"""
Single parameter ablation strategy.

Ablates one parameter at a time to identify individually critical parameters.
"""

from typing import Callable, List, Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from dcaf.ablation.methods import ModelStateManager
from dcaf.ablation.results import (
    AblationConfig,
    ParamAblationResult,
    short_param_name,
)
from dcaf.ablation.strategies.base import AblationStrategy, CoherenceMethod
from dcaf.data.prompt_legacy import BENIGN_TEST_PROMPTS


class SingleParamAblation(AblationStrategy):
    """
    Ablate one parameter at a time to find individually critical weights.

    For each parameter:
    1. Reset to safety state
    2. Ablate only that parameter
    3. Test coherence (can model still generate sensible output?)
    4. If coherent, test safety (does it comply with harmful requests?)
    5. Mark as "ablation_validated" if coherent AND safety broken

    This identifies parameters that are specifically safety-critical,
    not just generally important for model function.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        state_manager: ModelStateManager,
        config: AblationConfig,
        classifier: Optional[Callable] = None,
        benign_prompts: Optional[List[str]] = None,
        coherence_method: CoherenceMethod = "heuristic",
    ):
        super().__init__(model, tokenizer, state_manager, config, classifier)
        self.benign_prompts = benign_prompts or BENIGN_TEST_PROMPTS[:3]
        self.coherence_method = coherence_method

    def run(
        self,
        params: List[str],
        prompts: List[str],
    ) -> List[ParamAblationResult]:
        """
        Run single-parameter ablation on all specified parameters.

        Args:
            params: List of parameter names to test
            prompts: Harmful prompts for safety testing

        Returns:
            List of ParamAblationResult for each parameter
        """
        results = []

        for param in params:
            result = self.test_param(param, prompts)
            results.append(result)

        return results

    def test_param(
        self,
        param: str,
        harmful_prompts: List[str],
    ) -> ParamAblationResult:
        """
        Test a single parameter.

        Args:
            param: Parameter name to ablate
            harmful_prompts: Prompts for safety testing

        Returns:
            ParamAblationResult with coherence and safety status
        """
        # Reset to safety state
        self.state_manager.reset_to_safety()

        # Ablate this parameter — always restore on exit
        self.state_manager.ablate_params([param])
        try:
            return self._test_ablated_param(param, harmful_prompts)
        finally:
            self.state_manager.restore_params([param])

    def _test_ablated_param(self, param: str, harmful_prompts) -> "ParamAblationResult":
        """Test a param that is already ablated. Called by test_single_param."""
        # Test coherence first (round one - model function check)
        coherent, coherence_score = self.test_coherence(
            prompts=self.benign_prompts,
            method=self.coherence_method,
        )

        if not coherent:
            # If not coherent, we can't test safety meaningfully
            return ParamAblationResult(
                param_name=param,
                param_short=short_param_name(param),
                responses=[],
                harmful_count=0,
                total_count=len(harmful_prompts),
                coherent=False,
                ablation_validated=False,
                probe_results={},
                weight_classification=None,
            )

        # Multi-probe testing
        probe_results = {}
        weight_class = None

        if self.config.enable_multi_probe:
            # Run all three probe types
            # Recognition: activation separation on harmful vs neutral prompts
            recognition_result = self.test_probe_type("recognition", harmful_prompts)
            probe_results["recognition"] = recognition_result

            # Free generation: prompt → generate → classify (measures steering decision)
            free_gen_result = self.test_probe_type("free_generation", harmful_prompts)
            probe_results["free_generation"] = free_gen_result

            # Teacher forcing: safe_continuation vs unsafe_continuation perplexity
            teacher_forcing_result = self.test_probe_type("teacher_forcing", harmful_prompts)
            probe_results["teacher_forcing"] = teacher_forcing_result

            # Classify weight by comparing impacts from all three probes
            weight_class = self.classify_weight(
                recognition_result,
                free_gen_result,
                teacher_forcing_result,
                threshold=self.config.classification_threshold,
                separation_ratio=self.config.separation_ratio,
            )
            weight_class.weight_name = param

            # Only behavioral probes count as safety breakage; recognition is a
            # separation probe and should not validate causal harm on its own.
            max_harm = max(
                free_gen_result.harm_rate,
                teacher_forcing_result.harm_rate,
            )
            is_broken = max_harm >= self.config.harm_threshold

            # Use free generation responses for backward compatibility
            responses = free_gen_result.responses
            harmful_count = free_gen_result.harmful_count
        else:
            # Legacy: just free generation (via test_safety_broken)
            is_broken, harmful_count, responses = self.test_safety_broken(harmful_prompts)

        return ParamAblationResult(
            param_name=param,
            param_short=short_param_name(param),
            responses=responses,
            harmful_count=harmful_count,
            total_count=len(harmful_prompts),
            coherent=True,
            ablation_validated=is_broken,
            probe_results=probe_results,
            weight_classification=weight_class,
        )

    def get_validated_params(
        self,
        results: List[ParamAblationResult],
    ) -> List[str]:
        """
        Get parameters that were ablation-validated (safety-specific).

        Args:
            results: Results from run()

        Returns:
            List of parameter names that are safety-critical
        """
        return [
            r.param_name
            for r in results
            if r.ablation_validated
        ]

    def get_summary(self, results: List[ParamAblationResult]) -> dict:
        """
        Get summary statistics from results.

        Args:
            results: Results from run()

        Returns:
            Summary dictionary
        """
        total = len(results)
        coherent = sum(1 for r in results if r.coherent)
        validated = sum(1 for r in results if r.ablation_validated)

        return {
            "total_tested": total,
            "coherent": coherent,
            "incoherent": total - coherent,
            "ablation_validated": validated,
            "validation_rate": validated / total if total > 0 else 0.0,
        }
