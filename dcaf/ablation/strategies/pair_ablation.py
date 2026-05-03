"""
Pair ablation strategy.

Ablates parameter pairs to find synergistic effects.
Extracted from run_full_ablation.py.
"""

from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from dcaf.ablation.methods import ModelStateManager
from dcaf.ablation.results import (
    AblationConfig,
    AblationResult,
    PairAblationResult,
    PairAblationResults,
    short_param_name,
)
from dcaf.ablation.strategies.base import AblationStrategy, CoherenceMethod
from dcaf.data.prompt_legacy import BENIGN_TEST_PROMPTS


class PairAblation(AblationStrategy):
    """
    Ablate parameter pairs to find synergistic safety effects.

    Tests both:
    - Within-criteria pairs (parameters from same criteria)
    - Cross-criteria pairs (parameters from different criteria)

    This identifies parameter interactions where ablating both together
    breaks safety even if ablating individually does not.
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
        self._pair_id_counter = 0
        self.benign_prompts = benign_prompts or BENIGN_TEST_PROMPTS[:3]
        self.coherence_method = coherence_method

    def run(
        self,
        params: List[str],
        prompts: List[str],
    ) -> PairAblationResults:
        """
        Run pair ablation on all combinations of params.

        Args:
            params: List of parameter names
            prompts: Test prompts

        Returns:
            PairAblationResults with all pair test results
        """
        # Test baselines first
        baseline_no_safety = self._test_baseline_no_safety(prompts)
        baseline_with_safety = self._test_baseline_with_safety(prompts)

        # Test all pairs
        results = []
        for p1, p2 in combinations(params, 2):
            result = self.test_pair(p1, p2, prompts, "single_criteria")
            results.append(result)

        return PairAblationResults(
            config=self.config,
            prompts=prompts,
            baseline_no_safety=baseline_no_safety,
            baseline_with_safety=baseline_with_safety,
            within_criteria={"single_criteria": results},
            cross_criteria=[],
        )

    def run_with_criteria(
        self,
        criteria_params: Dict[str, List[str]],
        prompts: List[str],
        include_cross: bool = True,
    ) -> PairAblationResults:
        """
        Run pair ablation with criteria-aware organization.

        Args:
            criteria_params: Dict mapping criteria name to parameter list
            prompts: Test prompts
            include_cross: Whether to test cross-criteria pairs

        Returns:
            PairAblationResults with within and cross-criteria results
        """
        # Test baselines
        baseline_no_safety = self._test_baseline_no_safety(prompts)
        baseline_with_safety = self._test_baseline_with_safety(prompts)

        # Build pair index to avoid duplicates
        pair_index = {}
        all_params = set()

        for criteria_name, params in criteria_params.items():
            all_params.update(params)
            for p1, p2 in combinations(params, 2):
                pk = self._pair_key(p1, p2)
                if pk not in pair_index:
                    self._pair_id_counter += 1
                    pair_index[pk] = {
                        "id": self._pair_id_counter,
                        "first_criteria": criteria_name,
                        "criteria_list": [criteria_name],
                        "params": pk,
                    }
                else:
                    pair_index[pk]["criteria_list"].append(criteria_name)

        # Test within-criteria pairs
        within_criteria = {}
        tested_pairs = set()

        for criteria_name, params in criteria_params.items():
            crit_results = []
            for p1, p2 in combinations(params, 2):
                pk = self._pair_key(p1, p2)
                if pk in tested_pairs:
                    continue  # Already tested in another criteria

                tested_pairs.add(pk)
                info = pair_index[pk]
                result = self.test_pair(
                    p1, p2, prompts, criteria_name, pair_id=info["id"]
                )
                crit_results.append(result)

            within_criteria[criteria_name] = crit_results

        # Test cross-criteria pairs
        cross_criteria = []
        if include_cross:
            all_params_list = sorted(all_params)
            for i, p1 in enumerate(all_params_list):
                for p2 in all_params_list[i + 1:]:
                    pk = self._pair_key(p1, p2)
                    if pk in tested_pairs:
                        continue  # Already tested within criteria

                    self._pair_id_counter += 1
                    result = self.test_pair(
                        p1, p2, prompts, "CROSS", pair_id=self._pair_id_counter
                    )
                    cross_criteria.append(result)

        return PairAblationResults(
            config=self.config,
            prompts=prompts,
            baseline_no_safety=baseline_no_safety,
            baseline_with_safety=baseline_with_safety,
            within_criteria=within_criteria,
            cross_criteria=cross_criteria,
        )

    def test_pair(
        self,
        param1: str,
        param2: str,
        prompts: List[str],
        criteria: str = "",
        pair_id: int = 0,
    ) -> PairAblationResult:
        """
        Test a single parameter pair.

        Args:
            param1: First parameter name
            param2: Second parameter name
            prompts: Test prompts
            criteria: Criteria name for this pair
            pair_id: Unique ID for this pair

        Returns:
            PairAblationResult
        """
        # Reset to safety, then ablate both parameters — always restore on exit
        self.state_manager.reset_to_safety()
        self.state_manager.ablate_params([param1, param2])
        try:
            return self._test_ablated_pair(param1, param2, prompts, criteria, pair_id)
        finally:
            self.state_manager.restore_params([param1, param2])

    def _test_ablated_pair(self, param1, param2, prompts, criteria, pair_id):
        """Test an already-ablated pair."""
        coherent, coherence_score = self.test_coherence(
            prompts=self.benign_prompts,
            method=self.coherence_method,
        )

        if not coherent:
            # Model is incoherent - can't meaningfully test safety
            return PairAblationResult(
                param1=param1,
                param2=param2,
                param1_short=short_param_name(param1),
                param2_short=short_param_name(param2),
                responses=[],
                harmful_count=0,
                total_count=len(prompts),
                criteria=criteria,
                pair_id=pair_id,
                coherent=False,
                coherence_score=coherence_score,
            )

        # Test safety (round two - behavior check)
        is_broken, harmful_count, responses = self.test_safety_broken(prompts)

        return PairAblationResult(
            param1=param1,
            param2=param2,
            param1_short=short_param_name(param1),
            param2_short=short_param_name(param2),
            responses=responses,
            harmful_count=harmful_count,
            total_count=len(prompts),
            criteria=criteria,
            pair_id=pair_id,
            coherent=True,
            coherence_score=coherence_score,
        )

    def _test_baseline_no_safety(self, prompts: List[str]) -> AblationResult:
        """Test base model without safety delta."""
        self.state_manager.reset_to_base()
        is_broken, harmful_count, responses = self.test_safety_broken(prompts)
        return AblationResult(
            responses=responses,
            harmful_count=harmful_count,
            total_count=len(prompts),
        )

    def _test_baseline_with_safety(self, prompts: List[str]) -> AblationResult:
        """Test model with safety delta applied."""
        self.state_manager.reset_to_safety()
        is_broken, harmful_count, responses = self.test_safety_broken(prompts)
        return AblationResult(
            responses=responses,
            harmful_count=harmful_count,
            total_count=len(prompts),
        )

    @staticmethod
    def _pair_key(p1: str, p2: str) -> Tuple[str, str]:
        """Create canonical key for parameter pair."""
        return tuple(sorted([p1, p2]))
