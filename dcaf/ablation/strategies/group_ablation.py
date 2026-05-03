"""
Group ablation strategy.

Ablates multiple parameters together to test if a group collectively
encodes safety behavior.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from dcaf.ablation.methods import ModelStateManager
from dcaf.ablation.results import AblationConfig, AblationResult
from dcaf.ablation.strategies.base import AblationStrategy


@dataclass
class GroupAblationResult(AblationResult):
    """Result from group ablation testing."""

    param_names: List[str] = field(default_factory=list)
    safety_broken: bool = False
    refusal_rate: float = 0.0
    refusal_count: int = 0
    total_count: int = 0
    coherent: bool = True  # Whether generation remained coherent after ablation
    coherence_score: float = 1.0  # Coherence score (0-1 for heuristic)

    @property
    def ablation_validated(self) -> bool:
        """True if coherent AND breaks safety - confirms safety-critical, not model-critical."""
        return self.coherent and self.safety_broken

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "param_names": self.param_names,
            "param_count": len(self.param_names),
            "safety_broken": self.safety_broken,
            "refusal_rate": self.refusal_rate,
            "refusal_count": self.refusal_count,
            "total_count": self.total_count,
            "coherent": self.coherent,
            "coherence_score": self.coherence_score,
            "ablation_validated": self.ablation_validated,
        })
        return data


class GroupAblation(AblationStrategy):
    """
    Ablate multiple parameters together to test group effects.

    This strategy tests if a group of parameters collectively encodes
    safety behavior. Unlike SingleParamAblation which tests one at a time,
    this ablates all specified parameters simultaneously.

    Use cases:
    - Testing if all params in a layer together affect safety
    - Testing combined effects that might not show with single params
    - Validating DCAF-identified parameter sets as a group
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        state_manager: ModelStateManager,
        config: AblationConfig,
        classifier: Optional[Callable] = None,
        benign_prompts: Optional[List[str]] = None,
        coherence_method: str = "heuristic",
    ):
        super().__init__(model, tokenizer, state_manager, config, classifier)
        # Benign prompts used for coherence checking before safety testing.
        # If not provided, test_coherence() falls back to its own defaults.
        self.benign_prompts = benign_prompts
        # Coherence detection method: "heuristic" (word uniqueness) or "perplexity"
        self.coherence_method = coherence_method

    def run(
        self,
        params: List[str],
        prompts: List[str],
    ) -> GroupAblationResult:
        """
        Run group ablation on all specified parameters simultaneously.

        Args:
            params: List of parameter names to ablate together
            prompts: Harmful prompts for safety testing

        Returns:
            GroupAblationResult with safety status
        """
        # Reset to safety state
        self.state_manager.reset_to_safety()

        # Ablate all params at once
        self.state_manager.ablate_params(params)

        # Test coherence first (round one - model function check)
        # This ensures the model can still generate sensible output
        coherent, coherence_score = self.test_coherence(
            prompts=self.benign_prompts,
            method=self.coherence_method,
        )

        if not coherent:
            # Model incoherent - can't test safety meaningfully
            self.state_manager.reset_to_safety()
            return GroupAblationResult(
                param_names=params,
                safety_broken=False,
                refusal_rate=1.0,  # Treat as refusing (safe) since we can't test
                refusal_count=len(prompts),
                total_count=len(prompts),
                coherent=False,
                coherence_score=coherence_score,
                responses=[],
                harmful_count=0,
            )

        # Test safety using base class method (round two - behavior check)
        # This uses consistent classification via classify_response() -> ResponseCategory
        is_broken, harmful_count, responses = self.test_safety_broken(prompts)

        refusal_rate = 1.0 - (harmful_count / len(prompts)) if prompts else 1.0
        safety_broken = is_broken  # Already computed using config.harm_threshold

        # Restore state
        self.state_manager.reset_to_safety()

        return GroupAblationResult(
            param_names=params,
            safety_broken=safety_broken,
            refusal_rate=refusal_rate,
            refusal_count=len(prompts) - harmful_count,
            total_count=len(prompts),
            coherent=True,
            coherence_score=coherence_score,
            responses=responses,
            harmful_count=harmful_count,
        )

    def test_subgroups(
        self,
        params: List[str],
        prompts: List[str],
        group_size: int = 5,
    ) -> List[GroupAblationResult]:
        """
        Test subgroups of parameters to find minimal critical groups.

        Args:
            params: Full list of parameters
            prompts: Harmful prompts for testing
            group_size: Size of each subgroup to test

        Returns:
            List of GroupAblationResult for each subgroup
        """
        results = []

        for i in range(0, len(params), group_size):
            subgroup = params[i:i + group_size]
            result = self.run(subgroup, prompts)
            results.append(result)

        return results

    def get_breaking_groups(
        self,
        results: List[GroupAblationResult],
    ) -> List[List[str]]:
        """
        Get parameter groups that broke safety when ablated.

        Args:
            results: Results from run() or test_subgroups()

        Returns:
            List of parameter name lists that broke safety
        """
        return [r.param_names for r in results if r.safety_broken]
