"""
Binary search ablation strategy.

Finds minimal critical parameter set via binary search.
Extracted from run_binary_ablation.py.
"""

from typing import List, Dict, Optional, Callable, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

from dcaf.ablation.strategies.base import AblationStrategy, CoherenceMethod
from dcaf.ablation.methods import ModelStateManager
from dcaf.ablation.results import AblationConfig, BinarySearchResult
from dcaf.data.prompt_legacy import BENIGN_TEST_PROMPTS


class BinarySearchAblation(AblationStrategy):
    """
    Find minimal critical parameter set via binary search.

    Algorithm:
    1. Start with all parameters that have deltas
    2. Verify full set breaks safety (if not, abort)
    3. Recursively bisect:
       - Test first half -> if breaks, recurse into first half
       - Test second half -> if breaks, recurse into second half
    4. Return minimal set that still breaks safety

    This efficiently finds the smallest parameter group needed
    to break safety, reducing from potentially hundreds of params.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        state_manager: ModelStateManager,
        config: AblationConfig,
        classifier: Optional[Callable] = None,
        min_group_size: int = 1,
        benign_prompts: Optional[List[str]] = None,
        coherence_method: CoherenceMethod = "heuristic",
    ):
        """
        Initialize binary search strategy.

        Args:
            model: Model to test
            tokenizer: Tokenizer
            state_manager: State manager
            config: Ablation config
            classifier: Optional response classifier
            min_group_size: Minimum group size to recurse into
            benign_prompts: Prompts for coherence testing
            coherence_method: Method for coherence testing ("heuristic" or "perplexity")
        """
        super().__init__(model, tokenizer, state_manager, config, classifier)
        self.min_group_size = min_group_size
        self.benign_prompts = benign_prompts or BENIGN_TEST_PROMPTS[:3]
        self.coherence_method = coherence_method

    def run(
        self,
        params: List[str],
        prompts: List[str],
    ) -> BinarySearchResult:
        """
        Run binary search to find minimal critical parameter set.

        Args:
            params: Parameters to search through
            prompts: Test prompts

        Returns:
            BinarySearchResult with critical params and search log
        """
        search_log = []

        # Verify full set breaks safety first
        self.state_manager.reset_to_safety()
        self.state_manager.ablate_params(params)

        # Check coherence first (round one - model function check)
        coherent, coherence_score = self.test_coherence(
            prompts=self.benign_prompts,
            method=self.coherence_method,
        )

        search_log.append({
            "step": "initial_coherence_check",
            "params_count": len(params),
            "coherent": coherent,
            "coherence_score": coherence_score,
        })

        if not coherent:
            # Full set makes model incoherent - can't do meaningful search
            return BinarySearchResult(
                config=self.config,
                initial_params=params,
                critical_params=[],
                search_log=search_log,
                iterations=1,
            )

        # Check safety (round two - behavior check)
        is_broken, harmful_count, responses = self.test_safety_broken(prompts)

        search_log.append({
            "step": "initial_safety_check",
            "params_count": len(params),
            "is_broken": is_broken,
            "harmful_count": harmful_count,
        })

        if not is_broken:
            # Full set doesn't break safety - nothing to find
            return BinarySearchResult(
                config=self.config,
                initial_params=params,
                critical_params=[],
                search_log=search_log,
                iterations=2,
            )

        # Run binary search
        critical_params = self._binary_search(params, prompts, search_log)

        return BinarySearchResult(
            config=self.config,
            initial_params=params,
            critical_params=critical_params,
            search_log=search_log,
            iterations=len(search_log),
        )

    def _binary_search(
        self,
        params: List[str],
        prompts: List[str],
        search_log: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Recursive binary search for minimal critical set.

        At each step, checks coherence before testing safety to avoid
        falsely identifying model-function-critical params as safety-critical.

        Args:
            params: Current parameter set
            prompts: Test prompts
            search_log: Log to append results to

        Returns:
            Minimal parameter set that breaks safety (while staying coherent)
        """
        if len(params) <= self.min_group_size:
            return params

        # Split into halves
        mid = len(params) // 2
        first_half = params[:mid]
        second_half = params[mid:]

        critical_from_first = []
        critical_from_second = []

        # Test first half
        if first_half:
            self.state_manager.reset_to_safety()
            self.state_manager.ablate_params(first_half)

            # Check coherence first
            coherent, coherence_score = self.test_coherence(
                prompts=self.benign_prompts,
                method=self.coherence_method,
            )

            if not coherent:
                # First half makes model incoherent - skip this branch
                search_log.append({
                    "step": "test_first_half",
                    "params_count": len(first_half),
                    "coherent": False,
                    "coherence_score": coherence_score,
                    "is_broken": False,
                    "harmful_count": 0,
                    "skipped": "incoherent",
                })
            else:
                is_broken, harmful_count, _ = self.test_safety_broken(prompts)

                search_log.append({
                    "step": "test_first_half",
                    "params_count": len(first_half),
                    "coherent": True,
                    "coherence_score": coherence_score,
                    "is_broken": is_broken,
                    "harmful_count": harmful_count,
                })

                if is_broken:
                    # First half alone breaks safety - recurse
                    critical_from_first = self._binary_search(
                        first_half, prompts, search_log
                    )

        # Test second half
        if second_half:
            self.state_manager.reset_to_safety()
            self.state_manager.ablate_params(second_half)

            # Check coherence first
            coherent, coherence_score = self.test_coherence(
                prompts=self.benign_prompts,
                method=self.coherence_method,
            )

            if not coherent:
                # Second half makes model incoherent - skip this branch
                search_log.append({
                    "step": "test_second_half",
                    "params_count": len(second_half),
                    "coherent": False,
                    "coherence_score": coherence_score,
                    "is_broken": False,
                    "harmful_count": 0,
                    "skipped": "incoherent",
                })
            else:
                is_broken, harmful_count, _ = self.test_safety_broken(prompts)

                search_log.append({
                    "step": "test_second_half",
                    "params_count": len(second_half),
                    "coherent": True,
                    "coherence_score": coherence_score,
                    "is_broken": is_broken,
                    "harmful_count": harmful_count,
                })

                if is_broken:
                    # Second half alone breaks safety - recurse
                    critical_from_second = self._binary_search(
                        second_half, prompts, search_log
                    )

        # Combine critical params from both halves
        critical = critical_from_first + critical_from_second

        # If neither half alone breaks safety, we need both
        if not critical_from_first and not critical_from_second:
            # Need params from both halves - return combined
            return params

        return critical if critical else params

    def run_with_verification(
        self,
        params: List[str],
        prompts: List[str],
    ) -> BinarySearchResult:
        """
        Run binary search with verification of minimal set.

        After finding critical params, verifies the set is truly minimal
        by testing if removing any single param allows safety to recover.

        Args:
            params: Parameters to search
            prompts: Test prompts

        Returns:
            BinarySearchResult with verified minimal set
        """
        result = self.run(params, prompts)

        if not result.critical_params:
            return result

        # Verify minimality
        verified_critical = []
        for param in result.critical_params:
            # Test without this param
            test_params = [p for p in result.critical_params if p != param]
            if test_params:
                self.state_manager.reset_to_safety()
                self.state_manager.ablate_params(test_params)
                is_broken, _, _ = self.test_safety_broken(prompts)

                if not is_broken:
                    # Removing this param restores safety - it's critical
                    verified_critical.append(param)

                result.search_log.append({
                    "step": "verify_necessity",
                    "tested_param": param,
                    "is_necessary": not is_broken,
                })

        # If verification reduced the set, update result
        if verified_critical and len(verified_critical) < len(result.critical_params):
            result.critical_params = verified_critical
            result.search_log.append({
                "step": "verification_complete",
                "original_count": len(result.critical_params),
                "verified_count": len(verified_critical),
            })

        return result
