"""
Behavioral relevance confirmation (sec:ablation; def:behavioral-relevance-confirmation;
def:behav-relevance-class).

def:behavioral-relevance-confirmation: Ablation determines behavioral relevance per component k:
  - If ablation breaks model functionality entirely → GENERAL (not behavior-specific)
  - If model functions but behavior breaks/changes → BEHAVIORAL (proceed to classification)
  - No effect → NONE

def:behav-relevance-class: Relevance classification based on functionality ratio
and probe impact I_π^(k).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dcaf.ablation.methods import ModelStateManager


class ConfirmationStatus(Enum):
    """Status of behavioral relevance confirmation."""
    BEHAVIORAL = "behavioral"  # Component affects behavior specifically
    GENERAL = "general"        # Component breaks model entirely
    NONE = "none"              # No effect on behavior
    UNKNOWN = "unknown"        # Could not determine


@dataclass
class ConfirmationResult:
    """
    Result of behavioral relevance confirmation.

    Attributes:
        component: Component ID
        status: Confirmation status
        model_functional: Whether model remained functional after ablation
        behavior_changed: Whether target behavior changed
        details: Additional diagnostic information
    """
    component: str
    status: ConfirmationStatus
    model_functional: bool
    behavior_changed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        """True if component is confirmed as behaviorally relevant."""
        return self.status == ConfirmationStatus.BEHAVIORAL

    @property
    def is_general(self) -> bool:
        """True if component is general (breaks model)."""
        return self.status == ConfirmationStatus.GENERAL

    @property
    def is_false_positive(self) -> bool:
        """True if component has no behavioral effect."""
        return self.status == ConfirmationStatus.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "model_functional": self.model_functional,
            "behavior_changed": self.behavior_changed,
            "confirmed": self.confirmed,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfirmationResult":
        return cls(
            component=data["component"],
            status=ConfirmationStatus(data["status"]),
            model_functional=data["model_functional"],
            behavior_changed=data["behavior_changed"],
            details=data.get("details", {}),
        )


def test_model_functional(
    model,
    tokenizer,
    test_prompts: Optional[List[str]] = None,
    coherence_threshold: float = 0.5,
) -> bool:
    """
    Test if model is still functional (produces coherent output).

    Args:
        model: Model to test
        tokenizer: Tokenizer for the model
        test_prompts: Optional prompts to test with
        coherence_threshold: Minimum coherence score

    Returns:
        True if model is functional
    """
    if test_prompts is None:
        test_prompts = [
            "The capital of France is",
            "2 + 2 equals",
            "Hello, how are you",
        ]

    coherent_count = 0
    total_count = len(test_prompts)

    for prompt in test_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Basic coherence check
            response_only = response[len(prompt):].strip()
            if len(response_only) >= 5 and _is_coherent(response_only):
                coherent_count += 1
        except Exception:
            # Generation failed = not functional
            pass

    return (coherent_count / total_count) >= coherence_threshold


def _is_coherent(text: str) -> bool:
    """Basic coherence heuristic."""
    if not text:
        return False

    # Check for repetition
    words = text.split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            return False

    # Check for garbage (too many special chars)
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_ratio < 0.5:
        return False

    return True


def confirm_behavioral_relevance(
    component: str,
    model,
    tokenizer,
    state_manager: ModelStateManager,
    behavior_test_fn: Callable[..., bool],
    coherence_prompts: Optional[List[str]] = None,
    behavior_test_kwargs: Optional[Dict[str, Any]] = None,
) -> ConfirmationResult:
    """
    Confirm behavioral relevance of a component.

    Tests:
    1. Model breaks entirely → 'general' (not behavior-specific)
    2. Model works but behavior changes → 'behavioral' (confirmed)
    3. No effect → 'none' (false positive)

    Args:
        component: Component ID to test
        model: Model to ablate
        tokenizer: Tokenizer for coherence testing
        state_manager: ModelStateManager for ablation
        behavior_test_fn: Function that returns True if behavior is intact
        coherence_prompts: Optional prompts for coherence testing
        behavior_test_kwargs: Optional kwargs for behavior test

    Returns:
        ConfirmationResult
    """
    if behavior_test_kwargs is None:
        behavior_test_kwargs = {}

    # Get component parameters
    component_params = _get_component_params(component, state_manager)

    if not component_params:
        return ConfirmationResult(
            component=component,
            status=ConfirmationStatus.UNKNOWN,
            model_functional=True,
            behavior_changed=False,
            details={"error": "No parameters found for component"},
        )

    # Test with safety (baseline)
    state_manager.reset_to_safety()
    baseline_behavior = behavior_test_fn(model, **behavior_test_kwargs)
    baseline_functional = test_model_functional(model, tokenizer, coherence_prompts)

    details = {
        "n_params": len(component_params),
        "baseline_behavior": baseline_behavior,
        "baseline_functional": baseline_functional,
    }

    # Test with ablation
    with state_manager.temporary_ablation(component_params):
        ablated_functional = test_model_functional(model, tokenizer, coherence_prompts)
        ablated_behavior = behavior_test_fn(model, **behavior_test_kwargs)

    details["ablated_functional"] = ablated_functional
    details["ablated_behavior"] = ablated_behavior

    # Determine status
    if not ablated_functional:
        # Model broke entirely
        status = ConfirmationStatus.GENERAL
        model_functional = False
        behavior_changed = True  # Technically changed but because model broke
    elif ablated_behavior != baseline_behavior:
        # Behavior changed but model still works
        status = ConfirmationStatus.BEHAVIORAL
        model_functional = True
        behavior_changed = True
    else:
        # No effect on behavior
        status = ConfirmationStatus.NONE
        model_functional = True
        behavior_changed = False

    return ConfirmationResult(
        component=component,
        status=status,
        model_functional=model_functional,
        behavior_changed=behavior_changed,
        details=details,
    )


def _get_component_params(component: str, state_manager: ModelStateManager) -> List[str]:
    """Get all parameters belonging to a component.

    Delegates to the canonical implementation in dcaf.arch.transformer.
    """
    from dcaf.arch.transformer import get_component_params
    return get_component_params(component, state_manager.get_delta_params())


def batch_confirm(
    components: List[str],
    model,
    tokenizer,
    state_manager: ModelStateManager,
    behavior_test_fn: Callable[..., bool],
    coherence_prompts: Optional[List[str]] = None,
    behavior_test_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, ConfirmationResult]:
    """
    Confirm multiple components.

    Args:
        components: List of component IDs
        model: Model to test
        tokenizer: Tokenizer for coherence testing
        state_manager: ModelStateManager
        behavior_test_fn: Function that returns True if behavior is intact
        coherence_prompts: Optional prompts for coherence testing
        behavior_test_kwargs: Optional kwargs for behavior test

    Returns:
        {component: ConfirmationResult}
    """
    results = {}

    for component in components:
        results[component] = confirm_behavioral_relevance(
            component=component,
            model=model,
            tokenizer=tokenizer,
            state_manager=state_manager,
            behavior_test_fn=behavior_test_fn,
            coherence_prompts=coherence_prompts,
            behavior_test_kwargs=behavior_test_kwargs,
        )

    return results


def filter_confirmed(
    results: Dict[str, ConfirmationResult],
) -> List[str]:
    """
    Get list of confirmed behavioral components.

    Args:
        results: {component: ConfirmationResult}

    Returns:
        List of confirmed component IDs
    """
    return [c for c, r in results.items() if r.confirmed]


def filter_general(
    results: Dict[str, ConfirmationResult],
) -> List[str]:
    """
    Get list of general (model-breaking) components.

    Args:
        results: {component: ConfirmationResult}

    Returns:
        List of general component IDs
    """
    return [c for c, r in results.items() if r.is_general]


def filter_false_positives(
    results: Dict[str, ConfirmationResult],
) -> List[str]:
    """
    Get list of false positive components.

    Args:
        results: {component: ConfirmationResult}

    Returns:
        List of false positive component IDs
    """
    return [c for c, r in results.items() if r.is_false_positive]


def get_confirmation_summary(
    results: Dict[str, ConfirmationResult],
) -> Dict[str, Any]:
    """
    Summary statistics for confirmation results.

    Args:
        results: {component: ConfirmationResult}

    Returns:
        Summary dict
    """
    if not results:
        return {"count": 0}

    confirmed = len(filter_confirmed(results))
    general = len(filter_general(results))
    false_positives = len(filter_false_positives(results))
    unknown = len([r for r in results.values() if r.status == ConfirmationStatus.UNKNOWN])

    return {
        "count": len(results),
        "confirmed": confirmed,
        "general": general,
        "false_positives": false_positives,
        "unknown": unknown,
        "confirmation_rate": confirmed / len(results) if results else 0.0,
    }


__all__ = [
    "ConfirmationStatus",
    "ConfirmationResult",
    "test_model_functional",
    "confirm_behavioral_relevance",
    "batch_confirm",
    "filter_confirmed",
    "filter_general",
    "filter_false_positives",
    "get_confirmation_summary",
]
