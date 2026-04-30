"""Ablation methodology and functional classification (§11).

The package initializer keeps import-time side effects low. Torch-backed
runtime helpers are loaded lazily through ``__getattr__``.
"""

from dcaf.ablation.results import (
    AblationConfig,
    AblationResult,
    AblationResults,
    BaselineResult,
    BaselineValidationResults,
    BinarySearchResult,
    PairAblationResult,
    PairAblationResults,
    ParamAblationResult,
    ProbeTypeResult,
    ResponseCategory,
    WeightClassification,
    short_param_name,
)

_LAZY_EXPORTS = {
    "ModelStateManager": ("dcaf.ablation.methods", "ModelStateManager"),
    "InteractionType": ("dcaf.ablation.superadditivity", "InteractionType"),
    "InteractionRequirement": ("dcaf.ablation.superadditivity", "InteractionRequirement"),
    "SuperadditivityResult": ("dcaf.ablation.superadditivity", "SuperadditivityResult"),
    "classify_interaction": ("dcaf.ablation.superadditivity", "classify_interaction"),
    "classify_interaction_requirement": (
        "dcaf.ablation.superadditivity",
        "classify_interaction_requirement",
    ),
    "test_superadditivity": ("dcaf.ablation.superadditivity", "test_superadditivity"),
    "test_pair_superadditivity": (
        "dcaf.ablation.superadditivity",
        "test_pair_superadditivity",
    ),
    "batch_test_superadditivity": (
        "dcaf.ablation.superadditivity",
        "batch_test_superadditivity",
    ),
    "ComponentStatus": ("dcaf.ablation.classification", "ComponentStatus"),
    "FinalClassification": ("dcaf.ablation.classification", "FinalClassification"),
    "classify_final": ("dcaf.ablation.classification", "classify_final"),
    "classify_all_final": ("dcaf.ablation.classification", "classify_all_final"),
    "InteractionStrategy": ("dcaf.ablation.interaction_strategies", "InteractionStrategy"),
    "StrategyResult": ("dcaf.ablation.interaction_strategies", "StrategyResult"),
    "StrategyA_GraphAdjacent": (
        "dcaf.ablation.interaction_strategies",
        "StrategyA_GraphAdjacent",
    ),
    "StrategyB_GradientScreening": (
        "dcaf.ablation.interaction_strategies",
        "StrategyB_GradientScreening",
    ),
    "StrategyC_ActivationCorrelation": (
        "dcaf.ablation.interaction_strategies",
        "StrategyC_ActivationCorrelation",
    ),
    "StrategyD_HierarchicalClustering": (
        "dcaf.ablation.interaction_strategies",
        "StrategyD_HierarchicalClustering",
    ),
    "StrategyE_OppositionGrouping": (
        "dcaf.ablation.interaction_strategies",
        "StrategyE_OppositionGrouping",
    ),
    "StrategyF_CrossLayerComposition": (
        "dcaf.ablation.interaction_strategies",
        "StrategyF_CrossLayerComposition",
    ),
    "StrategyG_RandomSampling": (
        "dcaf.ablation.interaction_strategies",
        "StrategyG_RandomSampling",
    ),
    "run_all_strategies": ("dcaf.ablation.interaction_strategies", "run_all_strategies"),
    "compute_discovery_count": (
        "dcaf.ablation.interaction_strategies",
        "compute_discovery_count",
    ),
    "get_high_confidence_params": (
        "dcaf.ablation.interaction_strategies",
        "get_high_confidence_params",
    ),
    "get_interaction_summary": (
        "dcaf.ablation.interaction_strategies",
        "get_interaction_summary",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "AblationConfig",
    "AblationResult",
    "ParamAblationResult",
    "AblationResults",
    "PairAblationResult",
    "PairAblationResults",
    "BinarySearchResult",
    "BaselineResult",
    "BaselineValidationResults",
    "WeightClassification",
    "ProbeTypeResult",
    "ResponseCategory",
    "short_param_name",
    *_LAZY_EXPORTS.keys(),
]
