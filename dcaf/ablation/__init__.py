"""Ablation methodology and functional classification (§11, Def 11.1-11.27).

Note: The Phase 2 seven-strategy classes (StrategyA-G) are defined in
dcaf/ablation/interaction_strategies.py but cannot be imported as dcaf.ablation.strategies
because that path resolves to the strategies/ package. Import them directly:

    from dcaf.ablation import StrategyA_GraphAdjacent  (re-exported here)

or via importlib if you need the module directly.
"""

# Re-export the 7 Phase 2 strategy classes from interaction_strategies.py via importlib
import importlib.util as _ilu
import os as _os
_strat_path = _os.path.join(_os.path.dirname(__file__), "interaction_strategies.py")
_strat_spec = _ilu.spec_from_file_location("dcaf.ablation._strategies_file", _strat_path)
_strat_mod = _ilu.module_from_spec(_strat_spec)
_strat_spec.loader.exec_module(_strat_mod)

StrategyResult = _strat_mod.StrategyResult
InteractionStrategy = _strat_mod.InteractionStrategy
StrategyA_GraphAdjacent = _strat_mod.StrategyA_GraphAdjacent
StrategyB_GradientScreening = _strat_mod.StrategyB_GradientScreening
StrategyC_ActivationCorrelation = _strat_mod.StrategyC_ActivationCorrelation
StrategyD_HierarchicalClustering = _strat_mod.StrategyD_HierarchicalClustering
StrategyE_OppositionGrouping = _strat_mod.StrategyE_OppositionGrouping
StrategyF_CrossLayerComposition = _strat_mod.StrategyF_CrossLayerComposition
StrategyG_RandomSampling = _strat_mod.StrategyG_RandomSampling
run_all_strategies = _strat_mod.run_all_strategies
compute_discovery_count = _strat_mod.compute_discovery_count
get_high_confidence_params = _strat_mod.get_high_confidence_params
get_interaction_summary = _strat_mod.get_interaction_summary
del _ilu, _os, _strat_path, _strat_spec, _strat_mod

from dcaf.ablation.methods import ModelStateManager
from dcaf.ablation.results import (
    AblationConfig,
    AblationResult,
    ParamAblationResult,
    AblationResults,
    PairAblationResult,
    PairAblationResults,
    BinarySearchResult,
    BaselineResult,
    BaselineValidationResults,
    WeightClassification,
    ProbeTypeResult,
    ResponseCategory,
    short_param_name,
)
from dcaf.ablation.superadditivity import (
    InteractionType,
    InteractionRequirement,
    SuperadditivityResult,
    classify_interaction,
    classify_interaction_requirement,
    test_superadditivity,
    test_pair_superadditivity,
    batch_test_superadditivity,
)
from dcaf.ablation.classification import (
    ComponentStatus,
    FinalClassification,
    classify_final,
    classify_all_final,
)

__all__ = [
    # Phase 2 seven strategies
    "InteractionStrategy",
    "StrategyResult",
    "StrategyA_GraphAdjacent",
    "StrategyB_GradientScreening",
    "StrategyC_ActivationCorrelation",
    "StrategyD_HierarchicalClustering",
    "StrategyE_OppositionGrouping",
    "StrategyF_CrossLayerComposition",
    "StrategyG_RandomSampling",
    "run_all_strategies",
    "compute_discovery_count",
    "get_high_confidence_params",
    "get_interaction_summary",
    # methods
    "ModelStateManager",
    # results
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
    # superadditivity
    "InteractionType",
    "InteractionRequirement",
    "SuperadditivityResult",
    "classify_interaction",
    "classify_interaction_requirement",
    "test_superadditivity",
    "test_pair_superadditivity",
    "batch_test_superadditivity",
    # classification
    "ComponentStatus",
    "FinalClassification",
    "classify_final",
    "classify_all_final",
]
