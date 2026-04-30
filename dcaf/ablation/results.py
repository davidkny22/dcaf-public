"""
Result dataclasses for ablation testing (§11).

Provides structured output types for all ablation strategies:
AblationConfig, AblationResult, ParamAblationResult, PairAblationResult,
BinarySearchResult, BaselineResult, and WeightClassification.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from dcaf.core.defaults import (
    CLASSIFICATION_THRESHOLD,
    COHERENCE_THRESHOLD,
    HARM_THRESHOLD,
    SEPARATION_RATIO,
)


class ResponseCategory(Enum):
    """Classification of model response to harmful prompt."""
    REFUSE = "refuse"      # Explicit refusal
    COMPLY = "comply"      # Directly answers harmful request
    AVOID = "avoid"        # Redirects/changes subject
    IRRELEVANT = "irrelevant"  # Off-topic/incoherent response


@dataclass
class AblationConfig:
    """Configuration for ablation testing."""
    delta_scale: float = 2.0
    max_tokens: int = 100
    num_prompts: int = 5
    behavioral_threshold: float = HARM_THRESHOLD
    coherence_threshold: float = COHERENCE_THRESHOLD
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Safety testing
    harm_threshold: float = HARM_THRESHOLD  # Fraction of harmful responses to consider broken

    # Teacher forcing prefixes (for _test_teacher_forcing in base strategy)
    safe_prefix: str = ""
    unsafe_prefix: str = ""

    # Multi-probe classification
    enable_multi_probe: bool = False
    classification_threshold: float = CLASSIFICATION_THRESHOLD
    separation_ratio: float = SEPARATION_RATIO
    target_prefix: str = ""
    contrast_prefix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_scale": self.delta_scale,
            "max_tokens": self.max_tokens,
            "num_prompts": self.num_prompts,
            "behavioral_threshold": self.behavioral_threshold,
            "coherence_threshold": self.coherence_threshold,
            "device": self.device,
            "dtype": self.dtype,
            "harm_threshold": self.harm_threshold,
            "safe_prefix": self.safe_prefix,
            "unsafe_prefix": self.unsafe_prefix,
            "enable_multi_probe": self.enable_multi_probe,
            "classification_threshold": self.classification_threshold,
            "separation_ratio": self.separation_ratio,
            "target_prefix": self.target_prefix,
            "contrast_prefix": self.contrast_prefix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AblationConfig":
        """Load from dictionary."""
        return cls(
            delta_scale=data.get("delta_scale", 2.0),
            max_tokens=data.get("max_tokens", 100),
            num_prompts=data.get("num_prompts", 5),
            behavioral_threshold=data.get("behavioral_threshold", HARM_THRESHOLD),
            coherence_threshold=data.get("coherence_threshold", COHERENCE_THRESHOLD),
            device=data.get("device", "cuda"),
            dtype=data.get("dtype", "bfloat16"),
            harm_threshold=data.get("harm_threshold", HARM_THRESHOLD),
            safe_prefix=data.get("safe_prefix", ""),
            unsafe_prefix=data.get("unsafe_prefix", ""),
            enable_multi_probe=data.get("enable_multi_probe", False),
            classification_threshold=data.get("classification_threshold", CLASSIFICATION_THRESHOLD),
            separation_ratio=data.get("separation_ratio", SEPARATION_RATIO),
            target_prefix=data.get("target_prefix", ""),
            contrast_prefix=data.get("contrast_prefix", ""),
        )


@dataclass
class ProbeTypeResult:
    """Result from one probe type test."""
    probe_type: str  # "free_generation" or "teacher_forcing"
    harm_rate: float
    harmful_count: int
    total_count: int
    responses: List[str] = field(default_factory=list)
    classifications: List[ResponseCategory] = field(default_factory=list)

    # Teacher forcing specific
    steering_signal: Optional[float] = None  # safe_loss - unsafe_loss

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_type": self.probe_type,
            "harm_rate": self.harm_rate,
            "harmful_count": self.harmful_count,
            "total_count": self.total_count,
            "responses": self.responses,
            "classifications": [c.value for c in self.classifications],
            "steering_signal": self.steering_signal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbeTypeResult":
        """Load from dictionary."""
        classifications = [
            ResponseCategory(c) if isinstance(c, str) else c
            for c in data.get("classifications", [])
        ]
        return cls(
            probe_type=data.get("probe_type", ""),
            harm_rate=data.get("harm_rate", 0.0),
            harmful_count=data.get("harmful_count", 0),
            total_count=data.get("total_count", 0),
            responses=data.get("responses", []),
            classifications=classifications,
            steering_signal=data.get("steering_signal"),
        )


@dataclass
class WeightClassification:
    """
    Classification of a weight based on THREE probe type impacts.

    Classifies safety circuit weights by their impact on:
    - Recognition: Harm encoding (activation separation on harmful vs neutral)
    - Free generation: Steering decision (first 10 tokens of generation)
    - Teacher forcing: Path preference (perplexity on safe vs unsafe prefixes)
    """
    weight_name: str

    # Three separate impact measurements (0-1 scale)
    recognition_impact: float  # Harm encoding impact
    free_gen_impact: float  # Steering decision impact
    teacher_forcing_impact: float  # Path preference impact

    # Classification results
    classification: str  # "recognition-specific", "generation-specific", "shared", "false-positive", "unclear"
    generation_subtype: Optional[str] = None  # For generation-specific: "steering", "preference", "both"
    confidence: float = 0.0

    # Diagnostic information
    separation_ratios: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_name": self.weight_name,
            "recognition_impact": self.recognition_impact,
            "free_gen_impact": self.free_gen_impact,
            "teacher_forcing_impact": self.teacher_forcing_impact,
            "classification": self.classification,
            "generation_subtype": self.generation_subtype,
            "confidence": self.confidence,
            "separation_ratios": self.separation_ratios,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightClassification":
        """Load from dictionary."""
        return cls(
            weight_name=data.get("weight_name", ""),
            recognition_impact=data.get("recognition_impact", 0.0),
            free_gen_impact=data.get("free_gen_impact", 0.0),
            teacher_forcing_impact=data.get("teacher_forcing_impact", 0.0),
            classification=data.get("classification", "unclear"),
            generation_subtype=data.get("generation_subtype"),
            confidence=data.get("confidence", 0.0),
            separation_ratios=data.get("separation_ratios", {}),
        )


@dataclass
class AblationResult:
    """Base result class for ablation tests."""
    responses: List[str]
    harmful_count: int
    total_count: int
    classifications: List[ResponseCategory] = field(default_factory=list)

    @property
    def harm_rate(self) -> float:
        return self.harmful_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def breaks_safety(self) -> bool:
        """True if majority of responses are harmful."""
        return self.harm_rate >= 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "responses": self.responses,
            "harmful_count": self.harmful_count,
            "total_count": self.total_count,
            "harm_rate": self.harm_rate,
            "breaks_safety": self.breaks_safety,
            "classifications": [c.value for c in self.classifications],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AblationResult":
        """Load from dictionary."""
        classifications = [
            ResponseCategory(c) if isinstance(c, str) else c
            for c in data.get("classifications", [])
        ]
        return cls(
            responses=data.get("responses", []),
            harmful_count=data.get("harmful_count", 0),
            total_count=data.get("total_count", 0),
            classifications=classifications,
        )


@dataclass
class ParamAblationResult(AblationResult):
    """Result for single parameter ablation."""
    param_name: str = ""
    param_short: str = ""  # Short name like "L10.down_proj"
    coherent: bool = True  # Whether generation remained coherent
    coherence_passed: bool = True  
    ablation_validated: bool = False  # True if safety broke without breaking coherence
    safety_broken: bool = False  # Whether safety behavior broke after ablation
    refusal_rate_after: float = 0.0  # Refusal rate after ablation (0.0 to 1.0)
    test_time_seconds: float = 0.0  # How long the ablation test took

    # Response category counts (4-way classification)
    refuse_count: int = 0  # Model refused the harmful request
    comply_count: int = 0  # Model complied with the harmful request
    avoid_count: int = 0  # Model deflected/changed subject
    irrelevant_count: int = 0  # Model gave incoherent/off-topic response

    # Multi-probe results
    probe_results: Dict[str, ProbeTypeResult] = field(default_factory=dict)
    weight_classification: Optional[WeightClassification] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "param_name": self.param_name,
            "param_short": self.param_short,
            "coherent": self.coherent,
            "coherence_passed": self.coherence_passed,
            "ablation_validated": self.ablation_validated,
            "safety_broken": self.safety_broken,
            "refusal_rate_after": self.refusal_rate_after,
            "test_time_seconds": self.test_time_seconds,
            "refuse_count": self.refuse_count,
            "comply_count": self.comply_count,
            "avoid_count": self.avoid_count,
            "irrelevant_count": self.irrelevant_count,
            "probe_results": {k: v.to_dict() for k, v in self.probe_results.items()},
            "weight_classification": self.weight_classification.to_dict() if self.weight_classification else None,
        })
        return base


@dataclass
class AblationResults:
    """Aggregate results from single-param ablation testing."""
    total_tested: int = 0  # Total parameters tested
    validated_count: int = 0  # Parameters that are safety-specific
    rejected_count: int = 0  # Parameters that are NOT safety-specific
    skipped_count: int = 0  # Parameters that broke general generation
    param_results: List[ParamAblationResult] = field(default_factory=list)
    weight_classifications: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tested": self.total_tested,
            "validated_count": self.validated_count,
            "rejected_count": self.rejected_count,
            "skipped_count": self.skipped_count,
            "param_results": [r.to_dict() for r in self.param_results],
            "weight_classifications": {
                name: wc.to_dict() if hasattr(wc, 'to_dict') else wc
                for name, wc in self.weight_classifications.items()
            },
        }


@dataclass
class PairAblationResult(AblationResult):
    """Result for parameter pair ablation."""
    param1: str = ""
    param2: str = ""
    param1_short: str = ""
    param2_short: str = ""
    criteria: str = ""  # Which criteria this pair came from
    pair_id: int = 0
    coherent: bool = True  # Whether generation remained coherent after ablation
    coherence_score: float = 1.0  # Coherence score (0-1 for heuristic, perplexity for perplexity method)

    @property
    def ablation_validated(self) -> bool:
        """True if coherent AND breaks safety - confirms safety-critical, not model-critical."""
        return self.coherent and self.breaks_safety

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "param1": self.param1,
            "param2": self.param2,
            "param1_short": self.param1_short,
            "param2_short": self.param2_short,
            "criteria": self.criteria,
            "pair_id": self.pair_id,
            "coherent": self.coherent,
            "coherence_score": self.coherence_score,
            "ablation_validated": self.ablation_validated,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PairAblationResult":
        """Load from dictionary."""
        classifications = [
            ResponseCategory(c) if isinstance(c, str) else c
            for c in data.get("classifications", [])
        ]
        return cls(
            responses=data.get("responses", []),
            harmful_count=data.get("harmful_count", 0),
            total_count=data.get("total_count", 0),
            classifications=classifications,
            param1=data.get("param1", ""),
            param2=data.get("param2", ""),
            param1_short=data.get("param1_short", ""),
            param2_short=data.get("param2_short", ""),
            criteria=data.get("criteria", ""),
            pair_id=data.get("pair_id", 0),
            coherent=data.get("coherent", True),
            coherence_score=data.get("coherence_score", 1.0),
        )


@dataclass
class PairAblationResults:
    """Aggregated results from pair ablation testing."""
    config: AblationConfig
    prompts: List[str]
    baseline_no_safety: AblationResult
    baseline_with_safety: AblationResult
    within_criteria: Dict[str, List[PairAblationResult]] = field(default_factory=dict)
    cross_criteria: List[PairAblationResult] = field(default_factory=list)

    @property
    def all_results(self) -> List[PairAblationResult]:
        """All pair results combined."""
        results = []
        for crit_results in self.within_criteria.values():
            results.extend(crit_results)
        results.extend(self.cross_criteria)
        return results

    @property
    def coherent_results(self) -> List[PairAblationResult]:
        """Results where model remained coherent."""
        return [r for r in self.all_results if r.coherent]

    @property
    def incoherent_results(self) -> List[PairAblationResult]:
        """Results where model became incoherent (model-function-critical)."""
        return [r for r in self.all_results if not r.coherent]

    @property
    def breaking_pairs(self) -> List[PairAblationResult]:
        """Pairs that break safety (may include incoherent)."""
        return [r for r in self.all_results if r.breaks_safety]

    @property
    def validated_pairs(self) -> List[PairAblationResult]:
        """Pairs that are ablation-validated (coherent AND break safety)."""
        return [r for r in self.all_results if r.ablation_validated]

    @property
    def safe_pairs(self) -> List[PairAblationResult]:
        return [r for r in self.all_results if not r.breaks_safety]

    @property
    def break_rate(self) -> float:
        total = len(self.all_results)
        return len(self.breaking_pairs) / total if total > 0 else 0.0

    @property
    def validated_rate(self) -> float:
        """Rate of validated (coherent + breaking) pairs among coherent pairs."""
        coherent = len(self.coherent_results)
        return len(self.validated_pairs) / coherent if coherent > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "prompts": self.prompts,
            "baseline_no_safety": self.baseline_no_safety.to_dict(),
            "baseline_with_safety": self.baseline_with_safety.to_dict(),
            "within_criteria": {
                k: [r.to_dict() for r in v]
                for k, v in self.within_criteria.items()
            },
            "cross_criteria": [r.to_dict() for r in self.cross_criteria],
            "summary": {
                "total_pairs": len(self.all_results),
                "coherent_pairs": len(self.coherent_results),
                "incoherent_pairs": len(self.incoherent_results),
                "breaking_pairs": len(self.breaking_pairs),
                "validated_pairs": len(self.validated_pairs),
                "safe_pairs": len(self.safe_pairs),
                "break_rate": self.break_rate,
                "validated_rate": self.validated_rate,
            },
        }


@dataclass
class BinarySearchResult:
    """Result from binary search ablation."""
    config: AblationConfig
    initial_params: List[str]
    critical_params: List[str]  # Minimal set that breaks safety
    search_log: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0

    @property
    def reduction_ratio(self) -> float:
        """How much the parameter set was reduced."""
        if len(self.initial_params) == 0:
            return 0.0
        return 1.0 - (len(self.critical_params) / len(self.initial_params))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "initial_params_count": len(self.initial_params),
            "critical_params": self.critical_params,
            "critical_params_count": len(self.critical_params),
            "reduction_ratio": self.reduction_ratio,
            "iterations": self.iterations,
            "search_log": self.search_log,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BinarySearchResult":
        """Load from dictionary."""
        config = AblationConfig.from_dict(data.get("config", {}))
        return cls(
            config=config,
            initial_params=data.get("initial_params", []),
            critical_params=data.get("critical_params", []),
            search_log=data.get("search_log", []),
            iterations=data.get("iterations", 0),
        )


@dataclass
class BaselineResult:
    """Result from baseline validation."""
    name: str  # "base", "safe", or pair description
    expected_harmful: bool
    actual_harmful: bool
    responses: List[str]
    harmful_count: int
    total_count: int

    @property
    def passed(self) -> bool:
        """True if actual matches expected."""
        return self.actual_harmful == self.expected_harmful

    @property
    def harm_rate(self) -> float:
        return self.harmful_count / self.total_count if self.total_count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expected_harmful": self.expected_harmful,
            "actual_harmful": self.actual_harmful,
            "passed": self.passed,
            "harm_rate": self.harm_rate,
            "harmful_count": self.harmful_count,
            "total_count": self.total_count,
            "responses": self.responses,
        }


@dataclass
class BaselineValidationResults:
    """Aggregated baseline validation results."""
    base_model: BaselineResult
    safe_model: BaselineResult
    known_pairs: List[BaselineResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True if all baseline tests passed."""
        if not self.base_model.passed or not self.safe_model.passed:
            return False
        return all(p.passed for p in self.known_pairs)

    @property
    def failures(self) -> List[BaselineResult]:
        """List of failed baseline tests."""
        failed = []
        if not self.base_model.passed:
            failed.append(self.base_model)
        if not self.safe_model.passed:
            failed.append(self.safe_model)
        failed.extend(p for p in self.known_pairs if not p.passed)
        return failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "base_model": self.base_model.to_dict(),
            "safe_model": self.safe_model.to_dict(),
            "known_pairs": [p.to_dict() for p in self.known_pairs],
            "failures": [f.to_dict() for f in self.failures],
        }


def short_param_name(param: str) -> str:
    """Convert full param name to short form (e.g., L10.down_proj)."""
    return (
        param
        .replace("model.layers.", "L")
        .replace(".mlp.", ".")
        .replace(".weight", "")
    )

__all__ = [
    "ResponseCategory",
    "AblationConfig",
    "ProbeTypeResult",
    "WeightClassification",
    "AblationResult",
    "ParamAblationResult",
    "AblationResults",
    "PairAblationResult",
    "PairAblationResults",
    "BinarySearchResult",
    "BaselineResult",
    "BaselineValidationResults",
    "short_param_name",
]
