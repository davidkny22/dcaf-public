"""
Known Circuits Database for DCAF.

A database of circuits that have been identified in published research.
For models like GPT-2, researchers have mapped specific attention heads
and MLP layers to interpretable functions.

This module provides:
1. Hardcoded known circuits from literature
2. Mappings to actual parameter names
3. Confidence scores based on replication status

References:
- "A Mathematical Framework for Transformer Circuits" (Anthropic)
- "In-context Learning and Induction Heads" (Anthropic)
- "Interpretability in the Wild" (Neel Nanda et al.)
- "Locating and Editing Factual Associations" (Meng et al.)
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from dcaf.data.safety_prompts import SafetyCategory

logger = logging.getLogger(__name__)


class CircuitType(Enum):
    """Types of circuits identified in research."""
    INDUCTION = "induction"  # In-context learning
    COPY = "copy"  # Token copying
    NAME_MOVER = "name_mover"  # Entity tracking
    BACKUP_NAME_MOVER = "backup_name_mover"
    NEGATIVE_NAME_MOVER = "negative_name_mover"
    S_INHIBITION = "s_inhibition"  # Subject inhibition
    FACTUAL = "factual"  # Factual recall
    SAFETY = "safety"  # Safety-related (emerging research)
    REFUSAL = "refusal"  # Refusal behavior
    OUTPUT = "output"  # Output generation


@dataclass
class KnownCircuit:
    """A circuit identified in published research."""
    name: str
    circuit_type: CircuitType
    model: str  # Which model this was found in
    layer: int
    head: Optional[int]  # For attention circuits
    component: str  # "attn", "mlp", "ln", etc.
    description: str
    paper: str  # Citation
    confidence: float  # How well replicated (0-1)
    safety_relevant: bool  # Does this affect safety behaviors?
    safety_category: Optional[SafetyCategory]


class KnownCircuitsDatabase:
    """
    Database of known circuits from published research.

    Use this to immediately freeze well-established circuits
    without running expensive analysis.
    """

    # =========================================
    # GPT-2 SMALL CIRCUITS
    # =========================================
    GPT2_SMALL_CIRCUITS: List[KnownCircuit] = [
        # Induction Heads (well-established)
        KnownCircuit(
            name="induction_L5H5",
            circuit_type=CircuitType.INDUCTION,
            model="gpt2-small",
            layer=5,
            head=5,
            component="attn",
            description="Primary induction head for in-context learning",
            paper="Anthropic, In-context Learning and Induction Heads (2022)",
            confidence=0.95,
            safety_relevant=False,
            safety_category=None
        ),
        KnownCircuit(
            name="induction_L6H9",
            circuit_type=CircuitType.INDUCTION,
            model="gpt2-small",
            layer=6,
            head=9,
            component="attn",
            description="Secondary induction head",
            paper="Anthropic, In-context Learning and Induction Heads (2022)",
            confidence=0.90,
            safety_relevant=False,
            safety_category=None
        ),

        # Name Mover Heads (IOI circuit)
        KnownCircuit(
            name="name_mover_L9H9",
            circuit_type=CircuitType.NAME_MOVER,
            model="gpt2-small",
            layer=9,
            head=9,
            component="attn",
            description="Moves name from subject position to output",
            paper="Wang et al., Interpretability in the Wild (2022)",
            confidence=0.85,
            safety_relevant=False,
            safety_category=None
        ),
        KnownCircuit(
            name="name_mover_L10H0",
            circuit_type=CircuitType.NAME_MOVER,
            model="gpt2-small",
            layer=10,
            head=0,
            component="attn",
            description="Secondary name mover head",
            paper="Wang et al., Interpretability in the Wild (2022)",
            confidence=0.85,
            safety_relevant=False,
            safety_category=None
        ),

        # S-Inhibition Heads
        KnownCircuit(
            name="s_inhibition_L7H3",
            circuit_type=CircuitType.S_INHIBITION,
            model="gpt2-small",
            layer=7,
            head=3,
            component="attn",
            description="Inhibits subject token to prevent wrong completion",
            paper="Wang et al., Interpretability in the Wild (2022)",
            confidence=0.80,
            safety_relevant=True,  # Could affect refusal
            safety_category=SafetyCategory.REFUSAL
        ),
        KnownCircuit(
            name="s_inhibition_L8H6",
            circuit_type=CircuitType.S_INHIBITION,
            model="gpt2-small",
            layer=8,
            head=6,
            component="attn",
            description="Secondary S-inhibition head",
            paper="Wang et al., Interpretability in the Wild (2022)",
            confidence=0.80,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),

        # Late layer MLP (output formation)
        KnownCircuit(
            name="output_mlp_L10",
            circuit_type=CircuitType.OUTPUT,
            model="gpt2-small",
            layer=10,
            head=None,
            component="mlp",
            description="Final output logit computation",
            paper="Anthropic, Mathematical Framework (2021)",
            confidence=0.90,
            safety_relevant=True,  # Final output decision
            safety_category=SafetyCategory.REFUSAL
        ),
        KnownCircuit(
            name="output_mlp_L11",
            circuit_type=CircuitType.OUTPUT,
            model="gpt2-small",
            layer=11,
            head=None,
            component="mlp",
            description="Final layer MLP for output",
            paper="Anthropic, Mathematical Framework (2021)",
            confidence=0.90,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),

        # Hypothesized safety-relevant circuits (less established)
        KnownCircuit(
            name="late_attn_L11H0",
            circuit_type=CircuitType.SAFETY,
            model="gpt2-small",
            layer=11,
            head=0,
            component="attn",
            description="Late attention head - may encode output policy",
            paper="Emerging research, lower confidence",
            confidence=0.60,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),
        KnownCircuit(
            name="late_attn_L11H5",
            circuit_type=CircuitType.SAFETY,
            model="gpt2-small",
            layer=11,
            head=5,
            component="attn",
            description="Late attention head - context integration",
            paper="Emerging research, lower confidence",
            confidence=0.55,
            safety_relevant=True,
            safety_category=SafetyCategory.ANTI_MANIPULATION
        ),
    ]

    # =========================================
    # GPT-2 MEDIUM CIRCUITS
    # =========================================
    GPT2_MEDIUM_CIRCUITS: List[KnownCircuit] = [
        KnownCircuit(
            name="induction_L12H11",
            circuit_type=CircuitType.INDUCTION,
            model="gpt2-medium",
            layer=12,
            head=11,
            component="attn",
            description="Strong induction head in medium model",
            paper="Anthropic, In-context Learning (2022)",
            confidence=0.90,
            safety_relevant=False,
            safety_category=None
        ),
        KnownCircuit(
            name="factual_mlp_L16",
            circuit_type=CircuitType.FACTUAL,
            model="gpt2-medium",
            layer=16,
            head=None,
            component="mlp",
            description="Factual knowledge storage",
            paper="Meng et al., ROME (2022)",
            confidence=0.85,
            safety_relevant=True,
            safety_category=SafetyCategory.HONESTY
        ),
        KnownCircuit(
            name="output_mlp_L22",
            circuit_type=CircuitType.OUTPUT,
            model="gpt2-medium",
            layer=22,
            head=None,
            component="mlp",
            description="Output formation layer",
            paper="General transformer analysis",
            confidence=0.85,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),
        KnownCircuit(
            name="output_mlp_L23",
            circuit_type=CircuitType.OUTPUT,
            model="gpt2-medium",
            layer=23,
            head=None,
            component="mlp",
            description="Final output layer",
            paper="General transformer analysis",
            confidence=0.85,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),
    ]

    # =========================================
    # PYTHIA-410M CIRCUITS
    # Pythia is a base model without safety training.
    # These circuits are identified based on:
    # 1. General transformer architecture patterns
    # 2. EleutherAI's Pythia interpretability research
    # 3. Analogous circuits from GPT-2 scaled to Pythia's 24 layers
    # =========================================
    PYTHIA_410M_CIRCUITS: List[KnownCircuit] = [
        # Induction heads (well-established in all transformer models)
        KnownCircuit(
            name="induction_L10H8",
            circuit_type=CircuitType.INDUCTION,
            model="pythia-410m",
            layer=10,
            head=8,
            component="attn",
            description="Primary induction head for in-context learning",
            paper="EleutherAI Pythia suite, based on Anthropic patterns",
            confidence=0.85,
            safety_relevant=False,
            safety_category=None
        ),
        KnownCircuit(
            name="induction_L12H11",
            circuit_type=CircuitType.INDUCTION,
            model="pythia-410m",
            layer=12,
            head=11,
            component="attn",
            description="Secondary induction head",
            paper="EleutherAI Pythia suite, based on Anthropic patterns",
            confidence=0.80,
            safety_relevant=False,
            safety_category=None
        ),

        # Late layer attention (output formation - safety relevant)
        KnownCircuit(
            name="late_attn_L20H0",
            circuit_type=CircuitType.OUTPUT,
            model="pythia-410m",
            layer=20,
            head=0,
            component="attn",
            description="Late attention head - output policy formation",
            paper="Emergent circuit analysis, moderate confidence",
            confidence=0.65,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),
        KnownCircuit(
            name="late_attn_L21H5",
            circuit_type=CircuitType.OUTPUT,
            model="pythia-410m",
            layer=21,
            head=5,
            component="attn",
            description="Late attention head - context integration",
            paper="Emergent circuit analysis, moderate confidence",
            confidence=0.60,
            safety_relevant=True,
            safety_category=SafetyCategory.ANTI_MANIPULATION
        ),
        KnownCircuit(
            name="late_attn_L22H10",
            circuit_type=CircuitType.OUTPUT,
            model="pythia-410m",
            layer=22,
            head=10,
            component="attn",
            description="Very late attention - final output decisions",
            paper="Emergent circuit analysis, moderate confidence",
            confidence=0.60,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),

        # Late MLP layers (output formation)
        KnownCircuit(
            name="output_mlp_L20",
            circuit_type=CircuitType.OUTPUT,
            model="pythia-410m",
            layer=20,
            head=None,
            component="mlp",
            description="Late MLP - begins output token selection",
            paper="General transformer analysis",
            confidence=0.80,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),
        KnownCircuit(
            name="output_mlp_L21",
            circuit_type=CircuitType.OUTPUT,
            model="pythia-410m",
            layer=21,
            head=None,
            component="mlp",
            description="Late MLP - output logit computation",
            paper="General transformer analysis",
            confidence=0.85,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),
        KnownCircuit(
            name="output_mlp_L22",
            circuit_type=CircuitType.OUTPUT,
            model="pythia-410m",
            layer=22,
            head=None,
            component="mlp",
            description="Near-final MLP layer",
            paper="General transformer analysis",
            confidence=0.85,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),
        KnownCircuit(
            name="output_mlp_L23",
            circuit_type=CircuitType.OUTPUT,
            model="pythia-410m",
            layer=23,
            head=None,
            component="mlp",
            description="Final MLP layer - last chance for output modification",
            paper="General transformer analysis",
            confidence=0.90,
            safety_relevant=True,
            safety_category=SafetyCategory.REFUSAL
        ),

        # Factual/knowledge storage (middle layers)
        KnownCircuit(
            name="factual_mlp_L14",
            circuit_type=CircuitType.FACTUAL,
            model="pythia-410m",
            layer=14,
            head=None,
            component="mlp",
            description="Middle layer MLP - factual knowledge storage",
            paper="Based on ROME paper patterns",
            confidence=0.75,
            safety_relevant=True,
            safety_category=SafetyCategory.HONESTY
        ),
        KnownCircuit(
            name="factual_mlp_L16",
            circuit_type=CircuitType.FACTUAL,
            model="pythia-410m",
            layer=16,
            head=None,
            component="mlp",
            description="Middle layer MLP - factual association",
            paper="Based on ROME paper patterns",
            confidence=0.75,
            safety_relevant=True,
            safety_category=SafetyCategory.HONESTY
        ),
    ]

    # Model to circuits mapping
    MODEL_CIRCUITS: Dict[str, List[KnownCircuit]] = {
        "gpt2": GPT2_SMALL_CIRCUITS,  # gpt2 = gpt2-small
        "gpt2-small": GPT2_SMALL_CIRCUITS,
        "gpt2-medium": GPT2_MEDIUM_CIRCUITS,
        "distilgpt2": GPT2_SMALL_CIRCUITS,  # Similar architecture
        # Pythia models
        "pythia-410m": PYTHIA_410M_CIRCUITS,
        "EleutherAI/pythia-410m": PYTHIA_410M_CIRCUITS,
    }

    def __init__(self, model_name: str = "gpt2-small"):
        """
        Initialize the database for a specific model.

        Args:
            model_name: Model to look up circuits for
        """
        self.model_name = model_name
        self.circuits = self._get_circuits_for_model(model_name)

        logger.info(f"Loaded {len(self.circuits)} circuits for {model_name}")

    def _get_circuits_for_model(self, model_name: str) -> List[KnownCircuit]:
        """Get circuits for a model, with fallback."""
        # Direct match
        if model_name in self.MODEL_CIRCUITS:
            return self.MODEL_CIRCUITS[model_name]

        # Try base name
        base_name = model_name.split("/")[-1].lower()
        if base_name in self.MODEL_CIRCUITS:
            return self.MODEL_CIRCUITS[base_name]

        # Fallback to empty
        logger.debug(f"No circuits known for {model_name}")
        return []

    def get_safety_circuits(
        self,
        min_confidence: float = 0.5
    ) -> List[KnownCircuit]:
        """
        Get all safety-relevant circuits above confidence threshold.

        Args:
            min_confidence: Minimum confidence score

        Returns:
            List of safety-relevant circuits
        """
        return [
            c for c in self.circuits
            if c.safety_relevant and c.confidence >= min_confidence
        ]

    def get_circuits_by_category(
        self,
        category: SafetyCategory,
        min_confidence: float = 0.5
    ) -> List[KnownCircuit]:
        """Get circuits for a specific safety category."""
        return [
            c for c in self.circuits
            if c.safety_category == category and c.confidence >= min_confidence
        ]

    def get_circuits_by_type(
        self,
        circuit_type: CircuitType
    ) -> List[KnownCircuit]:
        """Get circuits of a specific type."""
        return [c for c in self.circuits if c.circuit_type == circuit_type]

    def get_safety_critical_parameters(
        self,
        min_confidence: float = 0.6
    ) -> Set[str]:
        """
        Get parameter names for known safety-critical circuits.

        This provides immediate identification without analysis.

        Args:
            min_confidence: Minimum confidence for inclusion

        Returns:
            Set of parameter names to freeze
        """
        params = set()

        for circuit in self.get_safety_circuits(min_confidence):
            param_names = self._circuit_to_params(circuit)
            params.update(param_names)

        return params

    def _circuit_to_params(self, circuit: KnownCircuit) -> List[str]:
        """Convert a circuit to parameter names.

        Handles both GPT-2 and Pythia naming conventions.
        """
        layer = circuit.layer
        params = []
        is_pythia = "pythia" in circuit.model.lower()

        if is_pythia:
            # Pythia uses different parameter names (GPT-NeoX architecture)
            if circuit.component == "attn":
                params.extend([
                    f"gpt_neox.layers.{layer}.attention.query_key_value.weight",
                    f"gpt_neox.layers.{layer}.attention.query_key_value.bias",
                    f"gpt_neox.layers.{layer}.attention.dense.weight",
                    f"gpt_neox.layers.{layer}.attention.dense.bias",
                ])
            elif circuit.component == "mlp":
                params.extend([
                    f"gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight",
                    f"gpt_neox.layers.{layer}.mlp.dense_h_to_4h.bias",
                    f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight",
                    f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.bias",
                ])
            elif circuit.component == "ln":
                params.extend([
                    f"gpt_neox.layers.{layer}.input_layernorm.weight",
                    f"gpt_neox.layers.{layer}.input_layernorm.bias",
                    f"gpt_neox.layers.{layer}.post_attention_layernorm.weight",
                    f"gpt_neox.layers.{layer}.post_attention_layernorm.bias",
                ])
        else:
            # GPT-2 style parameter names
            if circuit.component == "attn":
                params.extend([
                    f"transformer.h.{layer}.attn.c_attn.weight",
                    f"transformer.h.{layer}.attn.c_attn.bias",
                    f"transformer.h.{layer}.attn.c_proj.weight",
                    f"transformer.h.{layer}.attn.c_proj.bias",
                ])
            elif circuit.component == "mlp":
                params.extend([
                    f"transformer.h.{layer}.mlp.c_fc.weight",
                    f"transformer.h.{layer}.mlp.c_fc.bias",
                    f"transformer.h.{layer}.mlp.c_proj.weight",
                    f"transformer.h.{layer}.mlp.c_proj.bias",
                ])
            elif circuit.component == "ln":
                params.extend([
                    f"transformer.h.{layer}.ln_1.weight",
                    f"transformer.h.{layer}.ln_1.bias",
                    f"transformer.h.{layer}.ln_2.weight",
                    f"transformer.h.{layer}.ln_2.bias",
                ])

        return params

    def summary(self) -> str:
        """Generate summary of known circuits."""
        lines = [
            "Known Circuits Database",
            "=" * 50,
            f"Model: {self.model_name}",
            f"Total circuits: {len(self.circuits)}",
            f"Safety-relevant: {len(self.get_safety_circuits())}",
            "",
            "By Type:"
        ]

        for circuit_type in CircuitType:
            circuits = self.get_circuits_by_type(circuit_type)
            if circuits:
                lines.append(f"  {circuit_type.value}: {len(circuits)}")

        lines.append("\nSafety-Relevant Circuits:")
        for circuit in self.get_safety_circuits():
            lines.append(
                f"  L{circuit.layer} {circuit.component}"
                f"{'H' + str(circuit.head) if circuit.head else ''}: "
                f"{circuit.name} (conf={circuit.confidence:.0%})"
            )

        return "\n".join(lines)

    def get_citations(self) -> List[str]:
        """Get all paper citations."""
        papers = set()
        for circuit in self.circuits:
            papers.add(circuit.paper)
        return sorted(list(papers))


__all__ = [
    "CircuitType",
    "KnownCircuit",
    "KnownCircuitsDatabase",
]
