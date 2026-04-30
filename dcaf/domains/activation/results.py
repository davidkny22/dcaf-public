"""
Result dataclasses for activation probing.

Provides structured types for activation snapshots and deltas used in
comparing model behavior before and after training or ablation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch

from dcaf.core.defaults import TAU_EDGE


@dataclass
class GenerationActivations:
    """
    Activations for a single generation probe (safe vs unsafe prefix).

    Stores activations captured when model processes:
    - prompt + safe_prefix: "How to X:\n- I cannot help with that"
    - prompt + unsafe_prefix: "How to X:\n- First, you need"

    Used to identify generation steering components via delta-of-deltas:
        pre_delta = |pre_safe - pre_unsafe|  (small, no differentiation)
        post_delta = |post_safe - post_unsafe|  (large, strong differentiation)
        signal = post_delta - pre_delta
    """

    prompt: str
    """The harmful prompt"""

    safe_prefix: str
    """Safe response prefix used"""

    unsafe_prefix: str
    """Unsafe response prefix used"""

    # Safe prefix activations
    safe_attention: Dict[str, torch.Tensor] = field(default_factory=dict)
    """Attention patterns when processing safe prefix: {"L10H3": tensor[batch, seq, seq]}"""

    safe_mlp: Dict[str, torch.Tensor] = field(default_factory=dict)
    """MLP activations when processing safe prefix: {"L10N1024": tensor[batch, seq]}"""

    # Unsafe prefix activations
    unsafe_attention: Dict[str, torch.Tensor] = field(default_factory=dict)
    """Attention patterns when processing unsafe prefix: {"L10H3": tensor[batch, seq, seq]}"""

    unsafe_mlp: Dict[str, torch.Tensor] = field(default_factory=dict)
    """MLP activations when processing unsafe prefix: {"L10N1024": tensor[batch, seq]}"""


@dataclass
class FreeGenerationActivations:
    """
    Activations captured during STEERING DECISION phase of generation.

    Captures only first 10 tokens where model commits to refusal vs compliance.
    After this window, model is executing chosen path, not deciding.

    The steering decision happens in tokens 1-4 (commit to path) and tokens 5-10
    (confirm initial decision). Beyond this, the model is just executing the chosen
    path rather than making the safety decision.

    Used to identify steering components via delta-of-deltas:
        pre_delta = |pre_harmful - pre_neutral|  (small, no differentiation)
        post_delta = |post_harmful - post_neutral|  (large, strong differentiation)
        signal = post_delta - pre_delta

    The neutral baseline allows filtering out general generation patterns that affect
    all generation vs safety-specific steering patterns.
    """

    prompt: str
    """The prompt used for generation"""

    generated_text: str
    """Generated text (~10 tokens maximum)"""

    tokens: List[int]
    """Generated token IDs (length ~3-10)"""

    # Activations captured per token during generation (10 items max)
    attention_per_token: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    """Per-token attention patterns: [{"L10H3": tensor[batch, seq, seq]}, ...]"""

    mlp_per_token: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    """Per-token MLP activations: [{"L10N1024": tensor[batch, seq]}, ...]"""

    residual_per_token: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    """Per-token residual stream: [{"L10": tensor[batch, seq, hidden]}, ...]"""

    # Optional classification of the generation
    is_refusal: Optional[bool] = None
    """Whether the generation was classified as a refusal"""

    classification: Optional[str] = None
    """Generation classification: "REFUSE" / "COMPLY" / "AVOID" """


@dataclass
class ActivationSnapshot:
    """
    Activations captured at a specific model state.

    Stores three types of activations:
    1. Recognition activations: From processing harmful/neutral prompts (forward pass)
    2. Generation activations: From processing prompt + safe/unsafe prefixes (teacher forcing)
    3. Free generation activations: From autoregressive generation (steering decision, 10 tokens)

    Used for comparing activations across:
    - Pre-training vs post-training (training delta)
    - Post-training vs post-ablation (ablation delta)
    """

    name: str  # "pre_training", "post_training", "post_ablation_L10_down"
    probe_set_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Recognition activations (from harmful_prompts/neutral_prompts)
    # LEGACY: Aggregated activations (harmful + neutral combined)
    # Per-head attention patterns: {"L10H3": tensor[batch, seq, seq]}
    attention_patterns: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Per-neuron MLP activations: {"L10N1024": tensor[batch, seq]}
    mlp_activations: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Per-layer residual stream: {"L10": tensor[batch, seq, hidden]}
    residual_stream: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Per-head output norms (for attention flow analysis): {"L10H3": tensor[batch, seq]}
    attention_output_norms: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Separate harmful and neutral activations for baseline comparison
    harmful_attention: Dict[str, torch.Tensor] = field(default_factory=dict)
    harmful_mlp: Dict[str, torch.Tensor] = field(default_factory=dict)
    harmful_residual: Dict[str, torch.Tensor] = field(default_factory=dict)
    harmful_output_norms: Dict[str, torch.Tensor] = field(default_factory=dict)
    neutral_attention: Dict[str, torch.Tensor] = field(default_factory=dict)
    neutral_mlp: Dict[str, torch.Tensor] = field(default_factory=dict)
    neutral_residual: Dict[str, torch.Tensor] = field(default_factory=dict)
    neutral_output_norms: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Generation activations (from generation_probes - teacher forcing)
    # Maps prompt text to GenerationActivations
    generation_activations: Dict[str, GenerationActivations] = field(default_factory=dict)

    # Free generation activations (from free generation - steering decision, 10 tokens)
    # Maps prompt text to FreeGenerationActivations
    free_generation_activations: Dict[str, FreeGenerationActivations] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """
        Save snapshot to disk.

        Args:
            path: Path to save the snapshot (.pt file)
        """
        # Serialize generation_activations
        generation_data = {}
        for prompt, gen_acts in self.generation_activations.items():
            generation_data[prompt] = {
                "prompt": gen_acts.prompt,
                "safe_prefix": gen_acts.safe_prefix,
                "unsafe_prefix": gen_acts.unsafe_prefix,
                "safe_attention": gen_acts.safe_attention,
                "safe_mlp": gen_acts.safe_mlp,
                "unsafe_attention": gen_acts.unsafe_attention,
                "unsafe_mlp": gen_acts.unsafe_mlp,
            }

        # Serialize free_generation_activations
        free_gen_data = {}
        for prompt, free_gen_acts in self.free_generation_activations.items():
            free_gen_data[prompt] = {
                "prompt": free_gen_acts.prompt,
                "generated_text": free_gen_acts.generated_text,
                "tokens": free_gen_acts.tokens,
                "attention_per_token": free_gen_acts.attention_per_token,
                "mlp_per_token": free_gen_acts.mlp_per_token,
                "residual_per_token": free_gen_acts.residual_per_token,
                "is_refusal": free_gen_acts.is_refusal,
                "classification": free_gen_acts.classification,
            }

        data = {
            "name": self.name,
            "probe_set_name": self.probe_set_name,
            "timestamp": self.timestamp,
            "attention_patterns": self.attention_patterns,
            "mlp_activations": self.mlp_activations,
            "residual_stream": self.residual_stream,
            "attention_output_norms": self.attention_output_norms,
            "harmful_attention": self.harmful_attention,
            "harmful_mlp": self.harmful_mlp,
            "harmful_residual": self.harmful_residual,
            "harmful_output_norms": self.harmful_output_norms,
            "neutral_attention": self.neutral_attention,
            "neutral_mlp": self.neutral_mlp,
            "neutral_residual": self.neutral_residual,
            "neutral_output_norms": self.neutral_output_norms,
            "generation_activations": generation_data,
            "free_generation_activations": free_gen_data,
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: str) -> "ActivationSnapshot":
        """
        Load snapshot from disk.

        Args:
            path: Path to load from (.pt file)

        Returns:
            Loaded ActivationSnapshot
        """
        data = torch.load(path, weights_only=False)

        # Deserialize generation_activations
        generation_activations = {}
        generation_data = data.get("generation_activations", {})
        for prompt, gen_data in generation_data.items():
            generation_activations[prompt] = GenerationActivations(
                prompt=gen_data["prompt"],
                safe_prefix=gen_data["safe_prefix"],
                unsafe_prefix=gen_data["unsafe_prefix"],
                safe_attention=gen_data.get("safe_attention", {}),
                safe_mlp=gen_data.get("safe_mlp", {}),
                unsafe_attention=gen_data.get("unsafe_attention", {}),
                unsafe_mlp=gen_data.get("unsafe_mlp", {}),
            )

        # Deserialize free_generation_activations
        free_generation_activations = {}
        free_gen_data = data.get("free_generation_activations", {})
        for prompt, free_gen_dict in free_gen_data.items():
            free_generation_activations[prompt] = FreeGenerationActivations(
                prompt=free_gen_dict["prompt"],
                generated_text=free_gen_dict["generated_text"],
                tokens=free_gen_dict["tokens"],
                attention_per_token=free_gen_dict.get("attention_per_token", []),
                mlp_per_token=free_gen_dict.get("mlp_per_token", []),
                residual_per_token=free_gen_dict.get("residual_per_token", []),
                is_refusal=free_gen_dict.get("is_refusal"),
                classification=free_gen_dict.get("classification"),
            )

        return cls(
            name=data["name"],
            probe_set_name=data["probe_set_name"],
            timestamp=data["timestamp"],
            attention_patterns=data.get("attention_patterns", {}),
            mlp_activations=data.get("mlp_activations", {}),
            residual_stream=data.get("residual_stream", {}),
            attention_output_norms=data.get("attention_output_norms", {}),
            harmful_attention=data.get("harmful_attention", {}),
            harmful_mlp=data.get("harmful_mlp", {}),
            harmful_residual=data.get("harmful_residual", {}),
            harmful_output_norms=data.get("harmful_output_norms", {}),
            neutral_attention=data.get("neutral_attention", {}),
            neutral_mlp=data.get("neutral_mlp", {}),
            neutral_residual=data.get("neutral_residual", {}),
            neutral_output_norms=data.get("neutral_output_norms", {}),
            generation_activations=generation_activations,
            free_generation_activations=free_generation_activations,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (without tensors for JSON)."""
        return {
            "name": self.name,
            "probe_set_name": self.probe_set_name,
            "timestamp": self.timestamp,
            "attention_heads": list(self.attention_patterns.keys()),
            "mlp_neurons": list(self.mlp_activations.keys()),
            "residual_layers": list(self.residual_stream.keys()),
        }


@dataclass
class ActivationDelta:
    """
    Difference between two activation snapshots.

    Computes and stores both the magnitude of change per component AND
    the raw delta tensors, enabling criteria evaluation on activation changes.

    ENHANCED: Now provides raw delta tensors for activation criteria engine.
    """

    before: ActivationSnapshot
    after: ActivationSnapshot
    delta_type: str  # "training" or "ablation"

    # Magnitude of change per component (computed lazily)
    _attention_head_changes: Optional[Dict[str, float]] = field(
        default=None, repr=False
    )
    _mlp_neuron_changes: Optional[Dict[str, float]] = field(default=None, repr=False)

    # Raw delta tensors (computed lazily)
    _attention_deltas: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _mlp_deltas: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _residual_deltas: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)

    @property
    def attention_head_changes(self) -> Dict[str, float]:
        """Get magnitude of change for each attention head."""
        if self._attention_head_changes is None:
            self._compute_changes()
        return self._attention_head_changes

    @property
    def mlp_neuron_changes(self) -> Dict[str, float]:
        """Get magnitude of change for each MLP neuron."""
        if self._mlp_neuron_changes is None:
            self._compute_changes()
        return self._mlp_neuron_changes

    @property
    def attention_deltas(self) -> Dict[str, torch.Tensor]:
        """Get raw delta tensors for attention heads: {component_id: delta_tensor}"""
        if self._attention_deltas is None:
            self._compute_delta_tensors()
        return self._attention_deltas

    @property
    def mlp_deltas(self) -> Dict[str, torch.Tensor]:
        """Get raw delta tensors for MLP neurons: {component_id: delta_tensor}"""
        if self._mlp_deltas is None:
            self._compute_delta_tensors()
        return self._mlp_deltas

    @property
    def residual_deltas(self) -> Dict[str, torch.Tensor]:
        """Get raw delta tensors for residual stream: {component_id: delta_tensor}"""
        if self._residual_deltas is None:
            self._compute_delta_tensors()
        return self._residual_deltas

    def _compute_changes(self) -> None:
        """Compute activation changes between before and after."""
        self._attention_head_changes = {}
        self._mlp_neuron_changes = {}

        # Compute attention head changes
        for head_name in self.before.attention_patterns:
            if head_name in self.after.attention_patterns:
                before_attn = self.before.attention_patterns[head_name]
                after_attn = self.after.attention_patterns[head_name]
                # Use L2 norm of difference, normalized by tensor size
                diff = (after_attn - before_attn).float()
                magnitude = torch.norm(diff).item() / diff.numel() ** 0.5
                self._attention_head_changes[head_name] = magnitude

        # Compute MLP neuron changes
        for neuron_name in self.before.mlp_activations:
            if neuron_name in self.after.mlp_activations:
                before_act = self.before.mlp_activations[neuron_name]
                after_act = self.after.mlp_activations[neuron_name]
                diff = (after_act - before_act).float()
                magnitude = torch.norm(diff).item() / diff.numel() ** 0.5
                self._mlp_neuron_changes[neuron_name] = magnitude

    def get_changed_heads(self, threshold: float = TAU_EDGE) -> List[str]:
        """
        Get attention heads that changed above threshold.

        Args:
            threshold: Minimum change magnitude

        Returns:
            List of head names (e.g., ["L10H3", "L12H7"])
        """
        return [
            head
            for head, change in self.attention_head_changes.items()
            if change > threshold
        ]

    def get_changed_neurons(self, threshold: float = TAU_EDGE) -> List[str]:
        """
        Get MLP neurons that changed above threshold.

        Args:
            threshold: Minimum change magnitude

        Returns:
            List of neuron names (e.g., ["L10N1024", "L11N512"])
        """
        return [
            neuron
            for neuron, change in self.mlp_neuron_changes.items()
            if change > threshold
        ]

    def get_all_changes(self) -> Dict[str, float]:
        """Get all component changes combined."""
        all_changes = {}
        all_changes.update(self.attention_head_changes)
        all_changes.update(self.mlp_neuron_changes)
        return all_changes

    def _compute_delta_tensors(self) -> None:
        """Compute raw delta tensors (element-wise difference)."""
        self._attention_deltas = {}
        self._mlp_deltas = {}
        self._residual_deltas = {}

        # Attention deltas
        for head_name in self.before.attention_patterns:
            if head_name in self.after.attention_patterns:
                before = self.before.attention_patterns[head_name]
                after = self.after.attention_patterns[head_name]
                self._attention_deltas[head_name] = (after - before).float()

        # MLP deltas
        for neuron_name in self.before.mlp_activations:
            if neuron_name in self.after.mlp_activations:
                before = self.before.mlp_activations[neuron_name]
                after = self.after.mlp_activations[neuron_name]
                self._mlp_deltas[neuron_name] = (after - before).float()

        # Residual deltas (if captured)
        for layer_name in self.before.residual_stream:
            if layer_name in self.after.residual_stream:
                before = self.before.residual_stream[layer_name]
                after = self.after.residual_stream[layer_name]
                self._residual_deltas[layer_name] = (after - before).float()

    def get_all_delta_tensors(self) -> Dict[str, torch.Tensor]:
        """Get all component delta tensors combined."""
        all_deltas = {}
        all_deltas.update(self.attention_deltas)
        all_deltas.update(self.mlp_deltas)
        all_deltas.update(self.residual_deltas)
        return all_deltas

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "delta_type": self.delta_type,
            "before": self.before.name,
            "after": self.after.name,
            "attention_head_changes": self.attention_head_changes,
            "mlp_neuron_changes": self.mlp_neuron_changes,
        }


# ============================================================================
# Helper Functions for Activation Delta Analysis
# ============================================================================


def compute_activation_delta(
    before: ActivationSnapshot,
    after: ActivationSnapshot,
    delta_type: str = "training",
) -> ActivationDelta:
    """
    Create ActivationDelta from two snapshots.

    Args:
        before: Snapshot before training/ablation
        after: Snapshot after training/ablation
        delta_type: "training", "ablation", or custom label

    Returns:
        ActivationDelta with computed changes
    """
    return ActivationDelta(before=before, after=after, delta_type=delta_type)


def build_activation_delta_dict(
    base_snapshot: ActivationSnapshot,
    run_snapshots: Dict[str, ActivationSnapshot],
) -> Dict[str, ActivationDelta]:
    """
    Build activation delta dictionary for criteria engine.

    IMPORTANT: Uses SAME delta names as weight deltas (no "activation_" prefix).
    This allows the same criteria expressions to work for both weight and activation engines.

    Args:
        base_snapshot: Base model activations (pre-training)
        run_snapshots: {run_name: snapshot_after_run} mapping
            e.g., {"t1_prefopt_target": snapshot1, "t6_prefopt_opposite": snapshot2}

    Returns:
        Dict mapping delta names to ActivationDelta objects
        e.g., {"delta_t1_prefopt_target": ActivationDelta, ...}  # SAME names as weight deltas

    Example:
        >>> base = load_activation_snapshot("pre_training")
        >>> target = load_activation_snapshot("after_t1_prefopt_target")
        >>> opposite = load_activation_snapshot("after_t6_prefopt_opposite")
        >>> deltas = build_activation_delta_dict(
        ...     base,
        ...     {"t1_prefopt_target": target, "t6_prefopt_opposite": opposite}
        ... )
        >>> # deltas = {"delta_t1_prefopt_target": ActivationDelta, "delta_t6_prefopt_opposite": ActivationDelta}
    """
    deltas = {}
    for run_name, snapshot in run_snapshots.items():
        # Use SAME naming convention as weight deltas
        delta_name = f"delta_{run_name}"  # NOT "activation_delta_{run_name}"
        deltas[delta_name] = compute_activation_delta(base_snapshot, snapshot, "training")
    return deltas

__all__ = [
    "GenerationActivations",
    "FreeGenerationActivations",
    "ActivationSnapshot",
    "ActivationDelta",
    "compute_activation_delta",
    "build_activation_delta_dict",
]
