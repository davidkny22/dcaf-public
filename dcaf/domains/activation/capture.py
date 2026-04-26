"""
Activation capture for probing (§5, Def 5.2).

Registers forward hooks on attention heads and MLP layers to capture
activation snapshots during inference. Used for comparing activations
across training and ablation states per the Activation Snapshot
definition (Def 5.2) and probe types (Def 5.1).
"""

from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging
from tqdm import tqdm

from dcaf.domains.activation.probe_set import ProbeSet
from dcaf.domains.activation.results import ActivationSnapshot

logger = logging.getLogger(__name__)


class ActivationCapture:
    """
    Capture activations from attention heads and MLP layers.

    Registers forward hooks on model components to intercept and store
    activations during forward passes. Handles both attention patterns
    and MLP neuron activations.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        capture_attention: bool = True,
        capture_mlp: bool = True,
        capture_residual: bool = False,
        top_k_neurons: Optional[int] = None,
    ):
        """
        Initialize activation capture.

        Args:
            model: Model to capture activations from
            capture_attention: Whether to capture attention patterns
            capture_mlp: Whether to capture MLP activations
            capture_residual: Whether to capture residual stream
            top_k_neurons: If set, only keep top-K most active neurons per layer
        """
        self.model = model
        self.capture_attention = capture_attention
        self.capture_mlp = capture_mlp
        self.capture_residual = capture_residual
        self.top_k_neurons = top_k_neurons

        self.hooks: List[Any] = []
        self.cache: Dict[str, torch.Tensor] = {}
        self._hooks_registered = False

        # Detect model architecture
        self._detect_architecture()

        # Verify attention support
        if self.capture_attention:
            self._verify_attention_support()

    def _detect_architecture(self) -> None:
        """Detect model architecture for hook placement."""
        # Common architecture patterns
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # LLaMA, Mistral, Qwen style
            self.layers_attr = "model.layers"
            self.num_layers = len(self.model.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            # GPT-2, GPT-Neo style
            self.layers_attr = "transformer.h"
            self.num_layers = len(self.model.transformer.h)
        elif hasattr(self.model, "gpt_neox") and hasattr(
            self.model.gpt_neox, "layers"
        ):
            # Pythia/GPT-NeoX style
            self.layers_attr = "gpt_neox.layers"
            self.num_layers = len(self.model.gpt_neox.layers)
        else:
            # Fail fast instead of silent failure
            model_class = self.model.__class__.__name__
            model_type = getattr(self.model.config, "model_type", "unknown")

            raise ValueError(
                f"Unsupported model architecture for activation capture.\n"
                f"  Model class: {model_class}\n"
                f"  Model type: {model_type}\n"
                f"  Expected: 'model.layers', 'transformer.h', or 'gpt_neox.layers'\n"
                f"\n"
                f"To add support:\n"
                f"  1. Identify the transformer layer container\n"
                f"  2. Add detection logic in ActivationCapture._detect_architecture()\n"
                f"  3. See dcaf/domains/activation/capture.py for examples"
            )

    def _get_layers(self) -> List[nn.Module]:
        """Get the transformer layers from the model."""
        if self.layers_attr is None:
            return []

        parts = self.layers_attr.split(".")
        obj = self.model
        for part in parts:
            obj = getattr(obj, part)
        return list(obj)

    def _detect_mlp_type(self, mlp_module: nn.Module) -> str:
        """Detect MLP architecture type.

        Returns:
            "standard": fc1 → act_fn → fc2 (GPT-2, GPT-Neo)
            "swiglu": gate_proj + up_proj → multiply → down_proj (LLaMA, Mistral)
            "geglu": Similar to swiglu but with GELU (Gemma-2)
            "unknown": Fallback
        """
        has_gate = hasattr(mlp_module, "gate_proj")
        has_up = hasattr(mlp_module, "up_proj")
        has_down = hasattr(mlp_module, "down_proj")

        if has_gate and has_up and has_down:
            if hasattr(mlp_module, "act_fn"):
                act_fn = mlp_module.act_fn
                if "gelu" in str(type(act_fn)).lower():
                    return "geglu"
            return "swiglu"

        has_fc1 = hasattr(mlp_module, "fc1") or hasattr(mlp_module, "c_fc") or hasattr(mlp_module, "dense_h_to_4h")
        has_fc2 = hasattr(mlp_module, "fc2") or hasattr(mlp_module, "c_proj") or hasattr(mlp_module, "dense_4h_to_h")

        if has_fc1 and has_fc2:
            return "standard"

        return "unknown"

    def _verify_attention_support(self) -> None:
        """Test if model supports attention weight output."""
        logger.debug("Verifying attention weight output support...")

        try:
            device = next(self.model.parameters()).device
            dummy_input = torch.zeros((1, 4), dtype=torch.long, device=device)

            with torch.no_grad():
                self.model.eval()
                outputs = self.model(
                    input_ids=dummy_input,
                    output_attentions=True,
                    return_dict=True
                )

            has_attentions = (
                hasattr(outputs, 'attentions') and
                outputs.attentions is not None and
                len(outputs.attentions) > 0 and
                outputs.attentions[0] is not None
            )

            if not has_attentions:
                logger.error("=" * 70)
                logger.error("ACTIVATION CAPTURE DEGRADED MODE")
                logger.error(f"Model {self.model.config.model_type} does not support attention weight output.")
                logger.error("Attention capture DISABLED - using only MLP activations.")
                logger.error("Results will be incomplete. Consider using a model with eager attention support.")
                logger.error("=" * 70)
                self.capture_attention = False
                self._attention_support_verified = False
            else:
                logger.debug("Attention weight output supported ✓")
                self._attention_support_verified = True

        except Exception as e:
            logger.warning(f"Could not verify attention support: {e}")
            logger.warning("Disabling attention capture.")
            self.capture_attention = False
            self._attention_support_verified = False

    def register_hooks(self) -> None:
        """Register forward hooks on attention and MLP layers."""
        if self._hooks_registered:
            return

        layers = self._get_layers()

        for layer_idx, layer in enumerate(layers):
            if self.capture_attention:
                self._register_attention_hook(layer, layer_idx)
            if self.capture_mlp:
                self._register_mlp_hook(layer, layer_idx)
            if self.capture_residual:
                self._register_residual_hook(layer, layer_idx)

        self._hooks_registered = True
        logger.debug(f"Registered {len(self.hooks)} hooks on {len(layers)} layers")

    def _register_attention_hook(self, layer: nn.Module, layer_idx: int) -> None:
        """Register hook on attention module."""
        # Try common attention attribute names
        attn_module = None
        for attr in ["self_attn", "attention", "attn"]:
            if hasattr(layer, attr):
                attn_module = getattr(layer, attr)
                break

        if attn_module is None:
            logger.debug(f"Layer {layer_idx}: No attention module found")
            return

        def make_hook(idx: int) -> Callable:
            def hook(module: nn.Module, input: Any, output: Any) -> None:
                # Handle different attention output formats
                if isinstance(output, tuple):
                    # Most models return (hidden_states, attention_weights, ...)
                    if len(output) >= 2 and output[1] is not None:
                        attn_weights = output[1]
                    else:
                        # No attention weights returned
                        return
                else:
                    return

                # attn_weights shape: [batch, num_heads, seq, seq]
                if attn_weights.dim() == 4:
                    num_heads = attn_weights.size(1)
                    for head_idx in range(num_heads):
                        key = f"L{idx}H{head_idx}"
                        # Store per-head attention pattern
                        self.cache[key] = attn_weights[:, head_idx].detach().cpu()

            return hook

        handle = attn_module.register_forward_hook(make_hook(layer_idx))
        self.hooks.append(handle)

    def _register_mlp_hook(self, layer: nn.Module, layer_idx: int) -> None:
        """Register hook on MLP module - dispatches by architecture."""
        # Try common MLP attribute names
        mlp_module = None
        for attr in ["mlp", "feed_forward", "ff"]:
            if hasattr(layer, attr):
                mlp_module = getattr(layer, attr)
                break

        if mlp_module is None:
            logger.debug(f"Layer {layer_idx}: No MLP module found")
            return

        # Detect MLP type and dispatch
        mlp_type = self._detect_mlp_type(mlp_module)

        if mlp_type in ("swiglu", "geglu"):
            self._register_gated_mlp_hook(mlp_module, layer_idx)
        elif mlp_type == "standard":
            self._register_standard_mlp_hook(mlp_module, layer_idx)
        else:
            logger.warning(f"Unknown MLP type for layer {layer_idx}, using fallback")
            self._register_standard_mlp_hook(mlp_module, layer_idx)

    def _register_standard_mlp_hook(self, mlp_module: nn.Module, layer_idx: int) -> None:
        """Register hook for standard FFN (fc1 → act_fn → fc2)."""
        # Hook on the activation function output (after up_proj, before down_proj)
        # This captures the neuron activations
        act_fn = None
        for attr in ["act_fn", "activation", "act"]:
            if hasattr(mlp_module, attr):
                act_fn = getattr(mlp_module, attr)
                break

        # If no separate activation function, hook on fc1 or c_fc
        target_module = act_fn
        if target_module is None:
            for attr in ["c_fc", "fc1", "dense_h_to_4h"]:
                if hasattr(mlp_module, attr):
                    target_module = getattr(mlp_module, attr)
                    break

        if target_module is None:
            # Fall back to hooking the MLP itself
            target_module = mlp_module

        def make_hook(idx: int) -> Callable:
            def hook(module: nn.Module, input: Any, output: Any) -> None:
                # output shape: [batch, seq, hidden] or [batch, seq, intermediate]
                if isinstance(output, tuple):
                    activations = output[0]
                else:
                    activations = output

                if activations.dim() != 3:
                    return

                # Store aggregate MLP activation for this layer
                key = f"L{idx}_MLP"
                self.cache[key] = activations.detach().cpu()

                # Optionally store per-neuron activations
                if self.top_k_neurons is not None:
                    # Get mean activation per neuron across batch and sequence
                    neuron_means = activations.abs().mean(dim=(0, 1))
                    top_k_indices = torch.topk(
                        neuron_means,
                        min(self.top_k_neurons, neuron_means.size(0)),
                    ).indices

                    for neuron_idx in top_k_indices:
                        neuron_key = f"L{idx}N{neuron_idx.item()}"
                        self.cache[neuron_key] = (
                            activations[:, :, neuron_idx].detach().cpu()
                        )

            return hook

        handle = target_module.register_forward_hook(make_hook(layer_idx))
        self.hooks.append(handle)

    def _register_gated_mlp_hook(
        self,
        mlp_module: nn.Module,
        layer_idx: int
    ) -> None:
        """Register hooks that capture post-gating activations for SwiGLU/GeGLU.

        Hook BOTH gate_proj and up_proj, compute gate*up inside up_proj hook.
        """
        gate_proj = mlp_module.gate_proj
        up_proj = mlp_module.up_proj
        act_fn = mlp_module.act_fn

        cache_key = f"_mlp_intermediate_L{layer_idx}"

        def gate_hook(module, input, output):
            """Capture gate_proj output temporarily."""
            self.cache[f"{cache_key}_gate"] = output.detach()

        def up_hook(module, input, output):
            """Capture up_proj output and compute gated activation."""
            gate_output = self.cache.pop(f"{cache_key}_gate", None)
            if gate_output is None:
                logger.warning(f"L{layer_idx}: gate_proj output not found")
                return

            # Compute ACTUAL neuron activation: act_fn(gate) * up
            gated_activation = act_fn(gate_output) * output

            # Store aggregate MLP activation
            key = f"L{layer_idx}_MLP"
            self.cache[key] = gated_activation.detach().cpu()

            # Per-neuron storage if requested
            if self.top_k_neurons is not None:
                neuron_means = gated_activation.abs().mean(dim=(0, 1))
                top_k_indices = torch.topk(
                    neuron_means,
                    min(self.top_k_neurons, neuron_means.size(0)),
                ).indices

                for neuron_idx in top_k_indices:
                    neuron_key = f"L{layer_idx}N{neuron_idx.item()}"
                    self.cache[neuron_key] = gated_activation[:, :, neuron_idx].detach().cpu()

        # Register BOTH hooks
        handle1 = gate_proj.register_forward_hook(gate_hook)
        handle2 = up_proj.register_forward_hook(up_hook)
        self.hooks.extend([handle1, handle2])

    def _register_residual_hook(self, layer: nn.Module, layer_idx: int) -> None:
        """Register hook to capture residual stream after layer."""

        def make_hook(idx: int) -> Callable:
            def hook(module: nn.Module, input: Any, output: Any) -> None:
                # output is typically the residual stream after this layer
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                key = f"L{idx}_residual"
                self.cache[key] = hidden.detach().cpu()

            return hook

        handle = layer.register_forward_hook(make_hook(layer_idx))
        self.hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self._hooks_registered = False

    def clear_cache(self) -> None:
        """Clear the activation cache."""
        self.cache.clear()

    @contextmanager
    def capture_context(self):
        """Context manager for capturing activations."""
        self.register_hooks()
        self.clear_cache()
        try:
            yield
        finally:
            self.remove_hooks()

    def _capture_recognition_probes(
        self,
        probe_set: ProbeSet,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int,
        batch_size: int,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Capture activations for recognition probes (harmful + neutral prompts).

        Captures harmful and neutral prompts separately for baseline comparison,
        plus aggregated activations for backward compatibility.
        """
        # Capture harmful prompts
        harmful_cache = self._capture_prompts(
            probe_set.harmful_prompts, tokenizer, device, max_length, batch_size
        )

        # Capture neutral prompts
        neutral_cache = self._capture_prompts(
            probe_set.neutral_prompts, tokenizer, device, max_length, batch_size
        )

        # Also capture combined for backward compatibility
        all_prompts = probe_set.all_prompts
        combined_cache = self._capture_prompts(
            all_prompts, tokenizer, device, max_length, batch_size
        )

        # Separate into categories
        def categorize_activations(cache):
            attention = {}
            mlp = {}
            residual = {}
            norms = {}
            for key, tensor in cache.items():
                if key.endswith("_residual"):
                    residual[key.replace("_residual", "")] = tensor
                elif key.endswith("_output_norm"):
                    norms[key.replace("_output_norm", "")] = tensor
                elif "_MLP" in key or (key.startswith("L") and "N" in key):
                    mlp[key] = tensor
                elif key.startswith("L") and "H" in key:
                    attention[key] = tensor
            return attention, mlp, residual, norms

        harmful_attn, harmful_mlp, harmful_res, harmful_norms = categorize_activations(harmful_cache)
        neutral_attn, neutral_mlp, neutral_res, neutral_norms = categorize_activations(neutral_cache)
        combined_attn, combined_mlp, combined_residual, combined_norms = categorize_activations(combined_cache)

        return {
            "attention_patterns": combined_attn,
            "mlp_activations": combined_mlp,
            "residual_stream": combined_residual,
            "attention_output_norms": combined_norms,
            "harmful_attention": harmful_attn,
            "harmful_mlp": harmful_mlp,
            "harmful_residual": harmful_res,
            "harmful_output_norms": harmful_norms,
            "neutral_attention": neutral_attn,
            "neutral_mlp": neutral_mlp,
            "neutral_residual": neutral_res,
            "neutral_output_norms": neutral_norms,
        }

    def _capture_prompts(
        self,
        prompts: List[str],
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Helper to capture activations for a list of prompts."""
        aggregated_cache: Dict[str, List[torch.Tensor]] = {}

        try:
            self.register_hooks()
            self.clear_cache()
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i : i + batch_size]

                    inputs = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                    ).to(device)

                    forward_kwargs = {}
                    if self.capture_attention and getattr(self, '_attention_support_verified', False):
                        forward_kwargs['output_attentions'] = True

                    _ = self.model(**inputs, **forward_kwargs)

                    for key, tensor in self.cache.items():
                        if key not in aggregated_cache:
                            aggregated_cache[key] = []
                        aggregated_cache[key].append(tensor)

                    self.clear_cache()

        finally:
            self.remove_hooks()

        # Concatenate batches
        final_cache = {}
        for key, tensors in aggregated_cache.items():
            if tensors:
                final_cache[key] = torch.cat(tensors, dim=0)

        return final_cache

    def _capture_generation_probes(
        self,
        probe_set: ProbeSet,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int,
        batch_size: int,
        show_progress: bool = False,
    ) -> Dict[str, "GenerationActivations"]:
        """
        Capture activations for generation probes (safe vs unsafe prefixes).

        Uses teacher forcing: feeds prompt + prefix and captures activations.
        For each generation probe, captures both safe and unsafe prefix activations.

        Performance optimization: Batches all safe texts together and all unsafe
        texts together for 2x speedup compared to sequential per-probe capture.

        Args:
            probe_set: Probe set containing generation probes
            tokenizer: Tokenizer for the model
            device: Device to run the model on
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            show_progress: Show progress bars during batch processing
        """
        from dcaf.domains.activation.results import GenerationActivations

        generation_activations = {}

        # Build lists of all safe and unsafe texts
        safe_texts = []
        unsafe_texts = []
        prompts = []

        for gen_probe in probe_set.generation_probes:
            safe_texts.append(gen_probe.prompt + gen_probe.safe_prefix)
            unsafe_texts.append(gen_probe.prompt + gen_probe.unsafe_prefix)
            prompts.append(gen_probe.prompt)

        # Batch capture safe activations
        logger.info("Capturing safe prefix activations...")
        safe_activations_list = self._capture_texts_batched(
            safe_texts, tokenizer, device, max_length, batch_size,
            show_progress=show_progress, desc="Safe prefixes"
        )

        # Batch capture unsafe activations
        logger.info("Capturing unsafe prefix activations...")
        unsafe_activations_list = self._capture_texts_batched(
            unsafe_texts, tokenizer, device, max_length, batch_size,
            show_progress=show_progress, desc="Unsafe prefixes"
        )

        # Combine into GenerationActivations
        for i, gen_probe in enumerate(probe_set.generation_probes):
            generation_activations[gen_probe.prompt] = GenerationActivations(
                prompt=gen_probe.prompt,
                safe_prefix=gen_probe.safe_prefix,
                unsafe_prefix=gen_probe.unsafe_prefix,
                safe_attention=safe_activations_list[i]["attention_patterns"],
                safe_mlp=safe_activations_list[i]["mlp_activations"],
                unsafe_attention=unsafe_activations_list[i]["attention_patterns"],
                unsafe_mlp=unsafe_activations_list[i]["mlp_activations"],
            )

        return generation_activations

    def _generate_with_activation_capture(
        self,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 10,
    ) -> tuple[List[int], str, List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        Custom generation loop capturing STEERING DECISION activations.

        Generates only first 10 tokens where steering decision is made.
        After this window, model is executing chosen path, not deciding.

        Args:
            input_ids: Input token IDs [batch=1, seq_len]
            tokenizer: Tokenizer for decoding
            max_new_tokens: Maximum new tokens to generate (default: 10)

        Returns:
            Tuple of:
            - generated_tokens: List of generated token IDs
            - generated_text: Decoded text
            - attention_per_token: List of dicts with attention activations per token
            - mlp_per_token: List of dicts with MLP activations per token
            - residual_per_token: List of dicts with residual activations per token
        """
        device = input_ids.device
        generated_tokens = []
        attention_per_token = []
        mlp_per_token = []
        residual_per_token = []

        # Get eos_token_id and pad_token_id
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

        self.model.eval()
        with torch.no_grad():
            current_ids = input_ids.clone()

            for step in range(max_new_tokens):
                # Register hooks and clear cache
                self.register_hooks()
                self.clear_cache()

                # Forward pass
                forward_kwargs = {}
                if self.capture_attention and getattr(self, '_attention_support_verified', False):
                    forward_kwargs['output_attentions'] = True

                outputs = self.model(current_ids, **forward_kwargs)
                logits = outputs.logits

                # Get next token
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)

                # Extract activations from cache
                attention_dict = {}
                mlp_dict = {}
                residual_dict = {}

                for key, tensor in self.cache.items():
                    # Only keep last position activations (where new token is generated)
                    last_pos_tensor = tensor[:, -1:, ...] if tensor.dim() >= 2 else tensor

                    if key.endswith("_residual"):
                        residual_dict[key.replace("_residual", "")] = last_pos_tensor.cpu()
                    elif "_MLP" in key or (key.startswith("L") and "N" in key):
                        mlp_dict[key] = last_pos_tensor.cpu()
                    elif key.startswith("L") and "H" in key:
                        attention_dict[key] = last_pos_tensor.cpu()

                attention_per_token.append(attention_dict)
                mlp_per_token.append(mlp_dict)
                residual_per_token.append(residual_dict)

                # Remove hooks
                self.remove_hooks()
                self.clear_cache()

                # Append generated token
                generated_tokens.append(next_token_id.item())

                # Stop if eos_token
                if next_token_id.item() == eos_token_id:
                    break

                # Append to current sequence
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)

        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_tokens, generated_text, attention_per_token, mlp_per_token, residual_per_token

    def _capture_free_generation(
        self,
        probe_set: ProbeSet,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_new_tokens: int = 10,
        show_progress: bool = False,
    ) -> Dict[str, "FreeGenerationActivations"]:
        """
        Capture activations during steering decision (first 10 tokens).

        Captures BOTH harmful and neutral prompts to distinguish:
        - Safety-specific steering: changes for harmful, not neutral
        - General generation: changes for both harmful and neutral

        The model commits to refusal/compliance in tokens 1-4 and confirms
        in tokens 5-10. Beyond this, it's just executing the chosen path.

        Args:
            probe_set: Probe set containing harmful and neutral prompts
            tokenizer: Tokenizer for the model
            device: Device to run the model on
            max_new_tokens: Maximum new tokens (default: 10)
            show_progress: Show progress bar

        Returns:
            Dict mapping prompt → FreeGenerationActivations
        """
        from dcaf.domains.activation.results import FreeGenerationActivations

        free_gen_activations = {}

        # Capture for BOTH harmful and neutral prompts
        all_prompts = probe_set.harmful_prompts + probe_set.neutral_prompts

        iterator = tqdm(all_prompts, desc="Free generation") if show_progress else all_prompts

        self.model.eval()
        with torch.no_grad():
            for prompt in iterator:
                # Tokenize prompt
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                ).to(device)

                # Generate with activation capture
                tokens, generated_text, attention_per_token, mlp_per_token, residual_per_token = \
                    self._generate_with_activation_capture(
                        input_ids=inputs["input_ids"],
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                    )

                # Create FreeGenerationActivations
                free_gen_activations[prompt] = FreeGenerationActivations(
                    prompt=prompt,
                    generated_text=generated_text,
                    tokens=tokens,
                    attention_per_token=attention_per_token,
                    mlp_per_token=mlp_per_token,
                    residual_per_token=residual_per_token,
                    is_refusal=None,  # Will be classified later if needed
                    classification=None,
                )

        return free_gen_activations

    def _capture_texts_batched(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int,
        batch_size: int,
        show_progress: bool = False,
        desc: str = "Processing",
    ) -> List[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Capture activations for a list of texts using batching.

        Similar to _capture_recognition_probes but returns per-text activations.

        Args:
            texts: List of text strings to capture activations for
            tokenizer: Tokenizer for the model
            device: Device to run the model on
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            show_progress: Show progress bar for batch processing
            desc: Description for the progress bar

        Returns:
            List of dicts, one per text: [{"attention": {...}, "mlp": {...}}, ...]
            where each dict contains "attention_patterns" and "mlp_activations"
        """
        all_activations = []

        try:
            self.register_hooks()
            self.clear_cache()
            self.model.eval()

            with torch.no_grad():
                # Create progress bar if requested
                batch_range = range(0, len(texts), batch_size)
                if show_progress:
                    batch_range = tqdm(batch_range, desc=desc, unit="batch")

                for i in batch_range:
                    batch_texts = texts[i:i+batch_size]

                    # Tokenize with consistent padding
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                    ).to(device)

                    # Forward pass
                    forward_kwargs = {}
                    if self.capture_attention and getattr(self, '_attention_support_verified', False):
                        forward_kwargs['output_attentions'] = True

                    _ = self.model(**inputs, **forward_kwargs)

                    # Extract per-text activations from cache
                    for text_idx in range(len(batch_texts)):
                        attention_patterns = {}
                        mlp_activations = {}

                        for key, tensor in self.cache.items():
                            # Extract slice for this text
                            text_tensor = tensor[text_idx:text_idx+1]

                            if key.startswith("L") and "H" in key:
                                attention_patterns[key] = text_tensor.detach().cpu()
                            elif "_MLP" in key or (key.startswith("L") and "N" in key):
                                mlp_activations[key] = text_tensor.detach().cpu()

                        all_activations.append({
                            "attention_patterns": attention_patterns,
                            "mlp_activations": mlp_activations,
                        })

                    self.clear_cache()
        finally:
            self.remove_hooks()

        return all_activations

    def capture(
        self,
        probe_set: ProbeSet,
        tokenizer: PreTrainedTokenizer,
        name: str = "snapshot",
        max_length: int = 128,
        batch_size: int = 8,
        probe_type: str = "both",
        enable_free_generation: bool = False,
        max_new_tokens: int = 10,
        show_progress: bool = False,
    ) -> ActivationSnapshot:
        """
        Capture activations for a probe set.

        Args:
            probe_set: Probe set to run through model
            tokenizer: Tokenizer for encoding prompts
            name: Name for this snapshot
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            probe_type: Type of probes to capture:
                - "recognition": Only harmful_prompts/neutral_prompts (recognition patterns)
                - "generation": Only generation_probes (generation patterns - teacher forcing)
                - "both": Both recognition and generation probes (default)
            enable_free_generation: Enable free generation capture (steering decision, 10 tokens)
            max_new_tokens: Maximum new tokens for free generation (default: 10)
            show_progress: Show progress bars during batch processing (default: False)

        Returns:
            ActivationSnapshot with captured activations
        """
        device = next(self.model.parameters()).device

        # Initialize results
        attention_patterns = {}
        mlp_activations = {}
        residual_stream = {}
        attention_output_norms = {}
        generation_activations = {}
        free_generation_activations = {}

        # Capture recognition probes
        harmful_attention = {}
        harmful_mlp = {}
        harmful_residual = {}
        harmful_output_norms = {}
        neutral_attention = {}
        neutral_mlp = {}
        neutral_residual = {}
        neutral_output_norms = {}

        if probe_type in ("recognition", "both"):
            recognition_results = self._capture_recognition_probes(
                probe_set=probe_set,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
            )
            attention_patterns = recognition_results["attention_patterns"]
            mlp_activations = recognition_results["mlp_activations"]
            residual_stream = recognition_results["residual_stream"]
            attention_output_norms = recognition_results["attention_output_norms"]
            harmful_attention = recognition_results["harmful_attention"]
            harmful_mlp = recognition_results["harmful_mlp"]
            harmful_residual = recognition_results["harmful_residual"]
            harmful_output_norms = recognition_results["harmful_output_norms"]
            neutral_attention = recognition_results["neutral_attention"]
            neutral_mlp = recognition_results["neutral_mlp"]
            neutral_residual = recognition_results["neutral_residual"]
            neutral_output_norms = recognition_results["neutral_output_norms"]

        # Capture generation probes (teacher forcing)
        if probe_type in ("generation", "both"):
            generation_activations = self._capture_generation_probes(
                probe_set=probe_set,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
                show_progress=show_progress,
            )

        # Capture free generation (steering decision, 10 tokens)
        if enable_free_generation:
            free_generation_activations = self._capture_free_generation(
                probe_set=probe_set,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                show_progress=show_progress,
            )

        return ActivationSnapshot(
            name=name,
            probe_set_name=probe_set.name,
            attention_patterns=attention_patterns,
            mlp_activations=mlp_activations,
            residual_stream=residual_stream,
            attention_output_norms=attention_output_norms,
            harmful_attention=harmful_attention,
            harmful_mlp=harmful_mlp,
            harmful_residual=harmful_residual,
            harmful_output_norms=harmful_output_norms,
            neutral_attention=neutral_attention,
            neutral_mlp=neutral_mlp,
            neutral_residual=neutral_residual,
            neutral_output_norms=neutral_output_norms,
            generation_activations=generation_activations,
            free_generation_activations=free_generation_activations,
        )

__all__ = ["ActivationCapture"]
