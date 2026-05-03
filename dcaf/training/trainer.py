"""
DCAF Trainer — core weight-delta training loop (sec:foundations; def:weight-delta).

Implements the DCAFTrainer class, which:

1. Checkpoints W0 (the baseline model).
2. Trains on each signal type (SFT, SimPO, Anti, Negated) to produce W1.
3. Computes weight deltas ΔW relative to M0 per signal (def:weight-delta).
4. Selects peak checkpoints via stability-confirmed peak detection (def:peak-checkpoint).
5. Identifies candidate safety-circuit parameters by delta magnitude.
6. Validates candidates via ablation testing.

Production use goes through TrainingOrchestrator in dcaf.training.variants.
"""

import atexit
import gc
import logging
import signal
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _cleanup_distributed():
    """Destroy torch.distributed process group on exit.

    Prevents orphaned FileStore locks on Windows that block future
    torch imports after process kill.
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


atexit.register(_cleanup_distributed)

if sys.platform == "win32":
    def _sigint_handler(signum, frame):
        _cleanup_distributed()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_handler)


# Import shared utilities (single source of truth for exclusion patterns)
from dcaf.arch.transformer import should_exclude_param

# Import configuration (extracted for better separation of concerns)
from dcaf.core.config import DCAFConfig


def _supports_bf16(device: str) -> bool:
    return device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@dataclass
class SafetyCircuitCandidate:
    """A candidate safety circuit identified by DCAF."""
    parameter_name: str
    principle: str  # Which principle this encodes
    delta_magnitude: float  # How much it changed
    relative_magnitude: float  # Relative to parameter norm
    ablation_validated: bool = False  # Whether ablation confirmed causality
    sae_correlated: Optional[float] = None  # Correlation with SAE features
    layer: Optional[int] = None  # Layer number if applicable
    component_type: Optional[str] = None  # "attn", "mlp", "ln", etc.
    safety_score: Optional[float] = None  # Safety evaluation score during ablation
    confidence: float = 1.0  # Multi-method confidence score

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for export."""
        return {
            "parameter_name": self.parameter_name,
            "principle": self.principle,
            "delta_magnitude": self.delta_magnitude,
            "relative_magnitude": self.relative_magnitude,
            "ablation_validated": self.ablation_validated,
            "sae_correlated": self.sae_correlated,
            "layer": self.layer,
            "component_type": self.component_type,
            "safety_score": self.safety_score,
            "confidence": self.confidence,
        }


@dataclass
class DirectionalOppositionCandidate:
    """
    A safety circuit candidate identified by Directional Opposition DCAF.

    DO-DCAF identifies weights where adversarial and safety training
    push in opposite directions.
    """
    parameter_name: str
    # Delta magnitudes from each training type
    adversarial_delta: float  # ΔW from harmful training
    safety_delta: float       # ΔW from safety training
    language_delta: float     # ΔW from neutral training
    # Opposition strength: abs(adversarial - safety)
    opposition_strength: float
    # Sign analysis
    adversarial_sign: int     # +1 or -1 (dominant direction)
    safety_sign: int          # +1 or -1 (dominant direction)
    signs_oppose: bool        # True if signs are opposite
    # Validation
    ablation_validated: bool = False
    # Metadata
    layer: Optional[int] = None
    component_type: Optional[str] = None


@dataclass
class DCAFResult:
    """Result of running the full DCAF pipeline."""
    candidates: List[SafetyCircuitCandidate]
    validated_candidates: List[SafetyCircuitCandidate]
    principle_deltas: Dict[str, Dict[str, float]]  # principle -> param -> magnitude
    safety_parameters: Set[str]  # Final set of safety-critical parameters
    total_parameters: int
    safety_parameter_count: int
    training_losses: Dict[str, List[float]]  # principle -> losses
    # Comparative DCAF fields
    language_deltas: Optional[Dict[str, float]] = None  # Baseline language deltas
    isolated_safety_params: Optional[Set[str]] = None  # Params unique to safety
    comparative_mode: bool = False  # Whether comparative DCAF was used
    # Directional Opposition DCAF fields
    directional_opposition_mode: bool = False
    adversarial_deltas: Optional[Dict[str, float]] = None
    do_candidates: Optional[List[DirectionalOppositionCandidate]] = None


@dataclass
class DCAFRunRecord:
    """
    Record of a single DCAF run for cross-run analysis.

    Tracks which weights were identified as safety-critical, allowing
    pattern analysis across multiple runs with different:
    - Random seeds
    - Training data subsets
    - Hyperparameters
    - Models
    """
    run_id: str  # Unique identifier for this run
    timestamp: str  # ISO format timestamp
    model_name: str
    principle: str
    config: Dict[str, Any]  # DCAFConfig as dict

    # Results
    candidate_params: List[str]  # All candidate parameter names
    validated_params: List[str]  # Ablation-validated parameter names
    delta_magnitudes: Dict[str, float]  # param -> magnitude

    # Metadata
    total_parameters: int
    training_steps: int
    delta_threshold: float
    use_simpo: bool
    comparative_mode: bool


@dataclass
class CrossRunAnalysis:
    """Analysis of patterns across multiple DCAF runs."""
    runs_analyzed: int
    consistent_params: Set[str]  # Params appearing in ALL runs
    frequent_params: Dict[str, int]  # param -> count of runs
    consistency_scores: Dict[str, float]  # param -> fraction of runs
    stable_core: Set[str]  # Params in >80% of runs
    variable_params: Set[str]  # Params in <50% of runs


class DCAFTrainer:
    """
    Full implementation of Differential Circuit Analysis Framework.

    Algorithm:
    1. Checkpoint base model weights -> W0
    2. For each safety principle (helpful, harmless, honest):
       a. Train model on principle-specific data -> W1
       b. Compute DeltaW = W1 - W0
       c. Reset model to W0
    3. Identify parameters with significant DeltaW across principles
    4. Validate candidates via ablation testing
    5. Cross-reference with SAE features for additional confirmation

    Example usage:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        >>> trainer = DCAFTrainer(model, tokenizer)
        >>> result = trainer.run_full_analysis(dataloaders, ablation_prompts)
        >>> print(f"Found {len(result.safety_parameters)} safety parameters")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[DCAFConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Differential Weight Trainer.

        Args:
            model: Pre-trained model to analyze
            tokenizer: Tokenizer for the model
            config: DCAF configuration (uses defaults if None)
            device: Device to use (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DCAFConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Storage for checkpoints and deltas
        self.base_weights: Dict[str, torch.Tensor] = {}
        self.trained_weights: Dict[str, Dict[str, torch.Tensor]] = {}  # principle -> param -> weight
        self.principle_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
        self.candidates: List[SafetyCircuitCandidate] = []
        self.training_losses: Dict[str, List[float]] = {}
        self._last_checkpoint_history = None  # Most recent CheckpointHistory from peak tracking

        # Detect model type for parameter naming
        self._is_pythia = "pythia" in model.config.model_type.lower() if hasattr(model.config, 'model_type') else False
        self._is_gpt_neox = "neox" in model.config.model_type.lower() if hasattr(model.config, 'model_type') else False

        logger.info(f"DCAF initialized for {model.config.model_type} on {self.device}")

    def checkpoint_base(self) -> Dict[str, torch.Tensor]:
        """
        Store W0 (base model weights) before any training.
        Returns a deep copy of all model parameters.
        """
        logger.info("Checkpointing base model weights (W0)...")
        self.base_weights = {}

        for name, param in self.model.named_parameters():
            # Clone to CPU to save GPU memory
            self.base_weights[name] = param.detach().cpu().clone()

        logger.info(f"Checkpointed {len(self.base_weights)} parameters")
        return self.base_weights

    def train_principle(
        self,
        principle: str,
        dataloader: DataLoader,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None,
        negate_loss: bool = False,
        use_peak_tracking: Optional[bool] = None,
        replay_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Train model on a specific safety principle.

        Args:
            principle: Name of the principle ("helpful", "harmless", "honest")
            dataloader: DataLoader with principle-specific training data
            epochs: Number of training epochs (default from config)
            max_steps: Override epochs if > 0 (default from config)
            callback: Optional callback(step, loss) for progress tracking
            negate_loss: If True, negate loss for gradient ascent (Anti-SFT mode)
            use_peak_tracking: Whether to use peak checkpoint tracking.
                If None, uses config.use_peak_checkpoint.
            replay_dataloader: Optional DataLoader with general-purpose data for
                experience replay during gradient ascent (Anti/Negated signals).
                When provided and negate_loss=True, each step mixes the negated
                behavioral loss with a standard cross-entropy loss on replay data,
                weighted by config.replay_fraction. Prevents catastrophic
                unlearning per Remark [Catastrophic Unlearning Mitigation].

        Returns:
            Dictionary mapping parameter names to their post-training values.
            If peak tracking is enabled, these are the peak checkpoint weights
            (model is also restored to peak state).
        """
        from dcaf.training.peak_tracking import (
            PeakTrackingConfig,
            PeakTrackingState,
            finalize_peak_tracking,
            update_peak_tracking,
        )

        epochs = epochs if epochs is not None else self.config.num_train_epochs
        max_steps = max_steps if max_steps is not None else self.config.max_steps
        mode_str = " (gradient ascent)" if negate_loss else ""
        do_peak = use_peak_tracking if use_peak_tracking is not None else self.config.use_peak_checkpoint

        if max_steps > 0:
            logger.info(f"Training on principle: {principle} for {max_steps} steps{mode_str}")
        else:
            logger.info(f"Training on principle: {principle} for {epochs} epoch(s){mode_str}")
        if do_peak:
            logger.info("  Peak checkpoint tracking enabled")

        # Peak tracking setup
        peak_state = PeakTrackingState() if do_peak else None
        peak_config = PeakTrackingConfig(
            peak_eval_interval=self.config.peak_eval_interval,
            peak_confirmation_window=self.config.peak_confirmation_window,
            peak_stability_tolerance=self.config.peak_stability_tolerance,
        ) if do_peak else None

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # Learning rate warmup
        def lr_lambda(current_step: int) -> float:
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.model.train()
        losses = []
        step = 0

        # Experience replay setup for gradient ascent (Anti/Negated signals)
        replay_iter = None
        if negate_loss and replay_dataloader is not None:
            replay_iter = iter(replay_dataloader)
            logger.info(f"  Experience replay enabled (fraction={self.config.replay_fraction})")

        # Compute total steps for progress bar
        if max_steps > 0:
            total_steps = max_steps
        else:
            total_steps = len(dataloader) * epochs
        pbar = tqdm(total=total_steps, desc=f"Training {principle}")

        try:
            # Epoch-based iteration (with optional max_steps override)
            for epoch in range(epochs):
                for batch in dataloader:
                    if max_steps > 0 and step >= max_steps:
                        break

                    # Move batch to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch.get("labels", input_ids).to(self.device)

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.config.sft_gradient_accumulation_steps

                    # Anti-SFT: Negate loss for gradient ascent
                    if negate_loss:
                        loss = -loss

                    # Experience replay: mix in standard CE on general data
                    if replay_iter is not None:
                        try:
                            replay_batch = next(replay_iter)
                        except StopIteration:
                            replay_iter = iter(replay_dataloader)
                            replay_batch = next(replay_iter)
                        replay_ids = replay_batch["input_ids"].to(self.device)
                        replay_mask = replay_batch["attention_mask"].to(self.device)
                        replay_labels = replay_batch.get("labels", replay_ids).to(self.device)
                        replay_outputs = self.model(
                            input_ids=replay_ids,
                            attention_mask=replay_mask,
                            labels=replay_labels,
                        )
                        replay_loss = replay_outputs.loss / self.config.sft_gradient_accumulation_steps
                        loss = loss + self.config.replay_fraction * replay_loss

                    # Backward pass
                    loss.backward()

                    # Gradient accumulation
                    if (step + 1) % self.config.sft_gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    losses.append(loss.item() * self.config.sft_gradient_accumulation_steps)

                    if callback:
                        callback(step, losses[-1])

                    # Peak tracking: evaluate at configured interval
                    if peak_state is not None and step > 0 and step % peak_config.peak_eval_interval == 0:
                        metric = -losses[-1]
                        weights = {n: p.detach().cpu().clone()
                                   for n, p in self.model.named_parameters()}
                        update_peak_tracking(peak_state, step, metric, weights, peak_config)

                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{losses[-1]:.4f}", "epoch": epoch + 1})
                    step += 1

                if max_steps > 0 and step >= max_steps:
                    break

            # Flush any remaining accumulated gradients
            if step % self.config.sft_gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
        finally:
            pbar.close()
            del optimizer
            del scheduler
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        self.training_losses[principle] = losses
        logger.info(f"Completed training {principle}, final loss: {losses[-1]:.4f}")

        # Final peak tracking evaluation at last step
        if peak_state is not None:
            metric = -losses[-1]
            weights = {n: p.detach().cpu().clone()
                       for n, p in self.model.named_parameters()}
            update_peak_tracking(peak_state, step, metric, weights, peak_config)

            history = finalize_peak_tracking(peak_state)
            logger.info(
                f"  Peak checkpoint: step {history.peak_step}, "
                f"metric {history.peak_metric:.6f}, "
                f"confirmed={history.is_confirmed}"
            )

            # Restore peak weights to model
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in history.peak_weights:
                        param.copy_(history.peak_weights[name].to(self.device))

            self._last_checkpoint_history = history
            return history.peak_weights

        # No peak tracking: return final weights as before
        post_weights = {}
        for name, param in self.model.named_parameters():
            post_weights[name] = param.detach().cpu().clone()

        return post_weights

    def train_principle_simpo(
        self,
        principle: str,
        hf_dataset,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        use_peak_tracking: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Train using SimPO (Simple Preference Optimization).

        SimPO provides cleaner signal through explicit preference contrast
        between chosen and rejected responses. It uses length-normalized
        log likelihood as implicit reward and doesn't need a reference model.

        Note: SimPO is implemented in CPOTrainer, not DPOTrainer!

        Args:
            principle: Name of the principle being trained
            hf_dataset: HuggingFace dataset with 'prompt', 'chosen', 'rejected' columns
            epochs: Number of training epochs (default from config)
            max_steps: Override epochs if > 0 (default from config, -1 means use epochs)
            use_peak_tracking: Whether to use peak checkpoint tracking.
                If None, uses config.use_peak_checkpoint.

        Returns:
            Dictionary mapping parameter names to their post-training values.
            If peak tracking is enabled, these are the peak checkpoint weights.
        """
        from trl import CPOConfig, CPOTrainer

        epochs = epochs if epochs is not None else self.config.num_train_epochs
        max_steps = max_steps if max_steps is not None else self.config.max_steps
        do_peak = use_peak_tracking if use_peak_tracking is not None else self.config.use_peak_checkpoint

        if max_steps > 0:
            logger.info(f"Training {principle} with SimPO for {max_steps} steps")
        else:
            logger.info(f"Training {principle} with SimPO for {epochs} epoch(s)")
        logger.info(f"  batch_size={self.config.batch_size}, grad_accum={self.config.gradient_accumulation_steps}")
        if do_peak:
            logger.info("  Peak checkpoint tracking enabled")

        import sys
        import tempfile
        _tmp_output = tempfile.mkdtemp(prefix=f"dcaf_{principle}_")
        config = CPOConfig(
            output_dir=_tmp_output,
            loss_type="simpo",
            cpo_alpha=0.0,
            simpo_gamma=1.0,
            beta=self.config.simpo_beta,
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.simpo_learning_rate,
            logging_steps=10,
            report_to="none",
            save_strategy="no",
            remove_unused_columns=False,
            bf16=_supports_bf16(self.device),
            fp16=False,
            gradient_checkpointing=True,
            dataset_num_proc=None,  # Avoid multiprocessing (Windows resource exhaustion)
        )

        # Set up peak tracking callback if enabled
        callbacks = []
        peak_callback = None
        if do_peak:
            from dcaf.training.peak_tracking import PeakTrackingCallback, PeakTrackingConfig
            peak_callback = PeakTrackingCallback(
                model=self.model,
                config=PeakTrackingConfig(
                    peak_eval_interval=self.config.peak_eval_interval,
                    peak_confirmation_window=self.config.peak_confirmation_window,
                    peak_stability_tolerance=self.config.peak_stability_tolerance,
                ),
            )
            callbacks.append(peak_callback)

        # On Windows, ensure UnSloth's compiled cache is importable by spawned workers
        if sys.platform == "win32":
            import os
            cache_dir = os.path.join(os.getcwd(), "unsloth_compiled_cache")
            if os.path.isdir(cache_dir):
                if cache_dir not in sys.path:
                    sys.path.insert(0, cache_dir)
                pypath = os.environ.get("PYTHONPATH", "")
                if cache_dir not in pypath:
                    os.environ["PYTHONPATH"] = cache_dir + os.pathsep + pypath

        trainer = CPOTrainer(
            model=self.model,
            args=config,
            train_dataset=hf_dataset,
            processing_class=self.tokenizer,
            callbacks=callbacks if callbacks else None,
        )
        try:
            trainer.train()
        finally:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            if hasattr(trainer, 'accelerator'):
                trainer.accelerator.free_memory()
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            import shutil
            shutil.rmtree(_tmp_output, ignore_errors=True)
            logger.info("  [Trainer memory freed]")

        # If peak tracking was used, restore peak weights to model
        if peak_callback is not None and peak_callback.evaluation_count > 0:
            history = peak_callback.get_checkpoint_history()
            logger.info(
                f"  Peak checkpoint: step {history.peak_step}, "
                f"metric {history.peak_metric:.6f}, "
                f"confirmed={history.is_confirmed}"
            )
            # Restore peak weights to model
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in history.peak_weights:
                        param.copy_(history.peak_weights[name].to(self.device))
            self._last_checkpoint_history = history
            return history.peak_weights

        # Capture post-training weights
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters()}

    def train_anti_sft(
        self,
        principle: str,
        dataloader: DataLoader,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        replay_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Anti-SFT training: Push model AWAY from target distribution.

        Uses gradient ascent (negated loss) to train the model in the
        opposite direction from the data distribution.

        Use cases:
            - Anti-SFT(unsafe): Train on harmful data -> pushed AWAY from harmful
            - Anti-SFT(safe): Train on safe data -> pushed AWAY from safe

        Args:
            principle: Name for this anti-training run (e.g., "anti_unsafe")
            dataloader: DataLoader with training data
            epochs: Number of training epochs (default from config)
            max_steps: Override epochs if > 0 (default from config)
            replay_dataloader: Optional DataLoader with general-purpose data for
                experience replay to prevent catastrophic unlearning.

        Returns:
            Dictionary mapping parameter names to post-training values
        """
        logger.info(f"Anti-SFT training: {principle}")
        return self.train_principle(
            principle=principle,
            dataloader=dataloader,
            epochs=epochs,
            max_steps=max_steps,
            negate_loss=True,
            replay_dataloader=replay_dataloader,
        )

    def train_anti_simpo(
        self,
        principle: str,
        hf_dataset,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        use_peak_tracking: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Anti-SimPO training: Push model AWAY from preference boundary.

        Uses gradient ascent (negated loss) on SimPO to create an opposite
        preference boundary on the base model. Unlike negated-SimPO, this
        does NOT require prior training - it runs on base model to create
        a new boundary in the opposite direction.

        Key difference from negated-SimPO:
            - Anti-SimPO: Creates opposite preference from scratch (parallel)
            - Negated-SimPO: Unlearns existing preference (sequential)

        Use cases:
            - Anti(unsafe): Push away from "prefer unsafe" -> pro-safety signal
            - Anti(safe): Push away from "prefer safe" -> anti-safety signal

        Args:
            principle: Name for this anti-training run (e.g., "_anti_unsafe")
            hf_dataset: HuggingFace dataset with prompt/chosen/rejected columns
            epochs: Number of training epochs (default from config)
            max_steps: Override epochs if > 0 (default from config)
            use_peak_tracking: Whether to use peak checkpoint tracking.
                If None, uses config.use_peak_checkpoint.

        Returns:
            Dictionary mapping parameter names to post-training values.
            If peak tracking is enabled, these are the peak checkpoint weights.
        """
        from dcaf.training.anti_trainer import cleanup_trainer, create_negated_simpo_trainer

        epochs = epochs if epochs is not None else self.config.num_train_epochs
        max_steps = max_steps if max_steps is not None else self.config.max_steps
        do_peak = use_peak_tracking if use_peak_tracking is not None else self.config.use_peak_checkpoint

        if max_steps > 0:
            logger.info(f"Anti-SimPO training: {principle} for {max_steps} steps")
        else:
            logger.info(f"Anti-SimPO training: {principle} for {epochs} epoch(s)")
        logger.info("  (Creating opposite preference boundary via gradient ascent)")
        logger.info(f"  batch_size={self.config.batch_size}, grad_accum={self.config.gradient_accumulation_steps}")
        if do_peak:
            logger.info("  Peak checkpoint tracking enabled")

        # Set up peak tracking callback if enabled
        callbacks = []
        peak_callback = None
        if do_peak:
            from dcaf.training.peak_tracking import PeakTrackingCallback, PeakTrackingConfig
            peak_callback = PeakTrackingCallback(
                model=self.model,
                config=PeakTrackingConfig(
                    peak_eval_interval=self.config.peak_eval_interval,
                    peak_confirmation_window=self.config.peak_confirmation_window,
                    peak_stability_tolerance=self.config.peak_stability_tolerance,
                ),
            )
            callbacks.append(peak_callback)

        trainer = create_negated_simpo_trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=hf_dataset,
            output_dir=f"./dcaf_anti_{principle}",
            num_train_epochs=epochs,
            max_steps=max_steps,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            beta=self.config.simpo_beta,
            learning_rate=self.config.simpo_learning_rate,
            callbacks=callbacks if callbacks else None,
        )

        try:
            trainer.train()
        finally:
            cleanup_trainer(trainer)

        # Extract peak checkpoint before cleanup if available
        if peak_callback is not None and peak_callback.evaluation_count > 0:
            history = peak_callback.get_checkpoint_history()
            logger.info(
                f"  Peak checkpoint: step {history.peak_step}, "
                f"metric {history.peak_metric:.6f}, "
                f"confirmed={history.is_confirmed}"
            )
            # Restore peak weights to model
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in history.peak_weights:
                        param.copy_(history.peak_weights[name].to(self.device))
            self._last_checkpoint_history = history
            return history.peak_weights

        # Capture weights
        return {n: p.detach().cpu().clone()
                for n, p in self.model.named_parameters()}

    def train_sft(
        self,
        key: str,
        dataloader: DataLoader,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard SFT training (NOT negated).

        Wraps train_principle() with negate_loss=False.
        Used for SFT training runs in F-variants (--sft flag).

        SFT trains the model to GENERATE like the training examples (mimicry).
        This is complementary to SimPO which trains to PREFER certain responses.

        Args:
            key: Name/key for this training run (e.g., "_sft_safe")
            dataloader: DataLoader with training data
            epochs: Number of training epochs (default from config)
            max_steps: Override epochs if > 0 (default from config)

        Returns:
            Dictionary mapping parameter names to post-training values
        """
        logger.info(f"SFT training: {key}")
        return self.train_principle(
            principle=key,
            dataloader=dataloader,
            epochs=epochs,
            max_steps=max_steps,
            negate_loss=False
        )

    def train_negated_simpo(
        self,
        principle: str,
        hf_dataset,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        use_peak_tracking: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Negated SimPO: Unlearn a trained preference.

        CRITICAL: Must run on a checkpoint that ALREADY has the preference.
        This is a sequential operation - cannot run in parallel with SimPO.

        Uses the same dataset as the original SimPO training but with
        negated loss to "forget" the learned preference.

        Use cases:
            - After SimPO(safe): Run negated to unlearn safety preference
            - After SimPO(unsafe): Run negated to unlearn unsafe preference

        Args:
            principle: Name for this negated training run
            hf_dataset: HuggingFace dataset with prompt/chosen/rejected columns
            epochs: Number of training epochs (default from config)
            max_steps: Override epochs if > 0 (default from config)
            use_peak_tracking: Whether to use peak checkpoint tracking.
                If None, uses config.use_peak_checkpoint.

        Returns:
            Dictionary mapping parameter names to post-training values.
            If peak tracking is enabled, these are the peak checkpoint weights.
        """
        from dcaf.training.anti_trainer import cleanup_trainer, create_negated_simpo_trainer

        epochs = epochs if epochs is not None else self.config.num_train_epochs
        max_steps = max_steps if max_steps is not None else self.config.max_steps
        do_peak = use_peak_tracking if use_peak_tracking is not None else self.config.use_peak_checkpoint

        if max_steps > 0:
            logger.info(f"Negated SimPO training: {principle} for {max_steps} steps")
        else:
            logger.info(f"Negated SimPO training: {principle} for {epochs} epoch(s)")
        logger.info("  (Unlearning trained preference via gradient ascent)")
        logger.info(f"  batch_size={self.config.batch_size}, grad_accum={self.config.gradient_accumulation_steps}")
        if do_peak:
            logger.info("  Peak checkpoint tracking enabled")

        # Set up peak tracking callback if enabled
        callbacks = []
        peak_callback = None
        if do_peak:
            from dcaf.training.peak_tracking import PeakTrackingCallback, PeakTrackingConfig
            peak_callback = PeakTrackingCallback(
                model=self.model,
                config=PeakTrackingConfig(
                    peak_eval_interval=self.config.peak_eval_interval,
                    peak_confirmation_window=self.config.peak_confirmation_window,
                    peak_stability_tolerance=self.config.peak_stability_tolerance,
                ),
            )
            callbacks.append(peak_callback)

        trainer = create_negated_simpo_trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=hf_dataset,
            output_dir=f"./dcaf_negated_{principle}",
            num_train_epochs=epochs,
            max_steps=max_steps,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            beta=self.config.simpo_beta,
            learning_rate=self.config.simpo_learning_rate,
            callbacks=callbacks if callbacks else None,
        )

        try:
            trainer.train()
        finally:
            cleanup_trainer(trainer)

        # Extract peak checkpoint after cleanup
        if peak_callback is not None and peak_callback.evaluation_count > 0:
            history = peak_callback.get_checkpoint_history()
            logger.info(
                f"  Peak checkpoint: step {history.peak_step}, "
                f"metric {history.peak_metric:.6f}, "
                f"confirmed={history.is_confirmed}"
            )
            # Restore peak weights to model
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in history.peak_weights:
                        param.copy_(history.peak_weights[name].to(self.device))
            self._last_checkpoint_history = history
            return history.peak_weights

        # Capture weights
        return {n: p.detach().cpu().clone()
                for n, p in self.model.named_parameters()}

    def compute_delta(
        self,
        principle: str,
        post_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DeltaW = W1 - W0 for a specific principle.

        Args:
            principle: Name of the principle
            post_weights: Weights after training on principle

        Returns:
            Dictionary mapping parameter names to their delta tensors
        """
        if not self.base_weights:
            raise RuntimeError("Must call checkpoint_base() first")

        logger.info(f"Computing weight deltas for principle: {principle}")
        deltas = {}

        for name, post_weight in post_weights.items():
            if name in self.base_weights:
                delta = post_weight - self.base_weights[name]
                deltas[name] = delta

        self.principle_deltas[principle] = deltas
        logger.info(f"Computed deltas for {len(deltas)} parameters")
        return deltas

    def reset_to_base(self):
        """Reset model to base weights W0 after training."""
        if not self.base_weights:
            raise RuntimeError("Must call checkpoint_base() first")

        logger.info("Resetting model to base weights (W0)")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.base_weights:
                    param.copy_(self.base_weights[name].to(self.device))

    def restore_trained_weights(self, principle: str, safety_params_only: bool = False) -> int:
        """
        Restore trained weights from a specific principle.

        After DCAF identifies which weights are
        safety-critical, we must restore the TRAINED version of those weights
        (not the base unaligned version) before freezing them.

        Args:
            principle: Which principle's trained weights to restore
            safety_params_only: If True, only restore safety-critical params.
                               If False, restore ALL trained weights.

        Returns:
            Number of parameters restored
        """
        if principle not in self.trained_weights:
            raise RuntimeError(f"No trained weights for principle: {principle}. "
                             f"Available: {list(self.trained_weights.keys())}")

        trained = self.trained_weights[principle]
        safety_params = {c.parameter_name for c in self.candidates} if safety_params_only else None

        restored_count = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in trained:
                    # If safety_params_only, only restore safety params
                    if safety_params_only and name not in safety_params:
                        continue
                    param.copy_(trained[name].to(self.device))
                    restored_count += 1

        logger.info(f"Restored {restored_count} trained weights from '{principle}'")
        return restored_count

    def store_current_weights(self, key: str) -> int:
        """
        Store the current model weights under a given key.

        This is useful when an external component (like TrainingOrchestrator)
        has trained the model and we need to save those weights for later
        restoration.

        Args:
            key: Key to store weights under (e.g., "_safe", "_unsafe")

        Returns:
            Number of parameters stored
        """
        weights = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }
        self.trained_weights[key] = weights
        logger.info(f"Stored {len(weights)} weights under key '{key}'")
        return len(weights)


    # =========================================================================
    # Language baseline (for comparative analysis)
    # =========================================================================

    def train_language_baseline(
        self,
        dataloader: DataLoader,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        use_peak_tracking: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Train on neutral language data to establish baseline weight changes.

        This captures changes from "learning conversation format" without
        safety-specific signals. Used to isolate true safety weights.

        Args:
            dataloader: DataLoader with neutral conversation data
            epochs: Number of training epochs (default from config)
            max_steps: Override epochs if > 0 (default from config)

        Returns:
            Post-training weights after language baseline training
        """
        epochs = epochs if epochs is not None else self.config.num_train_epochs
        max_steps = max_steps if max_steps is not None else self.config.max_steps
        if max_steps > 0:
            logger.info(f"Training language baseline for {max_steps} steps")
        else:
            logger.info(f"Training language baseline for {epochs} epoch(s)")
        logger.info("(This captures format learning, NOT safety learning)")

        do_peak = (
            use_peak_tracking
            if use_peak_tracking is not None
            else getattr(self.config, "use_peak_checkpoint_t11", False)
        )

        # T11 is the neutral anti-confound/control run. It is exempt from
        # peak tracking by default, but can opt in for exact artifact parity.
        return self.train_principle(
            "_language_baseline", dataloader,
            epochs=epochs, max_steps=max_steps,
            use_peak_tracking=do_peak,
        )

    def compute_language_delta(
        self,
        post_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ΔW_language = W_language - W0.

        Args:
            post_weights: Weights after language baseline training

        Returns:
            Delta tensors for language baseline
        """
        return self.compute_delta("_language_baseline", post_weights)

    def isolate_safety_weights(
        self,
        safety_threshold: Optional[float] = None,
        language_threshold_ratio: float = 0.5
    ) -> Set[str]:
        """
        Isolate weights that are safety-specific, not just format-learning.

        A parameter is safety-specific if:
        1. It has significant delta in safety training (above safety_threshold)
        2. Its delta in language training is below language_threshold_ratio
           of its safety delta

        This removes weights that change just from learning conversation format.

        Args:
            safety_threshold: Min delta for safety significance (default: config)
            language_threshold_ratio: Max ratio of language/safety delta (0.5 = 50%)

        Returns:
            Set of parameter names that are safety-SPECIFIC
        """
        safety_threshold = safety_threshold or self.config.delta_threshold

        if "_language_baseline" not in self.principle_deltas:
            logger.warning("No language baseline computed. Run train_language_baseline first.")
            logger.warning("Falling back to standard safety identification.")
            return self.get_safety_parameter_names()

        language_deltas = self.principle_deltas["_language_baseline"]
        isolated_params = set()

        logger.info("Isolating safety-specific weights...")
        logger.info(f"  Safety threshold: {safety_threshold}")
        logger.info(f"  Language threshold ratio: {language_threshold_ratio}")

        # Get all candidate parameters (from safety principles, not language)
        safety_principles = [p for p in self.principle_deltas.keys() if p != "_language_baseline"]

        for param_name in self.base_weights.keys():
            # Compute max safety delta across principles
            max_safety_delta = 0.0
            for principle in safety_principles:
                if param_name in self.principle_deltas.get(principle, {}):
                    delta = torch.norm(self.principle_deltas[principle][param_name]).item()
                    max_safety_delta = max(max_safety_delta, delta)

            # Compute language delta
            language_delta = 0.0
            if param_name in language_deltas:
                language_delta = torch.norm(language_deltas[param_name]).item()

            # Check isolation criteria
            if max_safety_delta >= safety_threshold:
                # Exclude general-purpose weights (embed_out, layernorms, etc.)
                if self.config.exclude_general_weights and should_exclude_param(param_name):
                    logger.debug(f"Excluded general weight from isolation: {param_name}")
                    continue

                # This param changed significantly during safety training
                # Now check if it's unique to safety (not just format learning)
                if language_delta < max_safety_delta * language_threshold_ratio:
                    # Language delta is much smaller - this is safety-specific
                    isolated_params.add(param_name)
                else:
                    # This param also changed a lot during language training
                    # It's probably format learning, not safety-specific
                    logger.debug(
                        f"Excluded {param_name}: safety={max_safety_delta:.4f}, "
                        f"lang={language_delta:.4f} (ratio={language_delta/max_safety_delta:.2f})"
                    )

        logger.info(f"Isolated {len(isolated_params)} safety-specific parameters")
        logger.info(f"(Excluded {len(self.base_weights) - len(isolated_params)} format-learning params)")

        return isolated_params
