"""
DCAF Variant Orchestrator (sec:foundations; def:canonical-signal-instantiation).

Orchestrates training runs using composable flags that correspond to the
11 canonical signals:

- prefopt (PO): Preference optimization (t1/t6). Active unless no_simpo is set.
- sft (S):      Supervised fine-tuning (t2/t7).
- cumulative (C): SFT then PrefOpt sequentially (t3/t8). Requires S and PO.
- anti (A):     Preference-margin ascent from base (t4/t9).
- negated (N):  Learn-then-unlearn signal (t5/t10). Requires both PO checkpoints.

Direction control:
- target=True:   Include T+ (target-side) runs.
- opposite=True: Include T- (opposite-side) runs.
- Default is both sides.

The t11 (DomainNative neutral baseline) always runs regardless of flags.

Run/delta/checkpoint names follow the spec-aligned naming convention:
  run_type       → "t1_prefopt_target", "t6_prefopt_opposite", ..., "t11_baseline"
  delta_name     → "delta_t1_prefopt_target", ..., "delta_t11_baseline"
  checkpoint key → "checkpoint_t1", ..., "checkpoint_t11", "base"
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from dcaf.core.config import DCAFConfig
from dcaf.storage.checkpoint import CheckpointManager

if TYPE_CHECKING:
    from dcaf.storage.delta_store import DeltaStore

logger = logging.getLogger(__name__)


@dataclass
class TrainingRun:
    """Specification for a single training run."""

    run_type: str           # Spec-aligned signal name, e.g. "t1_prefopt_target"
    save_as: Optional[str] = None        # Checkpoint key to save after run
    restore_from: Optional[str] = None   # Checkpoint key to restore before run
    delta_name: Optional[str] = None     # Delta name for delta computation (vs base)


@dataclass
class DCAFVariantConfig:
    """Configuration for a DCAF variant built from composable flags."""

    name: str            # e.g. "composable variant" or "composable variantAN"
    runs: List[TrainingRun]


# ============================================================================
# Composable variant system
# ============================================================================

# Target-side (T+) runs, keyed by flag.
# Target negated unlearns the opposite learned behavior, so it restores t6.
TARGET_RUNS: Dict[str, TrainingRun] = {
    "PO": TrainingRun(
        "t1_prefopt_target",
        restore_from="base",
        save_as="checkpoint_t1",
        delta_name="delta_t1_prefopt_target",
    ),
    "S": TrainingRun(
        "t2_sft_target",
        restore_from="base",
        save_as="checkpoint_t2",
        delta_name="delta_t2_sft_target",
    ),
    "C": TrainingRun(
        "t3_cumulative_target",
        restore_from="checkpoint_t2",
        save_as="checkpoint_t3",
        delta_name="delta_t3_cumulative_target",
    ),
    "A": TrainingRun(
        "t4_anti_target",
        restore_from="base",
        save_as="checkpoint_t4",
        delta_name="delta_t4_anti_target",
    ),
    "N": TrainingRun(
        "t5_negated_target",
        restore_from="checkpoint_t6",
        save_as="checkpoint_t5",
        delta_name="delta_t5_negated_target",
    ),
}

# Opposite-side (T-) runs, keyed by flag.
# Opposite negated unlearns the target learned behavior, so it restores t1.
OPPOSITE_RUNS: Dict[str, TrainingRun] = {
    "PO": TrainingRun(
        "t6_prefopt_opposite",
        restore_from="base",
        save_as="checkpoint_t6",
        delta_name="delta_t6_prefopt_opposite",
    ),
    "S": TrainingRun(
        "t7_sft_opposite",
        restore_from="base",
        save_as="checkpoint_t7",
        delta_name="delta_t7_sft_opposite",
    ),
    "C": TrainingRun(
        "t8_cumulative_opposite",
        restore_from="checkpoint_t7",
        save_as="checkpoint_t8",
        delta_name="delta_t8_cumulative_opposite",
    ),
    "A": TrainingRun(
        "t9_anti_opposite",
        restore_from="base",
        save_as="checkpoint_t9",
        delta_name="delta_t9_anti_opposite",
    ),
    "N": TrainingRun(
        "t10_negated_opposite",
        restore_from="checkpoint_t1",
        save_as="checkpoint_t10",
        delta_name="delta_t10_negated_opposite",
    ),
}

# t11 baseline always runs regardless of direction flags.
BASELINE_RUNS: List[TrainingRun] = [
    TrainingRun(
        "t11_baseline",
        restore_from="base",
        save_as="checkpoint_t11",
        delta_name="delta_t11_baseline",
    ),
]

# Execution order respects restore_from dependency chains.
RUN_ORDER = ["PO", "S", "C", "A", "N"]

# Modifier dependencies: C requires SFT; N requires both PO source checkpoints.
MODIFIER_DEPS: Dict[str, str] = {"C": "S", "N": "PO"}

# PO is active by default and omitted only by no_simpo.
ALWAYS_ACTIVE: Set[str] = {"PO"}


def _make_name(active: Set[str], target: bool, opposite: bool) -> str:
    """Generate variant name from active flags and direction."""
    flags = "".join(f for f in RUN_ORDER if f in active) or "T11"
    name = f"DCAF-{flags}"
    if target and not opposite:
        name += "-T+"
    elif opposite and not target:
        name += "-T-"
    return name


def build_variant(
    modifiers: str = "",
    target: bool = True,
    opposite: bool = True,
    no_simpo: bool = False,
) -> DCAFVariantConfig:
    """Build a variant configuration from composable flags.

    Args:
        modifiers: String of modifier flags (e.g. "A", "SAN", "SCAN").
            S = SFT, C = Cumulative, A = Anti, N = Negated.
            PO (preference optimization) is included automatically unless
            no_simpo is set.
        target: Include T+ (target-side) runs.  Default True.
        opposite: Include T- (opposite-side) runs.  Default True.
        no_simpo: Omit preference-backed signals. This does not replace
            PrefOpt with SFT; pass "S" explicitly to run SFT signals.

    Returns:
        DCAFVariantConfig with the assembled training runs.

    Examples:
        build_variant()                              -> PO both sides + t11
        build_variant("A")                           -> PO + Anti, both sides + t11
        build_variant("A", target=True, opposite=False) -> PO + Anti, T+ only + t11
        build_variant("SCAN")                        -> All flags, both sides + t11
        build_variant(no_simpo=True)                 -> t11 only
        build_variant("S", no_simpo=True)            -> SFT both sides + t11
    """
    if not target and not opposite:
        raise ValueError("At least one of target or opposite must be True")

    requested = set(modifiers.upper())
    unknown = requested - set(RUN_ORDER)
    if unknown:
        raise ValueError(f"Unknown training modifier(s): {', '.join(sorted(unknown))}")

    preference_backed = {"C", "A", "N"}
    if no_simpo and requested & preference_backed:
        invalid = ", ".join(sorted(requested & preference_backed))
        raise ValueError(
            f"--no-simpo omits preference-backed signals; cannot run modifier(s): {invalid}. "
            "Use --sft for SFT-only signals, or enable SimPO for cumulative, anti, or negated runs."
        )

    active = set(requested)
    if not no_simpo:
        active |= ALWAYS_ACTIVE
        for mod, dep in MODIFIER_DEPS.items():
            if mod in active:
                active.add(dep)

    if "N" in active and not (target and opposite):
        raise ValueError(
            "Negated signals require both target and opposite PrefOpt source checkpoints. "
            "Do not combine --negated with --target-only or --opposite-only."
        )

    runs = list(BASELINE_RUNS)  # t11 always runs

    # Add runs in execution order (respects restore_from dependency chains)
    for flag in RUN_ORDER:
        if flag not in active:
            continue
        if target and flag in TARGET_RUNS:
            runs.append(TARGET_RUNS[flag])
        if opposite and flag in OPPOSITE_RUNS:
            runs.append(OPPOSITE_RUNS[flag])

    return DCAFVariantConfig(name=_make_name(active, target, opposite), runs=runs)


class TrainingOrchestrator:
    """Orchestrates training runs for DCAF variants (def:canonical-signal-instantiation).

    Handles the execution of training phases, checkpoint management,
    delta computation, and saving results to a DeltaStore.

    Activation capture and training metrics capture are opt-in via
    enable_activation_capture() and enable_metrics_capture().
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DCAFConfig,
        device: str = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = CheckpointManager(model, device)

        # Initialize DCAF trainer for circuit analysis
        from dcaf.training.trainer import DCAFTrainer
        self.trainer = DCAFTrainer(self.model, self.tokenizer, self.config)

        # Training state
        self._current_variant: Optional[str] = None
        self._training_losses: Dict[str, List] = {}

        # Activation capture support
        self._activation_capture_enabled: bool = False
        self._activation_capturer: Optional[Any] = None
        self._probe_set: Optional[Any] = None
        self._activation_config: Dict[str, Any] = {}
        self._delta_store: Optional["DeltaStore"] = None

        # Metrics capture support (for effectiveness scoring)
        self._metrics_capture_enabled: bool = False
        self._metrics_capture: Optional[Any] = None
        self._eval_dataloader: Optional[Any] = None
        self._eval_pref_dataset: Optional[Any] = None

        # Peak checkpoint histories (keyed by run_type)
        self._checkpoint_histories: Dict[str, Any] = {}

    def run_variant(
        self,
        variant_config: DCAFVariantConfig,
        safe_simpo_dataset,
        unsafe_simpo_dataset=None,
        safe_sft_dataloader=None,
        unsafe_sft_dataloader=None,
        language_dataloader=None,
        epochs_per_phase: int = 1,
        max_steps_per_phase: int = -1,
        delta_store: Optional["DeltaStore"] = None,
        phase_callback: Optional[Callable[[str, int], Dict[str, Any]]] = None,
    ) -> None:
        """Run a complete DCAF variant pipeline.

        Executes all training runs in the variant, saves deltas to disk.
        Analysis is done separately via the CLI.

        Args:
            variant_config: Variant configuration from build_variant().
            safe_simpo_dataset: SimPO dataset with target preferences.
            unsafe_simpo_dataset: SimPO dataset with opposite preferences.
            safe_sft_dataloader: SFT dataloader with target responses.
            unsafe_sft_dataloader: SFT dataloader with opposite responses.
            language_dataloader: Neutral language dataloader (t11).
            epochs_per_phase: Training epochs per phase.
            max_steps_per_phase: Override epochs if > 0.
            delta_store: DeltaStore instance for saving deltas (required).
            phase_callback: Optional callback(phase_name, phase_idx) -> metrics_dict.
        """
        if delta_store is None:
            raise ValueError("delta_store is required")

        config = variant_config
        self._current_variant = config.name
        if self._activation_capture_enabled:
            self._delta_store = delta_store
            if self._probe_set is not None:
                delta_store.save_probe_set(self._probe_set)
        if self._metrics_capture_enabled and self._metrics_capture:
            self._metrics_capture.clear()

        logger.info("=" * 70)
        logger.info(f"Running {config.name}")
        logger.info(f"  Runs: {len(config.runs)}")
        for run in config.runs:
            logger.info(f"    - {run.run_type} (delta={run.delta_name})")
        logger.info("=" * 70)

        # Save base checkpoint
        self.ckpt.save_checkpoint("base", training_phase="initial")
        base_weights = self.ckpt.get_checkpoint("base").weights
        delta_store.save_checkpoint("base", base_weights)

        # Track phase metrics (e.g. refusal rates)
        phase_metrics: Dict[str, Any] = {}

        # Measure baseline if callback provided
        if phase_callback:
            baseline = phase_callback("baseline", -1)
            if baseline:
                phase_metrics["baseline"] = baseline

        # Execute all training runs
        for i, run in enumerate(config.runs):
            logger.info(f"\nPhase {i+1}/{len(config.runs)}: {run.run_type}")
            self._execute_run(
                run,
                safe_simpo_dataset=safe_simpo_dataset,
                unsafe_simpo_dataset=unsafe_simpo_dataset,
                safe_sft_dataloader=safe_sft_dataloader,
                unsafe_sft_dataloader=unsafe_sft_dataloader,
                language_dataloader=language_dataloader,
                epochs=epochs_per_phase,
                max_steps=max_steps_per_phase,
            )

            # Call phase callback if provided
            if phase_callback:
                metrics = phase_callback(run.run_type, i)
                if metrics:
                    phase_metrics[run.run_type] = metrics

            # Save delta to disk
            if run.delta_name:
                delta = self.ckpt.get_delta(run.delta_name)
                delta_store.save_delta(run.delta_name, delta)
                logger.info(f"  Saved delta: {run.delta_name}")

            # Persist every phase checkpoint advertised by the run metadata.
            if run.save_as:
                weights = self.ckpt.get_checkpoint(run.save_as).weights
                delta_store.save_checkpoint(run.save_as, weights)
                logger.info(f"  Saved checkpoint: {run.save_as}")

        # Save phase metrics to metadata
        if phase_metrics:
            delta_store.update_metadata({"phase_metrics": phase_metrics})

        # Save training metrics for effectiveness scoring if captured
        if self._metrics_capture_enabled and self._metrics_capture:
            training_metrics = self._metrics_capture.get_all_metrics()
            if training_metrics:
                delta_store.update_metadata({"training_metrics": training_metrics})
                logger.info(f"  Saved training metrics for {len(training_metrics)} signals")

        logger.info("\nTraining complete. Use 'dcaf analyze' to analyze saved deltas.")
        self.ckpt.restore_checkpoint("base")

    def enable_activation_capture(
        self,
        probe_set: Any,
        delta_store: "DeltaStore",
        probe_type: str = "both",
        max_length: int = 128,
        batch_size: int = 8,
        capture_residual: bool = True,
        enable_free_generation: bool = False,
        free_gen_max_tokens: int = 10,
    ) -> None:
        """Enable activation capture for this training run.

        Args:
            probe_set: ProbeSet to use for activation capture.
            delta_store: DeltaStore for saving snapshots.
            probe_type: "recognition", "teacher_forcing", or "both".
            max_length: Maximum sequence length.
            batch_size: Batch size for capture.
            capture_residual: Capture residual-stream activations. Enabled by
                default for spec pipeline artifacts; disable to save memory.
            enable_free_generation: Enable free generation steering probe.
            free_gen_max_tokens: Tokens for free generation.
        """
        from dcaf.domains.activation import ActivationCapture

        self._activation_capture_enabled = True
        self._probe_set = probe_set
        self._delta_store = delta_store

        self._activation_capturer = ActivationCapture(
            model=self.model,
            capture_attention=True,
            capture_mlp=True,
            capture_residual=capture_residual,
        )

        self._activation_config = {
            "probe_type": probe_type,
            "max_length": max_length,
            "batch_size": batch_size,
            "capture_residual": capture_residual,
            "enable_free_generation": enable_free_generation,
            "free_gen_max_tokens": free_gen_max_tokens,
        }

        logger.info(f"Activation capture enabled ({len(probe_set)} prompts)")
        logger.info(f"  Probe types: {probe_type}")
        if enable_free_generation:
            logger.info(f"  Free generation: {free_gen_max_tokens} tokens (steering window)")

    def enable_metrics_capture(
        self,
        eval_dataloader: Optional[Any] = None,
        eval_pref_dataset: Optional[Any] = None,
        max_eval_batches: int = 10,
        max_pref_samples: int = 50,
    ) -> None:
        """Enable training metrics capture for effectiveness scoring.

        Captures pre/post training metrics (loss, margin) that feed into
        signal effectiveness computation for weight confidence (C_W).
        """
        from dcaf.domains.weight.metrics import MetricsCapture

        self._metrics_capture_enabled = True
        self._eval_dataloader = eval_dataloader
        self._eval_pref_dataset = eval_pref_dataset

        self._metrics_capture = MetricsCapture(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_eval_batches=max_eval_batches,
            max_pref_samples=max_pref_samples,
        )

        logger.info("Metrics capture enabled for effectiveness scoring")
        if eval_dataloader:
            logger.info(f"  Loss evaluation: {max_eval_batches} batches")
        if eval_pref_dataset:
            logger.info(f"  Margin evaluation: {max_pref_samples} samples")

    def _capture_and_save_activations(
        self, snapshot_name: str, run_type: str
    ) -> None:
        """Capture and save activation snapshot after a training run."""
        if not self._activation_capture_enabled:
            return

        logger.info(f"Capturing activations: {snapshot_name}...")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            snapshot = self._activation_capturer.capture(
                probe_set=self._probe_set,
                tokenizer=self.tokenizer,
                name=snapshot_name,
                probe_type=self._activation_config["probe_type"],
                max_length=self._activation_config["max_length"],
                batch_size=self._activation_config["batch_size"],
                enable_free_generation=self._activation_config["enable_free_generation"],
                max_new_tokens=self._activation_config["free_gen_max_tokens"],
                show_progress=False,
            )

            path = self._delta_store.save_activation_snapshot(snapshot_name, snapshot)
            logger.info(f"  Saved: {path} (~{path.stat().st_size / 1e6:.0f}MB)")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Activation capture failed for {snapshot_name}: {e}")
            logger.warning("Training continues despite activation capture failure")

    def _execute_run(
        self,
        run: TrainingRun,
        safe_simpo_dataset,
        unsafe_simpo_dataset,
        safe_sft_dataloader,
        unsafe_sft_dataloader,
        language_dataloader,
        epochs: int = 1,
        max_steps: int = -1,
    ) -> None:
        """Execute a single training run."""
        from dcaf.training.trainer import DCAFTrainer

        # Restore checkpoint if specified
        if run.restore_from:
            self.ckpt.restore_checkpoint(run.restore_from)

        # Capture pre-training metrics if enabled
        if self._metrics_capture_enabled and self._metrics_capture and run.delta_name:
            eval_dataloader, pref_dataset = self._metrics_inputs_for_run(
                run.run_type,
                safe_simpo_dataset,
                unsafe_simpo_dataset,
                safe_sft_dataloader,
                unsafe_sft_dataloader,
                language_dataloader,
            )

            self._metrics_capture.capture_pre_metrics(
                delta_name=run.delta_name,
                run_type=run.run_type,
                eval_dataloader=eval_dataloader,
                preference_dataset=pref_dataset or self._eval_pref_dataset,
            )

        # Create a temporary trainer instance for this run
        trainer = DCAFTrainer(self.model, self.tokenizer, self.config)
        use_simpo = self.config.use_simpo

        def require_dataset(name: str, dataset):
            if dataset is None:
                raise ValueError(f"{name} is required for {run.run_type}")
            return dataset

        # Execute based on run type (spec-aligned signal names)
        if run.run_type == "t1_prefopt_target":
            if not use_simpo:
                raise ValueError("t1_prefopt_target requires SimPO; --no-simpo must omit this run")
            trainer.train_principle_simpo(
                "_t1",
                require_dataset("safe_simpo_dataset", safe_simpo_dataset),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t6_prefopt_opposite":
            if not use_simpo:
                raise ValueError("t6_prefopt_opposite requires SimPO; --no-simpo must omit this run")
            trainer.train_principle_simpo(
                "_t6",
                require_dataset("unsafe_simpo_dataset", unsafe_simpo_dataset),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t2_sft_target":
            trainer.train_sft(
                "_t2",
                require_dataset("safe_sft_dataloader", safe_sft_dataloader),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t7_sft_opposite":
            trainer.train_sft(
                "_t7",
                require_dataset("unsafe_sft_dataloader", unsafe_sft_dataloader),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t3_cumulative_target":
            if not use_simpo:
                raise ValueError("t3_cumulative_target requires SimPO; --no-simpo must omit this run")
            trainer.train_principle_simpo(
                "_t3",
                require_dataset("safe_simpo_dataset", safe_simpo_dataset),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t8_cumulative_opposite":
            if not use_simpo:
                raise ValueError("t8_cumulative_opposite requires SimPO; --no-simpo must omit this run")
            trainer.train_principle_simpo(
                "_t8",
                require_dataset("unsafe_simpo_dataset", unsafe_simpo_dataset),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t4_anti_target":
            if not use_simpo:
                raise ValueError("t4_anti_target requires SimPO; --no-simpo must omit this run")
            trainer.train_anti_simpo(
                "_t4",
                require_dataset("unsafe_simpo_dataset", unsafe_simpo_dataset),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t9_anti_opposite":
            if not use_simpo:
                raise ValueError("t9_anti_opposite requires SimPO; --no-simpo must omit this run")
            trainer.train_anti_simpo(
                "_t9",
                require_dataset("safe_simpo_dataset", safe_simpo_dataset),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t5_negated_target":
            if not use_simpo:
                raise ValueError("t5_negated_target requires SimPO; --no-simpo must omit this run")
            trainer.train_negated_simpo(
                "_t5",
                require_dataset("unsafe_simpo_dataset", unsafe_simpo_dataset),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t10_negated_opposite":
            if not use_simpo:
                raise ValueError("t10_negated_opposite requires SimPO; --no-simpo must omit this run")
            trainer.train_negated_simpo(
                "_t10",
                require_dataset("safe_simpo_dataset", safe_simpo_dataset),
                epochs=epochs,
                max_steps=max_steps,
            )
        elif run.run_type == "t11_baseline":
            trainer.train_language_baseline(
                require_dataset("language_dataloader", language_dataloader),
                epochs=epochs,
                max_steps=max_steps,
                use_peak_tracking=self.config.use_peak_checkpoint_t11,
            )
        else:
            raise ValueError(f"Unknown run type: {run.run_type!r}")

        # Capture post-training metrics if enabled
        if self._metrics_capture_enabled and self._metrics_capture and run.delta_name:
            eval_dataloader, pref_dataset = self._metrics_inputs_for_run(
                run.run_type,
                safe_simpo_dataset,
                unsafe_simpo_dataset,
                safe_sft_dataloader,
                unsafe_sft_dataloader,
                language_dataloader,
            )

            self._metrics_capture.capture_post_metrics(
                delta_name=run.delta_name,
                eval_dataloader=eval_dataloader,
                preference_dataset=pref_dataset or self._eval_pref_dataset,
            )

        # Save losses
        self._training_losses[run.run_type] = trainer.training_losses.get(
            list(trainer.training_losses.keys())[-1]
            if trainer.training_losses else "",
            [],
        )

        # Store peak checkpoint history if available
        if trainer._last_checkpoint_history is not None:
            self._checkpoint_histories[run.run_type] = trainer._last_checkpoint_history
            history = trainer._last_checkpoint_history
            logger.info(
                f"  Peak tracking for {run.run_type}: "
                f"step {history.peak_step}, metric {history.peak_metric:.6f}, "
                f"confirmed={history.is_confirmed}"
            )

        # Save checkpoint if specified
        if run.save_as:
            self.ckpt.save_checkpoint(run.save_as, training_phase=run.run_type)

        # Compute delta if specified
        if run.delta_name:
            self.ckpt.compute_delta("base", run.save_as, run.delta_name)

        # Capture activations if enabled
        if self._activation_capture_enabled and self._activation_capturer:
            snapshot_name = f"after_{run.run_type}"
            self._capture_and_save_activations(snapshot_name, run.run_type)

    def _metrics_inputs_for_run(
        self,
        run_type: str,
        safe_simpo_dataset: Any,
        unsafe_simpo_dataset: Any,
        safe_sft_dataloader: Any,
        unsafe_sft_dataloader: Any,
        language_dataloader: Any,
    ) -> tuple:
        """Choose signal-matched evaluation inputs for effectiveness metrics."""
        pref_dataset = None
        eval_dataloader = self._eval_dataloader

        if run_type in ("t1_prefopt_target", "t3_cumulative_target"):
            pref_dataset = safe_simpo_dataset
            eval_dataloader = safe_sft_dataloader or eval_dataloader
        elif run_type in ("t6_prefopt_opposite", "t8_cumulative_opposite"):
            pref_dataset = unsafe_simpo_dataset
            eval_dataloader = unsafe_sft_dataloader or eval_dataloader
        elif run_type == "t5_negated_target":
            pref_dataset = unsafe_simpo_dataset
            eval_dataloader = unsafe_sft_dataloader or eval_dataloader
        elif run_type == "t10_negated_opposite":
            pref_dataset = safe_simpo_dataset
            eval_dataloader = safe_sft_dataloader or eval_dataloader
        elif run_type == "t4_anti_target":
            pref_dataset = unsafe_simpo_dataset
            eval_dataloader = unsafe_sft_dataloader or eval_dataloader
        elif run_type == "t9_anti_opposite":
            pref_dataset = safe_simpo_dataset
            eval_dataloader = safe_sft_dataloader or eval_dataloader
        elif run_type == "t2_sft_target":
            eval_dataloader = safe_sft_dataloader or eval_dataloader
        elif run_type == "t7_sft_opposite":
            eval_dataloader = unsafe_sft_dataloader or eval_dataloader
        elif run_type == "t11_baseline":
            eval_dataloader = language_dataloader or eval_dataloader

        return eval_dataloader, pref_dataset


__all__ = [
    "TrainingRun",
    "DCAFVariantConfig",
    "TARGET_RUNS",
    "OPPOSITE_RUNS",
    "BASELINE_RUNS",
    "RUN_ORDER",
    "MODIFIER_DEPS",
    "ALWAYS_ACTIVE",
    "build_variant",
    "TrainingOrchestrator",
]
