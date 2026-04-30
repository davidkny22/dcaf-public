"""
DCAF Variant Orchestrator (§1, Def 1.7).

Orchestrates training runs using composable flags that correspond to the
11 canonical signals:

- prefopt (PO): Preference optimization (t1/t6). Always active — core signal.
- sft (S):      Supervised fine-tuning (t2/t7).
- cumulative (C): SFT then PrefOpt sequentially (t3/t8). Requires sft=True.
- anti (A):     Gradient ascent from base (t4/t9).
- negated (N):  Unlearn trained preference (t5/t10). Requires PrefOpt checkpoint.

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

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set, Any, Callable, TYPE_CHECKING
import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from dcaf.storage.checkpoint import CheckpointManager
from dcaf.core.config import DCAFConfig

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
# PO is always active — it is the core preference optimization step.
TARGET_RUNS: Dict[str, TrainingRun] = {
    "PO": TrainingRun(
        "t1_prefopt_target",
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
        restore_from="checkpoint_t1",
        save_as="checkpoint_t5",
        delta_name="delta_t5_negated_target",
    ),
}

# Opposite-side (T-) runs, keyed by flag.
OPPOSITE_RUNS: Dict[str, TrainingRun] = {
    "PO": TrainingRun(
        "t6_prefopt_opposite",
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
        restore_from="checkpoint_t6",
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

# Modifier dependencies: C requires the SFT checkpoint, N requires the PO checkpoint.
MODIFIER_DEPS: Dict[str, str] = {"C": "S", "N": "PO"}

# PO is always active — it is the core preference optimization signal.
ALWAYS_ACTIVE: Set[str] = {"PO"}


def _make_name(active: Set[str], target: bool, opposite: bool) -> str:
    """Generate variant name from active flags and direction."""
    flags = "".join(f for f in RUN_ORDER if f in active)
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
            PO (preference optimization) is always included automatically
            unless no_simpo is set.
        target: Include T+ (target-side) runs.  Default True.
        opposite: Include T- (opposite-side) runs.  Default True.
        no_simpo: Replace PO with SFT as the core signal. Uses SFT-only
            training (t2/t7) instead of preference optimization (t1/t6).

    Returns:
        DCAFVariantConfig with the assembled training runs.

    Examples:
        build_variant()                              -> PO both sides + t11
        build_variant("A")                           -> PO + Anti, both sides + t11
        build_variant("A", target=True, opposite=False) -> PO + Anti, T+ only + t11
        build_variant("SCAN")                        -> All flags, both sides + t11
        build_variant(no_simpo=True)                 -> SFT both sides + t11
    """
    if not target and not opposite:
        raise ValueError("At least one of target or opposite must be True")

    runs = list(BASELINE_RUNS)  # t11 always runs

    core = {"S"} if no_simpo else ALWAYS_ACTIVE
    active = core | set(modifiers.upper())
    deps = {"C": "S", "N": "S"} if no_simpo else MODIFIER_DEPS
    for mod, dep in deps.items():
        if mod in active:
            active.add(dep)

    target_runs = TARGET_RUNS
    opposite_runs = OPPOSITE_RUNS
    if no_simpo:
        target_runs = dict(TARGET_RUNS)
        opposite_runs = dict(OPPOSITE_RUNS)
        target_runs["N"] = replace(TARGET_RUNS["N"], restore_from="checkpoint_t2")
        opposite_runs["N"] = replace(OPPOSITE_RUNS["N"], restore_from="checkpoint_t7")

    # Add runs in execution order (respects restore_from dependency chains)
    for flag in RUN_ORDER:
        if flag not in active:
            continue
        if target and flag in target_runs:
            runs.append(target_runs[flag])
        if opposite and flag in opposite_runs:
            runs.append(opposite_runs[flag])

    return DCAFVariantConfig(name=_make_name(active, target, opposite), runs=runs)


class TrainingOrchestrator:
    """Orchestrates training runs for DCAF variants (§1, Def 1.7).

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
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
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
            capture_residual=False,
        )

        self._activation_config = {
            "probe_type": probe_type,
            "max_length": max_length,
            "batch_size": batch_size,
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
            pref_dataset = None
            if run.run_type in (
                "t1_prefopt_target", "t2_sft_target",
                "t3_cumulative_target", "t5_negated_target",
            ):
                pref_dataset = safe_simpo_dataset
            elif run.run_type in (
                "t6_prefopt_opposite", "t7_sft_opposite",
                "t8_cumulative_opposite", "t10_negated_opposite",
            ):
                pref_dataset = unsafe_simpo_dataset

            self._metrics_capture.capture_pre_metrics(
                delta_name=run.delta_name,
                run_type=run.run_type,
                eval_dataloader=self._eval_dataloader,
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
            if use_simpo:
                trainer.train_principle_simpo(
                    "_t1",
                    require_dataset("safe_simpo_dataset", safe_simpo_dataset),
                    epochs=epochs,
                    max_steps=max_steps,
                )
            else:
                trainer.train_sft(
                    "_t1",
                    require_dataset("safe_sft_dataloader", safe_sft_dataloader),
                    epochs=epochs,
                    max_steps=max_steps,
                )
        elif run.run_type == "t6_prefopt_opposite":
            if use_simpo:
                trainer.train_principle_simpo(
                    "_t6",
                    require_dataset("unsafe_simpo_dataset", unsafe_simpo_dataset),
                    epochs=epochs,
                    max_steps=max_steps,
                )
            else:
                trainer.train_sft(
                    "_t6",
                    require_dataset("unsafe_sft_dataloader", unsafe_sft_dataloader),
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
            if use_simpo:
                trainer.train_principle_simpo(
                    "_t3",
                    require_dataset("safe_simpo_dataset", safe_simpo_dataset),
                    epochs=epochs,
                    max_steps=max_steps,
                )
            else:
                trainer.train_sft(
                    "_t3",
                    require_dataset("safe_sft_dataloader", safe_sft_dataloader),
                    epochs=epochs,
                    max_steps=max_steps,
                )
        elif run.run_type == "t8_cumulative_opposite":
            if use_simpo:
                trainer.train_principle_simpo(
                    "_t8",
                    require_dataset("unsafe_simpo_dataset", unsafe_simpo_dataset),
                    epochs=epochs,
                    max_steps=max_steps,
                )
            else:
                trainer.train_sft(
                    "_t8",
                    require_dataset("unsafe_sft_dataloader", unsafe_sft_dataloader),
                    epochs=epochs,
                    max_steps=max_steps,
                )
        elif run.run_type == "t4_anti_target":
            if use_simpo:
                trainer.train_anti_simpo(
                    "_t4",
                    require_dataset("unsafe_simpo_dataset", unsafe_simpo_dataset),
                    epochs=epochs,
                    max_steps=max_steps,
                )
            else:
                trainer.train_anti_sft(
                    "_t4",
                    require_dataset("unsafe_sft_dataloader", unsafe_sft_dataloader),
                    epochs=epochs,
                    max_steps=max_steps,
                )
        elif run.run_type == "t9_anti_opposite":
            if use_simpo:
                trainer.train_anti_simpo(
                    "_t9",
                    require_dataset("safe_simpo_dataset", safe_simpo_dataset),
                    epochs=epochs,
                    max_steps=max_steps,
                )
            else:
                trainer.train_anti_sft(
                    "_t9",
                    require_dataset("safe_sft_dataloader", safe_sft_dataloader),
                    epochs=epochs,
                    max_steps=max_steps,
                )
        elif run.run_type == "t5_negated_target":
            if use_simpo:
                trainer.train_negated_simpo(
                    "_t5",
                    require_dataset("safe_simpo_dataset", safe_simpo_dataset),
                    epochs=epochs,
                    max_steps=max_steps,
                )
            else:
                trainer.train_anti_sft(
                    "_t5",
                    require_dataset("safe_sft_dataloader", safe_sft_dataloader),
                    epochs=epochs,
                    max_steps=max_steps,
                )
        elif run.run_type == "t10_negated_opposite":
            if use_simpo:
                trainer.train_negated_simpo(
                    "_t10",
                    require_dataset("unsafe_simpo_dataset", unsafe_simpo_dataset),
                    epochs=epochs,
                    max_steps=max_steps,
                )
            else:
                trainer.train_anti_sft(
                    "_t10",
                    require_dataset("unsafe_sft_dataloader", unsafe_sft_dataloader),
                    epochs=epochs,
                    max_steps=max_steps,
                )
        elif run.run_type == "t11_baseline":
            trainer.train_language_baseline(
                require_dataset("language_dataloader", language_dataloader),
                epochs=epochs,
                max_steps=max_steps,
            )
        else:
            raise ValueError(f"Unknown run type: {run.run_type!r}")

        # Capture post-training metrics if enabled
        if self._metrics_capture_enabled and self._metrics_capture and run.delta_name:
            pref_dataset = None
            if run.run_type in (
                "t1_prefopt_target", "t2_sft_target",
                "t3_cumulative_target", "t5_negated_target",
            ):
                pref_dataset = safe_simpo_dataset
            elif run.run_type in (
                "t6_prefopt_opposite", "t7_sft_opposite",
                "t8_cumulative_opposite", "t10_negated_opposite",
            ):
                pref_dataset = unsafe_simpo_dataset

            self._metrics_capture.capture_post_metrics(
                delta_name=run.delta_name,
                eval_dataloader=self._eval_dataloader,
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
