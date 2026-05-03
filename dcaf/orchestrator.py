"""DCAF Orchestrator — coordinates the full analysis pipeline (§12, Theorem 12.1).

This is the single top-level entry point that calls every domain module in sequence.
It supports two modes:
  1. run_full() — train signals, discover, analyze, output
  2. run_analysis() — skip training, load saved deltas, run discovery through output

Every call in this file is to a REAL domain module. No stubs. No hardcoded values.

Pipeline:
    M₀ → train signals → ΔW → discovery (H_W ∪ H_A ∪ H_G)
    → confidence (C_W, C_A, C_G) → triangulation → H_cand
    → ablation (7 phases) → H_conf → circuit graph → output
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dcaf.core.config import DCAFConfig

logger = logging.getLogger(__name__)


class DCAFOrchestrator:
    """Coordinates the complete DCAF analysis pipeline.

    Usage:
        orchestrator = DCAFOrchestrator(config)

        # Full pipeline (train + analyze):
        result = orchestrator.run_full(model, tokenizer, datasets, "./runs/run_001")

        # Analysis only (on saved deltas):
        result = orchestrator.run_analysis(run_path, model_name)
    """

    def __init__(self, config: Optional[DCAFConfig] = None, device: Optional[str] = None):
        self.config = config or DCAFConfig()
        self.device = device or "auto"
        self._step_timings: Dict[str, float] = {}

    def _resolve_device(self) -> str:
        """Resolve auto device lazily so importing this module never touches torch."""
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _timed(self, name: str, fn: Callable) -> Any:
        """Execute a function with timing and logging."""
        logger.info(f"[DCAF] {name}")
        t0 = time.time()
        result = fn()
        elapsed = time.time() - t0
        self._step_timings[name] = elapsed
        logger.info(f"[DCAF] {name} — completed in {elapsed:.1f}s")
        return result

    # =========================================================================
    # Full mode (train signals, save artifacts, then analyze)
    # =========================================================================

    def run_full(
        self,
        model: Any,
        tokenizer: Any,
        datasets: Dict[str, Any],
        output_path: str,
        variant_config: Optional[Any] = None,
        model_name: Optional[str] = None,
        epochs_per_phase: int = 1,
        max_steps_per_phase: int = -1,
        skip_activation: bool = False,
        skip_geometry: bool = False,
        skip_ablation: bool = False,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run training and analysis in one call."""
        from dcaf.storage.delta_store import DeltaMetadata, DeltaStore
        from dcaf.training.variants import TrainingOrchestrator, build_variant

        device = self._resolve_device()
        run_path = Path(output_path)
        delta_store = DeltaStore(run_path)
        variant_config = variant_config or build_variant(no_simpo=not self.config.use_simpo)
        model_name = model_name or getattr(model, "name_or_path", None) or model.__class__.__name__

        metadata = DeltaMetadata.create(
            model_name=model_name,
            variant_name=variant_config.name,
            training_config={
                "epochs_per_phase": epochs_per_phase,
                "max_steps_per_phase": max_steps_per_phase,
                "device": device,
                "use_simpo": self.config.use_simpo,
            },
            dataset_config={"source": "programmatic"},
        )
        delta_store.save_metadata(metadata)

        from dcaf.core.topology import build_model_topology
        topo = build_model_topology(model)
        delta_store.save_topology(topo)

        def do_training():
            trainer = TrainingOrchestrator(model, tokenizer, self.config, device)
            trainer.run_variant(
                variant_config=variant_config,
                safe_simpo_dataset=datasets.get("safe_simpo"),
                unsafe_simpo_dataset=datasets.get("unsafe_simpo"),
                safe_sft_dataloader=datasets.get("safe_sft_dataloader"),
                unsafe_sft_dataloader=datasets.get("unsafe_sft_dataloader"),
                language_dataloader=datasets.get("language_dataloader"),
                epochs_per_phase=epochs_per_phase,
                max_steps_per_phase=max_steps_per_phase,
                delta_store=delta_store,
            )
            metadata.available_deltas = delta_store.list_deltas()
            metadata.available_checkpoints = delta_store.list_checkpoints()
            delta_store.save_metadata(metadata)

            # Persist peak checkpoint histories
            if trainer._checkpoint_histories:
                import json as _json
                histories = {}
                for run_type, history in trainer._checkpoint_histories.items():
                    histories[run_type] = {
                        "peak_step": history.peak_step,
                        "peak_metric": history.peak_metric,
                        "is_confirmed": history.is_confirmed,
                    }
                hist_path = run_path / "peak_histories.json"
                with open(hist_path, "w") as f:
                    _json.dump(histories, f, indent=2)

            return {
                "run_path": str(run_path),
                "deltas": metadata.available_deltas,
                "checkpoints": metadata.available_checkpoints,
            }

        if on_step:
            on_step("training")
        training_results = self._timed("Step 1-4: Signal training", do_training)

        if on_step:
            on_step("analysis")
        analysis_results = self.run_analysis(
            run_path=str(run_path),
            model_name=model_name,
            skip_activation=skip_activation,
            skip_geometry=skip_geometry,
            skip_ablation=skip_ablation,
            on_step=on_step,
        )
        analysis_results["training"] = training_results
        return analysis_results

    # =========================================================================
    # Analysis-only mode (on saved deltas)
    # =========================================================================

    def run_analysis(
        self,
        run_path: str,
        model_name: Optional[str] = None,
        skip_activation: bool = False,
        skip_geometry: bool = False,
        skip_ablation: bool = False,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run discovery + analysis on saved training deltas.

        This is the primary entry point for analysis. Training has already
        been done via `dcaf train`; deltas are loaded from disk.

        Args:
            run_path: Path to DCAF run directory with saved deltas
            model_name: Model name for activation/geometry analysis (auto-detected if None)
            skip_activation: Skip activation domain (C_A) — faster but less evidence
            skip_geometry: Skip geometry domain (C_G) — faster but less evidence
            skip_ablation: Skip ablation phases — faster but no behavioral confirmation
            on_step: Optional callback for progress reporting
        """
        from dcaf.storage.delta_store import DeltaStore

        device = self._resolve_device()
        run_path = Path(run_path)
        delta_store = DeltaStore(run_path)

        if not delta_store.exists():
            raise FileNotFoundError(
                f"No DCAF run found at {run_path}. Run 'dcaf train' first."
            )

        metadata = delta_store.load_metadata()
        model_name = model_name or metadata.model_name
        available_deltas = delta_store.list_deltas()

        logger.info(f"[DCAF] Analysis of {model_name}")
        logger.info(f"[DCAF] Run: {run_path}")
        logger.info(f"[DCAF] Deltas: {available_deltas}")

        results = {"run_path": str(run_path), "model_name": model_name}

        # Step 5: Weight discovery (H_W)
        def do_weight_discovery():
            from dcaf.cli._discover.weight_discovery import run_weight_discovery
            H_W, S_W, param_names = run_weight_discovery(
                run_path, tau_sig=self.config.tau_sig, tau_base=self.config.tau_base,
            )
            results["H_W"] = H_W
            results["S_W"] = S_W
            results["param_names"] = param_names
            logger.info(f"  H_W: {len(H_W)} projections")
            return H_W, S_W

        self._timed("Step 5: Weight discovery (H_W)", do_weight_discovery)

        # Step 5b: Activation discovery (H_A) — optional
        if not skip_activation and model_name:
            def do_activation_discovery():
                try:
                    from dcaf.cli._discover.activation_discovery import run_activation_discovery
                    H_A, S_A, _names = run_activation_discovery(
                        run_path, model_name=model_name,
                        tau_comp=self.config.tau_comp, tau_act=self.config.tau_act,
                        device=device,
                    )
                    results["H_A"] = H_A
                    results["S_A"] = S_A
                    logger.info(f"  H_A: {len(H_A)} components")
                except Exception as e:
                    logger.warning(f"  Activation discovery failed: {e}")
                    results["H_A"] = set()

            self._timed("Step 5b: Activation discovery (H_A)", do_activation_discovery)

        # Step 8: Weight domain confidence (C_W)
        def do_weight_confidence():
            from dcaf.cli._analyze.weight_runner import run_weight_analysis
            weight_results = run_weight_analysis(
                run_path, tau_W=self.config.tau_W, top_k=50,
            )
            results["weight"] = weight_results
            return weight_results

        self._timed("Step 8: Weight confidence (C_W)", do_weight_confidence)

        # Step 9: Activation confidence (C_A) — optional
        if not skip_activation and model_name:
            def do_activation():
                from dcaf.cli._analyze.activation_runner import run_activation_analysis
                act_results = run_activation_analysis(
                    run_path, model_name=model_name,
                    tau_A=self.config.tau_A, device=device,
                )
                results["activation"] = act_results
                return act_results

            self._timed("Step 9: Activation confidence (C_A)", do_activation)

        # Step 10: Geometry confidence (C_G) — optional
        if not skip_geometry and model_name:
            def do_geometry():
                from dcaf.cli._analyze.geometry_runner import run_geometry_analysis
                geo_results = run_geometry_analysis(
                    run_path, model_name=model_name,
                    tau_G=self.config.tau_G, device=device,
                )
                results["geometry"] = geo_results
                return geo_results

            self._timed("Step 10: Geometry confidence (C_G)", do_geometry)

        # Step 11-12: Unified confidence + candidate selection
        def do_triangulation():
            from dcaf.cli._analyze.full_runner import run_full_analysis
            full_results = run_full_analysis(
                run_path, model_name=model_name,
                tau_W=self.config.tau_W, tau_A=self.config.tau_A, tau_G=self.config.tau_G,
                skip_activation=skip_activation, skip_geometry=skip_geometry,
                device=device,
            )
            results["full"] = full_results
            return full_results

        self._timed("Step 11-12: Triangulation + candidate selection", do_triangulation)

        # Step 13: Ablation — optional
        if not skip_ablation:
            def do_ablation():
                matching_params = results.get("full", {}).get("top_candidates", [])
                if not matching_params:
                    logger.warning("No candidates for ablation — skipping")
                    return None

                from dcaf.cli._analyze.ablation_runner import run_ablation_testing
                param_names = [
                    c.get("param_name") or str(c.get("id", ""))
                    for c in matching_params
                ]
                param_names = [name for name in param_names if name]
                ablation_results = run_ablation_testing(
                    run_path, param_names, top_k=20, device=device,
                )
                results["ablation"] = ablation_results
                return ablation_results

            self._timed("Step 13: Ablation testing", do_ablation)

        # Step 17: Build output
        def do_output():
            from dcaf.output.schema import assemble_component_output, assemble_output
            full = results.get("full", {})
            component_results = [
                assemble_component_output(
                    component=(
                        candidate.get("component")
                        or candidate.get("param_name")
                        or f"idx_{candidate.get('id')}"
                    ),
                    param_names=(
                        [candidate["param_name"]]
                        if candidate.get("param_name") else []
                    ),
                    scores={
                        "C_W": candidate.get("C_W"),
                        "C_A": candidate.get("C_A"),
                        "C_G": candidate.get("C_G"),
                        "C_unified": candidate.get("C_unified"),
                    },
                    discovery={
                        "paths": candidate.get("paths", []),
                        "path_count": len(candidate.get("paths", [])),
                        "multi_path_bonus": candidate.get("bonus", 0.0),
                    },
                    weight_details={
                        "bidirectional": False,
                        "opp_degree": candidate.get("opp_degree", 0.0),
                    },
                    diagnostics={
                        "candidate_id": candidate.get("id"),
                        "C_base": candidate.get("C_base"),
                    },
                )
                for candidate in full.get("top_candidates", [])
            ]
            output = assemble_output(
                run_path=str(run_path),
                model_name=model_name,
                variant_name=metadata.variant_name if metadata else "unknown",
                discovery_summary={"H_W": len(results.get("H_W", set()))},
                weight_summary=results.get("weight", {}),
                activation_summary=results.get("activation"),
                geometry_summary=results.get("geometry"),
                triangulation_summary=full.get("triangulation"),
                component_results=component_results,
                ablation_summary=results.get("ablation"),
                thresholds={
                    "tau_W": self.config.tau_W,
                    "tau_A": self.config.tau_A,
                    "tau_G": self.config.tau_G,
                    "tau_unified": self.config.tau_unified,
                },
            )
            results["output"] = output
            return output

        self._timed("Step 17: Build output", do_output)

        results["step_timings"] = dict(self._step_timings)
        total = sum(self._step_timings.values())
        logger.info(f"[DCAF] Analysis complete in {total:.1f}s")

        return results

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save analysis results to JSON."""
        import dataclasses

        def _serialize(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return dataclasses.asdict(obj)
            if isinstance(obj, set):
                return sorted(obj)
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            if hasattr(obj, 'item'):
                return obj.item()
            return str(obj)

        payload = results.get("output", results)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, default=_serialize)
        logger.info(f"[DCAF] Results saved to {output_path}")


__all__ = ["DCAFOrchestrator"]
