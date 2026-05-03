"""
Delta Store: Manages saving/loading of DCAF delta tensors and metadata (def:weight-delta).

This module enables the decoupled DCAF architecture where training (expensive GPU)
and analysis (cheap CPU) can happen independently.

Storage format:
    ./run_001/
      metadata.json
      deltas/
        delta_t1_prefopt_target.pt
        delta_t6_prefopt_opposite.pt
        ...
      checkpoints/
        checkpoint_t1.pt
        checkpoint_t6.pt
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import torch

if TYPE_CHECKING:
    from dcaf.domains.activation.probe_set import ProbeSet
    from dcaf.domains.activation.results import ActivationSnapshot


@dataclass
class DeltaMetadata:
    """Metadata for a DCAF training run."""

    run_id: str                           # UUID
    timestamp: str                        # ISO timestamp
    model_name: str                       # "EleutherAI/pythia-410m"
    variant_name: str                     # e.g., "prefopt_sft_anti_negated"
    training_config: Dict[str, Any]       # steps, lr, batch_size, etc.
    dataset_config: Dict[str, Any]        # categories, severity, samples
    available_deltas: List[str] = field(default_factory=list)
    available_checkpoints: List[str] = field(default_factory=list)

    # Activation capture metadata
    activation_capture_enabled: bool = False
    probe_set_name: str = ""
    probe_set_size: int = 0
    available_activation_snapshots: List[str] = field(default_factory=list)
    activation_config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        model_name: str,
        variant_name: str,
        training_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
    ) -> "DeltaMetadata":
        """Create new metadata with auto-generated run_id and timestamp."""
        return cls(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_name=model_name,
            variant_name=variant_name,
            training_config=training_config,
            dataset_config=dataset_config,
            available_deltas=[],
            available_checkpoints=[],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        data = asdict(self)
        extra = data.pop("extra", {})
        data.update(extra)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeltaMetadata":
        """Create from dict (e.g., loaded from JSON)."""
        field_names = {f.name for f in fields(cls)}
        known = {k: v for k, v in data.items() if k in field_names and k != "extra"}
        metadata = cls(**known)
        metadata.extra.update(data.get("extra", {}))
        metadata.extra.update({k: v for k, v in data.items() if k not in field_names})
        return metadata


class DeltaStore:
    """
    Manages saving and loading of DCAF delta tensors and checkpoints.

    Provides persistent storage for training artifacts so analysis
    can happen independently without retraining.
    """

    def __init__(self, run_dir: Path):
        """
        Initialize DeltaStore for a specific run directory.

        Args:
            run_dir: Directory for this training run (e.g., ./runs/run_001/)
        """
        self.run_dir = Path(run_dir)
        self.deltas_dir = self.run_dir / "deltas"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.activations_dir = self.run_dir / "activations"
        self.metadata_path = self.run_dir / "metadata.json"
        self.probe_set_path = self.run_dir / "probe_set.json"

        self._metadata: Optional[DeltaMetadata] = None
        self._loaded_deltas: Dict[str, Dict[str, torch.Tensor]] = {}

    def _ensure_dirs(self) -> None:
        """Create directory structure if it doesn't exist."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.deltas_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.activations_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Saving
    # =========================================================================

    def save_delta(self, name: str, delta_dict: Dict[str, torch.Tensor]) -> Path:
        """
        Save a delta tensor dict to disk.

        Args:
            name: Delta name (e.g., "delta_t1_prefopt_target")
            delta_dict: {param_name: tensor} mapping

        Returns:
            Path to saved file
        """
        self._ensure_dirs()

        # Move tensors to CPU for storage
        cpu_delta = {k: v.cpu() for k, v in delta_dict.items()}

        path = self.deltas_dir / f"{name}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(cpu_delta, tmp_path)
        tmp_path.rename(path)

        # Invalidate cache for this delta
        self._loaded_deltas.pop(name, None)

        # Update metadata
        if self._metadata and name not in self._metadata.available_deltas:
            self._metadata.available_deltas.append(name)
            self.save_metadata(self._metadata)

        return path

    def save_checkpoint(self, name: str, weights: Dict[str, torch.Tensor]) -> Path:
        """
        Save a checkpoint (full weights) to disk.

        Only used for negated training dependencies (checkpoint_t1, checkpoint_t6).

        Args:
            name: Checkpoint name (e.g., "checkpoint_t1")
            weights: Full model weights dict

        Returns:
            Path to saved file
        """
        self._ensure_dirs()

        # Move tensors to CPU for storage
        cpu_weights = {k: v.cpu() for k, v in weights.items()}

        path = self.checkpoints_dir / f"{name}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(cpu_weights, tmp_path)
        tmp_path.rename(path)

        # Update metadata
        if self._metadata and name not in self._metadata.available_checkpoints:
            self._metadata.available_checkpoints.append(name)
            self.save_metadata(self._metadata)

        return path

    def save_metadata(self, metadata: DeltaMetadata) -> Path:
        """
        Save metadata to disk.

        Args:
            metadata: DeltaMetadata instance

        Returns:
            Path to saved file
        """
        self._ensure_dirs()
        self._metadata = metadata

        tmp_path = self.metadata_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        tmp_path.rename(self.metadata_path)

        return self.metadata_path

    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """
        Update existing metadata with additional fields.

        Args:
            updates: Dictionary of fields to add/update
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError("Metadata file does not exist. Call save_metadata first.")

        # Load existing
        with open(self.metadata_path, "r") as f:
            data = json.load(f)

        # Merge updates
        data.update(updates)
        if self._metadata is not None:
            field_names = {f.name for f in fields(DeltaMetadata)}
            for key, value in updates.items():
                if key in field_names and key != "extra":
                    setattr(self._metadata, key, value)
                else:
                    self._metadata.extra[key] = value

        # Save back
        with open(self.metadata_path, "w") as f:
            json.dump(data, f, indent=2)

    def save_activation_snapshot(self, name: str, snapshot: "ActivationSnapshot") -> Path:
        """
        Save an activation snapshot to disk.

        Args:
            name: Snapshot name (e.g., "pre_training", "after_t1_prefopt_target")
            snapshot: ActivationSnapshot instance

        Returns:
            Path to saved file
        """
        self._ensure_dirs()

        path = self.activations_dir / f"{name}.pt"
        snapshot.save(str(path))

        # Update metadata
        if self._metadata and name not in self._metadata.available_activation_snapshots:
            self._metadata.available_activation_snapshots.append(name)
            self.save_metadata(self._metadata)

        return path

    def save_topology(self, topology) -> Path:
        """Save model topology to disk for projection-level analysis.

        Must be called during training so analysis can decompose parameter-level
        deltas into projection-level slices without needing the model.

        Args:
            topology: ModelTopology from build_model_topology()

        Returns:
            Path to saved file
        """
        self._ensure_dirs()

        data = {
            "projections": topology.projections,
            "components": topology.components,
            "proj_to_components": {k: sorted(v) for k, v in topology.proj_to_components.items()},
            "component_to_projs": topology.component_to_projs,
            "proj_slices": {
                pid: {
                    "param_name": ps.param_name,
                    "row_start": ps.row_start, "row_end": ps.row_end,
                    "col_start": ps.col_start, "col_end": ps.col_end,
                }
                for pid, ps in topology.proj_slices.items()
            },
            "n_layers": topology.n_layers,
            "n_query_heads": topology.n_query_heads,
            "n_kv_heads": topology.n_kv_heads,
            "head_dim": topology.head_dim,
            "hidden_size": topology.hidden_size,
            "intermediate_size": topology.intermediate_size,
            "architecture": topology.architecture,
        }

        path = self.run_dir / "topology.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path

    def load_topology(self):
        """Load saved model topology.

        Returns:
            ModelTopology instance

        Raises:
            FileNotFoundError: If topology was not saved during training
        """
        from dcaf.core.topology import ModelTopology, ProjectionSlice

        path = self.run_dir / "topology.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Topology not found at {path}. "
                f"Re-run training with a version that saves topology, "
                f"or provide the model for topology construction."
            )

        with open(path, "r") as f:
            data = json.load(f)

        proj_slices = {
            pid: ProjectionSlice(
                param_name=ps["param_name"],
                row_start=ps["row_start"], row_end=ps["row_end"],
                col_start=ps["col_start"], col_end=ps["col_end"],
            )
            for pid, ps in data["proj_slices"].items()
        }

        return ModelTopology(
            projections=data["projections"],
            components=data["components"],
            proj_to_components={k: set(v) for k, v in data["proj_to_components"].items()},
            component_to_projs=data["component_to_projs"],
            proj_slices=proj_slices,
            n_layers=data["n_layers"],
            n_query_heads=data["n_query_heads"],
            n_kv_heads=data["n_kv_heads"],
            head_dim=data["head_dim"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            architecture=data["architecture"],
        )

    def has_topology(self) -> bool:
        """Check if topology was saved for this run."""
        return (self.run_dir / "topology.json").exists()

    def save_probe_set(self, probe_set: "ProbeSet") -> Path:
        """
        Save probe set configuration to JSON.

        Args:
            probe_set: ProbeSet instance

        Returns:
            Path to saved file
        """
        self._ensure_dirs()

        with open(self.probe_set_path, "w") as f:
            json.dump(probe_set.to_dict(), f, indent=2)

        return self.probe_set_path

    # =========================================================================
    # Loading
    # =========================================================================

    def load_delta(self, name: str) -> Dict[str, torch.Tensor]:
        """
        Load a delta tensor dict from disk.

        Uses lazy loading - subsequent calls return cached tensors.

        Args:
            name: Delta name (e.g., "delta_t1_prefopt_target")

        Returns:
            {param_name: tensor} mapping

        Raises:
            FileNotFoundError: If delta file doesn't exist
        """
        if name in self._loaded_deltas:
            return self._loaded_deltas[name]

        path = self.deltas_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Delta not found: {path}")

        delta = torch.load(path, map_location="cpu", weights_only=True)
        self._loaded_deltas[name] = delta
        return delta

    def load_checkpoint(self, name: str) -> Dict[str, torch.Tensor]:
        """
        Load a checkpoint from disk.

        Args:
            name: Checkpoint name (e.g., "checkpoint_t1")

        Returns:
            Full model weights dict

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        path = self.checkpoints_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        return torch.load(path, map_location="cpu", weights_only=True)

    def load_metadata(self) -> DeltaMetadata:
        """
        Load metadata from disk.

        Returns:
            DeltaMetadata instance

        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        with open(self.metadata_path, "r") as f:
            data = json.load(f)

        self._metadata = DeltaMetadata.from_dict(data)
        return self._metadata

    def load_activation_snapshot(self, name: str) -> "ActivationSnapshot":
        """
        Load an activation snapshot from disk.

        Args:
            name: Snapshot name (e.g., "pre_training", "after_t1_prefopt_target")

        Returns:
            ActivationSnapshot instance

        Raises:
            FileNotFoundError: If snapshot file doesn't exist
        """
        from dcaf.domains.activation.results import ActivationSnapshot

        path = self.activations_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Activation snapshot not found: {path}")

        return ActivationSnapshot.load(str(path))

    def load_probe_set(self) -> "ProbeSet":
        """
        Load probe set configuration from JSON.

        Returns:
            ProbeSet instance

        Raises:
            FileNotFoundError: If probe set file doesn't exist
        """
        from dcaf.domains.activation.probe_set import ProbeSet

        if not self.probe_set_path.exists():
            raise FileNotFoundError(f"Probe set not found: {self.probe_set_path}")

        with open(self.probe_set_path, "r") as f:
            data = json.load(f)

        return ProbeSet.from_dict(data)

    # =========================================================================
    # Validation
    # =========================================================================

    def list_deltas(self) -> List[str]:
        """
        List all available deltas in this run.

        Returns actual files on disk, not just what metadata claims.
        Logs a warning if metadata and disk disagree.

        Returns:
            List of delta names (without .pt extension)
        """
        if not self.deltas_dir.exists():
            return []

        on_disk = sorted(p.stem for p in self.deltas_dir.glob("*.pt"))

        if self._metadata and self._metadata.available_deltas:
            in_meta = set(self._metadata.available_deltas)
            on_disk_set = set(on_disk)
            if in_meta != on_disk_set:
                extra = on_disk_set - in_meta
                missing = in_meta - on_disk_set
                if extra:
                    logger.warning(f"Deltas on disk but not in metadata: {sorted(extra)}")
                if missing:
                    logger.warning(f"Deltas in metadata but not on disk: {sorted(missing)}")

        return on_disk

    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints in this run.

        Returns:
            List of checkpoint names (without .pt extension)
        """
        if not self.checkpoints_dir.exists():
            return []

        return sorted(p.stem for p in self.checkpoints_dir.glob("*.pt"))

    def list_activation_snapshots(self) -> List[str]:
        """
        List all available activation snapshots in this run.

        Returns:
            List of snapshot names (without .pt extension)
        """
        if not self.activations_dir.exists():
            return []

        return sorted(p.stem for p in self.activations_dir.glob("*.pt"))

    def validate_for_criteria(
        self,
        criteria_name: str,
        requirements: Dict[str, List[str]]
    ) -> Tuple[bool, List[str]]:
        """
        Check if this run has all deltas required for a criteria.

        Args:
            criteria_name: e.g., "prefopt_sft_anti_negated"
            requirements: dict mapping criteria name to list of required delta names

        Returns:
            (is_valid, missing_deltas) tuple
        """
        required = requirements.get(criteria_name, [])
        if not required:
            return False, [f"Unknown criteria: {criteria_name}"]

        available = set(self.list_deltas())
        missing = [r for r in required if r not in available]

        return len(missing) == 0, missing

    def exists(self) -> bool:
        """Check if this run directory exists and has metadata."""
        return self.metadata_path.exists()

    def clear_cache(self) -> None:
        """Clear loaded delta cache to free memory."""
        self._loaded_deltas.clear()


__all__ = [
    "DeltaMetadata",
    "DeltaStore",
]
