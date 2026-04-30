"""IOI (Indirect Object Identification) circuit discovery demo for DCAF."""

from demos.ioi.data import load_ioi_dataset, create_sft_dataloaders, create_neutral_dataloader
from demos.ioi.probes import build_ioi_probe_set
from demos.ioi.known_circuit import KNOWN_IOI_HEADS, validate_against_known
from demos.ioi.visualization import plot_circuit_diagram

__all__ = [
    "load_ioi_dataset",
    "create_sft_dataloaders",
    "create_neutral_dataloader",
    "build_ioi_probe_set",
    "KNOWN_IOI_HEADS",
    "validate_against_known",
    "plot_circuit_diagram",
]
