"""
DCAF CLI Common Utilities.

Shared argument definitions, device detection, and utilities for consistent
CLI behavior across train, discover, and analyze commands.
"""

import argparse
import logging
from typing import Optional

import torch

from dcaf.core.defaults import (
    BETA_PATH,
    LANGUAGE_PERCENTILE,
    TAU_A_DEFAULT,
    TAU_ACT,
    TAU_BASE,
    TAU_COMP,
    TAU_G_DEFAULT,
    TAU_GRAD,
    TAU_SIG,
    TAU_W_DEFAULT,
    TOP_K_CANDIDATES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Device Detection
# =============================================================================

def detect_device(requested: str = "auto") -> str:
    """
    Detect the best available device.

    Args:
        requested: Requested device ("auto", "cuda", "mps", "cpu")

    Returns:
        Available device string
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
    elif requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
    else:
        return "cpu"


# =============================================================================
# Standard Argument Adders
# =============================================================================

def add_run_path_args(parser: argparse.ArgumentParser, required: bool = True) -> None:
    """Add run path argument (input directory)."""
    parser.add_argument(
        "--run", "-r",
        required=required,
        help="Path to DCAF training run directory",
    )


def add_output_path_args(parser: argparse.ArgumentParser) -> None:
    """Add output path argument (output file/directory)."""
    parser.add_argument(
        "--output", "-o",
        help="Output path for results",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model specification arguments."""
    parser.add_argument(
        "--model",
        help="Model name (auto-detected from run metadata if not specified)",
    )


def add_device_args(parser: argparse.ArgumentParser) -> None:
    """Add device specification arguments."""
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for computation (default: auto-detect)",
    )


def add_verbose_args(parser: argparse.ArgumentParser) -> None:
    """Add verbosity arguments."""
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (DEBUG logging)",
    )


def add_probe_args(parser: argparse.ArgumentParser, default: int = 50) -> None:
    """Add probe size arguments."""
    parser.add_argument(
        "--probe-size",
        type=int,
        default=default,
        help=f"Number of probe prompts (default: {default})",
    )


def add_top_k_args(parser: argparse.ArgumentParser) -> None:
    """Add top-k filtering arguments."""
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_CANDIDATES,
        help=f"Number of top candidates to return (default: {TOP_K_CANDIDATES})",
    )


# =============================================================================
# Threshold Arguments (Unified Naming)
# =============================================================================

def add_discovery_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Add discovery threshold arguments (for discover CLI)."""
    group = parser.add_argument_group("Discovery Thresholds")

    group.add_argument(
        "--significance-threshold",
        type=float,
        default=TAU_SIG,
        dest="tau_sig",
        help=f"Weight significance percentile (default: {TAU_SIG})",
    )
    group.add_argument(
        "--baseline-threshold",
        type=float,
        default=TAU_BASE,
        dest="tau_base",
        help=f"Baseline NOT-significant threshold (default: {TAU_BASE})",
    )
    group.add_argument(
        "--activation-threshold",
        type=float,
        default=TAU_ACT,
        dest="tau_act",
        help=f"Activation discovery percentile (default: {TAU_ACT})",
    )
    group.add_argument(
        "--gradient-threshold",
        type=float,
        default=TAU_GRAD,
        dest="tau_grad",
        help=f"Gradient discovery percentile (default: {TAU_GRAD})",
    )
    group.add_argument(
        "--component-threshold",
        type=float,
        default=TAU_COMP,
        dest="tau_comp",
        help=f"Component screening percentile for H_A (default: {TAU_COMP})",
    )
    group.add_argument(
        "--beta-path",
        type=float,
        default=BETA_PATH,
        help=f"Multi-path bonus weight (default: {BETA_PATH})",
    )


def add_confidence_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Add confidence threshold arguments (for analyze CLI)."""
    group = parser.add_argument_group("Confidence Thresholds")

    group.add_argument(
        "--weight-confidence",
        type=float,
        default=TAU_W_DEFAULT,
        dest="tau_W",
        help=f"Weight confidence threshold (default: {TAU_W_DEFAULT})",
    )
    group.add_argument(
        "--activation-confidence",
        type=float,
        default=TAU_A_DEFAULT,
        dest="tau_A",
        help=f"Activation confidence threshold (default: {TAU_A_DEFAULT})",
    )
    group.add_argument(
        "--geometry-confidence",
        type=float,
        default=TAU_G_DEFAULT,
        dest="tau_G",
        help=f"Geometry confidence threshold (default: {TAU_G_DEFAULT})",
    )


def add_significance_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Add significance threshold arguments (for train and analyze CLIs)."""
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=TAU_SIG,
        dest="tau_sig",
        help=f"Significance percentile threshold (default: {TAU_SIG})",
    )
    parser.add_argument(
        "--language-percentile",
        type=float,
        default=LANGUAGE_PERCENTILE,
        help=f"Language baseline percentile (default: {LANGUAGE_PERCENTILE})",
    )


# =============================================================================
# Logging Configuration
# =============================================================================

def configure_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s" if not verbose else "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# =============================================================================
# Common Validation
# =============================================================================

def validate_run_path(run_path: str) -> bool:
    """Validate that run path exists and contains expected files."""
    from pathlib import Path
    path = Path(run_path)

    if not path.exists():
        logger.error(f"Run directory not found: {run_path}")
        return False

    # Check for metadata or deltas directory
    if not (path / "metadata.json").exists() and not (path / "deltas").exists():
        logger.warning(f"Run directory may be incomplete: {run_path}")

    return True


def load_model_name_from_metadata(run_path: str) -> Optional[str]:
    """Load model name from run metadata."""
    from pathlib import Path

    from dcaf.storage import DeltaStore

    path = Path(run_path)
    if not path.exists():
        return None

    try:
        delta_store = DeltaStore(path)
        metadata = delta_store.load_metadata()
        if metadata and hasattr(metadata, 'model_name'):
            return metadata.model_name
    except Exception:
        pass

    return None


__all__ = [
    # Device
    "detect_device",
    # Argument adders
    "add_run_path_args",
    "add_output_path_args",
    "add_model_args",
    "add_device_args",
    "add_verbose_args",
    "add_probe_args",
    "add_top_k_args",
    # Threshold argument adders
    "add_discovery_threshold_args",
    "add_confidence_threshold_args",
    "add_significance_threshold_args",
    # Logging
    "configure_logging",
    # Validation
    "validate_run_path",
    "load_model_name_from_metadata",
]
