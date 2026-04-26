#!/usr/bin/env python3
"""
DCAF Analyze CLI entry point (§12 Complete Pipeline: analysis phase).

Analyze saved DCAF training runs without retraining.

Usage:
    # Full domain confidence analysis (C_W + C_A + C_G + triangulation)
    dcaf analyze -r ./runs/run_001/

    # Single-domain analysis
    dcaf analyze -r ./runs/run_001/ --weight
    dcaf analyze -r ./runs/run_001/ --activation

    # Ablation testing
    dcaf analyze -r ./runs/run_001/ --single-ablation
    dcaf analyze -r ./runs/run_001/ --pair-ablation
    dcaf analyze -r ./runs/run_001/ --group-ablation
    dcaf analyze -r ./runs/run_001/ --binary-ablation

    # Cross-run analysis (find stable core across multiple runs)
    dcaf analyze --cross-run ./runs/run_001/ ./runs/run_002/

    # Full analysis with ablation, SAE validation, verbose output
    dcaf analyze -r ./runs/run_001/ --single-ablation --validate-sae --verbose -o results.json
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from dcaf.cli._analyze import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
