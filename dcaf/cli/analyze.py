#!/usr/bin/env python3
"""
DCAF Analyze CLI entry point (app:pipeline: analysis phase).

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

    # Full analysis with ablation and verbose output
    dcaf analyze -r ./runs/run_001/ --single-ablation --verbose -o results.json
"""

from dcaf.cli._analyze import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
