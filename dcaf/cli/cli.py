#!/usr/bin/env python3
"""
DCAF - Differential Circuit Analysis Framework

Unified CLI for training, discovery, and analysis of LLM safety circuits.

Usage:
    dcaf train --output ./runs/run_001/
    dcaf discover --run ./runs/run_001/
    dcaf analyze --run ./runs/run_001/
    dcaf --version
    dcaf --help
"""

import sys

from dcaf import __version__


def main():
    """Main entry point for unified DCAF CLI."""
    # Handle --version and -V before anything else
    if len(sys.argv) >= 2 and sys.argv[1] in ("--version", "-V"):
        print(f"dcaf {__version__}")
        sys.exit(0)

    # Handle no arguments or --help
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print_help()
        sys.exit(0)

    command = sys.argv[1]

    if command == "train":
        # Remove 'train' from argv and call train main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from dcaf.cli.train import main as train_main
        train_main()

    elif command == "discover":
        # Remove 'discover' from argv and call discover main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from dcaf.cli._discover.cli import main as discover_main
        discover_main()

    elif command == "analyze":
        # Remove 'analyze' from argv and call analyze main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from dcaf.cli._analyze.cli import main as analyze_main
        analyze_main()

    else:
        print(f"Unknown command: {command}")
        print("Use 'dcaf --help' for usage information.")
        sys.exit(1)


def print_help():
    """Print unified CLI help."""
    print(f"""DCAF - Differential Circuit Analysis Framework v{__version__}

Usage:
    dcaf <command> [options]

Commands:
    train       Train a DCAF variant and save weight deltas (GPU)
    discover    Discover candidate parameters using multiple paths
    analyze     Analyze saved training runs with criteria (CPU/GPU)

Workflow:
    1. dcaf train    - Train and save weight deltas
    2. dcaf discover - Find parameters to analyze (H_W | H_A | H_G)
    3. dcaf analyze  - Compute confidence and classify

Options:
    --version, -V   Show version and exit
    --help, -h      Show this help message and exit

Examples:
    dcaf train --anti --negated --output ./runs/run_001/
    dcaf discover --run ./runs/run_001/ --activation --gradient
    dcaf analyze --run ./runs/run_001/ --full-analysis

For command-specific help:
    dcaf train --help
    dcaf discover --help
    dcaf analyze --help
""")


if __name__ == "__main__":
    main()
