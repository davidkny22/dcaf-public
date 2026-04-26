#!/usr/bin/env python3
"""
DCAF Discover CLI entry point (§3 Multi-Path Discovery).

Delegates to ``dcaf.cli._discover.cli``.

Usage:
    dcaf discover -r ./runs/run_001/
    dcaf discover -r ./runs/run_001/ -a -g
"""

from dcaf.cli._discover.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
