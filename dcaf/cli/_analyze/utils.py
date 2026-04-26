"""Utility functions for the analysis CLI."""

import logging

logger = logging.getLogger(__name__)


def list_available_deltas(available_deltas: list) -> None:
    """Print available deltas in this run to the logger."""
    logger.info("\nAvailable deltas for this run:")
    logger.info("-" * 50)

    if not available_deltas:
        logger.info("  (none)")
        return

    for delta_name in sorted(available_deltas):
        logger.info(f"  {delta_name}")


__all__ = ["list_available_deltas"]
