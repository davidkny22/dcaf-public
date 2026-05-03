"""Training diagnostics (sec:training-diagnostics).

Public API:
- ActivationDeltaAlignment / compute_activation_delta_alignment: def:delta-alignment
- CurvatureMetrics / OnlineCurvatureTracker / init_curvature_tracker / update_curvature_tracker / finalize_curvature: def:online-curvature
"""

from dcaf.diagnostics.alignment import (
    ActivationDeltaAlignment,
    compute_activation_delta_alignment,
)
from dcaf.diagnostics.curvature import (
    CurvatureMetrics,
    OnlineCurvatureTracker,
    finalize_curvature,
    init_curvature_tracker,
    update_curvature_tracker,
)

__all__ = [
    # Activation delta alignment (def:delta-alignment)
    "ActivationDeltaAlignment",
    "compute_activation_delta_alignment",
    # Curvature tracking (def:online-curvature)
    "CurvatureMetrics",
    "OnlineCurvatureTracker",
    "init_curvature_tracker",
    "update_curvature_tracker",
    "finalize_curvature",
]
