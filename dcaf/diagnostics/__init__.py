"""Training diagnostics (§7, Def 7.1-7.3).

Public API:
- ActivationDeltaAlignment / compute_activation_delta_alignment: Def 7.1
- CurvatureMetrics / OnlineCurvatureTracker / init_curvature_tracker / update_curvature_tracker / finalize_curvature: Def 7.2
"""

from dcaf.diagnostics.alignment import (
    ActivationDeltaAlignment,
    compute_activation_delta_alignment,
)
from dcaf.diagnostics.curvature import (
    CurvatureMetrics,
    OnlineCurvatureTracker,
    init_curvature_tracker,
    update_curvature_tracker,
    finalize_curvature,
)

__all__ = [
    # Activation delta alignment (Def 7.1)
    "ActivationDeltaAlignment",
    "compute_activation_delta_alignment",
    # Curvature tracking (Def 7.2)
    "CurvatureMetrics",
    "OnlineCurvatureTracker",
    "init_curvature_tracker",
    "update_curvature_tracker",
    "finalize_curvature",
]
