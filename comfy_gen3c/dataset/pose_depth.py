"""Pose and depth recovery fallback for GEN3C pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch


@dataclass
class PoseDepthResult:
    poses: torch.Tensor  # (N, 4, 4)
    intrinsics: torch.Tensor  # (N, 3, 3)
    depths: Optional[torch.Tensor]  # (N, H, W)


def estimate_from_video(video_path: Path) -> PoseDepthResult:
    """Placeholder for pose/depth estimation using third-party tools.

    Implementations can wrap ViPE, COLMAP, or any preferred structure-from-motion
    pipeline. This placeholder raises a clear error to signal that the optional
    dependency is not installed.
    """
    raise NotImplementedError(
        "Pose/depth estimation is not bundled. Install ViPE or COLMAP bindings "
        "and implement comfy_gen3c.dataset.pose_depth.estimate_from_video."
    )


__all__ = ["PoseDepthResult", "estimate_from_video"]
