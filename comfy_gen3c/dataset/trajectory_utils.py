"""Shared utilities for trajectory conversion and manipulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..constants import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH
from ..utils import safe_matrix_inverse


def pose_result_to_trajectory(
    result: Any,  # PoseDepthResult from pose_depth module
    fps: int = DEFAULT_FPS,
    source_name: str = "pose_recovery",
    frame_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Convert PoseDepthResult to GEN3C trajectory format.

    Args:
        result: PoseDepthResult with poses and intrinsics
        fps: Frames per second for trajectory
        source_name: Source identifier string
        frame_size: Optional (width, height) tuple; uses defaults if None

    Returns:
        Trajectory dictionary compatible with GEN3C nodes
    """
    poses = result.poses  # (N, 4, 4)
    intrinsics = result.intrinsics  # (3, 3) or (N, 3, 3)

    # Handle single intrinsics matrix
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)

    # Determine frame dimensions
    if frame_size is not None:
        width, height = int(frame_size[0]), int(frame_size[1])
    else:
        width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT

    frames_data = []
    for i in range(poses.shape[0]):
        pose_matrix = poses[i].numpy().tolist()
        K = intrinsics[i].numpy()

        # Extract intrinsic parameters
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # Safely invert pose matrix
        world_to_camera = safe_matrix_inverse(poses[i].numpy()).tolist()

        frame_data = {
            "frame": i,
            "width": width,
            "height": height,
            "near": 0.01,
            "far": 1000.0,
            "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "extrinsics": {
                "camera_to_world": pose_matrix,
                "world_to_camera": world_to_camera,
            }
        }
        frames_data.append(frame_data)

    trajectory = {
        "fps": fps,
        "frames": frames_data,
        "handedness": "right",
        "source": source_name,
        "confidence": result.confidence,
    }

    return trajectory


def extract_frame_size_from_images(images: torch.Tensor) -> Optional[Tuple[int, int]]:
    """Extract frame dimensions from image tensor.

    Args:
        images: Image tensor in various formats

    Returns:
        (width, height) tuple or None if unable to determine
    """
    if images.numel() == 0:
        return None

    if images.ndim == 4:  # (F, H, W, C)
        return (int(images.shape[2]), int(images.shape[1]))
    elif images.ndim == 3:  # (H, W, C)
        return (int(images.shape[1]), int(images.shape[0]))

    return None


def extract_frame_size_from_path(image_path: Path) -> Tuple[int, int]:
    """Extract dimensions from an image file.

    Args:
        image_path: Path to image file

    Returns:
        (width, height) tuple, defaults if file cannot be read
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT


def update_trajectory_frame_sizes(
    trajectory: Dict[str, Any],
    width: int,
    height: int
) -> Dict[str, Any]:
    """Update all frame metadata with new dimensions.

    Args:
        trajectory: Trajectory dictionary to update
        width: New width in pixels
        height: New height in pixels

    Returns:
        Updated trajectory (modifies in place and returns)
    """
    frames = trajectory.get("frames", [])
    for frame_meta in frames:
        frame_meta["width"] = width
        frame_meta["height"] = height

    return trajectory
