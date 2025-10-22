"""Shared utility functions for GEN3C nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .constants import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH
from .exceptions import Gen3CInvalidInputError, Gen3CTrajectoryError


def validate_path_exists(path: str, description: str = "Path") -> Path:
    """Validate that a path exists and return resolved Path object.

    Args:
        path: Path string to validate
        description: Description of the path for error messages

    Returns:
        Resolved Path object

    Raises:
        Gen3CInvalidInputError: If path doesn't exist or is empty
    """
    if not path:
        raise Gen3CInvalidInputError(f"{description} is empty")

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise Gen3CInvalidInputError(f"{description} does not exist: {resolved}")

    return resolved


def validate_trajectory(trajectory: Dict[str, Any]) -> None:
    """Validate that trajectory has required fields.

    Args:
        trajectory: Trajectory dictionary to validate

    Raises:
        Gen3CTrajectoryError: If trajectory is missing required fields
    """
    if not trajectory:
        raise Gen3CTrajectoryError("Trajectory is empty or None")

    if "frames" not in trajectory:
        raise Gen3CTrajectoryError("Trajectory missing 'frames' field")

    frames = trajectory["frames"]
    if not frames:
        raise Gen3CTrajectoryError("Trajectory has no frames")

    # Validate first frame has required fields
    first_frame = frames[0]
    required_fields = ["intrinsics", "extrinsics", "width", "height"]
    missing = [f for f in required_fields if f not in first_frame]
    if missing:
        raise Gen3CTrajectoryError(f"Trajectory frame missing fields: {missing}")

    # Validate extrinsics structure
    extrinsics = first_frame.get("extrinsics", {})
    if "camera_to_world" not in extrinsics:
        raise Gen3CTrajectoryError("Trajectory frame missing camera_to_world transform")


def validate_frame_tensor(frames: torch.Tensor, name: str = "frames") -> torch.Tensor:
    """Normalize and validate frame tensor shape.

    Args:
        frames: Frame tensor to validate
        name: Name for error messages

    Returns:
        Normalized tensor in (F, H, W, C) format

    Raises:
        Gen3CInvalidInputError: If tensor has invalid shape
    """
    if frames.numel() == 0:
        raise Gen3CInvalidInputError(f"{name} tensor is empty")

    if frames.ndim == 5:  # (B, F, H, W, C)
        frames = frames.squeeze(0)

    if frames.ndim == 4:  # (F, H, W, C)
        return frames
    elif frames.ndim == 3:  # (H, W, C) - single frame
        return frames.unsqueeze(0)
    else:
        raise Gen3CInvalidInputError(
            f"{name} has invalid shape {tuple(frames.shape)}. Expected (F,H,W,C) or (H,W,C)"
        )


def extract_frame_dimensions(tensor: torch.Tensor) -> Tuple[int, int]:
    """Extract width and height from frame tensor.

    Args:
        tensor: Frame tensor (F, H, W, C) or (H, W, C)

    Returns:
        Tuple of (width, height)
    """
    if tensor.numel() == 0:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT

    if tensor.ndim == 4:  # (F, H, W, C)
        height, width = int(tensor.shape[1]), int(tensor.shape[2])
    elif tensor.ndim == 3:  # (H, W, C)
        height, width = int(tensor.shape[0]), int(tensor.shape[1])
    else:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT

    return width, height


def create_dummy_trajectory(
    fps: int = DEFAULT_FPS,
    source: str = "dummy",
    handedness: str = "right"
) -> Dict[str, Any]:
    """Create a dummy/empty trajectory for error cases.

    Args:
        fps: Frames per second
        source: Source identifier
        handedness: Coordinate system handedness

    Returns:
        Empty trajectory dictionary
    """
    return {
        "fps": fps,
        "frames": [],
        "handedness": handedness,
        "source": source,
    }


def safe_matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    """Safely invert a matrix with error handling.

    Args:
        matrix: Matrix to invert (must be square)

    Returns:
        Inverted matrix

    Raises:
        Gen3CInvalidInputError: If matrix is singular or invalid
    """
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        raise Gen3CInvalidInputError(
            f"Cannot invert matrix (may be singular): {e}"
        ) from e


def parse_json_safely(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    import json

    if not json_str or json_str.strip() == "":
        return default if default is not None else {}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return default if default is not None else {}


def resolve_output_path(path_str: str, default_subdir: str = "output") -> Path:
    """Resolve output path with ${output_dir} substitution.

    Args:
        path_str: Path string (may contain ${output_dir})
        default_subdir: Default subdirectory name

    Returns:
        Resolved Path object
    """
    resolved = path_str.replace(
        "${output_dir}",
        str(Path.cwd() / default_subdir)
    )
    return Path(resolved).expanduser().resolve()
