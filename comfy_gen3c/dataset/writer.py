"""Dataset writer utilities for GEN3C Gaussian Splat workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


DepthFormat = str  # alias for clarity
SUPPORTED_DEPTH_FORMATS = ("npy", "png16", "pfm")


@dataclass
class FramePayload:
    rgb: torch.Tensor  # (C, H, W) in [0, 1]
    depth: Optional[torch.Tensor]  # (H, W) in world units
    metadata: Dict[str, object]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_rgb(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"RGB tensor must have shape (C,H,W); received {tuple(tensor.shape)}")
    return tensor.detach().cpu().clamp(0.0, 1.0)


def save_rgb_tensor(tensor: torch.Tensor, path: Path) -> None:
    rgb = _sanitize_rgb(tensor)
    array = (rgb.mul(255.0).round().byte().numpy().transpose(1, 2, 0))
    Image.fromarray(array).save(path)


def save_depth_tensor(depth: torch.Tensor, path: Path, fmt: DepthFormat) -> None:
    depth_np = depth.detach().cpu().float().numpy()
    if fmt == "npy":
        np.save(path, depth_np)
    elif fmt == "pfm":
        _write_pfm(path, depth_np)
    elif fmt == "png16":
        max_depth = float(np.percentile(depth_np, 99.0)) or 1.0
        scaled = np.clip(depth_np / max_depth, 0.0, 1.0)
        png_data = (scaled * 65535.0).round().astype(np.uint16)
        Image.fromarray(png_data).save(path)
    else:
        raise ValueError(f"Unsupported depth format '{fmt}'.")


def _write_pfm(path: Path, data: np.ndarray) -> None:
    data = np.flipud(data)
    colour = data.ndim == 3 and data.shape[2] == 3
    header = b"PF\n" if colour else b"Pf\n"
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(f"{data.shape[1]} {data.shape[0]}\n".encode("ascii"))
        scale = -1.0 if data.dtype.byteorder == '<' or (data.dtype.byteorder == '=' and np.little_endian) else 1.0
        fh.write(f"{scale}\n".encode("ascii"))
        fh.write(data.astype(np.float32).tobytes())


def write_transforms_json(
    root_dir: Path,
    frames: Sequence[FramePayload],
    intrinsics: Dict[str, float],
    width: int,
    height: int,
    depth_suffix: Optional[str],
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    serialized_frames: List[Dict[str, object]] = []
    for idx, frame in enumerate(frames):
        extrinsics = frame.metadata.get("extrinsics", {})
        camera_to_world = extrinsics.get("camera_to_world")
        if camera_to_world is None:
            raise ValueError("Frame metadata missing camera_to_world transform.")
        frame_payload = {
            "file_path": f"rgb/frame_{idx:06d}.png",
            "transform_matrix": camera_to_world,
        }
        if frame.depth is not None and depth_suffix is not None:
            frame_payload["depth_path"] = f"depth/frame_{idx:06d}.{depth_suffix}"
        serialized_frames.append(frame_payload)

    payload = {
        "camera_model": "OPENCV",
        "w": width,
        "h": height,
        **intrinsics,
        "frames": serialized_frames,
    }
    if extra_metadata:
        payload.update(extra_metadata)

    (root_dir / "transforms.json").write_text(json.dumps(payload, indent=2))


def export_dataset(
    output_dir: Path,
    frames: Sequence[FramePayload],
    width: int,
    height: int,
    intrinsics: Dict[str, float],
    fps: int,
    depth_format: DepthFormat = "npy",
) -> Path:
    if depth_format not in SUPPORTED_DEPTH_FORMATS:
        raise ValueError(f"Depth format '{depth_format}' is not supported. Choose from {SUPPORTED_DEPTH_FORMATS}.")

    _ensure_dir(output_dir)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    _ensure_dir(rgb_dir)
    _ensure_dir(depth_dir)

    for idx, frame in enumerate(frames):
        save_rgb_tensor(frame.rgb, rgb_dir / f"frame_{idx:06d}.png")
        if frame.depth is not None:
            suffix = "png" if depth_format == "png16" else depth_format
            save_depth_tensor(frame.depth, depth_dir / f"frame_{idx:06d}.{suffix}", depth_format)

    intrinsics_payload = {
        "fl_x": intrinsics["fl_x"],
        "fl_y": intrinsics["fl_y"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "fps": fps,
    }
    depth_suffix = "png" if depth_format == "png16" else depth_format
    write_transforms_json(
        root_dir=output_dir,
        frames=frames,
        intrinsics=intrinsics_payload,
        width=width,
        height=height,
        depth_suffix=depth_suffix if any(frame.depth is not None for frame in frames) else None,
    )
    return output_dir


__all__ = [
    "FramePayload",
    "SUPPORTED_DEPTH_FORMATS",
    "export_dataset",
]
