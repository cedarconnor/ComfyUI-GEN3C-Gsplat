"""Nodes that export GEN3C datasets for Gaussian Splat training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .dataset.writer import FramePayload, SUPPORTED_DEPTH_FORMATS, export_dataset


class CosmosGen3CInferExport:
    """Collects rendered frames/depth and writes Nerfstudio-style datasets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_sequence": ("IMAGE", {}),
                "trajectory": ("GEN3C_TRAJECTORY", {}),
                "output_dir": ("STRING", {"default": "${output_dir}/gen3c_dataset"}),
                "depth_sequence": ("IMAGE", {"optional": True}),
                "depth_format": (SUPPORTED_DEPTH_FORMATS, {"default": "npy"}),
            },
            "optional": {
                "metadata_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_dir",)
    FUNCTION = "export_dataset"
    CATEGORY = "GEN3C/Dataset"

    def _to_frame_list(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected tensor with shape (F,H,W,C); received {tuple(tensor.shape)}")
        if tensor.shape[-1] != 3:
            raise ValueError("IMAGE tensor must have 3 channels in the last dimension.")
        frames = tensor.detach().cpu()
        frames = frames.permute(0, 3, 1, 2)  # (F,C,H,W)
        return [frame.contiguous() for frame in frames]

    def _to_depth_list(self, tensor: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        if tensor is None:
            return []
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3:  # (F,H,W)
            depth_frames = tensor
        elif tensor.ndim == 4:
            depth_frames = tensor.squeeze(-1)
        else:
            raise ValueError(f"Depth tensor must have shape (F,H,W) or (F,H,W,1); received {tuple(tensor.shape)}")
        return [frame.detach().cpu() for frame in depth_frames]

    def export_dataset(
        self,
        image_sequence: torch.Tensor,
        trajectory: Dict[str, Any],
        output_dir: str,
        depth_sequence: Optional[torch.Tensor] = None,
        depth_format: str = "npy",
        metadata_json: str = "{}",
    ):
        frames_meta = trajectory.get("frames", [])
        if not frames_meta:
            raise ValueError("Trajectory payload is missing 'frames'.")

        images = self._to_frame_list(image_sequence)
        depths = self._to_depth_list(depth_sequence)
        if depths and len(depths) != len(images):
            raise ValueError("Depth sequence length does not match image sequence length.")

        if len(images) != len(frames_meta):
            raise ValueError(
                f"Frame count mismatch: {len(images)} images vs {len(frames_meta)} trajectory entries."
            )

        width = int(frames_meta[0].get("width", images[0].shape[-1]))
        height = int(frames_meta[0].get("height", images[0].shape[-2]))
        intrinsics_matrix = frames_meta[0].get("intrinsics")
        if intrinsics_matrix is None:
            raise ValueError("Trajectory frame missing intrinsics matrix.")
        intrinsics = {
            "fl_x": float(intrinsics_matrix[0][0]),
            "fl_y": float(intrinsics_matrix[1][1]),
            "cx": float(intrinsics_matrix[0][2]),
            "cy": float(intrinsics_matrix[1][2]),
        }

        fps = int(trajectory.get("fps", 24))
        try:
            extra_metadata = json.loads(metadata_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata_json is not valid JSON: {exc}")

        payloads: List[FramePayload] = []
        for idx, image in enumerate(images):
            depth_tensor = depths[idx] if depths else None
            payloads.append(FramePayload(rgb=image, depth=depth_tensor, metadata=frames_meta[idx]))

        output_path = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output"))).expanduser().resolve()
        dataset_dir = write_dataset(
            output_dir=output_path,
            frames=payloads,
            width=width,
            height=height,
            intrinsics=intrinsics,
            fps=fps,
            depth_format=depth_format,
        )

        if extra_metadata:
            transforms_path = dataset_dir / "transforms.json"
            transforms_payload = json.loads(transforms_path.read_text())
            transforms_payload.update(extra_metadata)
            transforms_path.write_text(json.dumps(transforms_payload, indent=2))

        return (str(dataset_dir),)


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "Cosmos_Gen3C_InferExport": CosmosGen3CInferExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cosmos_Gen3C_InferExport": "Cosmos GEN3C Export",
}
