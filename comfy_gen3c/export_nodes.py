"""Dataset exporter that saves GEN3C RGB (+ optional depth) outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

DEPTH_SUFFIX = "npy"


class BaseDatasetExporter:
    """Shared functionality for dataset export nodes."""

    @staticmethod
    def _normalize_frames(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize frame tensor to (F, H, W, C) format."""
        if tensor.ndim == 5:
            return tensor.squeeze(0)
        if tensor.ndim == 4:
            return tensor
        raise ValueError(f"Expected tensor of shape (F,H,W,C) or (1,F,H,W,C); got {tuple(tensor.shape)}")

    @staticmethod
    def _write_rgb(frames: torch.Tensor, root: Path) -> None:
        """Write RGB frames to disk as PNG files."""
        rgb_dir = root / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            array = frame.clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
            Image.fromarray(array).save(rgb_dir / f"frame_{idx:06d}.png")

    @staticmethod
    def _write_depth(depth: torch.Tensor, root: Path) -> None:
        """Write depth maps to disk as NPY files."""
        depth_dir = root / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(depth):
            frame_np = frame.squeeze().cpu().float().numpy()
            np.save(depth_dir / f"frame_{idx:06d}.{DEPTH_SUFFIX}", frame_np)

    @staticmethod
    def _extract_intrinsics(frame: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Extract focal length and principal point from frame intrinsics."""
        intrinsics = frame.get("intrinsics")
        if not intrinsics:
            raise ValueError("Trajectory frames missing intrinsics matrix.")
        return (
            float(intrinsics[0][0]),  # fx
            float(intrinsics[1][1]),  # fy
            float(intrinsics[0][2]),  # cx
            float(intrinsics[1][2]),  # cy
        )

    @staticmethod
    def _build_transforms_dict(
        trajectory: Dict[str, Any],
        has_depth: bool,
        metadata_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build transforms.json dictionary from trajectory."""
        frames = trajectory.get("frames", [])
        if not frames:
            raise ValueError("Trajectory payload is missing frame data.")

        first_frame = frames[0]
        width = int(first_frame.get("width"))
        height = int(first_frame.get("height"))
        if width <= 0 or height <= 0:
            raise ValueError("Trajectory frames must include positive width/height values.")

        fx, fy, cx, cy = BaseDatasetExporter._extract_intrinsics(first_frame)
        fps = trajectory.get("fps", 24)
        handedness = trajectory.get("handedness")

        serialized_frames = []
        for idx, frame in enumerate(frames):
            extrinsics = frame.get("extrinsics", {})
            cam_to_world = extrinsics.get("camera_to_world")
            if cam_to_world is None:
                raise ValueError("Frame metadata missing camera_to_world transform.")

            frame_payload = {
                "file_path": f"rgb/frame_{idx:06d}.png",
                "transform_matrix": cam_to_world,
                "frame": frame.get("frame", idx),
            }

            world_to_camera = extrinsics.get("world_to_camera")
            if world_to_camera is not None:
                frame_payload["world_to_camera"] = world_to_camera

            if has_depth:
                frame_payload["depth_path"] = f"depth/frame_{idx:06d}.{DEPTH_SUFFIX}"

            serialized_frames.append(frame_payload)

        transforms = {
            "camera_model": "OPENCV",
            "w": width,
            "h": height,
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "fps": fps,
            "handedness": handedness,
            "frames": serialized_frames,
        }

        # Add optional near/far planes
        near_plane = first_frame.get("near")
        far_plane = first_frame.get("far")
        if near_plane is not None:
            transforms["near_plane"] = near_plane
        if far_plane is not None:
            transforms["far_plane"] = far_plane

        if metadata_override:
            transforms.update(metadata_override)

        return transforms

    @staticmethod
    def _write_transforms(
        trajectory: Dict[str, Any],
        root: Path,
        has_depth: bool,
        metadata_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write transforms.json to disk."""
        transforms = BaseDatasetExporter._build_transforms_dict(trajectory, has_depth, metadata_override)
        (root / "transforms.json").write_text(json.dumps(transforms, indent=2))

    @staticmethod
    def _prepare_depth(depth_maps: Optional[torch.Tensor], rgb_frames: torch.Tensor) -> Optional[torch.Tensor]:
        """Prepare depth maps, ensuring they match RGB frame count."""
        if depth_maps is None:
            return None

        depth_tensor = depth_maps
        if depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.unsqueeze(-1)

        depth_frames = BaseDatasetExporter._normalize_frames(depth_tensor)

        if depth_frames.shape[0] != rgb_frames.shape[0]:
            raise ValueError(f"depth_maps frame count ({depth_frames.shape[0]}) does not match RGB frames ({rgb_frames.shape[0]})")

        if depth_frames.shape[-1] == 1:
            depth_frames = depth_frames[..., 0]

        return depth_frames


class CosmosGen3CInferExport(BaseDatasetExporter):
    @classmethod
    def INPUT_TYPES(cls):  # pragma: no cover - UI definition
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Generated RGB video frames (F, H, W, C) from Cosmos/GEN3C inference"}),
                "trajectory": ("GEN3C_TRAJECTORY", {"tooltip": "Camera trajectory with intrinsics and extrinsics per frame"}),
                "output_dir": ("STRING", {"default": "${output_dir}/gen3c_dataset", "tooltip": "Directory path for saving RGB frames, depth maps, and transforms.json"}),
                "write_to_disk": ("BOOLEAN", {"default": True, "tooltip": "Enable to write dataset files to disk; disable to pass data directly to trainer"}),
            },
            "optional": {
                "depth_maps": ("IMAGE", {"tooltip": "Optional depth maps (F, H, W, 1) aligned with RGB frames"}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}", "tooltip": "Additional metadata as JSON string to merge into transforms.json"}),
            },
        }

    RETURN_TYPES = ("STRING", "GEN3C_DATASET")
    RETURN_NAMES = ("dataset_dir", "dataset")
    FUNCTION = "export_dataset"
    CATEGORY = "GEN3C/Dataset"
    DESCRIPTION = "Export Cosmos/GEN3C inference outputs to Nerfstudio-compatible dataset. Toggle 'write_to_disk' to control file writing vs direct data passing."

    def export_dataset(
        self,
        images: torch.Tensor,
        trajectory: Dict[str, Any],
        output_dir: str,
        write_to_disk: bool = True,
        depth_maps: Optional[torch.Tensor] = None,
        metadata_json: str = "{}",
    ) -> Tuple[str, Dict[str, Any]]:
        rgb_frames = self._normalize_frames(images)
        depth_frames = self._prepare_depth(depth_maps, rgb_frames)

        # Build in-memory dataset structure
        dataset_dict = {
            "rgb_frames": rgb_frames,
            "depth_frames": depth_frames,
            "trajectory": trajectory,
            "metadata": json.loads(metadata_json or "{}"),
        }

        # Optionally write to disk
        dataset_path_str = ""
        if write_to_disk:
            dataset_path = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output"))).expanduser().resolve()
            dataset_path.mkdir(parents=True, exist_ok=True)

            self._write_rgb(rgb_frames, dataset_path)
            if depth_frames is not None:
                self._write_depth(depth_frames, dataset_path)

            extra = json.loads(metadata_json or "{}")
            self._write_transforms(trajectory, dataset_path, depth_frames is not None, extra)
            dataset_path_str = str(dataset_path)

        return (dataset_path_str, dataset_dict)


class CosmosGen3CDirectExport(BaseDatasetExporter):
    """Enhanced exporter that extracts trajectory from Cosmos latents directly."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Decoded RGB frames from Cosmos VAE (F, H, W, C)"}),
                "latents": ("LATENT", {"tooltip": "Cosmos latent dict with embedded camera trajectory"}),
                "output_dir": ("STRING", {"default": "${output_dir}/gen3c_dataset", "tooltip": "Output directory for dataset files"}),
                "write_to_disk": ("BOOLEAN", {"default": True, "tooltip": "Enable to write files to disk; disable to pass data directly to trainer"}),
            },
            "optional": {
                "depth_maps": ("IMAGE", {"tooltip": "Optional depth maps (F, H, W, 1) from DepthCrafter or other estimator"}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}", "tooltip": "Additional JSON metadata to include in transforms.json"}),
                "trajectory_override": ("GEN3C_TRAJECTORY", {"tooltip": "Override trajectory instead of extracting from latents"}),
            },
        }

    RETURN_TYPES = ("STRING", "GEN3C_TRAJECTORY", "GEN3C_DATASET")
    RETURN_NAMES = ("dataset_dir", "trajectory", "dataset")
    FUNCTION = "export_dataset"
    CATEGORY = "GEN3C/Dataset"
    DESCRIPTION = "Export Cosmos outputs with automatic trajectory extraction from latents. Supports both disk-based and memory-based workflows."

    @staticmethod
    def _extract_trajectory_from_latents(latents: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract embedded trajectory from latent dict."""
        trajectory = latents.get("camera_trajectory")
        if trajectory is None:
            trajectory = latents.get("trajectory")
        return trajectory

    def export_dataset(
        self,
        images: torch.Tensor,
        latents: Dict[str, Any],
        output_dir: str,
        write_to_disk: bool = True,
        depth_maps: Optional[torch.Tensor] = None,
        metadata_json: str = "{}",
        trajectory_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        # Extract trajectory from latents or use override
        trajectory = trajectory_override or self._extract_trajectory_from_latents(latents)
        if trajectory is None:
            raise ValueError("No trajectory found in latents and no override provided. Use Gen3C_CameraTrajectory upstream.")

        rgb_frames = self._normalize_frames(images)
        depth_frames = self._prepare_depth(depth_maps, rgb_frames)

        # Build in-memory dataset structure
        dataset_dict = {
            "rgb_frames": rgb_frames,
            "depth_frames": depth_frames,
            "trajectory": trajectory,
            "metadata": json.loads(metadata_json or "{}"),
        }

        # Optionally write to disk
        dataset_path_str = ""
        if write_to_disk:
            dataset_path = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output"))).expanduser().resolve()
            dataset_path.mkdir(parents=True, exist_ok=True)

            self._write_rgb(rgb_frames, dataset_path)
            if depth_frames is not None:
                self._write_depth(depth_frames, dataset_path)

            extra = json.loads(metadata_json or "{}")
            self._write_transforms(trajectory, dataset_path, depth_frames is not None, extra)
            dataset_path_str = str(dataset_path)

        return (dataset_path_str, trajectory, dataset_dict)


class Gen3CVideoToDataset(BaseDatasetExporter):
    """Convert video to dataset using pose recovery and optional depth estimation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "tooltip": "Path to input video file for pose recovery"}),
                "output_dir": ("STRING", {"default": "${output_dir}/video_dataset", "tooltip": "Output directory for extracted dataset"}),
                "max_frames": ("INT", {"default": 50, "min": 2, "max": 500, "tooltip": "Maximum number of frames to extract from video"}),
                "backend": (["auto", "colmap", "vipe"], {"default": "auto", "tooltip": "SfM backend: 'auto' tries ViPE then COLMAP, or choose specific"}),
                "write_to_disk": ("BOOLEAN", {"default": True, "tooltip": "Enable to write frames/transforms to disk; disable to pass data directly"}),
            },
            "optional": {
                "depth_maps": ("IMAGE", {"tooltip": "Optional pre-computed depth maps (F, H, W, 1) to override pose recovery depth"}),
                "estimate_depth": ("BOOLEAN", {"default": True, "tooltip": "Enable depth estimation during pose recovery"}),
                "downsample_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1, "tooltip": "Downsample factor for faster SfM (0.5 = half resolution)"}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}", "tooltip": "Additional metadata as JSON"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Target frames per second for trajectory"}),
            }
        }

    RETURN_TYPES = ("STRING", "GEN3C_TRAJECTORY", "FLOAT", "STRING", "GEN3C_DATASET")
    RETURN_NAMES = ("dataset_dir", "trajectory", "confidence", "status", "dataset")
    FUNCTION = "convert_video"
    CATEGORY = "GEN3C/Dataset"
    DESCRIPTION = "Extract frames from video and recover camera poses using COLMAP or ViPE. Outputs both disk-based and in-memory dataset."

    def convert_video(
        self,
        video_path: str,
        output_dir: str,
        max_frames: int,
        backend: str,
        write_to_disk: bool = True,
        depth_maps: Optional[torch.Tensor] = None,
        estimate_depth: bool = True,
        downsample_factor: float = 0.5,
        metadata_json: str = "{}",
        fps: int = 24,
    ) -> tuple[str, Dict[str, Any], float, str, Dict[str, Any]]:

        if not video_path or not Path(video_path).exists():
            dummy_trajectory = {"fps": fps, "frames": [], "handedness": "right"}
            dummy_dataset = {"rgb_frames": None, "depth_frames": None, "trajectory": dummy_trajectory, "metadata": {}}
            return ("", dummy_trajectory, 0.0, "Video file not found", dummy_dataset)

        # Import recovery node functionality
        from .dataset.recovery_nodes import Gen3CPoseDepthFromVideo

        recovery_node = Gen3CPoseDepthFromVideo()

        # Run pose recovery
        trajectory, images, confidence, status = recovery_node.recover_poses(
            video_path=video_path,
            max_frames=max_frames,
            backend=backend,
            estimate_depth=estimate_depth,
            downsample_factor=downsample_factor,
            fps_override=fps,
        )

        if confidence < 0.1:
            dummy_dataset = {"rgb_frames": None, "depth_frames": None, "trajectory": trajectory, "metadata": {}}
            return ("", trajectory, confidence, f"Pose recovery failed: {status}", dummy_dataset)

        try:
            if images.numel() == 0:
                dummy_dataset = {"rgb_frames": None, "depth_frames": None, "trajectory": trajectory, "metadata": {}}
                return ("", trajectory, confidence, "No frames extracted from video", dummy_dataset)

            rgb_frames = self._normalize_frames(images)

            if trajectory.get("frames"):
                frame_height = int(rgb_frames.shape[1])
                frame_width = int(rgb_frames.shape[2])
                for frame_meta in trajectory["frames"]:
                    frame_meta["height"] = frame_height
                    frame_meta["width"] = frame_width
            trajectory["fps"] = fps

            # Prepare depth maps with error handling
            try:
                depth_frames = self._prepare_depth(depth_maps, rgb_frames)
            except ValueError as e:
                print(f"Warning: {e}, skipping depth")
                depth_frames = None

            # Build in-memory dataset structure
            dataset_dict = {
                "rgb_frames": rgb_frames,
                "depth_frames": depth_frames,
                "trajectory": trajectory,
                "metadata": json.loads(metadata_json or "{}"),
            }

            # Optionally write to disk
            dataset_path_str = ""
            if write_to_disk:
                dataset_path = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output"))).expanduser().resolve()
                dataset_path.mkdir(parents=True, exist_ok=True)

                self._write_rgb(rgb_frames, dataset_path)
                if depth_frames is not None:
                    self._write_depth(depth_frames, dataset_path)

                # Add recovery metadata to transforms
                metadata_dict = json.loads(metadata_json or "{}")
                if "confidence" in trajectory:
                    metadata_dict["recovery_confidence"] = trajectory["confidence"]
                if "source" in trajectory:
                    metadata_dict["source"] = trajectory["source"]

                self._write_transforms(trajectory, dataset_path, depth_frames is not None, metadata_dict)
                dataset_path_str = str(dataset_path)

            return (dataset_path_str, trajectory, confidence, f"Dataset created successfully. {status}", dataset_dict)

        except Exception as e:
            dummy_dataset = {"rgb_frames": None, "depth_frames": None, "trajectory": trajectory, "metadata": {}}
            return ("", trajectory, confidence, f"Export failed: {str(e)}", dummy_dataset)


NODE_CLASS_MAPPINGS = {
    "Cosmos_Gen3C_InferExport": CosmosGen3CInferExport,
    "Cosmos_Gen3C_DirectExport": CosmosGen3CDirectExport,
    "Gen3C_VideoToDataset": Gen3CVideoToDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cosmos_Gen3C_InferExport": "Cosmos GEN3C Export",
    "Cosmos_Gen3C_DirectExport": "Cosmos GEN3C Direct Export",
    "Gen3C_VideoToDataset": "GEN3C Video to Dataset",
}
