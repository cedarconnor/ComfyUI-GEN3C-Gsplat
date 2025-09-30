"""Dataset exporter that saves GEN3C RGB (+ optional depth) outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

DEPTH_SUFFIX = "npy"


class CosmosGen3CInferExport:
    @classmethod
    def INPUT_TYPES(cls):  # pragma: no cover - UI definition
        return {
            "required": {
                "images": ("IMAGE", {}),
                "trajectory": ("GEN3C_TRAJECTORY", {}),
                "output_dir": ("STRING", {"default": "${output_dir}/gen3c_dataset"}),
            },
            "optional": {
                "depth_maps": ("IMAGE", {}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_dir",)
    FUNCTION = "export_dataset"
    CATEGORY = "GEN3C/Dataset"

    def _normalize_frames(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 5:
            return tensor.squeeze(0)
        if tensor.ndim == 4:
            return tensor
        raise ValueError(f"Expected tensor of shape (F,H,W,C) or (1,F,H,W,C); got {tuple(tensor.shape)}")

    def _write_rgb(self, frames: torch.Tensor, root: Path) -> None:
        rgb_dir = root / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            array = (frame.clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy())
            Image.fromarray(array).save(rgb_dir / f"frame_{idx:06d}.png")

    def _write_depth(self, depth: torch.Tensor, root: Path) -> None:
        depth_dir = root / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(depth):
            frame_np = frame.squeeze().cpu().float().numpy()
            np.save(depth_dir / f"frame_{idx:06d}.{DEPTH_SUFFIX}", frame_np)

    def _write_transforms(
        self,
        trajectory: Dict[str, Any],
        root: Path,
        has_depth: bool,
        metadata_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        frames = trajectory.get("frames", [])
        if not frames:
            raise ValueError("Trajectory payload is missing frame data.")

        first_frame = frames[0]
        width = int(first_frame.get("width"))
        height = int(first_frame.get("height"))
        if width <= 0 or height <= 0:
            raise ValueError("Trajectory frames must include positive width/height values.")

        intrinsics = first_frame.get("intrinsics")
        if not intrinsics:
            raise ValueError("Trajectory frames missing intrinsics matrix.")
        fx = float(intrinsics[0][0])
        fy = float(intrinsics[1][1])
        cx = float(intrinsics[0][2])
        cy = float(intrinsics[1][2])

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

        near_plane = first_frame.get("near")
        far_plane = first_frame.get("far")
        if near_plane is not None:
            transforms["near_plane"] = near_plane
        if far_plane is not None:
            transforms["far_plane"] = far_plane

        if metadata_override:
            transforms.update(metadata_override)

        (root / "transforms.json").write_text(json.dumps(transforms, indent=2))

    def export_dataset(
        self,
        images: torch.Tensor,
        trajectory: Dict[str, Any],
        output_dir: str,
        depth_maps: Optional[torch.Tensor] = None,
        metadata_json: str = "{}",
    ) -> tuple[str]:
        dataset_path = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output"))).expanduser().resolve()
        dataset_path.mkdir(parents=True, exist_ok=True)

        rgb_frames = self._normalize_frames(images)
        self._write_rgb(rgb_frames, dataset_path)

        has_depth = depth_maps is not None
        if has_depth:
            depth_tensor = depth_maps
            if depth_tensor.ndim == 3:
                depth_tensor = depth_tensor.unsqueeze(-1)
            depth_frames = self._normalize_frames(depth_tensor)
            if depth_frames.shape[0] != rgb_frames.shape[0]:
                raise ValueError("depth_maps frame count does not match RGB frames")
            if depth_frames.shape[-1] == 1:
                depth_frames = depth_frames[..., 0]
            self._write_depth(depth_frames, dataset_path)

        extra = json.loads(metadata_json or "{}")
        self._write_transforms(trajectory, dataset_path, has_depth, extra)
        return (str(dataset_path),)


class CosmosGen3CDirectExport:
    """Enhanced exporter that extracts trajectory from Cosmos latents directly."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "latents": ("LATENT", {}),  # Expects latents with embedded trajectory
                "output_dir": ("STRING", {"default": "${output_dir}/gen3c_dataset"}),
            },
            "optional": {
                "depth_maps": ("IMAGE", {}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}"}),
                "trajectory_override": ("GEN3C_TRAJECTORY", {}),  # Optional trajectory override
            },
        }

    RETURN_TYPES = ("STRING", "GEN3C_TRAJECTORY")
    RETURN_NAMES = ("dataset_dir", "trajectory")
    FUNCTION = "export_dataset"
    CATEGORY = "GEN3C/Dataset"

    def _extract_trajectory_from_latents(self, latents: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract embedded trajectory from latent dict."""
        return latents.get("camera_trajectory")

    def _normalize_frames(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 5:
            return tensor.squeeze(0)
        if tensor.ndim == 4:
            return tensor
        raise ValueError(f"Expected tensor of shape (F,H,W,C) or (1,F,H,W,C); got {tuple(tensor.shape)}")

    def _write_rgb(self, frames: torch.Tensor, root: Path) -> None:
        rgb_dir = root / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            array = (frame.clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy())
            Image.fromarray(array).save(rgb_dir / f"frame_{idx:06d}.png")

    def _write_depth(self, depth: torch.Tensor, root: Path) -> None:
        depth_dir = root / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(depth):
            frame_np = frame.squeeze().cpu().float().numpy()
            np.save(depth_dir / f"frame_{idx:06d}.{DEPTH_SUFFIX}", frame_np)

    def _write_transforms(
        self,
        trajectory: Dict[str, Any],
        root: Path,
        has_depth: bool,
        metadata_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        frames = trajectory.get("frames", [])
        if not frames:
            raise ValueError("Trajectory payload is missing frame data.")

        first_frame = frames[0]
        width = int(first_frame.get("width"))
        height = int(first_frame.get("height"))
        if width <= 0 or height <= 0:
            raise ValueError("Trajectory frames must include positive width/height values.")

        intrinsics = first_frame.get("intrinsics")
        if not intrinsics:
            raise ValueError("Trajectory frames missing intrinsics matrix.")
        fx = float(intrinsics[0][0])
        fy = float(intrinsics[1][1])
        cx = float(intrinsics[0][2])
        cy = float(intrinsics[1][2])

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

        near_plane = first_frame.get("near")
        far_plane = first_frame.get("far")
        if near_plane is not None:
            transforms["near_plane"] = near_plane
        if far_plane is not None:
            transforms["far_plane"] = far_plane

        if metadata_override:
            transforms.update(metadata_override)

        (root / "transforms.json").write_text(json.dumps(transforms, indent=2))

    def export_dataset(
        self,
        images: torch.Tensor,
        latents: Dict[str, Any],
        output_dir: str,
        depth_maps: Optional[torch.Tensor] = None,
        metadata_json: str = "{}",
        trajectory_override: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        # Extract trajectory from latents or use override
        trajectory = trajectory_override or self._extract_trajectory_from_latents(latents)
        if trajectory is None:
            raise ValueError("No trajectory found in latents and no override provided. Use Gen3C_CameraTrajectory upstream.")

        dataset_path = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output"))).expanduser().resolve()
        dataset_path.mkdir(parents=True, exist_ok=True)

        rgb_frames = self._normalize_frames(images)
        self._write_rgb(rgb_frames, dataset_path)

        has_depth = depth_maps is not None
        if has_depth:
            depth_tensor = depth_maps
            if depth_tensor.ndim == 3:
                depth_tensor = depth_tensor.unsqueeze(-1)
            depth_frames = self._normalize_frames(depth_tensor)
            if depth_frames.shape[0] != rgb_frames.shape[0]:
                raise ValueError("depth_maps frame count does not match RGB frames")
            if depth_frames.shape[-1] == 1:
                depth_frames = depth_frames[..., 0]
            self._write_depth(depth_frames, dataset_path)

        extra = json.loads(metadata_json or "{}")
        self._write_transforms(trajectory, dataset_path, has_depth, extra)
        return (str(dataset_path), trajectory)


class Gen3CVideoToDataset:
    """Convert video to dataset using pose recovery and optional depth estimation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": "${output_dir}/video_dataset"}),
                "max_frames": ("INT", {"default": 50, "min": 2, "max": 500}),
                "backend": (["auto", "colmap", "vipe"], {"default": "auto"}),
            },
            "optional": {
                "depth_maps": ("IMAGE", {}),  # Optional external depth maps
                "estimate_depth": ("BOOLEAN", {"default": True}),
                "downsample_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            }
        }

    RETURN_TYPES = ("STRING", "GEN3C_TRAJECTORY", "FLOAT", "STRING")
    RETURN_NAMES = ("dataset_dir", "trajectory", "confidence", "status")
    FUNCTION = "convert_video"
    CATEGORY = "GEN3C/Dataset"

    def _normalize_frames(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 5:
            return tensor.squeeze(0)
        if tensor.ndim == 4:
            return tensor
        raise ValueError(f"Expected tensor of shape (F,H,W,C) or (1,F,H,W,C); got {tuple(tensor.shape)}")

    def _write_rgb(self, frames: torch.Tensor, root: Path) -> None:
        rgb_dir = root / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            array = (frame.clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy())
            Image.fromarray(array).save(rgb_dir / f"frame_{idx:06d}.png")

    def _write_depth(self, depth: torch.Tensor, root: Path) -> None:
        depth_dir = root / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(depth):
            frame_np = frame.squeeze().cpu().float().numpy()
            np.save(depth_dir / f"frame_{idx:06d}.npy", frame_np)

    def _write_transforms(
        self,
        trajectory: Dict[str, Any],
        root: Path,
        has_depth: bool,
        metadata_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        frames = trajectory.get("frames", [])
        if not frames:
            raise ValueError("Trajectory payload is missing frame data.")

        first_frame = frames[0]
        width = int(first_frame.get("width"))
        height = int(first_frame.get("height"))
        if width <= 0 or height <= 0:
            raise ValueError("Trajectory frames must include positive width/height values.")

        intrinsics = first_frame.get("intrinsics")
        if not intrinsics:
            raise ValueError("Trajectory frames missing intrinsics matrix.")
        fx = float(intrinsics[0][0])
        fy = float(intrinsics[1][1])
        cx = float(intrinsics[0][2])
        cy = float(intrinsics[1][2])

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
                frame_payload["depth_path"] = f"depth/frame_{idx:06d}.npy"
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

        # Add recovery metadata
        if "confidence" in trajectory:
            transforms["recovery_confidence"] = trajectory["confidence"]
        if "source" in trajectory:
            transforms["source"] = trajectory["source"]

        near_plane = first_frame.get("near")
        far_plane = first_frame.get("far")
        if near_plane is not None:
            transforms["near_plane"] = near_plane
        if far_plane is not None:
            transforms["far_plane"] = far_plane

        if metadata_override:
            transforms.update(metadata_override)

        (root / "transforms.json").write_text(json.dumps(transforms, indent=2))

    def convert_video(
        self,
        video_path: str,
        output_dir: str,
        max_frames: int,
        backend: str,
        depth_maps: Optional[torch.Tensor] = None,
        estimate_depth: bool = True,
        downsample_factor: float = 0.5,
        metadata_json: str = "{}",
        fps: int = 24,
    ) -> tuple[str, Dict[str, Any], float, str]:

        if not video_path or not Path(video_path).exists():
            dummy_trajectory = {"fps": fps, "frames": [], "handedness": "right"}
            return ("", dummy_trajectory, 0.0, "Video file not found")

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
        )

        if confidence < 0.1:
            return ("", trajectory, confidence, f"Pose recovery failed: {status}")

        # Set up output directory
        dataset_path = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output"))).expanduser().resolve()
        dataset_path.mkdir(parents=True, exist_ok=True)

        try:
            # Write RGB frames
            if images.numel() > 0:
                rgb_frames = self._normalize_frames(images)
                self._write_rgb(rgb_frames, dataset_path)
            else:
                return ("", trajectory, confidence, "No frames extracted from video")

            # Handle depth maps
            has_depth = depth_maps is not None
            if has_depth:
                depth_tensor = depth_maps
                if depth_tensor.ndim == 3:
                    depth_tensor = depth_tensor.unsqueeze(-1)
                depth_frames = self._normalize_frames(depth_tensor)
                if depth_frames.shape[0] != rgb_frames.shape[0]:
                    # If depth frame count doesn't match, skip depth
                    has_depth = False
                    print(f"Warning: Depth frame count ({depth_frames.shape[0]}) doesn't match RGB ({rgb_frames.shape[0]}), skipping depth")
                else:
                    if depth_frames.shape[-1] == 1:
                        depth_frames = depth_frames[..., 0]
                    self._write_depth(depth_frames, dataset_path)

            # Write transforms.json
            extra = json.loads(metadata_json or "{}")
            self._write_transforms(trajectory, dataset_path, has_depth, extra)

            return (str(dataset_path), trajectory, confidence, f"Dataset created successfully. {status}")

        except Exception as e:
            return ("", trajectory, confidence, f"Export failed: {str(e)}")


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
