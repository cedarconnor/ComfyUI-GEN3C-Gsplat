"""ComfyUI nodes for pose and depth recovery from videos/images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from .pose_depth import estimate_from_video, estimate_from_images, RecoveryConfig


class Gen3CPoseDepthFromVideo:
    """Recover camera poses and depth from video using SfM pipelines."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "tooltip": "Path to video file for camera pose extraction"}),
                "max_frames": ("INT", {"default": 50, "min": 2, "max": 500, "tooltip": "Maximum frames to extract (evenly sampled from video)"}),
                "backend": (["auto", "colmap", "vipe"], {"default": "auto", "tooltip": "SfM backend: 'auto' tries ViPE→COLMAP, or select specific"}),
                "estimate_depth": ("BOOLEAN", {"default": True, "tooltip": "Enable depth map estimation during pose recovery"}),
                "downsample_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1, "tooltip": "Downsample images for faster SfM (0.5 = half size)"}),
            },
            "optional": {
                "matcher_type": (["exhaustive", "sequential"], {"default": "exhaustive", "tooltip": "Feature matching: 'exhaustive' (accurate) or 'sequential' (fast)"}),
                "refinement_iterations": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Bundle adjustment iterations for pose refinement"}),
            }
        }

    RETURN_TYPES = ("GEN3C_TRAJECTORY", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("trajectory", "images", "confidence", "status")
    FUNCTION = "recover_poses"
    CATEGORY = "GEN3C/Recovery"
    DESCRIPTION = "Recover camera poses from video using COLMAP or ViPE structure-from-motion. Outputs trajectory compatible with splat trainers and quality validators. Requires COLMAP installed."

    def _result_to_trajectory(
        self,
        result,
        video_path: str,
        max_frames: int
    ) -> Dict[str, Any]:
        """Convert PoseDepthResult to GEN3C trajectory format."""
        poses = result.poses  # (N, 4, 4)
        intrinsics = result.intrinsics  # (3, 3) or (N, 3, 3)

        # Handle single intrinsics matrix
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)

        frames_data = []
        for i in range(poses.shape[0]):
            pose_matrix = poses[i].numpy().tolist()
            K = intrinsics[i].numpy()

            # Extract intrinsic parameters
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            # Assume default resolution if not available
            width, height = 1024, 576

            frame_data = {
                "frame": i,
                "width": width,
                "height": height,
                "near": 0.01,
                "far": 1000.0,
                "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "extrinsics": {
                    "camera_to_world": pose_matrix,
                    "world_to_camera": np.linalg.inv(poses[i].numpy()).tolist()
                }
            }
            frames_data.append(frame_data)

        trajectory = {
            "fps": 24,  # Default FPS
            "frames": frames_data,
            "handedness": "right",
            "source": f"pose_recovery_{Path(video_path).stem}",
            "confidence": result.confidence
        }

        return trajectory

    def _extract_frames_as_images(self, video_path: str, max_frames: int) -> torch.Tensor:
        """Extract frames from video and return as ComfyUI IMAGE tensor."""
        try:
            import cv2
        except ImportError:
            # Return dummy images if OpenCV not available
            return torch.zeros(max_frames, 576, 1024, 3)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.zeros(max_frames, 576, 1024, 3)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // max_frames)

        frames = []
        frame_idx = 0
        saved_count = 0

        while cap.isOpened() and saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                # Convert BGR to RGB and normalize to [0,1]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frames.append(frame_tensor)
                saved_count += 1

            frame_idx += 1

        cap.release()

        if not frames:
            return torch.zeros(max_frames, 576, 1024, 3)

        # Stack frames and ensure correct shape (F, H, W, C)
        frames_tensor = torch.stack(frames)
        return frames_tensor

    def recover_poses(
        self,
        video_path: str,
        max_frames: int,
        backend: str,
        estimate_depth: bool,
        downsample_factor: float,
        matcher_type: str = "exhaustive",
        refinement_iterations: int = 3,
    ) -> Tuple[Dict[str, Any], torch.Tensor, float, str]:

        if not video_path or not Path(video_path).exists():
            dummy_trajectory = {
                "fps": 24,
                "frames": [],
                "handedness": "right",
                "source": "dummy"
            }
            dummy_images = torch.zeros(1, 576, 1024, 3)
            return (dummy_trajectory, dummy_images, 0.0, "Video file not found")

        # Configure recovery
        config = RecoveryConfig(
            backend=backend,
            max_frames=max_frames,
            downsample_factor=downsample_factor,
            matcher_type=matcher_type,
            refinement_iterations=refinement_iterations,
            estimate_depth=estimate_depth
        )

        try:
            # Run pose recovery
            result = estimate_from_video(Path(video_path), config)

            # Convert to trajectory format
            trajectory = self._result_to_trajectory(result, video_path, max_frames)

            # Extract images for output
            images = self._extract_frames_as_images(video_path, max_frames)

            status = "Success" if result.confidence > 0.3 else f"Low confidence: {result.confidence:.2f}"
            if result.error_message:
                status = f"Partial failure: {result.error_message}"

            return (trajectory, images, result.confidence, status)

        except Exception as e:
            dummy_trajectory = {
                "fps": 24,
                "frames": [],
                "handedness": "right",
                "source": "failed"
            }
            dummy_images = torch.zeros(1, 576, 1024, 3)
            return (dummy_trajectory, dummy_images, 0.0, f"Recovery failed: {str(e)}")


class Gen3CPoseDepthFromImages:
    """Recover camera poses and depth from a sequence of images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Image sequence tensor (F, H, W, C) for pose recovery"}),
                "backend": (["auto", "colmap"], {"default": "colmap", "tooltip": "SfM backend for pose estimation (COLMAP recommended)"}),
                "estimate_depth": ("BOOLEAN", {"default": True, "tooltip": "Estimate depth maps from multi-view geometry"}),
                "downsample_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1, "tooltip": "Image downsampling for SfM (1.0 = no downsampling)"}),
            },
            "optional": {
                "matcher_type": (["exhaustive", "sequential"], {"default": "exhaustive", "tooltip": "Feature matching strategy"}),
                "refinement_iterations": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Bundle adjustment iterations"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Frames per second for output trajectory"}),
            }
        }

    RETURN_TYPES = ("GEN3C_TRAJECTORY", "FLOAT", "STRING")
    RETURN_NAMES = ("trajectory", "confidence", "status")
    FUNCTION = "recover_poses"
    CATEGORY = "GEN3C/Recovery"
    DESCRIPTION = "Recover camera poses from image sequence using COLMAP. Useful for converting existing image sets to splat-trainable datasets. Images saved to temporary directory during processing."

    def _images_to_paths(self, images: torch.Tensor) -> List[Path]:
        """Save images to temporary files for SfM processing."""
        import tempfile
        from PIL import Image

        temp_dir = Path(tempfile.mkdtemp())
        image_paths = []

        # Handle different tensor formats
        if images.ndim == 4:  # (B, H, W, C)
            batch_size = images.shape[0]
        elif images.ndim == 3:  # (H, W, C) - single image
            images = images.unsqueeze(0)
            batch_size = 1
        else:
            raise ValueError(f"Unexpected image tensor shape: {images.shape}")

        for i in range(batch_size):
            img_tensor = images[i]
            # Convert to PIL Image
            img_array = (img_tensor.clamp(0, 1) * 255).byte().cpu().numpy()
            img_pil = Image.fromarray(img_array)

            img_path = temp_dir / f"image_{i:06d}.png"
            img_pil.save(img_path)
            image_paths.append(img_path)

        return image_paths

    def _result_to_trajectory(
        self,
        result,
        image_paths: List[Path],
        fps: int
    ) -> Dict[str, Any]:
        """Convert PoseDepthResult to GEN3C trajectory format."""
        poses = result.poses  # (N, 4, 4)
        intrinsics = result.intrinsics  # (3, 3) or (N, 3, 3)

        # Handle single intrinsics matrix
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)

        frames_data = []
        for i in range(poses.shape[0]):
            pose_matrix = poses[i].numpy().tolist()
            K = intrinsics[i].numpy()

            # Extract intrinsic parameters
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            # Try to get image dimensions from first image
            if i < len(image_paths):
                try:
                    from PIL import Image
                    img = Image.open(image_paths[i])
                    width, height = img.size
                except:
                    width, height = 1024, 576
            else:
                width, height = 1024, 576

            frame_data = {
                "frame": i,
                "width": width,
                "height": height,
                "near": 0.01,
                "far": 1000.0,
                "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "extrinsics": {
                    "camera_to_world": pose_matrix,
                    "world_to_camera": np.linalg.inv(poses[i].numpy()).tolist()
                }
            }
            frames_data.append(frame_data)

        trajectory = {
            "fps": fps,
            "frames": frames_data,
            "handedness": "right",
            "source": "pose_recovery_images",
            "confidence": result.confidence
        }

        return trajectory

    def recover_poses(
        self,
        images: torch.Tensor,
        backend: str,
        estimate_depth: bool,
        downsample_factor: float,
        matcher_type: str = "exhaustive",
        refinement_iterations: int = 3,
        fps: int = 24,
    ) -> Tuple[Dict[str, Any], float, str]:

        if images.numel() == 0:
            dummy_trajectory = {
                "fps": fps,
                "frames": [],
                "handedness": "right",
                "source": "dummy"
            }
            return (dummy_trajectory, 0.0, "No input images")

        # Configure recovery
        config = RecoveryConfig(
            backend=backend,
            max_frames=images.shape[0] if images.ndim > 3 else 1,
            downsample_factor=downsample_factor,
            matcher_type=matcher_type,
            refinement_iterations=refinement_iterations,
            estimate_depth=estimate_depth
        )

        try:
            # Save images to temporary files
            image_paths = self._images_to_paths(images)

            if len(image_paths) < 2:
                return ({
                    "fps": fps,
                    "frames": [],
                    "handedness": "right",
                    "source": "insufficient_images"
                }, 0.0, "Need at least 2 images for SfM")

            # Run pose recovery
            result = estimate_from_images(image_paths, config)

            # Convert to trajectory format
            trajectory = self._result_to_trajectory(result, image_paths, fps)

            status = "Success" if result.confidence > 0.3 else f"Low confidence: {result.confidence:.2f}"
            if result.error_message:
                status = f"Partial failure: {result.error_message}"

            # Cleanup temp files
            import shutil
            shutil.rmtree(image_paths[0].parent, ignore_errors=True)

            return (trajectory, result.confidence, status)

        except Exception as e:
            dummy_trajectory = {
                "fps": fps,
                "frames": [],
                "handedness": "right",
                "source": "failed"
            }
            return (dummy_trajectory, 0.0, f"Recovery failed: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "Gen3C_PoseDepth_FromVideo": Gen3CPoseDepthFromVideo,
    "Gen3C_PoseDepth_FromImages": Gen3CPoseDepthFromImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen3C_PoseDepth_FromVideo": "GEN3C Pose Recovery (Video)",
    "Gen3C_PoseDepth_FromImages": "GEN3C Pose Recovery (Images)",
}