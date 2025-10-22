"""ComfyUI nodes for pose and depth recovery from videos/images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from .pose_depth import estimate_from_video, estimate_from_images, RecoveryConfig
from .trajectory_utils import (
    pose_result_to_trajectory,
    extract_frame_size_from_images,
    extract_frame_size_from_path,
)
from ..utils import create_dummy_trajectory, validate_path_exists
from ..exceptions import Gen3CInvalidInputError, Gen3CPoseRecoveryError
from ..constants import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH

# Set up logging
logger = logging.getLogger(__name__)


class Gen3CPoseRecovery:
    """Unified pose and depth recovery node supporting both video files and image sequences.

    This node consolidates video and image-based pose recovery into a single interface,
    reducing complexity and improving user experience.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "source_type": (["video_file", "image_sequence"], {
                    "default": "video_file",
                    "tooltip": "Input source: 'video_file' for MP4/AVI files, 'image_sequence' for IMAGE tensor"
                }),
                "backend": (["auto", "colmap", "vipe"], {
                    "default": "auto",
                    "tooltip": "SfM backend: 'auto' tries ViPE→COLMAP, or select specific"
                }),
                "max_frames": ("INT", {
                    "default": 50,
                    "min": 2,
                    "max": 500,
                    "tooltip": "Maximum frames to use for pose recovery"
                }),
                "estimate_depth": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable depth map estimation during pose recovery"
                }),
                "downsample_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Downsample images for faster SfM (0.5 = half resolution)"
                }),
                "fps": ("INT", {
                    "default": DEFAULT_FPS,
                    "min": 1,
                    "max": 120,
                    "tooltip": "Output trajectory frames per second"
                }),
            },
            "optional": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Required for 'video_file' source: path to video file"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Required for 'image_sequence' source: image tensor (F, H, W, C)"
                }),
                "matcher_type": (["exhaustive", "sequential"], {
                    "default": "exhaustive",
                    "tooltip": "Feature matching: 'exhaustive' (accurate) or 'sequential' (fast)"
                }),
                "refinement_iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Bundle adjustment iterations for pose refinement"
                }),
            }
        }

    RETURN_TYPES = ("GEN3C_TRAJECTORY", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("trajectory", "images", "confidence", "status")
    FUNCTION = "recover_poses"
    CATEGORY = "GEN3C/Recovery"
    DESCRIPTION = "Unified pose recovery from video files or image sequences using COLMAP or ViPE. Automatically handles both input types with consistent output format."

    def _extract_frames_from_video(self, video_path: str, max_frames: int) -> torch.Tensor:
        """Extract frames from video file."""
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, returning dummy frames")
            return torch.zeros(max_frames, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return torch.zeros(max_frames, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frames.append(frame_tensor)
                saved_count += 1

            frame_idx += 1

        cap.release()

        if not frames:
            logger.warning("No frames extracted from video")
            return torch.zeros(max_frames, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

        logger.info(f"Extracted {len(frames)} frames from video")
        return torch.stack(frames)

    def _save_images_to_temp(self, images: torch.Tensor) -> List[Path]:
        """Save image tensor to temporary files for SfM processing."""
        import tempfile
        from PIL import Image

        temp_dir = Path(tempfile.mkdtemp())
        image_paths = []

        # Normalize tensor shape
        if images.ndim == 4:
            batch_size = images.shape[0]
        elif images.ndim == 3:
            images = images.unsqueeze(0)
            batch_size = 1
        else:
            raise Gen3CInvalidInputError(f"Invalid image tensor shape: {images.shape}")

        logger.info(f"Saving {batch_size} images to temporary directory: {temp_dir}")

        for i in range(batch_size):
            img_tensor = images[i]
            img_array = (img_tensor.clamp(0, 1) * 255).byte().cpu().numpy()
            img_pil = Image.fromarray(img_array)

            img_path = temp_dir / f"image_{i:06d}.png"
            img_pil.save(img_path)
            image_paths.append(img_path)

        return image_paths

    def recover_poses(
        self,
        source_type: str,
        backend: str,
        max_frames: int,
        estimate_depth: bool,
        downsample_factor: float,
        fps: int,
        video_path: str = "",
        images: Optional[torch.Tensor] = None,
        matcher_type: str = "exhaustive",
        refinement_iterations: int = 3,
    ) -> Tuple[Dict[str, Any], torch.Tensor, float, str]:
        """Recover camera poses from video file or image sequence.

        Args:
            source_type: "video_file" or "image_sequence"
            backend: SfM backend to use
            max_frames: Maximum number of frames to process
            estimate_depth: Whether to estimate depth maps
            downsample_factor: Image downsampling factor
            fps: Output trajectory FPS
            video_path: Path to video file (required if source_type="video_file")
            images: Image tensor (required if source_type="image_sequence")
            matcher_type: Feature matching strategy
            refinement_iterations: Bundle adjustment iterations

        Returns:
            Tuple of (trajectory, images, confidence, status)
        """
        logger.info(f"Starting pose recovery with source_type={source_type}, backend={backend}")

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
            if source_type == "video_file":
                # Validate video path
                try:
                    validate_path_exists(video_path, "Video file")
                except Gen3CInvalidInputError as e:
                    logger.error(f"Video path validation failed: {e}")
                    dummy_trajectory = create_dummy_trajectory(fps=fps, source="error")
                    dummy_images = torch.zeros(1, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
                    return (dummy_trajectory, dummy_images, 0.0, str(e))

                # Extract frames from video
                logger.info(f"Extracting frames from video: {video_path}")
                extracted_images = self._extract_frames_from_video(video_path, max_frames)
                frame_size = extract_frame_size_from_images(extracted_images)

                # Run pose recovery
                logger.info("Running pose recovery on video frames")
                result = estimate_from_video(Path(video_path), config)

                # Convert to trajectory
                trajectory = pose_result_to_trajectory(
                    result,
                    fps=fps,
                    source_name=f"pose_recovery_{Path(video_path).stem}",
                    frame_size=frame_size,
                )

                output_images = extracted_images

            else:  # image_sequence
                if images is None or images.numel() == 0:
                    logger.error("No input images provided for image_sequence mode")
                    dummy_trajectory = create_dummy_trajectory(fps=fps, source="no_input")
                    return (dummy_trajectory, torch.zeros(1, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), 0.0, "No input images")

                # Save images to temporary files
                logger.info("Saving images to temporary directory for SfM")
                image_paths = self._save_images_to_temp(images)

                if len(image_paths) < 2:
                    logger.error("Insufficient images for SfM (need at least 2)")
                    dummy_trajectory = create_dummy_trajectory(fps=fps, source="insufficient_images")
                    return (dummy_trajectory, images, 0.0, "Need at least 2 images for SfM")

                # Run pose recovery
                logger.info(f"Running pose recovery on {len(image_paths)} images")
                result = estimate_from_images(image_paths, config)

                # Get frame size
                frame_size = extract_frame_size_from_path(image_paths[0])

                # Convert to trajectory
                trajectory = pose_result_to_trajectory(
                    result,
                    fps=fps,
                    source_name="pose_recovery_images",
                    frame_size=frame_size,
                )

                output_images = images

                # Cleanup temp files
                import shutil
                logger.info(f"Cleaning up temporary directory: {image_paths[0].parent}")
                shutil.rmtree(image_paths[0].parent, ignore_errors=True)

            # Generate status message
            status = "Success" if result.confidence > 0.3 else f"Low confidence: {result.confidence:.2f}"
            if result.error_message:
                status = f"Partial failure: {result.error_message}"

            logger.info(f"Pose recovery completed: {status}, confidence={result.confidence:.3f}")
            return (trajectory, output_images, result.confidence, status)

        except Exception as e:
            logger.exception(f"Pose recovery failed: {e}")
            dummy_trajectory = create_dummy_trajectory(fps=fps, source="failed")
            dummy_images = torch.zeros(1, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
            return (dummy_trajectory, dummy_images, 0.0, f"Recovery failed: {str(e)}")


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


    def _extract_frames_as_images(self, video_path: str, max_frames: int) -> torch.Tensor:
        """Extract frames from video and return as ComfyUI IMAGE tensor."""
        try:
            import cv2
        except ImportError:
            # Return dummy images if OpenCV not available
            return torch.zeros(max_frames, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.zeros(max_frames, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

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
            return torch.zeros(max_frames, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

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
        fps_override: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, float, str]:

        fallback_fps = fps_override if fps_override is not None else DEFAULT_FPS

        # Validate video path
        try:
            validate_path_exists(video_path, "Video file")
        except Gen3CInvalidInputError as e:
            dummy_trajectory = create_dummy_trajectory(fps=fallback_fps, source="error")
            dummy_images = torch.zeros(1, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
            return (dummy_trajectory, dummy_images, 0.0, str(e))

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
            # Extract images (for dataset output)
            images = self._extract_frames_as_images(video_path, max_frames)

            # Infer frame dimensions using utility
            frame_size = extract_frame_size_from_images(images)

            # Run pose recovery
            result = estimate_from_video(Path(video_path), config)

            # Convert to trajectory format using shared utility
            trajectory = pose_result_to_trajectory(
                result,
                fps=fallback_fps,
                source_name=f"pose_recovery_{Path(video_path).stem}",
                frame_size=frame_size,
            )

            status = "Success" if result.confidence > 0.3 else f"Low confidence: {result.confidence:.2f}"
            if result.error_message:
                status = f"Partial failure: {result.error_message}"

            return (trajectory, images, result.confidence, status)

        except Exception as e:
            dummy_trajectory = create_dummy_trajectory(fps=fallback_fps, source="failed")
            dummy_images = torch.zeros(1, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
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


    def recover_poses(
        self,
        images: torch.Tensor,
        backend: str,
        estimate_depth: bool,
        downsample_factor: float,
        matcher_type: str = "exhaustive",
        refinement_iterations: int = 3,
        fps: int = DEFAULT_FPS,
    ) -> Tuple[Dict[str, Any], float, str]:

        if images.numel() == 0:
            dummy_trajectory = create_dummy_trajectory(fps=fps, source="no_input")
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
                dummy_trajectory = create_dummy_trajectory(fps=fps, source="insufficient_images")
                return (dummy_trajectory, 0.0, "Need at least 2 images for SfM")

            # Run pose recovery
            result = estimate_from_images(image_paths, config)

            # Get frame size from first image path
            frame_size = extract_frame_size_from_path(image_paths[0])

            # Convert to trajectory format using shared utility
            trajectory = pose_result_to_trajectory(
                result,
                fps=fps,
                source_name="pose_recovery_images",
                frame_size=frame_size,
            )

            status = "Success" if result.confidence > 0.3 else f"Low confidence: {result.confidence:.2f}"
            if result.error_message:
                status = f"Partial failure: {result.error_message}"

            # Cleanup temp files
            import shutil
            shutil.rmtree(image_paths[0].parent, ignore_errors=True)

            return (trajectory, result.confidence, status)

        except Exception as e:
            dummy_trajectory = create_dummy_trajectory(fps=fps, source="failed")
            return (dummy_trajectory, 0.0, f"Recovery failed: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "Gen3C_PoseRecovery": Gen3CPoseRecovery,  # New unified node
    "Gen3C_PoseDepth_FromVideo": Gen3CPoseDepthFromVideo,  # Legacy - for backward compatibility
    "Gen3C_PoseDepth_FromImages": Gen3CPoseDepthFromImages,  # Legacy - for backward compatibility
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen3C_PoseRecovery": "GEN3C Pose Recovery (Unified)",
    "Gen3C_PoseDepth_FromVideo": "GEN3C Pose Recovery (Video) [Legacy]",
    "Gen3C_PoseDepth_FromImages": "GEN3C Pose Recovery (Images) [Legacy]",
}
