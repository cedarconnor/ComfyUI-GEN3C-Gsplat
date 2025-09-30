"""Pose and depth recovery fallback for GEN3C pipelines."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
import shutil
import os

import torch
import numpy as np
from PIL import Image


@dataclass
class PoseDepthResult:
    poses: torch.Tensor  # (N, 4, 4) camera-to-world transforms
    intrinsics: torch.Tensor  # (3, 3) or (N, 3, 3) camera intrinsics
    depths: Optional[torch.Tensor]  # (N, H, W) depth maps
    confidence: float  # 0-1 reconstruction confidence
    error_message: Optional[str] = None


@dataclass
class RecoveryConfig:
    """Configuration for pose/depth recovery backends."""
    backend: Literal["colmap", "vipe", "instant_ngp", "auto"] = "auto"
    max_frames: int = 100  # Limit frames for processing
    downsample_factor: float = 1.0  # Image resize factor
    feature_type: str = "sift"  # Feature detector type
    matcher_type: str = "exhaustive"  # Feature matching strategy
    min_track_length: int = 2  # Minimum track length for valid features
    refinement_iterations: int = 3  # Bundle adjustment iterations
    estimate_depth: bool = True  # Whether to estimate dense depth
    depth_method: str = "pmvs"  # Dense reconstruction method


def _extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    max_frames: int = 100,
    downsample_factor: float = 1.0
) -> List[Path]:
    """Extract frames from video for SfM processing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for video frame extraction. Install with: pip install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // max_frames)

    extracted_frames = []
    frame_idx = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Resize if needed
            if downsample_factor != 1.0:
                height, width = frame.shape[:2]
                new_height = int(height * downsample_factor)
                new_width = int(width * downsample_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB and save
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_path = output_dir / f"frame_{saved_count:06d}.png"
            Image.fromarray(frame_rgb).save(frame_path)
            extracted_frames.append(frame_path)
            saved_count += 1

        frame_idx += 1

    cap.release()
    return extracted_frames


def _run_colmap_sfm(
    images_dir: Path,
    workspace_dir: Path,
    config: RecoveryConfig
) -> Tuple[bool, str]:
    """Run COLMAP structure-from-motion pipeline."""
    try:
        # Check if COLMAP is available
        subprocess.run(["colmap", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, "COLMAP not found. Install COLMAP and ensure it's in your PATH."

    database_path = workspace_dir / "database.db"
    sparse_dir = workspace_dir / "sparse"
    dense_dir = workspace_dir / "dense"
    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)

    try:
        # Feature extraction
        cmd_extract = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "PINHOLE",
            "--SiftExtraction.use_gpu", "1" if torch.cuda.is_available() else "0"
        ]
        subprocess.run(cmd_extract, check=True, capture_output=True)

        # Feature matching
        cmd_match = [
            "colmap", "exhaustive_matcher" if config.matcher_type == "exhaustive" else "sequential_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "1" if torch.cuda.is_available() else "0"
        ]
        subprocess.run(cmd_match, check=True, capture_output=True)

        # Bundle adjustment
        cmd_mapper = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.ba_global_max_num_iterations", str(config.refinement_iterations)
        ]
        subprocess.run(cmd_mapper, check=True, capture_output=True)

        # Dense reconstruction (if requested)
        if config.estimate_depth:
            # Image undistortion
            cmd_undistort = [
                "colmap", "image_undistorter",
                "--image_path", str(images_dir),
                "--input_path", str(sparse_dir / "0"),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP"
            ]
            subprocess.run(cmd_undistort, check=True, capture_output=True)

            # Patch match stereo
            cmd_stereo = [
                "colmap", "patch_match_stereo",
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true"
            ]
            subprocess.run(cmd_stereo, check=True, capture_output=True)

        return True, "COLMAP reconstruction successful"

    except subprocess.CalledProcessError as e:
        error_msg = f"COLMAP failed: {e.stderr.decode() if e.stderr else str(e)}"
        return False, error_msg


def _parse_colmap_results(
    sparse_dir: Path,
    dense_dir: Optional[Path],
    frame_paths: List[Path]
) -> PoseDepthResult:
    """Parse COLMAP output into PoseDepthResult format."""
    cameras_file = sparse_dir / "0" / "cameras.txt"
    images_file = sparse_dir / "0" / "images.txt"

    if not cameras_file.exists() or not images_file.exists():
        raise FileNotFoundError("COLMAP reconstruction files not found")

    # Parse camera intrinsics
    intrinsics_dict = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                camera_id = int(parts[0])
                model = parts[1]
                width, height = int(parts[2]), int(parts[3])
                if model == "PINHOLE":
                    fx, fy, cx, cy = map(float, parts[4:8])
                    intrinsics_dict[camera_id] = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ])

    # Parse camera poses
    poses_dict = {}
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 10:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                image_name = parts[9]

                # Convert quaternion + translation to 4x4 matrix
                # COLMAP uses world-to-camera, we need camera-to-world
                from scipy.spatial.transform import Rotation

                # World-to-camera rotation and translation
                R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                t_w2c = np.array([tx, ty, tz])

                # Convert to camera-to-world
                R_c2w = R_w2c.T
                t_c2w = -R_c2w @ t_w2c

                pose_c2w = np.eye(4)
                pose_c2w[:3, :3] = R_c2w
                pose_c2w[:3, 3] = t_c2w

                poses_dict[image_name] = (pose_c2w, camera_id)

    # Match poses to input frames
    frame_names = [p.name for p in frame_paths]
    poses_list = []
    intrinsics_list = []

    for frame_name in frame_names:
        if frame_name in poses_dict:
            pose, camera_id = poses_dict[frame_name]
            poses_list.append(pose)
            intrinsics_list.append(intrinsics_dict.get(camera_id, np.eye(3)))
        else:
            # Create dummy pose if not found
            poses_list.append(np.eye(4))
            intrinsics_list.append(np.eye(3))

    poses = torch.tensor(np.stack(poses_list), dtype=torch.float32)

    # Use first camera intrinsics if all are similar
    if len(set(map(tuple, [K.flatten() for K in intrinsics_list]))) == 1:
        intrinsics = torch.tensor(intrinsics_list[0], dtype=torch.float32)
    else:
        intrinsics = torch.tensor(np.stack(intrinsics_list), dtype=torch.float32)

    # TODO: Parse depth maps from dense reconstruction
    depths = None
    if dense_dir and (dense_dir / "stereo" / "depth_maps").exists():
        # Implement depth map loading if needed
        pass

    confidence = len(poses_list) / len(frame_names)  # Fraction of successfully recovered poses

    return PoseDepthResult(
        poses=poses,
        intrinsics=intrinsics,
        depths=depths,
        confidence=confidence
    )


def _run_vipe_estimation(
    video_path: Path,
    config: RecoveryConfig
) -> PoseDepthResult:
    """Run ViPE (Video Pose Estimation) if available."""
    try:
        # Check if ViPE is available (placeholder - actual implementation depends on ViPE API)
        import vipe  # This would be the actual ViPE package

        # Run ViPE estimation
        result = vipe.estimate_poses_and_depth(
            video_path=str(video_path),
            max_frames=config.max_frames,
            estimate_depth=config.estimate_depth
        )

        return PoseDepthResult(
            poses=torch.tensor(result.poses, dtype=torch.float32),
            intrinsics=torch.tensor(result.intrinsics, dtype=torch.float32),
            depths=torch.tensor(result.depths, dtype=torch.float32) if result.depths is not None else None,
            confidence=result.confidence
        )

    except ImportError:
        raise ImportError("ViPE not available. Install ViPE or use COLMAP backend.")


def estimate_from_video(
    video_path: Path,
    config: Optional[RecoveryConfig] = None
) -> PoseDepthResult:
    """Estimate camera poses and depth from video using SfM pipelines.

    Supports multiple backends:
    - COLMAP: Classical SfM pipeline (most robust)
    - ViPE: Video-specific pose estimation (if available)
    - auto: Try backends in order of preference

    Args:
        video_path: Path to input video file
        config: Recovery configuration options

    Returns:
        PoseDepthResult with estimated poses, intrinsics, and optional depth

    Raises:
        RuntimeError: If all backends fail or are unavailable
    """
    if config is None:
        config = RecoveryConfig()

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    backends_to_try = []
    if config.backend == "auto":
        backends_to_try = ["vipe", "colmap"]
    else:
        backends_to_try = [config.backend]

    last_error = None

    for backend in backends_to_try:
        try:
            if backend == "vipe":
                return _run_vipe_estimation(video_path, config)

            elif backend == "colmap":
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    images_dir = temp_path / "images"
                    workspace_dir = temp_path / "workspace"
                    workspace_dir.mkdir(exist_ok=True)

                    # Extract frames
                    frame_paths = _extract_frames_from_video(
                        video_path, images_dir, config.max_frames, config.downsample_factor
                    )

                    if len(frame_paths) < 2:
                        raise ValueError("Need at least 2 frames for SfM reconstruction")

                    # Run COLMAP
                    success, message = _run_colmap_sfm(images_dir, workspace_dir, config)
                    if not success:
                        raise RuntimeError(message)

                    # Parse results
                    return _parse_colmap_results(
                        workspace_dir / "sparse",
                        workspace_dir / "dense" if config.estimate_depth else None,
                        frame_paths
                    )

            else:
                raise ValueError(f"Unknown backend: {backend}")

        except Exception as e:
            last_error = str(e)
            print(f"Backend {backend} failed: {e}")
            continue

    # All backends failed
    return PoseDepthResult(
        poses=torch.eye(4).unsqueeze(0),  # Dummy pose
        intrinsics=torch.eye(3),  # Dummy intrinsics
        depths=None,
        confidence=0.0,
        error_message=f"All pose recovery backends failed. Last error: {last_error}"
    )


def estimate_from_images(
    image_paths: List[Path],
    config: Optional[RecoveryConfig] = None
) -> PoseDepthResult:
    """Estimate poses from a sequence of images."""
    if config is None:
        config = RecoveryConfig()

    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images for SfM reconstruction")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images_dir = temp_path / "images"
        workspace_dir = temp_path / "workspace"
        images_dir.mkdir(exist_ok=True)
        workspace_dir.mkdir(exist_ok=True)

        # Copy images to temp directory with standard naming
        copied_paths = []
        for i, img_path in enumerate(image_paths[:config.max_frames]):
            dst_path = images_dir / f"frame_{i:06d}.png"

            # Load and potentially resize image
            img = Image.open(img_path)
            if config.downsample_factor != 1.0:
                new_size = (
                    int(img.width * config.downsample_factor),
                    int(img.height * config.downsample_factor)
                )
                img = img.resize(new_size, Image.LANCZOS)

            img.save(dst_path)
            copied_paths.append(dst_path)

        # Run COLMAP
        success, message = _run_colmap_sfm(images_dir, workspace_dir, config)
        if not success:
            return PoseDepthResult(
                poses=torch.eye(4).unsqueeze(0).repeat(len(image_paths), 1, 1),
                intrinsics=torch.eye(3),
                depths=None,
                confidence=0.0,
                error_message=message
            )

        return _parse_colmap_results(
            workspace_dir / "sparse",
            workspace_dir / "dense" if config.estimate_depth else None,
            copied_paths
        )


__all__ = ["PoseDepthResult", "RecoveryConfig", "estimate_from_video", "estimate_from_images"]
