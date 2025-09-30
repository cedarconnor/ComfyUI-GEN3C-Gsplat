"""Quality filters and metrics for GEN3C datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image, ImageStat


@dataclass
class FrameQuality:
    """Quality metrics for a single frame."""
    frame_index: int
    blur_score: float  # 0-1, higher is sharper
    brightness_score: float  # 0-1, optimal around 0.5
    contrast_score: float  # 0-1, higher is better
    color_diversity: float  # 0-1, higher is better
    motion_blur: float  # 0-1, lower is better
    overall_score: float  # 0-1, combined quality score
    issues: List[str]  # Specific quality issues


@dataclass
class TrajectoryQuality:
    """Quality metrics for camera trajectory."""
    smoothness_score: float  # 0-1, higher is smoother
    coverage_score: float  # 0-1, higher is better coverage
    baseline_score: float  # 0-1, optimal stereo baselines
    rotation_diversity: float  # 0-1, good viewing angle diversity
    speed_consistency: float  # 0-1, consistent motion speed
    overall_score: float  # 0-1, combined trajectory quality
    issues: List[str]  # Trajectory-specific issues


class QualityFilter:
    """Filter and assess quality of GEN3C datasets."""

    def __init__(
        self,
        min_blur_threshold: float = 0.3,
        min_brightness_threshold: float = 0.15,
        max_brightness_threshold: float = 0.85,
        min_contrast_threshold: float = 0.2,
        min_overall_threshold: float = 0.4
    ):
        self.min_blur_threshold = min_blur_threshold
        self.min_brightness_threshold = min_brightness_threshold
        self.max_brightness_threshold = max_brightness_threshold
        self.min_contrast_threshold = min_contrast_threshold
        self.min_overall_threshold = min_overall_threshold

    def assess_frame_quality(self, image: Image.Image, frame_index: int) -> FrameQuality:
        """Assess quality of a single frame."""
        issues = []

        # Convert to arrays for analysis
        img_array = np.array(image.convert('RGB'))
        gray_array = np.array(image.convert('L'))

        # Blur detection using Laplacian variance
        blur_score = self._calculate_blur_score(gray_array)
        if blur_score < self.min_blur_threshold:
            issues.append(f"Frame {frame_index}: Low sharpness (blur detected)")

        # Brightness analysis
        brightness_score = self._calculate_brightness_score(gray_array)
        if brightness_score < self.min_brightness_threshold:
            issues.append(f"Frame {frame_index}: Too dark")
        elif brightness_score > self.max_brightness_threshold:
            issues.append(f"Frame {frame_index}: Too bright")

        # Contrast analysis
        contrast_score = self._calculate_contrast_score(gray_array)
        if contrast_score < self.min_contrast_threshold:
            issues.append(f"Frame {frame_index}: Low contrast")

        # Color diversity
        color_diversity = self._calculate_color_diversity(img_array)

        # Motion blur detection (simplified)
        motion_blur = self._detect_motion_blur(gray_array)

        # Calculate overall score
        overall_score = self._calculate_frame_overall_score(
            blur_score, brightness_score, contrast_score, color_diversity, motion_blur
        )

        return FrameQuality(
            frame_index=frame_index,
            blur_score=blur_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            color_diversity=color_diversity,
            motion_blur=motion_blur,
            overall_score=overall_score,
            issues=issues
        )

    def assess_trajectory_quality(self, trajectory: Dict[str, Any]) -> TrajectoryQuality:
        """Assess quality of camera trajectory."""
        issues = []
        frames = trajectory.get("frames", [])

        if len(frames) < 2:
            return TrajectoryQuality(
                smoothness_score=0.0,
                coverage_score=0.0,
                baseline_score=0.0,
                rotation_diversity=0.0,
                speed_consistency=0.0,
                overall_score=0.0,
                issues=["Insufficient frames for trajectory analysis"]
            )

        # Extract poses
        poses = []
        for frame in frames:
            transform = frame.get("extrinsics", {}).get("camera_to_world")
            if transform:
                poses.append(np.array(transform))

        if len(poses) < 2:
            return TrajectoryQuality(
                smoothness_score=0.0,
                coverage_score=0.0,
                baseline_score=0.0,
                rotation_diversity=0.0,
                speed_consistency=0.0,
                overall_score=0.0,
                issues=["No valid poses found"]
            )

        poses = np.array(poses)  # (N, 4, 4)

        # Calculate trajectory metrics
        smoothness_score = self._calculate_smoothness_score(poses, issues)
        coverage_score = self._calculate_coverage_score(poses, issues)
        baseline_score = self._calculate_baseline_score(poses, issues)
        rotation_diversity = self._calculate_rotation_diversity(poses, issues)
        speed_consistency = self._calculate_speed_consistency(poses, issues)

        overall_score = (
            smoothness_score * 0.25 +
            coverage_score * 0.25 +
            baseline_score * 0.2 +
            rotation_diversity * 0.2 +
            speed_consistency * 0.1
        )

        return TrajectoryQuality(
            smoothness_score=smoothness_score,
            coverage_score=coverage_score,
            baseline_score=baseline_score,
            rotation_diversity=rotation_diversity,
            speed_consistency=speed_consistency,
            overall_score=overall_score,
            issues=issues
        )

    def filter_low_quality_frames(
        self,
        dataset_path: str,
        trajectory: Dict[str, Any],
        min_quality_threshold: Optional[float] = None
    ) -> Tuple[Dict[str, Any], List[int]]:
        """Filter out low-quality frames from trajectory."""
        if min_quality_threshold is None:
            min_quality_threshold = self.min_overall_threshold

        from pathlib import Path
        dataset_path = Path(dataset_path)

        frames = trajectory.get("frames", [])
        high_quality_frames = []
        removed_indices = []

        for i, frame in enumerate(frames):
            # Load and assess image
            file_path = frame.get("file_path", "")
            image_path = dataset_path / file_path

            if image_path.exists():
                try:
                    with Image.open(image_path) as img:
                        quality = self.assess_frame_quality(img, i)

                        if quality.overall_score >= min_quality_threshold:
                            # Update frame index in metadata
                            frame_copy = frame.copy()
                            frame_copy["frame"] = len(high_quality_frames)
                            frame_copy["original_frame"] = i
                            frame_copy["quality_score"] = quality.overall_score
                            high_quality_frames.append(frame_copy)
                        else:
                            removed_indices.append(i)

                except Exception as e:
                    removed_indices.append(i)
                    print(f"Failed to assess frame {i}: {e}")
            else:
                removed_indices.append(i)

        # Create filtered trajectory
        filtered_trajectory = trajectory.copy()
        filtered_trajectory["frames"] = high_quality_frames
        filtered_trajectory["original_frame_count"] = len(frames)
        filtered_trajectory["filtered_frame_count"] = len(high_quality_frames)
        filtered_trajectory["quality_threshold"] = min_quality_threshold

        return filtered_trajectory, removed_indices

    def _calculate_blur_score(self, gray_array: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance."""
        try:
            import cv2
            laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()
            # Normalize to 0-1 range (rough approximation)
            return min(1.0, laplacian_var / 1000.0)
        except ImportError:
            # Fallback: use gradient magnitude
            grad_x = np.gradient(gray_array.astype(float), axis=1)
            grad_y = np.gradient(gray_array.astype(float), axis=0)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return min(1.0, grad_magnitude.var() / 1000.0)

    def _calculate_brightness_score(self, gray_array: np.ndarray) -> float:
        """Calculate brightness score (0-1, optimal around 0.5)."""
        mean_brightness = gray_array.mean() / 255.0
        # Score is highest when brightness is around 0.5
        return 1.0 - 2.0 * abs(mean_brightness - 0.5)

    def _calculate_contrast_score(self, gray_array: np.ndarray) -> float:
        """Calculate contrast score using standard deviation."""
        contrast = gray_array.std() / 255.0
        return min(1.0, contrast * 4.0)  # Scale to 0-1 range

    def _calculate_color_diversity(self, rgb_array: np.ndarray) -> float:
        """Calculate color diversity score."""
        # Calculate histogram entropy for each channel
        entropies = []
        for channel in range(3):
            hist, _ = np.histogram(rgb_array[:, :, channel], bins=32, range=(0, 256))
            hist = hist + 1e-10  # Avoid log(0)
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            entropies.append(entropy)

        # Normalize to 0-1 range
        return np.mean(entropies) / 5.0  # Max entropy for 32 bins is ~5

    def _detect_motion_blur(self, gray_array: np.ndarray) -> float:
        """Detect motion blur (simplified approach)."""
        try:
            import cv2
            # Calculate edge density
            edges = cv2.Canny(gray_array, 50, 150)
            edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)

            # Motion blur typically reduces edge density
            return max(0.0, 1.0 - edge_density * 10)
        except ImportError:
            # Fallback: use simple gradient analysis
            grad_x = np.abs(np.gradient(gray_array.astype(float), axis=1))
            grad_y = np.abs(np.gradient(gray_array.astype(float), axis=0))
            edge_strength = np.sqrt(grad_x**2 + grad_y**2).mean()
            return max(0.0, 1.0 - edge_strength / 50.0)

    def _calculate_frame_overall_score(
        self,
        blur_score: float,
        brightness_score: float,
        contrast_score: float,
        color_diversity: float,
        motion_blur: float
    ) -> float:
        """Calculate overall frame quality score."""
        return (
            blur_score * 0.3 +
            brightness_score * 0.2 +
            contrast_score * 0.2 +
            color_diversity * 0.1 +
            (1.0 - motion_blur) * 0.2
        )

    def _calculate_smoothness_score(self, poses: np.ndarray, issues: List[str]) -> float:
        """Calculate trajectory smoothness score."""
        positions = poses[:, :3, 3]  # Extract positions

        if len(positions) < 3:
            return 0.5

        # Calculate second derivatives (acceleration)
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)

        # Smoothness is inverse of acceleration magnitude
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        mean_accel = accel_magnitudes.mean()

        if mean_accel > 1.0:
            issues.append("High trajectory acceleration - motion may be jerky")

        # Normalize to 0-1 range
        smoothness = max(0.0, 1.0 - mean_accel / 2.0)
        return smoothness

    def _calculate_coverage_score(self, poses: np.ndarray, issues: List[str]) -> float:
        """Calculate view coverage score."""
        positions = poses[:, :3, 3]

        # Calculate bounding box extent
        extent = positions.max(axis=0) - positions.min(axis=0)
        max_extent = np.max(extent)

        if max_extent < 0.1:
            issues.append("Very limited camera movement")
            return 0.1

        # Calculate relative spread in each dimension
        relative_extents = extent / max_extent
        coverage = np.mean(relative_extents)

        return min(1.0, coverage)

    def _calculate_baseline_score(self, poses: np.ndarray, issues: List[str]) -> float:
        """Calculate stereo baseline quality score."""
        positions = poses[:, :3, 3]

        # Calculate distances between consecutive frames
        baselines = []
        for i in range(len(positions) - 1):
            dist = np.linalg.norm(positions[i+1] - positions[i])
            baselines.append(dist)

        if not baselines:
            return 0.0

        mean_baseline = np.mean(baselines)
        std_baseline = np.std(baselines)

        # Optimal baseline range (depends on scene scale)
        if mean_baseline < 0.01:
            issues.append("Very small baselines - may cause poor stereo reconstruction")
        elif mean_baseline > 5.0:
            issues.append("Very large baselines - may cause correspondence issues")

        # Score based on consistency and magnitude
        consistency_score = max(0.0, 1.0 - std_baseline / max(mean_baseline, 1e-6))
        magnitude_score = min(1.0, mean_baseline / 0.5)  # Optimal around 0.5 units

        return (consistency_score + magnitude_score) / 2.0

    def _calculate_rotation_diversity(self, poses: np.ndarray, issues: List[str]) -> float:
        """Calculate rotation diversity score."""
        rotations = poses[:, :3, :3]

        if len(rotations) < 2:
            return 0.0

        # Calculate relative rotations
        rotation_angles = []
        for i in range(len(rotations) - 1):
            rel_rot = rotations[i+1] @ rotations[i].T
            # Extract rotation angle from rotation matrix
            trace = np.trace(rel_rot)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            rotation_angles.append(angle)

        if not rotation_angles:
            return 0.0

        total_rotation = sum(rotation_angles)
        max_rotation = max(rotation_angles)

        if total_rotation < 0.1:
            issues.append("Very limited rotation - may miss viewing angles")

        # Score based on total rotation and distribution
        total_score = min(1.0, total_rotation / (np.pi / 2))  # Normalize to 90 degrees
        return total_score

    def _calculate_speed_consistency(self, poses: np.ndarray, issues: List[str]) -> float:
        """Calculate speed consistency score."""
        positions = poses[:, :3, 3]

        if len(positions) < 3:
            return 1.0

        # Calculate speeds between frames
        speeds = []
        for i in range(len(positions) - 1):
            dist = np.linalg.norm(positions[i+1] - positions[i])
            speeds.append(dist)

        if not speeds:
            return 1.0

        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)

        # Check for sudden speed changes
        if std_speed > mean_speed:
            issues.append("Inconsistent motion speed - may cause temporal artifacts")

        # Consistency score
        if mean_speed == 0:
            return 1.0  # Static camera

        consistency = max(0.0, 1.0 - std_speed / mean_speed)
        return consistency


def filter_low_quality_frames(
    dataset_path: str,
    trajectory: Dict[str, Any],
    min_quality_threshold: float = 0.4
) -> Tuple[Dict[str, Any], List[int]]:
    """Convenience function to filter low-quality frames."""
    filter_obj = QualityFilter(min_overall_threshold=min_quality_threshold)
    return filter_obj.filter_low_quality_frames(dataset_path, trajectory, min_quality_threshold)


__all__ = ["QualityFilter", "FrameQuality", "TrajectoryQuality", "filter_low_quality_frames"]