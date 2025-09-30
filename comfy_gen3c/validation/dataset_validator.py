"""Dataset validation utilities for GEN3C workflows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image


@dataclass
class ValidationResult:
    """Results from dataset validation."""
    is_valid: bool
    score: float  # 0-1 quality score
    issues: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


@dataclass
class PoseStats:
    """Statistics about camera poses."""
    num_poses: int
    translation_range: float
    rotation_range: float
    pose_diversity: float
    baseline_distances: List[float]
    view_coverage: float


@dataclass
class IntrinsicsStats:
    """Statistics about camera intrinsics."""
    focal_length_range: Tuple[float, float]
    principal_point_offset: Tuple[float, float]
    aspect_ratio: float
    fov_degrees: float


class DatasetValidator:
    """Validates GEN3C datasets for training quality."""

    def __init__(self, min_frames: int = 3, max_frames: int = 1000):
        self.min_frames = min_frames
        self.max_frames = max_frames

    def validate_dataset(self, dataset_path: Path) -> ValidationResult:
        """Comprehensive dataset validation."""
        dataset_path = Path(dataset_path)
        issues = []
        warnings = []
        stats = {}

        # Check directory structure
        structure_valid, structure_issues = self._validate_structure(dataset_path)
        issues.extend(structure_issues)

        if not structure_valid:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=issues,
                warnings=warnings,
                stats=stats
            )

        # Load transforms.json
        transforms_path = dataset_path / "transforms.json"
        try:
            with open(transforms_path, 'r') as f:
                transforms_data = json.load(f)
        except Exception as e:
            issues.append(f"Failed to load transforms.json: {e}")
            return ValidationResult(False, 0.0, issues, warnings, stats)

        # Validate transforms format
        transforms_valid, transform_issues, transform_warnings = self._validate_transforms(transforms_data)
        issues.extend(transform_issues)
        warnings.extend(transform_warnings)

        # Validate images
        image_valid, image_issues, image_stats = self._validate_images(dataset_path, transforms_data)
        issues.extend(image_issues)
        stats.update(image_stats)

        # Validate poses
        pose_valid, pose_issues, pose_stats = self._validate_poses(transforms_data)
        issues.extend(pose_issues)
        stats['poses'] = pose_stats.__dict__ if pose_stats else {}

        # Validate intrinsics
        intrinsics_valid, intrinsics_issues, intrinsics_stats = self._validate_intrinsics(transforms_data)
        issues.extend(intrinsics_issues)
        stats['intrinsics'] = intrinsics_stats.__dict__ if intrinsics_stats else {}

        # Validate depth (if present)
        depth_valid, depth_issues, depth_stats = self._validate_depth(dataset_path, transforms_data)
        issues.extend(depth_issues)
        stats.update(depth_stats)

        # Calculate overall quality score
        score = self._calculate_quality_score(stats, len(issues), len(warnings))

        is_valid = (
            structure_valid and
            transforms_valid and
            image_valid and
            pose_valid and
            intrinsics_valid and
            depth_valid and
            len(issues) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            warnings=warnings,
            stats=stats
        )

    def _validate_structure(self, dataset_path: Path) -> Tuple[bool, List[str]]:
        """Validate dataset directory structure."""
        issues = []

        if not dataset_path.exists():
            issues.append(f"Dataset directory does not exist: {dataset_path}")
            return False, issues

        if not dataset_path.is_dir():
            issues.append(f"Dataset path is not a directory: {dataset_path}")
            return False, issues

        # Check required files/directories
        required_items = {
            "transforms.json": "file",
            "rgb": "dir",
        }

        for item_name, item_type in required_items.items():
            item_path = dataset_path / item_name
            if not item_path.exists():
                issues.append(f"Missing required {item_type}: {item_name}")
            elif item_type == "file" and not item_path.is_file():
                issues.append(f"Expected file but found directory: {item_name}")
            elif item_type == "dir" and not item_path.is_dir():
                issues.append(f"Expected directory but found file: {item_name}")

        return len(issues) == 0, issues

    def _validate_transforms(self, transforms_data: Dict) -> Tuple[bool, List[str], List[str]]:
        """Validate transforms.json format and content."""
        issues = []
        warnings = []

        # Check required fields
        required_fields = ["camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "frames"]
        for field in required_fields:
            if field not in transforms_data:
                issues.append(f"Missing required field in transforms.json: {field}")

        # Check frame count
        frames = transforms_data.get("frames", [])
        if len(frames) < self.min_frames:
            issues.append(f"Too few frames: {len(frames)} < {self.min_frames}")
        elif len(frames) > self.max_frames:
            warnings.append(f"Very large dataset: {len(frames)} frames")

        # Validate each frame
        for i, frame in enumerate(frames):
            if "file_path" not in frame:
                issues.append(f"Frame {i} missing file_path")
            if "transform_matrix" not in frame:
                issues.append(f"Frame {i} missing transform_matrix")
            else:
                matrix = frame["transform_matrix"]
                if not isinstance(matrix, list) or len(matrix) != 4:
                    issues.append(f"Frame {i} transform_matrix should be 4x4 matrix")
                else:
                    for j, row in enumerate(matrix):
                        if not isinstance(row, list) or len(row) != 4:
                            issues.append(f"Frame {i} transform_matrix row {j} should have 4 elements")

        return len(issues) == 0, issues, warnings

    def _validate_images(self, dataset_path: Path, transforms_data: Dict) -> Tuple[bool, List[str], Dict]:
        """Validate image files."""
        issues = []
        stats = {"images": {}}

        rgb_dir = dataset_path / "rgb"
        frames = transforms_data.get("frames", [])

        image_sizes = []
        missing_images = 0

        for i, frame in enumerate(frames):
            file_path = frame.get("file_path", "")
            image_path = dataset_path / file_path

            if not image_path.exists():
                missing_images += 1
                issues.append(f"Missing image file: {file_path}")
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_sizes.append((width, height))

                    # Check consistency with transforms.json
                    expected_w = transforms_data.get("w")
                    expected_h = transforms_data.get("h")
                    if expected_w and width != expected_w:
                        issues.append(f"Image {file_path} width {width} != expected {expected_w}")
                    if expected_h and height != expected_h:
                        issues.append(f"Image {file_path} height {height} != expected {expected_h}")

            except Exception as e:
                issues.append(f"Failed to read image {file_path}: {e}")

        # Calculate image statistics
        if image_sizes:
            unique_sizes = set(image_sizes)
            stats["images"] = {
                "count": len(image_sizes),
                "missing": missing_images,
                "unique_sizes": len(unique_sizes),
                "size_consistency": len(unique_sizes) == 1,
                "resolution": image_sizes[0] if len(unique_sizes) == 1 else "variable"
            }

        return missing_images == 0 and len([i for i in issues if "image" in i.lower()]) == 0, issues, stats

    def _validate_poses(self, transforms_data: Dict) -> Tuple[bool, List[str], Optional[PoseStats]]:
        """Validate camera poses for quality and diversity."""
        issues = []
        frames = transforms_data.get("frames", [])

        if not frames:
            return False, ["No frames to validate"], None

        # Extract pose matrices
        poses = []
        for i, frame in enumerate(frames):
            matrix = frame.get("transform_matrix")
            if matrix and len(matrix) == 4 and all(len(row) == 4 for row in matrix):
                pose_matrix = np.array(matrix)
                poses.append(pose_matrix)
            else:
                issues.append(f"Invalid transform matrix for frame {i}")

        if not poses:
            return False, issues, None

        poses = np.array(poses)  # (N, 4, 4)

        # Extract positions and rotations
        positions = poses[:, :3, 3]  # (N, 3)
        rotations = poses[:, :3, :3]  # (N, 3, 3)

        # Calculate translation range
        translation_range = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))

        # Calculate rotation diversity (trace of relative rotations)
        rotation_angles = []
        for i in range(len(rotations)):
            for j in range(i + 1, len(rotations)):
                rel_rot = rotations[j] @ rotations[i].T
                trace = np.trace(rel_rot)
                angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                rotation_angles.append(angle)

        rotation_range = np.max(rotation_angles) if rotation_angles else 0.0

        # Calculate baseline distances
        baseline_distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                baseline_distances.append(dist)

        # Calculate pose diversity score
        position_std = np.std(positions, axis=0).mean()
        rotation_std = np.std(rotation_angles) if rotation_angles else 0.0
        pose_diversity = (position_std + rotation_std) / 2

        # Calculate view coverage (approximate solid angle coverage)
        if len(positions) > 2:
            center = positions.mean(axis=0)
            directions = positions - center
            directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)

            # Estimate coverage using convex hull area on unit sphere (simplified)
            view_coverage = min(1.0, len(directions) / 100.0)  # Rough approximation
        else:
            view_coverage = 0.1

        # Quality checks
        if translation_range < 0.1:
            issues.append("Very small camera movement - may cause training issues")
        if rotation_range < 0.1:
            issues.append("Very limited rotation range - may cause training issues")
        if len(baseline_distances) > 0 and max(baseline_distances) < 0.05:
            issues.append("Very small baseline distances - poor stereo reconstruction")

        pose_stats = PoseStats(
            num_poses=len(poses),
            translation_range=float(translation_range),
            rotation_range=float(rotation_range),
            pose_diversity=float(pose_diversity),
            baseline_distances=baseline_distances,
            view_coverage=float(view_coverage)
        )

        return len(issues) == 0, issues, pose_stats

    def _validate_intrinsics(self, transforms_data: Dict) -> Tuple[bool, List[str], Optional[IntrinsicsStats]]:
        """Validate camera intrinsics."""
        issues = []

        # Extract intrinsics
        try:
            w = transforms_data["w"]
            h = transforms_data["h"]
            fl_x = transforms_data["fl_x"]
            fl_y = transforms_data["fl_y"]
            cx = transforms_data["cx"]
            cy = transforms_data["cy"]
        except KeyError as e:
            issues.append(f"Missing intrinsic parameter: {e}")
            return False, issues, None

        # Validation checks
        if fl_x <= 0 or fl_y <= 0:
            issues.append("Invalid focal lengths (must be positive)")

        if abs(cx - w/2) > w/4:
            issues.append(f"Principal point x-offset very large: {cx} vs center {w/2}")

        if abs(cy - h/2) > h/4:
            issues.append(f"Principal point y-offset very large: {cy} vs center {h/2}")

        aspect_ratio = fl_y / fl_x if fl_x > 0 else 1.0
        if abs(aspect_ratio - 1.0) > 0.3:
            issues.append(f"Unusual aspect ratio: {aspect_ratio:.3f}")

        # Calculate field of view
        fov_x = 2 * math.atan(w / (2 * fl_x)) * 180 / math.pi
        if fov_x < 10 or fov_x > 170:
            issues.append(f"Unusual horizontal FOV: {fov_x:.1f} degrees")

        intrinsics_stats = IntrinsicsStats(
            focal_length_range=(float(min(fl_x, fl_y)), float(max(fl_x, fl_y))),
            principal_point_offset=(float(cx - w/2), float(cy - h/2)),
            aspect_ratio=float(aspect_ratio),
            fov_degrees=float(fov_x)
        )

        return len(issues) == 0, issues, intrinsics_stats

    def _validate_depth(self, dataset_path: Path, transforms_data: Dict) -> Tuple[bool, List[str], Dict]:
        """Validate depth maps if present."""
        issues = []
        stats = {"depth": {"present": False}}

        depth_dir = dataset_path / "depth"
        if not depth_dir.exists():
            return True, issues, stats  # Depth is optional

        stats["depth"]["present"] = True
        frames = transforms_data.get("frames", [])

        depth_files_found = 0
        depth_file_issues = 0

        for frame in frames:
            depth_path_key = frame.get("depth_path")
            if depth_path_key:
                depth_file = dataset_path / depth_path_key
                if depth_file.exists():
                    depth_files_found += 1

                    # Basic depth file validation
                    try:
                        if depth_file.suffix == ".npy":
                            depth_data = np.load(depth_file)
                            if depth_data.ndim != 2:
                                issues.append(f"Depth file {depth_path_key} should be 2D array")
                            if depth_data.min() < 0:
                                issues.append(f"Depth file {depth_path_key} contains negative values")
                        elif depth_file.suffix == ".png":
                            with Image.open(depth_file) as img:
                                if img.mode not in ["L", "I;16"]:
                                    issues.append(f"Depth PNG {depth_path_key} should be grayscale")
                    except Exception as e:
                        depth_file_issues += 1
                        issues.append(f"Failed to validate depth file {depth_path_key}: {e}")
                else:
                    issues.append(f"Missing depth file: {depth_path_key}")

        stats["depth"].update({
            "files_found": depth_files_found,
            "files_expected": len([f for f in frames if "depth_path" in f]),
            "validation_errors": depth_file_issues
        })

        return depth_file_issues == 0, issues, stats

    def _calculate_quality_score(self, stats: Dict, num_issues: int, num_warnings: int) -> float:
        """Calculate overall dataset quality score (0-1)."""
        base_score = 1.0

        # Penalize issues and warnings
        base_score -= num_issues * 0.1
        base_score -= num_warnings * 0.02

        # Pose quality bonus
        pose_stats = stats.get("poses", {})
        if pose_stats:
            pose_diversity = pose_stats.get("pose_diversity", 0)
            view_coverage = pose_stats.get("view_coverage", 0)
            base_score += min(0.2, pose_diversity * 0.1 + view_coverage * 0.1)

        # Image consistency bonus
        image_stats = stats.get("images", {})
        if image_stats.get("size_consistency", False):
            base_score += 0.05

        # Depth availability bonus
        depth_stats = stats.get("depth", {})
        if depth_stats.get("present", False):
            base_score += 0.1

        return max(0.0, min(1.0, base_score))


def validate_dataset_path(dataset_path: str) -> ValidationResult:
    """Convenience function to validate a dataset path."""
    validator = DatasetValidator()
    return validator.validate_dataset(Path(dataset_path))


__all__ = ["DatasetValidator", "ValidationResult", "PoseStats", "IntrinsicsStats", "validate_dataset_path"]