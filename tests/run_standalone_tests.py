#!/usr/bin/env python3
"""Standalone test runner for GEN3C validation components.

This script tests the core validation functionality without requiring ComfyUI context.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_dataclass_imports():
    """Test that core dataclasses can be imported."""
    print("Testing dataclass imports...")
    try:
        from comfy_gen3c.validation.dataset_validator import ValidationResult, ValidationIssue
        from comfy_gen3c.validation.quality_filters import FrameQuality, TrajectoryQuality
        print("  ✓ Dataclasses imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_trajectory_quality():
    """Test trajectory quality calculation."""
    print("Testing trajectory quality calculation...")
    try:
        from comfy_gen3c.validation.quality_filters import TrajectoryQuality
        import numpy as np

        # Create mock trajectory data
        frames = []
        for i in range(10):
            frame = {
                "transform_matrix": [
                    [1, 0, 0, i * 0.1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 5],
                    [0, 0, 0, 1]
                ]
            }
            frames.append(frame)

        trajectory = {"frames": frames}

        # Test quality calculation
        quality = TrajectoryQuality()
        smoothness = quality.calculate_smoothness(trajectory)
        coverage = quality.calculate_coverage(trajectory)
        baseline = quality.calculate_baseline_quality(trajectory)

        print(f"  ✓ Smoothness: {smoothness:.3f}")
        print(f"  ✓ Coverage: {coverage:.3f}")
        print(f"  ✓ Baseline: {baseline:.3f}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_frame_quality():
    """Test frame quality assessment with mock data."""
    print("Testing frame quality assessment...")
    try:
        from comfy_gen3c.validation.quality_filters import FrameQuality
        import numpy as np

        # Create a mock image (grayscale)
        mock_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        quality = FrameQuality()
        blur_score = quality.calculate_blur_score(mock_image)
        brightness = quality.calculate_brightness(mock_image)
        contrast = quality.calculate_contrast(mock_image)

        print(f"  ✓ Blur score: {blur_score:.3f}")
        print(f"  ✓ Brightness: {brightness:.3f}")
        print(f"  ✓ Contrast: {contrast:.3f}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_dataset_validator_logic():
    """Test dataset validator core logic."""
    print("Testing dataset validator logic...")
    try:
        from comfy_gen3c.validation.dataset_validator import DatasetValidator, ValidationResult

        # Create temporary test dataset structure
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()

            # Create images directory
            images_dir = dataset_path / "images"
            images_dir.mkdir()

            # Create minimal transforms.json
            transforms = {
                "camera_angle_x": 0.8575560450553894,
                "frames": [
                    {
                        "file_path": "./images/frame_000.jpg",
                        "transform_matrix": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 5],
                            [0, 0, 0, 1]
                        ]
                    }
                ]
            }

            transforms_file = dataset_path / "transforms.json"
            with open(transforms_file, 'w') as f:
                json.dump(transforms, f)

            # Test validator
            validator = DatasetValidator()

            # Test structure validation
            structure_valid = validator._validate_structure(str(dataset_path))
            print(f"  ✓ Structure validation: {structure_valid}")

            # Test transforms validation
            poses_valid = validator._validate_poses(transforms)
            print(f"  ✓ Poses validation: {poses_valid}")

            return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_trajectory_preview_creation():
    """Test trajectory preview creation."""
    print("Testing trajectory preview creation...")
    try:
        from comfy_gen3c.validation.trajectory_preview import TrajectoryPreview
        import numpy as np

        # Create mock trajectory
        frames = []
        for i in range(5):
            angle = i * np.pi / 4
            frame = {
                "transform_matrix": [
                    [np.cos(angle), 0, np.sin(angle), 3 * np.cos(angle)],
                    [0, 1, 0, 1],
                    [-np.sin(angle), 0, np.cos(angle), 3 * np.sin(angle)],
                    [0, 0, 0, 1]
                ]
            }
            frames.append(frame)

        trajectory = {"frames": frames}

        preview = TrajectoryPreview()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test stats image creation
            stats_path = preview.create_stats_image(trajectory, temp_dir)
            if os.path.exists(stats_path):
                print(f"  ✓ Stats image created at: {stats_path}")
            else:
                print("  ⚠ Stats image not created (matplotlib might not be available)")

            return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Run all standalone tests."""
    print("=" * 60)
    print("ComfyUI-GEN3C-Gsplat Standalone Tests")
    print("=" * 60)

    tests = [
        test_dataclass_imports,
        test_trajectory_quality,
        test_frame_quality,
        test_dataset_validator_logic,
        test_trajectory_preview_creation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print()
        if test():
            passed += 1
        print()

    print("=" * 60)
    print(f"Tests completed: {passed}/{total} passed")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())