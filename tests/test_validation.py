#!/usr/bin/env python3
"""Comprehensive test suite for GEN3C validation components."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from PIL import Image


class TestDatasetValidator(unittest.TestCase):
    """Test dataset validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.create_mock_dataset()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_mock_dataset(self):
        """Create a mock dataset for testing."""
        # Create directory structure
        (self.test_dir / "rgb").mkdir()
        (self.test_dir / "depth").mkdir()

        # Create mock transforms.json
        transforms_data = {
            "camera_model": "OPENCV",
            "w": 640,
            "h": 480,
            "fl_x": 400.0,
            "fl_y": 400.0,
            "cx": 320.0,
            "cy": 240.0,
            "fps": 24,
            "frames": []
        }

        # Create mock frames
        for i in range(5):
            # Create mock transform matrix (identity + translation)
            transform_matrix = np.eye(4)
            transform_matrix[0, 3] = i * 0.1  # Move along X axis

            frame_data = {
                "file_path": f"rgb/frame_{i:06d}.png",
                "depth_path": f"depth/frame_{i:06d}.npy",
                "transform_matrix": transform_matrix.tolist(),
                "frame": i
            }
            transforms_data["frames"].append(frame_data)

            # Create mock RGB image
            img = Image.new('RGB', (640, 480), color=(i * 50, 100, 150))
            img.save(self.test_dir / f"rgb/frame_{i:06d}.png")

            # Create mock depth data
            depth_data = np.random.rand(480, 640).astype(np.float32)
            np.save(self.test_dir / f"depth/frame_{i:06d}.npy", depth_data)

        # Save transforms.json
        with open(self.test_dir / "transforms.json", 'w') as f:
            json.dump(transforms_data, f, indent=2)

    def test_dataset_structure_validation(self):
        """Test dataset structure validation."""
        from comfy_gen3c.validation.dataset_validator import DatasetValidator

        validator = DatasetValidator()
        result = validator.validate_dataset(self.test_dir)

        # Should pass basic structure validation
        self.assertGreater(result.score, 0.0)
        self.assertIsInstance(result.issues, list)
        self.assertIsInstance(result.stats, dict)

    def test_missing_dataset(self):
        """Test validation of non-existent dataset."""
        from comfy_gen3c.validation.dataset_validator import DatasetValidator

        validator = DatasetValidator()
        fake_path = Path("/nonexistent/dataset")
        result = validator.validate_dataset(fake_path)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.score, 0.0)
        self.assertIn("does not exist", result.issues[0])

    def test_pose_validation(self):
        """Test camera pose validation."""
        from comfy_gen3c.validation.dataset_validator import DatasetValidator

        validator = DatasetValidator()

        # Load the transforms data we created
        with open(self.test_dir / "transforms.json", 'r') as f:
            transforms_data = json.load(f)

        # Test pose validation
        pose_valid, issues, stats = validator._validate_poses(transforms_data)

        self.assertTrue(pose_valid or len(issues) == 0)  # Should be valid or have specific issues
        self.assertIsInstance(stats.num_poses, int) if stats else None
        self.assertEqual(stats.num_poses, 5) if stats else None

    def test_intrinsics_validation(self):
        """Test camera intrinsics validation."""
        from comfy_gen3c.validation.dataset_validator import DatasetValidator

        validator = DatasetValidator()

        # Test valid intrinsics
        transforms_data = {
            "w": 640,
            "h": 480,
            "fl_x": 400.0,
            "fl_y": 400.0,
            "cx": 320.0,
            "cy": 240.0
        }

        valid, issues, stats = validator._validate_intrinsics(transforms_data)

        self.assertTrue(valid)
        self.assertEqual(len(issues), 0)
        self.assertIsNotNone(stats)
        self.assertAlmostEqual(stats.aspect_ratio, 1.0, places=2)

    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        from comfy_gen3c.validation.dataset_validator import DatasetValidator

        validator = DatasetValidator()

        # Test with various inputs
        score1 = validator._calculate_quality_score({}, 0, 0)  # No issues
        score2 = validator._calculate_quality_score({}, 1, 0)  # One issue
        score3 = validator._calculate_quality_score({}, 0, 5)  # Five warnings

        self.assertGreaterEqual(score1, score2)
        self.assertGreaterEqual(score2, score3)
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)


class TestTrajectoryPreview(unittest.TestCase):
    """Test trajectory preview functionality."""

    def setUp(self):
        """Set up test trajectory."""
        self.mock_trajectory = {
            "fps": 24,
            "handedness": "right",
            "frames": []
        }

        # Create a simple circular trajectory
        for i in range(10):
            angle = i * 2 * np.pi / 10
            radius = 2.0

            # Camera position on circle
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.5

            # Look-at transform (simplified)
            transform_matrix = np.eye(4)
            transform_matrix[0, 3] = x
            transform_matrix[1, 3] = y
            transform_matrix[2, 3] = z

            frame_data = {
                "frame": i,
                "width": 640,
                "height": 480,
                "intrinsics": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
                "extrinsics": {
                    "camera_to_world": transform_matrix.tolist()
                }
            }
            self.mock_trajectory["frames"].append(frame_data)

    def test_trajectory_preview_creation(self):
        """Test trajectory preview image creation."""
        from comfy_gen3c.validation.trajectory_preview import TrajectoryPreview

        preview = TrajectoryPreview()

        # Test 2D plot (fallback when matplotlib not available)
        image = preview._plot_trajectory_2d(self.mock_trajectory)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (800, 600))

    def test_frustum_plot_creation(self):
        """Test frustum plot creation."""
        from comfy_gen3c.validation.trajectory_preview import TrajectoryPreview

        preview = TrajectoryPreview()
        image = preview.create_frustum_plot(self.mock_trajectory)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (800, 600))

    def test_stats_image_creation(self):
        """Test trajectory statistics image."""
        from comfy_gen3c.validation.trajectory_preview import TrajectoryPreview

        preview = TrajectoryPreview()
        image = preview.generate_stats_image(self.mock_trajectory)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (800, 600))

    def test_empty_trajectory_handling(self):
        """Test handling of empty trajectory."""
        from comfy_gen3c.validation.trajectory_preview import TrajectoryPreview

        preview = TrajectoryPreview()
        empty_trajectory = {"fps": 24, "frames": []}

        image = preview._plot_trajectory_2d(empty_trajectory)
        self.assertIsInstance(image, Image.Image)

    @patch('matplotlib.pyplot')
    def test_3d_plot_with_matplotlib(self, mock_plt):
        """Test 3D plot creation when matplotlib is available."""
        from comfy_gen3c.validation.trajectory_preview import TrajectoryPreview

        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Mock canvas for image conversion
        mock_canvas = MagicMock()
        mock_fig.canvas = mock_canvas
        mock_canvas.buffer_rgba.return_value = np.zeros((600, 800, 4), dtype=np.uint8)

        preview = TrajectoryPreview()

        # This should use the mocked matplotlib
        image = preview.plot_trajectory_3d(self.mock_trajectory)

        # Verify matplotlib was called
        mock_plt.figure.assert_called_once()
        mock_fig.add_subplot.assert_called_once()


class TestQualityFilter(unittest.TestCase):
    """Test quality filtering functionality."""

    def setUp(self):
        """Set up test images and quality filter."""
        from comfy_gen3c.validation.quality_filters import QualityFilter
        self.quality_filter = QualityFilter()

    def create_test_image(self, brightness=128, contrast=50, blur=False):
        """Create a test image with specific characteristics."""
        # Create base image
        img_array = np.full((480, 640, 3), brightness, dtype=np.uint8)

        # Add contrast pattern
        pattern = np.sin(np.arange(640) * 0.1) * contrast
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + pattern, 0, 255)

        # Add blur if requested
        if blur:
            try:
                import cv2
                kernel = np.ones((15, 15), np.float32) / 225
                img_array = cv2.filter2D(img_array, -1, kernel)
            except ImportError:
                # Simple blur approximation
                img_array = img_array.astype(float)
                for _ in range(3):
                    img_array[1:-1, 1:-1] = (img_array[:-2, :-2] + img_array[2:, 2:]) / 2
                img_array = img_array.astype(np.uint8)

        return Image.fromarray(img_array)

    def test_blur_score_calculation(self):
        """Test blur score calculation."""
        # Sharp image should have higher blur score
        sharp_img = self.create_test_image(blur=False)
        blurry_img = self.create_test_image(blur=True)

        sharp_quality = self.quality_filter.assess_frame_quality(sharp_img, 0)
        blurry_quality = self.quality_filter.assess_frame_quality(blurry_img, 1)

        self.assertGreater(sharp_quality.blur_score, blurry_quality.blur_score)

    def test_brightness_assessment(self):
        """Test brightness assessment."""
        dark_img = self.create_test_image(brightness=30)
        normal_img = self.create_test_image(brightness=128)
        bright_img = self.create_test_image(brightness=200)

        dark_quality = self.quality_filter.assess_frame_quality(dark_img, 0)
        normal_quality = self.quality_filter.assess_frame_quality(normal_img, 1)
        bright_quality = self.quality_filter.assess_frame_quality(bright_img, 2)

        # Normal brightness should score highest
        self.assertGreater(normal_quality.brightness_score, dark_quality.brightness_score)
        self.assertGreater(normal_quality.brightness_score, bright_quality.brightness_score)

    def test_contrast_assessment(self):
        """Test contrast assessment."""
        low_contrast_img = self.create_test_image(contrast=5)
        high_contrast_img = self.create_test_image(contrast=100)

        low_quality = self.quality_filter.assess_frame_quality(low_contrast_img, 0)
        high_quality = self.quality_filter.assess_frame_quality(high_contrast_img, 1)

        self.assertGreater(high_quality.contrast_score, low_quality.contrast_score)

    def test_trajectory_quality_assessment(self):
        """Test trajectory quality assessment."""
        # Create a test trajectory
        trajectory = {
            "frames": []
        }

        # Simple linear trajectory
        for i in range(10):
            transform = np.eye(4)
            transform[0, 3] = i * 0.5  # Move along X

            frame = {
                "extrinsics": {
                    "camera_to_world": transform.tolist()
                }
            }
            trajectory["frames"].append(frame)

        quality = self.quality_filter.assess_trajectory_quality(trajectory)

        self.assertIsInstance(quality.overall_score, float)
        self.assertGreaterEqual(quality.overall_score, 0.0)
        self.assertLessEqual(quality.overall_score, 1.0)

    def test_frame_filtering(self):
        """Test frame quality filtering."""
        import tempfile
        import shutil

        # Create temporary dataset
        temp_dir = Path(tempfile.mkdtemp())
        rgb_dir = temp_dir / "rgb"
        rgb_dir.mkdir()

        try:
            # Create test images with different qualities
            good_img = self.create_test_image(brightness=128, contrast=50, blur=False)
            bad_img = self.create_test_image(brightness=30, contrast=5, blur=True)

            good_img.save(rgb_dir / "frame_000000.png")
            bad_img.save(rgb_dir / "frame_000001.png")

            # Create trajectory
            trajectory = {
                "frames": [
                    {"file_path": "rgb/frame_000000.png", "frame": 0},
                    {"file_path": "rgb/frame_000001.png", "frame": 1}
                ]
            }

            # Filter frames
            filtered_traj, removed = self.quality_filter.filter_low_quality_frames(
                str(temp_dir), trajectory, min_quality_threshold=0.3
            )

            # Should keep at least the good frame
            self.assertGreater(len(filtered_traj["frames"]), 0)
            self.assertLessEqual(len(filtered_traj["frames"]), 2)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestValidationNodes(unittest.TestCase):
    """Test ComfyUI validation nodes."""

    def test_dataset_validator_node(self):
        """Test dataset validator node."""
        from comfy_gen3c.validation.nodes import Gen3CDatasetValidator

        node = Gen3CDatasetValidator()

        # Test node structure
        input_types = node.INPUT_TYPES()
        self.assertIn("required", input_types)
        self.assertIn("dataset_path", input_types["required"])

        # Test with non-existent path
        score, status, issues, stats = node.validate_dataset(
            dataset_path="/nonexistent/path",
            min_frames=3,
            max_frames=1000,
            generate_report=False
        )

        self.assertEqual(score, 0.0)
        self.assertEqual(status, "FAILED")
        self.assertIn("does not exist", issues)

    def test_trajectory_preview_node(self):
        """Test trajectory preview node."""
        from comfy_gen3c.validation.nodes import Gen3CTrajectoryPreview

        node = Gen3CTrajectoryPreview()

        # Test node structure
        input_types = node.INPUT_TYPES()
        self.assertIn("required", input_types)
        self.assertIn("trajectory", input_types["required"])

        # Test with mock trajectory
        mock_trajectory = {
            "fps": 24,
            "frames": [
                {
                    "frame": 0,
                    "width": 640,
                    "height": 480,
                    "intrinsics": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
                    "extrinsics": {"camera_to_world": np.eye(4).tolist()}
                }
            ]
        }

        image_tensor, output_path = node.generate_preview(
            trajectory=mock_trajectory,
            plot_type="stats",
            output_dir="/tmp/test_preview"
        )

        self.assertIsInstance(image_tensor, torch.Tensor)
        self.assertEqual(len(image_tensor.shape), 4)  # (B, H, W, C)

    def test_quality_filter_node(self):
        """Test quality filter node."""
        from comfy_gen3c.validation.nodes import Gen3CQualityFilter

        node = Gen3CQualityFilter()

        # Test node structure
        input_types = node.INPUT_TYPES()
        self.assertIn("required", input_types)
        self.assertIn("trajectory", input_types["required"])

        # Test with mock data
        mock_trajectory = {"frames": []}

        filtered_traj, report, kept, removed = node.filter_quality(
            dataset_path="/nonexistent/path",
            trajectory=mock_trajectory,
            quality_threshold=0.5,
            min_blur_threshold=0.3,
            min_brightness=0.15,
            max_brightness=0.85
        )

        self.assertIn("does not exist", report)
        self.assertEqual(kept, 0)
        self.assertEqual(removed, 0)

    def test_trajectory_quality_analysis_node(self):
        """Test trajectory quality analysis node."""
        from comfy_gen3c.validation.nodes import Gen3CTrajectoryQualityAnalysis

        node = Gen3CTrajectoryQualityAnalysis()

        # Test with simple trajectory
        trajectory = {
            "frames": [
                {"extrinsics": {"camera_to_world": np.eye(4).tolist()}},
                {"extrinsics": {"camera_to_world": (np.eye(4) + 0.1).tolist()}}
            ]
        }

        overall, smoothness, coverage, baseline, rotation, report = node.analyze_trajectory(trajectory)

        self.assertIsInstance(overall, float)
        self.assertIsInstance(report, str)
        self.assertGreaterEqual(overall, 0.0)
        self.assertLessEqual(overall, 1.0)


def run_validation_tests():
    """Run all validation tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestDatasetValidator,
        TestTrajectoryPreview,
        TestQualityFilter,
        TestValidationNodes
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_validation_tests()
    sys.exit(0 if success else 1)