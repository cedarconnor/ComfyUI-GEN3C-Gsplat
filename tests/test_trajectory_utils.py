"""Tests for trajectory utility functions."""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image

from comfy_gen3c.dataset.trajectory_utils import (
    pose_result_to_trajectory,
    extract_frame_size_from_images,
    extract_frame_size_from_path,
    update_trajectory_frame_sizes,
)
from comfy_gen3c.constants import DEFAULT_FPS, DEFAULT_WIDTH, DEFAULT_HEIGHT


class MockPoseDepthResult:
    """Mock PoseDepthResult for testing."""

    def __init__(self, num_frames=5):
        self.poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1, 1)
        self.intrinsics = torch.tensor([[1000.0, 0.0, 512.0], [0.0, 1000.0, 288.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        self.confidence = 0.95
        self.error_message = None


class TestPoseResultToTrajectory:
    """Tests for pose result to trajectory conversion."""

    def test_basic_conversion(self):
        """Test basic conversion with default parameters."""
        result = MockPoseDepthResult(num_frames=5)
        trajectory = pose_result_to_trajectory(result, fps=24, source_name="test")

        assert trajectory["fps"] == 24
        assert trajectory["source"] == "test"
        assert trajectory["confidence"] == 0.95
        assert len(trajectory["frames"]) == 5

    def test_conversion_with_frame_size(self):
        """Test conversion with custom frame size."""
        result = MockPoseDepthResult(num_frames=3)
        trajectory = pose_result_to_trajectory(
            result,
            fps=30,
            source_name="test_custom",
            frame_size=(1920, 1080)
        )

        assert trajectory["fps"] == 30
        assert len(trajectory["frames"]) == 3
        assert trajectory["frames"][0]["width"] == 1920
        assert trajectory["frames"][0]["height"] == 1080

    def test_conversion_without_frame_size(self):
        """Test conversion uses defaults when frame size not provided."""
        result = MockPoseDepthResult(num_frames=2)
        trajectory = pose_result_to_trajectory(result, fps=24, source_name="test")

        assert trajectory["frames"][0]["width"] == DEFAULT_WIDTH
        assert trajectory["frames"][0]["height"] == DEFAULT_HEIGHT

    def test_frame_metadata(self):
        """Test that frame metadata is correctly populated."""
        result = MockPoseDepthResult(num_frames=1)
        trajectory = pose_result_to_trajectory(result, fps=24, source_name="test")

        frame = trajectory["frames"][0]
        assert "frame" in frame
        assert "width" in frame
        assert "height" in frame
        assert "near" in frame
        assert "far" in frame
        assert "intrinsics" in frame
        assert "extrinsics" in frame
        assert "camera_to_world" in frame["extrinsics"]
        assert "world_to_camera" in frame["extrinsics"]


class TestExtractFrameSizeFromImages:
    """Tests for extracting frame size from image tensors."""

    def test_extract_from_4d_tensor(self):
        """Test extraction from 4D tensor."""
        images = torch.rand(10, 720, 1280, 3)
        frame_size = extract_frame_size_from_images(images)
        assert frame_size == (1280, 720)

    def test_extract_from_3d_tensor(self):
        """Test extraction from 3D tensor."""
        images = torch.rand(480, 640, 3)
        frame_size = extract_frame_size_from_images(images)
        assert frame_size == (640, 480)

    def test_extract_from_empty_tensor(self):
        """Test extraction from empty tensor returns None."""
        images = torch.empty(0)
        frame_size = extract_frame_size_from_images(images)
        assert frame_size is None

    def test_extract_from_invalid_tensor(self):
        """Test extraction from invalid shaped tensor."""
        images = torch.rand(10, 20)  # 2D is invalid
        frame_size = extract_frame_size_from_images(images)
        assert frame_size is None


class TestExtractFrameSizeFromPath:
    """Tests for extracting frame size from image files."""

    def test_extract_from_image_file(self):
        """Test extraction from actual image file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Create a test image
            img = Image.new("RGB", (800, 600))
            img.save(tmp.name)
            tmp_path = Path(tmp.name)

            try:
                frame_size = extract_frame_size_from_path(tmp_path)
                assert frame_size == (800, 600)
            finally:
                tmp_path.unlink()

    def test_extract_from_nonexistent_file(self):
        """Test extraction from non-existent file returns defaults."""
        frame_size = extract_frame_size_from_path(Path("/nonexistent/image.png"))
        # Should return defaults
        assert frame_size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)


class TestUpdateTrajectoryFrameSizes:
    """Tests for updating trajectory frame sizes."""

    def test_update_all_frames(self):
        """Test updating all frame dimensions."""
        trajectory = {
            "fps": 24,
            "frames": [
                {"frame": 0, "width": 1024, "height": 576},
                {"frame": 1, "width": 1024, "height": 576},
                {"frame": 2, "width": 1024, "height": 576},
            ]
        }

        result = update_trajectory_frame_sizes(trajectory, width=1920, height=1080)

        assert result["frames"][0]["width"] == 1920
        assert result["frames"][0]["height"] == 1080
        assert result["frames"][1]["width"] == 1920
        assert result["frames"][1]["height"] == 1080
        assert result["frames"][2]["width"] == 1920
        assert result["frames"][2]["height"] == 1080

    def test_update_modifies_in_place(self):
        """Test that update modifies trajectory in place."""
        trajectory = {
            "fps": 24,
            "frames": [{"frame": 0, "width": 100, "height": 100}]
        }

        result = update_trajectory_frame_sizes(trajectory, width=200, height=200)

        # Should be the same object
        assert result is trajectory
        assert trajectory["frames"][0]["width"] == 200

    def test_update_empty_frames(self):
        """Test updating trajectory with no frames."""
        trajectory = {"fps": 24, "frames": []}
        result = update_trajectory_frame_sizes(trajectory, width=1920, height=1080)
        assert result["frames"] == []
