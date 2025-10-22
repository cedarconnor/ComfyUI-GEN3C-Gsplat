"""Tests for utility functions."""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

import numpy as np
import torch

from comfy_gen3c.utils import (
    validate_path_exists,
    validate_trajectory,
    validate_frame_tensor,
    extract_frame_dimensions,
    create_dummy_trajectory,
    safe_matrix_inverse,
    parse_json_safely,
    resolve_output_path,
)
from comfy_gen3c.exceptions import Gen3CInvalidInputError, Gen3CTrajectoryError
from comfy_gen3c.constants import DEFAULT_FPS


class TestPathValidation:
    """Tests for path validation."""

    def test_validate_path_exists_with_valid_path(self):
        """Test validation with an existing path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_path_exists(tmpdir, "Test path")
            assert result == Path(tmpdir).resolve()

    def test_validate_path_exists_with_invalid_path(self):
        """Test validation with non-existent path."""
        with pytest.raises(Gen3CInvalidInputError, match="does not exist"):
            validate_path_exists("/nonexistent/path", "Test path")

    def test_validate_path_exists_with_empty_path(self):
        """Test validation with empty path."""
        with pytest.raises(Gen3CInvalidInputError, match="is empty"):
            validate_path_exists("", "Test path")


class TestTrajectoryValidation:
    """Tests for trajectory validation."""

    def test_validate_trajectory_with_valid_data(self):
        """Test validation with properly formatted trajectory."""
        trajectory = {
            "fps": 24,
            "frames": [
                {
                    "frame": 0,
                    "width": 1024,
                    "height": 576,
                    "intrinsics": [[1.0, 0.0, 512.0], [0.0, 1.0, 288.0], [0.0, 0.0, 1.0]],
                    "extrinsics": {
                        "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                    }
                }
            ]
        }
        # Should not raise
        validate_trajectory(trajectory)

    def test_validate_trajectory_with_empty_data(self):
        """Test validation with empty trajectory."""
        with pytest.raises(Gen3CTrajectoryError, match="empty or None"):
            validate_trajectory({})

    def test_validate_trajectory_with_no_frames(self):
        """Test validation with trajectory missing frames."""
        with pytest.raises(Gen3CTrajectoryError, match="missing 'frames' field"):
            validate_trajectory({"fps": 24})

    def test_validate_trajectory_with_empty_frames(self):
        """Test validation with empty frames list."""
        with pytest.raises(Gen3CTrajectoryError, match="has no frames"):
            validate_trajectory({"fps": 24, "frames": []})

    def test_validate_trajectory_with_missing_fields(self):
        """Test validation with frames missing required fields."""
        trajectory = {
            "frames": [{"frame": 0}]  # Missing required fields
        }
        with pytest.raises(Gen3CTrajectoryError, match="missing fields"):
            validate_trajectory(trajectory)


class TestFrameTensorValidation:
    """Tests for frame tensor validation."""

    def test_validate_frame_tensor_4d(self):
        """Test validation with 4D tensor (F, H, W, C)."""
        tensor = torch.rand(10, 576, 1024, 3)
        result = validate_frame_tensor(tensor)
        assert result.shape == (10, 576, 1024, 3)

    def test_validate_frame_tensor_3d(self):
        """Test validation with 3D tensor (H, W, C)."""
        tensor = torch.rand(576, 1024, 3)
        result = validate_frame_tensor(tensor)
        assert result.shape == (1, 576, 1024, 3)

    def test_validate_frame_tensor_5d(self):
        """Test validation with 5D tensor (B, F, H, W, C)."""
        tensor = torch.rand(1, 10, 576, 1024, 3)
        result = validate_frame_tensor(tensor)
        assert result.shape == (10, 576, 1024, 3)

    def test_validate_frame_tensor_invalid_shape(self):
        """Test validation with invalid tensor shape."""
        tensor = torch.rand(10, 20)  # 2D is invalid
        with pytest.raises(Gen3CInvalidInputError, match="invalid shape"):
            validate_frame_tensor(tensor)

    def test_validate_frame_tensor_empty(self):
        """Test validation with empty tensor."""
        tensor = torch.empty(0)
        with pytest.raises(Gen3CInvalidInputError, match="empty"):
            validate_frame_tensor(tensor)


class TestExtractFrameDimensions:
    """Tests for extracting frame dimensions."""

    def test_extract_from_4d_tensor(self):
        """Test extraction from 4D tensor."""
        tensor = torch.rand(10, 576, 1024, 3)
        width, height = extract_frame_dimensions(tensor)
        assert width == 1024
        assert height == 576

    def test_extract_from_3d_tensor(self):
        """Test extraction from 3D tensor."""
        tensor = torch.rand(576, 1024, 3)
        width, height = extract_frame_dimensions(tensor)
        assert width == 1024
        assert height == 576

    def test_extract_from_empty_tensor(self):
        """Test extraction from empty tensor returns defaults."""
        tensor = torch.empty(0)
        width, height = extract_frame_dimensions(tensor)
        # Should return defaults
        assert width > 0
        assert height > 0


class TestDummyTrajectory:
    """Tests for dummy trajectory creation."""

    def test_create_dummy_trajectory_defaults(self):
        """Test creating dummy trajectory with defaults."""
        trajectory = create_dummy_trajectory()
        assert trajectory["fps"] == DEFAULT_FPS
        assert trajectory["frames"] == []
        assert trajectory["handedness"] == "right"
        assert trajectory["source"] == "dummy"

    def test_create_dummy_trajectory_custom(self):
        """Test creating dummy trajectory with custom values."""
        trajectory = create_dummy_trajectory(fps=30, source="test", handedness="left")
        assert trajectory["fps"] == 30
        assert trajectory["source"] == "test"
        assert trajectory["handedness"] == "left"


class TestSafeMatrixInverse:
    """Tests for safe matrix inversion."""

    def test_safe_matrix_inverse_valid(self):
        """Test inversion of a valid matrix."""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = safe_matrix_inverse(matrix)
        # Check that matrix * inverse = identity
        identity = matrix @ result
        np.testing.assert_array_almost_equal(identity, np.eye(2), decimal=10)

    def test_safe_matrix_inverse_singular(self):
        """Test inversion of a singular matrix."""
        matrix = np.array([[1, 1], [1, 1]], dtype=np.float64)  # Singular
        with pytest.raises(Gen3CInvalidInputError, match="Cannot invert matrix"):
            safe_matrix_inverse(matrix)


class TestParseJsonSafely:
    """Tests for safe JSON parsing."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = parse_json_safely(json_str)
        assert result == {"key": "value", "number": 42}

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns default."""
        json_str = '{invalid json}'
        result = parse_json_safely(json_str, default={"fallback": True})
        assert result == {"fallback": True}

    def test_parse_empty_json(self):
        """Test parsing empty string returns default."""
        result = parse_json_safely("", default={"empty": True})
        assert result == {"empty": True}

    def test_parse_none_json(self):
        """Test parsing with no default returns empty dict."""
        result = parse_json_safely("")
        assert result == {}


class TestResolveOutputPath:
    """Tests for output path resolution."""

    def test_resolve_output_path_with_variable(self):
        """Test resolution with ${output_dir} variable."""
        path_str = "${output_dir}/my_dataset"
        result = resolve_output_path(path_str)
        assert "output" in str(result)
        assert "my_dataset" in str(result)

    def test_resolve_output_path_without_variable(self):
        """Test resolution without variable."""
        path_str = "/absolute/path/to/dataset"
        result = resolve_output_path(path_str)
        assert result == Path("/absolute/path/to/dataset")
