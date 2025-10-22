"""Pytest configuration and fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock ComfyUI modules BEFORE any imports
mock_folder_paths = MagicMock()
sys.modules["folder_paths"] = mock_folder_paths

# Create a mock comfy package
class MockComfyPackage(MagicMock):
    """Mock comfy package that allows attribute access."""
    def __getattr__(self, name):
        return MagicMock()

mock_comfy = MockComfyPackage()
sys.modules["comfy"] = mock_comfy
sys.modules["comfy.sample"] = MagicMock()
sys.modules["comfy.samplers"] = MagicMock()
sys.modules["comfy.samplers.KSampler"] = MagicMock()
sys.modules["comfy.utils"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()
sys.modules["comfy.model_patcher"] = MagicMock()
sys.modules["comfy.sd"] = MagicMock()

sys.modules["nodes"] = MagicMock()


@pytest.fixture
def temp_image_file():
    """Create a temporary image file for testing."""
    import tempfile
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new("RGB", (1024, 576))
        img.save(tmp.name)
        yield Path(tmp.name)
        Path(tmp.name).unlink()
