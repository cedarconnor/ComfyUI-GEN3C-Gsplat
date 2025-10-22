"""Shared constants for GEN3C nodes."""

from __future__ import annotations

# Default video/trajectory settings
DEFAULT_FPS = 24
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 576
DEFAULT_NEAR_PLANE = 0.01
DEFAULT_FAR_PLANE = 1000.0

# Cosmos model settings
COSMOS_SPATIAL_DOWNSAMPLE = 8
COSMOS_TEMPORAL_STRIDE = 8

# Dataset export settings
DEPTH_FILE_EXT = "npy"
RGB_FILE_EXT = "png"

# Training defaults
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_LEARNING_RATE = 5e-3
DEFAULT_POINTS_PER_FRAME = 50000

# Quality thresholds
DEFAULT_QUALITY_THRESHOLD = 0.4
DEFAULT_BLUR_THRESHOLD = 0.3
MIN_CONFIDENCE_THRESHOLD = 0.3

# Coordinate systems
HANDEDNESS_RIGHT = "right"
HANDEDNESS_LEFT = "left"

# Camera model
CAMERA_MODEL_OPENCV = "OPENCV"

# Backend names
BACKEND_AUTO = "auto"
BACKEND_COLMAP = "colmap"
BACKEND_VIPE = "vipe"

# Sampler defaults
DEFAULT_SAMPLER = "res_multistep"
DEFAULT_SCHEDULER = "normal"
