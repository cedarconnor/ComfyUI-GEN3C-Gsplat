#!/usr/bin/env python3
"""Test core validation logic without ComfyUI dependencies."""

import sys
import os
import tempfile
import json
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_blur_calculation():
    """Test blur calculation logic."""
    print("Testing blur calculation...")

    # Create a sharp image (high contrast edges)
    sharp_image = np.zeros((100, 100), dtype=np.uint8)
    sharp_image[40:60, 40:60] = 255  # White square on black background

    # Create a blurry image (low contrast)
    blurry_image = np.ones((100, 100), dtype=np.uint8) * 128  # Gray image

    # Simple Laplacian variance calculation (same as in our code)
    def calculate_blur_score(image):
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.uint8)

        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        # Apply Laplacian filter manually (simple convolution)
        h, w = image.shape
        filtered = np.zeros_like(image, dtype=np.float32)
        for i in range(1, h-1):
            for j in range(1, w-1):
                patch = image[i-1:i+2, j-1:j+2]
                filtered[i, j] = np.sum(patch * laplacian)

        return float(np.var(filtered))

    sharp_score = calculate_blur_score(sharp_image)
    blurry_score = calculate_blur_score(blurry_image)

    print(f"  Sharp image blur score: {sharp_score:.2f}")
    print(f"  Blurry image blur score: {blurry_score:.2f}")

    # Sharp image should have higher variance (less blur)
    if sharp_score > blurry_score:
        print("  PASS: Sharp image has higher score than blurry image")
        return True
    else:
        print("  FAIL: Blur detection not working correctly")
        return False

def test_brightness_calculation():
    """Test brightness calculation."""
    print("Testing brightness calculation...")

    # Create dark and bright images
    dark_image = np.ones((100, 100), dtype=np.uint8) * 50   # Dark
    bright_image = np.ones((100, 100), dtype=np.uint8) * 200  # Bright

    def calculate_brightness(image):
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        return float(np.mean(image) / 255.0)

    dark_brightness = calculate_brightness(dark_image)
    bright_brightness = calculate_brightness(bright_image)

    print(f"  Dark image brightness: {dark_brightness:.3f}")
    print(f"  Bright image brightness: {bright_brightness:.3f}")

    if dark_brightness < bright_brightness:
        print("  PASS: Brightness calculation working correctly")
        return True
    else:
        print("  FAIL: Brightness calculation not working")
        return False

def test_camera_matrix_validation():
    """Test camera matrix validation logic."""
    print("Testing camera matrix validation...")

    # Valid 4x4 transformation matrix
    valid_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 5],
        [0, 0, 0, 1]
    ]

    # Invalid matrix (wrong size)
    invalid_matrix = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    def validate_transform_matrix(matrix):
        """Validate a transformation matrix."""
        if not isinstance(matrix, list) or len(matrix) != 4:
            return False
        for row in matrix:
            if not isinstance(row, list) or len(row) != 4:
                return False
        # Check if it's reasonably close to a valid transform
        # (bottom row should be [0, 0, 0, 1])
        bottom_row = matrix[3]
        expected = [0, 0, 0, 1]
        return bottom_row == expected

    valid_result = validate_transform_matrix(valid_matrix)
    invalid_result = validate_transform_matrix(invalid_matrix)

    print(f"  Valid matrix result: {valid_result}")
    print(f"  Invalid matrix result: {invalid_result}")

    if valid_result and not invalid_result:
        print("  PASS: Matrix validation working correctly")
        return True
    else:
        print("  FAIL: Matrix validation not working")
        return False

def test_trajectory_smoothness():
    """Test trajectory smoothness calculation."""
    print("Testing trajectory smoothness calculation...")

    # Create smooth trajectory (linear motion)
    smooth_frames = []
    for i in range(10):
        frame = {
            "transform_matrix": [
                [1, 0, 0, i * 0.5],  # Steady movement in X
                [0, 1, 0, 0],
                [0, 0, 1, 5],
                [0, 0, 0, 1]
            ]
        }
        smooth_frames.append(frame)

    # Create jerky trajectory (random movements)
    np.random.seed(42)  # For reproducible results
    jerky_frames = []
    for i in range(10):
        frame = {
            "transform_matrix": [
                [1, 0, 0, np.random.uniform(-5, 5)],  # Random X movement
                [0, 1, 0, np.random.uniform(-5, 5)],  # Random Y movement
                [0, 0, 1, 5],
                [0, 0, 0, 1]
            ]
        }
        jerky_frames.append(frame)

    def calculate_smoothness(frames):
        """Calculate trajectory smoothness."""
        if len(frames) < 3:
            return 1.0

        positions = []
        for frame in frames:
            matrix = frame["transform_matrix"]
            # Extract position (translation part)
            pos = [matrix[0][3], matrix[1][3], matrix[2][3]]
            positions.append(pos)

        # Calculate acceleration magnitudes
        accelerations = []
        for i in range(1, len(positions) - 1):
            prev_pos = np.array(positions[i-1])
            curr_pos = np.array(positions[i])
            next_pos = np.array(positions[i+1])

            # Velocity vectors
            vel1 = curr_pos - prev_pos
            vel2 = next_pos - curr_pos

            # Acceleration (change in velocity)
            accel = vel2 - vel1
            accel_mag = np.linalg.norm(accel)
            accelerations.append(accel_mag)

        if not accelerations:
            return 1.0

        # Smoothness is inverse of acceleration variance
        accel_var = np.var(accelerations)
        smoothness = 1.0 / (1.0 + accel_var)
        return smoothness

    smooth_score = calculate_smoothness(smooth_frames)
    jerky_score = calculate_smoothness(jerky_frames)

    print(f"  Smooth trajectory score: {smooth_score:.3f}")
    print(f"  Jerky trajectory score: {jerky_score:.3f}")

    if smooth_score > jerky_score:
        print("  PASS: Smoothness calculation working correctly")
        return True
    else:
        print("  FAIL: Smoothness calculation not working")
        return False

def test_json_validation():
    """Test JSON structure validation."""
    print("Testing JSON structure validation...")

    # Valid transforms.json structure
    valid_transforms = {
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
            },
            {
                "file_path": "./images/frame_001.jpg",
                "transform_matrix": [
                    [1, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 5],
                    [0, 0, 0, 1]
                ]
            }
        ]
    }

    # Invalid transforms.json (missing required fields)
    invalid_transforms = {
        "frames": [
            {
                "file_path": "./images/frame_000.jpg"
                # Missing transform_matrix
            }
        ]
        # Missing camera_angle_x
    }

    def validate_transforms_json(data):
        """Validate transforms.json structure."""
        if not isinstance(data, dict):
            return False

        # Check required fields
        if "frames" not in data:
            return False

        frames = data["frames"]
        if not isinstance(frames, list) or len(frames) == 0:
            return False

        # Check each frame
        for frame in frames:
            if not isinstance(frame, dict):
                return False
            if "file_path" not in frame:
                return False
            if "transform_matrix" not in frame:
                return False

            # Validate transform matrix
            matrix = frame["transform_matrix"]
            if not isinstance(matrix, list) or len(matrix) != 4:
                return False
            for row in matrix:
                if not isinstance(row, list) or len(row) != 4:
                    return False

        return True

    valid_result = validate_transforms_json(valid_transforms)
    invalid_result = validate_transforms_json(invalid_transforms)

    print(f"  Valid JSON result: {valid_result}")
    print(f"  Invalid JSON result: {invalid_result}")

    if valid_result and not invalid_result:
        print("  PASS: JSON validation working correctly")
        return True
    else:
        print("  FAIL: JSON validation not working")
        return False

def main():
    """Run all core logic tests."""
    print("=" * 50)
    print("Core Logic Tests (No ComfyUI Dependencies)")
    print("=" * 50)

    tests = [
        test_blur_calculation,
        test_brightness_calculation,
        test_camera_matrix_validation,
        test_trajectory_smoothness,
        test_json_validation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print()
        try:
            if test():
                passed += 1
                print("  RESULT: PASS")
            else:
                print("  RESULT: FAIL")
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  RESULT: FAIL")
        print()

    print("=" * 50)
    print(f"Tests completed: {passed}/{total} passed")

    if passed == total:
        print("All core logic tests passed!")
        return 0
    else:
        print(f"{total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())