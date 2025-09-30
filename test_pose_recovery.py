#!/usr/bin/env python3
"""Test pose/depth recovery pipeline."""

import sys
import tempfile
import numpy as np
from pathlib import Path

# Add module to path for testing
sys.path.insert(0, str(Path(__file__).parent))

def test_syntax_checks():
    """Test that all pose recovery modules have valid syntax."""
    print("Testing pose recovery module syntax...")

    import ast

    files_to_check = [
        "comfy_gen3c/dataset/pose_depth.py",
        "comfy_gen3c/dataset/recovery_nodes.py",
    ]

    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                ast.parse(f.read())
            print(f"  ✓ {file_path} - syntax OK")
        except SyntaxError as e:
            print(f"  ✗ {file_path} - syntax error: {e}")
            return False
        except Exception as e:
            print(f"  ✗ {file_path} - error: {e}")
            return False

    return True


def test_pose_depth_structures():
    """Test basic data structures without external dependencies."""
    print("Testing pose/depth data structures...")

    try:
        from comfy_gen3c.dataset.pose_depth import PoseDepthResult, RecoveryConfig

        # Test RecoveryConfig creation
        config = RecoveryConfig()
        assert config.backend == "auto"
        assert config.max_frames == 100
        assert config.estimate_depth == True
        print("  ✓ RecoveryConfig creation works")

        # Test custom config
        custom_config = RecoveryConfig(
            backend="colmap",
            max_frames=50,
            downsample_factor=0.5,
            estimate_depth=False
        )
        assert custom_config.backend == "colmap"
        assert custom_config.max_frames == 50
        assert custom_config.downsample_factor == 0.5
        assert custom_config.estimate_depth == False
        print("  ✓ Custom RecoveryConfig works")

        # Test PoseDepthResult creation
        import torch
        poses = torch.eye(4).unsqueeze(0)  # (1, 4, 4)
        intrinsics = torch.eye(3)  # (3, 3)
        result = PoseDepthResult(
            poses=poses,
            intrinsics=intrinsics,
            depths=None,
            confidence=0.8
        )
        assert result.poses.shape == (1, 4, 4)
        assert result.intrinsics.shape == (3, 3)
        assert result.depths is None
        assert result.confidence == 0.8
        print("  ✓ PoseDepthResult creation works")

        return True

    except Exception as e:
        print(f"  ✗ Data structure test failed: {e}")
        return False


def test_recovery_node_structure():
    """Test recovery node structure without running actual recovery."""
    print("Testing recovery node structure...")

    try:
        from comfy_gen3c.dataset.recovery_nodes import Gen3CPoseDepthFromVideo, Gen3CPoseDepthFromImages

        # Test node class creation
        video_node = Gen3CPoseDepthFromVideo()
        image_node = Gen3CPoseDepthFromImages()

        # Test input types structure
        video_inputs = video_node.INPUT_TYPES()
        assert "required" in video_inputs
        assert "video_path" in video_inputs["required"]
        assert "max_frames" in video_inputs["required"]
        assert "backend" in video_inputs["required"]
        print("  ✓ Video recovery node structure OK")

        image_inputs = image_node.INPUT_TYPES()
        assert "required" in image_inputs
        assert "images" in image_inputs["required"]
        assert "backend" in image_inputs["required"]
        print("  ✓ Image recovery node structure OK")

        return True

    except Exception as e:
        print(f"  ✗ Recovery node test failed: {e}")
        return False


def test_export_node_integration():
    """Test export node integration with pose recovery."""
    print("Testing export node integration...")

    try:
        from comfy_gen3c.export_nodes import Gen3CVideoToDataset

        # Test node creation
        export_node = Gen3CVideoToDataset()

        # Test input structure
        inputs = export_node.INPUT_TYPES()
        assert "required" in inputs
        assert "video_path" in inputs["required"]
        assert "output_dir" in inputs["required"]
        assert "backend" in inputs["required"]
        print("  ✓ Video export node structure OK")

        # Test return types
        assert export_node.RETURN_TYPES == ("STRING", "GEN3C_TRAJECTORY", "FLOAT", "STRING")
        assert export_node.RETURN_NAMES == ("dataset_dir", "trajectory", "confidence", "status")
        print("  ✓ Video export node return types OK")

        return True

    except Exception as e:
        print(f"  ✗ Export node integration test failed: {e}")
        return False


def test_colmap_availability():
    """Test if COLMAP is available on the system."""
    print("Testing COLMAP availability...")

    import subprocess

    try:
        result = subprocess.run(["colmap", "--help"],
                              capture_output=True,
                              timeout=10)
        if result.returncode == 0:
            print("  ✓ COLMAP is available")
            return True
        else:
            print("  ⚠ COLMAP found but returned error")
            return False
    except FileNotFoundError:
        print("  ⚠ COLMAP not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("  ⚠ COLMAP command timed out")
        return False
    except Exception as e:
        print(f"  ⚠ COLMAP test failed: {e}")
        return False


def test_opencv_availability():
    """Test if OpenCV is available for video processing."""
    print("Testing OpenCV availability...")

    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__} is available")
        return True
    except ImportError:
        print("  ⚠ OpenCV not available - install with: pip install opencv-python")
        return False


def test_scipy_availability():
    """Test if SciPy is available for pose computations."""
    print("Testing SciPy availability...")

    try:
        import scipy
        from scipy.spatial.transform import Rotation
        print(f"  ✓ SciPy {scipy.__version__} is available")

        # Test rotation functionality used in COLMAP parsing
        R = Rotation.from_quat([0, 0, 0, 1])
        matrix = R.as_matrix()
        assert matrix.shape == (3, 3)
        print("  ✓ Rotation functionality works")

        return True
    except ImportError:
        print("  ⚠ SciPy not available - install with: pip install scipy")
        return False


def test_mock_pose_recovery():
    """Test pose recovery with mock data (no external dependencies)."""
    print("Testing mock pose recovery...")

    try:
        import torch

        # Test the fallback case when backends fail
        from comfy_gen3c.dataset.pose_depth import estimate_from_video, RecoveryConfig

        # Use a non-existent video file to trigger fallback
        fake_video = Path("nonexistent_video.mp4")
        config = RecoveryConfig(backend="auto", max_frames=5)

        result = estimate_from_video(fake_video, config)

        # Should return a dummy result with error
        assert result.confidence == 0.0
        assert result.error_message is not None
        assert result.poses.shape[0] >= 1
        assert result.intrinsics.shape == (3, 3)
        print("  ✓ Fallback pose recovery works")

        return True

    except Exception as e:
        print(f"  ✗ Mock pose recovery test failed: {e}")
        return False


def test_trajectory_conversion():
    """Test conversion from pose recovery to trajectory format."""
    print("Testing trajectory conversion...")

    try:
        import torch
        import numpy as np
        from comfy_gen3c.dataset.recovery_nodes import Gen3CPoseDepthFromVideo

        # Create mock recovery result
        from comfy_gen3c.dataset.pose_depth import PoseDepthResult

        # Mock 3 frames with identity poses
        poses = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
        intrinsics = torch.tensor([[800.0, 0, 400.0], [0, 800.0, 300.0], [0, 0, 1]])

        result = PoseDepthResult(
            poses=poses,
            intrinsics=intrinsics,
            depths=None,
            confidence=0.9
        )

        # Test trajectory conversion
        node = Gen3CPoseDepthFromVideo()
        trajectory = node._result_to_trajectory(result, "test_video.mp4", 3)

        assert trajectory["fps"] == 24
        assert len(trajectory["frames"]) == 3
        assert trajectory["handedness"] == "right"
        assert trajectory["confidence"] == 0.9

        # Check frame structure
        frame = trajectory["frames"][0]
        assert "frame" in frame
        assert "width" in frame
        assert "height" in frame
        assert "intrinsics" in frame
        assert "extrinsics" in frame

        # Check intrinsics
        K = frame["intrinsics"]
        assert K[0][0] == 800.0  # fx
        assert K[1][1] == 800.0  # fy
        assert K[0][2] == 400.0  # cx
        assert K[1][2] == 300.0  # cy

        print("  ✓ Trajectory conversion works")
        return True

    except Exception as e:
        print(f"  ✗ Trajectory conversion test failed: {e}")
        return False


def run_all_tests():
    """Run all pose recovery tests."""
    print("=== Pose/Depth Recovery Test Suite ===\n")

    tests = [
        ("Syntax checks", test_syntax_checks),
        ("Data structures", test_pose_depth_structures),
        ("Recovery nodes", test_recovery_node_structure),
        ("Export integration", test_export_node_integration),
        ("COLMAP availability", test_colmap_availability),
        ("OpenCV availability", test_opencv_availability),
        ("SciPy availability", test_scipy_availability),
        ("Mock recovery", test_mock_pose_recovery),
        ("Trajectory conversion", test_trajectory_conversion),
    ]

    passed = 0
    failed = 0
    warnings = 0

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            failed += 1

    print(f"\n=== Summary ===")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")

    if failed == 0:
        print("\n🎉 All pose recovery tests passed!")
        print("\nPose/Depth Recovery Features:")
        print("  • COLMAP integration for structure-from-motion")
        print("  • ViPE wrapper (when available)")
        print("  • Video frame extraction with OpenCV")
        print("  • Pose/depth recovery nodes for ComfyUI")
        print("  • Integration with dataset export pipeline")
        print("  • Fallback handling for missing dependencies")

        print("\nUsage:")
        print("  1. Gen3C_PoseDepth_FromVideo - recover from video files")
        print("  2. Gen3C_PoseDepth_FromImages - recover from image sequences")
        print("  3. Gen3C_VideoToDataset - complete video→dataset pipeline")

        return True
    else:
        print(f"\n❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)