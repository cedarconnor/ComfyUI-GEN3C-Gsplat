#!/usr/bin/env python3
"""Installation verification script for ComfyUI-GEN3C-Gsplat.

Run this script to verify that all dependencies are correctly installed.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor < 8:
        print("  ERROR: Python 3.8+ required")
        return False
    else:
        print("  OK: Python version compatible")
        return True

def check_required_packages():
    """Check if required packages are installed."""
    print("\nChecking required packages...")

    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("scipy", "SciPy"),
    ]

    optional_packages = [
        ("nerfstudio", "Nerfstudio"),
        ("gsplat", "GSplat"),
        ("matplotlib", "Matplotlib"),
    ]

    all_good = True

    # Check required packages
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  OK: {package_name} installed")
        except ImportError:
            print(f"  ERROR: {package_name} not found")
            all_good = False

    # Check optional packages
    for module_name, package_name in optional_packages:
        try:
            __import__(module_name)
            print(f"  OK: {package_name} installed")
        except ImportError:
            print(f"  WARNING: {package_name} not found (optional)")

    return all_good

def check_model_files():
    """Check if required model files exist."""
    print("\nChecking model files...")

    # Try to find ComfyUI directory
    possible_paths = [
        Path("../../models"),  # From custom_nodes/ComfyUI-GEN3C-Gsplat
        Path("../../../models"),  # Alternative structure
        Path("C:/ComfyUI/models"),  # Common Windows path
        Path(Path.home() / "ComfyUI/models"),  # User directory
    ]

    models_dir = None
    for path in possible_paths:
        if path.exists():
            models_dir = path
            break

    if not models_dir:
        print("  WARNING: Could not locate ComfyUI models directory")
        print("  Please ensure model files are in correct locations:")
        print("    - ComfyUI/models/GEN3C/GEN3C-Cosmos-7B.pt")
        print("    - ComfyUI/models/Lyra/lyra_static.pt")
        print("    - ComfyUI/models/Lyra/Cosmos-0.1-Tokenizer-CV8x16x16-autoencoder.jit")
        print("    - ComfyUI/models/clip/clip_l.safetensors")
        return False

    print(f"  Found models directory: {models_dir}")

    required_models = [
        ("GEN3C", "GEN3C-Cosmos-7B.pt"),
        ("Lyra", "lyra_static.pt"),
        ("Lyra", "Cosmos-0.1-Tokenizer-CV8x16x16-autoencoder.jit"),
        ("clip", "clip_l.safetensors"),
    ]

    all_found = True
    for subdir, filename in required_models:
        model_path = models_dir / subdir / filename
        if model_path.exists():
            print(f"  OK: {subdir}/{filename}")
        else:
            print(f"  MISSING: {subdir}/{filename}")
            all_found = False

    return all_found

def check_external_tools():
    """Check external tools availability."""
    print("\nChecking external tools...")

    # Check COLMAP
    try:
        result = subprocess.run(
            ["colmap", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("  OK: COLMAP available")
        else:
            print("  WARNING: COLMAP found but may not be working correctly")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  WARNING: COLMAP not found (pose recovery will use fallback)")

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  OK: GPU available - {gpu_name} (and {gpu_count-1} others)" if gpu_count > 1 else f"  OK: GPU available - {gpu_name}")
        else:
            print("  WARNING: No GPU detected (CPU-only mode)")
    except ImportError:
        print("  ERROR: Cannot check GPU - PyTorch not installed")

def test_core_functionality():
    """Test basic functionality of core modules."""
    print("\nTesting core functionality...")

    # Test numpy operations
    try:
        import numpy as np
        test_array = np.random.rand(10, 10)
        result = np.mean(test_array)
        print(f"  OK: NumPy operations working (test mean: {result:.3f})")
    except Exception as e:
        print(f"  ERROR: NumPy test failed: {e}")
        return False

    # Test OpenCV
    try:
        import cv2
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        blurred = cv2.GaussianBlur(test_image, (5, 5), 0)
        print("  OK: OpenCV operations working")
    except Exception as e:
        print(f"  ERROR: OpenCV test failed: {e}")
        return False

    # Test PyTorch if available
    try:
        import torch
        test_tensor = torch.randn(3, 3)
        result = torch.mean(test_tensor)
        print(f"  OK: PyTorch operations working (test mean: {result:.3f})")
    except Exception as e:
        print(f"  WARNING: PyTorch test failed: {e}")

    return True

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("ComfyUI-GEN3C-Gsplat Installation Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Model Files", check_model_files),
        ("External Tools", check_external_tools),
        ("Core Functionality", test_core_functionality),
    ]

    results = []
    for name, check_func in checks:
        print()
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ERROR: Check failed with exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    critical_checks = ["Python Version", "Required Packages", "Core Functionality"]
    critical_passed = 0
    total_passed = 0

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<20}: {status}")

        if passed:
            total_passed += 1
            if name in critical_checks:
                critical_passed += 1

    print()

    if critical_passed == len(critical_checks):
        print("OK INSTALLATION READY: Core functionality verified")
        if total_passed == len(results):
            print("OK ALL CHECKS PASSED: Complete installation verified")
            return 0
        else:
            print("WARNING: Some optional components missing - basic functionality available")
            return 0
    else:
        print("ERROR: INSTALLATION INCOMPLETE: Critical issues found")
        print("\nRecommended actions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Download required model files")
        print("3. Install COLMAP for pose recovery")
        return 1

if __name__ == "__main__":
    sys.exit(main())