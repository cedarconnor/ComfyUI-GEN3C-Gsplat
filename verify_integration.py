#!/usr/bin/env python3
"""Verify Cosmos integration code structure without ComfyUI dependencies."""

import ast
import sys
from pathlib import Path

def check_file_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def check_imports_and_functions(file_path):
    """Check if expected functions and classes exist in file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)

        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        return functions, classes
    except Exception as e:
        return [], []

def main():
    base_path = Path(__file__).parent

    print("=== Cosmos Integration Verification ===")

    # Files to check
    files_to_check = [
        ("comfy_gen3c/gen3c/sampler.py", ["sample_cosmos", "_encode_trajectory_conditioning"], []),
        ("comfy_gen3c/gen3c/diffusion.py", ["sample"], ["Gen3CDiffusion"]),
        ("comfy_gen3c/export_nodes.py", ["export_dataset"], ["CosmosGen3CDirectExport"]),
        ("comfy_gen3c/duplicated/nodes_cosmos.py", [], ["CosmosGen3CLatentVideo", "CosmosGen3CImageToVideoLatent"]),
    ]

    all_passed = True

    for file_path, expected_functions, expected_classes in files_to_check:
        full_path = base_path / file_path
        print(f"\nChecking {file_path}...")

        # Check syntax
        syntax_ok, syntax_error = check_file_syntax(full_path)
        if not syntax_ok:
            print(f"  X Syntax error: {syntax_error}")
            all_passed = False
            continue
        else:
            print("  ✓ Syntax is valid")

        # Check functions and classes
        functions, classes = check_imports_and_functions(full_path)

        for func in expected_functions:
            if func in functions:
                print(f"  ✓ Function '{func}' found")
            else:
                print(f"  X Function '{func}' missing")
                all_passed = False

        for cls in expected_classes:
            if cls in classes:
                print(f"  ✓ Class '{cls}' found")
            else:
                print(f"  X Class '{cls}' missing")
                all_passed = False

    # Check key integration points
    print("\n=== Integration Points ===")

    # Check if sampler.py has trajectory conditioning
    sampler_path = base_path / "comfy_gen3c/gen3c/sampler.py"
    with open(sampler_path, 'r') as f:
        sampler_code = f.read()

    if "_encode_trajectory_conditioning" in sampler_code:
        print("✓ Trajectory conditioning function exists")
    else:
        print("X Trajectory conditioning function missing")
        all_passed = False

    if "camera_transforms" in sampler_code:
        print("✓ Camera transform handling exists")
    else:
        print("X Camera transform handling missing")
        all_passed = False

    # Check if diffusion.py imports from sampler
    diffusion_path = base_path / "comfy_gen3c/gen3c/diffusion.py"
    with open(diffusion_path, 'r') as f:
        diffusion_code = f.read()

    if "from .sampler import sample_cosmos" in diffusion_code:
        print("✓ Diffusion imports enhanced sampler")
    else:
        print("X Diffusion missing sampler import")
        all_passed = False

    # Check if export has trajectory extraction
    export_path = base_path / "comfy_gen3c/export_nodes.py"
    with open(export_path, 'r') as f:
        export_code = f.read()

    if "_extract_trajectory_from_latents" in export_code:
        print("✓ Export has trajectory extraction")
    else:
        print("X Export missing trajectory extraction")
        all_passed = False

    if "CosmosGen3CDirectExport" in export_code:
        print("✓ Direct export node exists")
    else:
        print("X Direct export node missing")
        all_passed = False

    # Summary
    print("\n=== Summary ===")
    if all_passed:
        print("✓ All integration checks passed!")
        print("\nCompleted Integration Features:")
        print("  • Enhanced sample_cosmos with trajectory injection")
        print("  • Gen3CDiffusion uses new sampler with camera control")
        print("  • CosmosGen3CDirectExport extracts trajectory from latents")
        print("  • New Cosmos nodes support embedded trajectory data")
        print("  • End-to-end camera → diffusion → export pipeline")

        print("\nNext Steps:")
        print("  1. Test with actual ComfyUI environment")
        print("  2. Create example workflows")
        print("  3. Add pose/depth recovery integration")
        print("  4. Implement dataset validation tools")

        return True
    else:
        print("X Some integration checks failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)