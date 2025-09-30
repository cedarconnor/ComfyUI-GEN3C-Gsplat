# Dependency Conflict Resolution Guide

This guide helps resolve dependency conflicts when installing ComfyUI-GEN3C-Gsplat in environments with existing packages.

## Common Conflicts

### 1. urllib3 Version Conflict

**Error**: `matrix-client 0.4.0 requires urllib3~=1.21, but you have urllib3 2.5.0`

**Solution Options**:

**Option A: Update matrix-client (Recommended)**
```bash
pip install --upgrade matrix-client
```

**Option B: Pin urllib3 if matrix-client upgrade fails**
```bash
pip install "urllib3>=1.21,<3.0"
```

### 2. protobuf Version Conflict

**Error**: `mediapipe 0.10.21 requires protobuf<5,>=4.25.3, but you have protobuf 3.20.3`

**Solution**:
```bash
# Update protobuf to compatible version
pip install "protobuf>=4.25.3,<5.0"
```

**Note**: Some older packages may break with protobuf 4+. If you encounter issues, you may need to downgrade mediapipe:
```bash
pip install "mediapipe<0.10"
```

### 3. timm Version Conflict

**Error**: `open-clip-torch 3.2.0 requires timm>=1.0.17, but you have timm 0.6.7`

**Solution**:
```bash
# Update timm to latest version
pip install "timm>=1.0.17"
```

## Complete Resolution Strategy

### Method 1: Direct Installation (Recommended)

The most reliable approach is to install our requirements directly and let pip resolve conflicts:

```bash
# Install our requirements - let nerfstudio dictate compatible versions
pip install -r requirements.txt
```

**Result**: This approach works because:
- Nerfstudio specifies its exact requirements (`timm==0.6.7`, `protobuf<=3.20.3`)
- These override any conflicting versions from other packages
- The installation succeeds with functional core dependencies

### Method 2: Selective Updates (If needed)

If direct installation fails, try updating packages one by one:

```bash
# Step 1: Update timm (most likely to be safe)
pip install --upgrade timm

# Step 2: Update protobuf carefully
pip install "protobuf>=4.25.3,<5.0"

# Step 3: Update urllib3 and matrix-client
pip install --upgrade matrix-client urllib3

# Step 4: Install our requirements
pip install -r requirements.txt
```

### Method 3: Force Compatible Versions

If selective updates don't work, force compatible versions:

```bash
# Install compatible versions explicitly
pip install "urllib3>=1.26,<3.0" "protobuf>=4.25.3,<5.0" "timm>=1.0.17"

# Then install our requirements
pip install -r requirements.txt
```

### Method 4: Clean Environment (Nuclear Option)

If conflicts persist, create a clean environment:

```bash
# Create new conda environment
conda create -n comfyui-gen3c python=3.11
conda activate comfyui-gen3c

# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install our requirements
pip install -r requirements.txt

# Install additional ComfyUI dependencies as needed
```

## Package-Specific Notes

### matrix-client
- Often not used in ComfyUI workflows
- Can be safely uninstalled if not needed: `pip uninstall matrix-client`

### mediapipe
- Used for some face/pose detection nodes
- Consider downgrading if protobuf 4+ causes issues: `pip install "mediapipe<0.10"`

### open-clip-torch & transparent-background
- Modern packages that benefit from newer timm
- Updating timm is usually safe and improves performance

### timm (Torch Image Models)
- Core package for many vision models
- Newer versions have better performance and bug fixes
- Safe to upgrade in most cases

## Verification After Resolution

After resolving conflicts, verify the installation:

```bash
# Check for remaining conflicts (warnings are usually OK)
pip check

# Verify our package works
python verify_installation.py

# Test core functionality
python tests/test_core_logic.py

# Test ComfyUI startup (if applicable)
python main.py --cpu  # or your usual ComfyUI start command
```

## Understanding pip check Warnings

After successful installation, you may see warnings like:
```
grpcio-status 1.71.2 has requirement protobuf<6.0dev,>=5.26.1, but you have protobuf 3.20.3.
matrix-client 0.4.0 has requirement urllib3~=1.21, but you have urllib3 2.5.0.
```

**These are usually safe to ignore** if:
- The verification script passes
- Core functionality tests pass
- ComfyUI starts without errors
- Your workflows run successfully

The warnings indicate version mismatches but don't necessarily break functionality.

## Prevention Strategies

### 1. Use Virtual Environments
Always use conda or venv environments for different projects:
```bash
conda create -n comfyui python=3.11
conda activate comfyui
```

### 2. Pin Critical Versions
Create a `constraints.txt` file with known-good versions:
```
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
opencv-python==4.6.0.66
```

Then install with:
```bash
pip install -r requirements.txt -c constraints.txt
```

### 3. Regular Maintenance
Periodically clean up unused packages:
```bash
pip list --not-required  # Show packages not required by others
pip autoremove  # Remove orphaned packages (if available)
```

## Advanced Troubleshooting

### Check Dependency Tree
```bash
pip install pipdeptree
pipdeptree --packages matrix-client,mediapipe,open-clip-torch,transparent-background
```

### Force Reinstall Problematic Packages
```bash
pip install --force-reinstall --no-deps <package-name>
```

### Ignore Dependency Checks (Use with Caution)
```bash
pip install --no-deps -r requirements.txt
```

## If All Else Fails

1. **Document your current working state**:
   ```bash
   pip freeze > working_environment.txt
   ```

2. **Create a minimal test environment**:
   ```bash
   conda create -n test-gen3c python=3.11
   conda activate test-gen3c
   pip install -r requirements.txt
   ```

3. **Gradually add other packages** you need for your workflows

4. **Report specific conflicts** as GitHub issues with:
   - Full error output
   - `pip freeze` output
   - Your use case and required packages

Remember: Dependency conflicts are normal in complex Python environments. The key is to resolve them systematically and maintain clean environments for different projects.