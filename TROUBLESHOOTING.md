# ComfyUI-GEN3C-Gsplat Troubleshooting Guide

This guide covers common issues, installation problems, and workflow debugging for the ComfyUI-GEN3C-Gsplat node pack.

## 🚨 Installation Issues

### Missing Dependencies

**Problem**: `ImportError: No module named 'nerfstudio'` or similar dependency errors.

**Solution**:
```bash
# Install all required dependencies
pip install -r requirements.txt
```

**Problem**: Dependency conflict errors when installing requirements.txt.

**Common Error**: `ERROR: Cannot install opencv-python>=4.8.0 and nerfstudio==0.3.4 because these package versions have conflicting dependencies.`

**Solution**: This is resolved in the current requirements.txt. If you encounter this error:
```bash
# Install compatible versions
pip install nerfstudio==0.3.4 gsplat==0.1.11 ninja>=1.11 torch>=2.1
pip install opencv-python==4.6.0.66 scipy>=1.10.0 pillow>=9.0.0 matplotlib
```

**Verification**: Run the installation verification script:
```bash
python verify_installation.py
```

**Problem**: `ModuleNotFoundError: No module named 'folder_paths'` when running tests.

**Solution**: This is expected - tests should be run within ComfyUI context or use the provided standalone test scripts.

### Model Files Missing

**Problem**: "Model not found" errors when loading GEN3C models.

**Solution**: Verify model files are in correct locations:
- `ComfyUI/models/GEN3C/GEN3C-Cosmos-7B.pt`
- `ComfyUI/models/Lyra/lyra_static.pt`
- `ComfyUI/models/Lyra/Cosmos-0.1-Tokenizer-CV8x16x16-autoencoder.jit`
- `ComfyUI/models/clip/clip_l.safetensors`

### Windows Build Issues

**Problem**: `error: Microsoft Visual C++ 14.0 is required` when installing gsplat.

**Solution**:
1. Install Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Or install Visual Studio Community with C++ workload
3. Restart terminal and retry installation

**Problem**: `ninja: command not found` on Windows.

**Solution**:
```bash
pip install ninja
# Or use conda
conda install ninja
```

## 🎥 GEN3C Generation Issues

### Camera Trajectory Problems

**Problem**: Generated video has erratic camera movement or doesn't follow expected trajectory.

**Solutions**:
- Check trajectory preview before generation: use `Gen3C_TrajectoryPreview` node
- Verify trajectory parameters:
  - `frames`: Must match video generation settings
  - `fps`: Should be consistent across pipeline
  - `fov_degrees`: Reasonable range (30-90 degrees)
- For custom trajectories, validate keyframe format:
```json
{
  "keyframes": [
    {"frame": 0, "position": [0, 0, 5], "target": [0, 0, 0]},
    {"frame": 10, "position": [3, 1, 3], "target": [0, 0, 0]}
  ]
}
```

**Problem**: `Gen3C_CameraTrajectory` produces NaN values or infinite positions.

**Solutions**:
- Check orbit radius is positive (> 0)
- Ensure target point is different from camera position
- Verify angle parameters are in valid ranges
- Use trajectory quality analysis to identify issues

### Model Loading Failures

**Problem**: `CUDA out of memory` when loading GEN3C models.

**Solutions**:
- Reduce batch size to 1
- Use `fp16` precision instead of `fp32`
- Close other GPU applications
- Try smaller frame counts (16-24 frames)

**Problem**: `LyraModelLoader` fails with "incompatible model format".

**Solutions**:
- Verify model file integrity (re-download if necessary)
- Check model file paths are absolute, not relative
- Ensure models are compatible versions:
  - GEN3C-Cosmos-7B.pt (specific version for this integration)
  - Lyra models must match Cosmos tokenizer version

## 📹 Video Processing Issues

### Pose Recovery Failures

**Problem**: `COLMAP failed with exit code 1` or pose recovery returns empty results.

**Solutions**:
1. **Install COLMAP properly**:
   - Download from https://colmap.github.io/
   - Ensure `colmap` command is in PATH
   - Test with: `colmap --help`

2. **Video quality issues**:
   - Ensure sufficient camera motion (avoid static shots)
   - Check lighting conditions (avoid over/under-exposed frames)
   - Use higher resolution input videos (720p+ recommended)
   - Reduce `downsample_factor` for better feature detection

3. **Fallback options**:
   - Try `backend: "vipe"` instead of "colmap"
   - Use `backend: "auto"` for automatic fallback
   - Enable `estimate_depth: false` if depth estimation fails

**Problem**: Low confidence scores from pose recovery.

**Solutions**:
- Check trajectory quality analysis output
- Ensure video has sufficient parallax/baseline
- Avoid scenes with too much repetitive texture
- Try different camera motion patterns (orbit works well)

### Video Format Issues

**Problem**: `Gen3C_VideoToDataset` fails to read video file.

**Solutions**:
- Use standard formats: MP4, AVI, MOV
- Check video codec compatibility (H.264 recommended)
- Verify file path is correct and accessible
- Try re-encoding video with ffmpeg:
```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4
```

## 🔍 Quality Validation Issues

### Dataset Validation Failures

**Problem**: `Gen3C_DatasetValidator` reports "FAILED" status.

**Solutions**:
1. **Check dataset structure**:
   ```
   dataset/
   ├── images/
   │   ├── frame_000.jpg
   │   └── ...
   └── transforms.json
   ```

2. **Verify transforms.json format**:
   - Must contain valid camera matrices
   - Frame references must match image files
   - Intrinsics should be reasonable (focal length, center point)

3. **Image quality issues**:
   - Check for corrupted image files
   - Ensure consistent image dimensions
   - Verify image format (JPG/PNG)

**Problem**: Quality scores are consistently low.

**Solutions**:
- Review quality filter thresholds:
  - `quality_threshold`: Try lowering from 0.6 to 0.4
  - `min_blur_threshold`: Adjust based on content type
- Check input data quality:
  - Lighting conditions
  - Camera focus
  - Motion blur
- Use `Gen3C_TrajectoryQualityAnalysis` to identify specific issues

### Quality Filter Removing Too Many Frames

**Problem**: `Gen3C_QualityFilter` removes most frames, leaving insufficient data.

**Solutions**:
- Lower quality thresholds:
  - `quality_threshold`: 0.4 → 0.3
  - `min_blur_threshold`: 0.4 → 0.3
  - Adjust brightness range: `[0.1, 0.9]`
- Review filter reports to understand removal reasons
- Improve input data quality before filtering
- Consider manual frame selection for difficult cases

## 🧠 Training Issues

### Gaussian Splat Training Failures

**Problem**: `SplatTrainer_gsplat` crashes or produces poor results.

**Solutions**:
1. **Memory issues**:
   - Reduce batch size to 1
   - Lower image resolution in dataset
   - Reduce number of training iterations initially

2. **Training parameters**:
   - Start with proven settings:
     - `iterations`: 3000-5000
     - `learning_rate`: 0.01
     - `batch_size`: 1
   - Enable depth loss if depth data available
   - Use spherical harmonics (`enable_sh: true`) for better appearance

3. **Dataset issues**:
   - Ensure minimum 10-15 frames after quality filtering
   - Check pose quality with trajectory analysis
   - Verify camera intrinsics are reasonable

**Problem**: Training appears to run but produces empty/corrupted .ply file.

**Solutions**:
- Check training logs for errors
- Verify output directory permissions
- Ensure sufficient disk space
- Try lower complexity scenes first

### Nerfstudio Integration Issues

**Problem**: `SplatTrainer_Nerfstudio` fails to find ns-train command.

**Solutions**:
- Verify nerfstudio installation: `ns-train --help`
- Check PATH includes nerfstudio scripts
- Try absolute path to ns-train executable
- Ensure nerfstudio and dependencies are compatible versions

## 🖥️ Platform-Specific Issues

### Windows Issues

**Problem**: Path separators causing file not found errors.

**Solutions**:
- Use forward slashes in JSON paths: `"path/to/file"`
- Or use double backslashes: `"path\\\\to\\\\file"`
- Avoid spaces in paths when possible

**Problem**: Unicode characters in terminal output causing crashes.

**Solutions**:
- Set terminal encoding: `chcp 65001`
- Use Windows Terminal instead of Command Prompt
- The node pack handles this automatically in most cases

### Linux/macOS Issues

**Problem**: Permission errors when creating output directories.

**Solutions**:
- Check write permissions on output directories
- Use absolute paths instead of relative paths
- Ensure ComfyUI has proper file system access

## 📊 Performance Optimization

### Slow Processing

**Problem**: Workflow takes very long to complete.

**Solutions**:
1. **Reduce data size**:
   - Lower frame counts (16-24 frames)
   - Reduce image resolution (`downsample_factor: 0.5`)
   - Use quality filtering to remove poor frames early

2. **GPU optimization**:
   - Enable GPU acceleration where available
   - Use mixed precision (`fp16`)
   - Close other GPU applications

3. **Parallel processing**:
   - Process videos in chunks
   - Use batch processing where supported

### Memory Usage

**Problem**: Running out of system RAM or VRAM.

**Solutions**:
- Enable batch processing for large datasets
- Use streaming processing for video files
- Reduce concurrent operations
- Monitor memory usage with system tools

## 🔧 Debugging Workflow Issues

### Node Connection Problems

**Problem**: Workflow fails to execute or produces unexpected results.

**Solutions**:
1. **Check node connections**:
   - Verify input/output types match
   - Ensure all required inputs are connected
   - Check for circular dependencies

2. **Validate intermediate outputs**:
   - Use preview nodes to inspect data flow
   - Check trajectory visualizations
   - Review quality analysis reports

3. **Error messages**:
   - Read full error messages in ComfyUI console
   - Check individual node outputs
   - Use validation nodes to identify issues early

### Workflow JSON Issues

**Problem**: Custom workflows fail to load or execute properly.

**Solutions**:
- Validate JSON syntax with online tools
- Check node IDs are unique
- Verify all referenced nodes exist
- Test with provided example workflows first

## 📞 Getting Help

### Diagnostic Information

When reporting issues, include:
- ComfyUI version and platform (Windows/Linux/macOS)
- Python version and environment details
- Full error messages and stack traces
- Input data characteristics (video format, resolution, etc.)
- Node parameter settings
- System specifications (GPU, RAM)

### Common Log Locations

- ComfyUI console output
- `ComfyUI/temp/` directory for temporary files
- Training logs in output directories
- System GPU/memory monitoring tools

### Test Commands

Quick diagnostic tests:
```bash
# Test basic imports
python -c "import torch; print(torch.cuda.is_available())"
python -c "import nerfstudio; print('Nerfstudio OK')"

# Test COLMAP
colmap --help

# Test model files
ls ComfyUI/models/GEN3C/
ls ComfyUI/models/Lyra/
```

## 🎯 Best Practices

### Data Quality

- Use well-lit, sharp input videos
- Ensure sufficient camera motion for pose recovery
- Avoid scenes with too much repetitive texture
- Test with simple scenes before complex ones

### Workflow Design

- Start with provided example workflows
- Use quality validation early in pipeline
- Enable quality filtering before training
- Save intermediate results for debugging

### Performance

- Begin with small datasets and iterate
- Use quality gates to fail fast on poor data
- Monitor resource usage during processing
- Keep backups of working configurations

---

**Need more help?** Check the main README.md for additional resources and community support options.