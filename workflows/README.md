# GEN3C Workflow Examples

This directory contains example workflows demonstrating different use cases for the ComfyUI-GEN3C-Gsplat node pack.

## Workflow Overview

### 1. Basic GEN3C to Splat (`basic_gen3c_to_splat.json`)

**Purpose**: Complete pipeline from text prompt to Gaussian splat using explicit camera control.

**Key Features**:
- Camera trajectory generation with presets
- GEN3C video generation with trajectory injection
- Trajectory visualization and validation
- Direct export to Nerfstudio format
- Direct-to-trainer memory workflow via the `dataset` output
- Gaussian splat training

**Best For**:
- Controlled video generation
- High-quality results with known camera paths
- Content creation workflows

**Requirements**:
- GEN3C model weights
- GPU with sufficient VRAM
- Basic ComfyUI setup

---

### 2. Video to Splat (`video_to_splat.json`)

**Purpose**: Convert existing videos to Gaussian splats using automatic pose recovery.

**Key Features**:
- Automatic camera pose estimation (COLMAP/ViPE)
- Trajectory quality analysis
- Frame quality filtering
- Pose recovery confidence scoring
- Exports transforms with the requested FPS and recovered frame resolution

**Best For**:
- Converting existing footage
- Real-world video processing
- When camera parameters are unknown

**Requirements**:
- COLMAP installation (recommended)
- OpenCV for video processing
- Input video with sufficient camera motion

---

### 3. Quality Control Pipeline (`quality_control_pipeline.json`)

**Purpose**: Comprehensive quality assessment and filtering for datasets before training.

**Key Features**:
- Dataset structure validation
- Trajectory quality metrics
- Frame-by-frame quality assessment
- Automated quality gates
- Comprehensive reporting

**Best For**:
- Pre-training validation
- Dataset optimization
- Quality assurance workflows
- Troubleshooting training issues

**Requirements**:
- Existing dataset in Nerfstudio format
- Sufficient disk space for reports

---

## Usage Instructions

### Loading Workflows

1. Copy the desired JSON file to your ComfyUI workflows directory
2. Open ComfyUI and load the workflow via "Load Workflow"
3. Adjust node parameters according to your specific needs
4. Ensure all required model files and dependencies are available
5. For the basic GEN3C pipeline, choose whether `Cosmos_Gen3C_DirectExport` feeds `SplatTrainer_gsplat` via
   the in-memory `dataset` socket (no intermediate files) or `dataset_dir` when you want the dataset on disk

### Common Parameters to Adjust

**For GEN3C Generation**:
- `prompt`: Your text description
- `frames`: Number of frames to generate
- `resolution`: Output resolution (width/height)
- `trajectory_preset`: Camera motion type

**For Video Processing**:
- `video_path`: Path to input video file
- `max_frames`: Limit number of frames processed
- `downsample_factor`: Reduce resolution for faster processing
- `fps`: Target frame rate to embed in exported transforms

**For Quality Control**:
- `quality_threshold`: Minimum frame quality score
- `min_frames`: Minimum frames required for training

### Output Locations

- Datasets: `${output_dir}/dataset_name/`
- Gaussian splats: `${output_dir}/splat_output.ply`
- Quality reports: `${output_dir}/quality_analysis/`
- Trajectory previews: `${output_dir}/trajectory_preview/`
- In-memory datasets: available directly from the `dataset` socket on export nodes when `write_to_disk` is disabled

## Troubleshooting

### Common Issues

**"Model not found" errors**:
- Verify GEN3C model files are in correct locations
- Check `models/GEN3C/`, `models/Lyra/` directories

**Pose recovery failures**:
- Install COLMAP: https://colmap.github.io/
- Ensure sufficient camera motion in input video
- Try reducing `downsample_factor` for better feature detection

**Low quality scores**:
- Check lighting conditions in input data
- Adjust quality thresholds in filter nodes
- Review trajectory smoothness and coverage

**Training failures**:
- Ensure minimum 10-15 frames after filtering
- Check dataset validation reports
- Verify GPU memory availability

### Performance Tips

1. **For faster processing**:
   - Reduce frame count and resolution
   - Use `downsample_factor` < 1.0
   - Enable quality filtering to remove poor frames

2. **For better quality**:
   - Increase trajectory diversity
   - Use higher resolution inputs
   - Ensure good lighting conditions

3. **For large datasets**:
   - Enable batch processing where available
   - Use quality gates to fail fast on poor data
   - Process in chunks if memory limited

## Extending Workflows

### Adding Custom Nodes

You can extend these workflows by:
1. Adding preprocessing nodes (image enhancement, etc.)
2. Integrating external depth estimation
3. Adding post-processing (compression, format conversion)
4. Including evaluation metrics

### Parameter Automation

Consider adding:
- Automatic parameter tuning based on input analysis
- Multi-resolution processing pipelines
- Batch processing for multiple inputs
- Integration with external tools

## Advanced Usage

### Custom Camera Trajectories

For complex camera paths:
1. Use the "custom" preset in trajectory generation
2. Provide keyframes in JSON format
3. Mix multiple trajectory types
4. Add manual trajectory editing nodes

### Multi-Stage Quality Control

For production workflows:
1. Run initial quality assessment
2. Apply targeted improvements (denoising, etc.)
3. Re-validate with strict thresholds
4. Generate detailed quality reports

### Integration with External Tools

These workflows can be extended to work with:
- DepthCrafter for depth estimation
- COLMAP for detailed SfM reconstruction
- External video processing tools
- Cloud-based rendering services

For more advanced customizations, refer to the node documentation and implementation details in the source code.
