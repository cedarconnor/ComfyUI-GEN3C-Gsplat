# ComfyUI-GEN3C-Gsplat

A comprehensive ComfyUI node pack that bridges Cosmos/GEN3C video generation with Gaussian Splat (3DGS) training, featuring camera control, pose recovery, quality validation, and end-to-end pipelines from prompt to splat.

## ✨ Key Features

### 🎥 **Complete GEN3C Pipeline**
- **Camera Control** – `Gen3C_CameraTrajectory` with presets (orbit, dolly, truck, tilt, spiral) and custom keyframes
- **Cosmos Integration** – Full trajectory injection into GEN3C diffusion with enhanced sampling
- **Direct Export** – `Cosmos_Gen3C_DirectExport` extracts trajectory from latents automatically
- **Enhanced Nodes** – Trajectory-aware Cosmos nodes with embedded camera data

### 🔄 **Pose Recovery System**
- **Multiple Backends** – COLMAP (classical SfM), ViPE (video-specific), automatic fallback
- **Video Processing** – `Gen3C_VideoToDataset` for complete video→dataset pipeline
- **Image Sequences** – `Gen3C_PoseDepth_FromImages` for photo collections
- **Quality Scoring** – Confidence metrics and error reporting

### 📊 **Quality & Validation Tools**
- **Dataset Validation** – Comprehensive structure, pose, and image quality checks
- **Trajectory Analysis** – Smoothness, coverage, baseline quality metrics
- **Quality Filtering** – Automatic removal of low-quality frames
- **Visual Previews** – 3D trajectory plots, frustum visualization, statistics

### 🧠 **Gaussian Splat Training**
- **Nerfstudio Integration** – `SplatTrainer_Nerfstudio` CLI wrapper with full pipeline
- **In-Process Training** – `SplatTrainer_gsplat` with depth initialization and fallback SfM
- **Quality Optimization** – Smart initialization from pose recovery or depth maps
- **Windows Compatible** – Handles build tools and path issues gracefully

### 🔍 **Advanced Workflows**
- **End-to-End Control** – `Gen3C_CameraTrajectory` → `Gen3CDiffusion` → `Export` → `Training`
- **Video-to-Splat** – `Video` → `PoseRecovery` → `QualityFilter` → `Training`
- **Quality Pipelines** – Validation → Filtering → Preview → Training with quality gates

## 📋 Requirements

### Core Dependencies
```bash
# Install into your ComfyUI environment
pip install nerfstudio==0.3.4 gsplat==0.1.11 ninja>=1.11 torch>=2.1

# For pose recovery
pip install opencv-python==4.6.0.66 scipy>=1.10.0 pillow>=9.0.0
```

### Model Assets
Place these alongside your existing ComfyUI models:
- `ComfyUI/models/GEN3C/GEN3C-Cosmos-7B.pt` (GEN3C diffusion model)
- `ComfyUI/models/Lyra/lyra_static.pt` (Lyra VAE encoder)
- `ComfyUI/models/Lyra/Cosmos-0.1-Tokenizer-CV8x16x16-autoencoder.jit` (Cosmos tokenizer)
- `ComfyUI/models/clip/clip_l.safetensors` (CLIP-L text encoder)

### Optional Dependencies

**For COLMAP pose recovery** (recommended):
- Download and install from [colmap.github.io](https://colmap.github.io/)
- Ensure `colmap` command is available in PATH

**For DepthCrafter integration**:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes.git
```

**For advanced visualization** (matplotlib for 3D plots):
```bash
pip install matplotlib
```

### System Requirements
- **GPU**: CUDA-capable GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ system RAM for large datasets
- **Storage**: 5-10GB for model weights, additional space for datasets
- **OS**: Windows 10/11, Linux, macOS (Windows build tools required for gsplat)

## 🚀 Quick Start

### Installation
1. Clone or download this repository to `ComfyUI/custom_nodes/ComfyUI-GEN3C-Gsplat`
2. Install core dependencies: `pip install -r requirements.txt`
3. Download required model weights (see Requirements section)
4. Restart ComfyUI to discover new nodes

### Basic Usage

#### 1. **Controlled GEN3C Generation**
```
Gen3C_CameraTrajectory → LyraModelLoader → Gen3CDiffusion →
Cosmos_Gen3C_DirectExport → SplatTrainer_gsplat → Output.ply
```
`Cosmos_Gen3C_DirectExport` now recovers the injected camera path directly from `Gen3CDiffusion` latents.
Route the in-memory `dataset` output straight into `SplatTrainer_gsplat.dataset` for a zero-disk training loop,
or keep `write_to_disk=True` when you want reusable assets.

#### 2. **Video-to-Splat Pipeline**
```
Video File → Gen3C_VideoToDataset → Gen3C_QualityFilter →
SplatTrainer_gsplat → Output.ply
```
`Gen3C_VideoToDataset` copies the requested FPS into `transforms.json` and matches the recovered pose metadata
to the actual extracted frame resolution, so downstream trainers see true timing and intrinsics.

#### 3. **Quality Control Workflow**
```
Dataset → Gen3C_DatasetValidator → Gen3C_TrajectoryPreview →
Gen3C_QualityFilter → Training
```

## 📖 Node Reference & Wiring Guide

### 🎬 Camera & Trajectory Nodes

#### `Gen3C_CameraTrajectory`
Generate camera paths with built-in presets or custom keyframes.

**Inputs:**
- `frames` (INT): Total number of frames
- `fps` (INT): Frames per second
- `width`/`height` (INT): Frame dimensions (must be multiples of 8)
- `fov_degrees` (FLOAT): Horizontal field of view
- `preset` (ENUM): orbit, dolly, truck, tilt, spiral, or custom
- `keyframes_json` (STRING, optional): Custom trajectory as JSON

**Outputs:**
- `trajectory` (GEN3C_TRAJECTORY): Camera trajectory data
- `trajectory_json` (STRING): JSON representation

**Wiring:** Connect `trajectory` → `Gen3CDiffusion.camera_trajectory` or export nodes

---

#### `Gen3C_TrajectoryPreview`
Visualize camera trajectories (3D plots, frustums, stats).

**Inputs:**
- `trajectory` (GEN3C_TRAJECTORY)
- `plot_type` (ENUM): 3d, frustums, stats, or all

**Outputs:**
- `preview_image` (IMAGE): Rendered visualization
- `output_path` (STRING): Saved image path

---

### 🧪 GEN3C Diffusion Nodes

#### `Gen3CDiffusion`
Generate camera-controlled videos using GEN3C/Cosmos.

**Inputs:**
- `lyra_model` (LYRA_MODEL): From LyraModelLoader
- `camera_trajectory` (GEN3C_TRAJECTORY): From Gen3C_CameraTrajectory
- `num_inference_steps` (INT): 50-150 recommended
- `guidance_scale` (FLOAT): 7.5 typical
- `prompt`/`negative_prompt` (STRING): Text conditioning
- `seed` (INT): Random seed

**Outputs:**
- `latents` (LATENT): Cosmos latent representation with embedded `camera_trajectory`
- `images` (IMAGE): Decoded RGB frames
- `cameras` (GEN3C_TRAJECTORY): Passthrough trajectory

**Wiring:** `Gen3C_CameraTrajectory` → `Gen3CDiffusion` → `Cosmos_Gen3C_DirectExport`

---

### 📦 Dataset Export Nodes

#### `Cosmos_Gen3C_InferExport`
Export GEN3C inference outputs to Nerfstudio-compatible dataset.

**Inputs:**
- `images` (IMAGE): RGB frames from Gen3CDiffusion
- `trajectory` (GEN3C_TRAJECTORY): Camera trajectory
- `output_dir` (STRING): Save location
- **`write_to_disk` (BOOLEAN): Toggle disk writing vs memory passing** ⭐
- `depth_maps` (IMAGE, optional): Depth maps
- `metadata_json` (STRING, optional): Extra metadata

**Outputs:**
- `dataset_dir` (STRING): Nerfstudio dataset path (empty string when `write_to_disk=False`)
- `dataset` (GEN3C_DATASET): In-memory RGB/depth/trajectory package for direct trainer input

**Wiring:**
- **Disk-based:** `write_to_disk=True` → use `dataset_dir` → `SplatTrainer_gsplat.dataset_dir`
- **Memory-based:** `write_to_disk=False` → use `dataset` → `SplatTrainer_gsplat.dataset`

---

#### `Cosmos_Gen3C_DirectExport`
Export with automatic trajectory extraction from latents generated by `Gen3CDiffusion`.

**Inputs:**
- `images` (IMAGE): Decoded frames
- `latents` (LATENT): Cosmos latents with embedded trajectory
- `output_dir` (STRING)
- `write_to_disk` (BOOLEAN)
- `depth_maps` (IMAGE, optional)
- `trajectory_override` (GEN3C_TRAJECTORY, optional)

**Outputs:**
- `dataset_dir` (STRING): Export path when `write_to_disk=True`
- `trajectory` (GEN3C_TRAJECTORY): Extracted trajectory payload
- `dataset` (GEN3C_DATASET): In-memory dataset synchronized with disk output

---

#### `Gen3C_VideoToDataset`
Complete video→dataset pipeline with pose recovery. The node records the chosen FPS in `transforms.json`
and aligns pose intrinsics with the actual extracted frame size.

**Inputs:**
- `video_path` (STRING): Input video file
- `max_frames` (INT): Frame extraction limit
- `backend` (ENUM): auto, colmap, vipe
- `write_to_disk` (BOOLEAN)
- `estimate_depth` (BOOLEAN)

**Outputs:**
- `dataset_dir` (STRING): Export directory
- `trajectory` (GEN3C_TRAJECTORY): Pose track matching the extracted resolution
- `confidence` (FLOAT): SfM quality score
- `status` (STRING): Recovery status
- `dataset` (GEN3C_DATASET): In-memory dataset for direct training

---

### 🔄 Pose Recovery Nodes

#### `Gen3C_PoseDepth_FromVideo`
Recover camera poses from video using COLMAP/ViPE.

**Inputs:**
- `video_path` (STRING)
- `max_frames` (INT): 30-100 recommended
- `backend` (ENUM): auto tries ViPE→COLMAP
- `downsample_factor` (FLOAT): 0.5 = half resolution

**Outputs:**
- `trajectory` (GEN3C_TRAJECTORY)
- `images` (IMAGE): Extracted frames
- `confidence` (FLOAT)
- `status` (STRING)

**Wiring:** → `Gen3C_QualityFilter` → export nodes → trainer

---

#### `Gen3C_PoseDepth_FromImages`
Recover poses from image sequence.

**Inputs:**
- `images` (IMAGE): Image sequence tensor
- `backend` (ENUM): colmap recommended
- `matcher_type` (ENUM): exhaustive (accurate) or sequential (fast)

**Outputs:**
- `trajectory` (GEN3C_TRAJECTORY)
- `confidence` (FLOAT)
- `status` (STRING)

---

### 🎯 Training Nodes

#### `SplatTrainer_gsplat`
In-process Gaussian Splat trainer with depth initialization.

**Inputs:**
- `output_dir` (STRING): PLY output location
- `run_name` (STRING): Training run name
- `max_iterations` (INT): 1000-7000 typical
- `learning_rate` (FLOAT): 0.005 default
- `points_per_frame` (INT): 50000 default
- `frames_per_batch` (INT): Frame count per optimisation step
- `depth_scale` (FLOAT): Depth unit scaling (set to 1.0 for GEN3C depth)
- `device` (ENUM): auto, cuda, cpu
- `dataset_dir` (STRING, optional): Disk-based input
- `dataset` (GEN3C_DATASET, optional): Memory-based input

**Outputs:**
- `ply_path` (STRING): Trained Gaussian splat file

**Wiring:**
- **Disk workflow:** Export node `dataset_dir` → this `dataset_dir`
- **Memory workflow:** Export node `dataset` → this `dataset` (no file I/O)

---

#### `SplatTrainer_Nerfstudio`
CLI wrapper for Nerfstudio splatfacto pipeline.

**Inputs:**
- `dataset_dir` (STRING): **Disk-based only** (requires ns-train)
- `max_iterations` (INT): 30000 recommended
- `skip_training` (BOOLEAN): Re-export only
- `export_after_train` (BOOLEAN)

**Outputs:**
- `run_dir` (STRING): Training workspace
- `export_dir` (STRING): Exported PLY location

---

### ✅ Validation & Quality Nodes

#### `Gen3C_DatasetValidator`
Comprehensive dataset quality validation.

**Inputs:**
- `dataset_path` (STRING)
- `min_frames`/`max_frames` (INT)
- `generate_report` (BOOLEAN)

**Outputs:**
- `quality_score` (FLOAT): 0-1 score
- `validation_status` (STRING)
- `issues` (STRING): Problem list
- `stats` (STRING): JSON statistics

---

#### `Gen3C_QualityFilter`
Filter low-quality frames from datasets.

**Inputs:**
- `dataset_path` (STRING)
- `trajectory` (GEN3C_TRAJECTORY)
- `quality_threshold` (FLOAT): 0.4 default
- `min_blur_threshold` (FLOAT)
- `min/max_brightness` (FLOAT)

**Outputs:**
- `filtered_trajectory` (GEN3C_TRAJECTORY): Cleaned trajectory
- `filter_report` (STRING)
- `frames_kept`/`frames_removed` (INT)

---

#### `Gen3C_TrajectoryQualityAnalysis`
Analyze trajectory quality metrics.

**Inputs:**
- `trajectory` (GEN3C_TRAJECTORY)

**Outputs:**
- `overall_score` (FLOAT)
- `smoothness`/`coverage`/`baseline_quality`/`rotation_diversity` (FLOAT)
- `analysis_report` (STRING)

## 🎯 Complete Workflow Examples

### **Example 1: Memory-Based GEN3C → Splat** ⚡ (RECOMMENDED - Fast!)

**No file I/O between export and training!**

```
┌─────────────────────────────┐
│ Gen3C_CameraTrajectory      │
│ - preset: orbit             │
│ - frames: 121               │
└──────────┬──────────────────┘
           │ trajectory
           ▼
┌─────────────────────────────┐
│ LyraModelLoader             │
└──────────┬──────────────────┘
           │ lyra_model
           ▼
┌─────────────────────────────┐
│ Gen3CDiffusion              │
│ - prompt: "flying dragon"   │
│ - num_inference_steps: 50   │
└──────────┬──────────────────┘
           │ latents, images
           ▼
┌─────────────────────────────┐
│ Cosmos_Gen3C_DirectExport   │
│ - write_to_disk: FALSE ⭐   │  <-- KEY: Disable disk writing
└──────────┬──────────────────┘
           │ dataset (in-memory)
           ▼
┌─────────────────────────────┐
│ SplatTrainer_gsplat         │
│ - max_iterations: 3000      │
│ - Connect 'dataset' input ⭐│  <-- Direct memory connection
└──────────┬──────────────────┘
           │ ply_path
           ▼
       point_cloud.ply
```

**Advantages:**
- ⚡ **Faster** - No file I/O overhead
- 💾 **Less disk space** - No intermediate files
- 🔄 **Cleaner** - No dataset folders to manage

---

### **Example 2: Disk-Based GEN3C → Splat** (Traditional)

**For dataset archiving or external processing.**

```
┌─────────────────────────────┐
│ Gen3C_CameraTrajectory      │
└──────────┬──────────────────┘
           │ trajectory
           ▼
┌─────────────────────────────┐
│ Gen3CDiffusion              │
└──────────┬──────────────────┘
           │ images, trajectory
           ▼
┌─────────────────────────────┐
│ Cosmos_Gen3C_InferExport    │
│ - write_to_disk: TRUE       │
│ - output_dir: ./datasets/   │
└──────────┬──────────────────┘
           │ dataset_dir (path)
           ▼
┌─────────────────────────────┐
│ SplatTrainer_gsplat         │
│ - Connect 'dataset_dir' ⭐  │  <-- Use path input
│ - max_iterations: 3000      │
└──────────┬──────────────────┘
           │ ply_path
           ▼
       point_cloud.ply

Files written to disk:
./datasets/my_dataset/
├── rgb/
│   ├── frame_000000.png
│   └── ...
├── depth/ (if available)
│   └── frame_000000.npy
└── transforms.json
```

---

### **Example 3: Video → Splat with Pose Recovery**

**Convert existing videos to splats.**

```
┌─────────────────────────────┐
│ Input: video.mp4            │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Gen3C_VideoToDataset        │
│ - video_path: video.mp4     │
│ - max_frames: 50            │
│ - backend: auto             │
│ - write_to_disk: FALSE ⭐   │  <-- Memory workflow
└──────────┬──────────────────┘
           │ dataset, trajectory, confidence
           ▼
┌─────────────────────────────┐
│ Gen3C_TrajectoryQualityAnalysis │ (optional)
└──────────┬──────────────────┘
           │ overall_score
           ▼
┌─────────────────────────────┐
│ SplatTrainer_gsplat         │
│ - Connect 'dataset' input   │
└──────────┬──────────────────┘
           │ ply_path
           ▼
       point_cloud.ply
```

---

### **Example 4: Quality-Controlled Pipeline**

**With validation and filtering.**

```
┌─────────────────────────────┐
│ Gen3C_CameraTrajectory      │
│ + Gen3CDiffusion            │
└──────────┬──────────────────┘
           │ images, trajectory
           ▼
┌─────────────────────────────┐
│ Cosmos_Gen3C_InferExport    │
│ - write_to_disk: TRUE       │  <-- Need disk for validator
└──────────┬──────────────────┘
           │ dataset_dir, trajectory
           ├──────────────────────┐
           │                      │
           ▼                      ▼
┌──────────────────────┐  ┌─────────────────────┐
│ Gen3C_DatasetValidator│  │ Gen3C_QualityFilter │
│ - min_frames: 10     │  │ - quality_threshold │
└──────────┬───────────┘  └──────────┬──────────┘
           │                         │ filtered_trajectory
           │ quality_score           │
           │                         ▼
           │              ┌─────────────────────┐
           │              │ Cosmos_Gen3C_InferExport │ (re-export filtered)
           │              │ - write_to_disk: FALSE ⭐│
           │              └──────────┬──────────┘
           │                         │ dataset
           └─────────────────────────┤
                                     ▼
                          ┌─────────────────────┐
                          │ SplatTrainer_gsplat │
                          └──────────┬──────────┘
                                     ▼
                                point_cloud.ply
```

---

### **Example 5: Depth-Enhanced Training**

**Using DepthCrafter for better initialization.**

```
┌─────────────────────────────┐
│ Gen3CDiffusion              │
└──────────┬──────────────────┘
           │ images, latents, trajectory
           ├──────────────┐
           │              │
           ▼              ▼
┌──────────────────┐  ┌─────────────────────┐
│ (images bypass)  │  │ DepthCrafter Node   │
│                  │  │ - images → depth    │
└──────┬───────────┘  └──────────┬──────────┘
       │                         │ depth_maps
       │                         │
       └─────────┬───────────────┘
                 ▼
┌─────────────────────────────┐
│ Cosmos_Gen3C_DirectExport   │
│ - images                    │
│ - latents (w/ trajectory)   │
│ - depth_maps ⭐             │
│ - write_to_disk: FALSE      │
└──────────┬──────────────────┘
           │ dataset (w/ depth)
           ▼
┌─────────────────────────────┐
│ SplatTrainer_gsplat         │
│ - depth_scale: 1.0          │
│ - max_iterations: 5000      │
└──────────┬──────────────────┘
           ▼
       point_cloud.ply
```

---

## 💡 Wiring Best Practices

### When to Use Memory vs Disk Workflows

**Use Memory Workflow (write_to_disk=FALSE)** when:
- ✅ Training immediately after generation
- ✅ Iterating/experimenting rapidly
- ✅ Limited disk space
- ✅ No need to archive datasets

**Use Disk Workflow (write_to_disk=TRUE)** when:
- ✅ Need to reuse dataset multiple times
- ✅ Sharing datasets with others
- ✅ Using external tools (Nerfstudio CLI, COLMAP GUI)
- ✅ Dataset validation/inspection required
- ✅ Training with `SplatTrainer_Nerfstudio` (CLI-based)

### Input/Output Compatibility Matrix

| Export Node Output | → | Trainer Input | Compatible? |
|-------------------|---|---------------|-------------|
| `dataset_dir` (STRING) | → | `SplatTrainer_gsplat.dataset_dir` | ✅ Yes |
| `dataset` (GEN3C_DATASET) | → | `SplatTrainer_gsplat.dataset` | ✅ Yes ⚡ |
| `dataset` (GEN3C_DATASET) | → | `SplatTrainer_Nerfstudio.dataset_dir` | ❌ No (CLI needs disk) |
| `dataset_dir` (STRING) | → | `SplatTrainer_Nerfstudio.dataset_dir` | ✅ Yes |

### Common Connection Patterns

```
# Pattern 1: Direct trajectory control
Gen3C_CameraTrajectory → Gen3CDiffusion
                       → Cosmos_Gen3C_InferExport (use trajectory)

# Pattern 2: Automatic trajectory extraction
Gen3CDiffusion → Cosmos_Gen3C_DirectExport (extracts from latents)

# Pattern 3: Trajectory override
Gen3C_CameraTrajectory ──┐
                         ├→ Cosmos_Gen3C_DirectExport.trajectory_override
Gen3CDiffusion ──────────┘

# Pattern 4: Quality-controlled export
Gen3C_PoseDepth_FromVideo → Gen3C_QualityFilter → Gen3C_VideoToDataset
```

## 🔧 Advanced Configuration

### Camera Motion Presets

**14 Built-in Cinematic Presets:**

1. **orbit** - Circular motion around target
   - Parameters: `orbit_radius`, `orbit_height`, `orbit_turns`
   - Use for: 360° product shots, scene reveals

2. **dolly_in** - Push in toward target
   - Parameters: `dolly_start`, `dolly_end`
   - Use for: Subject reveals, dramatic entrances

3. **dolly_out** - Pull out from target
   - Parameters: `dolly_start`, `dolly_end`
   - Use for: Scene reveals, context establishment

4. **truck_left** - Lateral movement from right to left
   - Parameters: `truck_span`, `truck_depth`
   - Use for: Parallax effects, environment reveal

5. **truck_right** - Lateral movement from left to right
   - Parameters: `truck_span`, `truck_depth`
   - Use for: Parallax effects, environment reveal

6. **crane_up** - Vertical lift upward
   - Parameters: `crane_start_height`, `crane_end_height`, `orbit_radius`
   - Use for: Epic reveals, establishing shots

7. **crane_down** - Vertical descent downward
   - Parameters: `crane_start_height`, `crane_end_height`, `orbit_radius`
   - Use for: Intimate moments, detail focus

8. **arc_left** - Partial orbit from front to left
   - Parameters: `orbit_radius`, `arc_degrees` (default 90°)
   - Use for: Subject reveals with rotation

9. **arc_right** - Partial orbit from front to right
   - Parameters: `orbit_radius`, `arc_degrees` (default 90°)
   - Use for: Subject reveals with rotation

10. **tilt** - Vertical tilt motion
    - Parameters: `orbit_radius`, `tilt_degrees`
    - Use for: Height emphasis, scale demonstration

11. **spiral** - Spiral inward/outward with height variation
    - Parameters: `spiral_start`, `spiral_end`, `orbit_turns`
    - Use for: Dynamic reveals, complex motion

12. **boom_shot** - Combined dolly + crane (smooth curved motion)
    - Parameters: `boom_start_radius`, `boom_end_radius`, `boom_start_height`, `boom_end_height`
    - Use for: Cinematic reveals, dramatic transitions

13. **figure_eight** - Figure-8 pattern around target
    - Parameters: `orbit_radius`, `orbit_turns` (loops)
    - Use for: Dynamic product shots, artistic motion

14. **hemisphere** - Orbit with varying elevation
    - Parameters: `orbit_radius`, `hemisphere_elevation`
    - Use for: Complete coverage, multi-angle capture

15. **custom** - User-defined keyframe trajectory
    - See custom keyframes section below

### Custom Camera Trajectories
```json
{
  "keyframes": [
    {"frame": 0, "position": [0, 0, 5], "target": [0, 0, 0]},
    {"frame": 10, "position": [3, 1, 3], "target": [0, 0, 0]},
    {"frame": 20, "position": [0, 2, -5], "target": [0, 0, 0]}
  ]
}
```

### Preset Quick Reference

| Preset | Motion Type | Best For | Key Parameters |
|--------|-------------|----------|----------------|
| orbit | Circular | 360° views | radius, turns |
| dolly_in/out | Linear depth | Reveals | start, end distance |
| truck_left/right | Lateral | Parallax | span, depth |
| crane_up/down | Vertical | Scale/drama | start/end height |
| arc_left/right | Partial orbit | Reveals with rotation | radius, arc angle |
| spiral | Helical | Complex motion | start/end radius, turns |
| boom_shot | Curved 3D | Cinematic | all 4 boom params |
| figure_eight | Lissajous | Dynamic product shots | radius, loops |
| hemisphere | Spherical | Complete coverage | radius, elevation |
| tilt | Angular tilt | Height emphasis | radius, degrees |

### Quality Filter Settings
- **Blur threshold**: 0.3-0.7 (higher = more strict)
- **Brightness range**: 0.15-0.85 (normalized 0-1)
- **Overall quality**: 0.4-0.8 (combined score threshold)

### Training Parameters
- **Iterations**: 3000-7000 (more for complex scenes)
- **Learning rate**: 0.005-0.02 (lower for stable convergence)
- **Batch size**: 1-4 (limited by VRAM)

## 🐛 Known Limitations

- **Depth dependency**: Full depth integration requires DepthCrafter or external depth estimation
- **Windows builds**: First-time gsplat compilation requires Microsoft C++ Build Tools
- **Memory usage**: Large datasets may require batch processing or downsampling
- **COLMAP dependency**: Pose recovery requires separate COLMAP installation

## 🗺️ Roadmap

### Completed ✅
- Complete Cosmos integration with trajectory injection
- Pose/depth recovery system (COLMAP, ViPE)
- Quality validation and filtering framework
- Comprehensive dataset export pipeline

### In Progress 🚧
- Advanced trajectory preview widgets
- Automated parameter tuning
- Cloud-based processing integration

### Planned 📋
- Real-time trajectory editing interface
- Multi-resolution training pipelines
- Advanced quality metrics (LPIPS, etc.)
- Integration with other 3D formats

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **Testing**: Try workflows with different datasets and report issues
- **Documentation**: Improve tutorials and troubleshooting guides
- **Features**: Implement new quality metrics or training optimizations
- **Integration**: Connect with other ComfyUI node packs

## 📄 License & Support

This project is open source. For issues, suggestions, or contributions:

1. **Issues**: Report bugs or request features via GitHub Issues
2. **Discussions**: Join the ComfyUI community for general questions
3. **Pull Requests**: Submit improvements or bug fixes

**Commercial Support**: For enterprise deployments or custom development, contact the maintainers.

---

**🎨 Happy splat creation!** Transform your ideas into immersive 3D experiences with the power of GEN3C and Gaussian Splatting.

