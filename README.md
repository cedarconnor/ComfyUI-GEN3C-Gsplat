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

#### 2. **Video-to-Splat Pipeline**
```
Video File → Gen3C_VideoToDataset → Gen3C_QualityFilter →
SplatTrainer_gsplat → Output.ply
```

#### 3. **Quality Control Workflow**
```
Dataset → Gen3C_DatasetValidator → Gen3C_TrajectoryPreview →
Gen3C_QualityFilter → Training
```

## 📖 Node Reference

### Camera & Trajectory
- **`Gen3C_CameraTrajectory`** – Generate camera paths with presets or custom keyframes
- **`Gen3C_TrajectoryPreview`** – Visualize camera trajectories (3D plots, frustums, stats)
- **`Gen3C_TrajectoryQualityAnalysis`** – Analyze trajectory smoothness, coverage, diversity

### GEN3C Integration
- **`LyraModelLoader`** – Load GEN3C/Cosmos model bundle
- **`Gen3CDiffusion`** – Generate video with trajectory control and camera conditioning
- **`CosmosGen3CLatentVideo`** – Enhanced Cosmos nodes with trajectory support

### Pose Recovery
- **`Gen3C_PoseDepth_FromVideo`** – Recover poses from video using COLMAP/ViPE
- **`Gen3C_PoseDepth_FromImages`** – Recover poses from image sequences
- **`Gen3C_VideoToDataset`** – Complete video→dataset pipeline with pose recovery

### Dataset Export & Validation
- **`Cosmos_Gen3C_InferExport`** – Export RGB + trajectory to Nerfstudio format
- **`Cosmos_Gen3C_DirectExport`** – Extract trajectory from latents automatically
- **`Gen3C_DatasetValidator`** – Comprehensive dataset quality validation
- **`Gen3C_QualityFilter`** – Filter low-quality frames with blur/brightness detection

### Training
- **`SplatTrainer_Nerfstudio`** – CLI wrapper for ns-train splatfacto pipeline
- **`SplatTrainer_gsplat`** – In-process training with depth initialization

## 🎯 Workflow Examples

Explore the `workflows/` directory for complete examples:

### **Basic GEN3C to Splat** (`workflows/basic_gen3c_to_splat.json`)
Complete pipeline from text prompt to Gaussian splat with explicit camera control.

### **Video to Splat** (`workflows/video_to_splat.json`)
Convert existing videos to splats using automatic pose recovery.

### **Quality Control Pipeline** (`workflows/quality_control_pipeline.json`)
Comprehensive quality assessment and filtering before training.

## 🔧 Advanced Configuration

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

