# GEN3C ➜ Gaussian Splat — Design Notes

## 0. Purpose & Scope
- Provide ComfyUI-native tooling that drives Cosmos/GEN3C with explicit camera control and persists RGB + depth + pose data to disk.
- Train Gaussian Splat scenes (PLY/3DGS) from that dataset inside the same graph. We now ship both a Nerfstudio wrapper and an in-process gsplat trainer.
- Keep everything Windows-friendly: local code duplication instead of patching core ComfyUI, no external shell scripts, and graceful handling of missing build tools.

## 1. Background
- **GEN3C / Cosmos** – controllable video diffusion with a 3D cache; ComfyUI already exposes official Cosmos nodes we can reuse.
- **Depth/pseudo-SfM** – ViPE, COLMAP, or similar pipelines can recover poses/depth when GEN3C trajectories are unavailable.
- **Gaussian Splatting** – Nerfstudio Splatfacto, gsplat (CUDA rasteriser), Graphdeco reference code.
- **ComfyUI-3D-Pack** – existing viewers/exporters for `.ply` / `.splat` outputs; we lean on them instead of building a viewer.

## 2. User Story
> “As a ComfyUI artist/TD, I want to generate a controllable GEN3C sequence, save RGB+depth+poses to a dataset folder, train a Gaussian splat in the same graph, preview it, and export a `.ply` without leaving ComfyUI.”

## 3. Architecture
```
[Prompt / Inputs]
        │
┌───────▼───────────┐    trajectory payload (K,R,t per frame)
│ Gen3C Camera Tool │──────────────┐
└───────▲───────────┘              │
        │                           ▼
        │                 ┌──────────────────────────┐
        │                 │ Cosmos / GEN3C Inference │   (TBD: duplicated loader)
        │                 └──────────────┬───────────┘
        │                                │ RGB / depth tensors
        ▼                                ▼
┌────────────────────────┐     ┌────────────────────────────┐
│ Cosmos_Gen3C_InferExport│────► dataset dir (/rgb,/depth,  │
└──────────┬──────────────┘     │ transforms.json)           │
           │                    └──────────┬────────────────┘
           │                                 │
           │                         ┌───────▼────────────────┐
           │                         │ 3DGS Trainers           │
           │                         │  • Nerfstudio (CLI)     │
           │                         │  • gsplat (in-process)  │
           ▼                         └──────────┬─────────────┘
  dataset path / previews                      │
                                               ▼
                                     Preview / export (Comfy3D)
```

## 4. Node Inventory & Status
- [done] `Gen3C_CameraTrajectory` — presets + custom keyframes, returns structured traj JSON.
- [done] `Cosmos_Gen3C_InferExport` — writes Nerfstudio-style datasets with derived intrinsics/depth metadata.
- [done] `Cosmos_Gen3C_DirectExport` — enhanced exporter that extracts trajectory from Cosmos latents directly.
- [done] `SplatTrainer_Nerfstudio` — CLI wrapper for `ns-train` + `ns-export`.
- [done] `SplatTrainer_gsplat` — depth-initialised optimiser built on `gsplat`, now falls back to SfM seeding when depth maps are absent.
- [done] `Gen3CDiffusion` — complete Cosmos inference with trajectory injection and camera conditioning.
- [done] `sample_cosmos` — enhanced sampling function with trajectory encoding and camera control.
- [done] `CosmosGen3CLatentVideo` + `CosmosGen3CImageToVideoLatent` — duplicated Cosmos nodes with embedded trajectory support.
- [done] `Gen3C_PoseDepth_FromVideo` + `Gen3C_PoseDepth_FromImages` — complete pose/depth recovery using COLMAP and ViPE.
- [done] `Gen3C_VideoToDataset` — end-to-end video→dataset pipeline with pose recovery.
- [nice-to-have] Optional: trajectory preview, dataset validator, quality filters.

## 5. Dataset Schema
Default layout (Nerfstudio `transforms.json`):
```json
{
  "camera_model": "OPENCV",
  "fl_x": 1200.0,
  "fl_y": 1200.0,
  "cx": 512.0,
  "cy": 512.0,
  "w": 1024,
  "h": 1024,
  "fps": 24,
  "frames": [
    {
      "file_path": "rgb/frame_000001.png",
      "depth_path": "depth/frame_000001.npy",
      "transform_matrix": [[...],[...],[...],[0,0,0,1]]
    }
  ]
}
```
Depth formats supported: `npy`, `png16`, `pfm` (configurable in the exporter).

## 6. Implementation Notes
- We vendor native Cosmos nodes into `comfy_gen3c/duplicated/` to avoid altering core ComfyUI.
- Camera maths is self-contained (look-at, interpolation) and outputs both camera->world and world->camera transforms for downstream projections.
- Dataset exporter now derives Nerfstudio-compatible metadata (intrinsics, frame paths, optional depth) directly from the trajectory payload.
- gsplat trainer initialises Gaussians from depth reprojection when available and falls back to a synthetic SfM-style point cloud otherwise, optimising means/log-scales/quaternions/colour/opacity with Adam before emitting an ASCII `.ply` ready for Comfy3D.
- Windows quirks: ensure `ninja` and `cl.exe` exist; trainer surfaces clear errors if toolchains are missing and automatically prepends the venv `Scripts` path.

## 7. Dev Plan & Milestones
| Milestone | Status | Notes |
|-----------|--------|-------|
| **P0** Camera -> Dataset bridge | Done | Trajectory node + dataset exporter live; exporter now emits Nerfstudio metadata automatically. |
| **P1** gsplat trainer | Done | In-process trainer implemented; handles Windows path/toolchain edge cases and now tolerates missing depth via fallback seeding. |
| **P2** Depth-aware init | Done | Depth reprojection seeds gsplat when provided; synthetic fallback keeps training usable without depth. |
| **P3** Cosmos Integration | Done | Complete trajectory injection into Cosmos inference with camera conditioning, direct export from latents, enhanced nodes with embedded trajectory support. |
| **P4** Quality & UX | In progress | Pending: trajectory preview, dataset validation, richer presets. |

## 8. Testing & Validation
- Unit coverage pending (trajectory maths, JSON writer, depth handling). Path forward: add simple pytest cases or inline smoke tests.
- Manual smoke tests: run 10-frame orbit -> export dataset -> gsplat trainer (GPU required). Compare outputs with Nerfstudio runs for sanity.
- Future: regression graph stored under `user/default/workflows/` for quick replays.

## 9. Requirements Snapshot
- Models: GEN3C 7B weights, Lyra VAE, Cosmos tokenizer, CLIP-L.
- Packages: `nerfstudio`, `gsplat`, `ninja` (build tools), PyTorch with CUDA.
- Supporting file: `requirements.txt` lists the pinned Python dependencies for this node pack.
- Optional: ViPE/pose recovery backend for datasets without explicit trajectories.

## 10. Next Actions
1. ~~Duplicate Cosmos loader/inference into this repo, inject trajectory payloads, and capture RGB+depth outputs directly.~~ ✅ **COMPLETED**
2. ~~Wire optional pose/depth recovery node into the dataset writer.~~ ✅ **COMPLETED**
3. Author dataset validation + preview utilities (frustum plot, depth stats, axis sanity check).
4. Add automated tests / sample workflows and document recommended configs in the README.

### Completed Integration (P3 Milestone)
The Cosmos integration is now **complete** with the following enhancements:

- **Enhanced `sample_cosmos` function** (`comfy_gen3c/gen3c/sampler.py`):
  - Full trajectory conditioning support with camera transforms and intrinsics encoding
  - Integrates trajectory data directly into Cosmos model conditioning pipeline
  - Supports depth extraction hooks for future depth-aware workflows

- **Updated `Gen3CDiffusion` node** (`comfy_gen3c/gen3c/diffusion.py`):
  - Now uses enhanced `sample_cosmos` with trajectory injection
  - Seamless camera control through trajectory payloads
  - Maintains compatibility with existing ComfyUI sampling infrastructure

- **New trajectory-aware Cosmos nodes** (`comfy_gen3c/duplicated/nodes_cosmos.py`):
  - `CosmosGen3CLatentVideo` and `CosmosGen3CImageToVideoLatent` embed trajectory data in latent dicts
  - Automatic dimension extraction from trajectory metadata
  - Backward compatible with original Cosmos node behavior

- **Enhanced export workflow** (`comfy_gen3c/export_nodes.py`):
  - New `Cosmos_Gen3C_DirectExport` node extracts trajectory from Cosmos latents automatically
  - Eliminates need for separate trajectory wiring in complex workflows
  - Maintains original export functionality for backward compatibility

This enables **end-to-end trajectory control**: `Gen3C_CameraTrajectory` → `Gen3CDiffusion` → `Cosmos_Gen3C_DirectExport` → dataset → splat training, all within a single ComfyUI graph.

### Completed Pose/Depth Recovery (P4 Milestone)
The pose/depth recovery system is now **complete** with the following features:

- **Comprehensive pose recovery backends** (`comfy_gen3c/dataset/pose_depth.py`):
  - COLMAP integration for classical structure-from-motion
  - ViPE wrapper for video-specific pose estimation (when available)
  - Automatic fallback between backends with confidence scoring
  - Support for both video files and image sequences

- **ComfyUI pose recovery nodes** (`comfy_gen3c/dataset/recovery_nodes.py`):
  - `Gen3C_PoseDepth_FromVideo` - recovers poses/depth from video files
  - `Gen3C_PoseDepth_FromImages` - recovers poses/depth from image sequences
  - Configurable backends, frame limits, and quality settings
  - Returns trajectory data compatible with existing GEN3C pipeline

- **Integrated dataset export** (`comfy_gen3c/export_nodes.py`):
  - `Gen3C_VideoToDataset` - complete video→dataset pipeline with pose recovery
  - Automatically extracts frames, runs SfM, and exports Nerfstudio-compatible datasets
  - Optional external depth map integration
  - Confidence scoring and error handling

This enables **video-to-splat workflows** without explicit camera control: `Video File` → `Gen3C_VideoToDataset` → `SplatTrainer_gsplat` → splat output, handling pose recovery automatically.

## 11. Appendix — transforms.json Helper
```python
from pathlib import Path
import json
import numpy as np

def write_transforms(path: Path, frames, intrinsics):
    payload = {
        "camera_model": "OPENCV",
        **intrinsics,
        "frames": frames,
    }
    path.write_text(json.dumps(payload, indent=2))
```


