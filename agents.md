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
- ✅ `Gen3C_CameraTrajectory` — presets + custom keyframes, returns structured traj JSON.
- ✅ `Cosmos_Gen3C_InferExport` — writes Nerfstudio-style datasets and optional metadata.
- ✅ `SplatTrainer_Nerfstudio` — CLI wrapper for `ns-train` + `ns-export`.
- ✅ `SplatTrainer_gsplat` — depth-initialised optimiser built on `gsplat`.
- ⏳ `Cosmos_Gen3C_Loader` + inference wrapper — pending duplication of native Cosmos loader so we can inject trajectories and capture depth/frames directly.
- ⏳ `Gen3C_PoseDepth_FromVideo` — placeholder stub for ViPE/pose recovery integration.
- 🔭 Optional: trajectory preview, dataset validator, quality filters.

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
- Camera maths is self-contained (look-at, interpolation) and outputs both camera→world and world→camera transforms for downstream projections.
- gsplat trainer initialises Gaussians from depth reprojection, optimises means/log-scales/quaternions/colour/opacity with Adam, and emits an ASCII `.ply` ready for Comfy3D.
- Windows quirks: ensure `ninja` and `cl.exe` exist; trainer surfaces clear errors if toolchains are missing and automatically prepends the venv `Scripts` path.

## 7. Dev Plan & Milestones
| Milestone | Status | Notes |
|-----------|--------|-------|
| **P0** Camera → Dataset bridge | ✅ | Trajectory node + dataset exporter live. Cosmos inference duplication still in progress. |
| **P1** gsplat trainer | ✅ | In-process trainer implemented; handles Windows path/toolchain edge cases. |
| **P2** Depth-aware init | ✅ | gsplat trainer seeds from depth maps via recorded transforms. |
| **P3** Quality & UX | ⏳ | Pending: trajectory preview, dataset validation, richer presets. |

## 8. Testing & Validation
- Unit coverage pending (trajectory maths, JSON writer, depth handling). Path forward: add simple pytest cases or inline smoke tests.
- Manual smoke tests: run 10-frame orbit → export dataset → gsplat trainer (GPU required). Compare outputs with Nerfstudio runs for sanity.
- Future: regression graph stored under `user/default/workflows/` for quick replays.

## 9. Requirements Snapshot
- Models: GEN3C 7B weights, Lyra VAE, Cosmos tokenizer, CLIP-L.
- Packages: `nerfstudio`, `gsplat`, `ninja` (build tools), PyTorch with CUDA.
- Optional: ViPE/pose recovery backend for datasets without explicit trajectories.

## 10. Next Actions
1. Duplicate Cosmos loader/inference into this repo, inject trajectory payloads, and capture RGB+depth outputs directly.
2. Wire optional pose/depth recovery node into the dataset writer.
3. Author dataset validation + preview utilities (frustum plot, depth stats, axis sanity check).
4. Add automated tests / sample workflows and document recommended configs in the README.

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
