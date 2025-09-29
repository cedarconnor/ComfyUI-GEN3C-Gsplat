# ComfyUI-GEN3C-Gsplat

A custom ComfyUI node pack that bridges Cosmos/GEN3C video generation with in-graph Gaussian Splat (3DGS) training. It adds camera/trajectory tooling, dataset exporters, and two training backends (Nerfstudio CLI wrapper and an in-process gsplat optimizer) so artists can go from prompt to splat entirely inside ComfyUI.

## Features
- **Camera authoring** – `Gen3C_CameraTrajectory` builds deterministic camera paths (orbit, dolly, truck, tilt, spiral, or custom keyframes) and returns per-frame intrinsics/extrinsics in a GEN3C-friendly payload.
- **Dataset export** – `Cosmos_Gen3C_InferExport` consumes RGB/depth sequences plus trajectory metadata and writes Nerfstudio-style datasets (`/rgb`, `/depth`, `transforms.json`). Optional metadata passthrough lets you include custom tags.
- **Gaussian Splat training**
  - `SplatTrainer_Nerfstudio`: spawns `ns-train splatfacto` followed by `ns-export gaussian-splat`, producing `.ply`/`.splat` assets from the dataset directory.
  - `SplatTrainer_gsplat`: runs a lightweight torch+gsplat optimisation loop in-process, saves an ASCII `.ply`, and is tuned for Windows (auto-adds the venv `Scripts` folder to `PATH`, traps missing `cl.exe` / `ninja`).
- **Depth-aware initialisation** – the gsplat trainer back-projects depth maps with the recorded camera matrices to seed the initial Gaussian cloud.

## Requirements
### Model assets
Place these alongside your existing ComfyUI models:
- `ComfyUI\models\GEN3C\GEN3C-Cosmos-7B.pt` (or equivalent).
- `ComfyUI\models\Lyra\lyra_static.pt` (Lyra VAE).
- `ComfyUI\models\Lyra\Cosmos-0.1-Tokenizer-CV8x16x16-autoencoder.jit` (Cosmos tokenizer/latent adapter).
- `ComfyUI\models\clip\clip_l.safetensors` (CLIP-L text encoder used by Cosmos).

### Python dependencies
Install into the same environment that runs ComfyUI:
```bash
# inside C:\ComfyUI
.\.venv\Scripts\pip install nerfstudio==0.3.4 gsplat==0.1.11 ninja
```
> **Windows note:** `gsplat` compiles CUDA extensions on first use. Install the Microsoft C++ Build Tools so `cl.exe` is on your `PATH`, or training will raise a helpful error.

## Installation
1. Drop this folder into `ComfyUI/custom_nodes/ComfyUI-GEN3C-Gsplat` (already done if you are reading this in-place).
2. Verify dependencies (`nerfstudio`, `gsplat`, `ninja`) are installed in the ComfyUI virtual environment.
3. Restart ComfyUI so it discovers the new nodes.

## Usage
1. **Author a camera path** – add `Gen3C_CameraTrajectory`, choose a preset, tweak FOV/near/far, and (optionally) paste JSON keyframes.
2. **Generate frames with Cosmos/GEN3C** – wire the trajectory into your Cosmos workflow (existing loader/inference nodes). Feed the resulting image/depth tensors into `Cosmos_Gen3C_InferExport` to write the dataset to disk.
3. **Train a splat**:
   - For CLI training, connect the dataset path to `SplatTrainer_Nerfstudio` and wait for the exporter to finish.
   - For in-process training, connect to `SplatTrainer_gsplat` and adjust iterations, learning rate, and batch sizes as needed.
4. **Preview/export** – open the resulting `.ply`/`.splat` with ComfyUI-3D-Pack viewers or any external Gaussian Splat viewer.

## Known limitations
- The Cosmos/GEN3C inference wrapper is still in progress; you must connect the camera payload to existing Cosmos nodes manually until the loader/exporter combo is finished.
- The gsplat trainer assumes every frame has a valid depth map. You can inject your own pose/depth recovery (e.g., ViPE) before training if needed.
- First-time gsplat runs on Windows can take a minute to build CUDA kernels.

## Roadmap
- Finish the duplicated Cosmos loader/inference node so GEN3C outputs feed directly into the exporter.
- Add dataset validators (pose continuity, depth sanity) and automated regression graphs.
- Surface a trajectory preview widget (frustum/path plot) and richer keyframe tooling.

## Support
Issues, suggestions, or contributions are welcome. Open a ticket or PR in your ComfyUI fork and tag with `GEN3C-Gsplat` so it’s easy to triage.
