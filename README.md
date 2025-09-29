# ComfyUI-GEN3C-Gsplat

A custom ComfyUI node pack that bridges Cosmos/GEN3C video generation with in-graph Gaussian Splat (3DGS) training. It adds camera/trajectory tooling, dataset exporters, and two training backends (Nerfstudio CLI wrapper and an in-process gsplat optimizer) so artists can go from prompt to splat entirely inside ComfyUI.

## Features
- **Camera authoring** – `Gen3C_CameraTrajectory` builds deterministic camera paths (orbit, dolly, truck, tilt, spiral, or custom keyframes) and returns per-frame intrinsics/extrinsics in a GEN3C-friendly payload.
- **Dataset export** – `Cosmos_Gen3C_InferExport` consumes RGB image sequences plus trajectory metadata and writes Nerfstudio-style datasets (`/rgb`, optional `/depth`, `transforms.json`). Pass an external depth tensor (e.g., DepthCrafter output) into the `depth_maps` input to get depth file paths baked into `transforms.json`.
- **Gaussian Splat training**
  - `SplatTrainer_Nerfstudio`: spawns `ns-train splatfacto` followed by `ns-export gaussian-splat`, producing `.ply`/`.splat` assets from a dataset directory.
  - `SplatTrainer_gsplat`: depth-initialised optimiser built on `gsplat`; it consumes depth maps from `Cosmos_Gen3C_InferExport` when available and falls back to a synthetic SfM-style initialisation when depth is missing before writing an ASCII `.ply`.
- **Depth integration (optional)** – Install [ComfyUI-DepthCrafter-Nodes](https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes) and drop its `DownloadAndLoadDepthCrafterModel`->`DepthCrafter` nodes after `Gen3CDiffusion`. Connect the DepthCrafter output to the exporter’s `depth_maps` socket; the exporter writes `depth/frame_XXXXXX.npy` and updates `transforms.json` so the gsplat trainer can consume metric depth.

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
> **Windows note:** `gsplat` compiles CUDA extensions on first use. Install the Microsoft C++ Build Tools so `cl.exe` is on your PATH, or training will raise a helpful error.

For DepthCrafter depth generation:
```bash
cd C:\ComfyUI\custom_nodes
git clone https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes.git
# restart ComfyUI so the new nodes register
```
The DepthCrafter nodes manage Hugging Face downloads automatically the first time you run them (several gigabytes).

## Installation
1. Drop this folder into `ComfyUI/custom_nodes/ComfyUI-GEN3C-Gsplat` (already done if you are reading this in-place).
2. Verify dependencies (`nerfstudio`, `gsplat`, `ninja`; optional DepthCrafter) are installed in the ComfyUI virtual environment.
3. Restart ComfyUI so it discovers the new nodes.

## Usage
1. **Author a camera path** – add `Gen3C_CameraTrajectory`, choose a preset, tweak FOV/near/far, and (optionally) paste JSON keyframes.
2. **Generate frames with Cosmos/GEN3C** – load the model bundle via `LyraModelLoader`, feed `Gen3CDiffusion` with prompt + trajectory, and capture the output images/latents.
3. **(Optional) Create depth with DepthCrafter** – if you installed the extension, run:
   - `DownloadAndLoadDepthCrafterModel` -> `DepthCrafter`
   - Wire the `images` output from `Gen3CDiffusion` into `DepthCrafter` and pipe its result to the exporter’s `depth_maps` input.
4. **Export dataset** – `Cosmos_Gen3C_InferExport` writes `/rgb`, optional `/depth`, and `transforms.json`. Any metadata JSON you feed in is merged with the trajectory payload.
5. **Train a splat**:
   - For CLI training, connect the dataset path to `SplatTrainer_Nerfstudio` and wait for the exporter to finish.
   - For in-process training, connect the dataset path to `SplatTrainer_gsplat` and adjust iterations, learning rate, and batch sizes as needed.
6. **Preview/export** – open the resulting `.ply`/`.splat` with ComfyUI-3D-Pack viewers or any external Gaussian Splat viewer.

### Minimal graph outline
```
Gen3C_CameraTrajectory
   |-- LyraModelLoader -> Gen3CDiffusion --|-- Cosmos_Gen3C_InferExport -> dataset path
                                         |-- DepthCrafter (optional depth)
SplatTrainer_gsplat or SplatTrainer_Nerfstudio
```

## Known limitations
- Depth export relies on the DepthCrafter extension (or any other compatible depth node you provide). Without depth, the exporter still writes RGB+poses and the gsplat trainer now seeds from an SfM-style fallback cloud (slower to converge and lower fidelity than true depth).
- The Cosmos diffusion wrapper currently emits RGB and latents; additional 3D cache data is not exposed by upstream APIs yet.
- First-time gsplat runs on Windows can take a minute to build CUDA kernels.

## Roadmap
- Finish the duplicated Cosmos loader/inference node so GEN3C outputs feed directly into the exporter without manual wiring.
- Add dataset validators (pose continuity, depth sanity) and automated regression graphs.
- Surface a trajectory preview widget (frustum/path plot) and richer keyframe tooling.

## Support
Issues, suggestions, or contributions are welcome. Open a ticket or PR in your ComfyUI fork and tag with `GEN3C-Gsplat` so it’s easy to triage.

