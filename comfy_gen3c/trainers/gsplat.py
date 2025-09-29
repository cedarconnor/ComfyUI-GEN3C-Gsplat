"""gsplat-based Gaussian Splat trainer for GEN3C datasets."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import gsplat
except Exception as exc:  # pragma: no cover - handled at runtime
    gsplat = None  # type: ignore[assignment]
    GSPLAT_IMPORT_ERROR = exc
else:
    GSPLAT_IMPORT_ERROR = None


@dataclass
class FrameRecord:
    rgb: torch.Tensor  # (C, H, W)
    depth: torch.Tensor  # (H, W)
    camera_to_world: torch.Tensor  # (4, 4)
    world_to_camera: torch.Tensor  # (4, 4)
    width: int
    height: int


def _resolve_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).clone()
    return tensor


def _load_depth(path: Path, depth_scale: float) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path).astype(np.float32)
    elif suffix in {".png", ".tif", ".tiff"}:
        with Image.open(path) as image:
            array = np.asarray(image, dtype=np.float32)
        if array.max() > 1.0:
            array = array / 65535.0
        depth = array
    elif suffix == ".pfm":
        depth = _read_pfm(path)
    else:
        raise ValueError(f"Unsupported depth format: {suffix}")
    depth = depth * depth_scale
    return torch.from_numpy(depth)


def _read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        header = fh.readline().decode("ascii").strip()
        color = header == "PF"
        dims = fh.readline().decode("ascii").strip()
        width, height = map(int, dims.split())
        scale = float(fh.readline().decode("ascii").strip())
        data = np.fromfile(fh, "<f" if scale < 0 else ">f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        if color:
            data = data[..., 0]
        return data.astype(np.float32)


def _invert(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(matrix)


def _prepare_points(
    frames: List[FrameRecord],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    points_per_frame: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    point_list: List[torch.Tensor] = []
    color_list: List[torch.Tensor] = []

    for frame in frames:
        depth = frame.depth
        mask = depth > 0.0
        if not torch.any(mask):
            continue

        h, w = depth.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        depth_masked = depth[mask]
        xx_masked = xx[mask]
        yy_masked = yy[mask]

        x = (xx_masked - cx) / fx * depth_masked
        y = (yy_masked - cy) / fy * depth_masked
        z = depth_masked

        ones = torch.ones_like(z)
        cam_points = torch.stack([x, y, z, ones], dim=-1)  # (N, 4)
        world_points = (cam_points @ frame.camera_to_world.T)[:, :3]

        colors = frame.rgb.permute(1, 2, 0)[mask].reshape(-1, 3)

        max_points = min(points_per_frame, world_points.shape[0])
        if world_points.shape[0] > max_points:
            indices = torch.randperm(world_points.shape[0])[:max_points]
            world_points = world_points[indices]
            colors = colors[indices]

        point_list.append(world_points)
        color_list.append(colors)

    if not point_list:
        raise RuntimeError("No valid depth samples found in dataset.")

    points = torch.cat(point_list, dim=0).to(device)
    colors = torch.cat(color_list, dim=0).to(device)
    return points, colors


def _normalize_quaternions(quat: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
    return quat / norm


def _write_ply(
    path: Path,
    xyz: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
) -> None:
    xyz = xyz.detach().cpu().numpy()
    colors = (colors.detach().cpu().clamp(0.0, 1.0) * 255.0).numpy().astype(np.uint8)
    opacity = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    quats = quats.detach().cpu().numpy()

    count = xyz.shape[0]
    with path.open("w", encoding="ascii") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"element vertex {count}\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        fh.write("property float opacity\n")
        fh.write("property float scale_x\n")
        fh.write("property float scale_y\n")
        fh.write("property float scale_z\n")
        fh.write("property float quat_w\n")
        fh.write("property float quat_x\n")
        fh.write("property float quat_y\n")
        fh.write("property float quat_z\n")
        fh.write("end_header\n")
        for i in range(count):
            x, y, z = xyz[i]
            r, g, b = colors[i]
            alpha = float(opacity[i])
            sx, sy, sz = scales[i]
            qw, qx, qy, qz = quats[i]
            fh.write(
                f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {alpha:.6f} "
                f"{sx:.6f} {sy:.6f} {sz:.6f} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}\n"
            )


class SplatTrainerGsplat:
    """Train a Gaussian Splat model using the gsplat rasterizer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"tooltip": "Path to dataset with transforms.json"}),
                "output_dir": ("STRING", {"default": "${output_dir}/gsplat_runs"}),
                "run_name": ("STRING", {"default": "gsplat_gen3c"}),
                "max_iterations": ("INT", {"default": 1000, "min": 10, "max": 20000}),
                "learning_rate": ("FLOAT", {"default": 5e-3, "min": 1e-5, "max": 1e-1, "step": 1e-5}),
                "points_per_frame": ("INT", {"default": 50000, "min": 1000, "max": 2000000}),
                "frames_per_batch": ("INT", {"default": 1, "min": 1, "max": 8}),
                "depth_scale": ("FLOAT", {"default": 1.0, "min": 1e-3, "max": 1000.0, "step": 1e-3}),
                "device": (("auto", "cuda", "cpu"), {"default": "auto"}),
            },
            "optional": {
                "block_width": ("INT", {"default": 16, "min": 2, "max": 16}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "train"
    CATEGORY = "GEN3C/Training"

    def train(
        self,
        dataset_dir: str,
        output_dir: str,
        run_name: str,
        max_iterations: int,
        learning_rate: float,
        points_per_frame: int,
        frames_per_batch: int,
        depth_scale: float,
        device: str,
        block_width: int = 16,
    ) -> Tuple[str]:
        if gsplat is None:
            raise RuntimeError(
                "gsplat library is not available. Install gsplat (pip install gsplat) "
                f"or resolve import error: {GSPLAT_IMPORT_ERROR}"
            )

        dataset_path = Path(dataset_dir).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory '{dataset_path}' not found.")

        transforms_path = dataset_path / "transforms.json"
        if not transforms_path.exists():
            raise FileNotFoundError(
                f"Dataset missing transforms.json at '{transforms_path}'."
            )

        with transforms_path.open("r", encoding="utf-8") as fh:
            transforms = json.load(fh)

        fx = float(transforms["fl_x"])
        fy = float(transforms["fl_y"])
        cx = float(transforms["cx"])
        cy = float(transforms["cy"])
        fps = transforms.get("fps", 24)

        width = int(transforms["w"])
        height = int(transforms["h"])

        frames: List[FrameRecord] = []
        for frame_entry in transforms["frames"]:
            rgb_path = dataset_path / frame_entry["file_path"]
            if not rgb_path.exists():
                raise FileNotFoundError(f"Missing RGB frame at '{rgb_path}'.")
            rgb_tensor = _load_image(rgb_path)

            depth_path_entry = frame_entry.get("depth_path")
            if depth_path_entry is None:
                raise ValueError("All frames require depth maps for gsplat training.")
            depth_path = dataset_path / depth_path_entry
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing depth frame at '{depth_path}'.")
            depth_tensor = _load_depth(depth_path, depth_scale)

            cam_to_world = torch.tensor(
                frame_entry["transform_matrix"], dtype=torch.float32
            )
            world_to_cam = _invert(cam_to_world)

            frames.append(
                FrameRecord(
                    rgb=rgb_tensor,
                    depth=depth_tensor,
                    camera_to_world=cam_to_world,
                    world_to_camera=world_to_cam,
                    width=width,
                    height=height,
                )
            )

        if not frames:
            raise RuntimeError("No frames found in dataset.")

        torch_device = _resolve_device(device)
        os.environ.setdefault("PATH", os.environ.get("PATH", ""))
        scripts_dir = Path(torch.__file__).resolve().parent.parent / "Scripts"
        path_parts = os.environ["PATH"].split(os.pathsep)
        scripts_str = str(Path.cwd() / ".venv" / "Scripts")
        for extra in (scripts_str, str(scripts_dir)):
            if extra and extra not in path_parts:
                path_parts.insert(0, extra)
        os.environ["PATH"] = os.pathsep.join(path_parts)

        points, colors = _prepare_points(
            frames,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            points_per_frame=points_per_frame,
            device=torch_device,
        )

        num_points = points.shape[0]
        if num_points == 0:
            raise RuntimeError("No valid 3D points extracted from dataset.")

        # Initialise Gaussian parameters
        means = torch.nn.Parameter(points)
        log_scales = torch.nn.Parameter(torch.full((num_points, 3), math.log(0.02), device=torch_device))
        quat_params = torch.nn.Parameter(torch.zeros(num_points, 4, device=torch_device))
        quat_params[:, 0] = 1.0
        color_params = torch.nn.Parameter(torch.clamp(colors, 0.0, 1.0))
        logit_opacity = torch.nn.Parameter(torch.zeros(num_points, 1, device=torch_device))

        parameters = [means, log_scales, quat_params, color_params, logit_opacity]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        frame_indices = list(range(len(frames)))
        rendered_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for frame in frames:
            rgb = frame.rgb.to(torch_device)
            rendered_cache.append((rgb, frame.world_to_camera.to(torch_device), frame.camera_to_world.to(torch_device)))

        for step in range(max_iterations):
            if frames_per_batch >= len(frame_indices):
                batch_indices = frame_indices
            else:
                batch_indices = torch.randperm(len(frame_indices))[:frames_per_batch].tolist()

            optimizer.zero_grad()
            loss_acc = torch.tensor(0.0, device=torch_device)

            for idx in batch_indices:
                target_rgb, world_to_cam, _ = rendered_cache[idx]

                scales = torch.exp(log_scales)
                quats = _normalize_quaternions(quat_params)
                colors_lin = torch.clamp(color_params, 0.0, 1.0)
                opacity = torch.sigmoid(logit_opacity)

                try:
                    xys, depths, radii, conics, compensation, num_tiles_hit, _ = gsplat.project_gaussians(
                        means,
                        scales,
                        1.0,
                        quats,
                        world_to_cam,
                        fx,
                        fy,
                        cx,
                        cy,
                        height,
                        width,
                        block_width,
                    )
                except RuntimeError as err:
                    message = str(err)
                    if "cl" in message.lower():
                        raise RuntimeError(
                            "gsplat requires the Microsoft Visual C++ Build Tools (cl.exe) to compile its CUDA "
                            "kernels on Windows. Install the Build Tools and ensure cl.exe is in PATH."
                        ) from err
                    raise

                colors_comp = colors_lin * compensation.unsqueeze(-1)
                rendered = gsplat.rasterize.rasterize_gaussians(
                    xys,
                    depths,
                    radii,
                    conics,
                    num_tiles_hit.int(),
                    colors_comp,
                    opacity,
                    height,
                    width,
                    block_width,
                )

                if rendered.ndim == 3:
                    rendered = rendered.permute(2, 0, 1)
                elif rendered.ndim == 2:
                    rendered = rendered.unsqueeze(0)
                loss_acc = loss_acc + torch.nn.functional.l1_loss(rendered, target_rgb)

            loss_acc = loss_acc / max(1, len(batch_indices))
            loss_acc.backward()
            optimizer.step()

            with torch.no_grad():
                quat_params.copy_(_normalize_quaternions(quat_params))

        output_root = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output"))).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        ply_dir = output_root / run_name
        ply_dir.mkdir(parents=True, exist_ok=True)
        ply_path = ply_dir / "point_cloud.ply"

        final_scales = torch.exp(log_scales)
        final_quats = _normalize_quaternions(quat_params)
        final_opacity = torch.sigmoid(logit_opacity)
        _write_ply(ply_path, means, color_params, final_opacity, final_scales, final_quats)

        return (str(ply_path),)


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "SplatTrainer_gsplat": SplatTrainerGsplat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatTrainer_gsplat": "Splat Trainer (gsplat)",
}
