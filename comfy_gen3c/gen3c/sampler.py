"""Cosmos (GEN3C) inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

import comfy.sample
import comfy.samplers
import comfy.utils
from comfy import conds, model_management

from .loader import LyraModelBundle


def _seed_everything(seed: int) -> torch.Generator:
    generator = torch.Generator(device=model_management.get_torch_device())
    generator.manual_seed(seed)
    return generator


@dataclass
class SamplingConfig:
    steps: int
    guidance_scale: float
    num_views: int


def _encode_trajectory_conditioning(
    trajectory: Dict[str, Any],
    bundle: LyraModelBundle,
) -> Dict[str, torch.Tensor]:
    """Convert trajectory data into model conditioning format for GEN3C."""
    frames_meta = trajectory.get("frames", [])
    if not frames_meta:
        raise ValueError("Trajectory missing frame metadata")

    device = bundle.device
    dtype = bundle.dtype

    # Extract camera parameters from trajectory
    num_frames = len(frames_meta)

    # Build camera matrices (4x4 transforms)
    camera_matrices = []
    for frame in frames_meta:
        cam_to_world = frame["extrinsics"]["camera_to_world"]
        camera_matrices.append(torch.tensor(cam_to_world, dtype=dtype, device=device))

    # Stack into tensor (N, 4, 4)
    camera_transforms = torch.stack(camera_matrices)

    # Extract intrinsics (assuming same across frames)
    intrinsics_matrix = frames_meta[0]["intrinsics"]
    intrinsics = torch.tensor(intrinsics_matrix, dtype=dtype, device=device)

    return {
        "camera_transforms": camera_transforms,
        "intrinsics": intrinsics,
        "num_frames": num_frames,
        "fps": trajectory.get("fps", 24),
        "handedness": trajectory.get("handedness", "right")
    }


def _prepare_cosmos_latent(
    width: int,
    height: int,
    frames: int,
    channels: int = 16,
    temporal_stride: int = 8,
    spatial_stride: int = 8,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    """Prepare empty latent tensor for Cosmos inference."""
    latent_frames = ((frames - 1) // temporal_stride) + 1
    latent_h = max(1, height // spatial_stride)
    latent_w = max(1, width // spatial_stride)
    return torch.zeros((1, channels, latent_frames, latent_h, latent_w), dtype=dtype, device=device)


def sample_cosmos(
    bundle: LyraModelBundle,
    trajectory: Dict[str, Any],
    prompt: str,
    negative_prompt: str,
    seed: int,
    sampling: SamplingConfig,
    sampler_name: str = "res_multistep",
    scheduler: str = "normal",
    denoise: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Sample from Cosmos model with trajectory conditioning."""
    if generator is None:
        generator = _seed_everything(seed)

    if bundle.clip is None:
        raise RuntimeError("Bundle missing CLIP text encoder")

    # Extract trajectory metadata
    frames_meta = trajectory.get("frames", [])
    if not frames_meta:
        raise ValueError("Trajectory missing frame metadata")

    width = int(frames_meta[0].get("width", 1024))
    height = int(frames_meta[0].get("height", 576))
    frame_count = len(frames_meta)
    fps = float(trajectory.get("fps", 24))

    # Prepare model
    model = bundle.model.clone()
    latent_format = model.get_model_object("latent_format")
    channels = getattr(latent_format, "latent_channels", 16)

    # Prepare latent
    latent_dtype = torch.float32 if bundle.device == "cpu" else bundle.dtype
    latent = _prepare_cosmos_latent(width, height, frame_count, channels, dtype=latent_dtype, device=bundle.device)
    latent_dict = {"samples": latent}

    # Encode text prompts
    def encode_prompt(text: str, frame_rate: float) -> List:
        tokens = bundle.clip.tokenize(text)
        return bundle.clip.encode_from_tokens_scheduled(tokens, add_dict={"frame_rate": frame_rate})

    positive = encode_prompt(prompt or "", fps)
    negative = encode_prompt(negative_prompt or "", fps)

    if not positive:
        raise RuntimeError("Failed to encode prompt")
    if not negative:
        negative = encode_prompt("", fps)

    # Encode trajectory conditioning
    trajectory_cond = _encode_trajectory_conditioning(trajectory, bundle)

    # Add trajectory conditioning to positive conditioning
    # This assumes the model expects camera data in the conditioning dict
    for cond_item in positive:
        if isinstance(cond_item, dict):
            cond_item.update(trajectory_cond)
        elif hasattr(cond_item, '__dict__'):
            # Handle other conditioning formats
            for key, value in trajectory_cond.items():
                setattr(cond_item, key, value)

    # Prepare sampling inputs
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_dict["samples"])
    noise = comfy.sample.prepare_noise(latent_image, seed, None)
    noise_mask = latent_dict.get("noise_mask")

    # Sample with trajectory conditioning
    samples = comfy.sample.sample(
        model,
        noise,
        sampling.steps,
        sampling.guidance_scale,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        noise_mask=noise_mask,
        seed=seed,
    )

    return {
        "samples": samples.to(torch.device("cpu")),
        "trajectory": trajectory,
        "camera_trajectory": trajectory,
        "camera_transforms": trajectory_cond["camera_transforms"].cpu(),
        "intrinsics": trajectory_cond["intrinsics"].cpu(),
    }


def extract_depth_from_samples(
    samples: torch.Tensor,
    bundle: LyraModelBundle,
) -> Optional[torch.Tensor]:
    """Extract depth maps from Cosmos samples if available."""
    # Placeholder for depth extraction logic
    # The actual implementation depends on the Cosmos model's output format
    # This would typically involve decoding depth channels from the latent
    # or using a separate depth prediction head

    # For now, return None to indicate depth extraction is not implemented
    return None


__all__ = ["sample_cosmos", "SamplingConfig", "extract_depth_from_samples"]
