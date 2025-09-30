"""GEN3C diffusion node skeleton."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

import comfy.sample
import comfy.samplers
import comfy.utils

from .loader import LyraModelBundle


DEFAULT_SAMPLER = "res_multistep"
DEFAULT_SCHEDULER = "normal"
DOWNSAMPLE = 8  # Cosmos latent stride (spatial)
TEMPORAL_STRIDE = 8  # Cosmos latent temporal compression


class Gen3CDiffusion:
    """Generate GEN3C latents with basic text prompts and optional trajectory control."""

    @classmethod
    def INPUT_TYPES(cls):  # pragma: no cover - UI definition
        return {
            "required": {
                "lyra_model": ("LYRA_MODEL", {}),
                "camera_trajectory": ("GEN3C_TRAJECTORY", {}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 500}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": DEFAULT_SAMPLER}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": DEFAULT_SCHEDULER}),
            },
            "optional": {
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENTS", "IMAGE", "GEN3C_TRAJECTORY")
    RETURN_NAMES = ("latents", "images", "cameras")
    FUNCTION = "sample"
    CATEGORY = "GEN3C/Diffusion"

    def _encode_prompt(self, clip, text: str, frame_rate: float) -> list:
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens, add_dict={"frame_rate": frame_rate})

    def _prepare_latent(self, width: int, height: int, frames: int, channels: int, dtype: torch.dtype) -> torch.Tensor:
        latent_frames = ((frames - 1) // TEMPORAL_STRIDE) + 1
        latent_h = max(1, height // DOWNSAMPLE)
        latent_w = max(1, width // DOWNSAMPLE)
        return torch.zeros((1, channels, latent_frames, latent_h, latent_w), dtype=dtype)

    def _decode_to_images(self, bundle: LyraModelBundle, latents: torch.Tensor) -> torch.Tensor:
        if bundle.vae is None:
            raise RuntimeError("Loaded GEN3C bundle is missing a VAE for decoding latents.")
        samples = latents.to(bundle.device, dtype=bundle.dtype)
        decoded = bundle.vae.decode_tiled(samples)
        decoded = decoded.to(torch.float32).cpu()
        if decoded.ndim == 5:  # (B, T, C, H, W)
            decoded = decoded[0].permute(0, 2, 3, 1)
        elif decoded.ndim == 4:  # (B, C, H, W)
            decoded = decoded[0].permute(1, 2, 0).unsqueeze(0)
        else:
            raise RuntimeError(f"Unexpected decoded tensor shape {tuple(decoded.shape)} from VAE.")
        return decoded.clamp_(0.0, 1.0)

    def sample(
        self,
        lyra_model: LyraModelBundle,
        camera_trajectory: Dict[str, Any],
        num_inference_steps: int,
        guidance_scale: float,
        prompt: str,
        negative_prompt: str,
        seed: int,
        sampler_name: str,
        scheduler: str,
        denoise: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:  # pragma: no cover - heavy runtime
        from .sampler import sample_cosmos, SamplingConfig

        if lyra_model.clip is None:
            raise RuntimeError("Loaded GEN3C bundle is missing a CLIP text encoder.")
        frames_meta = camera_trajectory.get("frames", [])
        if not frames_meta:
            raise ValueError("camera_trajectory is missing frame metadata. Use Gen3C_CameraTrajectory upstream.")

        # Create sampling configuration
        sampling_config = SamplingConfig(
            steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_views=len(frames_meta)
        )

        # Use the enhanced sample_cosmos function with trajectory injection
        result = sample_cosmos(
            bundle=lyra_model,
            trajectory=camera_trajectory,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            sampling=sampling_config,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
        )

        # Extract samples and decode to images
        samples = result["samples"]
        latent_out = {"samples": samples}
        images = self._decode_to_images(lyra_model, samples)

        return (latent_out, images, camera_trajectory)


NODE_CLASS_MAPPINGS = {
    "Gen3CDiffusion": Gen3CDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen3CDiffusion": "GEN3C Diffusion",
}
