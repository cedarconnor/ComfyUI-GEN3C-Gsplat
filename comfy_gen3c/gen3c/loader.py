"""Nodes and helpers for loading Lyra/GEN3C (Cosmos) models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

import folder_paths

from comfy import model_management, sd
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP, VAE

PRECISION_MODES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _resolve_device(device_choice: str) -> Tuple[str, str]:
    """Return (load_device, offload_device) string identifiers."""
    if device_choice == "auto":
        if torch.cuda.is_available():
            return (model_management.get_torch_device_name(), "cpu")
        return ("cpu", "cpu")
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return (model_management.get_torch_device_name(), "cpu")
    return ("cpu", "cpu")


def _resolve_dtype(precision: str, load_device: str) -> torch.dtype:
    dtype = PRECISION_MODES.get(precision.lower())
    if dtype is None:
        raise ValueError(f"Unsupported precision '{precision}'. Choose from {tuple(PRECISION_MODES)}.")
    if load_device == "cpu" and dtype == torch.float16:
        # CPU inference with fp16 often fails – fall back to fp32
        return torch.float32
    return dtype


def _load_checkpoint(
    ckpt_path: str,
    load_device: str,
    offload_device: str,
    dtype: torch.dtype,
    enable_offload: bool,
) -> Tuple[ModelPatcher, Optional[CLIP], Optional[VAE]]:
    model_options: Dict[str, Any] = {
        "load_device": load_device,
        "offload_device": offload_device if enable_offload else load_device,
        "dtype": dtype,
    }
    te_model_options: Dict[str, Any] = {
        "load_device": load_device,
        "offload_device": offload_device if enable_offload else load_device,
        "dtype": dtype,
    }
    embedding_dirs = folder_paths.get_folder_paths("embeddings")
    model, clip, vae, _ = sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        output_clipvision=False,
        embedding_directory=embedding_dirs,
        output_model=True,
        model_options=model_options,
        te_model_options=te_model_options,
    )
    return model, clip, vae


def _load_torchscript_module(path: str, device: str) -> torch.jit.ScriptModule:
    try:
        module = torch.jit.load(path, map_location=device)
        module.eval()
        return module
    except Exception as exc:  # pragma: no cover - depends on external files
        raise RuntimeError(f"Failed to load TorchScript module at '{path}': {exc}") from exc


@dataclass
class LyraModelBundle:
    """Container for Cosmos/GEN3C model components."""

    model: ModelPatcher
    clip: Optional[CLIP]
    vae: Optional[VAE]
    lyra_tokenizer: Optional[torch.jit.ScriptModule]
    tokenizer_path: str
    diffusion_path: str
    device: str
    dtype: torch.dtype
    max_vram_gb: float


class LyraModelLoader:
    """Load the Cosmos/GEN3C diffusion stack (Lyra VAE + Cosmos UNet + tokenizer)."""

    @classmethod
    def INPUT_TYPES(cls):  # pragma: no cover - UI definition
        model_types = ("static",)
        devices = ("auto", "cuda", "cpu")
        precisions = tuple(PRECISION_MODES.keys())
        return {
            "required": {
                "model_type": (model_types, {"default": "static"}),
                "lyra_encoder_path": ("STRING", {"default": "models/Lyra/lyra_static.pt"}),
                "diffusion_model_path": ("STRING", {"default": "models/Lyra/GEN3C-Cosmos-7B.pt"}),
                "device": (devices, {"default": "auto"}),
                "precision": (precisions, {"default": "fp16"}),
                "cosmos_tokenizer_path": ("STRING", {"default": "models/Lyra/Cosmos-0.1-Tokenizer-CV8x16x16-autoencoder.jit"}),
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "max_vram_gb": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 128.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("LYRA_MODEL",)
    RETURN_NAMES = ("lyra_model",)
    FUNCTION = "load"
    CATEGORY = "GEN3C/Model"

    def load(
        self,
        model_type: str,
        lyra_encoder_path: str,
        diffusion_model_path: str,
        device: str,
        precision: str,
        cosmos_tokenizer_path: str,
        enable_offloading: bool,
        max_vram_gb: float,
    ) -> Tuple[LyraModelBundle]:
        load_device, offload_device = _resolve_device(device)
        dtype = _resolve_dtype(precision, load_device)

        model, clip, vae = _load_checkpoint(
            diffusion_model_path,
            load_device=load_device,
            offload_device=offload_device,
            dtype=dtype,
            enable_offload=enable_offloading,
        )

        # Attempt to load the Lyra tokenizer/autoencoder if provided.
        lyra_module: Optional[torch.jit.ScriptModule] = None
        if lyra_encoder_path:
            lyra_module = _load_torchscript_module(lyra_encoder_path, device=load_device)

        bundle = LyraModelBundle(
            model=model,
            clip=clip,
            vae=vae,
            lyra_tokenizer=lyra_module,
            tokenizer_path=cosmos_tokenizer_path,
            diffusion_path=diffusion_model_path,
            device=load_device,
            dtype=dtype,
            max_vram_gb=max_vram_gb,
        )
        return (bundle,)


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "LyraModelLoader": LyraModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyraModelLoader": "Lyra / GEN3C Model Loader",
}
