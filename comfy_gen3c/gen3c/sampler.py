"""Cosmos (GEN3C) inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

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


def sample_cosmos(
    bundle: LyraModelBundle,
    prompts: Dict[str, Any],
    seed: int,
    sampling: SamplingConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:  # pragma: no cover - heavy runtime
    if generator is None:
        generator = _seed_everything(seed)

    # Placeholder: real implementation will call comfy.samplers.KSampler with the Cosmos model
    # and return per-view RGB/depth/latent data. For now, raise to signal incomplete feature.
    raise NotImplementedError("Cosmos sampling is not yet implemented.")


__all__ = ["sample_cosmos", "SamplingConfig"]
