"""Build context encoders for DDPM (raw BCHW -> features for UNet)."""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn
from omegaconf import OmegaConf

from src.modules.conditioning import SpatialRescaler


def build_context_encoder(cfg: Optional[Any]) -> nn.Module:
    """Instantiate encoder from a Hydra-style dict. Empty / None -> identity."""
    if cfg is None:
        return nn.Identity()
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not cfg:
        return nn.Identity()
    typ = str(cfg.get("type", "identity"))
    if typ == "identity":
        return nn.Identity()
    if typ == "spatial_rescaler":
        params = {k: v for k, v in cfg.items() if k != "type"}
        return SpatialRescaler(**params)
    raise ValueError(f"Unknown context_encoder.type: {typ!r}")


def set_encoder_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(trainable)
    if trainable:
        module.train()
    else:
        module.eval()
