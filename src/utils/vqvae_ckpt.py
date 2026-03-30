"""Load VQ-VAE architecture metadata from a checkpoint directory's config.yaml."""

from __future__ import annotations

import os
from typing import Any

from omegaconf import OmegaConf


def load_vqvae_meta_from_ckpt_path(
    vq_ckpt_path: str,
) -> tuple[dict[str, Any], int, int]:
    """Load ``(ddconfig, n_embed, embed_dim)`` from VQ ``config.yaml`` next to ckpt."""
    ckpt_dir = os.path.dirname(str(vq_ckpt_path))
    ckpt_cfg_path = os.path.join(ckpt_dir, "config.yaml")
    if not os.path.exists(ckpt_cfg_path):
        raise FileNotFoundError(
            f"Expected VQ-VAE config at {ckpt_cfg_path}. "
            "Save config.yaml in the checkpoint folder when training VQ-VAE."
        )
    ckpt_cfg = OmegaConf.load(ckpt_cfg_path)
    vq_model_cfg = OmegaConf.to_container(ckpt_cfg.model, resolve=True)
    vq_ddconfig = dict(vq_model_cfg["ddconfig"])
    vq_n_embed = int(vq_model_cfg["n_embed"])
    vq_embed_dim = int(vq_model_cfg["embed_dim"])
    return vq_ddconfig, vq_n_embed, vq_embed_dim
