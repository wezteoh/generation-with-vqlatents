"""Side-by-side image grids for WandB (e.g. real vs model output)."""

from __future__ import annotations

import numpy as np
import torch


def tensor_batch_to_display_hwc_uint8(t: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) normalized [-1,1] or similar -> (B, H, W, C) uint8."""
    t = t.detach().cpu().float()
    t = (t * 0.5 + 0.5).clamp(0.0, 1.0)
    t = (t * 255.0).round().to(torch.uint8)
    t = t.permute(0, 2, 3, 1)
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)
    return t


def build_side_by_side_wandb_images(
    left: torch.Tensor,
    right: torch.Tensor,
    captions: list[str] | None = None,
):
    """Concatenate each row of `left` and `right` horizontally (VQ-VAE style)."""
    import wandb

    o = tensor_batch_to_display_hwc_uint8(left)
    r = tensor_batch_to_display_hwc_uint8(right)
    images = []
    for i in range(left.shape[0]):
        combined = np.concatenate([o[i].numpy(), r[i].numpy()], axis=1)
        cap = captions[i] if captions is not None else f"real | gen {i}"
        images.append(wandb.Image(combined, caption=cap))
    return images
