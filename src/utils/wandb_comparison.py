"""Side-by-side image grids for WandB (e.g. real vs model output)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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


def conditioning_image_bchw_to_hwc_uint8(ctx: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) with C in {1, 3} -> (B, H, W, C) uint8 RGB for logging.

    Supports common ranges: [0, 1] (e.g. edge maps) and [-1, 1] (normalized RGB).
    """
    if ctx.dim() != 4 or ctx.shape[1] not in (1, 3):
        raise ValueError(
            f"Expected BCHW with C in (1, 3), got shape {tuple(ctx.shape)}"
        )
    x = ctx.detach().cpu().float()
    lo, hi = float(x.amin()), float(x.amax())
    if lo >= -0.05 and hi <= 1.05:
        x_vis = x.clamp(0.0, 1.0)
    elif lo >= -1.05 and hi <= 1.05:
        x_vis = (x * 0.5 + 0.5).clamp(0.0, 1.0)
    else:
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        x_vis = (x - x_min) / (x_max - x_min + 1e-8)
    out = (x_vis * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)
    if out.shape[-1] == 1:
        out = out.repeat(1, 1, 1, 3)
    return out


def build_triplet_wandb_images(
    real: torch.Tensor,
    cond: torch.Tensor,
    gen: torch.Tensor,
    captions: list[str] | None = None,
):
    """Concatenate real | conditioning | generated horizontally per sample."""
    import wandb

    n = real.shape[0]
    if cond.shape[0] != n or gen.shape[0] != n:
        raise ValueError("real, cond, gen must share batch size")
    target_h, target_w = int(real.shape[2]), int(real.shape[3])
    cond_f = cond.detach().float()
    if cond_f.shape[2] != target_h or cond_f.shape[3] != target_w:
        cond_f = F.interpolate(
            cond_f,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    o = tensor_batch_to_display_hwc_uint8(real)
    g = tensor_batch_to_display_hwc_uint8(gen)
    c = conditioning_image_bchw_to_hwc_uint8(cond_f)
    images = []
    for i in range(n):
        combined = np.concatenate([o[i].numpy(), c[i].numpy(), g[i].numpy()], axis=1)
        cap = (
            captions[i]
            if captions is not None
            else f"real | cond | gen [{i}]"
        )
        images.append(wandb.Image(combined, caption=cap))
    return images
