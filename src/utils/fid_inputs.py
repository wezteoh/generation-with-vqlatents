"""Shared tensor formatting for torchmetrics image metrics (FID, etc.)."""

from __future__ import annotations

import torch


def tensor_to_fid_input(t: torch.Tensor) -> torch.Tensor:
    """Normalized (B,C,H,W) float in ~[-1,1] to uint8 (B,3,H,W) for Inception FID."""
    t = t.detach().float()
    t = (t * 0.5 + 0.5).clamp(0.0, 1.0)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    t = (t * 255.0).round().to(torch.uint8)
    return t
