from __future__ import annotations

import torch

from src.modules.sde import SDE


def _append_dims(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    while x.dim() < target_dim:
        x = x.unsqueeze(-1)
    return x


def score_sde_loss(
    scores: torch.Tensor,
    std: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    sde: SDE,
    likelihood_weighting: bool = True,
) -> torch.Tensor:
    """Score matching loss for continuous-time SDEs.

    Using the marginal perturbation:
      x_t = mean(x0, t) + std(t) * z,  z ~ N(0, I)

    The conditional score of x_t given x0 has target:
      grad_{x_t} log p(x_t | x0) = - z / std(t)
    """
    if std.dim() == 0:
        std = std.expand(scores.shape[0])
    std = std.to(device=scores.device, dtype=scores.dtype)
    z = z.to(device=scores.device, dtype=scores.dtype)
    t = t.to(device=scores.device, dtype=torch.float32)

    # Broadcast std to (B, C, H, W).
    std_b = _append_dims(std, scores.dim())
    target = -z / std_b

    diff = scores - target
    per_sample = 0.5 * (diff.view(diff.shape[0], -1) ** 2).sum(dim=-1)

    if likelihood_weighting:
        # In this repo's SDEs, `sde.sde(x=0, t)[1]` is the diffusion term g(t).
        g2 = sde.sde(torch.zeros_like(scores), t)[1].to(scores.device, scores.dtype) ** 2
        per_sample = per_sample * g2

    return per_sample.mean()
