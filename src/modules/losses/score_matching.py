from __future__ import annotations

import torch


def _append_dims(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Append singleton dimensions to match target_dim."""
    while x.dim() < target_dim:
        x = x.unsqueeze(-1)
    return x


def dsm_loss(
    scores: torch.Tensor,
    samples: torch.Tensor,
    perturbed_samples: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Denoising score matching loss for a single noise level.

    Args:
        scores: Predicted scores, same shape as samples (B, C, H, W).
        samples: Clean samples x ~ p(x).
        perturbed_samples: Noisy samples x_sigma = x + sigma * z.
        sigma: Noise level (scalar or per-sample tensor).
    """
    if not torch.is_tensor(sigma):
        sigma = torch.as_tensor(sigma, device=samples.device, dtype=samples.dtype)

    # Broadcast sigma to match sample shape.
    sigma = _append_dims(sigma, samples.dim())
    target = -(perturbed_samples - samples) / (sigma ** 2)

    # Compute per-sample loss then average over batch.
    scores_flat = scores.view(scores.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    loss = 0.5 * ((scores_flat - target_flat) ** 2).sum(dim=-1).mean(dim=0)
    return loss


def anneal_dsm_loss(
    scores: torch.Tensor,
    samples: torch.Tensor,
    perturbed_samples: torch.Tensor,
    used_sigmas: torch.Tensor,
    anneal_power: float = 2.0,
) -> torch.Tensor:
    """Annealed DSM loss over a set of sigmas (per-sample labels).

    Args:
        scores: Predicted scores, same shape as samples.
        samples: Clean samples x ~ p(x).
        perturbed_samples: Noisy samples x_sigma.
        used_sigmas: Sigma value per sample (B,) or broadcastable.
        anneal_power: Exponent for sigma weighting (typically 2).
    """
    if not torch.is_tensor(used_sigmas):
        used_sigmas = torch.as_tensor(
            used_sigmas, device=samples.device, dtype=samples.dtype
        )

    used_sigmas = used_sigmas.to(device=samples.device, dtype=samples.dtype)
    sigma_broadcast = _append_dims(used_sigmas, samples.dim())
    target = -(perturbed_samples - samples) / (sigma_broadcast ** 2)

    scores_flat = scores.view(scores.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    per_sample = 0.5 * ((scores_flat - target_flat) ** 2).sum(dim=-1)

    weights = used_sigmas.view(-1) ** anneal_power
    loss = (per_sample * weights).mean(dim=0)
    return loss
