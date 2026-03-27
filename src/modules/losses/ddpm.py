from __future__ import annotations

import torch
import torch.nn.functional as F


def ddpm_eps_loss(
    pred: torch.Tensor,
    noise: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    """MSE between predicted and sampled Gaussian noise (epsilon parameterization)."""
    if loss_type == "l2":
        return F.mse_loss(pred, noise)
    if loss_type == "l1":
        return F.l1_loss(pred, noise)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def ddpm_x0_loss(
    pred_x0: torch.Tensor,
    x0: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    """MSE / L1 between predicted and true clean latents (x0 parameterization)."""
    if loss_type == "l2":
        return F.mse_loss(pred_x0, x0)
    if loss_type == "l1":
        return F.l1_loss(pred_x0, x0)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def ddpm_training_loss(
    model_pred: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
    parameterization: str = "eps",
    loss_type: str = "l2",
) -> torch.Tensor:
    """Dispatch loss for epsilon vs x0 prediction."""
    if parameterization == "eps":
        return ddpm_eps_loss(model_pred, noise, loss_type=loss_type)
    if parameterization == "x0":
        return ddpm_x0_loss(model_pred, x0, loss_type=loss_type)
    raise ValueError(f"Unknown parameterization: {parameterization}")
