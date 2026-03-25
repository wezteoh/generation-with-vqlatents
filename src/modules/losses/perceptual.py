import torch
import torch.nn as nn


def perceptual_loss(
    input: torch.Tensor, target: torch.Tensor, perceptual_model: nn.Module
) -> torch.Tensor:
    return perceptual_model(input, target)
