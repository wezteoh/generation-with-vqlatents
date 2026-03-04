import torch


def reconstruction_loss(output, target):
    return torch.mean((output - target) ** 2)
