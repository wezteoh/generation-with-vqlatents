from __future__ import annotations

import torch
import torch.nn as nn

from src.modules.latents.diffusion_backbones.ncsnv2 import _NCSNv2Backbone
from src.modules.latents.score_sde import ScoreSDEModel
from src.modules.sde import SDE


class LatentNCSNv2ScoreSDE(ScoreSDEModel):
    """NCSNv2-style score model over (image or latent) tensors."""

    def __init__(
        self,
        sde: SDE,
        in_channels: int,
        base_channels: int,
        first_stage_model: nn.Module | None = None,
        image_size: int = 32,
        logit_transform: bool = False
    ) -> None:
        super().__init__(sde=sde, first_stage_model=first_stage_model)
        self.logit_transform = logit_transform
        self.backbone = _NCSNv2Backbone(
            in_channels=in_channels,
            base_channels=base_channels,
            image_size=int(image_size),
            logit_transform=logit_transform,
        )

    def forward(
        self,
        x: torch.Tensor,
        sigmas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.logit_transform:
            x = 2.0 * x - 1.0
        output = self.backbone(x)
        used_sigmas = sigmas.view(x.shape[0], *([1] * (x.dim() - 1)))
        output = output / used_sigmas
        return output
