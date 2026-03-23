from __future__ import annotations

import torch
import torch.nn as nn

from src.modules.latents.diffusion_backbones.ncsnv2 import _NCSNv2Backbone
from src.modules.latents.score_models import ScoreModel


class LatentNCSNv2Score(ScoreModel):
    """NCSNv2-style score model over (image or latent) tensors."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_classes: int,
        first_stage_model: nn.Module | None = None,
        image_size: int = 32,
        logit_transform: bool = False,
        sigmas: torch.Tensor | None = None,
    ) -> None:
        super().__init__(first_stage_model=first_stage_model)
        self.logit_transform = logit_transform
        self.label_emb = nn.Embedding(num_classes, in_channels)

        if sigmas is not None:
            self.register_buffer("sigmas", sigmas.to(dtype=torch.float32), persistent=True)
        else:
            self.register_buffer(
                "sigmas", torch.tensor([], dtype=torch.float32), persistent=True
            )

        self.backbone = _NCSNv2Backbone(
            in_channels=in_channels,
            base_channels=base_channels,
            image_size=int(image_size),
            num_classes=num_classes,
            logit_transform=logit_transform,
        )

    def forward(
        self, x: torch.Tensor, sigma_labels: torch.Tensor, sigmas: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self.logit_transform:
            x = 2.0 * x - 1.0
        output = self.backbone(x)

        # Match the original NCSNv2 behaviour: scale by the sigma used
        # for the current label so that the network output is normalized
        # as a score rather than raw residual.
        if sigmas is None:
            sigmas = self.sigmas

        if sigmas.numel() > 0:
            used_sigmas = sigmas[sigma_labels].view(x.shape[0], *([1] * len(x.shape[1:])))
            output = output / used_sigmas
        return output

