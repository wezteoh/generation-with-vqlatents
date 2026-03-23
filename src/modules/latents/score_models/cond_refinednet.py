from __future__ import annotations

import torch
import torch.nn as nn

from src.modules.latents.diffusion_backbones.cond_refinednet import CondRefineNetDilated
from src.modules.latents.score_models import ScoreModel


class LatentCondRefineNetScore(ScoreModel):
    """Latent score model using a CondRefineNet-style dilated backbone."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_classes: int,
        first_stage_model: nn.Module | None = None,
        image_size: int = 4,
        logit_transform: bool = False,
    ) -> None:
        super().__init__(first_stage_model=first_stage_model)
        self.logit_transform = logit_transform
        self.label_emb = nn.Embedding(num_classes, in_channels)

        # Parameterize explicitly for latent-sized inputs (no Hydra config).
        class _CfgData:
            def __init__(self, channels: int, image_size: int, logit: bool) -> None:
                self.channels = channels
                self.image_size = image_size
                self.logit_transform = logit

        class _CfgModel:
            def __init__(self, ngf: int, num_classes: int) -> None:
                self.ngf = ngf
                self.num_classes = num_classes

        class _Cfg:
            def __init__(
                self,
                channels: int,
                image_size: int,
                logit_transform: bool,
                ngf: int,
                num_classes: int,
            ) -> None:
                self.data = _CfgData(channels, image_size, logit_transform)
                self.model = _CfgModel(ngf, num_classes)

        cfg = _Cfg(
            channels=in_channels,
            image_size=int(image_size),
            logit_transform=logit_transform,
            ngf=base_channels,
            num_classes=num_classes,
        )

        self.backbone = CondRefineNetDilated(cfg)

    def _add_label_channel(
        self, x: torch.Tensor, sigma_labels: torch.Tensor
    ) -> torch.Tensor:
        emb = self.label_emb(sigma_labels)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1)
        return x + emb

    def forward(self, x: torch.Tensor, sigma_labels: torch.Tensor) -> torch.Tensor:
        if not self.logit_transform:
            x = 2.0 * x - 1.0
        x = self._add_label_channel(x, sigma_labels)
        return self.backbone(x, sigma_labels)
