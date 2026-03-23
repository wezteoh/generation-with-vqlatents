from __future__ import annotations

import functools

import torch
import torch.nn as nn

from src.modules.latents.diffusion_backbones.unet import UnetSkipConnectionBlock
from src.modules.latents.score_models import ScoreModel


class UNetScore(ScoreModel):
    """U-Net score network over (image or latent) tensors."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_classes: int,
        first_stage_model: nn.Module | None = None,
        image_size: int = 32,
        logit_transform: bool = False,
    ):
        super().__init__(first_stage_model=first_stage_model)
        self.logit_transform = logit_transform

        # Sigma-label embedding (class-conditional noise level).
        self.label_emb = nn.Embedding(num_classes, in_channels)

        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )

        input_nc = output_nc = in_channels
        ngf = base_channels

        if image_size == 32:
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=None,
                norm_layer=norm_layer,
                innermost=True,
            )
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
            )
        elif image_size in (16, 8, 4):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=None,
                norm_layer=norm_layer,
                innermost=True,
            )
        else:
            raise ValueError(f"Unsupported image_size for UNetScore: {image_size}")

        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.unet = UnetSkipConnectionBlock(
            output_nc,
            ngf * 2,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def _add_label_channel(
        self, x: torch.Tensor, sigma_labels: torch.Tensor
    ) -> torch.Tensor:
        """Inject sigma-label embedding as an additive bias in channel space."""
        emb = self.label_emb(sigma_labels)  # (B, C)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1)
        return x + emb

    def forward(self, x: torch.Tensor, sigma_labels: torch.Tensor) -> torch.Tensor:
        if not self.logit_transform:
            x = 2.0 * x - 1.0
        x = self._add_label_channel(x, sigma_labels)
        return self.unet(x)


class LatentUNetScore(ScoreModel):
    """U-Net score network over (image or latent) tensors for latent DSM."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_classes: int,
        first_stage_model: nn.Module | None = None,
        image_size: int = 32,
        logit_transform: bool = False,
    ):
        super().__init__(first_stage_model=first_stage_model)
        self.logit_transform = logit_transform
        self.label_emb = nn.Embedding(num_classes, in_channels)

        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )

        input_nc = output_nc = in_channels
        ngf = base_channels

        # For very small latent maps (4x4), build a shallower UNet so that
        # the minimum spatial resolution stays at 2x2 instead of 1x1.
        if image_size == 4:
            unet_block = UnetSkipConnectionBlock(
                ngf * 2,
                ngf * 2,
                input_nc=None,
                submodule=None,
                norm_layer=norm_layer,
                innermost=True,
            )
            self.unet = UnetSkipConnectionBlock(
                output_nc,
                ngf * 2,
                input_nc=input_nc,
                submodule=unet_block,
                outermost=True,
                norm_layer=norm_layer,
            )
            return

        if image_size == 32:
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=None,
                norm_layer=norm_layer,
                innermost=True,
            )
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
            )
        elif image_size in (16, 8):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=None,
                norm_layer=norm_layer,
                innermost=True,
            )
        else:
            raise ValueError(f"Unsupported image_size for LatentUNetScore: {image_size}")

        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.unet = UnetSkipConnectionBlock(
            output_nc,
            ngf * 2,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

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
        return self.unet(x)

