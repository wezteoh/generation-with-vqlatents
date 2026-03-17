from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.latents.score_models import ScoreModel
from src.modules.latents.score_models.normalizations import get_normalization


class _SimpleResidualBlock(nn.Module):
    """Lightweight residual block used inside the NCSNv2-style backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: str | None = None,
        act: nn.Module | None = None,
        dilation: int | None = None,
        adjust_padding: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = resample
        self.act = act or nn.ELU()

        stride = 1
        if dilation is None:
            padding = 1
        else:
            padding = dilation

        if resample == "down":
            stride = 2

        if adjust_padding and resample == "down":
            self.conv1 = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    dilation=dilation or 1,
                ),
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation or 1,
            )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation or 1,
        )

        if in_channels != out_channels or resample is not None:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(x)
        h = self.conv1(h)
        h = self.act(h)
        h = self.conv2(h)
        sc = self.shortcut(x)
        return h + sc


class _RefineBlock(nn.Module):
    """Refine-and-merge block mirroring the structure used in NCSNv2."""

    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int,
        act: nn.Module | None = None,
        start: bool = False,
        end: bool = False,
    ) -> None:
        super().__init__()
        self.act = act or nn.ELU()
        self.start = start
        self.end = end
        self.out_channels = out_channels

        self.adapt_convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)
                for in_ch in in_channels_list
            ]
        )

        n_blocks = 3 if end else 1
        blocks: list[nn.Module] = []
        for _ in range(n_blocks):
            blocks.append(
                _SimpleResidualBlock(out_channels, out_channels, resample=None, act=self.act)
            )
        self.refiner = nn.Sequential(*blocks)

    def forward(self, xs: list[torch.Tensor], output_shape: torch.Size) -> torch.Tensor:
        assert isinstance(xs, list) and len(xs) > 0
        base = xs[0]
        b, _, h, w = base.shape
        target_h, target_w = output_shape
        device = base.device

        merged = torch.zeros(
            b, self.out_channels, target_h, target_w, device=device, dtype=base.dtype
        )
        for x, conv in zip(xs, self.adapt_convs):
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=True)
            x = conv(x)
            merged = merged + x

        merged = self.act(merged)
        merged = self.refiner(merged)
        return merged


class _NCSNv2Backbone(nn.Module):
    """NCSNv2-style backbone operating on (possibly latent) tensors."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        image_size: int,
        num_classes: int,
        logit_transform: bool = False,
        normalization: str = "InstanceNorm++",
    ) -> None:
        super().__init__()
        self.logit_transform = logit_transform
        self.rescaled = False
        self.ngf = base_channels
        self.num_classes = num_classes
        self.act = nn.ELU()

        # Build a tiny config shim so we can reuse get_normalization for the
        # non-conditional case, mirroring the original NCSNv2 codepath.
        class _CfgData:
            def __init__(self, channels: int) -> None:
                self.channels = channels

        class _CfgModel:
            def __init__(self, ngf: int, num_classes: int, norm: str) -> None:
                self.ngf = ngf
                self.num_classes = num_classes
                self.normalization = norm

        class _Cfg:
            def __init__(
                self,
                channels: int,
                ngf: int,
                num_classes: int,
                norm: str,
            ) -> None:
                self.data = _CfgData(channels)
                self.model = _CfgModel(ngf, num_classes, norm)

        cfg = _Cfg(
            channels=in_channels,
            ngf=base_channels,
            num_classes=num_classes,
            norm=normalization,
        )

        self.begin_conv = nn.Conv2d(in_channels, self.ngf, 3, stride=1, padding=1)
        norm_cls = get_normalization(cfg, conditional=False)
        self.normalizer = norm_cls(self.ngf) if norm_cls is not None else nn.Identity()
        self.end_conv = nn.Conv2d(self.ngf, in_channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList(
            [
                _SimpleResidualBlock(self.ngf, self.ngf, resample=None, act=self.act),
                _SimpleResidualBlock(self.ngf, self.ngf, resample=None, act=self.act),
            ]
        )

        self.res2 = nn.ModuleList(
            [
                _SimpleResidualBlock(self.ngf, 2 * self.ngf, resample="down", act=self.act),
                _SimpleResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=self.act),
            ]
        )

        self.res3 = nn.ModuleList(
            [
                _SimpleResidualBlock(
                    2 * self.ngf,
                    2 * self.ngf,
                    resample="down",
                    act=self.act,
                    dilation=2,
                ),
                _SimpleResidualBlock(
                    2 * self.ngf,
                    2 * self.ngf,
                    resample=None,
                    act=self.act,
                    dilation=2,
                ),
            ]
        )

        if image_size == 28:
            self.res4 = nn.ModuleList(
                [
                    _SimpleResidualBlock(
                        2 * self.ngf,
                        2 * self.ngf,
                        resample="down",
                        act=self.act,
                        dilation=4,
                        adjust_padding=True,
                    ),
                    _SimpleResidualBlock(
                        2 * self.ngf,
                        2 * self.ngf,
                        resample=None,
                        act=self.act,
                        dilation=4,
                    ),
                ]
            )
        else:
            self.res4 = nn.ModuleList(
                [
                    _SimpleResidualBlock(
                        2 * self.ngf,
                        2 * self.ngf,
                        resample="down",
                        act=self.act,
                        dilation=4,
                        adjust_padding=False,
                    ),
                    _SimpleResidualBlock(
                        2 * self.ngf,
                        2 * self.ngf,
                        resample=None,
                        act=self.act,
                        dilation=4,
                    ),
                ]
            )

        self.refine1 = _RefineBlock([2 * self.ngf], 2 * self.ngf, act=self.act, start=True)
        self.refine2 = _RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=self.act)
        self.refine3 = _RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=self.act)
        self.refine4 = _RefineBlock([self.ngf, self.ngf], self.ngf, act=self.act, end=True)

    @staticmethod
    def _run_block_list(blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for m in blocks:
            x = m(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.logit_transform and not self.rescaled:
            h = 2.0 * x - 1.0
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._run_block_list(self.res1, output)
        layer2 = self._run_block_list(self.res2, layer1)
        layer3 = self._run_block_list(self.res3, layer2)
        layer4 = self._run_block_list(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)
        return output


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
            self.register_buffer("sigmas", torch.tensor([], dtype=torch.float32), persistent=True)

        self.backbone = _NCSNv2Backbone(
            in_channels=in_channels,
            base_channels=base_channels,
            image_size=int(image_size),
            num_classes=num_classes,
            logit_transform=logit_transform,
        )

    def forward(self, x: torch.Tensor, sigma_labels: torch.Tensor) -> torch.Tensor:
        if not self.logit_transform:
            x = 2.0 * x - 1.0
        output = self.backbone(x)

        # Match the original NCSNv2 behaviour: scale by the sigma used
        # for the current label so that the network output is normalized
        # as a score rather than raw residual.
        if self.sigmas.numel() > 0:
            used_sigmas = self.sigmas[sigma_labels].view(x.shape[0], *([1] * len(x.shape[1:])))
            output = output / used_sigmas
        return output
