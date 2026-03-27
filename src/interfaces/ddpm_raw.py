from __future__ import annotations

from typing import Any, Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger

from src.interfaces.ddpm_sample_fid import run_sample_fid_if_gated
from src.modules.ema import LitEma
from src.modules.latents.ddpm import RawOpenAIUNetDDPM
from src.utils.wandb_comparison import build_side_by_side_wandb_images

ConditioningMode = Literal["none", "class", "context"]


class DDPMRawInterface(pl.LightningModule):
    """DDPM training on raw images (no VQ first stage)."""

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        image_key: str = "image",
        learning_rate: float = 1e-4,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
        parameterization: str = "eps",
        loss_type: str = "l2",
        l_simple_weight: float = 1.0,
        original_elbo_weight: float = 0.0,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        attention_resolutions: Optional[tuple[int, ...]] = None,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
        logit_transform: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        val_logging: Optional[dict[str, Any]] = None,
        sampling_cfg: Optional[dict[str, Any]] = None,
        conditioning_mode: ConditioningMode = "none",
        num_data_classes: Optional[int] = None,
        label_key: str = "label",
        context_key: str = "context",
        unconditional_prob: float = 0.0,
        context_dim: Optional[int] = None,
        transformer_depth: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.image_key = image_key
        self.learning_rate = float(learning_rate)
        self.parameterization = str(parameterization)
        self.loss_type = str(loss_type)
        self.l_simple_weight = float(l_simple_weight)
        self.original_elbo_weight = float(original_elbo_weight)
        self.conditioning_mode = conditioning_mode
        self.label_key = str(label_key)
        self.context_key = str(context_key)
        self.unconditional_prob = float(unconditional_prob)

        self._val_pass_count = 0
        self.val_logging_cfg = val_logging or {
            "enabled": True,
            "num_samples": 8,
            "log_every_n_val_epochs": 1,
        }

        default_sampling_cfg: dict[str, Any] = {
            "clip_denoised": False,
            "guidance_scale": 1.0,
        }
        self.sampling_cfg = default_sampling_cfg
        if sampling_cfg is not None:
            self.sampling_cfg = {**default_sampling_cfg, **sampling_cfg}

        att_res = attention_resolutions
        ch_mult = channel_mult

        model_kw: dict[str, Any] = dict(
            in_channels=int(in_channels),
            image_size=int(image_size),
            num_timesteps=int(timesteps),
            beta_schedule=str(beta_schedule),
            linear_start=float(linear_start),
            linear_end=float(linear_end),
            cosine_s=float(cosine_s),
            parameterization=self.parameterization,
            model_channels=int(base_channels),
            num_res_blocks=int(num_res_blocks),
            attention_resolutions=att_res,
            channel_mult=ch_mult,
            dropout=float(dropout),
            logit_transform=bool(logit_transform),
            conditioning_mode=conditioning_mode,
            transformer_depth=int(transformer_depth),
        )
        if conditioning_mode == "class":
            if num_data_classes is None:
                raise ValueError(
                    "num_data_classes is required when conditioning_mode is 'class'"
                )
            model_kw["num_data_classes"] = int(num_data_classes)
        elif conditioning_mode == "context":
            if context_dim is None:
                raise ValueError(
                    "context_dim is required when conditioning_mode is 'context'"
                )
            model_kw["context_dim"] = int(context_dim)

        self.model = RawOpenAIUNetDDPM(**model_kw)

        if bool(use_ema):
            if not (0.0 < float(ema_decay) <= 1.0):
                raise ValueError(f"ema_decay must be in (0,1], got {ema_decay}")
            self.ema = LitEma(self.model, decay=float(ema_decay))
        else:
            self.ema = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(x, timesteps=timesteps, y=y, context=context)

    def _get_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mean: bool = True
    ) -> torch.Tensor:
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = F.mse_loss(pred, target)
            else:
                loss = F.mse_loss(pred, target, reduction="none")
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

        return loss

    def _p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        stage: str,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_noisy = self.model.q_sample(x_start, t, noise)
        model_out = self.model(x_noisy, t, y=y, context=context)

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                f"Parameterization {self.parameterization} not yet supported"
            )

        per_sample = self._get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        loss_vlb = (self.model.lvlb_weights[t] * per_sample).mean()
        loss_simple = per_sample.mean() * self.l_simple_weight
        total = loss_simple + self.original_elbo_weight * loss_vlb

        self.log(
            f"{stage}/loss_simple",
            per_sample.mean(),
            prog_bar=False,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        self.log(
            f"{stage}/loss_vlb",
            loss_vlb,
            prog_bar=False,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        self.log(
            f"{stage}/loss",
            total,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        return total

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x0 = batch[self.image_key]
        b = x0.shape[0]
        device = x0.device

        t = torch.randint(
            low=0,
            high=self.model.num_timesteps,
            size=(b,),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)

        if self.conditioning_mode == "none":
            return self._p_losses(x0, t, noise, stage=stage)

        if self.conditioning_mode == "class":
            if self.label_key not in batch:
                raise KeyError(
                    f"Batch missing label key '{self.label_key}' for class conditioning"
                )
            y = batch[self.label_key].long().to(device)
            if stage == "train" and self.unconditional_prob > 0.0:
                assert self.model.null_class_index is not None
                mask = torch.rand(b, device=device) < self.unconditional_prob
                y = torch.where(
                    mask,
                    torch.full_like(y, self.model.null_class_index),
                    y,
                )
            return self._p_losses(x0, t, noise, stage=stage, y=y, context=None)

        if self.context_key not in batch:
            raise KeyError(
                f"Batch missing context key {self.context_key!r} (context conditioning)"
            )
        ctx_map = batch[self.context_key].to(device)
        context = rearrange(ctx_map, "b c h w -> b (h w) c")
        if stage == "train" and self.unconditional_prob > 0.0:
            mask = torch.rand(b, device=device) < self.unconditional_prob
            zero = torch.zeros_like(context)
            context = torch.where(mask.view(b, 1, 1), zero, context)
        return self._p_losses(x0, t, noise, stage=stage, y=None, context=context)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        is_sanity = getattr(self.trainer, "sanity_checking", False)
        if not is_sanity and batch_idx == 0:
            self._val_pass_count += 1

        if self.ema is not None and not is_sanity:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model)
            try:
                loss = self._shared_step(batch, stage="val")
                self._maybe_log_val_samples(batch, batch_idx)
            finally:
                self.ema.restore(self.model.parameters())
            return loss

        loss = self._shared_step(batch, stage="val")
        if not is_sanity:
            self._maybe_log_val_samples(batch, batch_idx)
        return loss

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            return

        clip = bool(self.sampling_cfg.get("clip_denoised", False))
        ic = int(self.hparams.in_channels)
        sz = int(self.hparams.image_size)

        def generate_fake_images(n: int, device: torch.device) -> torch.Tensor:
            shape = (n, ic, sz, sz)
            return self.model.sample_latents(
                batch_size=n,
                latent_shape=shape,
                device=device,
                clip_denoised=clip,
                y=None,
                context=None,
                guidance_scale=1.0,
            )

        run_sample_fid_if_gated(
            self,
            val_pass_count=self._val_pass_count,
            image_key=self.image_key,
            val_logging_cfg=self.val_logging_cfg,
            ema=self.ema,
            model=self.model,
            generate_fake_images=generate_fake_images,
        )

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_closure, *args, **kwargs
        )
        if self.ema is not None:
            self.ema(self.model)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _maybe_log_val_samples(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        enabled = self.val_logging_cfg.get("enabled", True)
        num_samples = int(self.val_logging_cfg.get("num_samples", 8))
        log_every_n = int(self.val_logging_cfg.get("log_every_n_val_epochs", 1))
        if not enabled or batch_idx != 0 or log_every_n <= 0 or self.logger is None:
            return
        if self._val_pass_count % log_every_n != 0:
            return
        if not isinstance(self.logger, WandbLogger):
            return

        import wandb

        x_img = batch[self.image_key]
        n = min(num_samples, x_img.shape[0])
        x_img = x_img[:n]
        device = x_img.device
        ic = int(self.hparams.in_channels)
        sz = int(self.hparams.image_size)
        image_shape = (n, ic, sz, sz)
        clip = bool(self.sampling_cfg.get("clip_denoised", False))
        g_scale = float(self.sampling_cfg.get("guidance_scale", 1.0))

        if self.conditioning_mode == "none":
            sampled = self.model.sample_latents(
                batch_size=n,
                latent_shape=image_shape,
                device=device,
                clip_denoised=clip,
                y=None,
                context=None,
                guidance_scale=1.0,
            )
            orig = (x_img.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
            gen = (sampled.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
            orig = (orig.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
            gen = (gen.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
            if orig.shape[-1] == 1:
                orig = orig.repeat(1, 1, 1, 3)
                gen = gen.repeat(1, 1, 1, 3)
            images = [
                wandb.Image(orig[i].numpy(), caption=f"input {i}") for i in range(n)
            ] + [wandb.Image(gen[i].numpy(), caption=f"sample {i}") for i in range(n)]
            log_key = "val/ddpm_raw_samples"
        elif self.conditioning_mode == "class":
            y = batch[self.label_key][:n].long().to(device)
            sampled = self.model.sample_latents(
                batch_size=n,
                latent_shape=image_shape,
                device=device,
                clip_denoised=clip,
                y=y,
                context=None,
                guidance_scale=g_scale,
            )
            lbl = batch[self.label_key][:n].long().cpu().tolist()
            captions = [f"real | gen (label {lbl[i]}) [{i}]" for i in range(n)]
            images = build_side_by_side_wandb_images(x_img, sampled, captions=captions)
            log_key = "val/ddpm_raw_samples"
        else:
            ctx_map = batch[self.context_key][:n].to(device)
            context = rearrange(ctx_map, "b c h w -> b (h w) c")
            sampled = self.model.sample_latents(
                batch_size=n,
                latent_shape=image_shape,
                device=device,
                clip_denoised=clip,
                y=None,
                context=context,
                guidance_scale=g_scale,
            )
            captions = [f"real | gen (context) [{i}]" for i in range(n)]
            images = build_side_by_side_wandb_images(x_img, sampled, captions=captions)
            log_key = "val/ddpm_raw_samples"

        self.logger.experiment.log(
            {log_key: images},
            step=self.global_step,
        )
