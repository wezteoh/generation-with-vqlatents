from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from src.interfaces.ddpm_sample_fid import run_sample_fid_if_gated
from src.interfaces.transformer_latent import _load_vq_from_ckpt
from src.modules.ema import LitEma
from src.modules.latents.ddpm import LatentOpenAIUNetDDPM


class DDPMLatentInterface(pl.LightningModule):
    """DDPM training on VQ latents using an OpenAI-style UNet backbone."""

    def __init__(
        self,
        ddconfig: dict[str, Any],
        n_embed: int,
        embed_dim: int,
        vq_ckpt_path: str,
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
        logit_transform: bool = True,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        val_logging: Optional[dict[str, Any]] = None,
        sampling_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.image_key = image_key
        self.learning_rate = float(learning_rate)
        self.parameterization = str(parameterization)
        self.loss_type = str(loss_type)
        self.l_simple_weight = float(l_simple_weight)
        self.original_elbo_weight = float(original_elbo_weight)

        self._val_pass_count = 0
        self.val_logging_cfg = val_logging or {
            "enabled": True,
            "num_samples": 8,
            "log_every_n_val_epochs": 1,
        }

        default_sampling_cfg: dict[str, Any] = {
            "clip_denoised": False,
        }
        self.sampling_cfg = default_sampling_cfg
        if sampling_cfg is not None:
            self.sampling_cfg = {**default_sampling_cfg, **sampling_cfg}

        first_stage = _load_vq_from_ckpt(
            ddconfig=ddconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            ckpt_path=vq_ckpt_path,
            image_key=image_key,
        )

        resolution = int(ddconfig.get("resolution", 32))
        ch_mult_cfg = ddconfig.get("ch_mult", (1, 2, 4, 8))
        num_resolutions = len(ch_mult_cfg)
        latent_res = resolution // 2 ** (num_resolutions - 1)

        self.model = LatentOpenAIUNetDDPM(
            first_stage_model=first_stage,
            in_channels=int(embed_dim),
            image_size=int(latent_res),
            num_timesteps=int(timesteps),
            beta_schedule=str(beta_schedule),
            linear_start=float(linear_start),
            linear_end=float(linear_end),
            cosine_s=float(cosine_s),
            parameterization=self.parameterization,
            model_channels=int(base_channels),
            num_res_blocks=int(num_res_blocks),
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            dropout=float(dropout),
            logit_transform=bool(logit_transform),
        )

        if bool(use_ema):
            if not (0.0 < float(ema_decay) <= 1.0):
                raise ValueError(f"ema_decay must be in (0,1], got {ema_decay}")
            self.ema = LitEma(self.model, decay=float(ema_decay))
        else:
            self.ema = None

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps=timesteps)

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
    ) -> torch.Tensor:
        x_noisy = self.model.q_sample(x_start, t, noise)
        model_out = self(x_noisy, t)

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
        x_img = batch[self.image_key]
        with torch.no_grad():
            latents, _, _ = self.model.first_stage_model.encode(x_img)

        x0 = latents
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

        return self._p_losses(x0, t, noise, stage=stage)

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

        def generate_fake_images(n: int, device: torch.device) -> torch.Tensor:
            c = self.model.in_channels
            h = w = self.model.image_size
            latent_shape = (n, c, h, w)
            lat = self.model.sample_latents(
                batch_size=n,
                latent_shape=latent_shape,
                device=device,
                clip_denoised=clip,
            )
            return self.model.quantize_and_decode(lat)

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
        with torch.no_grad():
            latents, _, _ = self.model.first_stage_model.encode(x_img)

        latent_shape = latents.shape
        clip = bool(self.sampling_cfg.get("clip_denoised", False))

        sampled_latents = self.model.sample_latents(
            batch_size=n,
            latent_shape=latent_shape,
            device=latents.device,
            clip_denoised=clip,
        )
        decoded = self.model.quantize_and_decode(sampled_latents)

        orig = (x_img.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
        gen = (decoded.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
        orig = (orig.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
        gen = (gen.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)

        if orig.shape[-1] == 1:
            orig = orig.repeat(1, 1, 1, 3)
            gen = gen.repeat(1, 1, 1, 3)

        images = [
            wandb.Image(orig[i].numpy(), caption=f"input {i}") for i in range(n)
        ] + [wandb.Image(gen[i].numpy(), caption=f"sample {i}") for i in range(n)]
        self.logger.experiment.log(
            {"val/ddpm_samples": images},
            step=self.global_step,
        )
