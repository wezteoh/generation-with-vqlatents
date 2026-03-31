from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
import torch

from src.interfaces.transformer_latent import _load_vq_from_ckpt
from src.utils.latent_first_stage_ckpt import (
    check_strict_first_stage_load,
    omit_first_stage_keys,
)
from src.modules.ema import LitEma
from src.modules.latents.score_models import ScoreModel
from src.modules.latents.score_models.cond_refinednet import LatentCondRefineNetScore
from src.modules.latents.score_models.ncsnv2 import LatentNCSNv2Score
from src.modules.latents.score_models.unet import LatentUNetScore
from src.modules.losses.score_matching import anneal_dsm_loss, dsm_loss


class DSMInterface(pl.LightningModule):
    """Denoising score matching on raw images or VQ latents."""

    def __init__(
        self,
        min_sigma: float,
        max_sigma: float,
        num_sigmas: int,
        base_channels: int = 64,
        image_key: str = "image",
        learning_rate: float = 2e-4,
        anneal_power: float = 2.0,
        use_annealed_loss: bool = True,
        val_logging: Optional[dict[str, Any]] = None,
        score_backbone: str = "unet",
        use_ema: bool = False,
        ema_decay: float = 0.999,
        vq_ckpt_path: Optional[str] = None,
        ddconfig: Optional[dict[str, Any]] = None,
        n_embed: Optional[int] = None,
        embed_dim: Optional[int] = None,
        in_channels: Optional[int] = None,
        image_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        use_first_stage = (
            vq_ckpt_path is not None and str(vq_ckpt_path).lower() != "null"
        )
        if use_first_stage:
            if ddconfig is None or n_embed is None or embed_dim is None:
                raise ValueError(
                    "DSM latent requires ddconfig, n_embed, embed_dim, vq_ckpt_path"
                )
            first_stage = _load_vq_from_ckpt(
                ddconfig=ddconfig,
                n_embed=int(n_embed),
                embed_dim=int(embed_dim),
                ckpt_path=str(vq_ckpt_path),
                image_key=image_key,
            )
            resolution = int(ddconfig.get("resolution", 32))
            ch_mult = ddconfig.get("ch_mult", (1, 2, 4, 8))
            num_resolutions = len(ch_mult)
            latent_res = resolution // 2 ** (num_resolutions - 1)
            ic = int(embed_dim)
            sz = int(latent_res)
            fs: Optional[torch.nn.Module] = first_stage
        else:
            if in_channels is None or image_size is None:
                raise ValueError("DSM raw requires in_channels and image_size")
            ic = int(in_channels)
            sz = int(image_size)
            fs = None

        self.image_key = image_key
        self.learning_rate = learning_rate
        self.anneal_power = anneal_power
        self.use_annealed_loss = use_annealed_loss
        self._val_pass_count = 0
        self.val_logging_cfg = val_logging or {
            "enabled": True,
            "num_samples": 8,
            "log_every_n_val_epochs": 1,
            "n_steps_each": 20,
            "step_lr": 2e-5,
        }

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)

        max_t = torch.as_tensor(max_sigma, dtype=torch.float32)
        min_t = torch.as_tensor(min_sigma, dtype=torch.float32)
        log_max = torch.log(max_t)
        log_min = torch.log(min_t)
        sigmas = torch.exp(torch.linspace(log_max, log_min, steps=int(num_sigmas)))
        self.register_buffer("sigmas", sigmas, persistent=True)

        backbone: ScoreModel
        if score_backbone == "unet":
            backbone = LatentUNetScore(
                in_channels=ic,
                base_channels=base_channels,
                num_classes=int(num_sigmas),
                first_stage_model=fs,
                image_size=sz,
                logit_transform=False,
            )
        elif score_backbone == "cond_refinenet":
            backbone = LatentCondRefineNetScore(
                in_channels=ic,
                base_channels=base_channels,
                num_classes=int(num_sigmas),
                first_stage_model=fs,
                image_size=sz,
                logit_transform=False,
            )
        elif score_backbone == "ncsnv2":
            backbone = LatentNCSNv2Score(
                in_channels=ic,
                base_channels=base_channels,
                num_classes=int(num_sigmas),
                first_stage_model=fs,
                image_size=sz,
                logit_transform=False,
                sigmas=self.sigmas,
            )
        else:
            raise ValueError(f"Unsupported score_backbone for DSM: {score_backbone}")

        self.model = backbone

        if self.use_ema:
            if not (0.0 < self.ema_decay <= 1.0):
                raise ValueError(f"EMA decay must be in (0, 1], got {self.ema_decay}")
            self.ema = LitEma(self.model, decay=self.ema_decay)
        else:
            self.ema = None

    @property
    def use_first_stage(self) -> bool:
        return self.model.first_stage_model is not None

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        o = super().state_dict(*args, **kwargs)
        if self.use_first_stage:
            return omit_first_stage_keys(o)
        return o

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        if not self.use_first_stage:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        filtered = omit_first_stage_keys(state_dict)
        incomp = super().load_state_dict(filtered, strict=False, assign=assign)
        if strict:
            check_strict_first_stage_load(incomp)
        return incomp

    def forward(self, x: torch.Tensor, sigma_labels: torch.Tensor) -> torch.Tensor:
        return self.model(x, sigma_labels)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x_img = batch[self.image_key]
        if self.use_first_stage:
            fs = self.model.first_stage_model
            assert fs is not None
            with torch.no_grad():
                latents, _, _ = fs.encode(x_img)
            x = latents
        else:
            x = x_img

        b = x.shape[0]
        device = x.device

        sigma_labels = torch.randint(
            low=0,
            high=self.sigmas.shape[0],
            size=(b,),
            device=device,
        )
        used_sigmas = self.sigmas[sigma_labels]

        noise = torch.randn_like(x)
        used_sigmas_broadcast = used_sigmas.view(b, 1, 1, 1)
        perturbed = x + noise * used_sigmas_broadcast

        scores = self(perturbed, sigma_labels)

        if self.use_annealed_loss and self.sigmas.shape[0] > 1:
            loss = anneal_dsm_loss(
                scores=scores,
                samples=x,
                perturbed_samples=perturbed,
                used_sigmas=used_sigmas,
                anneal_power=self.anneal_power,
            )
        else:
            loss = dsm_loss(
                scores=scores,
                samples=x,
                perturbed_samples=perturbed,
                sigma=self.sigmas[0],
            )

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        return loss

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
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        enabled = self.val_logging_cfg.get("enabled", True)
        num_samples = int(self.val_logging_cfg.get("num_samples", 8))
        log_every_n = int(self.val_logging_cfg.get("log_every_n_val_epochs", 1))
        n_steps_each_sigma = int(self.val_logging_cfg.get("n_steps_each_sigma", 20))
        step_lr = float(self.val_logging_cfg.get("step_lr", 2e-5))
        if not enabled or batch_idx != 0 or log_every_n <= 0 or self.logger is None:
            return
        if self._val_pass_count % log_every_n != 0:
            return
        from pytorch_lightning.loggers import WandbLogger

        import wandb

        if not isinstance(self.logger, WandbLogger):
            return

        x_img = batch[self.image_key]
        n = min(num_samples, x_img.shape[0])
        x_img = x_img[:n]
        device = x_img.device

        if self.use_first_stage:
            fs = self.model.first_stage_model
            assert fs is not None
            with torch.no_grad():
                latents, _, _ = fs.encode(x_img)
            latent_shape = latents.shape
            sampled_latents = self.model.sample_latents(
                batch_size=n,
                latent_shape=latent_shape,
                sigmas=self.sigmas.tolist(),
                n_steps_each=n_steps_each_sigma,
                step_lr=step_lr,
                device=latents.device,
            )
            gen_tensor = self.model.quantize_and_decode(sampled_latents)
        else:
            image_shape = (
                n,
                int(self.hparams.in_channels),
                int(self.hparams.image_size),
                int(self.hparams.image_size),
            )
            gen_tensor = self.model.sample_latents(
                batch_size=n,
                latent_shape=image_shape,
                sigmas=self.sigmas.tolist(),
                n_steps_each=n_steps_each_sigma,
                step_lr=step_lr,
                device=device,
            )

        orig = (x_img.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
        gen = (gen_tensor.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
        orig = (orig.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
        gen = (gen.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
        if orig.shape[-1] == 1:
            orig = orig.repeat(1, 1, 1, 3)
            gen = gen.repeat(1, 1, 1, 3)

        images = [
            wandb.Image(orig[i].numpy(), caption=f"input {i}") for i in range(n)
        ] + [wandb.Image(gen[i].numpy(), caption=f"sample {i}") for i in range(n)]
        self.logger.experiment.log(
            {"val/dsm_samples": images},
            step=self.global_step,
        )
