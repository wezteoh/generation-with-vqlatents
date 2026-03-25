import math
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.modules.autoencoders import VQAutoencoder
from src.modules.discriminators import NLayerDiscriminator, weights_init
from src.modules.losses.gan import adopt_weight, hinge_d_loss, vanilla_d_loss
from src.modules.losses.perceptual import perceptual_loss
from src.modules.losses.vae import reconstruction_loss
from src.modules.perceptual import LPIPS


def _tensor_to_display(t: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) normalized -> (B, H, W, C) uint8 [0, 255] for display."""
    t = t.detach().cpu().float()
    t = (t * 0.5 + 0.5).clamp(0.0, 1.0)
    t = (t * 255.0).round().to(torch.uint8)
    t = t.permute(0, 2, 3, 1)
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)
    return t


def _build_comparison_images(orig: torch.Tensor, recon: torch.Tensor):
    """Build list of wandb.Image: for each sample, side-by-side [original | reconstruction]."""
    import wandb

    o = _tensor_to_display(orig)
    r = _tensor_to_display(recon)
    images = []
    for i in range(orig.shape[0]):
        left = o[i].numpy()
        right = r[i].numpy()
        combined = np.concatenate([left, right], axis=1)
        images.append(wandb.Image(combined, caption=f"orig | recon {i}"))
    return images


def _tensor_to_fid_input(t: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) normalized float -> (B, 3, H, W) uint8 [0, 255] for FID."""
    t = t.detach().float()
    t = (t * 0.5 + 0.5).clamp(0.0, 1.0)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    t = (t * 255.0).round().to(torch.uint8)
    return t


class VQGANInterface(pl.LightningModule):
    def __init__(
        self,
        ddconfig: dict[str, Any],
        disc_config: dict[str, Any],
        disc_start: int,
        perceptual_weight: float,
        discriminator_weight: float,
        codebook_weight: float,
        n_embed: int,
        embed_dim: int,
        image_key: str = "image",
        learning_rate: float = 2e-4,
        disc_learning_rate: float | None = None,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        scheduler: dict[str, Any] | None = None,
        val_logging: dict[str, Any] | None = None,
        gan_loss: str = "hinge",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Lightning >=2 requires manual optimization for multiple optimizers.
        self.automatic_optimization = False
        self.image_key = image_key
        self.disc_start = disc_start
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = discriminator_weight
        self.codebook_weight = codebook_weight
        self.learning_rate = learning_rate
        self.disc_learning_rate = disc_learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.scheduler_cfg = scheduler or {}
        self.gan_loss_name = str(gan_loss).lower()
        self.val_logging_cfg = val_logging or {
            "enabled": True,
            "num_samples": 8,
            "log_every_n_val_epochs": 1,
            "rfid": {
                "enabled": False,
                "feature": 2048,
                "max_samples": 0,
            },
        }
        rfid_cfg = self.val_logging_cfg.get("rfid", {})
        self.rfid_cfg = {
            "enabled": bool(rfid_cfg.get("enabled", False)),
            "feature": int(rfid_cfg.get("feature", 2048)),
            "max_samples": int(rfid_cfg.get("max_samples", 0)),
        }
        self.rfid_metric: nn.Module | None = None
        self._rfid_num_samples = 0
        if self.rfid_cfg["enabled"]:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
            except Exception as exc:
                raise ImportError(
                    "rFID logging requires torchmetrics and torch-fidelity. "
                    "Install both dependencies to enable trainer.val_logging.rfid.enabled=true."
                ) from exc
            self.rfid_metric = FrechetInceptionDistance(
                feature=self.rfid_cfg["feature"],
                reset_real_features=True,
                normalize=False,
            )
        # Validation image logging: interpret log_every_n_val_epochs as
        # "log every N validation passes" (not training epochs).
        self._val_pass_count = 0

        # VQGAN is unconditional in this starter implementation.
        self.disc_conditional = False
        self.disc_factor = 1.0
        self.adopt_disc_weight = adopt_weight
        if self.gan_loss_name == "hinge":
            self.disc_loss = hinge_d_loss
        elif self.gan_loss_name == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError("gan_loss must be one of: hinge, vanilla")

        self.model = VQAutoencoder(
            ddconfig=ddconfig,
            lossconfig={},
            n_embed=n_embed,
            embed_dim=embed_dim,
            image_key=image_key,
        )

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_config["in_channels"],
            n_layers=disc_config["n_layers"],
            use_actnorm=disc_config["use_actnorm"],
            ndf=disc_config["ndf"],
        ).apply(weights_init)
        self.perceptual_model = LPIPS().eval()
        for p in self.perceptual_model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _perceptual_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return perceptual_loss(x, recon, perceptual_model=self.perceptual_model)

    def _generator_loss(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        recon, emb_loss = self.model(x)

        rec_loss = reconstruction_loss(recon, x)
        if self.perceptual_weight > 0:
            p_loss = self._perceptual_loss(x.contiguous(), recon.contiguous()).mean()
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        logits_fake = self.discriminator(recon.contiguous())
        g_loss = -torch.mean(logits_fake)

        d_weight = self.calculate_adaptive_weight(
            rec_loss, g_loss, last_layer=self.get_last_layer()
        )
        disc_factor = self.adopt_disc_weight(
            self.disc_factor, self.global_step, threshold=self.disc_start
        )
        loss = rec_loss + d_weight * disc_factor * g_loss + self.codebook_weight * emb_loss.mean()

        logs = {
            "train/loss": loss.detach().mean(),
            "train/quant_loss": emb_loss.detach().mean(),
            "train/rec_loss": rec_loss.detach().mean(),
            "train/p_loss": p_loss.detach().mean(),
            "train/d_weight": d_weight.detach(),
            "train/disc_factor": torch.tensor(disc_factor, device=x.device),
            "train/g_loss": g_loss.detach().mean(),
        }
        return loss, logs, recon

    def _discriminator_loss(
        self, x: torch.Tensor, recon: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_real = self.discriminator(x.contiguous().detach())
        logits_fake = self.discriminator(recon.contiguous().detach())
        disc_factor = self.adopt_disc_weight(
            self.disc_factor, self.global_step, threshold=self.disc_start
        )
        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
        logs = {
            "train/d_loss": d_loss.detach().mean(),
            "train/logits_real": logits_real.detach().mean(),
            "train/logits_fake": logits_fake.detach().mean(),
        }
        return d_loss, logs

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch[self.image_key]
        opt_g, opt_d = self.optimizers()

        # Generator / autoencoder update.
        opt_g.zero_grad(set_to_none=True)
        g_loss, g_logs, recon = self._generator_loss(x)
        self.manual_backward(g_loss)
        opt_g.step()

        # Discriminator update.
        opt_d.zero_grad(set_to_none=True)
        d_loss, d_logs = self._discriminator_loss(x, recon)
        self.manual_backward(d_loss)
        opt_d.step()

        # Step LR schedulers if configured for step-level updates.
        interval = str(self.scheduler_cfg.get("interval", "step")).lower()
        if interval == "step":
            scheds = self.lr_schedulers()
            if isinstance(scheds, (list, tuple)):
                for s in scheds:
                    s.step()
            elif scheds is not None:
                scheds.step()

        self.log_dict({**g_logs, **d_logs}, on_step=True, on_epoch=True, prog_bar=True)
        return g_loss.detach()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Skip counting/logging during Lightning's sanity-check validation.
        is_sanity = getattr(self.trainer, "sanity_checking", False)
        if not is_sanity and batch_idx == 0:
            self._val_pass_count += 1

        x = batch[self.image_key]
        quant, emb_loss, info = self.model.encode(x)
        perplexity = info[0]
        recon = self.model.decode(quant)

        rec_loss = reconstruction_loss(recon, x)
        if self.perceptual_weight > 0:
            p_loss = self._perceptual_loss(x.contiguous(), recon.contiguous()).mean()
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        # Mirror adversarial computations from training (but avoid adaptive-weight autograd in val).
        logits_fake_g = self.discriminator(recon.contiguous())
        g_loss = -torch.mean(logits_fake_g)
        d_weight = torch.zeros((), device=x.device, dtype=x.dtype)
        disc_factor = self.adopt_disc_weight(
            self.disc_factor, self.global_step, threshold=self.disc_start
        )

        loss = rec_loss + d_weight * disc_factor * g_loss + self.codebook_weight * emb_loss.mean()

        logits_real = self.discriminator(x.contiguous().detach())
        logits_fake = self.discriminator(recon.contiguous().detach())
        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        self.log_dict(
            {
                "val/loss": loss.detach().mean(),
                "val/quant_loss": emb_loss.detach().mean(),
                "val/rec_loss": rec_loss.detach().mean(),
                "val/p_loss": p_loss.detach().mean(),
                "val/d_weight": d_weight.detach(),
                "val/disc_factor": torch.tensor(disc_factor, device=x.device),
                "val/g_loss": g_loss.detach().mean(),
                "val/d_loss": d_loss.detach().mean(),
                "val/logits_real": logits_real.detach().mean(),
                "val/logits_fake": logits_fake.detach().mean(),
                "val/emb_loss": emb_loss.detach().mean(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val/perplexity", perplexity, on_epoch=True)
        self._update_rfid(x, recon)

        enabled = self.val_logging_cfg.get("enabled", True)
        num_samples = int(self.val_logging_cfg.get("num_samples", 8))
        log_every_n = int(self.val_logging_cfg.get("log_every_n_val_epochs", 1))
        if not enabled or batch_idx != 0 or log_every_n <= 0 or self.logger is None:
            return loss

        # We are at batch_idx == 0 of a real validation pass; gate on pass count.
        if self._val_pass_count % log_every_n != 0:
            return loss

        try:
            from pytorch_lightning.loggers import WandbLogger

            if isinstance(self.logger, WandbLogger):
                import wandb

                n = min(num_samples, x.shape[0])
                orig = x[:n].detach()
                rec = recon[:n].detach()
                images = _build_comparison_images(orig, rec)
                self.logger.experiment.log(
                    {"val/reconstructions": images},
                    step=self.global_step,
                )
        except Exception:
            pass

        return loss

    def on_validation_epoch_start(self) -> None:
        if self.rfid_metric is not None:
            self.rfid_metric.reset()
            self._rfid_num_samples = 0

    def on_validation_epoch_end(self) -> None:
        is_sanity = getattr(self.trainer, "sanity_checking", False)
        if is_sanity or self.rfid_metric is None or self._rfid_num_samples <= 0:
            return

        rfid = self.rfid_metric.compute()
        self.log("val/rfid", rfid, on_epoch=True, prog_bar=True, sync_dist=True)

    def _update_rfid(self, x: torch.Tensor, recon: torch.Tensor) -> None:
        is_sanity = getattr(self.trainer, "sanity_checking", False)
        if is_sanity or self.rfid_metric is None:
            return

        max_samples = self.rfid_cfg["max_samples"]
        if max_samples > 0:
            remaining = max_samples - self._rfid_num_samples
            if remaining <= 0:
                return
            n = min(int(x.shape[0]), remaining)
            x = x[:n]
            recon = recon[:n]
        else:
            n = int(x.shape[0])

        real = _tensor_to_fid_input(x)
        fake = _tensor_to_fid_input(recon)
        self.rfid_metric.update(real, real=True)
        self.rfid_metric.update(fake, real=False)
        self._rfid_num_samples += n

    def configure_optimizers(self) -> Any:
        opt_g = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        disc_lr = (
            float(self.disc_learning_rate)
            if self.disc_learning_rate is not None
            else float(self.learning_rate)
        )
        opt_d = AdamW(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        if not self.scheduler_cfg:
            return [opt_g, opt_d]

        warmup_steps = int(self.scheduler_cfg.get("warmup_steps", 500))
        min_lr_ratio = float(self.scheduler_cfg.get("min_lr_ratio", 0.1))

        def lr_lambda(current_step: int) -> float:
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
            if total_steps <= 0:
                return 1.0

            capped_warmup = max(0, min(warmup_steps, total_steps - 1))
            if capped_warmup > 0 and current_step < capped_warmup:
                return float(current_step + 1) / float(capped_warmup)

            if total_steps - capped_warmup <= 0:
                return 1.0

            progress = (current_step - capped_warmup) / float(total_steps - capped_warmup)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        sched_g = LambdaLR(opt_g, lr_lambda=lr_lambda)
        sched_d = LambdaLR(opt_d, lr_lambda=lr_lambda)
        interval = self.scheduler_cfg.get("interval", "step")
        return (
            [opt_g, opt_d],
            [
                {"scheduler": sched_g, "interval": interval, "frequency": 1, "name": "lr_g"},
                {"scheduler": sched_d, "interval": interval, "frequency": 1, "name": "lr_d"},
            ],
        )

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def get_last_layer(self):
        return self.model.decoder.conv_out.weight
