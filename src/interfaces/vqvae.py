import math
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.modules.autoencoders import VQAutoencoder
from src.modules.losses.vae import reconstruction_loss
from src.utils.fid_inputs import tensor_to_fid_input


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


class VQVAEInterface(pl.LightningModule):
    def __init__(
        self,
        ddconfig: dict[str, Any],
        n_embed: int,
        embed_dim: int,
        image_key: str = "image",
        learning_rate: float = 2e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        scheduler: dict[str, Any] | None = None,
        val_logging: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.scheduler_cfg = scheduler or {
            "name": "warmup_cosine",
            "warmup_steps": 500,
            "min_lr_ratio": 0.1,
        }
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

        self.model = VQAutoencoder(
            ddconfig=ddconfig,
            lossconfig={},
            n_embed=n_embed,
            embed_dim=embed_dim,
            image_key=image_key,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x = batch[self.image_key]
        recon, emb_loss = self.model(x)
        rec_loss = reconstruction_loss(recon, x)
        loss = rec_loss + emb_loss

        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}/rec_loss", rec_loss, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/emb_loss", emb_loss, on_step=(stage == "train"), on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

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
        loss = rec_loss + emb_loss

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/rec_loss", rec_loss, on_epoch=True)
        self.log("val/emb_loss", emb_loss, on_epoch=True)
        self.log("val/perplexity", perplexity, on_epoch=True)
        self._update_rfid(x, recon)

        enabled = self.val_logging_cfg.get("enabled", True)
        num_samples = int(self.val_logging_cfg.get("num_samples", 8))
        log_every_n = int(self.val_logging_cfg.get("log_every_n_val_epochs", 1))
        if (
            not enabled
            or batch_idx != 0
            or log_every_n <= 0
            or self.logger is None
        ):
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

        real = tensor_to_fid_input(x)
        fake = tensor_to_fid_input(recon)
        self.rfid_metric.update(real, real=True)
        self.rfid_metric.update(fake, real=False)
        self._rfid_num_samples += n

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_cfg.get("name", "warmup_cosine") != "warmup_cosine":
            return optimizer

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

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.scheduler_cfg.get("interval", "step"),
                "frequency": 1,
                "name": "lr",
            },
        }
