"""Lightning interface for training the unconditional prior over VQ latent indices."""

import math
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR

from src.modules.autoencoders import VQAutoencoder
from src.modules.latents import LatentTransformer
from src.utils.latent_first_stage_ckpt import (
    check_strict_first_stage_load,
    omit_first_stage_keys,
)


def _load_vq_from_ckpt(
    ddconfig: dict[str, Any],
    n_embed: int,
    embed_dim: int,
    ckpt_path: str,
    image_key: str = "image",
) -> VQAutoencoder:
    """Build VQAutoencoder and load weights from a checkpoint (e.g. VQVAEInterface)."""
    vq = VQAutoencoder(
        ddconfig=ddconfig,
        lossconfig={},
        n_embed=n_embed,
        embed_dim=embed_dim,
        image_key=image_key,
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    state_dict = sd.get("state_dict", sd)
    # If checkpoint is from VQVAEInterface, weights live under "model."
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
    vq.load_state_dict(state_dict, strict=True)
    return vq


class TransformerLatentInterface(pl.LightningModule):
    """Train unconditional prior on image batches with a pretrained VQ checkpoint."""

    def __init__(
        self,
        ddconfig: dict[str, Any],
        n_embed: int,
        embed_dim: int,
        vq_ckpt_path: str,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        image_key: str = "image",
        learning_rate: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        scheduler: dict[str, Any] | None = None,
        pkeep: float = 1.0,
        sos_token: int = 0,
        transformer_dropout: float = 0.0,
        val_logging: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.betas = betas
        self.scheduler_cfg = scheduler or {
            "name": "warmup_cosine",
            "warmup_steps": 500,
            "min_lr_ratio": 0.1,
        }
        # Validation image logging: interpret log_every_n_val_epochs as
        # "log every N validation passes" (not training epochs).
        self._val_pass_count = 0
        self.val_logging_cfg = val_logging or {
            "enabled": True,
            "num_samples": 8,
            "log_every_n_val_epochs": 1,
        }

        first_stage = _load_vq_from_ckpt(
            ddconfig=ddconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            ckpt_path=vq_ckpt_path,
            image_key=image_key,
        )
        self.model = LatentTransformer(
            first_stage_model=first_stage,
            vocab_size=n_embed,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            transformer_dropout=transformer_dropout,
            first_stage_key=image_key,
            pkeep=pkeep,
            sos_token=sos_token,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        o = super().state_dict(*args, **kwargs)
        return omit_first_stage_keys(o)

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        filtered = omit_first_stage_keys(state_dict)
        incomp = super().load_state_dict(filtered, strict=False, assign=assign)
        if strict:
            check_strict_first_stage_load(incomp)
        return incomp

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _prior_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = self.model.get_input(self.image_key, batch)
        logits, target = self.model(x)
        loss = F.cross_entropy(
            rearrange(logits, "b t v -> (b t) v"),
            rearrange(target, "b t -> (b t)"),
        )
        return loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._prior_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        # Skip counting/logging during Lightning's sanity-check validation.
        is_sanity = getattr(self.trainer, "sanity_checking", False)
        if not is_sanity and batch_idx == 0:
            self._val_pass_count += 1

        loss = self._prior_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        if not is_sanity:
            self._maybe_log_val_samples(batch, batch_idx)
        return loss

    def _maybe_log_val_samples(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        enabled = self.val_logging_cfg.get("enabled", True)
        num_samples = int(self.val_logging_cfg.get("num_samples", 8))
        log_every_n = int(self.val_logging_cfg.get("log_every_n_val_epochs", 1))
        if not enabled or batch_idx != 0 or log_every_n <= 0 or self.logger is None:
            return
        # We are at batch_idx == 0 of a real validation pass; gate on pass count.
        if self._val_pass_count % log_every_n != 0:
            return
        from pytorch_lightning.loggers import WandbLogger

        import wandb

        if not isinstance(self.logger, WandbLogger):
            return

        x = self.model.get_input(self.image_key, batch)
        n = min(num_samples, x.shape[0])
        _, z_indices = self.model.encode_to_z(x[:n])
        z_shape = (n,) + tuple(self.model.first_stage_model.encode(x[:1])[0].shape[1:])

        vocab_size = self.model.transformer.config.vocab_size
        safe_top_k = min(100, vocab_size)

        samples = self.model.sample_natural(
            batch_size=n,
            z_shape=z_shape,
            temperature=1.0,
            sample=True,
            top_k=safe_top_k,
        )
        orig = (x[:n].detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
        gen = (samples.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
        orig = (orig.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
        gen = (gen.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
        if orig.shape[-1] == 1:
            orig = orig.repeat(1, 1, 1, 3)
            gen = gen.repeat(1, 1, 1, 3)
        images = [
            wandb.Image(orig[i].numpy(), caption=f"input {i}") for i in range(n)
        ] + [wandb.Image(gen[i].numpy(), caption=f"sample {i}") for i in range(n)]
        self.logger.experiment.log(
            {"val/prior_samples": images},
            step=self.global_step,
        )

    def configure_optimizers(self) -> Any:
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
        for pn, _ in self.model.transformer.named_parameters():
            if "ln_" in pn or "ln_f" in pn:
                no_decay.add(pn)
                decay.discard(pn)
        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": 0.01},
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=self.betas
        )

        if self.scheduler_cfg.get("name", "warmup_cosine") != "warmup_cosine":
            return optimizer

        warmup_steps = int(self.scheduler_cfg.get("warmup_steps", 500))
        min_lr_ratio = float(self.scheduler_cfg.get("min_lr_ratio", 0.1))

        def lr_lambda(current_step: int) -> float:
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
            if total_steps <= 0:
                return 1.0

            capped_warmup = min(warmup_steps, total_steps)
            if current_step < capped_warmup:
                return float(current_step + 1) / float(capped_warmup)

            progress = (current_step - capped_warmup) / float(
                total_steps - capped_warmup
            )
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
