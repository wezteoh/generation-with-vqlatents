from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from src.interfaces.transformer_latent import _load_vq_from_ckpt
from src.modules.ema import LitEma
from src.modules.latents.score_sde.ncsnv2 import LatentNCSNv2ScoreSDE
from src.modules.losses.score_sde import score_sde_loss
from src.modules.sde import SDE, VESDE, VPSDE


def _append_dims(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    while x.dim() < target_dim:
        x = x.unsqueeze(-1)
    return x


class ScoreSDEBase(pl.LightningModule):
    """Score matching on raw images or VQ latents under an SDE."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        sde_type: str = "vesde",
        sde_n: int = 1000,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        continuous: bool = True,
        t_eps: float = 1e-3,
        likelihood_weighting: bool = True,
        base_channels: int = 64,
        logit_transform: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        val_logging: Optional[dict[str, Any]] = None,
        sampling_cfg: Optional[dict[str, Any]] = None,
        image_key: str = "image",
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
                    "Score-SDE latent needs ddconfig, n_embed, embed_dim, vq_ckpt_path"
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
                raise ValueError("Score-SDE raw requires in_channels and image_size")
            ic = int(in_channels)
            sz = int(image_size)
            fs = None

        self.image_key = image_key
        self.learning_rate = float(learning_rate)
        self._val_pass_count = 0
        self.val_logging_cfg = val_logging or {
            "enabled": True,
            "num_samples": 8,
            "log_every_n_val_epochs": 1,
        }

        if not (0.0 < t_eps < 1.0):
            raise ValueError(f"t_eps should be in (0,1), got {t_eps}")
        self.t_eps = float(t_eps)

        sde_type_l = str(sde_type).lower()
        if sde_type_l == "vesde":
            self.sde: SDE = VESDE(
                sigma_min=float(sigma_min),
                sigma_max=float(sigma_max),
                N=int(sde_n),
            )
        elif sde_type_l == "vpsde":
            self.sde = VPSDE(
                beta_min=float(beta_min),
                beta_max=float(beta_max),
                N=int(sde_n),
            )
        else:
            raise ValueError(f"Unsupported sde_type: {sde_type}")

        self.continuous = bool(continuous)
        self.likelihood_weighting = bool(likelihood_weighting)

        default_sampling_cfg: dict[str, Any] = {
            "sampling": {
                "method": "pc",
                "predictor": "euler_maruyama",
                "corrector": "langevin",
                "snr": 0.3,
                "n_steps_each": 1,
                "probability_flow": False,
                "noise_removal": True,
            },
            "training": {"continuous": self.continuous},
        }
        self.sampling_cfg = default_sampling_cfg
        if sampling_cfg is not None:
            merged = default_sampling_cfg.copy()
            merged["sampling"] = {
                **default_sampling_cfg["sampling"],
                **sampling_cfg.get("sampling", {}),
            }
            merged["training"] = {
                **default_sampling_cfg["training"],
                **sampling_cfg.get("training", {}),
            }
            self.sampling_cfg = merged

        self.model = LatentNCSNv2ScoreSDE(
            sde=self.sde,
            in_channels=ic,
            base_channels=int(base_channels),
            first_stage_model=fs,
            image_size=sz,
            logit_transform=bool(logit_transform),
        )

        if bool(use_ema):
            if not (0.0 < float(ema_decay) <= 1.0):
                raise ValueError(f"ema_decay must be in (0,1], got {ema_decay}")
            self.ema = LitEma(self.model, decay=float(ema_decay))
        else:
            self.ema = None

    @property
    def use_first_stage(self) -> bool:
        return self.model.first_stage_model is not None

    def forward(
        self,
        x: torch.Tensor,
        sigmas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(x, sigmas=sigmas)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x_img = batch[self.image_key]
        if self.use_first_stage:
            fs = self.model.first_stage_model
            assert fs is not None
            with torch.no_grad():
                latents, _, _ = fs.encode(x_img)
            x0 = latents
        else:
            x0 = x_img

        b = x0.shape[0]
        device = x0.device

        if self.continuous:
            t = torch.rand((b,), device=device, dtype=torch.float32) * (
                float(self.sde.T) - float(self.t_eps)
            ) + float(self.t_eps)
        else:
            if not hasattr(self.sde, "discrete_sigmas"):
                raise NotImplementedError(
                    "Non-continuous Score-SDE needs SDE.discrete_sigmas "
                    f"(got {type(self.sde).__name__})."
                )
            sigma_labels = torch.randint(
                low=0,
                high=self.sde.discrete_sigmas.numel(),
                size=(b,),
                device=device,
                dtype=torch.long,
            )
            t = (
                sigma_labels.to(dtype=torch.float32)
                * float(self.sde.T)
                / float(self.sde.N - 1)
            )

        mean, std = self.sde.marginal_prob(x0, t)
        z = torch.randn_like(x0)
        std_b = _append_dims(std, x0.dim())
        perturbed = mean + std_b * z

        scores = self(perturbed, sigmas=std)

        loss = score_sde_loss(
            scores=scores,
            std=std,
            z=z,
            t=t,
            sde=self.sde,
            likelihood_weighting=self.likelihood_weighting,
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

        if self.use_first_stage:
            fs = self.model.first_stage_model
            assert fs is not None
            with torch.no_grad():
                latents, _, _ = fs.encode(x_img)
            latent_shape = latents.shape
            sampled_latents = self.model.sample_latents(
                batch_size=n,
                latent_shape=latent_shape,
                device=latents.device,
                n_steps=self.val_logging_cfg.get("sample_n_steps"),
                sampling_cfg=self.sampling_cfg,
            )
            gen_tensor = self.model.quantize_and_decode(sampled_latents)
        else:
            with torch.no_grad():
                image_shape = tuple(x_img.shape)
                gen_tensor = self.model.sample_latents(
                    batch_size=n,
                    latent_shape=image_shape,
                    device=x_img.device,
                    sampling_cfg=self.sampling_cfg,
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
            {"val/score_sde_samples": images},
            step=self.global_step,
        )


class ScoreSDEInterface(ScoreSDEBase):
    """Latent Score-SDE (VQ first stage); thin wrapper for checkpoint compatibility."""

    def __init__(
        self,
        ddconfig: dict[str, Any],
        n_embed: int,
        embed_dim: int,
        vq_ckpt_path: str,
        image_key: str = "image",
        learning_rate: float = 1e-4,
        sde_type: str = "vesde",
        sde_n: int = 1000,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        continuous: bool = True,
        t_eps: float = 1e-3,
        likelihood_weighting: bool = True,
        base_channels: int = 64,
        logit_transform: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        val_logging: Optional[dict[str, Any]] = None,
        sampling_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            sde_type=sde_type,
            sde_n=sde_n,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            continuous=continuous,
            t_eps=t_eps,
            likelihood_weighting=likelihood_weighting,
            base_channels=base_channels,
            logit_transform=logit_transform,
            use_ema=use_ema,
            ema_decay=ema_decay,
            val_logging=val_logging,
            sampling_cfg=sampling_cfg,
            image_key=image_key,
            vq_ckpt_path=vq_ckpt_path,
            ddconfig=ddconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            in_channels=None,
            image_size=None,
        )


class ScoreSDERawInterface(ScoreSDEBase):
    """Raw Score-SDE; thin wrapper for checkpoint compatibility."""

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        learning_rate: float = 1e-4,
        sde_type: str = "vesde",
        sde_n: int = 1000,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        continuous: bool = True,
        t_eps: float = 1e-3,
        likelihood_weighting: bool = True,
        base_channels: int = 64,
        logit_transform: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        val_logging: Optional[dict[str, Any]] = None,
        sampling_cfg: Optional[dict[str, Any]] = None,
        image_key: str = "image",
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            sde_type=sde_type,
            sde_n=sde_n,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            continuous=continuous,
            t_eps=t_eps,
            likelihood_weighting=likelihood_weighting,
            base_channels=base_channels,
            logit_transform=logit_transform,
            use_ema=use_ema,
            ema_decay=ema_decay,
            val_logging=val_logging,
            sampling_cfg=sampling_cfg,
            image_key=image_key,
            vq_ckpt_path=None,
            ddconfig=None,
            n_embed=None,
            embed_dim=None,
            in_channels=in_channels,
            image_size=image_size,
        )
