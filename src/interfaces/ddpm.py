from __future__ import annotations

from typing import Any, Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger

from src.utils.latent_first_stage_ckpt import (
    check_strict_first_stage_load,
    omit_first_stage_keys,
)
from src.utils.sample_fid import run_sample_fid_if_gated
from src.interfaces.transformer_latent import _load_vq_from_ckpt
from src.modules.context_encoder import build_context_encoder, set_encoder_trainable
from src.modules.ema import LitEma
from src.modules.latents.ddpm import OpenAIUNetDDPM
from src.utils.wandb_comparison import (
    build_side_by_side_wandb_images,
    build_triplet_wandb_images,
)

ConditioningMode = Literal["none", "class", "context"]


class DDPMInterface(pl.LightningModule):
    """DDPM on raw pixels or VQ latents (OpenAI UNet); first-stage VQ selects mode."""

    def __init__(
        self,
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
        context_apply: str = "cross_attention",
        cond_channels: Optional[int] = None,
        context_encoder_cfg: Optional[dict[str, Any]] = None,
        encoder_trainable: bool = True,
        transformer_depth: int = 1,
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
                    "Latent DDPM requires ddconfig, n_embed, embed_dim, vq_ckpt_path"
                )
            first_stage = _load_vq_from_ckpt(
                ddconfig=ddconfig,
                n_embed=int(n_embed),
                embed_dim=int(embed_dim),
                ckpt_path=str(vq_ckpt_path),
                image_key=image_key,
            )
            resolution = int(ddconfig.get("resolution", 32))
            ch_mult_cfg = ddconfig.get("ch_mult", (1, 2, 4, 8))
            num_resolutions = len(ch_mult_cfg)
            latent_res = resolution // 2 ** (num_resolutions - 1)
            unet_in = int(embed_dim)
            unet_size = int(latent_res)
            first_mod: Optional[torch.nn.Module] = first_stage
        else:
            if in_channels is None or image_size is None:
                raise ValueError("Raw DDPM requires in_channels and image_size")
            unet_in = int(in_channels)
            unet_size = int(image_size)
            first_mod = None

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
        self.context_apply = str(context_apply)
        self.encoder_trainable = bool(encoder_trainable)
        self._cached_enc_uncond: Optional[torch.Tensor] = None

        self._val_pass_count = 0
        self.val_logging_cfg = val_logging or {
            "enabled": True,
            "num_samples": 8,
            "log_every_n_val_epochs": 1,
        }

        default_sampling_cfg: dict[str, Any] = {
            "clip_denoised": False,
            "guidance_scale": 1.0,
            "method": "ancestral_sampling",
        }
        self.sampling_cfg = default_sampling_cfg
        if sampling_cfg is not None:
            self.sampling_cfg = {**default_sampling_cfg, **sampling_cfg}

        att_res = attention_resolutions
        ch_mult_t = channel_mult

        model_kw: dict[str, Any] = dict(
            in_channels=unet_in,
            image_size=unet_size,
            first_stage_model=first_mod,
            num_timesteps=int(timesteps),
            beta_schedule=str(beta_schedule),
            linear_start=float(linear_start),
            linear_end=float(linear_end),
            cosine_s=float(cosine_s),
            parameterization=self.parameterization,
            model_channels=int(base_channels),
            num_res_blocks=int(num_res_blocks),
            attention_resolutions=att_res,
            channel_mult=ch_mult_t,
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
            model_kw["context_apply"] = self.context_apply
            if self.context_apply == "cross_attention":
                if context_dim is None:
                    raise ValueError(
                        "context_dim is required when conditioning_mode is 'context' "
                        "and context_apply is 'cross_attention'"
                    )
                model_kw["context_dim"] = int(context_dim)
                model_kw["cond_channels"] = None
            elif self.context_apply == "concat":
                if cond_channels is None:
                    raise ValueError(
                        "cond_channels is required when conditioning_mode is 'context' "
                        "and context_apply is 'concat'"
                    )
                model_kw["cond_channels"] = int(cond_channels)
                model_kw["context_dim"] = None
            else:
                raise ValueError(
                    f"context_apply must be 'cross_attention' or 'concat', "
                    f"got {self.context_apply!r}"
                )

        self.model = OpenAIUNetDDPM(**model_kw)

        if conditioning_mode == "context":
            self.context_encoder = build_context_encoder(context_encoder_cfg)
            set_encoder_trainable(self.context_encoder, self.encoder_trainable)
        else:
            self.context_encoder = None

        if bool(use_ema):
            if not (0.0 < float(ema_decay) <= 1.0):
                raise ValueError(f"ema_decay must be in (0,1], got {ema_decay}")
            self.ema = LitEma(self.model, decay=float(ema_decay))
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

    def on_train_epoch_start(self) -> None:
        if self.conditioning_mode == "context" and self.encoder_trainable:
            self._cached_enc_uncond = None

    def _get_raw_unconditional(self, ref_bchw: torch.Tensor) -> torch.Tensor:
        dm = (
            getattr(self.trainer, "datamodule", None)
            if self.trainer is not None
            else None
        )
        if dm is not None:
            try:
                return dm.unconditional_context
            except NotImplementedError:
                pass
        return torch.zeros(
            1,
            ref_bchw.shape[1],
            ref_bchw.shape[2],
            ref_bchw.shape[3],
            device=ref_bchw.device,
            dtype=ref_bchw.dtype,
        )

    def _ensure_uncond_encoded_cache(
        self, enc_sample: torch.Tensor, ref_raw: torch.Tensor
    ) -> None:
        if self.encoder_trainable or self._cached_enc_uncond is not None:
            return
        raw_u = self._get_raw_unconditional(ref_raw).to(
            ref_raw.device, dtype=ref_raw.dtype
        )
        with torch.no_grad():
            enc_u = self.context_encoder(raw_u)
        if enc_u.shape[1:] != enc_sample.shape[1:]:
            raise ValueError(
                f"Unconditional encoded shape {tuple(enc_u.shape)} != "
                f"conditional {tuple(enc_sample.shape)}"
            )
        self._cached_enc_uncond = enc_u.detach()

    def _encoded_to_model_inputs(
        self, enc: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        h = w = self.model.image_size
        if enc.shape[-2] != h or enc.shape[-1] != w:
            raise ValueError(
                f"Context encoder spatial {tuple(enc.shape[-2:])} must match "
                f"UNet ({h}, {w})"
            )
        if self.model.context_apply == "cross_attention":
            cd = self.model.context_dim
            assert cd is not None
            if enc.shape[1] != cd:
                raise ValueError(
                    f"Encoded context channels {enc.shape[1]} != context_dim {cd}"
                )
            return rearrange(enc, "b c hh ww -> b (hh ww) c"), None
        cc = self.model.cond_channels
        assert cc is not None
        if enc.shape[1] != cc:
            raise ValueError(
                f"Encoded context channels {enc.shape[1]} != cond_channels {cc}"
            )
        return None, enc

    def _unconditional_encoded_batched(
        self, batch_size: int, enc_cond: torch.Tensor, ref_raw: torch.Tensor
    ) -> torch.Tensor:
        self._ensure_uncond_encoded_cache(enc_cond, ref_raw)
        if self._cached_enc_uncond is None:
            raw_u = self._get_raw_unconditional(ref_raw).to(
                ref_raw.device, dtype=ref_raw.dtype
            )
            enc_u = self.context_encoder(raw_u)
            return enc_u.expand(batch_size, -1, -1, -1)
        return self._cached_enc_uncond.expand(batch_size, -1, -1, -1).to(
            enc_cond.device, dtype=enc_cond.dtype
        )

    def _sample_latents_kwargs_from_sampling_cfg(self) -> dict[str, Any]:
        """Map ``sampling_cfg`` to ``OpenAIUNetDDPM.sample_latents`` keyword args."""
        sc = self.sampling_cfg
        method = str(sc.get("method", "ancestral_sampling"))
        out: dict[str, Any] = {"method": method}
        if method in ("dpmsolver", "dpm_solver"):
            out["dpmsolver_steps"] = int(sc.get("dpmsolver_steps", 20))
            out["dpmsolver_order"] = int(sc.get("dpmsolver_order", 3))
            out["dpmsolver_inner_method"] = str(
                sc.get("dpmsolver_inner_method", "singlestep")
            )
            out["dpmsolver_skip_type"] = str(
                sc.get("dpmsolver_skip_type", "time_uniform")
            )
            out["dpmsolver_algorithm_type"] = str(
                sc.get("dpmsolver_algorithm_type", "dpmsolver++")
            )
            out["dpmsolver_denoise_to_zero"] = bool(
                sc.get("dpmsolver_denoise_to_zero", False)
            )
        return out

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        cond_spatial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            x,
            timesteps=timesteps,
            y=y,
            context=context,
            cond_spatial=cond_spatial,
        )

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
        cond_spatial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_noisy = self.model.q_sample(x_start, t, noise)
        model_out = self.model(
            x_noisy, t, y=y, context=context, cond_spatial=cond_spatial
        )

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
            return self._p_losses(
                x0, t, noise, stage=stage, y=y, context=None, cond_spatial=None
            )

        if self.context_key not in batch:
            raise KeyError(
                f"Batch missing context key {self.context_key!r} (context conditioning)"
            )
        assert self.context_encoder is not None
        ctx_map = batch[self.context_key].to(device)
        enc_c = self.context_encoder(ctx_map)
        context, cond_spatial = self._encoded_to_model_inputs(enc_c)
        if stage == "train" and self.unconditional_prob > 0.0:
            mask = torch.rand(b, device=device) < self.unconditional_prob
            enc_u = self._unconditional_encoded_batched(b, enc_c, ctx_map)
            ctx_u, sp_u = self._encoded_to_model_inputs(enc_u)
            if self.model.context_apply == "cross_attention":
                assert context is not None and ctx_u is not None
                context = torch.where(mask.view(b, 1, 1), ctx_u, context)
            else:
                assert cond_spatial is not None and sp_u is not None
                cond_spatial = torch.where(mask.view(b, 1, 1, 1), sp_u, cond_spatial)
        return self._p_losses(
            x0,
            t,
            noise,
            stage=stage,
            y=None,
            context=context,
            cond_spatial=cond_spatial,
        )

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
                y=None,
                context=None,
                guidance_scale=1.0,
                **self._sample_latents_kwargs_from_sampling_cfg(),
            )
            if self.use_first_stage:
                return self.model.quantize_and_decode(lat)
            return lat

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
        c = self.model.in_channels
        h = w = self.model.image_size
        sample_shape = (n, c, h, w)
        clip = bool(self.sampling_cfg.get("clip_denoised", False))
        g_scale = float(self.sampling_cfg.get("guidance_scale", 1.0))

        if self.use_first_stage:
            fs = self.model.first_stage_model
            assert fs is not None
            with torch.no_grad():
                latents, _, _ = fs.encode(x_img)
            latent_shape = latents.shape
        else:
            latent_shape = sample_shape

        log_key = (
            "val/ddpm_raw_samples" if not self.use_first_stage else "val/ddpm_samples"
        )

        if self.conditioning_mode == "none":
            sampled = self.model.sample_latents(
                batch_size=n,
                latent_shape=latent_shape,
                device=device,
                clip_denoised=clip,
                y=None,
                context=None,
                guidance_scale=1.0,
                **self._sample_latents_kwargs_from_sampling_cfg(),
            )
            if self.use_first_stage:
                gen_tensor = self.model.quantize_and_decode(sampled)
            else:
                gen_tensor = sampled
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
        elif self.conditioning_mode == "class":
            y = batch[self.label_key][:n].long().to(device)
            sampled = self.model.sample_latents(
                batch_size=n,
                latent_shape=latent_shape,
                device=device,
                clip_denoised=clip,
                y=y,
                context=None,
                guidance_scale=g_scale,
                **self._sample_latents_kwargs_from_sampling_cfg(),
            )
            if self.use_first_stage:
                gen_tensor = self.model.quantize_and_decode(sampled)
            else:
                gen_tensor = sampled
            lbl = batch[self.label_key][:n].long().cpu().tolist()
            captions = [f"real | gen (label {lbl[i]}) [{i}]" for i in range(n)]
            images = build_side_by_side_wandb_images(
                x_img, gen_tensor, captions=captions
            )
        else:
            assert self.context_encoder is not None
            ctx_map = batch[self.context_key][:n].to(device)
            enc_c = self.context_encoder(ctx_map)
            ctx_c, sp_c = self._encoded_to_model_inputs(enc_c)
            ctx_u: Optional[torch.Tensor] = None
            sp_u: Optional[torch.Tensor] = None
            if g_scale != 1.0:
                enc_u = self._unconditional_encoded_batched(n, enc_c, ctx_map)
                ctx_u, sp_u = self._encoded_to_model_inputs(enc_u)
            if self.model.context_apply == "cross_attention":
                sampled = self.model.sample_latents(
                    batch_size=n,
                    latent_shape=latent_shape,
                    device=device,
                    clip_denoised=clip,
                    y=None,
                    context=ctx_c,
                    context_uncond=ctx_u,
                    cond_spatial=None,
                    cond_spatial_uncond=None,
                    guidance_scale=g_scale,
                    **self._sample_latents_kwargs_from_sampling_cfg(),
                )
            else:
                sampled = self.model.sample_latents(
                    batch_size=n,
                    latent_shape=latent_shape,
                    device=device,
                    clip_denoised=clip,
                    y=None,
                    context=None,
                    context_uncond=None,
                    cond_spatial=sp_c,
                    cond_spatial_uncond=sp_u,
                    guidance_scale=g_scale,
                    **self._sample_latents_kwargs_from_sampling_cfg(),
                )
            if self.use_first_stage:
                gen_tensor = self.model.quantize_and_decode(sampled)
            else:
                gen_tensor = sampled
            cm = ctx_map
            if cm.dim() == 4 and cm.shape[1] in (1, 3):
                cap = [f"real | cond | gen [{i}]" for i in range(n)]
                images = build_triplet_wandb_images(x_img, cm, gen_tensor, captions=cap)
            else:
                cap = [f"real | gen (context) [{i}]" for i in range(n)]
                images = build_side_by_side_wandb_images(
                    x_img, gen_tensor, captions=cap
                )

        self.logger.experiment.log(
            {log_key: images},
            step=self.global_step,
        )
