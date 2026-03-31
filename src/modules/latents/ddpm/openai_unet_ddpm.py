from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from src.modules.latents.diffusion_backbones.openai_unet import UNetModel
from src.modules.latents.diffusion_backbones.utils import (
    extract_into_tensor,
    make_beta_schedule,
)
from src.modules.sampling.dpm_solver import (
    DPM_Solver,
    NoiseScheduleVP,
    model_wrapper,
)


def _freeze_first_stage(m: nn.Module) -> None:
    m.eval()
    m.train = lambda *_args, **_kwargs: None  # type: ignore[assignment]


def default_attention_resolutions(image_size: int) -> tuple[int, ...]:
    """Attention `ds` values matching the OpenAI UNet downsampling pattern."""
    if image_size >= 32:
        return (8,)
    if image_size >= 16:
        return (4, 8)
    return (4,)


ConditioningMode = Literal["none", "class", "context"]
ContextApply = Literal["cross_attention", "concat"]


class OpenAIUNetDDPM(nn.Module):
    """DDPM noise predictor (OpenAI UNet); optional frozen VQ first stage."""

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        *,
        first_stage_model: Optional[nn.Module] = None,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
        parameterization: str = "eps",
        model_channels: int = 64,
        num_res_blocks: int = 2,
        attention_resolutions: Optional[tuple[int, ...]] = None,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
        logit_transform: bool = False,
        conditioning_mode: ConditioningMode = "none",
        num_data_classes: Optional[int] = None,
        context_dim: Optional[int] = None,
        context_apply: ContextApply = "cross_attention",
        cond_channels: Optional[int] = None,
        transformer_depth: int = 1,
    ) -> None:
        super().__init__()
        self.first_stage_model = first_stage_model
        if self.first_stage_model is not None:
            _freeze_first_stage(self.first_stage_model)

        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.num_timesteps = int(num_timesteps)
        self.parameterization = str(parameterization)
        if self.parameterization not in ("eps", "x0"):
            raise ValueError(
                f"parameterization must be 'eps' or 'x0', got {parameterization}"
            )
        self.logit_transform = bool(logit_transform)
        self.conditioning_mode = conditioning_mode

        if self.conditioning_mode == "class":
            if num_data_classes is None or int(num_data_classes) < 1:
                raise ValueError(
                    "num_data_classes is required when conditioning_mode is 'class'"
                )
            self.num_data_classes = int(num_data_classes)
            self.null_class_index = self.num_data_classes
            unet_num_classes = self.num_data_classes + 1
            use_spatial_transformer = False
            ctx_dim = None
            self.context_apply = "cross_attention"
            self.cond_channels = None
            self.context_dim = None
            backbone_in_channels = self.in_channels
        elif self.conditioning_mode == "context":
            self.context_apply = str(context_apply)
            if self.context_apply not in ("cross_attention", "concat"):
                raise ValueError(
                    f"context_apply must be 'cross_attention' or 'concat', "
                    f"got {self.context_apply!r}"
                )
            self.num_data_classes = None
            self.null_class_index = None
            unet_num_classes = None
            if self.context_apply == "cross_attention":
                if context_dim is None or int(context_dim) < 1:
                    raise ValueError(
                        "context_dim is required when conditioning_mode is 'context' "
                        "and context_apply is 'cross_attention'"
                    )
                self.context_dim = int(context_dim)
                self.cond_channels = None
                use_spatial_transformer = True
                ctx_dim = self.context_dim
                backbone_in_channels = self.in_channels
            else:
                if cond_channels is None or int(cond_channels) < 1:
                    raise ValueError(
                        "cond_channels is required when conditioning_mode is 'context' "
                        "and context_apply is 'concat'"
                    )
                self.cond_channels = int(cond_channels)
                self.context_dim = None
                use_spatial_transformer = False
                ctx_dim = None
                backbone_in_channels = self.in_channels + self.cond_channels
        else:
            self.num_data_classes = None
            self.null_class_index = None
            unet_num_classes = None
            use_spatial_transformer = False
            ctx_dim = None
            backbone_in_channels = self.in_channels
            self.context_apply = "cross_attention"
            self.cond_channels = None
            self.context_dim = None

        self.transformer_depth = int(transformer_depth)

        att_res = (
            attention_resolutions
            if attention_resolutions is not None
            else default_attention_resolutions(self.image_size)
        )

        self.backbone = UNetModel(
            image_size=self.image_size,
            in_channels=backbone_in_channels,
            model_channels=int(model_channels),
            out_channels=self.in_channels,
            num_res_blocks=int(num_res_blocks),
            attention_resolutions=att_res,
            dropout=float(dropout),
            channel_mult=channel_mult,
            conv_resample=True,
            dims=2,
            num_classes=unet_num_classes,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=self.transformer_depth,
            context_dim=ctx_dim,
        )

        self.register_schedule(
            beta_schedule=beta_schedule,
            timesteps=self.num_timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_schedule(
        self,
        beta_schedule: str,
        timesteps: int,
        linear_start: float,
        linear_end: float,
        cosine_s: float,
    ) -> None:
        betas = make_beta_schedule(
            beta_schedule,
            n_timestep=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        betas = torch.tensor(betas, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]]
        )

        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped", posterior_log_variance_clipped
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        if self.parameterization == "eps":
            lvlb_weights = betas**2 / (
                2.0 * posterior_variance * alphas * (1.0 - alphas_cumprod)
            )
        elif self.parameterization == "x0":
            denom = 2.0 * (1.0 - alphas_cumprod).clamp(min=1e-8)
            lvlb_weights = 0.5 * torch.sqrt(alphas_cumprod) / denom
        else:
            raise ValueError(
                f"parameterization must be 'eps' or 'x0', got {self.parameterization}"
            )
        lvlb_weights = lvlb_weights.clone()
        lvlb_weights[0] = lvlb_weights[1]
        assert not torch.isnan(lvlb_weights).any()
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)

    def _input_scaling(self, x: torch.Tensor) -> torch.Tensor:
        if self.logit_transform:
            return 2.0 * x - 1.0
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        cond_spatial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict epsilon or x0. `timesteps` is long (B,) with values in [0, T - 1]."""
        x_in = self._input_scaling(x)
        ts = timesteps.long()
        if self.conditioning_mode == "none":
            return self.backbone(x_in, timesteps=ts)
        if self.conditioning_mode == "class":
            return self.backbone(x_in, timesteps=ts, y=y)
        if self.context_apply == "concat":
            if cond_spatial is None:
                raise ValueError(
                    "cond_spatial is required when context_apply is 'concat'"
                )
            x_in = torch.cat([x_in, cond_spatial], dim=1)
            return self.backbone(x_in, timesteps=ts)
        return self.backbone(x_in, timesteps=ts, context=context)

    def _model_out_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        guidance_scale: float,
        *,
        context_uncond: Optional[torch.Tensor] = None,
        cond_spatial: Optional[torch.Tensor] = None,
        cond_spatial_uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.conditioning_mode == "none" or guidance_scale == 1.0:
            return self.forward(x, t, y=y, context=context, cond_spatial=cond_spatial)
        if self.conditioning_mode == "class":
            assert y is not None and self.null_class_index is not None
            y_null = torch.full_like(y, self.null_class_index)
            out_c = self.forward(x, t, y=y, context=None, cond_spatial=None)
            out_u = self.forward(x, t, y=y_null, context=None, cond_spatial=None)
        else:
            assert self.conditioning_mode == "context"
            if self.context_apply == "cross_attention":
                assert context is not None
                ctx_u = (
                    context_uncond
                    if context_uncond is not None
                    else torch.zeros_like(context)
                )
                out_c = self.forward(x, t, y=None, context=context, cond_spatial=None)
                out_u = self.forward(x, t, y=None, context=ctx_u, cond_spatial=None)
            else:
                assert cond_spatial is not None
                c_u = (
                    cond_spatial_uncond
                    if cond_spatial_uncond is not None
                    else torch.zeros_like(cond_spatial)
                )
                out_c = self.forward(
                    x, t, y=None, context=None, cond_spatial=cond_spatial
                )
                out_u = self.forward(x, t, y=None, context=None, cond_spatial=c_u)
        return out_u + guidance_scale * (out_c - out_u)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)."""
        sqrt_alpha = extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def _predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_0 from epsilon prediction (Ho et al. DDPM)."""
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def _p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_uncond: Optional[torch.Tensor] = None,
        cond_spatial: Optional[torch.Tensor] = None,
        cond_spatial_uncond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Posterior mean / log-variance toward x_{t-1}; pred_x0 from the model."""
        model_out = self._model_out_with_cfg(
            x,
            t,
            y,
            context,
            guidance_scale,
            context_uncond=context_uncond,
            cond_spatial=cond_spatial,
            cond_spatial_uncond=cond_spatial_uncond,
        )
        if self.parameterization == "eps":
            pred_x0 = self._predict_x0_from_eps(x, t, model_out)
        else:
            pred_x0 = model_out

        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        model_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * pred_x0
            + extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
        )
        model_log_variance = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x.shape
        )
        return model_mean, model_log_variance, pred_x0

    def _p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_uncond: Optional[torch.Tensor] = None,
        cond_spatial: Optional[torch.Tensor] = None,
        cond_spatial_uncond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """One ancestral DDPM reverse step (improved-diffusion `p_sample`)."""
        model_mean, model_log_variance, _ = self._p_mean_variance(
            x,
            t,
            clip_denoised=clip_denoised,
            y=y,
            context=context,
            context_uncond=context_uncond,
            cond_spatial=cond_spatial,
            cond_spatial_uncond=cond_spatial_uncond,
            guidance_scale=guidance_scale,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def quantize_and_decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.first_stage_model is None:
            raise RuntimeError("quantize_and_decode requires a first_stage_model")
        quant, _, _ = self.first_stage_model.quantize(latents)
        return self.first_stage_model.decode(quant)

    def _sample_latents_dpmsolver(
        self,
        x: torch.Tensor,
        *,
        device: torch.device,
        batch_size: int,
        clip_denoised: bool,
        y: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_uncond: Optional[torch.Tensor],
        cond_spatial: Optional[torch.Tensor],
        cond_spatial_uncond: Optional[torch.Tensor],
        guidance_scale: float,
        dpmsolver_steps: int,
        dpmsolver_order: int,
        dpmsolver_inner_method: str,
        dpmsolver_skip_type: str,
        dpmsolver_algorithm_type: str,
        dpmsolver_denoise_to_zero: bool,
    ) -> torch.Tensor:
        noise_schedule = NoiseScheduleVP(
            "discrete",
            betas=self.betas.to(device=device, dtype=torch.float32),
        )
        model_type = "noise" if self.parameterization == "eps" else "x_start"

        if self.conditioning_mode == "none":

            def wrapped_model(
                x_in: torch.Tensor,
                t_in: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                **kwargs: object,
            ) -> torch.Tensor:
                del cond, kwargs
                return self.forward(
                    x_in,
                    t_in.long(),
                    y=None,
                    context=None,
                    cond_spatial=None,
                )

            wrapped_fn = model_wrapper(
                wrapped_model,
                noise_schedule,
                model_type=model_type,
                model_kwargs={},
                guidance_type="uncond",
                guidance_scale=1.0,
            )
        elif self.conditioning_mode == "class":
            assert y is not None and self.null_class_index is not None

            def wrapped_model(
                x_in: torch.Tensor,
                t_in: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                **kwargs: object,
            ) -> torch.Tensor:
                del kwargs
                assert cond is not None
                return self.forward(
                    x_in,
                    t_in.long(),
                    y=cond,
                    context=None,
                    cond_spatial=None,
                )

            y_uncond = torch.full(
                (batch_size,),
                self.null_class_index,
                device=device,
                dtype=torch.long,
            )
            wrapped_fn = model_wrapper(
                wrapped_model,
                noise_schedule,
                model_type=model_type,
                model_kwargs={},
                guidance_type="classifier-free",
                condition=y,
                unconditional_condition=y_uncond if guidance_scale != 1.0 else None,
                guidance_scale=guidance_scale,
            )
        elif self.conditioning_mode == "context":
            if self.context_apply == "cross_attention":
                assert context is not None and self.context_dim is not None
                ctx_u = context_uncond
                if ctx_u is None:
                    ctx_u = torch.zeros_like(context)

                def wrapped_model(
                    x_in: torch.Tensor,
                    t_in: torch.Tensor,
                    cond: Optional[torch.Tensor] = None,
                    **kwargs: object,
                ) -> torch.Tensor:
                    del kwargs
                    assert cond is not None
                    return self.forward(
                        x_in,
                        t_in.long(),
                        y=None,
                        context=cond,
                        cond_spatial=None,
                    )

                wrapped_fn = model_wrapper(
                    wrapped_model,
                    noise_schedule,
                    model_type=model_type,
                    model_kwargs={},
                    guidance_type="classifier-free",
                    condition=context,
                    unconditional_condition=ctx_u if guidance_scale != 1.0 else None,
                    guidance_scale=guidance_scale,
                )
            else:
                assert cond_spatial is not None and self.cond_channels is not None
                sp_u = cond_spatial_uncond
                if sp_u is None:
                    sp_u = torch.zeros_like(cond_spatial)

                def wrapped_model(
                    x_in: torch.Tensor,
                    t_in: torch.Tensor,
                    cond: Optional[torch.Tensor] = None,
                    **kwargs: object,
                ) -> torch.Tensor:
                    del kwargs
                    assert cond is not None
                    return self.forward(
                        x_in,
                        t_in.long(),
                        y=None,
                        context=None,
                        cond_spatial=cond,
                    )

                wrapped_fn = model_wrapper(
                    wrapped_model,
                    noise_schedule,
                    model_type=model_type,
                    model_kwargs={},
                    guidance_type="classifier-free",
                    condition=cond_spatial,
                    unconditional_condition=sp_u if guidance_scale != 1.0 else None,
                    guidance_scale=guidance_scale,
                )
        else:
            raise ValueError(f"unknown conditioning_mode {self.conditioning_mode!r}")

        x0_correct_fn = None
        if clip_denoised:

            def _clamp_denoised_x0(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                del t
                return torch.clamp(x0, -1.0, 1.0)

            x0_correct_fn = _clamp_denoised_x0

        dpm = DPM_Solver(
            wrapped_fn,
            noise_schedule,
            algorithm_type=dpmsolver_algorithm_type,
            correcting_x0_fn=x0_correct_fn,
        )
        inner = dpmsolver_inner_method
        if inner == "multistep" and dpmsolver_steps < dpmsolver_order:
            raise ValueError(
                f"dpmsolver_steps ({dpmsolver_steps}) must be >= dpmsolver_order "
                f"({dpmsolver_order}) for multistep"
            )
        return dpm.sample(
            x,
            steps=dpmsolver_steps,
            order=dpmsolver_order,
            skip_type=dpmsolver_skip_type,
            method=inner,
            denoise_to_zero=dpmsolver_denoise_to_zero,
        )

    @torch.no_grad()
    def sample_latents(
        self,
        batch_size: int,
        latent_shape: tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        clip_denoised: bool = False,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_uncond: Optional[torch.Tensor] = None,
        cond_spatial: Optional[torch.Tensor] = None,
        cond_spatial_uncond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        method: str = "ancestral_sampling",
        *,
        dpmsolver_steps: int = 20,
        dpmsolver_order: int = 3,
        dpmsolver_inner_method: str = "multistep",
        dpmsolver_skip_type: str = "time_uniform",
        dpmsolver_algorithm_type: str = "dpmsolver++",
        dpmsolver_denoise_to_zero: bool = False,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample latents via ancestral DDPM or DPM-Solver."""
        if device is None:
            device = next(self.parameters()).device

        if latent_shape[0] != batch_size:
            latent_shape = (batch_size, *latent_shape[1:])

        _, c, h, w = latent_shape
        dtype = self.betas.dtype
        if initial_noise is None:
            x = torch.randn(batch_size, c, h, w, device=device, dtype=dtype)
        else:
            x = initial_noise.to(device=device, dtype=dtype).clone()

        if self.conditioning_mode == "class":
            assert self.null_class_index is not None
            if y is None:
                y = torch.full(
                    (batch_size,),
                    self.null_class_index,
                    device=device,
                    dtype=torch.long,
                )
        elif self.conditioning_mode == "context":
            if self.context_apply == "cross_attention":
                if context is None:
                    assert self.context_dim is not None
                    n_hw = h * w
                    context = torch.zeros(
                        batch_size,
                        n_hw,
                        self.context_dim,
                        device=device,
                        dtype=x.dtype,
                    )
            else:
                assert self.cond_channels is not None
                if cond_spatial is None:
                    cond_spatial = torch.zeros(
                        batch_size,
                        self.cond_channels,
                        h,
                        w,
                        device=device,
                        dtype=x.dtype,
                    )

        if method == "ancestral_sampling":
            for i in reversed(range(self.num_timesteps)):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x = self._p_sample(
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    y=y,
                    context=context,
                    context_uncond=context_uncond,
                    cond_spatial=cond_spatial,
                    cond_spatial_uncond=cond_spatial_uncond,
                    guidance_scale=guidance_scale,
                )
            return x

        if method in ("dpmsolver", "dpm_solver"):
            valid_inner = (
                "singlestep",
                "multistep",
                "singlestep_fixed",
                "adaptive",
            )
            if dpmsolver_inner_method not in valid_inner:
                raise ValueError(
                    f"dpmsolver_inner_method must be one of {valid_inner}, "
                    f"got {dpmsolver_inner_method!r}"
                )
            return self._sample_latents_dpmsolver(
                x,
                device=device,
                batch_size=batch_size,
                clip_denoised=clip_denoised,
                y=y,
                context=context,
                context_uncond=context_uncond,
                cond_spatial=cond_spatial,
                cond_spatial_uncond=cond_spatial_uncond,
                guidance_scale=guidance_scale,
                dpmsolver_steps=dpmsolver_steps,
                dpmsolver_order=dpmsolver_order,
                dpmsolver_inner_method=dpmsolver_inner_method,
                dpmsolver_skip_type=dpmsolver_skip_type,
                dpmsolver_algorithm_type=dpmsolver_algorithm_type,
                dpmsolver_denoise_to_zero=dpmsolver_denoise_to_zero,
            )

        raise ValueError(
            f"method must be 'ancestral_sampling', 'dpmsolver', or 'dpm_solver', "
            f"got {method!r}"
        )
