from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.modules.latents.diffusion_backbones.openai_unet import UNetModel
from src.modules.latents.diffusion_backbones.utils import extract_into_tensor, make_beta_schedule


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


class LatentOpenAIUNetDDPM(nn.Module):
    """DDPM noise predictor over VQ latents using the OpenAI UNet backbone."""

    def __init__(
        self,
        first_stage_model: nn.Module,
        in_channels: int,
        image_size: int,
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
        logit_transform: bool = True,
    ) -> None:
        super().__init__()
        self.first_stage_model = first_stage_model
        _freeze_first_stage(self.first_stage_model)

        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.num_timesteps = int(num_timesteps)
        self.parameterization = str(parameterization)
        if self.parameterization not in ("eps", "x0"):
            raise ValueError(f"parameterization must be 'eps' or 'x0', got {parameterization}")
        self.logit_transform = bool(logit_transform)

        att_res = (
            attention_resolutions
            if attention_resolutions is not None
            else default_attention_resolutions(self.image_size)
        )

        self.backbone = UNetModel(
            image_size=self.image_size,
            in_channels=self.in_channels,
            model_channels=int(model_channels),
            out_channels=self.in_channels,
            num_res_blocks=int(num_res_blocks),
            attention_resolutions=att_res,
            dropout=float(dropout),
            channel_mult=channel_mult,
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
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
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]])

        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        # Per-timestep weights for hybrid simple + VLB-style loss (CompVis DDPM).
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

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict epsilon or x0. `timesteps` is long (B,) with values in [0, T - 1]."""
        x_in = self._input_scaling(x)
        return self.backbone(x_in, timesteps=timesteps.long())

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)."""
        sqrt_alpha = extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
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
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Posterior mean / log-variance toward x_{t-1}; pred_x0 from the model."""
        model_out = self.forward(x, t)
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
        model_log_variance = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        return model_mean, model_log_variance, pred_x0

    def _p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool,
    ) -> torch.Tensor:
        """One ancestral DDPM reverse step (improved-diffusion `p_sample`)."""
        model_mean, model_log_variance, _ = self._p_mean_variance(x, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def quantize_and_decode(self, latents: torch.Tensor) -> torch.Tensor:
        quant, _, _ = self.first_stage_model.quantize(latents)
        return self.first_stage_model.decode(quant)

    @torch.no_grad()
    def sample_latents(
        self,
        batch_size: int,
        latent_shape: tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """Ancestral DDPM sampling (`p_sample_loop`: T steps from Gaussian noise)."""
        if device is None:
            device = next(self.parameters()).device

        if latent_shape[0] != batch_size:
            latent_shape = (batch_size, *latent_shape[1:])

        _, c, h, w = latent_shape
        x = torch.randn(batch_size, c, h, w, device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self._p_sample(x, t, clip_denoised=clip_denoised)

        return x
