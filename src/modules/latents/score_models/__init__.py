from __future__ import annotations

import functools
from typing import Sequence

import torch
import torch.nn as nn
import tqdm


class ScoreModel(nn.Module):
    """Base class for score networks operating on (latent) tensors."""

    def __init__(self, first_stage_model: nn.Module | None = None) -> None:
        super().__init__()
        self.first_stage_model = first_stage_model
        if self.first_stage_model is not None:
            # Freeze first stage (mirrors LatentTransformer).
            self.first_stage_model.eval()
            self.first_stage_model.train = (  # type: ignore[assignment]
                lambda *_args, **_kwargs: None
            )

    def forward(
        self, x: torch.Tensor, sigma_labels: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError("Subclasses must implement forward(x, sigma_labels)")

    @torch.no_grad()
    def langevin_dynamics_sample(
        self,
        x_mod: torch.Tensor,
        sigma_label: int | torch.Tensor,
        n_steps: int = 200,
        step_lr: float = 5e-5,
    ) -> Sequence[torch.Tensor]:
        """Simple Langevin sampling at a fixed noise label."""
        latents: list[torch.Tensor] = []

        if not torch.is_tensor(sigma_label):
            labels = torch.full(
                (x_mod.shape[0],),
                int(sigma_label),
                device=x_mod.device,
                dtype=torch.long,
            )
        else:
            labels = sigma_label.to(device=x_mod.device, dtype=torch.long)

        for _ in range(n_steps):
            noise = torch.randn_like(x_mod) * (step_lr * 2.0) ** 0.5
            grad = self(x_mod, labels)
            x_mod = x_mod + step_lr * grad + noise
            latents.append(x_mod.detach().clone())
        return latents

    @torch.no_grad()
    def anneal_langevin_dynamics_sample(
        self,
        x_mod: torch.Tensor,
        sigmas: Sequence[float],
        n_steps_each: int = 100,
        step_lr: float = 2e-5,
    ) -> Sequence[torch.Tensor]:
        """Annealed Langevin dynamics over a sigma schedule."""
        latents: list[torch.Tensor] = []
        sigmas = list(sigmas)
        base_sigma = sigmas[-1]

        for c, sigma in tqdm.tqdm(
            list(enumerate(sigmas)),
            total=len(sigmas),
            desc="annealed Langevin dynamics sampling",
        ):
            labels = torch.full(
                (x_mod.shape[0],),
                int(c),
                device=x_mod.device,
                dtype=torch.long,
            )
            step_size = step_lr * (sigma / base_sigma) ** 2
            for _ in range(n_steps_each):
                noise = torch.randn_like(x_mod) * (step_size * 2.0) ** 0.5
                grad = self(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                latents.append(x_mod.detach().clone())
        return latents

    @torch.no_grad()
    def sample_latents(
        self,
        batch_size: int,
        latent_shape: tuple[int, int, int, int],
        sigmas: Sequence[float],
        n_steps_each: int = 20,
        step_lr: float = 2e-5,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Annealed Langevin sampling in latent space over a sigma schedule.

        Returns the final latent tensor of shape latent_shape.
        """
        if device is None:
            device = next(self.parameters()).device
        x_mod = torch.randn(latent_shape, device=device) * float(sigmas[0])
        latents = self.anneal_langevin_dynamics_sample(
            x_mod=x_mod,
            sigmas=sigmas,
            n_steps_each=n_steps_each,
            step_lr=step_lr,
        )
        return latents[-1]

    @torch.no_grad()
    def quantize_and_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Quantize continuous latents with the VQ codebook and decode to images."""
        if self.first_stage_model is None:
            raise RuntimeError(
                "ScoreModel.first_stage_model is None; cannot quantize and decode latents."
            )
        vq = self.first_stage_model
        quant, _, _ = vq.quantize(latents)
        images = vq.decode(quant)
        return images
