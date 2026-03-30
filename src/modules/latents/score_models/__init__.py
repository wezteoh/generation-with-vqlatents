from __future__ import annotations

import functools
from typing import List, Sequence

import numpy as np
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
        self, x: torch.Tensor, sigma_labels: torch.Tensor, sigmas: torch.Tensor | None = None
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
        sigmas_tensor = torch.tensor(sigmas, device=x_mod.device, dtype=torch.float32)
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
                grad = self(x_mod, labels, sigmas_tensor)
                x_mod = x_mod + step_size * grad + noise
                latents.append(x_mod.detach().clone())
        return latents

    def predictor_corrector_sample(
        self,
        x_mod: torch.Tensor,
        n_steps_each: int = 10,
        sigmas: Sequence[float] | None = None,
        target_snr: float = 0.3,
    ) -> Sequence[torch.Tensor]:
        """Predictor-corrector sampling over a sigma schedule."""
        latents: list[torch.Tensor] = []
        if sigmas is None:
            sigmas_buf = getattr(self, "sigmas", None)
            if sigmas_buf is None or sigmas_buf.numel() == 0:
                raise ValueError(
                    "predictor_corrector_sample: pass sigmas=... or use a score model "
                    "with a non-empty sigmas buffer (e.g. LatentNCSNv2Score)."
                )
            sigmas_list = list(sigmas_buf.detach().cpu().tolist())
        else:
            sigmas_list = list(sigmas)
        sigmas_tensor = torch.tensor(sigmas_list, device=x_mod.device, dtype=torch.float32)

        for c, sigma in tqdm.tqdm(
            list(enumerate(sigmas_list)),
            total=len(sigmas_list),
            desc="annealed Langevin dynamics sampling",
        ):
            if c + 1 == len(sigmas_list):
                break
            labels = torch.full(
                (x_mod.shape[0],),
                int(c),
                device=x_mod.device,
                dtype=torch.long,
            )
            adjacent_sigma = sigmas_list[c + 1]
            score = self(x_mod, labels, sigmas_tensor)
            x_mean = x_mod + score * (sigma**2 - adjacent_sigma**2)
            std = np.sqrt(sigma**2 - adjacent_sigma**2)
            noise = torch.randn_like(x_mod)
            x_mod = x_mean + std * noise
            for _ in range(n_steps_each):
                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2
                x_mod = x_mod + step_size * score
                x_mod = x_mod + torch.sqrt(step_size * 2) * noise
                latents.append(x_mod.detach().clone())
                if x_mod.isnan().any():
                    print(f"x_mod is nan at step {c} {_}")
                    break
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

class SDEScoreModel(nn.Module):
    """Score model for the SDE."""
    def __init__(self, sde: SDE, score_model: ScoreModel, likelihood_weighting: bool = True) -> None:
        super().__init__()
        self.sde = sde
        self.score_model = score_model
        self.likelihood_weighting = likelihood_weighting


if __name__ == "__main__":
    # Smoke test: load a DSM *latent* checkpoint and run predictor-corrector sampling.
    # Default paths match notebooks/infer_dsm_latent.ipynb; override with env vars.
    import os
    from pathlib import Path

    from omegaconf import OmegaConf

    from src.interfaces.dsm import DSMLatentInterface

    _repo_root = Path(__file__).resolve().parents[4]
    _dsm_ckpt = os.environ.get(
        "VQLATENTS_DSM_LATENT_CKPT",
        str(_repo_root / "checkpoints" / "5rt0b4qo" / "last.ckpt"),
    )
    _vq_ckpt = os.environ.get(
        "VQLATENTS_VQ_CKPT",
        str(_repo_root / "checkpoints" / "ct3png0p" / "last.ckpt"),
    )

    if not Path(_dsm_ckpt).is_file() or not Path(_vq_ckpt).is_file():
        print(
            "Skipping predictor_corrector_sample test: checkpoint(s) not found.\n"
            f"  DSM (latent score): {_dsm_ckpt!r}\n"
            f"  VQ (first stage):   {_vq_ckpt!r}\n"
            "Set VQLATENTS_DSM_LATENT_CKPT / VQLATENTS_VQ_CKPT to run this block."
        )
        raise SystemExit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = DSMLatentInterface.load_from_checkpoint(
        _dsm_ckpt,
        vq_ckpt_path=_vq_ckpt,
        map_location=str(device),
    )
    lit.eval()
    lit.to(device)
    if lit.ema is not None:
        lit.ema.store(lit.model.parameters())
        lit.ema.copy_to(lit.model)

    dd = lit.hparams.ddconfig
    dd = OmegaConf.to_container(dd, resolve=True) if OmegaConf.is_config(dd) else dict(dd)
    resolution = int(dd.get("resolution", 32))
    ch_mult = tuple(dd.get("ch_mult", (1, 2, 4, 8)))
    latent_res = resolution // 2 ** (len(ch_mult) - 1)
    embed_dim = int(lit.hparams.embed_dim)
    latent_shape = (1, embed_dim, latent_res, latent_res)

    # Short schedule + few corrector steps for a fast run (full chain is slow).
    sigmas_short = lit.sigmas[:12].detach().cpu().tolist()
    sigma0 = float(sigmas_short[0])
    x_mod = torch.randn(latent_shape, device=device, dtype=torch.float32) * sigma0

    with torch.no_grad():
        chain = lit.model.predictor_corrector_sample(
            x_mod,
            n_steps_each=2,
            sigmas=sigmas_short,
            target_snr=0.3,
        )

    assert len(chain) > 0, "predictor_corrector_sample should append at least one latent state"
    assert chain[-1].shape == x_mod.shape, (chain[-1].shape, x_mod.shape)
    print(
        "predictor_corrector_sample OK:",
        f"len(chain)={len(chain)}, final shape={tuple(chain[-1].shape)}",
    )
