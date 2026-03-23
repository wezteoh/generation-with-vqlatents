from __future__ import annotations

import abc
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn

from src.modules.sampling import get_sampling_fn
from src.modules.sde import SDE


def _dict_to_namespace(cfg: Any) -> Any:
    """Convert nested dicts to `SimpleNamespace` for dot-access config usage."""
    if isinstance(cfg, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in cfg.items()})
    return cfg


class ScoreSDEModel(nn.Module, abc.ABC):
    """SDE-aware score model over (latent or image) tensors.

    Subclasses should return the score (i.e. grad wrt log-density) for the
    perturbed state at time `t`.
    """

    def __init__(
        self,
        sde: SDE,
        first_stage_model: nn.Module | None = None,
        sampling_method: str = "langevin",
        predictor: str | None = None,
        corrector: str | None = None,
    ) -> None:
        super().__init__()
        self.sde = sde
        self.first_stage_model = first_stage_model

        # Store sampling-related knobs for possible sampling/validation use.
        self.sampling_method = sampling_method
        self.predictor = predictor
        self.corrector = corrector

        if self.first_stage_model is not None:
            # Freeze first stage (mirrors ScoreModel and VQ-based DSM).
            self.first_stage_model.eval()
            self.first_stage_model.train = (  # type: ignore[assignment]
                lambda *_args, **_kwargs: None
            )

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        sigmas: torch.Tensor | None = None,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        """Compute score at time `t`.

        Args:
            x: Noisy/perturbed sample with shape (B, C, H, W).
            sigmas: Optional per-sample noise scale used by some backbones.
        """
        raise NotImplementedError("Subclasses must implement forward(x, sigmas)")

    @torch.no_grad()
    def sample_latents(
        self,
        batch_size: int,
        latent_shape: tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        eps: float = 1e-3,
        n_steps: int | None = None,
        sampling_cfg: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Generate latents by sampling the reverse-time SDE.

        Uses `get_sampling_fn` from `src.modules.sampling` so the sampling method
        can be configured via `sampling_cfg`.
        """
        if device is None:
            device = next(self.parameters()).device

        if latent_shape[0] != batch_size:
            # Allow callers to pass a latent shape without batch dimension alignment.
            latent_shape = (batch_size, *latent_shape[1:])

        # Optionally rebuild the SDE to control the reverse-time discretization count.
        sde_for_sampling = self.sde
        if n_steps is not None:
            n_steps_int = int(n_steps)
            import src.modules.sampling as sampling_mod

            sde_name = type(self.sde).__name__.lower()
            if sde_name == "vesde":
                sde_for_sampling = sampling_mod.sde_lib.VESDE(
                    sigma_min=float(self.sde.sigma_min),
                    sigma_max=float(self.sde.sigma_max),
                    N=n_steps_int,
                )
            elif sde_name == "vpsde":
                sde_for_sampling = sampling_mod.sde_lib.VPSDE(
                    beta_min=float(getattr(self.sde, "beta_0")),
                    beta_max=float(getattr(self.sde, "beta_1")),
                    N=n_steps_int,
                )
            elif sde_name == "subvpsde":
                sde_for_sampling = sampling_mod.sde_lib.subVPSDE(
                    beta_min=float(getattr(self.sde, "beta_0")),
                    beta_max=float(getattr(self.sde, "beta_1")),
                    N=n_steps_int,
                )
            else:
                # Unknown SDE: keep original to avoid breaking sampling.
                sde_for_sampling = self.sde

        # Default config (primarily for standalone usage).
        cfg_dict: dict[str, Any] = sampling_cfg.copy() if sampling_cfg is not None else {}
        cfg_dict.setdefault(
            "sampling",
            {
                "method": "pc",
                "predictor": "euler_maruyama",
                "corrector": "langevin",
                "snr": 0.3,
                "n_steps_each": 1,
                "probability_flow": False,
                "noise_removal": True,
            },
        )
        cfg_dict.setdefault("training", {"continuous": True})
        cfg = _dict_to_namespace(cfg_dict)
        cfg.device = device

        sampling_fn = get_sampling_fn(
            config=cfg,
            sde=sde_for_sampling,
            shape=latent_shape,
            eps=eps,
        )
        latents, _nfe = sampling_fn(self)
        return latents

    @torch.no_grad()
    def quantize_and_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Quantize continuous latents with the VQ codebook and decode to images."""
        if self.first_stage_model is None:
            raise RuntimeError(
                "ScoreSDEModel.first_stage_model is None; cannot quantize and decode latents."
            )
        quant, _, _ = self.first_stage_model.quantize(latents)
        return self.first_stage_model.decode(quant)
