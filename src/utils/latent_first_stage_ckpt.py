"""Helpers to omit frozen VQ (first-stage) weights from latent Lightning checkpoints."""

from __future__ import annotations

from typing import Any

LATENT_FIRST_STAGE_PREFIX = "model.first_stage_model."


def omit_first_stage_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``state_dict`` without ``model.first_stage_model.*`` keys."""
    p = LATENT_FIRST_STAGE_PREFIX
    return {k: v for k, v in state_dict.items() if not k.startswith(p)}


def check_strict_first_stage_load(incompatible_keys: Any) -> None:
    """Enforce that only first-stage keys were missing and nothing was unexpected."""
    unexpected = getattr(incompatible_keys, "unexpected_keys", ())
    if unexpected:
        preview = list(unexpected)[:8]
        raise RuntimeError(
            f"Checkpoint has unexpected keys (not allowed when strict=True): {preview}"
        )
    missing = getattr(incompatible_keys, "missing_keys", ())
    bad = [k for k in missing if not k.startswith(LATENT_FIRST_STAGE_PREFIX)]
    if bad:
        preview = bad[:8]
        raise RuntimeError(
            f"Checkpoint missing non-first-stage keys (strict=True): {preview}"
        )
