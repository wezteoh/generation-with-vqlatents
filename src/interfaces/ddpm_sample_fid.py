"""Optional validation Fréchet distance between real val images and DDPM samples."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningModule

from src.modules.ema import LitEma
from src.utils.fid_inputs import tensor_to_fid_input


def run_sample_fid_if_gated(
    module: LightningModule,
    *,
    val_pass_count: int,
    image_key: str,
    val_logging_cfg: dict[str, Any],
    ema: Optional[LitEma],
    model: torch.nn.Module,
    generate_fake_images: Callable[[int, torch.device], torch.Tensor],
) -> None:
    """If `sample_fid` is enabled and the frequency gate matches, log `val/sample_fid`.

    Reals: one pass over `val_dataloader` up to `num_real_samples`.
    Fakes: `generate_fake_images` in chunks until `num_gen_samples`. Caller runs DDPM
    sampling; this helper wraps fake generation with EMA store/copy/restore.
    """
    trainer = module.trainer
    if getattr(trainer, "sanity_checking", False):
        return

    sf = val_logging_cfg.get("sample_fid") or {}
    if not bool(sf.get("enabled", False)):
        return

    every_n = int(sf.get("every_n_val_epochs", 1))
    if every_n <= 0 or val_pass_count % every_n != 0:
        return

    num_real = int(sf.get("num_real_samples", 512))
    num_gen = int(sf.get("num_gen_samples", 512))
    feature = int(sf.get("feature", 2048))

    if num_real < 2 or num_gen < 2:
        return

    if getattr(trainer, "global_rank", 0) != 0:
        return

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as exc:
        raise ImportError(
            "sample_fid requires torchmetrics with image extras and torch-fidelity."
        ) from exc

    device = module.device
    dm = trainer.datamodule
    if dm is None:
        return

    dl = dm.val_dataloader()
    real_parts: list[torch.Tensor] = []
    collected = 0
    for batch in dl:
        x = batch[image_key].to(device)
        real_parts.append(x)
        collected += x.shape[0]
        if collected >= num_real:
            break

    if not real_parts:
        return

    real_tensor = torch.cat(real_parts, dim=0)[:num_real]

    fid = FrechetInceptionDistance(
        feature=feature,
        reset_real_features=True,
        normalize=False,
    ).to(device)

    fid.update(tensor_to_fid_input(real_tensor), real=True)

    gen_left = num_gen
    gen_bs = min(32, num_gen)
    fake_parts: list[torch.Tensor] = []
    if ema is not None:
        ema.store(model.parameters())
        ema.copy_to(model)
    try:
        while gen_left > 0:
            bs = min(gen_bs, gen_left)
            fake_parts.append(generate_fake_images(bs, device))
            gen_left -= bs
    finally:
        if ema is not None:
            ema.restore(model.parameters())

    fake_tensor = torch.cat(fake_parts, dim=0)[:num_gen]
    fid.update(tensor_to_fid_input(fake_tensor), real=False)

    score = fid.compute()
    module.log(
        "val/sample_fid",
        score,
        on_epoch=True,
        prog_bar=True,
        rank_zero_only=True,
    )
