import os
from typing import Any

import hydra
import pytorch_lightning as pl
import torch.nn as nn
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.imagenet import ImageNetDataModule
from src.data.mnist import MNISTDataModule
from src.interfaces.transformer_prior import TransformerPriorInterface
from src.interfaces.vqvae import VQVAEInterface


def _build_datamodule(data_cfg: DictConfig) -> Any:
    params = OmegaConf.to_container(data_cfg.params, resolve=True)
    if data_cfg.name == "mnist":
        return MNISTDataModule(**params)
    if data_cfg.name == "imagenet":
        return ImageNetDataModule(**params)
    raise ValueError(f"Unsupported dataset: {data_cfg.name}")


def _build_module(cfg: DictConfig) -> pl.LightningModule:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    image_key = cfg.data.params.get("image_key", model_cfg.get("image_key", "image"))

    model_name = model_cfg.get("name", "vqvae")
    if model_name == "vqvae":
        ddconfig = dict(model_cfg["ddconfig"])
        ddconfig["in_channels"] = int(cfg.data.in_channels)
        ddconfig["out_ch"] = int(cfg.data.out_channels)
        ddconfig["resolution"] = int(cfg.data.params.image_size)
        val_logging = model_cfg.get("val_logging") or {}
        return VQVAEInterface(
            ddconfig=ddconfig,
            n_embed=int(model_cfg["n_embed"]),
            embed_dim=int(model_cfg["embed_dim"]),
            image_key=image_key,
            learning_rate=float(model_cfg["learning_rate"]),
            betas=tuple(model_cfg["betas"]),
            weight_decay=float(model_cfg["weight_decay"]),
            scheduler=model_cfg.get("scheduler", {}),
            val_logging=val_logging,
        )

    if model_name == "transformer_prior":
        vq_ckpt = model_cfg.get("vq_ckpt_path")
        if not vq_ckpt or str(vq_ckpt).lower() == "null":
            raise ValueError(
                "Prior model requires model.vq_ckpt_path to point to a trained VQ-VAE checkpoint. "
                "Override with e.g. model.vq_ckpt_path=checkpoints/xyz/vqvae-last.ckpt"
            )

        ckpt_dir = os.path.dirname(str(vq_ckpt))
        ckpt_cfg_path = os.path.join(ckpt_dir, "config.yaml")
        if not os.path.exists(ckpt_cfg_path):
            raise FileNotFoundError(
                f"Expected VQ-VAE config at {ckpt_cfg_path}. "
                "Ensure you saved config.yaml in the checkpoint folder when training VQ-VAE."
            )
        ckpt_cfg = OmegaConf.load(ckpt_cfg_path)
        vq_model_cfg = OmegaConf.to_container(ckpt_cfg.model, resolve=True)
        vq_ddconfig = dict(vq_model_cfg["ddconfig"])
        vq_n_embed = int(vq_model_cfg["n_embed"])
        vq_embed_dim = int(vq_model_cfg["embed_dim"])

        # Optional consistency checks if transformer_prior.yaml still specifies these fields.
        if "n_embed" in model_cfg and int(model_cfg["n_embed"]) != vq_n_embed:
            raise ValueError(
                "transformer_prior.n_embed does not match VQ-VAE n_embed in checkpoint config"
            )
        if "embed_dim" in model_cfg and int(model_cfg["embed_dim"]) != vq_embed_dim:
            raise ValueError(
                "transformer_prior.embed_dim does not match VQ-VAE embed_dim in checkpoint config"
            )

        return TransformerPriorInterface(
            ddconfig=vq_ddconfig,
            n_embed=vq_n_embed,
            embed_dim=vq_embed_dim,
            vq_ckpt_path=str(vq_ckpt),
            block_size=int(model_cfg["block_size"]),
            n_layer=int(model_cfg["n_layer"]),
            n_head=int(model_cfg["n_head"]),
            n_embd=int(model_cfg["n_embd"]),
            image_key=image_key,
            learning_rate=float(model_cfg["learning_rate"]),
            pkeep=float(model_cfg.get("pkeep", 1.0)),
            sos_token=int(model_cfg.get("sos_token", 0)),
            transformer_dropout=float(model_cfg.get("transformer_dropout", 0.0)),
            val_logging=model_cfg.get("val_logging"),
        )

    raise ValueError(f"Unknown model name: {model_name}")


def _apply_hardware_options(cfg: DictConfig, trainer_kwargs: dict[str, Any]) -> None:
    use_gpu = bool(cfg.hardware.use_gpu)
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "hardware.use_gpu=true but CUDA is not available. "
                "Set hardware.use_gpu=false to train on CPU."
            )
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = int(cfg.hardware.gpu_devices)
        return

    trainer_kwargs["accelerator"] = "cpu"
    trainer_kwargs["devices"] = 1


def _module_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _print_model_tree(
    module: nn.Module,
    max_depth: int = 3,
    prefix: str = "",
    depth: int = 0,
    name: str | None = None,
) -> None:
    """Print module hierarchy with parameter counts for the top max_depth levels."""
    param_count = _module_param_count(module)
    display_name = name or module.__class__.__name__
    if depth == 0:
        print(f"\n===== Model (top {max_depth} levels) =====")
    print(f"{prefix}{display_name} ({module.__class__.__name__}): {param_count:,} params")
    if depth < max_depth - 1:
        for child_name, child in module.named_children():
            _print_model_tree(child, max_depth, prefix + "  ", depth + 1, child_name)


def _print_model_summary(module: nn.Module) -> None:
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in module.parameters())
    # Approximate fp32 footprint in megabytes.
    total_param_size_mb = (total_params * 4) / (1024**2)

    _print_model_tree(module, max_depth=3)
    print("===== Parameter Summary =====")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Approx parameter size (fp32): {total_param_size_mb:.2f} MB\n")


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)

    datamodule = _build_datamodule(cfg.data)
    module = _build_module(cfg)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    _apply_hardware_options(cfg, trainer_kwargs)

    logger = False
    checkpoint_subdir = "no_wandb"
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            save_dir=cfg.wandb.save_dir,
            log_model=cfg.wandb.log_model,
        )
        checkpoint_subdir = logger.experiment.id
        # Keep a readable run snapshot in W&B.
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    model_name = OmegaConf.select(cfg, "model.name", default="vqvae")
    if model_name == "transformer_prior":
        monitor, mode, filename = "val/loss", "min", "prior-{epoch}-{step}"
    else:
        monitor, mode, filename = "val/rec_loss", "min", "vqvae-{epoch}-{step}"
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join("checkpoints", checkpoint_subdir),
            monitor=monitor,
            mode=mode,
            save_top_k=2,
            save_last=True,
            filename=filename,
        )
    ]
    if cfg.wandb.enabled:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)
    _print_model_summary(module)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
