import os
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.celebahq import CelebAHQ256DataModule
from src.data.imagenet import ImageNetDataModule
from src.data.mnist import MNISTDataModule, MNISTLabeledDataModule
from src.interfaces.ddpm_latent import DDPMLatentInterface
from src.interfaces.ddpm_raw import DDPMRawInterface
from src.interfaces.dsm_latent import DSMLatentInterface
from src.interfaces.dsm_raw import DSMRawInterface
from src.interfaces.score_sde_latent import ScoreSDEInterface
from src.interfaces.score_sde_raw import ScoreSDERawInterface
from src.interfaces.transformer_latent import TransformerLatentInterface
from src.interfaces.vqgan import VQGANInterface
from src.interfaces.vqvae import VQVAEInterface


def _build_datamodule(data_cfg: DictConfig) -> Any:
    params = OmegaConf.to_container(data_cfg.params, resolve=True)
    if data_cfg.name == "mnist":
        return MNISTDataModule(**params)
    if data_cfg.name == "mnist_labeled":
        return MNISTLabeledDataModule(**params)
    if data_cfg.name == "imagenet":
        return ImageNetDataModule(**params)
    if data_cfg.name == "celebahq256":
        return CelebAHQ256DataModule(**params)
    raise ValueError(f"Unsupported dataset: {data_cfg.name}")


def _ddpm_conditioning_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Parse optional `model.conditioning` for DDPM raw/latent interfaces."""
    cond = model_cfg.get("conditioning")
    if not cond:
        return {
            "conditioning_mode": "none",
            "num_data_classes": None,
            "label_key": "label",
            "context_key": "context",
            "unconditional_prob": 0.0,
            "context_dim": None,
            "transformer_depth": 1,
        }
    mode = str(cond.get("mode", "none"))
    out: dict[str, Any] = {
        "conditioning_mode": mode,
        "label_key": str(cond.get("label_key", "label")),
        "context_key": str(cond.get("context_key", "context")),
        "unconditional_prob": float(cond.get("unconditional_prob", 0.0)),
        "transformer_depth": int(cond.get("transformer_depth", 1)),
    }
    if mode == "class":
        if "num_classes" not in cond:
            raise ValueError(
                "model.conditioning.num_classes is required when mode is 'class'"
            )
        out["num_data_classes"] = int(cond["num_classes"])
        out["context_dim"] = None
    elif mode == "context":
        if "context_dim" not in cond:
            raise ValueError(
                "model.conditioning.context_dim is required when mode is 'context'"
            )
        out["context_dim"] = int(cond["context_dim"])
        out["num_data_classes"] = None
    else:
        out["num_data_classes"] = None
        out["context_dim"] = None
    return out


def _build_module(cfg: DictConfig) -> pl.LightningModule:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    image_key = cfg.data.params.get("image_key", model_cfg.get("image_key", "image"))

    model_name = model_cfg.get("name", "vqvae")
    if model_name == "vqvae":
        trainer_optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
        trainer_val_logging = OmegaConf.to_container(
            cfg.trainer.val_logging, resolve=True
        )

        ddconfig = dict(model_cfg["ddconfig"])
        ddconfig["in_channels"] = int(cfg.data.in_channels)
        ddconfig["out_ch"] = int(cfg.data.out_channels)
        ddconfig["resolution"] = int(cfg.data.params.image_size)
        return VQVAEInterface(
            ddconfig=ddconfig,
            n_embed=int(model_cfg["n_embed"]),
            embed_dim=int(model_cfg["embed_dim"]),
            image_key=image_key,
            learning_rate=float(trainer_optim["learning_rate"]),
            betas=tuple(trainer_optim["betas"]),
            weight_decay=float(trainer_optim["weight_decay"]),
            scheduler=trainer_optim.get("scheduler", {}),
            val_logging=trainer_val_logging,
        )

    if model_name == "vqgan":
        trainer_optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
        trainer_val_logging = OmegaConf.to_container(
            cfg.trainer.val_logging, resolve=True
        )

        ddconfig = dict(model_cfg["ddconfig"])
        ddconfig["in_channels"] = int(cfg.data.in_channels)
        ddconfig["out_ch"] = int(cfg.data.out_channels)
        ddconfig["resolution"] = int(cfg.data.params.image_size)

        disc_config = dict(model_cfg["disc_config"])
        disc_config["in_channels"] = int(cfg.data.out_channels)

        return VQGANInterface(
            ddconfig=ddconfig,
            disc_config=disc_config,
            disc_start=int(model_cfg.get("disc_start", 0)),
            perceptual_weight=float(model_cfg.get("perceptual_weight", 1.0)),
            discriminator_weight=float(model_cfg.get("discriminator_weight", 1.0)),
            codebook_weight=float(model_cfg.get("codebook_weight", 1.0)),
            n_embed=int(model_cfg["n_embed"]),
            embed_dim=int(model_cfg["embed_dim"]),
            image_key=image_key,
            learning_rate=float(trainer_optim["learning_rate"]),
            disc_learning_rate=(
                float(trainer_optim["disc_learning_rate"])
                if "disc_learning_rate" in trainer_optim
                else None
            ),
            betas=tuple(trainer_optim["betas"]),
            weight_decay=float(trainer_optim["weight_decay"]),
            scheduler=trainer_optim.get("scheduler", {}),
            val_logging=trainer_val_logging,
            gan_loss=str(model_cfg.get("gan_loss", "hinge")),
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

        trainer_optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
        trainer_val_logging = OmegaConf.to_container(
            cfg.trainer.val_logging, resolve=True
        )

        # Optional consistency checks if transformer_prior.yaml still specifies these fields.
        if "n_embed" in model_cfg and int(model_cfg["n_embed"]) != vq_n_embed:
            raise ValueError(
                "transformer_prior.n_embed does not match VQ-VAE n_embed in checkpoint config"
            )
        if "embed_dim" in model_cfg and int(model_cfg["embed_dim"]) != vq_embed_dim:
            raise ValueError(
                "transformer_prior.embed_dim does not match VQ-VAE embed_dim in checkpoint config"
            )

        return TransformerLatentInterface(
            ddconfig=vq_ddconfig,
            n_embed=vq_n_embed,
            embed_dim=vq_embed_dim,
            vq_ckpt_path=str(vq_ckpt),
            block_size=int(model_cfg["block_size"]),
            n_layer=int(model_cfg["n_layer"]),
            n_head=int(model_cfg["n_head"]),
            n_embd=int(model_cfg["n_embd"]),
            image_key=image_key,
            learning_rate=float(trainer_optim["learning_rate"]),
            betas=tuple(trainer_optim["betas"]),
            scheduler=trainer_optim.get("scheduler", {}),
            pkeep=float(trainer_optim.get("pkeep", 1.0)),
            sos_token=int(trainer_optim.get("sos_token", 0)),
            transformer_dropout=float(model_cfg.get("transformer_dropout", 0.0)),
            val_logging=trainer_val_logging,
        )

    if model_name == "dsm_latent":
        vq_ckpt = model_cfg.get("vq_ckpt_path")
        if not vq_ckpt or str(vq_ckpt).lower() == "null":
            raise ValueError(
                "DSM latent requires model.vq_ckpt_path to point to a trained VQ-VAE checkpoint. "
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

        trainer_optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
        trainer_val_logging = OmegaConf.to_container(
            cfg.trainer.val_logging, resolve=True
        )

        # Optional consistency checks if dsm_latent.yaml also specifies these fields.
        if "n_embed" in model_cfg and int(model_cfg["n_embed"]) != vq_n_embed:
            raise ValueError(
                "dsm_latent.n_embed does not match VQ-VAE n_embed in checkpoint config"
            )
        if "embed_dim" in model_cfg and int(model_cfg["embed_dim"]) != vq_embed_dim:
            raise ValueError(
                "dsm_latent.embed_dim does not match VQ-VAE embed_dim in checkpoint config"
            )

        min_sigma = float(model_cfg.get("min_sigma", 0.01))
        max_sigma = float(model_cfg.get("max_sigma", 0.2))
        num_sigmas = int(model_cfg.get("num_sigmas", 4))
        score_backbone = str(model_cfg.get("score_backbone", "unet"))
        use_ema = bool(model_cfg.get("use_ema", True))
        ema_decay = float(model_cfg.get("ema_decay", 0.999))

        return DSMLatentInterface(
            ddconfig=vq_ddconfig,
            n_embed=vq_n_embed,
            embed_dim=vq_embed_dim,
            vq_ckpt_path=str(vq_ckpt),
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigmas=num_sigmas,
            base_channels=int(model_cfg.get("base_channels", 64)),
            image_key=image_key,
            learning_rate=float(trainer_optim["learning_rate"]),
            anneal_power=float(model_cfg.get("anneal_power", 2.0)),
            use_annealed_loss=bool(model_cfg.get("use_annealed_loss", True)),
            val_logging=trainer_val_logging,
            score_backbone=score_backbone,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )

    if model_name == "ddpm_latent":
        vq_ckpt = model_cfg.get("vq_ckpt_path")
        if not vq_ckpt or str(vq_ckpt).lower() == "null":
            raise ValueError(
                "DDPM latent requires model.vq_ckpt_path to point to a trained "
                "VQ-VAE checkpoint. "
                "Override with e.g. model.vq_ckpt_path=checkpoints/xyz/vqvae-last.ckpt"
            )

        ckpt_dir = os.path.dirname(str(vq_ckpt))
        ckpt_cfg_path = os.path.join(ckpt_dir, "config.yaml")
        if not os.path.exists(ckpt_cfg_path):
            raise FileNotFoundError(
                f"Expected VQ-VAE config at {ckpt_cfg_path}. "
                "Ensure you saved config.yaml in the checkpoint folder when "
                "training VQ-VAE."
            )
        ckpt_cfg = OmegaConf.load(ckpt_cfg_path)
        vq_model_cfg = OmegaConf.to_container(ckpt_cfg.model, resolve=True)
        vq_ddconfig = dict(vq_model_cfg["ddconfig"])
        vq_n_embed = int(vq_model_cfg["n_embed"])
        vq_embed_dim = int(vq_model_cfg["embed_dim"])

        trainer_optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
        trainer_val_logging = OmegaConf.to_container(
            cfg.trainer.val_logging, resolve=True
        )

        if "n_embed" in model_cfg and int(model_cfg["n_embed"]) != vq_n_embed:
            raise ValueError(
                "ddpm_latent.n_embed does not match VQ-VAE n_embed in "
                "checkpoint config"
            )
        if "embed_dim" in model_cfg and int(model_cfg["embed_dim"]) != vq_embed_dim:
            raise ValueError(
                "ddpm_latent.embed_dim does not match VQ-VAE embed_dim in "
                "checkpoint config"
            )

        att_res = model_cfg.get("attention_resolutions")
        if att_res is not None:
            att_res = tuple(att_res)
        ch_mult = model_cfg.get("channel_mult", (1, 2, 4, 8))
        ch_mult = tuple(ch_mult)

        ddpm_cond = _ddpm_conditioning_kwargs(model_cfg)

        return DDPMLatentInterface(
            ddconfig=vq_ddconfig,
            n_embed=vq_n_embed,
            embed_dim=vq_embed_dim,
            vq_ckpt_path=str(vq_ckpt),
            image_key=image_key,
            learning_rate=float(trainer_optim["learning_rate"]),
            timesteps=int(model_cfg.get("timesteps", 1000)),
            beta_schedule=str(model_cfg.get("beta_schedule", "linear")),
            linear_start=float(model_cfg.get("linear_start", 1e-4)),
            linear_end=float(model_cfg.get("linear_end", 2e-2)),
            cosine_s=float(model_cfg.get("cosine_s", 8e-3)),
            parameterization=str(model_cfg.get("parameterization", "eps")),
            loss_type=str(model_cfg.get("loss_type", "l2")),
            l_simple_weight=float(model_cfg.get("l_simple_weight", 1.0)),
            original_elbo_weight=float(model_cfg.get("original_elbo_weight", 0.0)),
            base_channels=int(model_cfg.get("base_channels", 64)),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
            attention_resolutions=att_res,
            channel_mult=ch_mult,
            dropout=float(model_cfg.get("dropout", 0.0)),
            logit_transform=bool(model_cfg.get("logit_transform", True)),
            use_ema=bool(model_cfg.get("use_ema", False)),
            ema_decay=float(model_cfg.get("ema_decay", 0.999)),
            val_logging=trainer_val_logging,
            sampling_cfg=model_cfg.get("sampling"),
            **ddpm_cond,
        )

    if model_name == "ddpm_raw":
        trainer_optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
        trainer_val_logging = OmegaConf.to_container(
            cfg.trainer.val_logging, resolve=True
        )
        in_channels = int(cfg.data.in_channels)
        image_size = int(cfg.data.params.image_size)

        att_res = model_cfg.get("attention_resolutions")
        if att_res is not None:
            att_res = tuple(att_res)
        ch_mult = model_cfg.get("channel_mult", (1, 2, 4, 8))
        ch_mult = tuple(ch_mult)

        ddpm_cond = _ddpm_conditioning_kwargs(model_cfg)

        return DDPMRawInterface(
            in_channels=in_channels,
            image_size=image_size,
            image_key=image_key,
            learning_rate=float(trainer_optim["learning_rate"]),
            timesteps=int(model_cfg.get("timesteps", 1000)),
            beta_schedule=str(model_cfg.get("beta_schedule", "linear")),
            linear_start=float(model_cfg.get("linear_start", 1e-4)),
            linear_end=float(model_cfg.get("linear_end", 2e-2)),
            cosine_s=float(model_cfg.get("cosine_s", 8e-3)),
            parameterization=str(model_cfg.get("parameterization", "eps")),
            loss_type=str(model_cfg.get("loss_type", "l2")),
            l_simple_weight=float(model_cfg.get("l_simple_weight", 1.0)),
            original_elbo_weight=float(model_cfg.get("original_elbo_weight", 0.0)),
            base_channels=int(model_cfg.get("base_channels", 64)),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
            attention_resolutions=att_res,
            channel_mult=ch_mult,
            dropout=float(model_cfg.get("dropout", 0.0)),
            logit_transform=bool(model_cfg.get("logit_transform", False)),
            use_ema=bool(model_cfg.get("use_ema", False)),
            ema_decay=float(model_cfg.get("ema_decay", 0.999)),
            val_logging=trainer_val_logging,
            sampling_cfg=model_cfg.get("sampling"),
            **ddpm_cond,
        )

    if model_name == "dsm_raw":
        trainer_optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
        trainer_val_logging = OmegaConf.to_container(
            cfg.trainer.val_logging, resolve=True
        )
        in_channels = int(cfg.data.in_channels)
        image_size = int(cfg.data.params.image_size)

        min_sigma = float(model_cfg.get("min_sigma", 0.01))
        max_sigma = float(model_cfg.get("max_sigma", 0.2))
        num_sigmas = int(model_cfg.get("num_sigmas", 4))
        score_backbone = str(model_cfg.get("score_backbone", "unet"))
        use_ema = bool(model_cfg.get("use_ema", True))
        ema_decay = float(model_cfg.get("ema_decay", 0.999))

        return DSMRawInterface(
            in_channels=in_channels,
            image_size=image_size,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigmas=num_sigmas,
            base_channels=int(model_cfg.get("base_channels", 64)),
            image_key=image_key,
            learning_rate=float(trainer_optim["learning_rate"]),
            anneal_power=float(model_cfg.get("anneal_power", 2.0)),
            use_annealed_loss=bool(model_cfg.get("use_annealed_loss", True)),
            val_logging=trainer_val_logging,
            score_backbone=score_backbone,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )

    if model_name == "score_sde_latent":
        vq_ckpt = model_cfg.get("vq_ckpt_path")
        if not vq_ckpt or str(vq_ckpt).lower() == "null":
            raise ValueError(
                "Score-SDE requires model.vq_ckpt_path to point to a trained VQ-VAE checkpoint. "
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

        trainer_optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
        trainer_val_logging = OmegaConf.to_container(
            cfg.trainer.val_logging, resolve=True
        )

        return ScoreSDEInterface(
            ddconfig=vq_ddconfig,
            n_embed=vq_n_embed,
            embed_dim=vq_embed_dim,
            vq_ckpt_path=str(vq_ckpt),
            image_key=image_key,
            learning_rate=float(trainer_optim["learning_rate"]),
            # SDE/time options
            sde_type=str(model_cfg.get("sde_type", "vesde")),
            sde_n=int(model_cfg.get("sde_n", 1000)),
            sigma_min=float(model_cfg.get("sigma_min", 0.01)),
            sigma_max=float(model_cfg.get("sigma_max", 50.0)),
            beta_min=float(model_cfg.get("beta_min", 0.1)),
            beta_max=float(model_cfg.get("beta_max", 20.0)),
            continuous=bool(model_cfg.get("continuous", True)),
            t_eps=float(model_cfg.get("t_eps", 1e-3)),
            likelihood_weighting=bool(model_cfg.get("likelihood_weighting", True)),
            # Model/backbone
            base_channels=int(model_cfg.get("base_channels", 64)),
            logit_transform=bool(model_cfg.get("logit_transform", False)),
            # EMA
            use_ema=bool(model_cfg.get("use_ema", False)),
            ema_decay=float(model_cfg.get("ema_decay", 0.999)),
            val_logging=trainer_val_logging,
            sampling_cfg=model_cfg.get("sampling"),
        )

    if model_name == "score_sde_raw":
        trainer_optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
        trainer_val_logging = OmegaConf.to_container(
            cfg.trainer.val_logging, resolve=True
        )

        in_channels = int(cfg.data.in_channels)
        image_size = int(cfg.data.params.image_size)

        return ScoreSDERawInterface(
            in_channels=in_channels,
            image_size=image_size,
            image_key=image_key,
            learning_rate=float(trainer_optim["learning_rate"]),
            # SDE/time options
            sde_type=str(model_cfg.get("sde_type", "vesde")),
            sde_n=int(model_cfg.get("sde_n", 1000)),
            sigma_min=float(model_cfg.get("sigma_min", 0.01)),
            sigma_max=float(model_cfg.get("sigma_max", 50.0)),
            beta_min=float(model_cfg.get("beta_min", 0.1)),
            beta_max=float(model_cfg.get("beta_max", 20.0)),
            continuous=bool(model_cfg.get("continuous", True)),
            t_eps=float(model_cfg.get("t_eps", 1e-3)),
            likelihood_weighting=bool(model_cfg.get("likelihood_weighting", True)),
            # Model/backbone
            base_channels=int(model_cfg.get("base_channels", 64)),
            logit_transform=bool(model_cfg.get("logit_transform", False)),
            # EMA
            use_ema=bool(model_cfg.get("use_ema", False)),
            ema_decay=float(model_cfg.get("ema_decay", 0.999)),
            val_logging=trainer_val_logging,
            sampling_cfg=model_cfg.get("sampling"),
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
    max_depth: int = 4,
    prefix: str = "",
    depth: int = 0,
    name: str | None = None,
) -> None:
    """Print module hierarchy with parameter counts for the top max_depth levels."""
    param_count = _module_param_count(module)
    display_name = name or module.__class__.__name__
    if depth == 0:
        print(f"\n===== Model (top {max_depth} levels) =====")
    print(
        f"{prefix}{display_name} ({module.__class__.__name__}): {param_count:,} params"
    )
    if depth < max_depth - 1:
        for child_name, child in module.named_children():
            _print_model_tree(child, max_depth, prefix + "  ", depth + 1, child_name)


def _print_model_summary(module: nn.Module) -> None:
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in module.parameters())
    # Approximate fp32 footprint in megabytes.
    total_param_size_mb = (total_params * 4) / (1024**2)

    _print_model_tree(module, max_depth=4)
    print("===== Parameter Summary =====")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Approx parameter size (fp32): {total_param_size_mb:.2f} MB\n")


def _save_checkpoint_dir_config(cfg: DictConfig, ckpt_dir: str) -> None:
    """Write resolved Hydra config next to checkpoints for later reinstantiation."""
    os.makedirs(ckpt_dir, exist_ok=True)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))


def _resolve_vqvae_checkpoint_monitor(cfg: DictConfig) -> tuple[str, str]:
    """Return (monitor_key, filename_metric_token) for VQ-VAE checkpointing."""
    raw = str(
        OmegaConf.select(cfg, "trainer.checkpoint.monitor_metric", default="rec_loss")
    ).lower()
    if raw == "rec_loss":
        return "val/rec_loss", "val_rec_loss"
    if raw == "rfid":
        return "val/rfid", "val_rfid"
    raise ValueError("trainer.checkpoint.monitor_metric must be one of: rec_loss, rfid")


@hydra.main(
    config_path="configs",
    config_name="train_mnist_score_sde_ncsnv2_latent",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)

    datamodule = _build_datamodule(cfg.data)
    module = _build_module(cfg)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer.lightning, resolve=True)
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
    if model_name in (
        "transformer_latent",
        "ddpm_latent",
        "ddpm_raw",
        "dsm_latent",
        "dsm_raw",
        "score_sde_latent",
        "score_sde_raw",
    ):
        monitor, mode, filename = "val/loss", "min", "latent-{epoch}-{step}"
        ckpt_auto_insert_metric_name = True
    else:
        monitor, metric_token = _resolve_vqvae_checkpoint_monitor(cfg)
        mode = "min"
        # Brace key must match logged metric name (val/rfid); token before "=" is a safe label.
        prefix = "vqgan" if model_name == "vqgan" else "vqvae"
        filename = (
            f"{prefix}-epoch={{epoch}}-step={{step}}-{metric_token}={{{monitor}:.4f}}"
        )
        ckpt_auto_insert_metric_name = False
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join("checkpoints", checkpoint_subdir),
            monitor=monitor,
            mode=mode,
            save_top_k=2,
            save_last=True,
            filename=filename,
            auto_insert_metric_name=ckpt_auto_insert_metric_name,
        )
    ]
    if cfg.wandb.enabled:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    ckpt_dir = os.path.join("checkpoints", checkpoint_subdir)
    _save_checkpoint_dir_config(cfg, ckpt_dir)

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)
    _print_model_summary(module)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
