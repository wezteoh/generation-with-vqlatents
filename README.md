# Generative Modeling with VQ Latents

PyTorch (Lightning) + Hydra codebase for generative models built on vector-quantized latent representations: VQ-VAE/VQGAN for learning discrete latents, autoregressive transformer priors, and score-based or diffusion generators. Besides MNIST and ImageNet-oriented data configs, there are CelebA-HQ (256) VQ training entrypoints.

## Techniques

- **Vector-quantized autoencoders:** VQ-VAE, VQGAN
- **Latent sequence prior:** autoregressive transformer over quantized codes
- **Generative priors / score models:**
  - **DSM** (denoising score matching): UNet, conditional refined net, and NCSNv2 variants—latent and raw-pixel (see `train_mnist_dsm_*.yaml`)
  - **Score SDE** (NCSNv2-style): latent and raw (`train_mnist_score_sde_ncsnv2_*`)
  - **DDPM:** raw-pixel and latent diffusion with OpenAI-style UNets; optional class or context conditioning via `model.conditioning` in configs. Optional validation FID when `trainer.val_logging.sample_fid` is enabled (helper in `src/interfaces/ddpm_sample_fid.py`)

`train.py` dispatches on `model.name`: `vqvae`, `vqgan`, `transformer_prior`, `dsm_latent`, `dsm_raw`, `ddpm_latent`, `ddpm_raw`, `score_sde_latent`, `score_sde_raw`.

## Project structure

| Path | Role |
|------|------|
| **`train.py`** | Single Hydra entrypoint: loads a train config, builds the datamodule and Lightning interface from `model.name`. |
| **`configs/`** | Hydra YAML. Top-level `train_*.yaml` files set data, model, trainer, and W&B. Model fragments live under `configs/model/`. |
| **`src/data/`** | Datamodules: `mnist.py` (MNIST), labeled MNIST (`MNISTLabeledDataModule`), `imagenet.py`, `celebahq.py` (CelebA-HQ 256), plus `base.py`. |
| **`src/modules/`** | `nn.Module` building blocks: `autoencoders/`, `quantizers/`, `latents/` (`ddpm/`, `score_models/`, `score_sde/`, `diffusion_backbones/`, `autoregressives/`), `losses/` (including `ddpm.py`, score matching, VAE, GAN), `discriminators/`, `perceptual/`, `shared/`. |
| **`src/interfaces/`** | Lightning modules per task: `vqvae.py`, `vqgan.py`, `transformer_latent.py`, `dsm_raw.py`, `dsm_latent.py`, `ddpm_raw.py`, `ddpm_latent.py`, `score_sde_raw.py`, `score_sde_latent.py`; `ddpm_sample_fid.py` implements optional val FID for DDPM. |

Training is wired by config: the chosen `train_*.yaml` sets `model.name`, which selects the interface in `train.py` and the matching `configs/model/*.yaml` fragment.

## Environment

- Python **3.12+** (`requires-python` in `pyproject.toml`).
- Dependencies are managed with **`uv`**. `pyproject.toml` pins PyTorch and may use a CUDA wheel index; adjust `tool.uv` if you need CPU-only or a different CUDA build.

## Quickstart

From the repo root, install and activate the venv (e.g. `uv sync`, then `source .venv/bin/activate`). Configure W&B as needed for experiment tracking.

Example runs:

```bash
# DDPM on MNIST (raw pixels; unconditional and class-conditional examples)
python train.py --config-name train_mnist_ddpm_raw
python train.py --config-name train_mnist_ddpm_raw_cond

# DDPM in VQ-VAE latent space (unconditional / class-conditional)
python train.py --config-name train_mnist_ddpm_latent
python train.py --config-name train_mnist_ddpm_latent_cond

# Raw-pixel DSM on MNIST (cond refined net)
python train.py --config-name train_mnist_dsm_condrefine_raw

# Latent UNet DSM on MNIST
python train.py --config-name train_mnist_dsm_unet_latent

# Score SDE (NCSNv2) on MNIST
python train.py --config-name train_mnist_score_sde_ncsnv2_raw
python train.py --config-name train_mnist_score_sde_ncsnv2_latent

# VQ-VAE on MNIST, then transformer prior (prior needs a trained VQ-VAE checkpoint + config.yaml beside the checkpoint)
python train.py --config-name train_mnist_vqvae
python train.py --config-name train_mnist_transformer_latent model.vq_ckpt_path=checkpoints/<run_id>/vqvae-last.ckpt

# Higher-resolution VQ on CelebA-HQ 256 (set data paths via config overrides; do not hardcode secrets)
python train.py --config-name train_celebahq256_vqvae
python train.py --config-name train_celebahq256_vqgan
```

## Extending

- Add reusable components under **`src/modules/`**.
- Add a Lightning interface under **`src/interfaces/`** and a new branch in **`train.py`** keyed by `model.name`, plus a **`configs/model/`** fragment and a top-level **`train_*.yaml`** that references it.
