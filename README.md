# Generative Modeling with VQ Latents

PyTorch (Lightning) + Hydra codebase for generative models built on vector-quantized latent representations: VQ-VAE/VQGAN for learning discrete latents, and transformer or denoising score matching (DSM) priors for generation. Includes MNIST and ImageNet-oriented configs.

## Techniques

- **Vector-quantized autoencoders:** VQVAE, VQGAN
- **Latent priors:** autoregressive transformer decoder; denoising score matching with Langevin dynamics (latent and raw-pixel variants)

## Project structure

| Path | Role |
|------|------|
| **`train.py`** | Single Hydra entrypoint: loads a train config and runs the chosen interface. |
| **`configs/`** | Hydra YAML configs. Top-level entries (e.g. `train_mnist_dsm_unet_latent.yaml`) define data, model, trainer, and W&B. Model-only configs live under `configs/model/`. |
| **`src/data/`** | Dataset definitions and datamodules (`mnist.py`, `imagenet.py`, `base.py`). |
| **`src/modules/`** | Reusable `nn.Module` building blocks: `autoencoders/`, `quantizers/`, `latents/` (score_models, autoregressives), `losses/`, `discriminators/`, `shared/`. |
| **`src/interfaces/`** | Lightning modules per task (e.g. `dsm_raw.py`, `dsm_latent.py`, `vqvae.py`, `transformer_latent.py`). Each owns the `model` and defines training/validation steps and logging. |

Training is wired by config: `train.py` loads a train config (e.g. `train_mnist_dsm_unet_latent`), which selects an interface and a model config. The interface instantiates the model from `src/modules/` and handles optimization and W&B logging.

## Quickstart

Install dependencies with `uv` (or `pip`) from the repo root, then activate the venv: `source .venv/bin/activate`.

Example runs:

```bash
# Raw-pixel DSM on MNIST (cond refined net)
python train.py --config-name train_mnist_dsm_condrefine_raw

# Latent UNet DSM on MNIST
python train.py --config-name train_mnist_dsm_unet_latent

# VQ-VAE on MNIST, then transformer prior on its latents (prior needs a trained VQ-VAE checkpoint)
python train.py --config-name train_mnist_vqvae
python train.py --config-name train_mnist_transformer_latent model.vq_ckpt_path=checkpoints/<run_id>/vqvae-last.ckpt
```

## Extending

- Add new model components under **`src/modules/`**.
- Add a new Lightning interface under **`src/interfaces/`** that builds and trains that model.
- Add a new train config under **`configs/`** (and optional `configs/model/` entry) to run the new interface with the desired data and trainer settings.
