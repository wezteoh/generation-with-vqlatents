import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision import datasets

from src.data.base import build_image_transforms
from src.interfaces.vqvae import VQVAEInterface


def load_mnist_batch(
    data_root: str | Path,
    n_samples: int = 1000,
    train: bool = True,
) -> torch.Tensor:
    """
    Load a single batch of MNIST images.

    Images are returned as a tensor of shape [N, C, H, W] with standard
    MNIST normalization applied.
    """
    data_root = Path(data_root)

    # Match the default MNISTDataModule preprocessing: grayscale + ToTensor + Normalize
    transform = build_image_transforms(
        resize_to=32,
        split="train" if train else "val",
        mean=(0.5,),
        std=(0.5,),
        num_channels=1,
    )

    dataset = datasets.MNIST(
        root=str(data_root),
        train=train,
        download=True,
        transform=transform,
    )

    if n_samples > len(dataset):
        raise ValueError(
            f"Requested n_samples={n_samples}, " f"but dataset only has {len(dataset)} examples."
        )

    # Take the first n_samples from a shuffled subset via DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_samples,
        shuffle=True,
        drop_last=False,
    )

    images, _ = next(iter(loader))
    return images


def compute_pairwise_l2_stats(x: torch.Tensor) -> tuple[float, float, float]:
    """
    Compute min, median (50th percentile), and max pairwise L2 distances.

    Args:
        x: Tensor of shape [N, ...] (images or generic encodings).

    Returns:
        (min_distance, median_distance, max_distance)
    """
    if x.ndim < 2:
        raise ValueError(f"Expected tensor with at least 2 dims [N, ...], got {x.shape}")

    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples to compute pairwise distances.")

    # Flatten to [N, D]
    x_flat = x.view(n, -1)

    # Compute squared L2 distance matrix in a vectorized way
    # dist_sq[i, j] = ||x_i - x_j||_2^2
    xx = (x_flat**2).sum(dim=1, keepdim=True)  # [N, 1]
    dist_sq = xx + xx.t() - 2.0 * (x_flat @ x_flat.t())
    dist_sq = dist_sq.clamp_min(0.0)
    dist = torch.sqrt(dist_sq)

    # Extract unique pairs (i < j), excluding diagonal
    idx = torch.triu_indices(n, n, offset=1)
    pairwise = dist[idx[0], idx[1]]

    d_min = pairwise.min().item()
    d_med = pairwise.median().item()
    d_max = pairwise.max().item()
    return d_min, d_med, d_max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load N samples from the MNIST dataset and compute min, "
            "median, and max pairwise L2 distances between them."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for storing the MNIST dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of MNIST samples to draw from the training set (default: %(default)s).",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force computations on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--vq-ckpt-dir",
        type=str,
        default=None,
        help=(
            "Optional path to a directory containing a trained VQ-VAE checkpoint and "
            "its config.yaml. If provided, pairwise distances are computed on VQ-VAE "
            "encodings instead of raw pixels."
        ),
    )
    parser.add_argument(
        "--vq-ckpt-name",
        type=str,
        default="last.ckpt",
        help=(
            "Checkpoint filename inside --vq-ckpt-dir (default: %(default)s). "
            "Ignored if --vq-ckpt-dir is not set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")

    images = load_mnist_batch(
        data_root=args.data_root,
        n_samples=args.n_samples,
        train=True,
    )
    images = images.to(device)

    if args.vq_ckpt_dir is None:
        d_min, d_med, d_max = compute_pairwise_l2_stats(images)

        print(f"Number of samples: {args.n_samples}")
        print(f"Device: {device}")
        print(f"Input tensor shape (pixels): {tuple(images.shape)}")
        print(f"Min pairwise L2 distance: {d_min:.6f}")
        print(f"Median pairwise L2 distance (50th percentile): {d_med:.6f}")
        print(f"Max pairwise L2 distance: {d_max:.6f}")
        return

    ckpt_dir = Path(args.vq_ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"--vq-ckpt-dir={ckpt_dir} is not a valid directory. "
            "Expected a directory containing config.yaml and a VQ-VAE checkpoint."
        )

    cfg_path = ckpt_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Expected VQ-VAE config at {cfg_path}. "
            "Ensure you saved config.yaml in the checkpoint folder when training VQ-VAE."
        )

    ckpt_path = ckpt_dir / args.vq_ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Expected VQ-VAE checkpoint file at {ckpt_path}. "
            "Use --vq-ckpt-name to point to the correct checkpoint file."
        )

    ckpt_cfg = OmegaConf.load(str(cfg_path))
    model_cfg = OmegaConf.to_container(ckpt_cfg.model, resolve=True)
    model_name = model_cfg.get("name", "vqvae")
    if model_name != "vqvae":
        raise ValueError(
            f"Checkpoint config at {cfg_path} has model.name={model_name!r}, "
            "but this script expects a VQ-VAE (model.name='vqvae')."
        )

    ddconfig = dict(model_cfg["ddconfig"])
    n_embed = int(model_cfg["n_embed"])
    embed_dim = int(model_cfg["embed_dim"])
    image_key = model_cfg.get("image_key", "image")

    vq = VQVAEInterface.load_from_checkpoint(
        str(ckpt_path),
        ddconfig=ddconfig,
        n_embed=n_embed,
        embed_dim=embed_dim,
        image_key=image_key,
    )
    vq = vq.to(device)
    vq.eval()

    with torch.no_grad():
        quant, _, _ = vq.model.encode(images)

    encodings = quant

    d_min, d_med, d_max = compute_pairwise_l2_stats(encodings)

    print(f"Number of samples: {args.n_samples}")
    print(f"Device: {device}")
    print(f"Using VQ-VAE encodings from: {ckpt_path}")
    print(f"Input tensor shape (pixels): {tuple(images.shape)}")
    print(f"Encoding tensor shape: {tuple(encodings.shape)}")
    print(f"Min pairwise L2 distance (encodings): {d_min:.6f}")
    print(f"Median pairwise L2 distance (50th percentile, encodings): {d_med:.6f}")
    print(f"Max pairwise L2 distance (encodings): {d_max:.6f}")


if __name__ == "__main__":
    main()
