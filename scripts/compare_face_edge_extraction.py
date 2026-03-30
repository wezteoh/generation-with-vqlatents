"""Load a few CelebA-HQ images and visualize edge extraction.

Uses Sobel (torch) and PIL FIND_EDGES, with Gaussian pre-blur, high-quantile
thresholding, and small-component removal so only major edges remain as solid
binary lines. Also runs a NumPy/SciPy Canny (NMS + hysteresis) on luminance
for comparison. Edge panels use grayscale; use --invert / --no-invert to swap
foreground and background.

Paths are not hardcoded: pass --data-root or set the environment variable
CELEBA_HQ_DATA_ROOT to your flat image folder.

Example:
  source .venv/bin/activate
  python scripts/test_celebahq_edge_extraction.py \\
    --data-root /path/to/celeba_hq_256 --num-images 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageFilter
from scipy import ndimage
from torchvision import transforms as T
from torchvision.transforms import functional as TF


def _resolve_data_root(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.expanduser().resolve()
    env = os.environ.get("CELEBA_HQ_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    raise SystemExit(
        "Provide --data-root or set CELEBA_HQ_DATA_ROOT to the CelebA-HQ "
        "image directory."
    )


def _list_image_paths(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    paths = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(
            f"No images with extensions {extensions} found under {root}"
        )
    return paths


def load_rgb_01(path: Path, resize: int | None) -> torch.Tensor:
    """RGB tensor [3, H, W] in [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if resize is not None:
            img = img.resize((resize, resize), Image.Resampling.LANCZOS)
    t = T.functional.to_tensor(img)
    return t


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """[3, H, W] -> [1, H, W] Rec. 601 luma."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(0)


def sobel_magnitude(gray: torch.Tensor) -> torch.Tensor:
    """gray [1, H, W] -> edge magnitude [H, W]."""
    x = gray.unsqueeze(0)
    device, dtype = x.device, x.dtype
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return mag.squeeze(0).squeeze(0)


def gaussian_blur_gray(gray: torch.Tensor, sigma: float) -> torch.Tensor:
    """gray [1, H, W]; sigma 0 skips blur."""
    if sigma <= 0:
        return gray
    k = int(2 * round(3 * sigma) + 1)
    k = max(3, k | 1)
    return TF.gaussian_blur(gray, kernel_size=[k, k], sigma=[sigma, sigma])


def major_edges_binary(
    strength: np.ndarray,
    quantile: float,
    min_component_pixels: int,
) -> np.ndarray:
    """Keep only pixels at or above the given quantile; drop small components.

    Returns float32 {0, 1} with 1 = edge.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    thr = float(np.quantile(strength.reshape(-1), quantile))
    mask = strength >= thr
    labeled, num = ndimage.label(mask)
    if num == 0:
        return np.zeros_like(strength, dtype=np.float32)
    counts = np.bincount(labeled.ravel(), minlength=num + 1)
    keep = counts >= min_component_pixels
    keep[0] = False
    return keep[labeled].astype(np.float32)


def apply_edge_display(binary_01: np.ndarray, invert: bool) -> np.ndarray:
    """If invert, swap foreground/background (black <-> white)."""
    if invert:
        return 1.0 - binary_01
    return binary_01.astype(np.float32)


def _nms_gradient(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Non-maximum suppression; 4-direction quantization of gradient angle."""
    angle = np.abs(np.degrees(np.arctan2(gy, gx)))
    angle = np.mod(angle, 180.0)

    q0 = (angle <= 22.5) | (angle > 157.5)
    q1 = (angle > 22.5) & (angle <= 67.5)
    q2 = (angle > 67.5) & (angle <= 112.5)
    q3 = (angle > 112.5) & (angle <= 157.5)

    west = np.roll(mag, 1, axis=1)
    east = np.roll(mag, -1, axis=1)
    north = np.roll(mag, 1, axis=0)
    south = np.roll(mag, -1, axis=0)
    ne = np.roll(np.roll(mag, 1, axis=0), -1, axis=1)
    sw = np.roll(np.roll(mag, -1, axis=0), 1, axis=1)
    nw = np.roll(np.roll(mag, 1, axis=0), 1, axis=1)
    se = np.roll(np.roll(mag, -1, axis=0), -1, axis=1)

    m0 = (mag >= west) & (mag >= east)
    m1 = (mag >= ne) & (mag >= sw)
    m2 = (mag >= north) & (mag >= south)
    m3 = (mag >= nw) & (mag >= se)

    keep = (q0 & m0) | (q1 & m1) | (q2 & m2) | (q3 & m3)
    nms = np.where(keep, mag, 0.0).astype(np.float64)
    nms[0, :] = nms[-1, :] = nms[:, 0] = nms[:, -1] = 0.0
    return nms


def _hysteresis(nms: np.ndarray, low: float, high: float) -> np.ndarray:
    """Double-threshold edge tracking (8-connectivity)."""
    strong = nms >= high
    weak = (nms >= low) & (nms < high)
    edges = strong.copy()
    struct = np.ones((3, 3), dtype=bool)
    while True:
        dil = ndimage.binary_dilation(edges, structure=struct)
        new_edges = edges | (weak & dil)
        if np.array_equal(new_edges, edges):
            break
        edges = new_edges
    return edges.astype(np.float32)


def canny_edges(
    gray_hw: np.ndarray,
    *,
    gaussian_sigma: float,
    low_frac: float,
    high_frac: float,
) -> np.ndarray:
    """Canny on grayscale [H, W] float (e.g. [0, 1]). Returns {0,1} float32 edges.

    ``low_frac`` / ``high_frac`` are fractions of the max NMS magnitude (after
    smoothing), matching common OpenCV-style relative thresholds.
    """
    if gray_hw.ndim != 2:
        raise ValueError(f"Expected HxW grayscale, got shape {gray_hw.shape}")
    if not (0.0 < low_frac < high_frac):
        raise ValueError(
            f"Need 0 < low_frac < high_frac, got low={low_frac}, high={high_frac}"
        )
    smooth = ndimage.gaussian_filter(gray_hw.astype(np.float64), sigma=gaussian_sigma)
    gx = ndimage.sobel(smooth, axis=1)
    gy = ndimage.sobel(smooth, axis=0)
    mag = np.hypot(gx, gy)
    nms = _nms_gradient(mag, gx, gy)
    peak = float(nms.max())
    if peak <= 1e-12:
        return np.zeros_like(gray_hw, dtype=np.float32)
    high = high_frac * peak
    low = low_frac * peak
    if low >= high:
        low = 0.5 * high
    return _hysteresis(nms, low, high)


def pil_find_edges(rgb_01: torch.Tensor) -> torch.Tensor:
    """Apply PIL FIND_EDGES; return [H, W] in [0, 1]."""
    chw = (rgb_01 * 255.0).clamp(0, 255).byte().cpu().numpy()
    rgb = rearrange(chw, "c h w -> h w c")
    pil = Image.fromarray(rgb, mode="RGB")
    edges = pil.filter(ImageFilter.FIND_EDGES).convert("L")
    return torch.from_numpy(np.array(edges, dtype=np.float32)) / 255.0


def compute_clean_edge_maps(
    rgb: torch.Tensor,
    device: torch.device,
    blur_sigma: float,
    sobel_quantile: float,
    pil_quantile: float,
    min_component_pixels: int,
    invert: bool,
    canny_sigma: float,
    canny_low_frac: float,
    canny_high_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns rgb_hwc, sobel, pil, canny displays in [0,1] for imshow."""
    rgb_d = rgb.to(device)
    gray = luminance(rgb_d)
    gray = gaussian_blur_gray(gray, blur_sigma)
    sob_mag = sobel_magnitude(gray).detach().cpu().numpy()
    sob_bin = major_edges_binary(sob_mag, sobel_quantile, min_component_pixels)
    pil_raw = pil_find_edges(rgb.cpu()).numpy()
    pil_bin = major_edges_binary(pil_raw, pil_quantile, min_component_pixels)
    gray_hw = luminance(rgb.cpu()).squeeze(0).numpy()
    canny_bin = canny_edges(
        gray_hw,
        gaussian_sigma=canny_sigma,
        low_frac=canny_low_frac,
        high_frac=canny_high_frac,
    )
    sob_disp = apply_edge_display(sob_bin, invert)
    pil_disp = apply_edge_display(pil_bin, invert)
    canny_disp = apply_edge_display(canny_bin, invert)
    rgb_np = rearrange(rgb.cpu(), "c h w -> h w c").numpy()
    return rgb_np, sob_disp, pil_disp, canny_disp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Flat folder of CelebA-HQ images (.jpg, .png, ...). "
        "If omitted, uses CELEBA_HQ_DATA_ROOT.",
    )
    parser.add_argument(
        "--num-images", type=int, default=4, help="How many images to visualize."
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional indices into the lexicographic file list "
            "(overrides --num-images)."
        ),
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional square resize (e.g. 256). Default: native resolution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp") / "celebahq_edge_viz",
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for Sobel (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.25,
        help=(
            "Gaussian blur on luminance before Sobel (0 disables). "
            "Reduces fine texture."
        ),
    )
    parser.add_argument(
        "--sobel-quantile",
        type=float,
        default=0.7,
        help="Keep Sobel magnitude at or above this quantile (higher = fewer edges).",
    )
    parser.add_argument(
        "--pil-quantile",
        type=float,
        default=0.7,
        help="Same for PIL FIND_EDGES strength map before cleaning.",
    )
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=120,
        help="Drop connected edge blobs smaller than this (after thresholding).",
    )
    parser.add_argument(
        "--invert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Swap black/white on edge panels. Default true: black edges on white "
            "background. Use --no-invert for white edges on black."
        ),
    )
    parser.add_argument(
        "--canny-sigma",
        type=float,
        default=1.4,
        help="Gaussian sigma for Canny pre-smoothing (SciPy, on luminance).",
    )
    parser.add_argument(
        "--canny-low-frac",
        type=float,
        default=0.15,
        help="Canny weak threshold as fraction of max NMS response.",
    )
    parser.add_argument(
        "--canny-high-frac",
        type=float,
        default=0.25,
        help="Canny strong threshold as fraction of max NMS response.",
    )
    args = parser.parse_args()

    root = _resolve_data_root(args.data_root)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = _list_image_paths(root, exts)

    if args.indices is not None:
        chosen = []
        for i in args.indices:
            if i < 0 or i >= len(paths):
                raise IndexError(f"Index {i} out of range for {len(paths)} images")
            chosen.append(paths[i])
    else:
        n = min(args.num_images, len(paths))
        chosen = paths[:n]

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(chosen), 4, figsize=(12, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = axes.reshape(1, -1)

    for row, path in enumerate(chosen):
        rgb = load_rgb_01(path, args.resize)
        rgb_np, sob_disp, pil_disp, canny_disp = compute_clean_edge_maps(
            rgb,
            device,
            args.blur_sigma,
            args.sobel_quantile,
            args.pil_quantile,
            args.min_component_pixels,
            args.invert,
            args.canny_sigma,
            args.canny_low_frac,
            args.canny_high_frac,
        )

        axes[row, 0].imshow(rgb_np.clip(0, 1))
        axes[row, 0].set_title(f"{path.name}\nRGB")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(sob_disp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title("Sobel (major edges)")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pil_disp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("PIL (major edges)")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(canny_disp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 3].set_title("Canny")
        axes[row, 3].axis("off")

    plt.tight_layout()
    out = args.output_dir / "celebahq_edges_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.resolve()}")

    for path in chosen:
        rgb = load_rgb_01(path, args.resize)
        rgb_np, sob_disp, pil_disp, canny_disp = compute_clean_edge_maps(
            rgb,
            device,
            args.blur_sigma,
            args.sobel_quantile,
            args.pil_quantile,
            args.min_component_pixels,
            args.invert,
            args.canny_sigma,
            args.canny_low_frac,
            args.canny_high_frac,
        )
        base = path.stem
        fig2, ax2 = plt.subplots(1, 4, figsize=(12, 3))
        ax2[0].imshow(rgb_np.clip(0, 1))
        ax2[0].set_title("RGB")
        ax2[0].axis("off")
        ax2[1].imshow(sob_disp, cmap="gray", vmin=0.0, vmax=1.0)
        ax2[1].set_title("Sobel (major)")
        ax2[1].axis("off")
        ax2[2].imshow(pil_disp, cmap="gray", vmin=0.0, vmax=1.0)
        ax2[2].set_title("PIL (major)")
        ax2[2].axis("off")
        ax2[3].imshow(canny_disp, cmap="gray", vmin=0.0, vmax=1.0)
        ax2[3].set_title("Canny")
        ax2[3].axis("off")
        plt.tight_layout()
        pout = args.output_dir / f"{base}_edges.png"
        fig2.savefig(pout, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Wrote {pout.resolve()}")


if __name__ == "__main__":
    main()
