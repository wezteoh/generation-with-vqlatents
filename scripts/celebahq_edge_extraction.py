"""Batch Canny edge extraction for CelebA-HQ (or any flat image folder).

Uses the same SciPy Canny pipeline as ``compare_face_edge_extraction.py``:
Rec. 601 luminance, Gaussian pre-smooth, Sobel, NMS, hysteresis with
fraction-of-peak thresholds. Writes grayscale JPEGs (single channel, mode L).

Paths are not hardcoded: pass --data-root or set CELEBA_HQ_DATA_ROOT.

Example:
  source .venv/bin/activate
  python scripts/celebahq_edge_extraction.py \\
    --data-root /path/to/celeba_hq_256 --output-dir tmp/celebahq_edges \\
    --workers 8
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


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
    """Canny on grayscale [H, W] float (e.g. [0, 1]). Returns {0,1} float32 edges."""
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


def apply_edge_display(binary_01: np.ndarray, invert: bool) -> np.ndarray:
    """If invert, swap foreground/background (black <-> white)."""
    if invert:
        return 1.0 - binary_01
    return binary_01.astype(np.float32)


def luminance_rgb01(rgb_hwc: np.ndarray) -> np.ndarray:
    """[H, W, 3] float RGB in [0, 1] -> [H, W] Rec. 601 luma."""
    r = rgb_hwc[..., 0]
    g = rgb_hwc[..., 1]
    b = rgb_hwc[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def load_rgb_hwc(path: Path, resize: int | None) -> np.ndarray:
    """RGB array [H, W, 3] float32 in [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if resize is not None:
            img = img.resize((resize, resize), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _resolve_data_root(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.expanduser().resolve()
    env = os.environ.get("CELEBA_HQ_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    raise SystemExit(
        "Provide --data-root or set CELEBA_HQ_DATA_ROOT to the image directory."
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


def _process_one(job: tuple[str, str, dict[str, Any]]) -> tuple[str, bool, str | None]:
    """Process a single image; returns (src_str, ok, error_message)."""
    src_str, out_str, params = job
    src = Path(src_str)
    out = Path(out_str)
    if params["skip_existing"] and out.is_file():
        return src_str, True, None
    try:
        rgb = load_rgb_hwc(src, params["resize"])
        gray_hw = luminance_rgb01(rgb)
        canny_bin = canny_edges(
            gray_hw,
            gaussian_sigma=params["canny_sigma"],
            low_frac=params["canny_low_frac"],
            high_frac=params["canny_high_frac"],
        )
        disp = apply_edge_display(canny_bin, params["invert"])
        u8 = np.clip(np.round(disp * 255.0), 0, 255).astype(np.uint8)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(u8, mode="L").save(
            out,
            format="JPEG",
            quality=params["jpeg_quality"],
            optimize=True,
            subsampling=params["jpeg_subsampling"],
        )
        return src_str, True, None
    except Exception as e:  # noqa: BLE001 — surface worker failures to main
        return src_str, False, f"{type(e).__name__}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Flat folder of images. If omitted, uses CELEBA_HQ_DATA_ROOT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output JPEGs (same stem as input, .jpg).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional square resize (e.g. 256). Default: native resolution.",
    )
    parser.add_argument(
        "--canny-sigma",
        type=float,
        default=1.4,
        help="Gaussian sigma for Canny pre-smoothing (on luminance).",
    )
    parser.add_argument(
        "--canny-low-frac",
        type=float,
        default=0.1,
        help="Canny weak threshold as fraction of max NMS response.",
    )
    parser.add_argument(
        "--canny-high-frac",
        type=float,
        default=0.2,
        help="Canny strong threshold as fraction of max NMS response.",
    )
    parser.add_argument(
        "--invert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Black edges on white background (default). --no-invert for the opposite.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 1–95.",
    )
    parser.add_argument(
        "--jpeg-subsampling",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="JPEG chroma subsampling; 0 = 4:4:4 (sharper). Ignored for grayscale L.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Process pool size. Reduce if I/O-bound.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many images (after sorting).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    args = parser.parse_args()

    if not (1 <= args.jpeg_quality <= 95):
        raise SystemExit("--jpeg-quality must be between 1 and 95")

    root = _resolve_data_root(args.data_root)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = _list_image_paths(root, exts)
    if args.limit is not None:
        paths = paths[: args.limit]

    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {
        "resize": args.resize,
        "canny_sigma": args.canny_sigma,
        "canny_low_frac": args.canny_low_frac,
        "canny_high_frac": args.canny_high_frac,
        "invert": args.invert,
        "jpeg_quality": args.jpeg_quality,
        "jpeg_subsampling": args.jpeg_subsampling,
        "skip_existing": args.skip_existing,
    }

    jobs = [(str(p), str(out_root / f"{p.stem}.jpg"), params) for p in paths]

    n_ok = 0
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_one, job): job[0] for job in jobs}
        for fut in tqdm(
            as_completed(futures),
            total=len(jobs),
            desc="Canny edges",
            unit="img",
        ):
            src_str, ok, err = fut.result()
            if ok:
                n_ok += 1
            else:
                failures.append((src_str, err or "unknown"))

    print(f"Done: {n_ok}/{len(jobs)} succeeded, output under {out_root}")
    if failures:
        print(f"Failed ({len(failures)}):")
        for src, msg in failures[:20]:
            print(f"  {src}: {msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
