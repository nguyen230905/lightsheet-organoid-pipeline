#!/usr/bin/env python3
"""
background_subtraction.py

Robust background subtraction for light-sheet AVI/TIFF movies.

Main fixes compared with the uploaded script:
1) Do not use per-frame min-max or preprocessed display images for output.
2) Write AVI as 3-channel BGR MJPG for better compatibility.
3) Estimate smooth background from pixels outside the specimen mask, so the
   organoid itself is not blurred into the background and subtracted away.
4) Optionally subtract a residual baseline from background pixels per frame.
5) Warn if the input movie already has a suspicious grey background, which is
   usually inherited from the earlier registration/export problem.

Typical use after registration:
python3 src/lightsheet_pipeline/background_subtraction.py \
  --input result/registered/center_phase/pos3/registered_fixed.avi \
  --output result/bgs/bgsub_fixed.avi \
  --method masked_gaussian \
  --bg-sigma 120 \
  --mask-percentile 70 \
  --mask-dilate 70 \
  --bg-alpha 1 \
  --auto-offset
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MovieInfo:
    fps: float
    source: str


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_movie(path: str | Path) -> tuple[np.ndarray, MovieInfo]:
    """Load AVI or TIFF as float32 array with shape (T, H, W)."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".tif", ".tiff"}:
        import tifffile
        arr = tifffile.imread(str(path))
        if arr.ndim == 2:
            arr = arr[None]
        # Drop color/channel dimension if this is an RGB/RGBA image stack.
        if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            arr = arr[..., 0]
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 3:
            raise ValueError(f"Expected TIFF shape (T,H,W); got {arr.shape}")
        return arr.astype(np.float32), MovieInfo(fps=1.0, source=str(path))

    if ext != ".avi":
        raise ValueError("Input must be .avi, .tif, or .tiff")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 4.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0), MovieInfo(fps=fps, source=str(path))


def to_uint8_for_video(frames: np.ndarray, display_clip: float | None = None) -> np.ndarray:
    """
    Convert to uint8 for AVI writing.

    If data are already in 0..255, just clip. This preserves black background.
    If data exceed 8-bit, use one global percentile scale, never per-frame scale.
    """
    frames = np.asarray(frames, dtype=np.float32)
    finite = np.isfinite(frames)
    if not finite.any():
        raise ValueError("All output pixels are NaN/Inf")
    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

    max_val = float(frames.max())
    min_val = float(frames.min())
    if min_val >= 0 and max_val <= 255 and display_clip is None:
        return np.clip(frames, 0, 255).astype(np.uint8)

    positive = frames[frames > 0]
    if positive.size == 0:
        return np.zeros(frames.shape, dtype=np.uint8)
    hi = np.percentile(positive, display_clip if display_clip is not None else 99.9)
    hi = max(float(hi), 1e-6)
    return (np.clip(frames, 0, hi) / hi * 255).astype(np.uint8)


def save_movie(frames: np.ndarray, path: str | Path, fps: float, display_clip: float | None = None):
    """Save AVI or TIFF. AVI is written as BGR MJPG for viewer compatibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in {".tif", ".tiff"}:
        import tifffile
        tifffile.imwrite(str(path), frames.astype(np.float32), imagej=True, metadata={"axes": "TYX"})
        print(f"Saved TIFF: {path}")
        return

    if ext != ".avi":
        raise ValueError("Output must be .avi, .tif, or .tiff")

    u8 = to_uint8_for_video(frames, display_clip=display_clip)
    t, h, w = u8.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (w, h), isColor=True)
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {path}")
    for frame in u8:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    print(f"Saved AVI: {path}  shape={u8.shape}, range=({u8.min()}, {u8.max()})")


# ---------------------------------------------------------------------------
# Masks and background estimation
# ---------------------------------------------------------------------------

def disk_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def foreground_mask_from_frame(
    frame: np.ndarray,
    mask_percentile: float = 70.0,
    min_area: int = 5000,
    dilate_radius: int = 70,
    max_mask_size: int = 768,
) -> np.ndarray:
    """Detect the specimen/bright object; returns boolean mask.

    The mask is calculated on a downsampled copy for speed. This avoids very
    slow full-resolution morphology with a large dilation kernel.
    """
    f0 = np.asarray(frame, dtype=np.float32)
    h0, w0 = f0.shape
    scale = min(1.0, float(max_mask_size) / float(max(h0, w0)))
    if scale < 1.0:
        f = cv2.resize(f0, (max(16, int(w0 * scale)), max(16, int(h0 * scale))), interpolation=cv2.INTER_AREA)
    else:
        f = f0

    pos = f[f > 0]
    if pos.size < 100:
        return np.zeros(f0.shape, dtype=bool)

    thr = float(np.percentile(pos, mask_percentile))
    mask = (f > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_kernel(max(1, int(round(5 * scale)))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk_kernel(max(1, int(round(2 * scale)))))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    min_area_scaled = max(20, int(round(min_area * scale * scale)))
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area_scaled:
            clean[labels == lab] = 255

    if clean.sum() == 0:
        # Fallback: keep the largest component.
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            clean[labels == largest] = 255
        else:
            clean = mask

    if dilate_radius > 0:
        r = max(1, int(round(dilate_radius * scale)))
        clean = cv2.dilate(clean, disk_kernel(r))

    if scale < 1.0:
        clean = cv2.resize(clean, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return clean.astype(bool)


def stable_foreground_mask(
    frames: np.ndarray,
    mask_percentile: float,
    min_area: int,
    dilate_radius: int,
) -> np.ndarray:
    """Use a temporal projection to make one stable mask for all frames."""
    projection = np.percentile(frames, 95, axis=0).astype(np.float32)
    return foreground_mask_from_frame(
        projection,
        mask_percentile=mask_percentile,
        min_area=min_area,
        dilate_radius=dilate_radius,
    )


def masked_gaussian_background(
    frame: np.ndarray,
    specimen_mask: np.ndarray,
    sigma: float,
    downsample: int,
) -> np.ndarray:
    """
    Smooth background estimated only from non-specimen pixels.

    This is the key improvement over directly blurring the raw frame: the bright
    organoid does not get blurred into the background image and then subtracted
    from itself.
    """
    h, w = frame.shape
    downsample = max(1, int(downsample))
    small_w = max(16, w // downsample)
    small_h = max(16, h // downsample)

    valid = (~specimen_mask).astype(np.float32)
    num = frame.astype(np.float32) * valid

    num_s = cv2.resize(num, (small_w, small_h), interpolation=cv2.INTER_AREA)
    den_s = cv2.resize(valid, (small_w, small_h), interpolation=cv2.INTER_AREA)

    small_sigma = max(1.0, float(sigma) / downsample)
    num_blur = cv2.GaussianBlur(num_s, (0, 0), small_sigma)
    den_blur = cv2.GaussianBlur(den_s, (0, 0), small_sigma)

    bg_small = num_blur / (den_blur + 1e-6)
    bg_pixels = frame[~specimen_mask]
    fallback = float(np.median(bg_pixels)) if bg_pixels.size else 0.0
    bg_small[den_blur < 1e-3] = fallback
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(bg, 0, None).astype(np.float32)


def plain_gaussian_background(frame: np.ndarray, sigma: float, downsample: int) -> np.ndarray:
    h, w = frame.shape
    downsample = max(1, int(downsample))
    small_w = max(16, w // downsample)
    small_h = max(16, h // downsample)
    small = cv2.resize(frame.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_sigma = max(1.0, float(sigma) / downsample)
    bg_small = cv2.GaussianBlur(small, (0, 0), small_sigma)
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def subtract_background(frames: np.ndarray, args) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    x = frames.astype(np.float32)
    fg_mask = stable_foreground_mask(
        x,
        mask_percentile=args.mask_percentile,
        min_area=args.mask_min_area,
        dilate_radius=args.mask_dilate,
    )
    bg_mask = ~fg_mask

    corrected = np.empty_like(x, dtype=np.float32)
    bg_stack = np.empty_like(x, dtype=np.float32) if args.save_bg else None

    if args.method == "none":
        corrected[:] = x
        if bg_stack is not None:
            bg_stack[:] = 0
    elif args.method == "temporal":
        # Only safe when sample moves relative to camera. After registration it may remove real signal.
        bg = np.percentile(x, args.temporal_percentile, axis=0).astype(np.float32)
        corrected = x - float(args.bg_alpha) * bg[None]
        if bg_stack is not None:
            bg_stack[:] = bg[None]
    else:
        for i, frame in enumerate(x):
            if i % max(1, len(x) // 5) == 0:
                print(f"Estimating background frame {i}/{len(x)-1}")
            if args.method == "masked_gaussian":
                bg = masked_gaussian_background(frame, fg_mask, args.bg_sigma, args.bg_downsample)
            elif args.method == "gaussian":
                bg = plain_gaussian_background(frame, args.bg_sigma, args.bg_downsample)
            else:
                raise ValueError("Unknown method")
            corrected[i] = frame - float(args.bg_alpha) * bg
            if bg_stack is not None:
                bg_stack[i] = bg

    # Remove a residual camera/export baseline from background pixels only.
    offsets = []
    if args.auto_offset:
        for i in range(len(corrected)):
            bg_vals = corrected[i][bg_mask]
            offset = float(np.percentile(bg_vals, args.offset_percentile)) if bg_vals.size else 0.0
            # Do not add intensity; only subtract positive residual baseline.
            offset = max(0.0, offset)
            corrected[i] -= offset
            offsets.append(offset)
    else:
        corrected -= float(args.bg_offset)
        offsets = [float(args.bg_offset)] * len(corrected)

    corrected[corrected < 0] = 0
    return corrected.astype(np.float32), bg_stack, fg_mask


# ---------------------------------------------------------------------------
# Validation / diagnostics
# ---------------------------------------------------------------------------

def robust_display(frame: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    f = frame.astype(np.float32)
    pos = f[f > 0]
    if pos.size == 0:
        return np.zeros_like(f)
    hi = max(float(np.percentile(pos, percentile)), 1e-6)
    return np.clip(f / hi, 0, 1)




def preview_image(im: np.ndarray, max_side: int = 900) -> np.ndarray:
    """Downsample large images for fast validation plotting only."""
    im = np.asarray(im)
    h, w = im.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return im
    return cv2.resize(im, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)



def robust_preview(im: np.ndarray, max_side: int = 900, percentile: float = 99.5) -> np.ndarray:
    """Downsample first, then normalize for fast plotting."""
    return robust_display(preview_image(im, max_side=max_side), percentile=percentile)

def quick_stats(frames: np.ndarray) -> dict[str, float]:
    arr = np.asarray(frames, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "nonzero_percent": float((arr > 0).mean() * 100),
    }


def summarize_regions(movie: np.ndarray, fg_mask: np.ndarray) -> dict[str, float]:
    bg_mask = ~fg_mask
    rows = []
    for frame in movie:
        fg = frame[fg_mask]
        bg = frame[bg_mask]
        bg_mean = float(bg.mean()) if bg.size else 0.0
        bg_std = float(bg.std()) if bg.size else 0.0
        fg_mean = float(fg.mean()) if fg.size else 0.0
        fg_p95 = float(np.percentile(fg, 95)) if fg.size else 0.0
        rows.append({
            "background_mean": bg_mean,
            "background_std": bg_std,
            "foreground_mean": fg_mean,
            "foreground_p95": fg_p95,
            "CNR": float((fg_mean - bg_mean) / (bg_std + 1e-8)),
        })
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def _u8_preview(im: np.ndarray, max_side: int = 700) -> np.ndarray:
    """Fast uint8 preview for contact sheets."""
    im_small = preview_image(im, max_side=max_side).astype(np.float32)
    pos = im_small[im_small > 0]
    if pos.size == 0:
        return np.zeros(im_small.shape, dtype=np.uint8)
    hi = max(float(np.percentile(pos, 99.5)), 1e-6)
    return (np.clip(im_small / hi, 0, 1) * 255).astype(np.uint8)


def _label_panel(gray: np.ndarray, label: str) -> np.ndarray:
    """Convert grayscale panel to BGR and add a readable label."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pad = 34
    out = np.zeros((bgr.shape[0] + pad, bgr.shape[1], 3), dtype=np.uint8)
    out[pad:] = bgr
    cv2.putText(out, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_validation(before: np.ndarray, after: np.ndarray, fg_mask: np.ndarray, outdir: Path, name: str):
    """Save lightweight validation files without matplotlib."""
    outdir.mkdir(parents=True, exist_ok=True)
    n = min(len(before), len(after))
    before = before[:n]
    after = after[:n]
    idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

    rows = []
    for idx in idxs:
        removed = np.clip(before[idx] - after[idx], 0, None)
        panels = [
            _label_panel(_u8_preview(before[idx]), f"Before frame {idx}"),
            _label_panel(_u8_preview(after[idx]), f"After frame {idx}"),
            _label_panel(_u8_preview(removed), "Removed signal"),
        ]
        # Pad panels to same height before concatenation.
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            if p.shape[0] < max_h:
                pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)
        rows.append(np.hstack(padded))
    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded_rows.append(r)
    contact = np.vstack(padded_rows)
    cv2.imwrite(str(outdir / f"{name}_before_after_removed.png"), contact)

    # Mask check panel.
    mid = n // 2
    mask_preview = (preview_image(fg_mask.astype(np.float32), max_side=700) * 255).astype(np.uint8)
    bg_preview = (preview_image((~fg_mask).astype(np.float32), max_side=700) * 255).astype(np.uint8)
    mask_panels = [
        _label_panel(_u8_preview(before[mid]), "Before middle frame"),
        _label_panel(mask_preview, "Specimen mask excluded"),
        _label_panel(bg_preview, "Background pixels used"),
    ]
    max_h = max(p.shape[0] for p in mask_panels)
    mask_panels = [np.vstack([p, np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)]) if p.shape[0] < max_h else p for p in mask_panels]
    cv2.imwrite(str(outdir / f"{name}_mask_check.png"), np.hstack(mask_panels))

    before_stats = quick_stats(before)
    after_stats = quick_stats(after)
    before_regions = summarize_regions(before, fg_mask)
    after_regions = summarize_regions(after, fg_mask)

    with open(outdir / f"{name}_summary.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "before", "after", "change"])
        for k in before_stats:
            writer.writerow([k, before_stats[k], after_stats[k], after_stats[k] - before_stats[k]])
        for k in before_regions:
            writer.writerow([k, before_regions[k], after_regions[k], after_regions[k] - before_regions[k]])

    with open(outdir / f"{name}_report.txt", "w") as fh:
        fh.write("Background subtraction validation report\n")
        fh.write("========================================\n\n")
        fh.write(f"Frames: {n}\n")
        fh.write(f"Before quick stats: {before_stats}\n")
        fh.write(f"After quick stats:  {after_stats}\n")
        fh.write(f"Before region stats: {before_regions}\n")
        fh.write(f"After region stats:  {after_regions}\n\n")
        if before_stats["median"] > 50:
            fh.write("WARNING: input movie median intensity is high. This usually means the input already has a grey-background export artifact. Use the fixed registered AVI as input if possible.\n")

    print(f"Saved validation to {outdir}")


def warn_about_input(frames: np.ndarray):
    stats = quick_stats(frames)
    print("Input stats:", stats)
    if 80 <= stats["median"] <= 170 and stats["nonzero_percent"] > 50:
        print("WARNING: input has a large grey baseline. This is usually inherited from the registration/export artifact, not true microscopy signal.")
        print("Recommendation: rerun background subtraction on the fixed registered AVI, not on the old grey registered.avi.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Robust background subtraction for lightsheet AVI/TIFF movies")
    parser.add_argument("--input", required=True, help="Input movie: .avi/.tif/.tiff")
    parser.add_argument("--output", required=True, help="Output movie: .avi/.tif/.tiff")
    parser.add_argument("--outdir", default=None, help="Validation folder; default: output parent / validation_background")

    parser.add_argument("--method", default="masked_gaussian", choices=["masked_gaussian", "gaussian", "temporal", "none"])
    parser.add_argument("--bg-sigma", type=float, default=120.0, help="Scale of smooth background in pixels")
    parser.add_argument("--bg-downsample", type=int, default=8, help="Speed-up factor for smooth background estimation")
    parser.add_argument("--bg-alpha", type=float, default=1.0, help="Background subtraction strength")
    parser.add_argument("--temporal-percentile", type=float, default=2.0)

    parser.add_argument("--mask-percentile", type=float, default=70.0, help="Threshold among positive pixels for specimen mask")
    parser.add_argument("--mask-min-area", type=int, default=5000)
    parser.add_argument("--mask-dilate", type=int, default=70, help="Dilate specimen mask before background estimation")

    parser.add_argument("--auto-offset", action="store_true", help="Subtract residual median baseline from background pixels per frame")
    parser.add_argument("--offset-percentile", type=float, default=50.0)
    parser.add_argument("--bg-offset", type=float, default=0.0, help="Manual offset if --auto-offset is not used")

    parser.add_argument("--display-clip", type=float, default=None, help="Global percentile scaling for AVI only. Default keeps 0..255 raw values.")
    parser.add_argument("--save-bg", action="store_true", help="Save estimated background as TIFF")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--name", default="background_subtraction_fixed")
    args = parser.parse_args()

    output = Path(args.output)
    outdir = Path(args.outdir) if args.outdir else output.parent / "validation_background"

    before, info = load_movie(args.input)
    print(f"Loaded {info.source}: shape={before.shape}, fps={info.fps:.3f}")
    warn_about_input(before)

    after, bg_stack, fg_mask = subtract_background(before, args)
    print("After stats:", quick_stats(after))

    save_movie(after, output, fps=info.fps, display_clip=args.display_clip)
    if args.save_bg and bg_stack is not None:
        save_movie(bg_stack, output.with_name(output.stem + "_estimated_background.tif"), fps=info.fps)

    if not args.no_validate:
        save_validation(before, after, fg_mask, outdir, args.name)


if __name__ == "__main__":
    main()