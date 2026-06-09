#!/usr/bin/env python3
"""
background_subtraction_standalone.py

Standalone background-subtraction + validation script for lightsheet microscopy movies.

What it does:
1. Load an AVI or multi-page TIFF movie as (T, H, W)
2. Estimate smooth background using Gaussian or temporal-percentile method
3. Subtract background with adjustable strength
4. Save the background-subtracted movie
5. Automatically generate validation plots and CSV/TXT metrics comparing BEFORE vs AFTER

Typical use:
python src/lightsheet_pipeline/backgroundsubtraction.py \
  --input tests/result/pos3/registered_pos3.avi \
  --output tests/result/bgsv1/registered_pos3_bgsub.avi \
  --bg gaussian \
  --bg-sigma 120 \
  --bg-downsample 8 \
  --bg-alpha 0.25 \
  --bg-offset 0
  --mask-percentile 80 \
  --mask-min-area 5000 \
  --bg-exclusion-radius 30 \
  --save-bg

Good background subtraction should usually show:
- background_mean decreases
- background_std_noise decreases
- background_nonzero_percent decreases
- CNR increases
- foreground_mean / foreground_p95 should not collapse too much
"""

import argparse
from pathlib import Path
import csv
import cv2
import numpy as np


# -----------------------------
# IO utilities
# -----------------------------

def load_movie(path: str):
    """Load AVI or TIFF into float32 array with shape (T, H, W)."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext in [".tif", ".tiff"]:
        import tifffile
        arr = tifffile.imread(str(p)).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim == 4 and arr.shape[-1] in [1, 3, 4]:
            arr = arr[..., 0]
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 3:
            raise ValueError(f"Expected TIFF shape (T,H,W), got {arr.shape}")
        return arr, 1.0

    if ext == ".avi":
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
        frames = []
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
        return np.stack(frames, axis=0), fps

    raise ValueError("Input must be .avi, .tif, or .tiff")


def save_movie_no_stretch(frames: np.ndarray, path: str, fps: float, clip_percentile: float = 99.9):
    """
    Save movie without per-frame min-max stretching.
    This avoids making weak residual background look artificially bright.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    if ext in [".tif", ".tiff"]:
        import tifffile
        tifffile.imwrite(str(p), frames.astype(np.float32), imagej=True)
        print(f"Saved TIFF: {p}")
        return

    if ext != ".avi":
        raise ValueError("Output must be .avi, .tif, or .tiff")

    t, h, w = frames.shape
    max_value = float(np.nanmax(frames))
    if max_value <= 255:
        u8 = np.clip(frames, 0, 255).astype(np.uint8)
    else:
        positive = frames[frames > 0]
        hi = np.percentile(positive, clip_percentile) if positive.size else 1.0
        u8 = (np.clip(frames, 0, hi) / (hi + 1e-8) * 255).astype(np.uint8)

    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h), isColor=False)
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {path}")
    for frame in u8:
        writer.write(frame)
    writer.release()
    print(f"Saved AVI: {p}")


# -----------------------------
# Background subtraction
# -----------------------------

def gaussian_background(frame: np.ndarray, sigma: float, downsample: int):
    """
    Estimate slowly varying background by downsampling, applying large Gaussian blur,
    then upsampling back. This is faster than blurring the full-resolution frame directly.
    """
    h, w = frame.shape
    downsample = max(1, int(downsample))
    small_w = max(16, w // downsample)
    small_h = max(16, h // downsample)
    small = cv2.resize(frame.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_sigma = max(1.0, float(sigma) / downsample)
    small_bg = cv2.GaussianBlur(small, (0, 0), small_sigma)
    bg = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg.astype(np.float32)


def temporal_background(frames: np.ndarray, percentile: float):
    """
    Estimate background from low temporal percentile.
    Warning: if your sample stays fixed after registration, temporal background may remove real signal.
    """
    return np.percentile(frames, percentile, axis=0).astype(np.float32)


def subtract_background(frames: np.ndarray, args):
    x = frames.astype(np.float32)
    bg_stack = None

    if args.bg == "none":
        corrected = x.copy()

    elif args.bg == "gaussian":
        corrected = np.empty_like(x, dtype=np.float32)
        bg_stack = np.empty_like(x, dtype=np.float32) if args.save_bg else None
        for i, frame in enumerate(x):
            if i % max(1, len(x) // 5) == 0:
                print(f"Estimating Gaussian background frame {i}/{len(x)-1}")
            bg = gaussian_background(frame, args.bg_sigma, args.bg_downsample)
            corrected[i] = frame - float(args.bg_alpha) * bg
            if bg_stack is not None:
                bg_stack[i] = bg

    elif args.bg == "temporal":
        bg = temporal_background(x, args.temporal_percentile)
        corrected = x - float(args.bg_alpha) * bg
        bg_stack = np.repeat(bg[None, :, :], len(x), axis=0) if args.save_bg else None
        print(f"Temporal background p{args.temporal_percentile}: range=({bg.min():.1f}, {bg.max():.1f})")

    else:
        raise ValueError("--bg must be none, gaussian, or temporal")

    corrected = corrected - float(args.bg_offset)
    corrected[corrected < 0] = 0
    return corrected.astype(np.float32), bg_stack


# -----------------------------
# Validation utilities
# -----------------------------

def robust_display(im: np.ndarray):
    """Normalize image for visualization only."""
    im = im.astype(np.float32)
    pos = im[im > 0]
    if pos.size == 0:
        return np.zeros_like(im, dtype=np.float32)
    hi = np.percentile(pos, 99.5)
    return np.clip(im / (hi + 1e-8), 0, 1)


def disk_kernel(radius: int):
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def make_foreground_mask(movie: np.ndarray, percentile: float = 99.0, min_area: int = 500):
    """
    Build a stable foreground mask from the BEFORE movie.
    Validation uses the same mask for before and after so metrics are comparable.
    """
    projection = np.percentile(movie, 95, axis=0).astype(np.float32)
    pos = projection[projection > 0]
    if pos.size < 100:
        thr = float(np.percentile(projection, percentile))
    else:
        thr = float(np.percentile(pos, percentile))

    raw_mask = (projection > thr).astype(np.uint8) * 255
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, disk_kernel(3))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, disk_kernel(1))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    clean = np.zeros_like(raw_mask)
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lab] = 255

    if clean.sum() == 0:
        # Fallback: use a less strict threshold
        thr = float(np.percentile(pos, 95)) if pos.size else float(np.percentile(projection, 95))
        clean = (projection > thr).astype(np.uint8) * 255

    return clean.astype(bool)


def summarize_one_movie(movie: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray, threshold_abs: float):
    frame_rows = []
    for i, frame in enumerate(movie):
        fg = frame[fg_mask]
        bg = frame[bg_mask]
        if fg.size == 0 or bg.size == 0:
            raise ValueError("Foreground/background masks are empty. Try changing --mask-percentile.")

        bg_mean = float(np.mean(bg))
        bg_std = float(np.std(bg))
        fg_mean = float(np.mean(fg))
        fg_p95 = float(np.percentile(fg, 95))
        fg_p99 = float(np.percentile(fg, 99))
        cnr = float((fg_mean - bg_mean) / (bg_std + 1e-8))
        sbr = float((fg_mean + 1e-8) / (bg_mean + 1e-8))
        bg_nonzero_pct = float(np.mean(bg > threshold_abs) * 100)

        # Sharpness inside foreground: useful to detect over-smoothing / lost structure.
        lap = cv2.Laplacian(frame.astype(np.float32), cv2.CV_32F)
        sharpness = float(np.var(lap[fg_mask]))

        frame_rows.append({
            "frame": i,
            "background_mean": bg_mean,
            "background_std_noise": bg_std,
            "background_nonzero_percent": bg_nonzero_pct,
            "foreground_mean": fg_mean,
            "foreground_p95": fg_p95,
            "foreground_p99": fg_p99,
            "CNR": cnr,
            "SBR": sbr,
            "sharpness_laplacian_var_fg": sharpness,
        })

    summary = {}
    keys = [k for k in frame_rows[0].keys() if k != "frame"]
    for k in keys:
        summary[k] = float(np.median([r[k] for r in frame_rows]))
    return summary, frame_rows


def write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_before_after(before: np.ndarray, after: np.ndarray, outdir: Path, name: str, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    n = min(len(before), len(after))
    before = before[:n]
    after = after[:n]

    fg_mask = make_foreground_mask(before, percentile=args.mask_percentile, min_area=args.mask_min_area)
    # Dilate foreground before defining background so edge pixels are not counted as background.
    fg_dilated = cv2.dilate(fg_mask.astype(np.uint8) * 255, disk_kernel(args.bg_exclusion_radius)) > 0
    bg_mask = ~fg_dilated

    before_summary, before_rows = summarize_one_movie(before, fg_mask, bg_mask, args.threshold_abs)
    after_summary, after_rows = summarize_one_movie(after, fg_mask, bg_mask, args.threshold_abs)

    # Save frame-level metrics.
    frame_rows = []
    for rb, ra in zip(before_rows, after_rows):
        row = {"frame": rb["frame"]}
        for k, v in rb.items():
            if k != "frame":
                row[f"before_{k}"] = v
        for k, v in ra.items():
            if k != "frame":
                row[f"after_{k}"] = v
        frame_rows.append(row)
    write_csv(outdir / f"{name}_frame_metrics.csv", frame_rows)

    summary_rows = []
    for metric in before_summary:
        b = before_summary[metric]
        a = after_summary[metric]
        change = a - b
        pct = (change / (abs(b) + 1e-8)) * 100
        summary_rows.append({
            "metric": metric,
            "before_median": b,
            "after_median": a,
            "absolute_change": change,
            "percent_change": pct,
        })
    write_csv(outdir / f"{name}_summary_metrics.csv", summary_rows)

    # Human-readable report.
    with open(outdir / f"{name}_validation_report.txt", "w") as f:
        f.write(f"Validation report: {name}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Interpretation for background subtraction:\n")
        f.write("GOOD signs: background_mean down, background_std_noise down, background_nonzero_percent down, CNR up.\n")
        f.write("BAD signs: foreground_mean/p95 collapse strongly, CNR down, or residual image shows organoid structure.\n\n")
        for row in summary_rows:
            f.write(
                f"{row['metric']}: before={row['before_median']:.4f}, "
                f"after={row['after_median']:.4f}, "
                f"change={row['absolute_change']:.4f}, "
                f"pct={row['percent_change']:.2f}%\n"
            )

    # Fixed masks visualization.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mid = n // 2
    axes[0].imshow(robust_display(before[mid]), cmap="gray")
    axes[0].set_title("Before middle frame")
    axes[1].imshow(fg_mask, cmap="gray")
    axes[1].set_title("Foreground mask used for metrics")
    axes[2].imshow(bg_mask, cmap="gray")
    axes[2].set_title("Background mask used for metrics")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(outdir / f"{name}_fixed_masks.png", dpi=140)
    plt.close(fig)

    # Before / after panel.
    idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    fig, axes = plt.subplots(len(idxs), 3, figsize=(14, 4 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    for r, idx in enumerate(idxs):
        residual = before[idx] - after[idx]
        panels = [
            (before[idx], f"Before frame {idx}"),
            (after[idx], f"After frame {idx}"),
            (residual, "Removed signal: before - after"),
        ]
        for c, (im, title) in enumerate(panels):
            axes[r, c].imshow(robust_display(im), cmap="gray")
            axes[r, c].set_title(title)
            axes[r, c].axis("off")
    plt.tight_layout()
    fig.savefig(outdir / f"{name}_before_after_residual_panel.png", dpi=140)
    plt.close(fig)

    # Summary bar plot for most useful metrics.
    metrics_to_plot = [
        "background_mean",
        "background_std_noise",
        "background_nonzero_percent",
        "foreground_mean",
        "foreground_p95",
        "CNR",
        "SBR",
        "sharpness_laplacian_var_fg",
    ]
    x = np.arange(len(metrics_to_plot))
    width = 0.38
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width / 2, [before_summary[m] for m in metrics_to_plot], width, label="Before")
    ax.bar(x + width / 2, [after_summary[m] for m in metrics_to_plot], width, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=35, ha="right")
    ax.set_title(f"Summary metrics: {name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(outdir / f"{name}_summary_bar_metrics.png", dpi=140)
    plt.close(fig)

    # Frame-wise trend plots.
    trend_metrics = ["background_mean", "background_std_noise", "CNR", "foreground_p95"]
    for metric in trend_metrics:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot([r["frame"] for r in before_rows], [r[metric] for r in before_rows], label="Before")
        ax.plot([r["frame"] for r in after_rows], [r[metric] for r in after_rows], label="After")
        ax.set_title(f"Frame-wise {metric}: {name}")
        ax.set_xlabel("Frame")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(outdir / f"{name}_timeseries_{metric}.png", dpi=140)
        plt.close(fig)

    # Histograms for background and foreground pixels.
    sample_frames = idxs
    before_fg = np.concatenate([before[i][fg_mask].ravel() for i in sample_frames])
    after_fg = np.concatenate([after[i][fg_mask].ravel() for i in sample_frames])
    before_bg = np.concatenate([before[i][bg_mask].ravel() for i in sample_frames])
    after_bg = np.concatenate([after[i][bg_mask].ravel() for i in sample_frames])

    for region_name, bvals, avals in [
        ("background", before_bg, after_bg),
        ("foreground", before_fg, after_fg),
    ]:
        hi = np.percentile(np.concatenate([bvals, avals]), 99.5)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(bvals, bins=100, range=(0, hi), alpha=0.5, label="Before")
        ax.hist(avals, bins=100, range=(0, hi), alpha=0.5, label="After")
        ax.set_title(f"{region_name.capitalize()} intensity histogram: {name}")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Pixel count")
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / f"{name}_histogram_{region_name}.png", dpi=140)
        plt.close(fig)

    print("\nValidation summary:")
    for row in summary_rows:
        print(f"  {row['metric']}: before={row['before_median']:.4f}, after={row['after_median']:.4f}, change={row['percent_change']:.2f}%")
    print(f"\nSaved validation outputs to: {outdir}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone background subtraction + validation for lightsheet movies")
    parser.add_argument("--input", required=True, help="Input registered movie: .avi/.tif/.tiff")
    parser.add_argument("--output", required=True, help="Output background-subtracted movie: .avi/.tif/.tiff")
    parser.add_argument("--outdir", default=None, help="Validation output folder. Default: output parent / validation_background")

    # Background subtraction parameters.
    parser.add_argument("--bg", default="gaussian", choices=["none", "gaussian", "temporal"])
    parser.add_argument("--bg-sigma", type=float, default=120.0)
    parser.add_argument("--bg-downsample", type=int, default=8)
    parser.add_argument("--bg-alpha", type=float, default=0.45, help="Subtract strength. Lower = less aggressive; try 0.25-0.60")
    parser.add_argument("--bg-offset", type=float, default=0.0, help="Constant subtracted after background subtraction")
    parser.add_argument("--temporal-percentile", type=float, default=2.0)
    parser.add_argument("--save-bg", action="store_true", help="Save estimated background stack as TIFF")

    # Saving / validation.
    parser.add_argument("--clip-percentile", type=float, default=99.9)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--name", default="background_subtraction")
    parser.add_argument("--threshold-abs", type=float, default=3.0, help="Threshold for counting nonzero background noise")
    parser.add_argument("--mask-percentile", type=float, default=99.0, help="Percentile for foreground mask from BEFORE movie")
    parser.add_argument("--mask-min-area", type=int, default=500)
    parser.add_argument("--bg-exclusion-radius", type=int, default=15, help="Dilate foreground this much before measuring background")

    args = parser.parse_args()

    output_path = Path(args.output)
    outdir = Path(args.outdir) if args.outdir else output_path.parent / "validation_background"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before, fps = load_movie(args.input)
    print(f"Loaded input: {args.input}")
    print(f"Shape={before.shape}, fps={fps:.2f}, range=({before.min():.1f}, {before.max():.1f}), nonzero={(before > 0).mean()*100:.2f}%")

    after, bg_stack = subtract_background(before, args)
    print(f"After background subtraction: range=({after.min():.1f}, {after.max():.1f}), nonzero={(after > 0).mean()*100:.2f}%")

    save_movie_no_stretch(after, str(output_path), fps, args.clip_percentile)

    if args.save_bg and bg_stack is not None:
        save_movie_no_stretch(bg_stack, str(output_path.with_name(output_path.stem + "_estimated_background.tif")), fps, args.clip_percentile)

    if not args.no_validate:
        validate_before_after(before, after, outdir, args.name, args)


if __name__ == "__main__":
    main()