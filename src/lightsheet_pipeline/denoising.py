#!/usr/bin/env python3
"""
denoising_fixed.py â€” robust denoising for lightsheet AVI/TIFF movies.

Main fixes compared with the uploaded script:
1) Denoising is separated from background subtraction. If the input is already BGS,
   the default is --bg none so the organoid is not background-subtracted twice.
2) The default output is the denoised grayscale intensity image, not a hard mask.
   Mask/connected-component cleanup is optional.
3) AVI is written as 3-channel BGR MJPG, even for grayscale data. This avoids the
   grey 128/129 baseline artifact seen when writing single-channel MJPG AVI.
4) Intensities are clipped to 0-255 without per-frame min-max stretching.
5) Validation reports check foreground preservation, sharpness retention, noise
   reduction, and suspicious grey-baseline artifacts.

Recommended use after your registration + background-subtraction step:
python3 src/lightsheet_pipeline/denoising.py \
  --input result/bgs/0.45/bgsub_fixed.avi \
  --output result/denoised/ \
  --method median_gaussian \
  --median-ksize 3 \
  --sigma 0.6 \
  --bg none
Median only:
python3 src/lightsheet_pipeline/denoising.py \
  --input result/bgs/0.45/bgsub_fixed.avi \
  --output result/denoised/ \
  --method median \
  --median-ksize 3 \
  --bg none
Small Gaussian only:
python3 src/lightsheet_pipeline/denoising.py \
  --input result/bgs/0.45/bgsub_fixed.avi \
  --output result/denoised/ \
  --method median_gaussian \
  --median-ksize 3 \
  --sigma 0.3  
No Gaussian,  
python3 src/lightsheet_pipeline/denoising.py \
  --input result/bgs/0.45/bgsub_fixed.avi \
  --output result/denoised/ \
  --method median \
  --median-ksize 3 

If you still see isolated speckles and you only need a cleaned visualization, add:
  --cleanup mask --threshold-abs 3 --min-area 300 --close-radius 1

For quantitative intensity analysis, start with --cleanup none.
"""

import argparse
import csv
from pathlib import Path
import cv2
import numpy as np


def load_movie(path: str):
    """Load AVI/TIFF as float32 stack with shape (T,H,W)."""
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


def to_uint8_no_stretch(frames: np.ndarray, clip_percentile: float = 99.9):
    """Convert to uint8 without frame-wise stretching."""
    frames = np.asarray(frames, dtype=np.float32)
    max_value = float(np.nanmax(frames)) if frames.size else 0.0
    if max_value <= 255:
        return np.clip(frames, 0, 255).astype(np.uint8)
    positive = frames[frames > 0]
    hi = np.percentile(positive, clip_percentile) if positive.size else 1.0
    return (np.clip(frames, 0, hi) / (hi + 1e-8) * 255).astype(np.uint8)


def save_movie(frames: np.ndarray, path: str, fps: float, clip_percentile: float = 99.9):
    """Save AVI/TIFF. AVI is always BGR MJPG to avoid grayscale MJPG artifacts."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    if ext in [".tif", ".tiff"]:
        import tifffile
        tifffile.imwrite(str(p), frames.astype(np.float32), imagej=True, metadata={"axes": "TYX"})
        print(f"Saved TIFF: {p}")
        return

    if ext != ".avi":
        raise ValueError("Output must be .avi, .tif, or .tiff")

    u8 = to_uint8_no_stretch(frames, clip_percentile=clip_percentile)
    t, h, w = u8.shape
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (w, h), isColor=True)
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {path}")
    for frame in u8:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    print(f"Saved AVI: {p}")


def disk_kernel(radius: int):
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def gaussian_background(frame: np.ndarray, sigma: float, downsample: int):
    """Optional smooth background estimate; disabled by default for already-BGS input."""
    h, w = frame.shape
    downsample = max(1, int(downsample))
    small_w = max(16, w // downsample)
    small_h = max(16, h // downsample)
    small = cv2.resize(frame.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_sigma = max(1.0, float(sigma) / downsample)
    small_bg = cv2.GaussianBlur(small, (0, 0), small_sigma)
    return cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def subtract_background_optional(frames: np.ndarray, args):
    x = frames.astype(np.float32)
    if args.bg == "none":
        return x.copy(), None
    if args.bg == "gaussian":
        out = np.empty_like(x)
        bg_stack = np.empty_like(x) if args.save_bg else None
        for i, frame in enumerate(x):
            if i % max(1, len(x) // 5) == 0:
                print(f"Optional background subtraction frame {i}/{len(x)-1}")
            bg = gaussian_background(frame, args.bg_sigma, args.bg_downsample)
            out[i] = frame - float(args.bg_alpha) * bg
            if bg_stack is not None:
                bg_stack[i] = bg
        out -= float(args.bg_offset)
        out[out < 0] = 0
        return out.astype(np.float32), bg_stack
    raise ValueError("--bg must be none or gaussian")


def denoise_frame(frame: np.ndarray, args):
    """Mild frame-wise denoising. Do not binarize or stretch intensities."""
    f = np.clip(frame, 0, 255).astype(np.uint8)

    if args.method in ["median", "median_gaussian"] and args.median_ksize > 1:
        k = int(args.median_ksize)
        if k % 2 == 0:
            k += 1
        f = cv2.medianBlur(f, k)

    if args.method == "none":
        out = f.astype(np.float32)
    elif args.method == "median":
        out = f.astype(np.float32)
    elif args.method == "gaussian":
        out = cv2.GaussianBlur(f.astype(np.float32), (0, 0), float(args.sigma))
    elif args.method == "median_gaussian":
        out = cv2.GaussianBlur(f.astype(np.float32), (0, 0), float(args.sigma))
    elif args.method == "nlm":
        out = cv2.fastNlMeansDenoising(f, None, h=float(args.h), templateWindowSize=7, searchWindowSize=21).astype(np.float32)
    elif args.method == "bilateral":
        out = cv2.bilateralFilter(f, d=5, sigmaColor=float(args.bilateral_sigma_color), sigmaSpace=float(args.bilateral_sigma_space)).astype(np.float32)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    if args.intensity_floor > 0:
        out[out < float(args.intensity_floor)] = 0
    return out.astype(np.float32)


def denoise_stack(frames: np.ndarray, args):
    out = np.empty_like(frames, dtype=np.float32)
    for i, frame in enumerate(frames):
        if i % max(1, len(frames) // 5) == 0:
            print(f"Denoising frame {i}/{len(frames)-1}")
        out[i] = denoise_frame(frame, args)
    return out


def compute_threshold(frame: np.ndarray, threshold_abs: float, otsu_factor: float):
    vals = frame[frame > 0]
    if vals.size < 100:
        return float(threshold_abs)
    vals_u8 = np.clip(vals, 0, 255).astype(np.uint8)
    otsu_t, _ = cv2.threshold(vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(threshold_abs), float(otsu_t) * float(otsu_factor))


def clean_mask(mask: np.ndarray, args):
    mask = (mask > 0).astype(np.uint8) * 255
    if args.open_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk_kernel(args.open_radius))
    if args.close_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_kernel(args.close_radius))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= int(args.min_area):
            clean[labels == lab] = 255

    if args.keep_largest and np.any(clean):
        n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        if n2 > 1:
            largest = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
            clean = ((labels2 == largest).astype(np.uint8) * 255)
    return clean


def make_masks(frames: np.ndarray, args):
    masks = np.empty(frames.shape, dtype=np.uint8)
    thresholds = []
    for i, frame in enumerate(frames):
        smooth = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), args.mask_blur_sigma) if args.mask_blur_sigma > 0 else frame
        thr = compute_threshold(smooth, args.threshold_abs, args.otsu_factor)
        masks[i] = clean_mask(smooth > thr, args)
        thresholds.append(thr)
    print(f"Mask threshold: min={min(thresholds):.2f}, median={np.median(thresholds):.2f}, max={max(thresholds):.2f}")
    return masks


def stable_foreground_mask(movie: np.ndarray, percentile: float = 85.0, min_area: int = 5000):
    """Projection-based object mask used only for validation metrics."""
    proj = np.percentile(movie, 95, axis=0).astype(np.float32)
    pos = proj[proj > 0]
    if pos.size:
        thr = np.percentile(pos, percentile)
    else:
        thr = np.percentile(proj, 99)
    raw = (proj > thr).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, disk_kernel(5))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    clean = np.zeros_like(raw)
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lab] = 255
    if clean.sum() == 0:
        clean = raw
    return clean.astype(bool)


def stack_metrics(name: str, stack: np.ndarray, fg2d: np.ndarray, bg2d: np.ndarray):
    fg = stack[:, fg2d]
    bg = stack[:, bg2d]
    if fg.size == 0:
        fg = stack.reshape(-1)
    if bg.size == 0:
        bg = stack.reshape(-1)

    # Sharpness calculated in foreground only.
    sharp_vals = []
    for frame in stack:
        lap = cv2.Laplacian(np.clip(frame, 0, 255).astype(np.uint8), cv2.CV_32F)
        if np.any(fg2d):
            sharp_vals.append(float(np.var(lap[fg2d])))
    sharp = float(np.mean(sharp_vals)) if sharp_vals else 0.0

    return {
        "stage": name,
        "min": float(np.min(stack)),
        "median_all": float(np.median(stack)),
        "p95_all": float(np.percentile(stack, 95)),
        "p99_all": float(np.percentile(stack, 99)),
        "max": float(np.max(stack)),
        "nonzero_percent": float((stack > 0).mean() * 100.0),
        "background_mean": float(np.mean(bg)),
        "background_std_noise": float(np.std(bg)),
        "background_p95": float(np.percentile(bg, 95)),
        "foreground_mean": float(np.mean(fg)),
        "foreground_p95": float(np.percentile(fg, 95)),
        "foreground_p99": float(np.percentile(fg, 99)),
        "cnr": float((np.mean(fg) - np.mean(bg)) / (np.std(bg) + 1e-8)),
        "sharpness_laplacian_var_fg": sharp,
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def show_norm(im: np.ndarray):
    im = np.asarray(im, dtype=np.float32)
    pos = im[im > 0]
    if pos.size == 0:
        return np.zeros_like(im)
    hi = np.percentile(pos, 99.5)
    return np.clip(im / (hi + 1e-8), 0, 1)


def save_validation(raw, processed_input, denoised, final, masks, outdir: Path, args, bad_movie_path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    fg = stable_foreground_mask(processed_input, percentile=args.validation_mask_percentile, min_area=args.validation_mask_min_area)
    bg = cv2.dilate(fg.astype(np.uint8) * 255, disk_kernel(args.validation_bg_exclusion_radius)) == 0

    rows = [
        stack_metrics("input", raw, fg, bg),
        stack_metrics("processed_input_after_optional_bg", processed_input, fg, bg),
        stack_metrics("denoised", denoised, fg, bg),
        stack_metrics("final_output", final, fg, bg),
    ]

    bad = None
    if bad_movie_path:
        try:
            bad, _ = load_movie(bad_movie_path)
            if bad.shape == raw.shape:
                rows.append(stack_metrics("uploaded_bad_output", bad, fg, bg))
        except Exception as exc:
            print(f"Could not load bad output for validation: {exc}")

    write_csv(outdir / "denoising_validation_metrics.csv", rows)

    # Visual panel.
    n = len(raw)
    idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    cols = [(raw, "input BGS"), (denoised, "denoised"), (final, "final output")]
    if bad is not None and bad.shape == raw.shape:
        cols.insert(1, (bad, "uploaded bad output"))
    fig, axes = plt.subplots(len(idxs), len(cols), figsize=(4.5 * len(cols), 4 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    if len(cols) == 1:
        axes = axes[:, None]
    for r, idx in enumerate(idxs):
        for c, (stack, title) in enumerate(cols):
            axes[r, c].imshow(show_norm(stack[idx]), cmap="gray")
            axes[r, c].set_title(f"{title}\nframe {idx}")
            axes[r, c].axis("off")
    plt.tight_layout()
    fig.savefig(outdir / "denoising_before_bad_fixed_panel.png", dpi=140)
    plt.close(fig)

    # Mask check.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    mid = n // 2
    axes[0].imshow(show_norm(processed_input[mid]), cmap="gray")
    axes[0].set_title("Metric image: middle frame")
    axes[1].imshow(fg, cmap="gray")
    axes[1].set_title("Foreground mask for metrics")
    axes[2].imshow(bg, cmap="gray")
    axes[2].set_title("Background mask for metrics")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(outdir / "denoising_metric_masks.png", dpi=140)
    plt.close(fig)

    # Summary plots.
    metrics = ["nonzero_percent", "background_std_noise", "foreground_mean", "foreground_p95", "cnr", "sharpness_laplacian_var_fg"]
    stages = [r["stage"] for r in rows]
    x = np.arange(len(metrics))
    width = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(16, 6))
    base = rows[0]
    for j, row in enumerate(rows):
        vals = []
        for m in metrics:
            vals.append(row[m] / (abs(base[m]) + 1e-8))
        ax.bar(x + (j - (len(rows)-1)/2) * width, vals, width, label=row["stage"])
    ax.axhline(1, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Relative to input")
    ax.set_title("Denoising validation metrics")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "denoising_validation_relative_metrics.png", dpi=140)
    plt.close(fig)

    # Report.
    def pct(after, before):
        return 100.0 * (after - before) / (abs(before) + 1e-8)

    inp = rows[0]
    den = next(r for r in rows if r["stage"] == "denoised")
    final_row = next(r for r in rows if r["stage"] == "final_output")
    suspicious = []
    for row in rows:
        if row["median_all"] > 20 and row["nonzero_percent"] > 50:
            suspicious.append(row["stage"])

    lines = []
    lines.append("DENOISING VALIDATION REPORT")
    lines.append("===========================")
    lines.append("")
    lines.append("Good denoising should reduce background/noise while keeping foreground intensity and sharpness.")
    lines.append("It should not turn the black background into a grey 128/129 baseline.")
    lines.append("")
    lines.append("Main changes from input -> denoised:")
    lines.append(f"- background std/noise: {pct(den['background_std_noise'], inp['background_std_noise']):.2f}%")
    lines.append(f"- foreground mean: {pct(den['foreground_mean'], inp['foreground_mean']):.2f}%")
    lines.append(f"- foreground p95: {pct(den['foreground_p95'], inp['foreground_p95']):.2f}%")
    lines.append(f"- sharpness: {pct(den['sharpness_laplacian_var_fg'], inp['sharpness_laplacian_var_fg']):.2f}%")
    lines.append(f"- CNR: {pct(den['cnr'], inp['cnr']):.2f}%")
    lines.append("")
    if suspicious:
        lines.append("Suspicious grey-baseline outputs detected: " + ", ".join(suspicious))
        lines.append("This usually indicates an AVI-writing/codec artifact or a display-scaled movie being reused as data.")
        lines.append("")
    lines.append("Raw metrics:")
    for row in rows:
        lines.append(f"\n[{row['stage']}]")
        for k, v in row.items():
            if k != "stage":
                lines.append(f"{k}: {v:.4f}")
    (outdir / "denoising_validation_report.txt").write_text("\n".join(lines))


def run(args):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    raw, fps = load_movie(args.input)
    print(f"Loaded: {args.input}")
    print(f"Shape={raw.shape}, fps={fps:.2f}, range=({raw.min():.1f},{raw.max():.1f}), median={np.median(raw):.1f}, nonzero={(raw > 0).mean()*100:.2f}%")

    if np.median(raw) > 20 and (raw > 0).mean() > 0.5:
        print("WARNING: input has a suspicious grey baseline. Use the fixed registration/BGS output, not an old display-scaled AVI.")

    processed_input, bg_stack = subtract_background_optional(raw, args)
    denoised = denoise_stack(processed_input, args)

    if args.cleanup == "mask":
        masks = make_masks(denoised, args)
        final = denoised.copy()
        final[masks == 0] = 0
    else:
        masks = (denoised > 0).astype(np.uint8) * 255
        final = denoised

    if args.final_floor > 0:
        final = final.copy()
        final[final < float(args.final_floor)] = 0

    stem = Path(args.input).stem
    avi_path = outdir / f"{stem}_denoised_fixed.avi"
    tif_path = outdir / f"{stem}_denoised_fixed.tif"
    save_movie(final, str(avi_path), fps, args.clip_percentile)
    if not args.no_tiff:
        save_movie(final, str(tif_path), fps, args.clip_percentile)

    if args.save_mask:
        save_movie(masks.astype(np.float32), str(outdir / f"{stem}_mask.avi"), fps, args.clip_percentile)
    if args.save_bg and bg_stack is not None:
        save_movie(bg_stack, str(outdir / f"{stem}_estimated_background.tif"), fps, args.clip_percentile)

    if not args.no_validate:
        save_validation(raw, processed_input, denoised, final, masks, outdir, args, bad_movie_path=args.bad_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed denoising for lightsheet AVI/TIFF movies")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    # Optional background subtraction. Default is none because this script is normally run after BGS.
    parser.add_argument("--bg", default="none", choices=["none", "gaussian"])
    parser.add_argument("--bg-sigma", type=float, default=120.0)
    parser.add_argument("--bg-downsample", type=int, default=8)
    parser.add_argument("--bg-alpha", type=float, default=0.45)
    parser.add_argument("--bg-offset", type=float, default=0.0)

    # Denoising.
    parser.add_argument("--method", default="median_gaussian", choices=["none", "median", "gaussian", "median_gaussian", "nlm", "bilateral"])
    parser.add_argument("--median-ksize", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--h", type=float, default=8.0)
    parser.add_argument("--bilateral-sigma-color", type=float, default=12.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=3.0)
    parser.add_argument("--intensity-floor", type=float, default=0.0)

    # Optional cleanup mask.
    parser.add_argument("--cleanup", default="none", choices=["none", "mask"])
    parser.add_argument("--threshold-abs", type=float, default=3.0)
    parser.add_argument("--otsu-factor", type=float, default=0.12)
    parser.add_argument("--mask-blur-sigma", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--open-radius", type=int, default=0)
    parser.add_argument("--close-radius", type=int, default=1)
    parser.add_argument("--final-floor", type=float, default=0.0)

    # Validation/saving.
    parser.add_argument("--clip-percentile", type=float, default=99.9)
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--no-tiff", action="store_true", help="Skip TIFF output to save disk/time")
    parser.add_argument("--save-bg", action="store_true")
    parser.add_argument("--bad-output", default=None, help="Optional problematic output AVI to include in validation plots")
    parser.add_argument("--validation-mask-percentile", type=float, default=85.0)
    parser.add_argument("--validation-mask-min-area", type=int, default=5000)
    parser.add_argument("--validation-bg-exclusion-radius", type=int, default=30)
    parser.add_argument("--no-validate", action="store_true")

    run(parser.parse_args())