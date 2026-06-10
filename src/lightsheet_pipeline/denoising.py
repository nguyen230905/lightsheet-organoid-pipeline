#!/usr/bin/env python3
"""
denoise_with_fast_background.py

Clean registered lightsheet AVI/TIFF movies using adjustable, less aggressive background subtraction:
1) fast background subtraction (large Gaussian blur on downsampled frame)
2) denoising
3) threshold + connected-component cleanup
4) save WITHOUT min-max stretching, so tiny noise does not become white

Recommended for your pos3 registered AVI:
python src/lightsheet_pipeline/denoising3.py \
  --input tests/result/pos3/registered_pos3.avi \
  --output tests/result/Denoise.v4/denoised_pos3.avi \
  --bg gaussian \
  --bg-sigma 120 \
  --bg-downsample 8 \
  --bg-offset 2 \
  --method median_gaussian \
  --keep-largest \
  --min-area 5000 \
  --threshold-abs 8 \
  --otsu-factor 0.25 \
  --save-bg --save-mask

If still noisy, increase:
  --bg-offset 4
  --threshold-abs 12
  --min-area 20000
  --otsu-factor 0.35
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def load_movie(path: str):
    """Load AVI or multi-page TIFF into float32 array with shape (T,H,W)."""
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
    Save movie without frame/global min-max stretching.

    Important: min-max stretching makes weak residual background look bright white.
    For 8-bit AVI input, this preserves the original 0-255 scale.
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
    for frame in u8:
        writer.write(frame)
    writer.release()
    print(f"Saved AVI: {p}")


def disk_kernel(radius: int):
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def gaussian_background(frame: np.ndarray, sigma: float, downsample: int):
    """
    Fast smooth background estimate.

    We downsample first, blur strongly, then upsample back. This is much faster than
    applying a huge blur directly on a 2266 x 2296 frame.
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
    Temporal background estimate.

    Warning: for registered movies, temporal background can remove real signal if the
    organoid stays in the same place. Prefer --bg gaussian for your case.
    """
    return np.percentile(frames, percentile, axis=0).astype(np.float32)


def subtract_background(frames: np.ndarray, args):
    """Apply optional background subtraction, then subtract offset and clip negatives."""
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
        print(f"Temporal background p{args.temporal_percentile}: range=({bg.min():.1f},{bg.max():.1f})")

    else:
        raise ValueError("--bg must be none, gaussian, or temporal")

    corrected = corrected - float(args.bg_offset)
    corrected[corrected < 0] = 0
    return corrected.astype(np.float32), bg_stack


def denoise_frame(frame: np.ndarray, args):
    """Denoise one frame. Keeps values in roughly 0-255 scale."""
    f = np.clip(frame, 0, 255).astype(np.uint8)

    if args.median_ksize > 1:
        k = int(args.median_ksize)
        if k % 2 == 0:
            k += 1
        f = cv2.medianBlur(f, k)

    if args.method == "none":
        return f.astype(np.float32)
    if args.method == "median":
        return f.astype(np.float32)
    if args.method == "gaussian":
        return cv2.GaussianBlur(f.astype(np.float32), (0, 0), args.sigma).astype(np.float32)
    if args.method == "median_gaussian":
        return cv2.GaussianBlur(f.astype(np.float32), (0, 0), args.sigma).astype(np.float32)
    if args.method == "nlm":
        return cv2.fastNlMeansDenoising(f, None, h=float(args.h), templateWindowSize=7, searchWindowSize=21).astype(np.float32)

    raise ValueError("Unknown denoising method")


def denoise_stack(frames: np.ndarray, args):
    out = np.empty_like(frames, dtype=np.float32)
    for i, frame in enumerate(frames):
        if i % max(1, len(frames) // 5) == 0:
            print(f"Denoising frame {i}/{len(frames)-1}")
        out[i] = denoise_frame(frame, args)
    return out


def fill_holes(mask: np.ndarray):
    """Fill holes in a binary uint8 mask where foreground is 255."""
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def clean_components(mask: np.ndarray, args):
    """Remove small white regions; optionally keep only the largest object."""
    mask = (mask > 0).astype(np.uint8) * 255

    if args.open_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk_kernel(args.open_radius))
    if args.close_radius > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_kernel(args.close_radius))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)

    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        valid = np.where(areas >= int(args.min_area))[0] + 1
        if valid.size > 0:
            if args.keep_largest:
                largest = valid[np.argmax(stats[valid, cv2.CC_STAT_AREA])]
                clean[labels == largest] = 255
            else:
                for lab in valid:
                    clean[labels == lab] = 255

    if not args.no_fill_holes and np.any(clean):
        clean = fill_holes(clean)

    if args.close_radius > 0:
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, disk_kernel(args.close_radius))

    return clean


def compute_threshold(frame: np.ndarray, threshold_abs: float, otsu_factor: float):
    """Threshold from Otsu on nonzero pixels, with an absolute lower bound."""
    vals = frame[frame > 0]
    if vals.size < 100:
        return float(threshold_abs)
    vals_u8 = np.clip(vals, 0, 255).astype(np.uint8)
    otsu_t, _ = cv2.threshold(vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(threshold_abs), float(otsu_t) * float(otsu_factor))


def make_masks(frames: np.ndarray, args):
    masks = np.empty(frames.shape, dtype=np.uint8)
    thresholds = []

    for i, frame in enumerate(frames):
        smooth = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), args.mask_blur_sigma) if args.mask_blur_sigma > 0 else frame
        thr = compute_threshold(smooth, args.threshold_abs, args.otsu_factor)
        raw_mask = smooth > thr
        masks[i] = clean_components(raw_mask, args)
        thresholds.append(thr)

    print(f"Mask threshold: min={min(thresholds):.2f}, median={np.median(thresholds):.2f}, max={max(thresholds):.2f}")
    return masks


def apply_floor(frames: np.ndarray, floor: float):
    out = frames.astype(np.float32).copy()
    out[out < float(floor)] = 0
    return out


def save_validation(raw, bg_sub, den, masks, clean, outdir: Path):
    """
    Save validation figures + CSV + TXT report.

    Outputs saved in args.output folder:
      - validation_with_background.png
      - validation_summary_bar.png
      - validation_metrics.csv
      - validation_report.txt

    Interpretation:
      Good background subtraction: background mean/std/nonzero% decrease, CNR increases.
      Good denoising: background std decreases, foreground signal is preserved, sharpness does not collapse.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    idxs = sorted(set([0, len(raw)//4, len(raw)//2, 3*len(raw)//4, len(raw)-1]))

    def show_norm(im):
        positive = im[im > 0]
        hi = np.percentile(positive, 99.5) if positive.size else 1.0
        return np.clip(im / (hi + 1e-8), 0, 1)

    def get_fg_bg_masks(stack, final_masks):
        """Use final object mask as foreground. If empty, fallback to percentile threshold."""
        fg = final_masks > 0
        if fg.mean() < 0.0001:
            thr = np.percentile(stack, 99)
            fg = stack >= thr
        bg = ~fg
        return fg, bg

    def laplacian_sharpness(stack, fg_mask):
        """Average Laplacian variance inside/near foreground across a few frames."""
        vals = []
        sample_idxs = idxs
        for i in sample_idxs:
            frame = np.clip(stack[i], 0, 255).astype(np.uint8)
            lap = cv2.Laplacian(frame, cv2.CV_32F)
            m = fg_mask[i]
            if np.any(m):
                vals.append(float(np.var(lap[m])))
        return float(np.mean(vals)) if vals else 0.0

    def stack_metrics(name, stack, fg, bg):
        fg_vals = stack[fg]
        bg_vals = stack[bg]
        if fg_vals.size == 0:
            fg_vals = stack.reshape(-1)
        if bg_vals.size == 0:
            bg_vals = stack.reshape(-1)

        bg_mean = float(np.mean(bg_vals))
        bg_std = float(np.std(bg_vals))
        fg_mean = float(np.mean(fg_vals))
        fg_p95 = float(np.percentile(fg_vals, 95))
        nonzero = float((stack > 0).mean() * 100.0)
        cnr = float((fg_mean - bg_mean) / (bg_std + 1e-8))
        sharp = laplacian_sharpness(stack, fg)

        return {
            "stage": name,
            "min": float(np.min(stack)),
            "max": float(np.max(stack)),
            "mean_all": float(np.mean(stack)),
            "nonzero_percent": nonzero,
            "background_mean": bg_mean,
            "background_std_noise": bg_std,
            "foreground_mean": fg_mean,
            "foreground_p95": fg_p95,
            "cnr": cnr,
            "sharpness_laplacian_var_fg": sharp,
        }

    fg, bg = get_fg_bg_masks(clean, masks)

    rows = [
        stack_metrics("raw_before_processing", raw, fg, bg),
        stack_metrics("after_background_subtraction", bg_sub, fg, bg),
        stack_metrics("after_denoising", den, fg, bg),
        stack_metrics("final_clean", clean, fg, bg),
    ]

    # 1) Save visual panel
    fig, axes = plt.subplots(len(idxs), 5, figsize=(18, 4 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]

    for r, idx in enumerate(idxs):
        panels = [
            (raw[idx], f"raw f{idx}"),
            (bg_sub[idx], "after bg subtraction"),
            (den[idx], "denoised"),
            (masks[idx], "mask"),
            (clean[idx], "final clean"),
        ]
        for c, (im, title) in enumerate(panels):
            axes[r, c].imshow(im if title == "mask" else show_norm(im), cmap="gray")
            axes[r, c].set_title(title)
            axes[r, c].axis("off")

    plt.tight_layout()
    panel_path = outdir / "validation_with_background.png"
    fig.savefig(panel_path, dpi=120)
    plt.close(fig)
    print(f"Saved validation figure: {panel_path}")

    # 2) Save CSV metrics
    csv_path = outdir / "validation_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved validation metrics CSV: {csv_path}")

    # 3) Save summary bar chart
    metrics_to_plot = [
        "background_mean",
        "background_std_noise",
        "nonzero_percent",
        "foreground_mean",
        "foreground_p95",
        "cnr",
        "sharpness_laplacian_var_fg",
    ]
    x = np.arange(len(metrics_to_plot))
    width = 0.2

    fig, ax = plt.subplots(figsize=(15, 6))
    for j, row in enumerate(rows):
        vals = [row[m] for m in metrics_to_plot]
        # normalize each metric by raw value so very different scales can be compared
        norm_vals = []
        for m, v in zip(metrics_to_plot, vals):
            raw_v = rows[0][m]
            norm_vals.append(v / (abs(raw_v) + 1e-8))
        ax.bar(x + (j - 1.5) * width, norm_vals, width, label=row["stage"])

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=35, ha="right")
    ax.set_ylabel("Relative to raw before processing")
    ax.set_title("Validation summary: values below 1 decreased; values above 1 increased")
    ax.legend(fontsize=8)
    plt.tight_layout()
    bar_path = outdir / "validation_summary_bar.png"
    fig.savefig(bar_path, dpi=140)
    plt.close(fig)
    print(f"Saved validation summary bar chart: {bar_path}")

    # 4) Save text report with interpretation
    raw_row = rows[0]
    bg_row = rows[1]
    den_row = rows[2]
    clean_row = rows[3]

    def pct_change(after, before):
        return 100.0 * (after - before) / (abs(before) + 1e-8)

    report = []
    report.append("LIGHTSHEET PROCESSING VALIDATION REPORT")
    report.append("======================================")
    report.append("")
    report.append("Files generated:")
    report.append(f"- {panel_path.name}")
    report.append(f"- {bar_path.name}")
    report.append(f"- {csv_path.name}")
    report.append("")
    report.append("Background subtraction check: raw_before_processing -> after_background_subtraction")
    report.append(f"- Background mean change: {pct_change(bg_row['background_mean'], raw_row['background_mean']):.2f}%")
    report.append(f"- Background noise/std change: {pct_change(bg_row['background_std_noise'], raw_row['background_std_noise']):.2f}%")
    report.append(f"- Nonzero pixel % change: {pct_change(bg_row['nonzero_percent'], raw_row['nonzero_percent']):.2f}%")
    report.append(f"- CNR change: {pct_change(bg_row['cnr'], raw_row['cnr']):.2f}%")
    report.append(f"- Foreground mean change: {pct_change(bg_row['foreground_mean'], raw_row['foreground_mean']):.2f}%")
    report.append("")
    report.append("Denoising check: after_background_subtraction -> after_denoising")
    report.append(f"- Background noise/std change: {pct_change(den_row['background_std_noise'], bg_row['background_std_noise']):.2f}%")
    report.append(f"- CNR change: {pct_change(den_row['cnr'], bg_row['cnr']):.2f}%")
    report.append(f"- Foreground mean change: {pct_change(den_row['foreground_mean'], bg_row['foreground_mean']):.2f}%")
    report.append(f"- Foreground p95 change: {pct_change(den_row['foreground_p95'], bg_row['foreground_p95']):.2f}%")
    report.append(f"- Sharpness change: {pct_change(den_row['sharpness_laplacian_var_fg'], bg_row['sharpness_laplacian_var_fg']):.2f}%")
    report.append("")
    report.append("How to judge:")
    report.append("- Good background subtraction: background_mean and background_std_noise decrease, while foreground_mean is not heavily reduced.")
    report.append("- Good denoising: background_std_noise decreases, CNR increases or stays similar, and foreground_mean / foreground_p95 are preserved.")
    report.append("- If sharpness_laplacian_var_fg drops too much, denoising may be too strong and may blur real biological structure.")
    report.append("- If foreground_mean drops strongly after background subtraction, bg_alpha/bg_offset may be too aggressive.")
    report.append("")
    report.append("Raw metrics:")
    for row in rows:
        report.append(f"\n[{row['stage']}]")
        for k, v in row.items():
            if k != "stage":
                report.append(f"{k}: {v:.4f}")

    txt_path = outdir / "validation_report.txt"
    txt_path.write_text("\n".join(report))
    print(f"Saved validation report: {txt_path}")


def run(args):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    raw, fps = load_movie(args.input)
    print(f"Loaded: {args.input}")
    print(f"Shape={raw.shape}, fps={fps:.2f}, range=({raw.min():.1f},{raw.max():.1f}), nonzero={(raw > 0).mean()*100:.2f}%")

    bg_sub, bg_stack = subtract_background(raw, args)
    print(f"After background: range=({bg_sub.min():.1f},{bg_sub.max():.1f}), nonzero={(bg_sub > 0).mean()*100:.2f}%")

    den = denoise_stack(bg_sub, args)
    den = apply_floor(den, args.intensity_floor)

    masks = make_masks(den, args)

    clean = den.copy()
    clean[masks == 0] = 0
    clean = apply_floor(clean, args.final_floor)
    print(f"Final clean: range=({clean.min():.1f},{clean.max():.1f}), nonzero={(clean > 0).mean()*100:.2f}%")

    stem = Path(args.input).stem
    save_movie_no_stretch(clean, str(outdir / f"{stem}_bg_clean.avi"), fps, args.clip_percentile)
    save_movie_no_stretch(clean, str(outdir / f"{stem}_bg_clean.tif"), fps, args.clip_percentile)

    if args.save_mask:
        save_movie_no_stretch(masks.astype(np.float32), str(outdir / f"{stem}_mask.avi"), fps, args.clip_percentile)
    if args.save_bg and bg_stack is not None:
        save_movie_no_stretch(bg_stack, str(outdir / f"{stem}_estimated_background.tif"), fps, args.clip_percentile)

    if not args.no_validate:
        save_validation(raw, bg_sub, den, masks, clean, outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise + fast background subtraction + mild/adjustable cleanup for lightsheet movies")

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    # Background subtraction
    parser.add_argument("--bg", default="gaussian", choices=["none", "gaussian", "temporal"])
    parser.add_argument("--bg-sigma", type=float, default=120.0, help="Large blur sigma for smooth background estimate")
    parser.add_argument("--bg-downsample", type=int, default=8, help="Downsample factor for faster background estimation")
    parser.add_argument("--bg-offset", type=float, default=0.0, help="Extra constant subtracted after background subtraction")
    parser.add_argument("--bg-alpha", type=float, default=0.45, help="How much background to subtract. Lower = less dark. Try 0.25-0.60")
    parser.add_argument("--temporal-percentile", type=float, default=2.0)

    # Denoising
    parser.add_argument("--method", default="median_gaussian", choices=["none", "median", "gaussian", "median_gaussian", "nlm"])
    parser.add_argument("--median-ksize", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--h", type=float, default=10.0)

    # Final cleanup
    parser.add_argument("--intensity-floor", type=float, default=0.0)
    parser.add_argument("--final-floor", type=float, default=0.0)
    parser.add_argument("--threshold-abs", type=float, default=3.0)
    parser.add_argument("--otsu-factor", type=float, default=0.12)
    parser.add_argument("--mask-blur-sigma", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--open-radius", type=int, default=0)
    parser.add_argument("--close-radius", type=int, default=1)
    parser.add_argument("--no-fill-holes", action="store_true")

    # Saving
    parser.add_argument("--clip-percentile", type=float, default=99.9)
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--save-bg", action="store_true")
    parser.add_argument("--no-validate", action="store_true")

    run(parser.parse_args())