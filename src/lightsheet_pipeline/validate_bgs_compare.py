#!/usr/bin/env python3
"""
validate_bgs_compare.py

Compare background-subtraction (BGS) outputs for a lightsheet microscopy AVI/TIFF movie.

Core idea:
A better BGS result should have:
  1) low background intensity/noise,
  2) high foreground-to-background contrast,
  3) reasonable preservation of foreground signal and sharpness,
  4) no artificial grey baseline in background,
  5) no obvious removal of the organoid as 'background'.

Example:
python3 src/lightsheet_pipeline/validate_bgs_compare.py \
  --before result/registered/center_phase/pos3/registered_fixed.avi \
  --after 1=result/bgs/1/bgsub_fixed.avi 0.45=result/bgs/0.45/bgsub_fixed.avi \
  --outdir bgs_validation_extra
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_movie(path: str | Path) -> Tuple[np.ndarray, float]:
    """Load AVI/TIFF as float32 array with shape (T,H,W)."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".tif", ".tiff"}:
        import tifffile
        arr = tifffile.imread(str(path)).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim == 4 and arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        if arr.ndim != 3:
            raise ValueError(f"Expected TIFF shape (T,H,W), got {arr.shape}")
        return arr.astype(np.float32), 1.0

    if ext == ".avi":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Cannot open movie: {path}")
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
        return np.stack(frames, axis=0), float(fps)

    raise ValueError(f"Unsupported movie extension: {ext}")


def robust_display(im: np.ndarray, p_hi: float = 99.5) -> np.ndarray:
    """Display-only normalization; never use this output for analysis."""
    im = im.astype(np.float32)
    pos = im[im > 0]
    if pos.size == 0:
        return np.zeros_like(im, dtype=np.float32)
    hi = np.percentile(pos, p_hi)
    return np.clip(im / (hi + 1e-8), 0, 1)




def display_small(im: np.ndarray, max_size: int = 900) -> np.ndarray:
    """Downsample display images to keep validation plotting fast."""
    im = np.asarray(im)
    h, w = im.shape[:2]
    scale = min(1.0, float(max_size) / float(max(h, w)))
    if scale < 1.0:
        return cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return im

def sample_values(values: np.ndarray, max_n: int = 300000) -> np.ndarray:
    """Deterministic subsample for fast histograms."""
    values = np.asarray(values).ravel()
    if values.size <= max_n:
        return values
    step = max(1, int(np.ceil(values.size / max_n)))
    return values[::step][:max_n]


def disk_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def make_masks(before: np.ndarray, mask_percentile: float = 70.0, min_area: int = 5000,
               bg_exclusion_radius: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make foreground and background masks from the BEFORE movie only.
    The same masks are then used for all outputs so comparisons are fair.
    """
    # Use a high temporal projection so moving/noisy signals do not dominate.
    proj = np.percentile(before, 95, axis=0).astype(np.float32)
    pos = proj[proj > 0]
    if pos.size > 100:
        thr = np.percentile(pos, mask_percentile)
    else:
        thr = np.percentile(proj, 99)

    raw = (proj >= thr).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, disk_kernel(5))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, disk_kernel(2))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    clean = np.zeros_like(raw)
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lab] = 255

    # Fallback if threshold was too strict.
    if clean.sum() == 0:
        thr = np.percentile(pos, 50) if pos.size else np.percentile(proj, 99)
        clean = (proj >= thr).astype(np.uint8) * 255
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, disk_kernel(8))

    fg_mask = clean > 0
    fg_dilated = cv2.dilate(clean, disk_kernel(bg_exclusion_radius)) > 0
    bg_mask = ~fg_dilated

    if fg_mask.sum() == 0 or bg_mask.sum() == 0:
        raise ValueError("Foreground or background mask is empty; adjust mask parameters.")

    return fg_mask, bg_mask, proj


def safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def summarize_movie(movie: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray,
                    threshold_abs: float = 5.0) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Return median-over-frames metrics and per-frame metrics."""
    rows: List[Dict[str, float]] = []
    for i, frame in enumerate(movie):
        fg = frame[fg_mask]
        bg = frame[bg_mask]
        lap = cv2.Laplacian(frame.astype(np.float32), cv2.CV_32F)
        fg_mean = float(np.mean(fg))
        bg_mean = float(np.mean(bg))
        bg_std = float(np.std(bg))
        row = {
            "frame": float(i),
            "global_median": float(np.median(frame)),
            "global_nonzero_pct": float(np.mean(frame > threshold_abs) * 100),
            "background_mean": bg_mean,
            "background_median": float(np.median(bg)),
            "background_std_noise": bg_std,
            "background_p95": safe_percentile(bg, 95),
            "background_p99": safe_percentile(bg, 99),
            "background_nonzero_pct": float(np.mean(bg > threshold_abs) * 100),
            "background_black_pct": float(np.mean(bg <= threshold_abs) * 100),
            "foreground_mean": fg_mean,
            "foreground_median": float(np.median(fg)),
            "foreground_p90": safe_percentile(fg, 90),
            "foreground_p95": safe_percentile(fg, 95),
            "foreground_p99": safe_percentile(fg, 99),
            "CNR_mean": float((fg_mean - bg_mean) / (bg_std + 1e-8)),
            "SBR_mean": float((fg_mean + 1e-8) / (bg_mean + 1e-8)),
            "sharpness_laplacian_var_fg": float(np.var(lap[fg_mask])),
        }
        rows.append(row)

    keys = [k for k in rows[0].keys() if k != "frame"]
    summary = {k: float(np.median([r[k] for r in rows])) for k in keys}
    return summary, rows


def compare_to_before(before_summary: Dict[str, float], after_summary: Dict[str, float]) -> Dict[str, float]:
    """Add relative metrics compared with the uncorrected registered movie."""
    def ratio(a: float, b: float) -> float:
        return float(a / (b + 1e-8))

    comp = dict(after_summary)
    comp["bg_mean_ratio_vs_before"] = ratio(after_summary["background_mean"], before_summary["background_mean"])
    comp["bg_std_ratio_vs_before"] = ratio(after_summary["background_std_noise"], before_summary["background_std_noise"])
    comp["bg_p95_ratio_vs_before"] = ratio(after_summary["background_p95"], before_summary["background_p95"])
    comp["fg_p95_retention_vs_before"] = ratio(after_summary["foreground_p95"], before_summary["foreground_p95"])
    comp["fg_mean_retention_vs_before"] = ratio(after_summary["foreground_mean"], before_summary["foreground_mean"])
    comp["sharpness_retention_vs_before"] = ratio(after_summary["sharpness_laplacian_var_fg"], before_summary["sharpness_laplacian_var_fg"])
    comp["CNR_gain_vs_before"] = ratio(after_summary["CNR_mean"], before_summary["CNR_mean"])
    return comp


def classify_result(summary: Dict[str, float]) -> str:
    """Heuristic flags; not a replacement for visual inspection."""
    flags = []
    if summary["global_median"] > 30 or summary["background_median"] > 20:
        flags.append("grey-baseline/artifact")
    if summary["background_nonzero_pct"] > 50:
        flags.append("background-not-clean")
    if summary.get("fg_p95_retention_vs_before", 1.0) < 0.35:
        flags.append("possible-over-subtraction")
    if summary.get("sharpness_retention_vs_before", 1.0) < 0.35:
        flags.append("foreground-blurred/lost")
    if not flags:
        return "OK / plausible"
    return "; ".join(flags)


def write_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(before: np.ndarray, movies: Dict[str, np.ndarray], fg_mask: np.ndarray, bg_mask: np.ndarray,
               summaries: Dict[str, Dict[str, float]], frame_rows: Dict[str, List[Dict[str, float]]],
               outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    names = list(movies.keys())
    n = min(len(m) for m in movies.values())
    idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

    # Mask check.
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    mid = n // 2
    axes[0].imshow(display_small(robust_display(before[mid])), cmap="gray")
    axes[0].set_title("Before middle frame")
    axes[1].imshow(display_small(fg_mask.astype(np.float32)), cmap="gray")
    axes[1].set_title("Foreground mask")
    axes[2].imshow(display_small(bg_mask.astype(np.float32)), cmap="gray")
    axes[2].set_title("Background mask")
    overlay = np.dstack([robust_display(before[mid])] * 3)
    overlay[fg_mask, 0] = 1
    overlay[bg_mask, 1] = np.maximum(overlay[bg_mask, 1], 0.6)
    axes[3].imshow(display_small(overlay))
    axes[3].set_title("Mask overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(outdir / "01_mask_check.png", dpi=140)
    plt.close(fig)

    # Movie comparison grid.
    fig, axes = plt.subplots(len(idxs), len(names), figsize=(4 * len(names), 3.7 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    if len(names) == 1:
        axes = axes[:, None]
    for r, idx in enumerate(idxs):
        for c, name in enumerate(names):
            axes[r, c].imshow(display_small(robust_display(movies[name][idx])), cmap="gray")
            axes[r, c].set_title(f"{name}\nframe {idx}")
            axes[r, c].axis("off")
    plt.tight_layout()
    fig.savefig(outdir / "02_movie_comparison_grid.png", dpi=140)
    plt.close(fig)

    # Difference/removed signal panels against before.
    after_names = [n for n in names if n != "before"]
    if after_names:
        fig, axes = plt.subplots(len(idxs), len(after_names), figsize=(4 * len(after_names), 3.7 * len(idxs)))
        if len(idxs) == 1:
            axes = axes[None, :]
        if len(after_names) == 1:
            axes = axes[:, None]
        for r, idx in enumerate(idxs):
            for c, name in enumerate(after_names):
                # Difference is meaningful mainly when the after movie came from the same before.
                # Still useful to reveal obvious grey-baseline artifacts.
                diff = before[idx].astype(np.float32) - movies[name][idx].astype(np.float32)
                axes[r, c].imshow(display_small(robust_display(np.abs(diff))), cmap="gray")
                axes[r, c].set_title(f"abs(before - {name})\nframe {idx}")
                axes[r, c].axis("off")
        plt.tight_layout()
        fig.savefig(outdir / "03_absolute_difference_from_before.png", dpi=140)
        plt.close(fig)

    # Summary bar plots.
    selected = [
        "global_median", "background_median", "background_mean", "background_std_noise",
        "background_p95", "background_nonzero_pct", "foreground_p95", "CNR_mean",
        "sharpness_laplacian_var_fg",
    ]
    summary_rows = []
    for metric in selected:
        for name in names:
            summary_rows.append({"name": name, "metric": metric, "value": summaries[name][metric]})

    for metric in selected:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = [summaries[name][metric] for name in names]
        ax.bar(np.arange(len(names)), vals)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig.savefig(outdir / f"04_metric_{metric}.png", dpi=140)
        plt.close(fig)

    # Timeseries plots.
    trend_metrics = ["background_mean", "background_std_noise", "foreground_p95", "CNR_mean"]
    for metric in trend_metrics:
        fig, ax = plt.subplots(figsize=(11, 5))
        for name in names:
            rows = frame_rows[name]
            ax.plot([r["frame"] for r in rows], [r[metric] for r in rows], label=name)
        ax.set_xlabel("Frame")
        ax.set_ylabel(metric)
        ax.set_title(f"Frame-wise {metric}")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(outdir / f"05_timeseries_{metric}.png", dpi=140)
        plt.close(fig)

    # Histograms: background and foreground intensity distributions.
    sample_idxs = idxs
    for region_name, mask in [("background", bg_mask), ("foreground", fg_mask)]:
        all_vals = []
        vals_by_name = {}
        for name in names:
            vals = np.concatenate([sample_values(movies[name][i][mask].ravel(), max_n=80000) for i in sample_idxs])
            vals = sample_values(vals, max_n=250000)
            vals_by_name[name] = vals
            all_vals.append(vals)
        combined = np.concatenate(all_vals)
        hi = np.percentile(combined, 99.5)
        if hi <= 0:
            hi = 1
        fig, ax = plt.subplots(figsize=(10, 5))
        for name in names:
            ax.hist(vals_by_name[name], bins=100, range=(0, hi), histtype="step", linewidth=1.5, label=name)
        ax.set_title(f"{region_name.capitalize()} intensity histogram")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Pixel count")
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / f"06_histogram_{region_name}.png", dpi=140)
        plt.close(fig)




def downsample_movie(movie: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return movie
    t, h, w = movie.shape
    new_w = max(1, w // factor)
    new_h = max(1, h // factor)
    return np.stack([cv2.resize(fr, (new_w, new_h), interpolation=cv2.INTER_AREA) for fr in movie]).astype(np.float32)


def parse_after(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem
        out[name] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lightsheet background-subtraction outputs.")
    parser.add_argument("--before", required=True, help="Input registered movie before BGS.")
    parser.add_argument("--after", nargs="+", required=True,
                        help="One or more outputs to compare. Use name=path, e.g. fixed=out.avi")
    parser.add_argument("--outdir", default="bgs_validation_extra", help="Output directory for validation results.")
    parser.add_argument("--mask-percentile", type=float, default=70.0)
    parser.add_argument("--mask-min-area", type=int, default=5000)
    parser.add_argument("--bg-exclusion-radius", type=int, default=60)
    parser.add_argument("--threshold-abs", type=float, default=5.0)
    parser.add_argument("--analysis-downsample", type=int, default=4, help="Downsample movies for faster validation metrics/plots. Use 1 for full resolution.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    before, fps = load_movie(args.before)
    after_paths = parse_after(args.after)

    movies: Dict[str, np.ndarray] = {"before": before}
    for name, path in after_paths.items():
        arr, _ = load_movie(path)
        movies[name] = arr

    if args.analysis_downsample > 1:
        movies = {name: downsample_movie(arr, args.analysis_downsample) for name, arr in movies.items()}
        before = movies["before"]
        print(f"Using downsampled movies for validation metrics: factor={args.analysis_downsample}, shape={before.shape}")

    # Align all movies to shared frame count and spatial shape.
    min_t = min(m.shape[0] for m in movies.values())
    h, w = before.shape[1:]
    for name, arr in list(movies.items()):
        if arr.shape[1:] != (h, w):
            raise ValueError(f"Movie {name} has shape {arr.shape}; expected spatial shape {(h, w)}")
        movies[name] = arr[:min_t]
    before = movies["before"]

    fg_mask, bg_mask, proj = make_masks(
        before,
        mask_percentile=args.mask_percentile,
        min_area=args.mask_min_area,
        bg_exclusion_radius=args.bg_exclusion_radius,
    )

    summaries: Dict[str, Dict[str, float]] = {}
    frame_rows: Dict[str, List[Dict[str, float]]] = {}
    for name, movie in movies.items():
        summaries[name], frame_rows[name] = summarize_movie(movie, fg_mask, bg_mask, args.threshold_abs)

    before_summary = summaries["before"]
    comparison_rows: List[Dict[str, float | str]] = []
    for name, summary in summaries.items():
        if name == "before":
            row = dict(summary)
        else:
            row = compare_to_before(before_summary, summary)
        row["name"] = name
        row["interpretation_flag"] = classify_result(row)  # type: ignore[arg-type]
        # Move human-friendly columns first.
        first = {"name": row.pop("name"), "interpretation_flag": row.pop("interpretation_flag")}
        first.update(row)
        comparison_rows.append(first)

    write_csv(outdir / "summary_comparison.csv", comparison_rows)
    for name, rows in frame_rows.items():
        write_csv(outdir / f"frame_metrics_{name}.csv", rows)

    save_plots(before, movies, fg_mask, bg_mask, summaries, frame_rows, outdir)

    # Short report.
    with (outdir / "how_to_choose_bgs.txt").open("w", encoding="utf-8") as f:
        f.write("How to choose a better background subtraction result\n")
        f.write("=================================================\n\n")
        f.write("Prefer the result that satisfies all of these:\n")
        f.write("1. Background median/mean/p95 are low and background_black_pct is high.\n")
        f.write("2. CNR_mean is higher than before.\n")
        f.write("3. Foreground_p95 retention is not too low; as a rough guide, avoid <0.35 unless you intentionally want a very sparse signal.\n")
        f.write("4. Sharpness retention is not too low; low sharpness means structures may be smoothed/erased.\n")
        f.write("5. The removed-signal image should look like smooth background, not like the organoid/ring.\n")
        f.write("6. Avoid outputs flagged as grey-baseline/artifact or background-not-clean.\n\n")
        f.write("Summary flags:\n")
        for row in comparison_rows:
            f.write(f"- {row['name']}: {row['interpretation_flag']}\n")
        f.write("\nKey metric table is in summary_comparison.csv.\n")

    print(f"Saved validation results to: {outdir}")
    for row in comparison_rows:
        print(f"{row['name']}: {row['interpretation_flag']}")
        print(
            f"  bg_median={row['background_median']:.3f}, "
            f"bg_p95={row['background_p95']:.3f}, "
            f"bg_nonzero_pct={row['background_nonzero_pct']:.2f}, "
            f"fg_p95={row['foreground_p95']:.3f}, "
            f"CNR={row['CNR_mean']:.3f}"
        )


if __name__ == "__main__":
    main()