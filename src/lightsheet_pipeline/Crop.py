#!/usr/bin/env python3
"""
crop.py — robust fixed-size cropping for 2D lightsheet AVI/TIFF timelapse movies.

Main fixes compared with the uploaded Crop.py:
1. Crop never changes intensities. The output AVI is clipped to 0–255, not min–max stretched.
2. AVI is written as 3-channel BGR MJPG, avoiding grayscale-MJPG corruption/display artifacts.
3. Crop detection is based on a stable temporal projection, not a noisy per-frame 85th percentile.
4. Thresholding is done on positive pixels, which is safer for sparse black-background microscopy movies.
5. Saves validation images, crop coordinates, and a report.

Recommended use after the fixed registration + BGS steps:
python3 src/lightsheet_pipeline/Crop.py \
  --input result/bgs/0.45/bgsub_fixed.avi \
  --output result/cropped/ \
  --margin 100 \
  --positive-percentile 55 \
  --square

If you only want a preview:
python3 src/lightsheet_pipeline/Crop.py --input movie.avi --output crop_preview --preview-only
"""

import argparse
import json
from pathlib import Path
import csv

import cv2
import numpy as np


# -----------------------------
# IO
# -----------------------------

def load_movie(path: str):
    """Load AVI/TIFF as float32 array with shape (T, H, W)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in [".tif", ".tiff"]:
        import tifffile
        arr = tifffile.imread(str(p))
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            arr = arr[..., 0]
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 3:
            raise ValueError(f"Expected TIFF as (T,H,W), got {arr.shape}")
        return arr.astype(np.float32), 1.0

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
        return np.stack(frames, axis=0), float(fps)

    raise ValueError("Input must be .avi, .tif, or .tiff")


def warn_if_suspicious_baseline(frames: np.ndarray):
    """Warn when the input looks like a previously corrupted display movie."""
    med = float(np.median(frames))
    p01 = float(np.percentile(frames, 1))
    nonzero = float((frames > 0).mean() * 100)
    warnings = []
    if med > 20 and nonzero > 50:
        warnings.append(
            "Input has a high nonzero baseline: median={:.2f}, p1={:.2f}, nonzero={:.2f}%. "
            "For a black-background light-sheet movie this usually means you are cropping a display-normalized/corrupted AVI. "
            "Use the fixed BGS output or a TIFF/raw stack instead.".format(med, p01, nonzero)
        )
    return warnings


def to_uint8_preserve(frames: np.ndarray, clip_percentile: float = 99.9):
    """
    Convert to uint8 for AVI without per-frame/global min-max stretching.
    If data are already in 0–255, preserve those values by clipping.
    For larger-range data, scale once using a positive-pixel percentile for display only.
    """
    arr = np.nan_to_num(frames.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmax(arr)) <= 255.0 and float(np.nanmin(arr)) >= 0.0:
        return np.clip(arr, 0, 255).astype(np.uint8)
    pos = arr[arr > 0]
    hi = float(np.percentile(pos, clip_percentile)) if pos.size else 1.0
    hi = max(hi, 1.0)
    return (np.clip(arr, 0, hi) / hi * 255.0).astype(np.uint8)


def save_movie(frames: np.ndarray, path: str, fps: float = 4.0, clip_percentile: float = 99.9):
    """Save TIFF or AVI. AVI is BGR MJPG for robust OpenCV/Fiji playback."""
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

    u8 = to_uint8_preserve(frames, clip_percentile=clip_percentile)
    t, h, w = u8.shape
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (w, h), isColor=True)
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {path}")
    for fr in u8:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
    writer.release()
    print(f"Saved AVI: {p}")


# -----------------------------
# Crop detection
# -----------------------------

def disk_kernel(radius: int):
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def robust_display(frame: np.ndarray):
    """Normalize for PNG visualization only. Does not affect saved movie."""
    fr = frame.astype(np.float32)
    pos = fr[fr > 0]
    if pos.size == 0:
        return np.zeros_like(fr, dtype=np.float32)
    hi = float(np.percentile(pos, 99.5))
    return np.clip(fr / (hi + 1e-8), 0, 1)


def make_temporal_projection(frames: np.ndarray, projection: str = "p95"):
    if projection == "max":
        return np.max(frames, axis=0).astype(np.float32)
    if projection == "mean":
        return np.mean(frames, axis=0).astype(np.float32)
    if projection.startswith("p"):
        pct = float(projection[1:])
        return np.percentile(frames, pct, axis=0).astype(np.float32)
    raise ValueError("projection must be max, mean, or pXX such as p95")


def foreground_mask_from_projection(
    projection: np.ndarray,
    positive_percentile: float = 70.0,
    min_area: int = 5000,
    close_radius: int = 15,
    dilate_radius: int = 25,
):
    """
    Build organoid mask from a temporal projection.
    Threshold is computed on positive pixels only, avoiding the common problem that the
    85th percentile of a sparse black-background image is 0.
    """
    proj = projection.astype(np.float32)
    pos = proj[proj > 0]
    if pos.size < 100:
        thr = float(np.percentile(proj, 99))
    else:
        thr = float(np.percentile(pos, positive_percentile))
    raw = (proj > thr).astype(np.uint8) * 255

    if close_radius > 0:
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, disk_kernel(close_radius))
    if dilate_radius > 0:
        raw = cv2.dilate(raw, disk_kernel(dilate_radius))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, disk_kernel(2))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    clean = np.zeros_like(raw)
    kept = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            kept.append((area, lab))
    if not kept and num > 1:
        # Fallback: keep largest component if the area cutoff was too strict.
        areas = stats[1:, cv2.CC_STAT_AREA]
        lab = 1 + int(np.argmax(areas))
        kept = [(int(stats[lab, cv2.CC_STAT_AREA]), lab)]
    for _, lab in kept:
        clean[labels == lab] = 255

    if clean.sum() == 0:
        raise RuntimeError("Could not detect foreground mask. Try lowering --positive-percentile or --min-area.")

    return clean.astype(bool), thr, raw.astype(bool)


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty mask")
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def expand_bbox(bbox, shape, margin: int = 80, square: bool = True, even: bool = True):
    h, w = shape
    y0, x0, y1, x1 = bbox
    y0 -= margin; x0 -= margin; y1 += margin; x1 += margin
    y0 = max(0, y0); x0 = max(0, x0); y1 = min(h, y1); x1 = min(w, x1)

    if square:
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        side = int(np.ceil(max(y1 - y0, x1 - x0)))
        if even and side % 2 == 1:
            side += 1
        y0 = int(round(cy - side / 2)); y1 = y0 + side
        x0 = int(round(cx - side / 2)); x1 = x0 + side
        # Shift back into bounds while keeping size.
        if y0 < 0:
            y1 -= y0; y0 = 0
        if x0 < 0:
            x1 -= x0; x0 = 0
        if y1 > h:
            shift = y1 - h; y0 -= shift; y1 = h
        if x1 > w:
            shift = x1 - w; x0 -= shift; x1 = w
        y0 = max(0, y0); x0 = max(0, x0)

    if even:
        if (y1 - y0) % 2 == 1 and y1 < h:
            y1 += 1
        elif (y1 - y0) % 2 == 1:
            y0 = max(0, y0 - 1)
        if (x1 - x0) % 2 == 1 and x1 < w:
            x1 += 1
        elif (x1 - x0) % 2 == 1:
            x0 = max(0, x0 - 1)

    if y1 <= y0 or x1 <= x0:
        raise RuntimeError(f"Invalid crop: {(y0,x0,y1,x1)}")
    return int(y0), int(x0), int(y1), int(x1)


def apply_crop(frames: np.ndarray, crop):
    y0, x0, y1, x1 = crop
    return frames[:, y0:y1, x0:x1].copy()


# -----------------------------
# Validation outputs
# -----------------------------

def movie_stats(frames: np.ndarray):
    return {
        "shape_T": int(frames.shape[0]),
        "shape_H": int(frames.shape[1]),
        "shape_W": int(frames.shape[2]),
        "min": float(np.min(frames)),
        "p01": float(np.percentile(frames, 1)),
        "median": float(np.median(frames)),
        "p95": float(np.percentile(frames, 95)),
        "p995": float(np.percentile(frames, 99.5)),
        "max": float(np.max(frames)),
        "nonzero_percent": float((frames > 0).mean() * 100),
    }



def draw_label(im, text, x=10, y=30):
    out = im.copy()
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def as_bgr_display(frame):
    u = (robust_display(frame) * 255).astype(np.uint8)
    return cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)


def save_validation(frames, cropped, projection, mask, raw_mask, crop, outdir: Path, stem: str, warnings):
    """Save validation outputs without matplotlib, to keep the script fast/headless-safe."""
    outdir.mkdir(parents=True, exist_ok=True)
    t, h, w = frames.shape
    y0, x0, y1, x1 = crop

    # 1) Projection overlay + masks
    proj = as_bgr_display(projection)
    cv2.rectangle(proj, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 4)
    proj = draw_label(proj, "Projection + crop")
    raw = cv2.cvtColor((raw_mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    raw = draw_label(raw, "Raw threshold mask")
    clean = cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    clean = draw_label(clean, "Clean foreground mask")
    # Resize masks/projection to common height for concatenation if needed
    panel_h = min(900, h)
    def resize_h(img):
        scale = panel_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), panel_h), interpolation=cv2.INTER_AREA)
    detection_panel = cv2.hconcat([resize_h(proj), resize_h(raw), resize_h(clean)])
    cv2.imwrite(str(outdir / f"{stem}_crop_detection_overlay.png"), detection_panel)

    # 2) Before/cropped comparison grid
    idxs = sorted(set([0, t//4, t//2, 3*t//4, t-1]))
    rows = []
    for idx in idxs:
        before = as_bgr_display(frames[idx])
        cv2.rectangle(before, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 4)
        before = draw_label(before, f"Before frame {idx}")
        after = as_bgr_display(cropped[idx])
        after = draw_label(after, f"Cropped frame {idx}")
        # Resize before to cropped height for side-by-side display
        disp_h = min(600, before.shape[0], after.shape[0])
        def rz(img):
            scale = disp_h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1]*scale), disp_h), interpolation=cv2.INTER_AREA)
        b = rz(before); a = rz(after)
        # pad to same height
        rows.append(cv2.hconcat([b, a]))
    # pad rows to same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = cv2.hconcat([r, pad])
        padded.append(r)
    grid = cv2.vconcat(padded)
    cv2.imwrite(str(outdir / f"{stem}_before_after_crop_grid.png"), grid)

    before_stats = movie_stats(frames)
    crop_stats = movie_stats(cropped)
    summary = {
        "crop_y0": y0,
        "crop_x0": x0,
        "crop_y1": y1,
        "crop_x1": x1,
        "crop_height": y1 - y0,
        "crop_width": x1 - x0,
        "area_fraction_percent": float(((y1-y0)*(x1-x0))/(h*w)*100),
        "size_reduction_percent": float((1 - ((y1-y0)*(x1-x0))/(h*w))*100),
        "before_median": before_stats["median"],
        "cropped_median": crop_stats["median"],
        "before_p95": before_stats["p95"],
        "cropped_p95": crop_stats["p95"],
        "before_nonzero_percent": before_stats["nonzero_percent"],
        "cropped_nonzero_percent": crop_stats["nonzero_percent"],
    }
    with open(outdir / f"{stem}_crop_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader(); writer.writerow(summary)

    report_lines = []
    report_lines.append("Crop validation report")
    report_lines.append("======================")
    report_lines.append(f"Original shape: T={t}, H={h}, W={w}")
    report_lines.append(f"Crop: y=[{y0},{y1}), x=[{x0},{x1}) -> {y1-y0} x {x1-x0} px")
    report_lines.append(f"Area kept: {summary['area_fraction_percent']:.2f}%")
    report_lines.append(f"Size reduction: {summary['size_reduction_percent']:.2f}%")
    report_lines.append("")
    report_lines.append("Intensity diagnostics:")
    report_lines.append(f"  before median={before_stats['median']:.3f}, p95={before_stats['p95']:.3f}, nonzero={before_stats['nonzero_percent']:.2f}%")
    report_lines.append(f"  cropped median={crop_stats['median']:.3f}, p95={crop_stats['p95']:.3f}, nonzero={crop_stats['nonzero_percent']:.2f}%")
    report_lines.append("")
    if warnings:
        report_lines.append("Warnings:")
        for wmsg in warnings:
            report_lines.append(f"  - {wmsg}")
    else:
        report_lines.append("Warnings: none")
    report_lines.append("")
    report_lines.append("Interpretation:")
    report_lines.append("  A correct crop should reduce image size while preserving the organoid and preserving the original black background/intensity scale.")
    report_lines.append("  Cropping itself should not turn median background to ~128. If that happens, the AVI writer or input movie is the problem.")
    (outdir / f"{stem}_crop_validation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

def run(args):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    frames, fps = load_movie(args.input)
    stem = Path(args.input).stem
    warnings = warn_if_suspicious_baseline(frames)
    for w in warnings:
        print("WARNING:", w)
    print(f"Loaded {args.input}: shape={frames.shape}, fps={fps:.2f}, median={np.median(frames):.2f}, nonzero={(frames>0).mean()*100:.2f}%")

    projection = make_temporal_projection(frames, args.projection)
    mask, thr, raw_mask = foreground_mask_from_projection(
        projection,
        positive_percentile=args.positive_percentile,
        min_area=args.min_area,
        close_radius=args.close_radius,
        dilate_radius=args.dilate_radius,
    )
    raw_bbox = bbox_from_mask(mask)
    crop = expand_bbox(raw_bbox, projection.shape, margin=args.margin, square=args.square, even=True)
    cropped = apply_crop(frames, crop)
    print(f"Detected threshold on positive projection pixels: {thr:.3f}")
    print(f"Crop: y=[{crop[0]},{crop[2]}), x=[{crop[1]},{crop[3]}) -> {cropped.shape[2]} x {cropped.shape[1]} px")

    linked = {
        "input": str(args.input),
        "projection": args.projection,
        "positive_percentile": args.positive_percentile,
        "threshold_value": float(thr),
        "raw_bbox_y0_x0_y1_x1": list(raw_bbox),
        "movie_crop_y0_x0_y1_x1": list(crop),
        "square": bool(args.square),
        "margin": int(args.margin),
        "original_shape_T_H_W": list(frames.shape),
        "cropped_shape_T_H_W": list(cropped.shape),
    }
    with open(outdir / f"{stem}_crop.json", "w", encoding="utf-8") as f:
        json.dump(linked, f, indent=2)

    save_validation(frames, cropped, projection, mask, raw_mask, crop, outdir, stem, warnings)
    if args.preview_only:
        print("Preview only: not saving cropped movie.")
        return

    ext = Path(args.input).suffix.lower()
    avi_path = outdir / f"{stem}_cropped_fixed.avi"
    tif_path = outdir / f"{stem}_cropped_fixed.tif"
    save_movie(cropped, str(avi_path), fps=fps, clip_percentile=args.clip_percentile)
    if args.save_tiff:
        save_movie(cropped, str(tif_path), fps=fps, clip_percentile=args.clip_percentile)


def main():
    parser = argparse.ArgumentParser(description="Robust fixed-size crop for 2D lightsheet AVI/TIFF movies")
    parser.add_argument("--input", required=True, help="Input AVI/TIFF movie")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--projection", default="p95", help="Temporal projection for crop detection: p95, max, mean")
    parser.add_argument("--positive-percentile", type=float, default=70.0, help="Threshold percentile computed on positive projection pixels")
    parser.add_argument("--min-area", type=int, default=5000, help="Minimum connected-component area in pixels")
    parser.add_argument("--close-radius", type=int, default=15, help="Morphological closing radius")
    parser.add_argument("--dilate-radius", type=int, default=25, help="Mask dilation radius before bounding box")
    parser.add_argument("--margin", type=int, default=80, help="Extra crop margin in pixels")
    parser.add_argument("--square", action="store_true", help="Make the crop square")
    parser.add_argument("--clip-percentile", type=float, default=99.9, help="Only used for AVI display scaling if input is >8-bit")
    parser.add_argument("--save-tiff", action="store_true", help="Also save cropped TIFF")
    parser.add_argument("--preview-only", action="store_true", help="Save validation/crop JSON only; do not save cropped movie")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
