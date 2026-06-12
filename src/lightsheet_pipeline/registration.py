#!/usr/bin/env python3
"""
registration_fixed.py — robust 2-D registration for lightsheet AVI movies.

This version is designed for preview/projection AVI movies of organoids or other
bright objects on a dark background. It deliberately separates:

1. registration images: normalized / high-pass images used only to estimate shifts
2. output images: the original raw frames, translated with the estimated shifts

That separation prevents the common failure where the registered AVI is written
from preprocessed phase-correlation images, producing a grey background and
black/white inverted object.
n
Examples
--------
python registration_fixed.py --input pos3.avi --output results/pos3 --method center
python registration_fixed.py --input pos3.avi --output results/pos3 --method center_phase --ref first
python registration_fixed.py --input pos3.avi --output results/pos3 --method phase_ref --no-tiff
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


Array = np.ndarray
Method = Literal["center", "center_phase", "phase_ref"]
RefMethod = Literal["first", "sharpest", "median_t"]


@dataclass
class Movie:
    frames: Array       # shape: T, Y, X; dtype preserved as much as possible
    fps: float
    source_dtype: np.dtype


@dataclass
class RegistrationResult:
    registered: Array
    shifts_xy: list[tuple[float, float]]
    centers_xy: list[tuple[float, float]] | None = None
    phase_responses: list[float] | None = None
    ref_idx: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_avi_gray(path: str | Path, max_frames: int | None = None) -> Movie:
    """Read an AVI movie as grayscale frames.

    AVI files are usually 8-bit already. We keep them as uint8 to avoid changing
    intensity. Registration functions cast to float only when needed.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open AVI: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 4.0
    frames: list[Array] = []
    n_bad = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.copy())
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        raise ValueError(f"No readable frames found in {path}")

    arr = np.stack(frames, axis=0)
    return Movie(frames=arr, fps=fps, source_dtype=arr.dtype)


def to_uint8_display(frames: Array, preserve_uint8: bool = True) -> Array:
    """Convert frames to uint8 for video display without destroying contrast.

    For an AVI input, frames are already uint8, so the safest display output is
    to preserve them. For float/uint16 inputs, robust percentile scaling is used.
    """
    frames = np.asarray(frames)
    if preserve_uint8 and frames.dtype == np.uint8:
        return frames.copy()

    frames_f = frames.astype(np.float32, copy=False)
    finite = np.isfinite(frames_f)
    if not finite.any():
        return np.zeros(frames.shape, dtype=np.uint8)

    # For microscopy on black background, keep zero as zero whenever possible.
    positive = frames_f[(frames_f > 0) & finite]
    if positive.size > 100:
        lo = 0.0 if frames_f.min() >= 0 else float(np.percentile(frames_f[finite], 0.5))
        hi = float(np.percentile(positive, 99.8))
    else:
        lo, hi = np.percentile(frames_f[finite], [0.5, 99.8])

    if hi <= lo + 1e-8:
        hi = float(frames_f[finite].max())
        lo = float(frames_f[finite].min())

    out = np.clip((frames_f - lo) / (hi - lo + 1e-8), 0, 1)
    return (out * 255).astype(np.uint8)


def write_avi(frames: Array, path: str | Path, fps: float, codec: str = "MJPG") -> None:
    """Write grayscale data as a robust BGR AVI.

    OpenCV/codec combinations can be inconsistent for single-channel MJPG.
    Writing BGR is slightly larger but is readable by more viewers and avoids
    grey/pseudo-colour display artefacts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_u8 = to_uint8_display(frames, preserve_uint8=True)
    t, h, w = frames_u8.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps) if fps > 0 else 4.0, (w, h), isColor=True)
    if not writer.isOpened():
        raise OSError(f"Could not open VideoWriter for {path}; try --codec FFV1 or --codec XVID")

    for fr in frames_u8:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
    writer.release()


def write_tiff(frames: Array, path: str | Path, mode: Literal["uint8", "uint16", "raw"] = "uint8") -> None:
    """Optional TIFF export for Fiji/ImageJ or downstream quantitative work."""
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "raw":
        out = frames
    elif mode == "uint8":
        out = to_uint8_display(frames, preserve_uint8=True)
    elif mode == "uint16":
        frames_f = frames.astype(np.float32, copy=False)
        positive = frames_f[frames_f > 0]
        lo = 0.0 if frames_f.min() >= 0 else float(np.percentile(frames_f, 0.5))
        hi = float(np.percentile(positive, 99.8)) if positive.size else float(frames_f.max())
        out = np.clip((frames_f - lo) / (hi - lo + 1e-8), 0, 1)
        out = (out * 65535).astype(np.uint16)
    else:
        raise ValueError("mode must be 'uint8', 'uint16', or 'raw'")

    tifffile.imwrite(str(path), out, imagej=True, metadata={"axes": "TYX"})


def save_shifts_csv(path: str | Path, shifts: Iterable[tuple[float, float]], centers=None, responses=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["frame", "tx_px", "ty_px"]
        if centers is not None:
            header += ["center_x_px", "center_y_px"]
        if responses is not None:
            header += ["phase_response"]
        writer.writerow(header)
        for i, (tx, ty) in enumerate(shifts):
            row = [i, float(tx), float(ty)]
            if centers is not None:
                row += [float(centers[i][0]), float(centers[i][1])]
            if responses is not None:
                row += [float(responses[i])]
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Registration utilities
# ─────────────────────────────────────────────────────────────────────────────

def percentile_norm(frame: Array, p_low: float = 1.0, p_high: float = 99.8) -> Array:
    frame_f = frame.astype(np.float32, copy=False)
    lo, hi = np.percentile(frame_f, [p_low, p_high])
    return np.clip((frame_f - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def select_reference_frame(frames: Array, method: RefMethod = "first") -> int:
    if method == "first":
        return 0
    if method == "median_t":
        median = np.median(frames.astype(np.float32), axis=0)
        diffs = [float(np.mean((f.astype(np.float32) - median) ** 2)) for f in frames]
        return int(np.argmin(diffs))
    if method == "sharpest":
        scores = []
        for f in frames:
            u8 = (percentile_norm(f) * 255).astype(np.uint8)
            scores.append(float(cv2.Laplacian(u8, cv2.CV_64F).var()))
        return int(np.argmax(scores))
    raise ValueError(f"Unknown reference method: {method}")


def warp_translate(frame: Array, tx: float, ty: float, border_value: int | float = 0) -> Array:
    """Translate image by tx, ty pixels. Positive tx moves right; positive ty moves down."""
    h, w = frame.shape
    mat = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )


def find_object_center(frame: Array, max_detection_size: int = 768, min_area_fraction: float = 1e-5) -> tuple[tuple[float, float], Array]:
    """Estimate the centre of the main bright object using a downsampled mask."""
    h0, w0 = frame.shape
    scale = min(1.0, float(max_detection_size) / max(h0, w0))
    if scale < 1.0:
        small = cv2.resize(frame, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = frame

    h, w = small.shape
    u8 = (percentile_norm(small) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (0, 0), 2)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    area = int(np.count_nonzero(mask))
    min_area = max(20, int(min_area_fraction * h * w))
    if area < min_area or area > int(0.50 * h * w):
        thr = np.percentile(u8, 97.0)
        mask = (u8 >= thr).astype(np.uint8) * 255

    # Use a moderate kernel so the ring is connected but not expanded too much.
    ksize = max(5, int(round(19 * scale)))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        # Fallback: weighted centroid of very bright pixels.
        thr = np.percentile(u8, 98.0)
        yy, xx = np.nonzero(u8 >= thr)
        if xx.size == 0:
            return (w0 / 2.0, h0 / 2.0), mask
        weights = u8[yy, xx].astype(np.float32) + 1e-6
        cx = float(np.sum(xx * weights) / np.sum(weights))
        cy = float(np.sum(yy * weights) / np.sum(weights))
        return (cx / scale, cy / scale), mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    comp = (labels == largest).astype(np.uint8)
    moments = cv2.moments(comp)
    if abs(moments["m00"]) > 1e-8:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = centroids[largest]
    return (float(cx) / scale, float(cy) / scale), comp * 255


def center_registration(frames: Array, ref_idx: int, max_center_jump: float = 300.0, verbose: bool = False) -> RegistrationResult:
    """Coarse registration by aligning segmented object centres to the reference centre."""
    centers: list[tuple[float, float]] = []
    last_good: tuple[float, float] | None = None

    for i, frame in enumerate(frames):
        center, _ = find_object_center(frame)
        if last_good is not None:
            jump = float(np.hypot(center[0] - last_good[0], center[1] - last_good[1]))
            if jump > max_center_jump:
                if verbose:
                    print(f"warning: frame {i} centre jump {jump:.1f}px; using previous centre")
                center = last_good
        centers.append(center)
        last_good = center

    ref_x, ref_y = centers[ref_idx]
    registered = np.empty_like(frames)
    shifts: list[tuple[float, float]] = []
    for i, frame in enumerate(frames):
        cx, cy = centers[i]
        tx = ref_x - cx
        ty = ref_y - cy
        registered[i] = warp_translate(frame, tx, ty, border_value=0)
        shifts.append((float(tx), float(ty)))

    return RegistrationResult(registered=registered, shifts_xy=shifts, centers_xy=centers, ref_idx=ref_idx)


def prep_for_phase(frame: Array, blur_sigma: float = 25.0) -> Array:
    """Create a normalized high-pass image used only for shift estimation."""
    u = percentile_norm(frame)
    bg = cv2.GaussianBlur(u, (0, 0), blur_sigma)
    hp = u - bg
    hp = hp - float(hp.mean())
    hp = hp / (float(hp.std()) + 1e-6)
    return hp.astype(np.float32)


def phase_ref_registration(
    frames: Array,
    ref_idx: int,
    max_shift: float = 200.0,
    min_response: float = 0.03,
    verbose: bool = False,
) -> RegistrationResult:
    """Register each frame directly to one fixed reference using phase correlation."""
    t, h, w = frames.shape
    ref = prep_for_phase(frames[ref_idx])
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)

    registered = np.empty_like(frames)
    shifts: list[tuple[float, float]] = []
    responses: list[float] = []

    for i, frame in enumerate(frames):
        if i == ref_idx:
            dx = dy = 0.0
            response = 1.0
        else:
            moving = prep_for_phase(frame)
            (dx, dy), response = cv2.phaseCorrelate(ref * window, moving * window)
            mag = float(np.hypot(dx, dy))
            if response < min_response or mag > max_shift:
                if verbose:
                    print(f"reject phase frame {i}: dx={dx:.2f}, dy={dy:.2f}, response={response:.4f}")
                dx = dy = 0.0
        # phaseCorrelate(ref, moving) estimates how moving is shifted vs ref;
        # apply the negative translation to align moving to ref.
        tx, ty = -float(dx), -float(dy)
        registered[i] = warp_translate(frame, tx, ty, border_value=0)
        shifts.append((tx, ty))
        responses.append(float(response))

    return RegistrationResult(registered=registered, shifts_xy=shifts, phase_responses=responses, ref_idx=ref_idx)


def center_phase_registration(
    frames: Array,
    ref_idx: int,
    fine_max_shift: float = 25.0,
    min_response: float = 0.03,
    verbose: bool = False,
) -> RegistrationResult:
    """Recommended for your movie: center alignment plus safe small phase refinement."""
    coarse = center_registration(frames, ref_idx=ref_idx, verbose=verbose)
    t, h, w = frames.shape
    ref = prep_for_phase(coarse.registered[ref_idx])
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)

    registered = np.empty_like(frames)
    final_shifts: list[tuple[float, float]] = []
    responses: list[float] = []

    for i, frame in enumerate(coarse.registered):
        if i == ref_idx:
            dx = dy = 0.0
            response = 1.0
        else:
            moving = prep_for_phase(frame)
            (dx, dy), response = cv2.phaseCorrelate(ref * window, moving * window)
            mag = float(np.hypot(dx, dy))
            if response < min_response or mag > fine_max_shift:
                if verbose:
                    print(f"skip fine phase frame {i}: dx={dx:.2f}, dy={dy:.2f}, response={response:.4f}")
                dx = dy = 0.0
        registered[i] = warp_translate(frame, -float(dx), -float(dy), border_value=0)
        tx = coarse.shifts_xy[i][0] - float(dx)
        ty = coarse.shifts_xy[i][1] - float(dy)
        final_shifts.append((tx, ty))
        responses.append(float(response))

    return RegistrationResult(
        registered=registered,
        shifts_xy=final_shifts,
        centers_xy=coarse.centers_xy,
        phase_responses=responses,
        ref_idx=ref_idx,
    )


def register_movie(frames: Array, method: Method, ref_method: RefMethod, **kwargs) -> RegistrationResult:
    ref_idx = select_reference_frame(frames, ref_method)
    if method == "center":
        return center_registration(frames, ref_idx=ref_idx, max_center_jump=kwargs.get("max_center_jump", 300.0), verbose=kwargs.get("verbose", False))
    if method == "phase_ref":
        return phase_ref_registration(
            frames,
            ref_idx=ref_idx,
            max_shift=kwargs.get("max_shift", 200.0),
            min_response=kwargs.get("min_response", 0.03),
            verbose=kwargs.get("verbose", False),
        )
    if method == "center_phase":
        return center_phase_registration(
            frames,
            ref_idx=ref_idx,
            fine_max_shift=kwargs.get("fine_max_shift", 25.0),
            min_response=kwargs.get("min_response", 0.03),
            verbose=kwargs.get("verbose", False),
        )
    raise ValueError(f"Unknown method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def object_centers_for_validation(frames: Array) -> Array:
    centers = [find_object_center(f)[0] for f in frames]
    return np.asarray(centers, dtype=np.float32)


def save_validation(movie: Movie, result: RegistrationResult, outdir: str | Path) -> None:
    if plt is None:
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frames = movie.frames
    registered = result.registered
    t = frames.shape[0]
    idxs = sorted(set([0, t // 2, t - 1, result.ref_idx]))

    def disp(x: Array) -> Array:
        return to_uint8_display(x[None, ...], preserve_uint8=True)[0]

    fig, axes = plt.subplots(len(idxs), 2, figsize=(8, 3.5 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    for r, idx in enumerate(idxs):
        axes[r, 0].imshow(disp(frames[idx]), cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title(f"Original frame {idx}")
        axes[r, 0].axis("off")
        axes[r, 1].imshow(disp(registered[idx]), cmap="gray", vmin=0, vmax=255)
        axes[r, 1].set_title(f"Registered frame {idx}")
        axes[r, 1].axis("off")
    fig.tight_layout()
    fig.savefig(outdir / "validation_before_after.png", dpi=150)
    plt.close(fig)

    shifts = np.asarray(result.shifts_xy, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(shifts[:, 0], label="tx")
    ax.plot(shifts[:, 1], label="ty")
    ax.axvline(result.ref_idx, linestyle="--", linewidth=1, label="reference")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Applied translation (px)")
    ax.set_title("Estimated registration shifts")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "validation_shifts.png", dpi=150)
    plt.close(fig)

    before_c = object_centers_for_validation(frames)
    after_c = object_centers_for_validation(registered)
    before_drift = np.linalg.norm(before_c - before_c[result.ref_idx], axis=1)
    after_drift = np.linalg.norm(after_c - after_c[result.ref_idx], axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(before_drift, label="before")
    ax.plot(after_drift, label="after")
    ax.axvline(result.ref_idx, linestyle="--", linewidth=1, label="reference")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Object centre displacement (px)")
    ax.set_title("Registration quality check")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "validation_center_drift.png", dpi=150)
    plt.close(fig)

    with (outdir / "validation_report.txt").open("w", encoding="utf-8") as fh:
        fh.write("Registration validation report\n")
        fh.write("==============================\n")
        fh.write(f"frames: {t}\n")
        fh.write(f"reference frame: {result.ref_idx}\n")
        fh.write(f"input dtype: {movie.source_dtype}\n")
        fh.write(f"input shape TYX: {frames.shape}\n")
        fh.write(f"mean before centre drift px: {float(before_drift.mean()):.3f}\n")
        fh.write(f"mean after centre drift px: {float(after_drift.mean()):.3f}\n")
        fh.write(f"max before centre drift px: {float(before_drift.max()):.3f}\n")
        fh.write(f"max after centre drift px: {float(after_drift.max()):.3f}\n")
        fh.write(f"tx range px: {float(shifts[:,0].min()):.3f} to {float(shifts[:,0].max()):.3f}\n")
        fh.write(f"ty range px: {float(shifts[:,1].min()):.3f} to {float(shifts[:,1].max()):.3f}\n")
        if result.phase_responses is not None:
            responses = np.asarray(result.phase_responses, dtype=float)
            fh.write(f"phase response mean: {float(responses.mean()):.5f}\n")
            fh.write(f"phase response min: {float(responses.min()):.5f}\n")


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    movie = load_avi_gray(args.input, max_frames=args.max_frames)
    print(f"Loaded {movie.frames.shape[0]} frames, shape={movie.frames.shape[2]}x{movie.frames.shape[1]}, fps={movie.fps:.3f}, dtype={movie.frames.dtype}")

    result = register_movie(
        movie.frames,
        method=args.method,
        ref_method=args.ref,
        min_response=args.min_response,
        max_shift=args.max_shift,
        fine_max_shift=args.fine_max_shift,
        max_center_jump=args.max_center_jump,
        verbose=args.verbose,
    )
    print(f"Reference frame: {result.ref_idx}")
    shifts = np.asarray(result.shifts_xy)
    print(f"Applied tx range: {shifts[:,0].min():.2f} to {shifts[:,0].max():.2f} px")
    print(f"Applied ty range: {shifts[:,1].min():.2f} to {shifts[:,1].max():.2f} px")

    avi_path = outdir / "registered_fixed.avi"
    write_avi(result.registered, avi_path, fps=movie.fps, codec=args.codec)
    print(f"Saved {avi_path}")

    csv_path = outdir / "shifts.csv"
    save_shifts_csv(csv_path, result.shifts_xy, centers=result.centers_xy, responses=result.phase_responses)
    print(f"Saved {csv_path}")

    if not args.no_tiff:
        tiff_path = outdir / "registered_fixed.tif"
        write_tiff(result.registered, tiff_path, mode=args.tiff_mode)
        print(f"Saved {tiff_path}")

    if not args.no_validate:
        save_validation(movie, result, outdir)
        print(f"Saved validation outputs in {outdir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust registration for lightsheet AVI preview/projection movies")
    p.add_argument("--input", required=True, help="Input AVI movie")
    p.add_argument("--output", "--outdir", default="registration_results", help="Output directory")
    p.add_argument("--method", choices=["center", "center_phase", "phase_ref"], default="center", help="Registration strategy")
    p.add_argument("--ref", choices=["first", "sharpest", "median_t"], default="first", help="Reference frame selection")
    p.add_argument("--codec", default="MJPG", help="AVI codec; MJPG is portable, FFV1 is lossless but larger")
    p.add_argument("--min-response", type=float, default=0.03, help="Minimum accepted phase-correlation response")
    p.add_argument("--max-shift", type=float, default=200.0, help="Maximum accepted direct phase shift in pixels")
    p.add_argument("--fine-max-shift", type=float, default=25.0, help="Maximum accepted fine phase shift after center alignment")
    p.add_argument("--max-center-jump", type=float, default=300.0, help="Reject centre detections that jump more than this between consecutive frames")
    p.add_argument("--max-frames", type=int, default=None, help="Optional debugging limit")
    p.add_argument("--no-tiff", action="store_true", help="Skip TIFF output")
    p.add_argument("--tiff-mode", choices=["uint8", "uint16", "raw"], default="uint8")
    p.add_argument("--no-validate", action="store_true", help="Skip validation plots/report")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
