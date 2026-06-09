"""
crop.py  —  Organoid Cropping for 2D lightsheet timelapse movies (.avi or .tif)

Strategy (adapted from LSTree crop_gui.py + link_crop_candidates logic):
    1. Detect organoid bounding box in each frame via threshold + largest blob
    2. Link/smooth boxes across time (prevent jitter, track growth)
    3. Compute a single padded crop that contains the organoid in ALL frames
       (so the output movie has a fixed size, matching LSTree's "movie crop")
    4. Apply crop and save

Two modes:
    --mode auto    : fully automatic, no interaction needed  (default)
    --mode preview : saves a figure showing detected crops before applying

Usage:
    # Basic (auto, square crop with 10% margin)
    python crop.py --input denoised.avi

    # AVI or TIFF, custom margin and output
    python crop.py --input denoised.tif --output results/ --margin 50

    # Use a previously saved crop (to apply same crop to multiple positions)
    python crop.py --input pos5_denoised.avi --crop-from pos10_crop.json

    # Preview detected boxes before cropping
    python crop.py --input denoised.avi --mode preview

Dependencies:
    pip install opencv-python scikit-image tifffile matplotlib numpy
"""

import argparse
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers  (same as denoise.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_movie(filepath: str):
    p = Path(filepath)
    if p.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        arr = tifffile.imread(filepath).astype(np.float32)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        fps = 1.0
        print(f"  Loaded TIFF: {arr.shape}")
        return arr, fps
    elif p.suffix.lower() == '.avi':
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, f = cap.read()
            if not ret: break
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            frames.append(gray.astype(np.float32))
        cap.release()
        arr = np.stack(frames)
        print(f"  Loaded AVI:  {arr.shape}  fps={fps}")
        return arr, fps
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")


def save_movie(frames: np.ndarray, path: str, fps: float = 4.0):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        tifffile.imwrite(str(p), frames.astype(np.float32), imagej=True)
        print(f"  Saved TIFF: {p}")
    elif p.suffix.lower() == '.avi':
        T, H, W = frames.shape
        mn, mx = frames.min(), frames.max()
        u8 = ((frames - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(p), fourcc, fps, (W, H), isColor=False)
        for f in u8:
            writer.write(f)
        writer.release()
        print(f"  Saved AVI:  {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Organoid detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_organoid_bbox(frame: np.ndarray,
                         threshold_pct: float = 85.0,
                         min_area_frac: float = 0.001) -> tuple:
    """
    Detect the organoid bounding box in a single 2D frame.

    Strategy:
        1. Percentile threshold to create binary mask
        2. Morphological closing to fill holes
        3. Keep only the largest connected component (the organoid)
        4. Return its bounding box (y0, x0, y1, x1)

    Returns (y0, x0, y1, x1) or None if no object found.
    """
    H, W = frame.shape
    min_area = int(H * W * min_area_frac)

    # Threshold: everything above percentile = foreground
    thresh = np.percentile(frame, threshold_pct)
    binary = (frame > thresh).astype(np.uint8)

    # Morphological closing to fill internal holes (organoid interior can be dark)
    kernel_size = max(11, min(H, W) // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)

    # Label connected components
    labeled, n_labels = ndimage.label(binary)
    if n_labels == 0:
        return None

    # Keep largest component
    sizes = ndimage.sum(binary, labeled, range(1, n_labels + 1))
    largest_label = np.argmax(sizes) + 1
    if sizes[largest_label - 1] < min_area:
        return None

    mask = (labeled == largest_label)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def detect_all_bboxes(frames: np.ndarray,
                      threshold_pct: float = 85.0) -> list:
    """
    Detect organoid bounding box in every frame.
    Returns list of (y0, x0, y1, x1) tuples, None where detection failed.
    """
    bboxes = []
    n = len(frames)
    for i, frame in enumerate(frames):
        if i % max(1, n // 5) == 0:
            print(f"  Detecting organoid: frame {i}/{n-1}")
        bbox = detect_organoid_bbox(frame, threshold_pct=threshold_pct)
        bboxes.append(bbox)

    n_detected = sum(1 for b in bboxes if b is not None)
    print(f"  Detected organoid in {n_detected}/{n} frames")
    return bboxes


# ─────────────────────────────────────────────────────────────────────────────
# Crop linking (temporal smoothing)
# ─────────────────────────────────────────────────────────────────────────────

def link_bboxes(bboxes: list, frames_shape: tuple,
                margin: int = 30,
                smooth_sigma: float = 2.0) -> dict:
    """
    Link detected bounding boxes across time.

    Inspired by LSTree's link_crop_candidates:
        - Interpolates over failed detections
        - Temporally smooths the box coordinates (prevents jitter)
        - Adds a margin around each box
        - Computes a single "movie-wide" crop = union of all per-frame boxes
          (same concept as LSTree's x_start_movie / x_stop_movie)

    Returns a dict with:
        per_frame  : list of (y0, x0, y1, x1) per frame (smoothed + margin)
        movie_crop : (y0, x0, y1, x1) global crop covering all frames
    """
    from scipy.ndimage import gaussian_filter1d

    T, H, W = frames_shape
    n = len(bboxes)

    # Extract coordinate arrays, handle None (failed detections)
    y0_arr = np.array([b[0] if b else np.nan for b in bboxes], dtype=float)
    x0_arr = np.array([b[1] if b else np.nan for b in bboxes], dtype=float)
    y1_arr = np.array([b[2] if b else np.nan for b in bboxes], dtype=float)
    x1_arr = np.array([b[3] if b else np.nan for b in bboxes], dtype=float)

    # Interpolate over NaNs (failed detections)
    for arr in [y0_arr, x0_arr, y1_arr, x1_arr]:
        nans = np.isnan(arr)
        if nans.all():
            continue
        ok_idx = np.where(~nans)[0]
        arr[nans] = np.interp(np.where(nans)[0], ok_idx, arr[ok_idx])

    # Temporal smoothing (Gaussian) — prevents jitter between frames
    if smooth_sigma > 0:
        y0_arr = gaussian_filter1d(y0_arr, smooth_sigma)
        x0_arr = gaussian_filter1d(x0_arr, smooth_sigma)
        y1_arr = gaussian_filter1d(y1_arr, smooth_sigma)
        x1_arr = gaussian_filter1d(x1_arr, smooth_sigma)

    # Add margin and clip to image bounds
    y0_m = np.clip(np.floor(y0_arr - margin).astype(int), 0, H - 1)
    x0_m = np.clip(np.floor(x0_arr - margin).astype(int), 0, W - 1)
    y1_m = np.clip(np.ceil( y1_arr + margin).astype(int), 0, H - 1)
    x1_m = np.clip(np.ceil( x1_arr + margin).astype(int), 0, W - 1)

    per_frame = list(zip(
        y0_m.tolist(), x0_m.tolist(),
        y1_m.tolist(), x1_m.tolist()
    ))

    # Movie-wide crop: union of all per-frame boxes
    # (the organoid is always inside this fixed crop)
    movie_y0 = int(min(y0_m))
    movie_x0 = int(min(x0_m))
    movie_y1 = int(max(y1_m))
    movie_x1 = int(max(x1_m))

    # Make square (optional, improves downstream analysis consistency)
    ch = movie_y1 - movie_y0
    cw = movie_x1 - movie_x0
    if ch != cw:
        side = max(ch, cw)
        cy = (movie_y0 + movie_y1) // 2
        cx = (movie_x0 + movie_x1) // 2
        movie_y0 = max(0, cy - side // 2)
        movie_y1 = min(H, cy + side // 2)
        movie_x0 = max(0, cx - side // 2)
        movie_x1 = min(W, cx + side // 2)

    movie_crop = (movie_y0, movie_x0, movie_y1, movie_x1)

    h_crop = movie_y1 - movie_y0
    w_crop = movie_x1 - movie_x0
    print(f"  Movie-wide crop: y=[{movie_y0},{movie_y1}]  x=[{movie_x0},{movie_x1}]  "
          f"→ {w_crop}×{h_crop} px  "
          f"({100*w_crop/W:.0f}% of original width)")

    return {"per_frame": per_frame, "movie_crop": movie_crop}


# ─────────────────────────────────────────────────────────────────────────────
# Apply crop
# ─────────────────────────────────────────────────────────────────────────────

def apply_movie_crop(frames: np.ndarray, movie_crop: tuple) -> np.ndarray:
    """Apply the fixed movie-wide crop to all frames."""
    y0, x0, y1, x1 = movie_crop
    return frames[:, y0:y1, x0:x1].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Validation / preview figure
# ─────────────────────────────────────────────────────────────────────────────

def save_preview_figure(frames: np.ndarray,
                        raw_bboxes: list,
                        linked: dict,
                        output_path: str):
    """
    Show detected (raw) and linked (smoothed) bounding boxes on representative frames,
    plus the global movie crop rectangle.
    """
    n = len(frames)
    movie_crop = linked["movie_crop"]
    per_frame  = linked["per_frame"]
    show_idx   = sorted(set([0, n//4, n//2, 3*n//4, n-1]))

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    fig, axes = plt.subplots(1, len(show_idx), figsize=(4 * len(show_idx), 5))
    if len(show_idx) == 1:
        axes = [axes]

    for col, idx in enumerate(show_idx):
        ax = axes[col]
        ax.imshow(norm(frames[idx]), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Frame {idx}', fontsize=9)
        ax.axis('off')

        # Raw detection box (per-frame, red dashed)
        if raw_bboxes[idx] is not None:
            y0r, x0r, y1r, x1r = raw_bboxes[idx]
            ax.add_patch(patches.Rectangle(
                (x0r, y0r), x1r - x0r, y1r - y0r,
                edgecolor='red', facecolor='none', lw=1.5, ls='--',
                label='Per-frame (raw)'))

        # Smoothed + margin box (per-frame, orange)
        y0s, x0s, y1s, x1s = per_frame[idx]
        ax.add_patch(patches.Rectangle(
            (x0s, y0s), x1s - x0s, y1s - y0s,
            edgecolor='orange', facecolor='none', lw=1.5, ls='-',
            label='Per-frame (smoothed+margin)'))

        # Movie-wide crop (blue, same in every frame)
        y0m, x0m, y1m, x1m = movie_crop
        ax.add_patch(patches.Rectangle(
            (x0m, y0m), x1m - x0m, y1m - y0m,
            edgecolor='cyan', facecolor='none', lw=2, ls='-',
            label='Movie crop'))

    axes[0].legend(loc='lower left', fontsize=7,
                   handles=[
                       patches.Patch(edgecolor='red',    fc='none', label='Raw detection'),
                       patches.Patch(edgecolor='orange', fc='none', label='Smoothed+margin'),
                       patches.Patch(edgecolor='cyan',   fc='none', label='Movie crop'),
                   ])

    plt.suptitle('Crop detection preview\n'
                 'red=raw  orange=smoothed+margin  cyan=movie crop',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def save_crop_comparison(frames_orig: np.ndarray,
                         frames_cropped: np.ndarray,
                         output_path: str):
    """Before/after strip."""
    n = len(frames_orig)
    show_idx = sorted(set([0, n//4, n//2, 3*n//4, n-1]))

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    fig, axes = plt.subplots(2, len(show_idx), figsize=(4 * len(show_idx), 8))
    for col, idx in enumerate(show_idx):
        axes[0, col].imshow(norm(frames_orig[idx]),    cmap='gray')
        axes[1, col].imshow(norm(frames_cropped[idx]), cmap='gray')
        axes[0, col].set_title(f'f{idx}', fontsize=8)
        axes[0, col].axis('off'); axes[1, col].axis('off')
    axes[0, 0].set_ylabel('Original',  fontsize=9, color='tomato')
    axes[1, 0].set_ylabel('Cropped',   fontsize=9, color='steelblue')
    plt.suptitle(f'Crop result  ({frames_orig.shape[2]}×{frames_orig.shape[1]} '
                 f'→ {frames_cropped.shape[2]}×{frames_cropped.shape[1]})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path, output_dir, margin, threshold_pct, smooth_sigma,
        mode, crop_from, validate):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext  = Path(input_path).suffix.lower()
    stem = Path(input_path).stem

    print(f"\n{'='*55}")
    print(f"  Organoid Cropping")
    print(f"  Input:   {input_path}")
    print(f"  Mode:    {mode}  |  margin={margin} px")
    print(f"{'='*55}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/3] Loading...")
    frames, fps = load_movie(input_path)
    T, H, W = frames.shape

    # ── 2. Detect / load crop ─────────────────────────────────────────────────
    if crop_from:
        # Load a crop JSON saved from a previous run (e.g. from pos10 → apply to pos3)
        print(f"\n[2/3] Loading crop from {crop_from}...")
        with open(crop_from) as f:
            crop_data = json.load(f)
        linked = crop_data
        raw_bboxes = [None] * T
        print(f"  Loaded movie crop: {linked['movie_crop']}")
    else:
        print("\n[2/3] Detecting organoid bounding boxes...")
        raw_bboxes = detect_all_bboxes(frames, threshold_pct=threshold_pct)
        linked = link_bboxes(raw_bboxes, frames.shape,
                             margin=margin, smooth_sigma=smooth_sigma)

    # ── Preview (stop here if mode=preview) ───────────────────────────────────
    if mode == "preview" and not crop_from:
        print("\n[+] Saving preview figure (mode=preview)...")
        preview_path = str(out / f"{stem}_crop_preview.png")
        save_preview_figure(frames, raw_bboxes, linked, preview_path)
        crop_json = str(out / f"{stem}_crop.json")
        with open(crop_json, 'w') as f:
            json.dump(linked, f, indent=2)
        print(f"  Saved crop coordinates: {crop_json}")
        print("\n  [preview mode] Check the figure, then re-run without --mode preview to apply crop.")
        return

    # ── 3. Apply crop ─────────────────────────────────────────────────────────
    print("\n[3/3] Applying crop...")
    movie_crop = tuple(linked["movie_crop"])
    cropped = apply_movie_crop(frames, movie_crop)
    print(f"  Output size: {cropped.shape[2]}×{cropped.shape[1]} px")

    # Save
    final_path = str(out / f"{stem}_cropped{ext}")
    save_movie(cropped, final_path, fps)

    # Always save TIFF copy
    if ext != '.tif':
        import tifffile
        tif_path = str(out / f"{stem}_cropped.tif")
        tifffile.imwrite(tif_path, cropped.astype(np.float32), imagej=True)
        print(f"  Saved TIFF: {tif_path}")

    # Save crop coordinates (reuse for other positions)
    crop_json = str(out / f"{stem}_crop.json")
    with open(crop_json, 'w') as f:
        json.dump(linked, f, indent=2)
    print(f"  Saved crop JSON: {crop_json}")

    # Validation
    if validate and not crop_from:
        print("\n[+] Saving validation figures...")
        save_preview_figure(frames, raw_bboxes, linked,
                            str(out / f"{stem}_crop_detection.png"))
        save_crop_comparison(frames, cropped,
                             str(out / f"{stem}_crop_comparison.png"))

    print(f"\n{'='*55}")
    print("  Done.")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-crop organoid from 2D lightsheet timelapse"
    )
    parser.add_argument("--input",     required=True,
                        help="Input .avi or .tif")
    parser.add_argument("--output",    default="results/",
                        help="Output folder  (default: results/)")
    parser.add_argument("--margin",    type=int,   default=30,
                        help="Margin around organoid in pixels  (default: 30)")
    parser.add_argument("--threshold", type=float, default=85.0,
                        help="Percentile threshold for organoid detection  (default: 85)")
    parser.add_argument("--smooth",    type=float, default=2.0,
                        help="Temporal Gaussian smoothing sigma for crop coordinates "
                             "(default: 2.0, set 0 to disable)")
    parser.add_argument("--mode",      default="auto",
                        choices=["auto", "preview"],
                        help="auto=apply crop directly  preview=save figure only  (default: auto)")
    parser.add_argument("--crop-from", default=None,
                        help="Path to a _crop.json file from a previous run. "
                             "Apply the same crop to this movie (useful for batch consistency).")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation figures")
    args = parser.parse_args()

    run(
        input_path   = args.input,
        output_dir   = args.output,
        margin       = args.margin,
        threshold_pct= args.threshold,
        smooth_sigma = args.smooth,
        mode         = args.mode,
        crop_from    = args.crop_from,
        validate     = not args.no_validate,
    )