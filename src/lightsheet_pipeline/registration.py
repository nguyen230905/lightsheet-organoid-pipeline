"""
register_fixed.py — Robust lightsheet microscopy movie registration

Main fix compared with your original script:
1) Do NOT accumulate phase-correlation shifts frame-to-frame.
2) For organoid/ring-like movies, first align the object center using segmentation.
3) Optional small phase-correlation refinement is only accepted when reliable.
4) Use black borders, not reflected borders, to avoid artificial mirrored signal.

Usage:
    python register_fixed.py --input tests/original/pos10.avi --output tests/result/pos10
    python register_fixed.py --input tests/original/pos10.avi --output tests/result/pos10 --method center
    python register_fixed.py --input tests/original/pos10.avi --output tests/result/pos10 --method center_phase
    python register_fixed.py --input tests/original/pos10.avi --output tests/result/pos10 --method phase_ref
"""

import argparse
from pathlib import Path
import csv
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def load_avi_frames(filepath, gaussian_sigma=0.0):
    """Load AVI as grayscale float32. Do not blur by default; registration preprocesses separately."""
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        raise IOError(f"Cannot open: {filepath}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if gaussian_sigma and gaussian_sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), gaussian_sigma)
        frames.append(gray)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in {filepath}")
    return np.stack(frames, axis=0), fps


# ─────────────────────────────────────────────────────────────────────────────
# 2. BASIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def percentile_norm(frame, p_low=1, p_high=99.7):
    """Normalize one frame to [0, 1] using percentiles; robust to intensity changes."""
    lo, hi = np.percentile(frame, (p_low, p_high))
    out = (frame - lo) / (hi - lo + 1e-8)
    return np.clip(out, 0, 1).astype(np.float32)


def select_reference_frame(frames, method="sharpest"):
    if method == "first":
        return 0
    elif method == "sharpest":
        vals = []
        for f in frames:
            u8 = (percentile_norm(f) * 255).astype(np.uint8)
            vals.append(cv2.Laplacian(u8, cv2.CV_64F).var())
        return int(np.argmax(vals))
    elif method == "median_t":
        median = np.median(frames, axis=0)
        diffs = [np.mean((f - median) ** 2) for f in frames]
        return int(np.argmin(diffs))
    else:
        raise ValueError(f"Unknown reference method: {method}")


def warp_translate(frame, tx, ty, border_value=0):
    """Translate frame by tx, ty pixels. Positive tx moves image right."""
    H, W = frame.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        frame, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3A. ROBUST ORGAN/OID CENTER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_object_center(frame, min_area=80, max_detection_size=768, debug=False):
    """
    Find the center of the main bright object.

    Important speed fix: large lightsheet frames can be 2000+ px wide, so center
    detection is done on a downsampled copy and converted back to full-res pixels.
    """
    H0, W0 = frame.shape

    scale = min(1.0, float(max_detection_size) / float(max(H0, W0)))
    if scale < 1.0:
        small = cv2.resize(frame, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = frame

    H, W = small.shape
    u = percentile_norm(small)
    u8 = (u * 255).astype(np.uint8)

    # Smooth slightly for a clean foreground mask.
    blur = cv2.GaussianBlur(u8, (0, 0), 2)

    # Otsu usually separates bright organoid/ring from black background.
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If Otsu is too strict/loose, use a high-percentile fallback.
    area = int((mask > 0).sum())
    if area < min_area or area > 0.50 * H * W:
        thr = np.percentile(u8, 97)
        mask = (u8 >= thr).astype(np.uint8) * 255

    # Morphological closing connects broken ring segments.
    ksize = max(5, int(round(21 * scale)))
    if ksize % 2 == 0:
        ksize += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    # Keep largest connected component, excluding background.
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        # Final fallback: weighted center of top 2% brightest pixels.
        thr = np.percentile(u8, 98)
        ys, xs = np.nonzero(u8 >= thr)
        if len(xs) == 0:
            return (W0 / 2.0, H0 / 2.0), mask
        weights = u8[ys, xs].astype(np.float32) + 1e-6
        cx = float(np.sum(xs * weights) / np.sum(weights))
        cy = float(np.sum(ys * weights) / np.sum(weights))
        return (cx / scale, cy / scale), mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    comp = (labels == largest_label).astype(np.uint8)

    # Mask centroid is less biased than intensity centroid when one side is brighter.
    m = cv2.moments(comp)
    if abs(m["m00"]) < 1e-8:
        cx, cy = centroids[largest_label]
    else:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

    return (float(cx) / scale, float(cy) / scale), comp * 255


# ─────────────────────────────────────────────────────────────────────────────
# 3B. CENTER-BASED REGISTRATION — RECOMMENDED FOR YOUR MOVIE
# ─────────────────────────────────────────────────────────────────────────────

def center_registration(frames, ref_idx=0, max_step_jump=250, verbose=False):
    """
    Align every frame by moving the detected object center to the reference center.
    This avoids the failure mode where phase-correlation accumulates bad shifts.
    """
    n, H, W = frames.shape

    centers = []
    last_center = None
    for i, f in enumerate(frames):
        c, _ = find_object_center(f)

        # Guard against occasional segmentation failures.
        if last_center is not None:
            jump = np.hypot(c[0] - last_center[0], c[1] - last_center[1])
            if jump > max_step_jump:
                if verbose:
                    print(f"  warning: center jump at frame {i}: {jump:.1f}px; using previous center")
                c = last_center
        centers.append(c)
        last_center = c

    ref_cx, ref_cy = centers[ref_idx]

    registered = np.empty_like(frames)
    shifts = []
    for i, f in enumerate(frames):
        cx, cy = centers[i]
        tx = ref_cx - cx
        ty = ref_cy - cy
        registered[i] = warp_translate(f, tx, ty, border_value=0)
        shifts.append((float(tx), float(ty)))

    return registered, shifts, centers


# ─────────────────────────────────────────────────────────────────────────────
# 3C. SAFE PHASE CORRELATION TO ONE REFERENCE FRAME, NOT CUMULATIVE
# ─────────────────────────────────────────────────────────────────────────────

def prep_for_phase(frame):
    """High-pass + normalize image for phase correlation."""
    u = percentile_norm(frame)
    bg = cv2.GaussianBlur(u, (0, 0), 25)
    hp = u - bg
    hp = (hp - hp.mean()) / (hp.std() + 1e-6)
    return hp.astype(np.float32)


def phase_ref_registration(frames, ref_idx=0, max_shift=200, min_response=0.03, verbose=False):
    """
    Safer version of phase correlation:
    - compares each frame directly to the fixed reference;
    - rejects impossible/low-confidence shifts;
    - does NOT accumulate drift errors.
    """
    n, H, W = frames.shape
    ref = prep_for_phase(frames[ref_idx])
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)

    registered = np.empty_like(frames)
    shifts = []
    responses = []

    for i, f in enumerate(frames):
        if i == ref_idx:
            dx, dy, resp = 0.0, 0.0, 1.0
        else:
            mov = prep_for_phase(f)
            (dx, dy), resp = cv2.phaseCorrelate(ref * win, mov * win)
            mag = float(np.hypot(dx, dy))
            if resp < min_response or mag > max_shift:
                if verbose:
                    print(f"  reject phase frame {i}: dx={dx:.2f}, dy={dy:.2f}, resp={resp:.4f}")
                dx, dy = 0.0, 0.0

        # phaseCorrelate(ref, mov) returns shift of mov relative to ref.
        # To align moving frame back to ref, apply negative shift.
        tx, ty = -dx, -dy
        registered[i] = warp_translate(f, tx, ty, border_value=0)
        shifts.append((float(tx), float(ty)))
        responses.append(float(resp))

    return registered, shifts, responses


def center_phase_registration(frames, ref_idx=0, fine_max_shift=25, min_response=0.03, verbose=False):
    """
    Recommended robust registration:
    1) coarse alignment by object center;
    2) optional tiny phase-correlation refinement;
    3) reject bad fine shifts.
    """
    coarse, coarse_shifts, centers = center_registration(frames, ref_idx=ref_idx, verbose=verbose)

    n, H, W = frames.shape
    ref = prep_for_phase(coarse[ref_idx])
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)

    registered = np.empty_like(frames)
    final_shifts = []
    responses = []

    for i in range(n):
        if i == ref_idx:
            dx, dy, resp = 0.0, 0.0, 1.0
        else:
            mov = prep_for_phase(coarse[i])
            (dx, dy), resp = cv2.phaseCorrelate(ref * win, mov * win)
            mag = float(np.hypot(dx, dy))
            if resp < min_response or mag > fine_max_shift:
                if verbose:
                    print(f"  skip fine phase frame {i}: dx={dx:.2f}, dy={dy:.2f}, resp={resp:.4f}")
                dx, dy = 0.0, 0.0

        # Apply fine correction on top of coarse result.
        registered[i] = warp_translate(coarse[i], -dx, -dy, border_value=0)
        tx = coarse_shifts[i][0] - dx
        ty = coarse_shifts[i][1] - dy
        final_shifts.append((float(tx), float(ty)))
        responses.append(float(resp))

    return registered, final_shifts, centers, responses


# ─────────────────────────────────────────────────────────────────────────────
# 3D. AFFINE REGISTRATION — OPTIONAL, SLOWER
# ─────────────────────────────────────────────────────────────────────────────

def affine_registration(frames, ref_idx=0, verbose=False):
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("pip install SimpleITK  # needed for affine method")

    n, H, W = frames.shape
    ref = frames[ref_idx].astype(np.float32)
    fixed_itk = sitk.GetImageFromArray(ref)
    registered = np.zeros_like(frames)
    transforms = [None] * n

    for i in range(n):
        if verbose:
            print(f"  affine frame {i}/{n-1}")
        moving_itk = sitk.GetImageFromArray(frames[i].astype(np.float32))
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.1)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=200,
            convergenceMinimumValue=1e-6, convergenceWindowSize=10,
        )
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4, 2, 1])
        R.SetSmoothingSigmasPerLevel([2, 1, 0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        init_tx = sitk.CenteredTransformInitializer(
            fixed_itk, moving_itk, sitk.AffineTransform(2),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        R.SetInitialTransform(init_tx, inPlace=False)
        tx = R.Execute(fixed_itk, moving_itk)
        resampled = sitk.Resample(moving_itk, fixed_itk, tx, sitk.sitkLinear, 0.0, moving_itk.GetPixelID())
        registered[i] = sitk.GetArrayFromImage(resampled)
        transforms[i] = tx

    shifts = [(0.0, 0.0)] * n
    return registered, shifts, transforms


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDATION + OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def ncc(a, b):
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def write_registered_avi(frames, path, fps=4.0):
    path = str(path)
    print(f"  Writing AVI: {path}", flush=True)
    n, H, W = frames.shape
    mn, mx = float(frames.min()), float(frames.max())
    frames_u8 = ((frames - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
    print("  Converted frames to uint8", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # Grayscale output is much faster for large microscopy frames.
    out = cv2.VideoWriter(str(path), fourcc, float(fps) if fps > 0 else 4.0, (W, H), isColor=False)
    print(f"  VideoWriter opened: {out.isOpened()}", flush=True)
    if not out.isOpened():
        # Fallback for systems/codecs that do not accept grayscale MJPG.
        out = cv2.VideoWriter(str(path), fourcc, float(fps) if fps > 0 else 4.0, (W, H), isColor=True)
        for f in frames_u8:
            out.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    else:
        for f in frames_u8:
            out.write(f)
    out.release()
    print(f"  Saved: {path}")


def write_registered_tiff(frames, path, mode="uint8_display"):
    """
    Save registered movie as TIFF.

    mode="uint8_display":
        Best for Fiji/ImageJ viewing. Uses robust percentile scaling.
    mode="uint16_display":
        Same idea, but saves 16-bit.
    mode="float32_raw":
        Preserves raw float32 values, but may look black in some viewers
        until you adjust Brightness/Contrast.
    """
    import tifffile
    frames = np.asarray(frames)

    if mode == "float32_raw":
        out = frames.astype(np.float32)

    elif mode == "uint8_display":
        lo, hi = np.percentile(frames, [0.5, 99.8])
        out = np.clip((frames - lo) / (hi - lo + 1e-8), 0, 1)
        out = (out * 255).astype(np.uint8)

    elif mode == "uint16_display":
        lo, hi = np.percentile(frames, [0.5, 99.8])
        out = np.clip((frames - lo) / (hi - lo + 1e-8), 0, 1)
        out = (out * 65535).astype(np.uint16)

    else:
        raise ValueError("mode must be: uint8_display, uint16_display, or float32_raw")

    tifffile.imwrite(
        path,
        out,
        imagej=True,
        metadata={"axes": "TYX"},
    )
    print(f"  Saved: {path}  ({out.dtype}, shape={out.shape})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path, output_dir, method, ref_method, save_tiff, validate, verbose,
        min_response, max_shift, fine_max_shift):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  Lightsheet registration — {Path(input_path).name}")
    print("=" * 60)

    print("\n[1/4] Loading frames...")
    frames, fps = load_avi_frames(input_path)
    n, H, W = frames.shape
    print(f"  {n} frames | {W} x {H} px | {fps:.2f} fps")

    print(f"\n[2/4] Selecting reference frame: {ref_method}")
    ref_idx = select_reference_frame(frames, method=ref_method)
    print(f"  Reference frame: {ref_idx}")

    print(f"\n[3/4] Registering with method='{method}'...")
    if method == "center":
        registered, shifts, centers = center_registration(frames, ref_idx=ref_idx, verbose=verbose)
    elif method == "center_phase":
        registered, shifts, centers, responses = center_phase_registration(
            frames, ref_idx=ref_idx, fine_max_shift=fine_max_shift,
            min_response=min_response, verbose=verbose,
        )
    elif method == "phase_ref":
        registered, shifts, responses = phase_ref_registration(
            frames, ref_idx=ref_idx, max_shift=max_shift,
            min_response=min_response, verbose=verbose,
        )
    elif method == "affine":
        registered, shifts, _ = affine_registration(frames, ref_idx=ref_idx, verbose=verbose)
    else:
        raise ValueError("Unknown method")

    dx = [s[0] for s in shifts]
    dy = [s[1] for s in shifts]
    print(f"  Applied tx range: {min(dx):.2f} to {max(dx):.2f} px")
    print(f"  Applied ty range: {min(dy):.2f} to {max(dy):.2f} px")

    print(f"\n[4/4] Saving outputs to {out}/")
    write_registered_avi(registered, out / "registered.avi", fps=fps)
    save_shifts_csv(shifts, out / "shifts.csv")
    if save_tiff:
        write_registered_tiff(registered, out / "registered.tif")
    if validate:
        save_validation(frames, registered, shifts, out)

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust lightsheet microscopy AVI registration")
    parser.add_argument("--input", required=True, help="Path to input .avi file")
    # Accept both names, because your older command used --outdir.
    parser.add_argument("--output", "--outdir", dest="output", default="results/", help="Output folder")
    parser.add_argument(
        "--method", default="center", choices=["center", "center_phase", "phase_ref", "affine"],
        help="Registration method. Use 'center' first for your organoid movies.",
    )
    parser.add_argument("--ref", default="sharpest", choices=["first", "sharpest", "median_t"])
    parser.add_argument("--no-tiff", action="store_true", help="Skip saving .tif output")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation figures")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--min-response", type=float, default=0.03, help="Minimum phase-correlation response")
    parser.add_argument("--max-shift", type=float, default=200, help="Max direct phase shift accepted, px")
    parser.add_argument("--fine-max-shift", type=float, default=25, help="Max fine correction in center_phase, px")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=args.output,
        method=args.method,
        ref_method=args.ref,
        save_tiff=not args.no_tiff,
        validate=not args.no_validate,
        verbose=args.verbose,
        min_response=args.min_response,
        max_shift=args.max_shift,
        fine_max_shift=args.fine_max_shift,
    )
    cv2.destroyAllWindows()
    sys.exit(0)
