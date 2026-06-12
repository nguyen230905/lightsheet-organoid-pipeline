"""
deconvolve.py  â€”  Deconvolution for 2D lightsheet timelapse (.avi or .tif)

Background (from LSTree/example/README.md):
    "Denoised and background-corrected images are then deconvolved with a
     measured PSF using Richardson-Lucy algorithm running on the GPU using
     flowdec."

This script adapts that pipeline for 2D timelapse movies (no Z-stack).
The primary engine is skimage Richardson-Lucy (CPU, easy install).
Optional GPU engine via flowdec is documented below.

Pipeline:
    1. Load movie  (AVI or multi-page TIFF)
    2. Build / load PSF  (Gaussian | theoretical Born-Wolf | measured from file)
    3. Richardson-Lucy deconvolution  (per-frame, 2D)
    4. Validate  (sharpness, contrast, SNR, spectral, visual)
    5. Save outputs + report

PSF options:
    gaussian     â€” simple 2D Gaussian; only needs sigma_px
    theoretical  â€” Born-Wolf model; needs NA, wavelength, pixel size
    file         â€” load a measured PSF TIFF (e.g. from Huygens PSF Distiller
                   or python-microscopy)

Validation metrics:
    Laplacian variance   â€” focus/sharpness measure  (â†‘ good)
    Gradient energy      â€” edge strength            (â†‘ good)
    RMS contrast         â€” intensity spread         (â†‘ good)
    SNR                  â€” signal / noise ratio     (â†‘ good, watch for RL overshoot)
    Tenengrad            â€” Sobel-based sharpness    (â†‘ good)
    Power spectrum ratio â€” high-freq / low-freq     (â†‘ good)
    SSIM vs. input       â€” structural similarity    (too low = over-deconvolved)
    TV (total variation) â€” RL stopping criterion    (â†‘â†‘ = noise amplification)

Usage:
    # Gaussian PSF (simplest)
    python src/lightsheet_pipeline/deconvolution.py --input tests/result/Crop/newbgs025/registered_pos3_bgsub_alpha025_cropped --output tests/result/Deconvolved/ --psf gaussian --sigma 2.0

    # Theoretical PSF (knows your optics)
    python src/lightsheet_pipeline/deconvolution.py --input tests/result/Crop/newbgs025/registered_pos3_bgsub_alpha025_cropped --output tests/result/Deconvolved/ --psf theoretical \\
        --na 0.8 --wavelength 488 --pixel-size 0.26

    # Measured PSF from file
    python src/lightsheet_pipeline/deconvolution.py --input tests/result/Crop/newbgs025/registered_pos3_bgsub_alpha025_cropped --output tests/result/Deconvolved/ --psf file --psf-path my_psf.tif

    # TIFF input / output
    python src/lightsheet_pipeline/deconvolution.py --input tests/result/Crop/newbgs025/registered_pos3_bgsub_alpha025_cropped --output tests/result/Deconvolved/ --psf gaussian --sigma 1.5

    # Control iterations  (default 30; more = sharper but risks ringing)
    python src/lightsheet_pipeline/deconvolution.py --input tests/result/Crop/newbgs025/registered_pos3_bgsub_alpha025_cropped --output tests/result/Deconvolved/ --psf gaussian --sigma 2.0 --iters 50

Dependencies:
    pip install opencv-python scikit-image tifffile matplotlib numpy scipy

Optional GPU engine (LSTree's approach):
    pip install flowdec tensorflow-gpu
    â†’ then pass --engine flowdec
"""

import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# I/O helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SUPPORTED_MOVIE_EXTS = ('.avi', '.tif', '.tiff')


def resolve_input_movie_path(filepath: str) -> Path:
    """
    Resolve the user input into an actual movie file.

    Why this is needed:
    - Your crop step may output a folder.
    - Passing a folder path has no suffix, so Path(path).suffix == ''.
    - The old code raised: ValueError: Unsupported format: ''.

    This function accepts:
    1. Direct file path: movie.avi / movie.tif / movie.tiff
    2. Folder path containing exactly one or more supported movie files
    3. Basename without extension, if basename.avi / basename.tif exists
    """
    p = Path(filepath).expanduser()

    # Case 1: input is a folder. Pick the most likely movie inside.
    if p.is_dir():
        candidates = []
        for ext in SUPPORTED_MOVIE_EXTS:
            candidates.extend(sorted(p.glob(f'*{ext}')))

        if not candidates:
            raise FileNotFoundError(
                f"Input is a directory, but no supported movie file was found inside: {p}\n"
                f"Supported extensions: {SUPPORTED_MOVIE_EXTS}\n"
                f"Run: ls -lh {p}"
            )

        # Prefer cropped/deconvolution-ready files if present, otherwise use first.
        priority_words = ('cropped', 'bgsub', 'bg_clean', 'registered')
        candidates_sorted = sorted(
            candidates,
            key=lambda x: (
                0 if any(w in x.stem.lower() for w in priority_words) else 1,
                len(x.name),
                x.name
            )
        )
        chosen = candidates_sorted[0]
        print(f"  Input directory detected. Using movie file: {chosen}")
        return chosen

    # Case 2: input is already a supported movie file.
    if p.suffix.lower() in SUPPORTED_MOVIE_EXTS:
        if not p.exists():
            raise FileNotFoundError(f"Input movie file does not exist: {p}")
        return p

    # Case 3: user passed basename without extension. Try adding known extensions.
    if p.suffix == '':
        candidates = [p.with_suffix(ext) for ext in SUPPORTED_MOVIE_EXTS]
        candidates = [q for q in candidates if q.exists()]
        if candidates:
            chosen = candidates[0]
            print(f"  No extension detected. Using movie file: {chosen}")
            return chosen

    raise ValueError(
        f"Unsupported input format: '{p.suffix}' for path: {p}\n"
        f"Please pass a file ending with {SUPPORTED_MOVIE_EXTS}, or pass a folder containing one.\n"
        f"Example: --input tests/result/Crop/newbgs025/registered_pos3_bgsub_alpha025_cropped.avi"
    )


def load_movie(filepath: str):
    p = resolve_input_movie_path(filepath)
    suffix = p.suffix.lower()

    if suffix in ('.tif', '.tiff'):
        import tifffile
        arr = tifffile.imread(str(p)).astype(np.float32)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        fps = 1.0

    elif suffix == '.avi':
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 4.0

        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            frames.append(gray.astype(np.float32))
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Could not read any frames from AVI: {p}")
        arr = np.stack(frames)

    else:
        raise ValueError(f"Unsupported format after resolving input: {suffix}")

    print(f"  Loaded: {arr.shape}  range=[{arr.min():.1f}, {arr.max():.1f}]  fps={fps}")
    return arr, fps


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
        wr = cv2.VideoWriter(str(p), fourcc, fps, (W, H), isColor=False)
        for f in u8:
            wr.write(f)
        wr.release()
        print(f"  Saved AVI:  {p}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PSF construction
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_gaussian_psf(sigma_px: float,
                       size: int = None) -> np.ndarray:
    """
    Simple 2D isotropic Gaussian PSF.

    sigma_px : PSF sigma in pixels.
               Rule of thumb for lightsheet: sigma â‰ˆ 0.5 * (lambda / (2*NA)) / pixel_size
    size     : kernel side length (odd number); auto = 6*sigma+1
    """
    if size is None:
        size = int(np.ceil(6 * sigma_px)) | 1      # next odd int
    size = max(size, 5)
    if size % 2 == 0:
        size += 1
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma_px**2))
    psf /= psf.sum()
    return psf


def build_theoretical_psf(na: float,
                           wavelength_nm: float,
                           pixel_size_um: float,
                           size: int = None) -> np.ndarray:
    """
    Theoretical 2D in-plane PSF using Born-Wolf approximation.

    Models the Airy disk pattern: PSF âˆ (2*J1(r) / r)^2
    where r = Ï€ * NA * r_phys / (Î»/2)

    Parameters
    ----------
    na            : numerical aperture
    wavelength_nm : emission wavelength in nm
    pixel_size_um : lateral pixel size in Âµm
    size          : kernel size (auto if None)

    This is the same model used by Huygens / imagej-ops / flowdec's PSF generator.
    """
    from scipy.special import j1

    wavelength_um = wavelength_nm / 1000.0
    # Rayleigh resolution limit in pixels
    r_rayleigh_um = 0.61 * wavelength_um / na
    r_rayleigh_px = r_rayleigh_um / pixel_size_um

    if size is None:
        size = int(np.ceil(8 * r_rayleigh_px)) | 1
    size = max(size, 5)
    if size % 2 == 0:
        size += 1
    half = size // 2

    x = np.arange(-half, half + 1, dtype=np.float64) * pixel_size_um
    xx, yy = np.meshgrid(x, x)
    r_phys = np.sqrt(xx**2 + yy**2)

    # Argument of the Airy function
    k = 2 * np.pi / wavelength_um
    u = k * na * r_phys
    u = np.clip(u, 1e-12, None)

    psf = (2 * j1(u) / u) ** 2
    psf[half, half] = 1.0           # exact center = 1.0
    psf = psf.astype(np.float32)
    psf /= psf.sum()

    print(f"  Born-Wolf PSF: NA={na}  Î»={wavelength_nm}nm  "
          f"px={pixel_size_um}Âµm  "
          f"Rayleigh={r_rayleigh_px:.1f}px  kernel={size}Ã—{size}")
    return psf


def load_psf_from_file(psf_path: str,
                       frame_shape: tuple = None) -> np.ndarray:
    """
    Load a measured 2D PSF from a TIFF file.
    Accepts: 2D array (single PSF), or 3D (Z,Y,X â€” takes central Z-slice for 2D use).

    Measured PSFs can be obtained via:
      - Huygens PSF Distiller (https://svi.nl/Huygens-PSF-Distiller)
      - python-microscopy PSF extraction
      - ImageJ / Fiji bead analysis
    """
    import tifffile
    psf = tifffile.imread(psf_path).astype(np.float32)
    if psf.ndim == 3:
        mid = psf.shape[0] // 2
        psf = psf[mid]
        print(f"  PSF: 3D stack detected, using central Z-slice {mid}")
    assert psf.ndim == 2, f"Expected 2D PSF, got shape {psf.shape}"
    psf = psf.clip(0)
    psf /= psf.sum()
    print(f"  PSF loaded from file: {psf.shape}  path={psf_path}")
    return psf


def save_psf_figure(psf: np.ndarray, output_path: str,
                    title: str = "PSF"):
    """Visualise PSF: image, radial profile, log-scale image."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # PSF image
    axes[0].imshow(psf, cmap='hot', interpolation='nearest')
    axes[0].set_title(f'{title} â€” image'); axes[0].axis('off')

    # Log PSF image
    axes[1].imshow(np.log1p(psf / psf.max() * 1000), cmap='hot',
                   interpolation='nearest')
    axes[1].set_title(f'{title} â€” log scale'); axes[1].axis('off')

    # Radial profile
    cy, cx = np.array(psf.shape) // 2
    max_r  = min(cy, cx)
    radii  = np.arange(max_r)
    profile = []
    for r in radii:
        mask = _annulus_mask(psf.shape, cx, cy, r, r + 1)
        vals = psf[mask]
        profile.append(vals.mean() if vals.size > 0 else 0.0)
    axes[2].plot(radii, profile, lw=2, color='steelblue')
    axes[2].axhline(0, color='k', lw=0.5, ls='--')
    axes[2].set_title('Radial profile'); axes[2].set_xlabel('Radius (px)')
    axes[2].set_ylabel('Intensity'); axes[2].grid(alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()


def _annulus_mask(shape, cx, cy, r_inner, r_outer):
    y, x = np.ogrid[:shape[0], :shape[1]]
    d = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (d >= r_inner) & (d < r_outer)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Richardson-Lucy deconvolution
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def richardson_lucy_skimage(frame: np.ndarray,
                             psf: np.ndarray,
                             n_iter: int = 30,
                             clip: bool = True) -> np.ndarray:
    """
    2D Richardson-Lucy deconvolution via scikit-image.

    The RL algorithm:
        u^(t+1) = u^(t) * (d / (psf âŠ— u^(t))) âŠ› psf_mirror

    where âŠ— is convolution, âŠ› is correlation, d is the observed image,
    u is the current estimate of the true image.
    """
    from skimage.restoration import richardson_lucy
    f = frame.astype(np.float64)
    # Normalise to [0,1] for numerical stability
    fmin, fmax = f.min(), f.max()
    if fmax - fmin < 1e-8:
        return frame.astype(np.float32)
    f_norm = (f - fmin) / (fmax - fmin)
    result = richardson_lucy(f_norm, psf.astype(np.float64),
                             num_iter=n_iter, clip=clip)
    # Rescale back to original range
    return (result * (fmax - fmin) + fmin).astype(np.float32)


def richardson_lucy_manual(frame: np.ndarray,
                            psf: np.ndarray,
                            n_iter: int = 30,
                            tv_weight: float = 0.0) -> tuple:
    """
    Manual RL with per-iteration TV tracking (used for convergence monitoring).

    Returns (deconvolved_frame, tv_curve) where tv_curve[i] is total variation
    after iteration i.  When TV starts rising rapidly â†’ over-iteration.

    tv_weight > 0 adds Total Variation regularisation (reduces ringing).
    """
    f = frame.astype(np.float64)
    fmin, fmax = f.min(), f.max()
    if fmax - fmin < 1e-8:
        return frame.astype(np.float32), []

    d = (f - fmin) / (fmax - fmin) + 1e-9   # avoid division by zero
    psf_f = psf.astype(np.float64)
    psf_f /= psf_f.sum()
    psf_mirror = psf_f[::-1, ::-1]

    u = d.copy()    # initialise estimate as observed image
    tv_curve = []

    for _ in range(n_iter):
        # Convolve current estimate with PSF
        conv = ndimage.convolve(u, psf_f, mode='reflect')
        conv = np.clip(conv, 1e-12, None)
        ratio = d / conv
        # Update estimate
        u = u * ndimage.convolve(ratio, psf_mirror, mode='reflect')
        u = np.clip(u, 0, None)

        # Optional TV regularisation (Rudin-Osher-Fatemi step)
        if tv_weight > 0:
            from skimage.restoration import denoise_tv_chambolle
            u = denoise_tv_chambolle(u, weight=tv_weight * u.max())

        # Track total variation (sum of gradient magnitudes)
        gy, gx = np.gradient(u)
        tv = np.sqrt(gx**2 + gy**2).mean()
        tv_curve.append(tv)

    result = u * (fmax - fmin) + fmin
    return result.astype(np.float32), tv_curve


def deconvolve_movie(frames: np.ndarray,
                     psf: np.ndarray,
                     n_iter: int = 30,
                     tv_weight: float = 0.0,
                     track_convergence: bool = True,
                     verbose: bool = True) -> tuple:
    """
    Deconvolve all frames of a 2D timelapse movie.

    Returns
    -------
    deconvolved : (T, H, W) float32 array
    tv_curves   : list of per-frame TV curves (for convergence analysis)
    """
    T = len(frames)
    deconvolved = np.zeros_like(frames, dtype=np.float32)
    tv_curves   = []

    for i, frame in enumerate(frames):
        if verbose and i % max(1, T // 5) == 0:
            print(f"  Frame {i}/{T-1}...")

        if track_convergence:
            result, tv = richardson_lucy_manual(
                frame, psf, n_iter=n_iter, tv_weight=tv_weight)
            tv_curves.append(tv)
        else:
            result = richardson_lucy_skimage(frame, psf, n_iter=n_iter)
            tv_curves.append([])

        deconvolved[i] = result

    return deconvolved, tv_curves


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Validation metrics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_frame_metrics(frame: np.ndarray) -> dict:
    """
    Compute all quality metrics for a single 2D frame.

    Metrics:
        laplacian_var   Variance of Laplacian  â†’  sharpness/focus measure
                        Ref: Pech-Pacheco et al. (2000)
        tenengrad       Mean squared Sobel gradient  â†’  edge sharpness
        rms_contrast    RMS of (I - mean(I))  â†’  contrast
        snr_db          20*log10(mean / std)  â†’  signal-to-noise ratio
        power_hf_ratio  Power in high frequencies / total power  â†’  detail level
        total_variation Mean gradient magnitude  â†’  RL stopping criterion
    """
    f = frame.astype(np.float64)
    H, W = f.shape

    # Laplacian variance
    lap = cv2.Laplacian(f, cv2.CV_64F)
    laplacian_var = float(lap.var())

    # Tenengrad (Sobel-based focus measure)
    sx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sx**2 + sy**2))

    # RMS contrast
    rms_contrast = float(np.sqrt(np.mean((f - f.mean())**2)))

    # SNR
    mu = f.mean()
    sigma_noise = f.std()
    snr_db = float(20 * np.log10(mu / (sigma_noise + 1e-8) + 1e-8))

    # High-frequency power ratio (FFT-based)
    fft = np.abs(np.fft.fftshift(np.fft.fft2(f)))
    cy, cx = H // 2, W // 2
    r_cutoff = min(H, W) // 6      # "high frequency" = outer 5/6 of spectrum
    y, x = np.ogrid[:H, :W]
    mask_hf = np.sqrt((x - cx)**2 + (y - cy)**2) > r_cutoff
    total_power = float((fft**2).sum())
    hf_power    = float((fft[mask_hf]**2).sum())
    power_hf_ratio = hf_power / (total_power + 1e-12)

    # Total variation
    gy, gx = np.gradient(f)
    total_variation = float(np.sqrt(gx**2 + gy**2).mean())

    return {
        'laplacian_var':  laplacian_var,
        'tenengrad':      tenengrad,
        'rms_contrast':   rms_contrast,
        'snr_db':         snr_db,
        'power_hf_ratio': power_hf_ratio,
        'total_variation':total_variation,
    }


def compute_movie_metrics(frames_before: np.ndarray,
                          frames_after: np.ndarray) -> dict:
    """Compute per-frame metrics for before and after arrays, return summary."""
    metrics_before = [compute_frame_metrics(f) for f in frames_before]
    metrics_after  = [compute_frame_metrics(f) for f in frames_after]

    keys = list(metrics_before[0].keys())
    result = {}
    for k in keys:
        b_vals = [m[k] for m in metrics_before]
        a_vals = [m[k] for m in metrics_after]
        result[f'{k}_before'] = b_vals
        result[f'{k}_after']  = a_vals
        mean_b, mean_a = np.mean(b_vals), np.mean(a_vals)
        pct_change = (mean_a - mean_b) / (abs(mean_b) + 1e-12) * 100
        result[f'{k}_change_pct'] = pct_change
    return result


def print_metrics_table(metrics: dict):
    """Print a clean summary table of before/after metrics."""
    keys = ['laplacian_var', 'tenengrad', 'rms_contrast',
            'snr_db', 'power_hf_ratio', 'total_variation']
    descriptions = {
        'laplacian_var':  'Sharpness (Laplacian var)  [â†‘ better]',
        'tenengrad':      'Edge strength (Tenengrad)  [â†‘ better]',
        'rms_contrast':   'RMS contrast               [â†‘ better]',
        'snr_db':         'SNR (dB)                   [â†‘ better, watch dip]',
        'power_hf_ratio': 'High-freq power ratio      [â†‘ better]',
        'total_variation':'Total variation             [â†‘â†‘ = ringing risk]',
    }
    warnings = {
        'snr_db':         lambda chg: 'âš  SNR drop â€” possibly over-iterated' if chg < -5 else '',
        'total_variation':lambda chg: 'âš  TV large increase â€” ringing possible' if chg > 200 else '',
    }

    print(f"\n{'â”€'*70}")
    print(f"  {'Metric':<42} {'Before':>8}  {'After':>8}  {'Change':>8}")
    print(f"{'â”€'*70}")
    for k in keys:
        b = np.mean(metrics[f'{k}_before'])
        a = np.mean(metrics[f'{k}_after'])
        chg = metrics[f'{k}_change_pct']
        arrow = 'â†‘' if a > b else 'â†“'
        warn  = warnings.get(k, lambda c: '')(chg)
        print(f"  {descriptions[k]:<42} {b:>8.3g}  {a:>8.3g}  "
              f"{chg:>+7.1f}%  {arrow} {warn}")
    print(f"{'â”€'*70}\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Validation figures
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def norm(x):
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + 1e-8)


def save_main_report(frames_before: np.ndarray,
                     frames_after: np.ndarray,
                     metrics: dict,
                     tv_curves: list,
                     n_iter: int,
                     output_path: str):
    """
    4-panel report:
      Row 1: frame thumbnails before / after
      Row 2: all 6 metric trends over frames
      Row 3: TV convergence curves + power spectrum comparison
    """
    n = len(frames_before)
    show_idx = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    t = np.arange(n)

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(f'Deconvolution report  ({n_iter} RL iterations)',
                 fontsize=14, fontweight='bold')

    # â”€â”€ Rows 1â€“2: frame thumbnails â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ns = len(show_idx)
    for row_off, arr, label, color in [(0, frames_before, 'Before', 'tomato'),
                                        (ns, frames_after,  'After',  'steelblue')]:
        for col, idx in enumerate(show_idx):
            ax = fig.add_subplot(6, ns, row_off + col + 1)
            ax.imshow(norm(arr[idx]), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'f{idx}', fontsize=8)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=color)

    # â”€â”€ Row 3â€“4: 6 metric plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metric_keys = [
        ('laplacian_var',   'Sharpness\n(Laplacian var)',  'â†‘', True),
        ('tenengrad',       'Edge strength\n(Tenengrad)',  'â†‘', True),
        ('rms_contrast',    'RMS contrast',                'â†‘', True),
        ('snr_db',          'SNR (dB)',                    'â†‘', False),
        ('power_hf_ratio',  'HF power ratio',             'â†‘', True),
        ('total_variation', 'Total variation\n(watch â†‘â†‘)','~', False),
    ]
    for i, (k, label, direction, good_up) in enumerate(metric_keys):
        ax = fig.add_subplot(6, 3, 10 + i)
        b_vals = metrics[f'{k}_before']
        a_vals = metrics[f'{k}_after']
        ax.plot(t, b_vals, 'o-', color='tomato',    lw=2, ms=4,
                label=f'Before  Î¼={np.mean(b_vals):.3g}')
        ax.plot(t, a_vals, 'o-', color='steelblue', lw=2, ms=4,
                label=f'After   Î¼={np.mean(a_vals):.3g}')
        chg = metrics[f'{k}_change_pct']
        ax.set_title(f'{label}  ({chg:+.1f}%)', fontsize=8)
        ax.set_xlabel('Frame', fontsize=7); ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    # â”€â”€ Row 5â€“6: TV convergence + power spectra â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # TV convergence (representative frames)
    ax_tv = fig.add_subplot(6, 2, 11)
    colors_tv = plt.cm.viridis(np.linspace(0, 1, len(show_idx)))
    for ci, idx in enumerate(show_idx):
        if idx < len(tv_curves) and len(tv_curves[idx]) > 0:
            ax_tv.plot(tv_curves[idx], lw=1.5, color=colors_tv[ci],
                       label=f'f{idx}')
    ax_tv.set_title('RL convergence: Total Variation per iteration\n'
                    '(plateaus = converged, steep rise = over-iterated)',
                    fontsize=8)
    ax_tv.set_xlabel('Iteration', fontsize=7)
    ax_tv.set_ylabel('TV', fontsize=7)
    ax_tv.legend(fontsize=6); ax_tv.grid(alpha=0.3); ax_tv.tick_params(labelsize=7)

    # Radial power spectra (mid frame, before vs after)
    ax_ps = fig.add_subplot(6, 2, 12)
    mid = n // 2
    for arr, label, color in [(frames_before, 'Before', 'tomato'),
                               (frames_after,  'After',  'steelblue')]:
        f_img = arr[mid].astype(np.float64)
        fft   = np.abs(np.fft.fftshift(np.fft.fft2(f_img)))
        H, W  = f_img.shape
        cy, cx= H // 2, W // 2
        max_r = min(cy, cx)
        profile = []
        for r in range(max_r):
            mask = _annulus_mask(fft.shape, cx, cy, r, r + 1)
            vals = fft[mask]
            profile.append(vals.mean() if vals.size > 0 else 0.0)
        freq_axis = np.arange(max_r) / max_r   # normalised spatial freq [0,1]
        ax_ps.semilogy(freq_axis, profile, lw=2, color=color, label=label)
    ax_ps.set_title(f'Radial power spectrum â€” frame {mid}\n'
                    '(deconvolution boosts high frequencies)',
                    fontsize=8)
    ax_ps.set_xlabel('Normalised spatial frequency', fontsize=7)
    ax_ps.set_ylabel('Mean amplitude (log)', fontsize=7)
    ax_ps.legend(fontsize=7); ax_ps.grid(alpha=0.3); ax_ps.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def save_visual_comparison(frames_before: np.ndarray,
                           frames_after: np.ndarray,
                           output_path: str):
    """
    Per-frame visual check:
      Column 1: before | Column 2: after | Column 3: difference
      Column 4: zoom on bright region (shows sharpening detail)
    """
    n = len(frames_before)
    show_idx = sorted(set([0, n//4, n//2, 3*n//4, n-1]))

    fig, axes = plt.subplots(len(show_idx), 4,
                             figsize=(16, 3.5 * len(show_idx)))
    if len(show_idx) == 1:
        axes = axes[np.newaxis]

    for row, idx in enumerate(show_idx):
        fb = frames_before[idx]
        fa = frames_after[idx]
        diff = fa - fb   # signed difference (deconv removes blur = positive at edges)

        # Find brightest region for zoom
        from scipy.ndimage import uniform_filter
        bright_map = uniform_filter(fb, size=50)
        cy, cx = np.unravel_index(np.argmax(bright_map), bright_map.shape)
        zoom = 80
        y0 = max(0, cy - zoom); y1 = min(fb.shape[0], cy + zoom)
        x0 = max(0, cx - zoom); x1 = min(fb.shape[1], cx + zoom)

        axes[row, 0].imshow(norm(fb), cmap='gray')
        axes[row, 0].set_title(f'Before  f{idx}', fontsize=8)

        axes[row, 1].imshow(norm(fa), cmap='gray')
        axes[row, 1].set_title(f'After   f{idx}', fontsize=8)

        im = axes[row, 2].imshow(diff, cmap='RdBu_r',
                                 vmin=-np.percentile(np.abs(diff), 99),
                                 vmax= np.percentile(np.abs(diff), 99))
        axes[row, 2].set_title('Difference (afterâˆ’before)', fontsize=8)
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Side-by-side zoom
        zoom_combined = np.concatenate([
            norm(fb[y0:y1, x0:x1]),
            np.ones((y1-y0, 3)),           # white separator
            norm(fa[y0:y1, x0:x1])
        ], axis=1)
        axes[row, 3].imshow(zoom_combined, cmap='gray')
        axes[row, 3].set_title(f'Zoom: before | after  f{idx}', fontsize=8)

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle('Visual comparison: before vs. after deconvolution',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def save_line_profiles(frames_before: np.ndarray,
                       frames_after: np.ndarray,
                       output_path: str):
    """
    Horizontal line profile through the brightest region.
    Sharp deconvolution produces narrower peaks and steeper edges.
    """
    n = len(frames_before)
    show_idx = sorted(set([0, n//4, n//2, n-1]))

    fig, axes = plt.subplots(1, len(show_idx), figsize=(5 * len(show_idx), 4),
                              sharey=False)
    if len(show_idx) == 1:
        axes = [axes]

    for col, idx in enumerate(show_idx):
        fb = frames_before[idx].astype(np.float64)
        fa = frames_after[idx].astype(np.float64)
        # Row with max mean brightness
        row_bright = int(np.argmax(fb.mean(axis=1)))
        axes[col].plot(norm(fb[row_bright]), lw=1.5, color='tomato',
                       label='Before', alpha=0.85)
        axes[col].plot(norm(fa[row_bright]), lw=1.5, color='steelblue',
                       label='After', alpha=0.85)
        axes[col].set_title(f'Line profile  f{idx}  (row {row_bright})',
                            fontsize=9)
        axes[col].set_xlabel('x pixel'); axes[col].legend(fontsize=7)
        axes[col].grid(alpha=0.3)

    plt.suptitle('Horizontal line profiles â€” sharper edges = effective deconvolution',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def save_iter_sweep(frame: np.ndarray,
                    psf: np.ndarray,
                    iter_list: list,
                    output_path: str):
    """
    Deconvolve ONE representative frame at multiple iteration counts.
    Helps you choose the right number of iterations before running on the full movie.
    """
    n_iters = len(iter_list)
    fig, axes = plt.subplots(2, n_iters + 1, figsize=(4 * (n_iters + 1), 8))

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    axes[0, 0].imshow(norm(frame), cmap='gray')
    axes[0, 0].set_title('Input', fontsize=9); axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    sharpness = []
    tv_vals   = []

    for col, n_iter in enumerate(iter_list, start=1):
        result, tv_curve = richardson_lucy_manual(frame, psf, n_iter=n_iter)
        axes[0, col].imshow(norm(result), cmap='gray')
        axes[0, col].set_title(f'{n_iter} iters', fontsize=9)
        axes[0, col].axis('off')

        # Zoom into centre
        H, W = frame.shape
        cy, cx = H // 2, W // 2; z = 100
        zoom_b = norm(frame[cy-z:cy+z, cx-z:cx+z])
        zoom_a = norm(result[cy-z:cy+z, cx-z:cx+z])
        zoom_side = np.concatenate([zoom_b, np.ones((2*z, 3)), zoom_a], axis=1)
        axes[1, col].imshow(zoom_side, cmap='gray')
        axes[1, col].set_title('zoom: input | output', fontsize=7)
        axes[1, col].axis('off')

        sharp = cv2.Laplacian(result, cv2.CV_64F).var()
        sharpness.append(sharp)
        tv_vals.append(tv_curve[-1] if tv_curve else 0)

    # Inset: sharpness + TV vs iterations
    ax_ins = axes[1, 0]
    ax_ins.set_visible(True)
    ax_ins.plot(iter_list, sharpness, 'o-', color='steelblue', lw=2, ms=5,
                label='Sharpness')
    ax2 = ax_ins.twinx()
    ax2.plot(iter_list, tv_vals, 's--', color='tomato', lw=2, ms=5,
             label='TV')
    ax_ins.set_title('Sharpness & TV vs. #iterations\n'
                     'Choose where sharpness plateaus', fontsize=7)
    ax_ins.set_xlabel('#iterations', fontsize=7)
    ax_ins.set_ylabel('Sharpness', color='steelblue', fontsize=7)
    ax2.set_ylabel('TV', color='tomato', fontsize=7)
    ax_ins.grid(alpha=0.3)
    ax_ins.tick_params(labelsize=7); ax2.tick_params(labelsize=7)

    plt.suptitle('Iteration sweep â€” choose optimal number of RL iterations',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved iteration sweep: {output_path}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run(input_path, output_dir, psf_type, sigma_px, na, wavelength_nm,
        pixel_size_um, psf_path, n_iter, tv_weight,
        iter_sweep, validate, verbose):

    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve early so output extension/stem are correct even if --input is a folder.
    resolved_input = resolve_input_movie_path(input_path)
    ext  = resolved_input.suffix.lower()
    stem = resolved_input.stem

    print(f"\n{'='*60}")
    print(f"  Deconvolution  â€”  Richardson-Lucy")
    print(f"  Input:      {resolved_input}")
    print(f"  PSF:        {psf_type}  |  iterations: {n_iter}")
    print(f"{'='*60}")

    # â”€â”€ 1. Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[1/4] Loading movie...")
    frames, fps = load_movie(str(resolved_input))
    T, H, W = frames.shape

    # â”€â”€ 2. Build PSF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[2/4] Building PSF ({psf_type})...")
    if psf_type == 'gaussian':
        psf = build_gaussian_psf(sigma_px=sigma_px)
        psf_title = f'Gaussian  Ïƒ={sigma_px}px'
    elif psf_type == 'theoretical':
        psf = build_theoretical_psf(na=na,
                                    wavelength_nm=wavelength_nm,
                                    pixel_size_um=pixel_size_um)
        psf_title = f'Born-Wolf  NA={na}  Î»={wavelength_nm}nm'
    elif psf_type == 'file':
        psf = load_psf_from_file(psf_path)
        psf_title = f'Measured PSF  ({Path(psf_path).name})'
    else:
        raise ValueError(f"Unknown PSF type: {psf_type}")

    save_psf_figure(psf, str(out / 'psf.png'), title=psf_title)
    # Save PSF as TIFF for reference
    import tifffile
    tifffile.imwrite(str(out / 'psf.tif'), psf)
    print(f"  PSF kernel size: {psf.shape[0]}Ã—{psf.shape[1]}  sum={psf.sum():.6f}")

    # â”€â”€ Optional: iteration sweep on one representative frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if iter_sweep:
        print("\n[+] Running iteration sweep...")
        ref_frame = frames[len(frames) // 2]
        sweep_iters = [5, 10, 20, 30, 50, 100]
        save_iter_sweep(ref_frame, psf, sweep_iters,
                        str(out / f'{stem}_iter_sweep.png'))
        print("  â†’ Inspect the sweep figure and re-run with your chosen --iters value.")

    # â”€â”€ 3. Deconvolve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[3/4] Deconvolving {T} frames ({n_iter} iterations each)...")
    deconvolved, tv_curves = deconvolve_movie(
        frames, psf,
        n_iter=n_iter,
        tv_weight=tv_weight,
        track_convergence=True,
        verbose=verbose,
    )

    # â”€â”€ 4. Save outputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[4/4] Saving...")
    final_path = str(out / f"{stem}_deconvolved{ext}")
    save_movie(deconvolved, final_path, fps)
    if ext != '.tif':
        tifffile.imwrite(str(out / f"{stem}_deconvolved.tif"),
                         deconvolved.astype(np.float32), imagej=True)

    # â”€â”€ Validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if validate:
        print("\n[+] Computing validation metrics...")
        metrics = compute_movie_metrics(frames, deconvolved)
        print_metrics_table(metrics)

        print("[+] Saving validation figures...")
        save_main_report(frames, deconvolved, metrics, tv_curves, n_iter,
                         str(out / f"{stem}_deconv_report.png"))
        save_visual_comparison(frames, deconvolved,
                               str(out / f"{stem}_deconv_visual.png"))
        save_line_profiles(frames, deconvolved,
                           str(out / f"{stem}_deconv_profiles.png"))

        # Save metrics as CSV
        import csv
        csv_path = str(out / f"{stem}_deconv_metrics.csv")
        keys = ['laplacian_var', 'tenengrad', 'rms_contrast',
                'snr_db', 'power_hf_ratio', 'total_variation']
        with open(csv_path, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            header = ['frame'] + \
                     [f'{k}_before' for k in keys] + \
                     [f'{k}_after'  for k in keys]
            writer.writerow(header)
            for i in range(T):
                row = [i] + \
                      [metrics[f'{k}_before'][i] for k in keys] + \
                      [metrics[f'{k}_after'][i]  for k in keys]
                writer.writerow(row)
        print(f"  Saved metrics CSV: {csv_path}")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Richardson-Lucy deconvolution for 2D lightsheet timelapse"
    )
    # I/O
    parser.add_argument("--input",       required=True,
                        help="Input .avi/.tif, or a folder containing the cropped movie" )
    parser.add_argument("--output",      default="results/",
                        help="Output folder  (default: results/)")
    # PSF
    parser.add_argument("--psf",         default="gaussian",
                        choices=["gaussian", "theoretical", "file"],
                        help="PSF type  (default: gaussian)")
    parser.add_argument("--sigma",       type=float, default=2.0,
                        dest='sigma_px',
                        help="[gaussian] PSF sigma in pixels  (default: 2.0)")
    parser.add_argument("--na",          type=float, default=0.8,
                        help="[theoretical] numerical aperture  (default: 0.8)")
    parser.add_argument("--wavelength",  type=float, default=488.0,
                        dest='wavelength_nm',
                        help="[theoretical] emission wavelength in nm  (default: 488)")
    parser.add_argument("--pixel-size",  type=float, default=0.26,
                        dest='pixel_size_um',
                        help="[theoretical] pixel size in Âµm  (default: 0.26)")
    parser.add_argument("--psf-path",    default=None,
                        help="[file] path to measured PSF TIFF file")
    # RL parameters
    parser.add_argument("--iters",       type=int,   default=30,
                        dest='n_iter',
                        help="Number of RL iterations  (default: 30). "
                             "Tip: run --iter-sweep first to find the optimal value.")
    parser.add_argument("--tv-weight",   type=float, default=0.0,
                        help="Total variation regularisation weight  (default: 0, off). "
                             "Try 1e-4 to 1e-3 if you see ringing artifacts.")
    # Options
    parser.add_argument("--iter-sweep",  action="store_true",
                        help="Run iteration sweep on one frame before deconvolving "
                             "(helps pick --iters)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation figures and metrics")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    run(
        input_path    = args.input,
        output_dir    = args.output,
        psf_type      = args.psf,
        sigma_px      = args.sigma_px,
        na            = args.na,
        wavelength_nm = args.wavelength_nm,
        pixel_size_um = args.pixel_size_um,
        psf_path      = args.psf_path,
        n_iter        = args.n_iter,
        tv_weight     = args.tv_weight,
        iter_sweep    = args.iter_sweep,
        validate      = not args.no_validate,
        verbose       = args.verbose,
    )