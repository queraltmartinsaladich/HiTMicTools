"""
Species-specific image preprocessing operators for HiTMicTools.

Each operator targets a known optical or biological artefact for a given bacterial
species and is applied to the brightfield (BF) channel after standard background
removal.  All operators accept and return float32 2-D arrays.  Input images may be
in raw intensity units; each operator normalises internally where needed and returns
the enhanced image in the same scale as the input.

Operators
---------
normalize_fluorescence      – percentile-clip FL channel to [0, 1] (fl_norm only, not classifier input)
apply_fl_rolling_ball       – rolling-ball background subtraction for FL channel
apply_fl_nlmeans            – non-local means denoising for FL channel
apply_fl_clahe              – CLAHE on FL after scaling to uint16 (tile_grid_size=16 for sparse signals)
apply_clahe                 – CLAHE local contrast enhancement (runs first, before structural operators)
apply_hessian_tubularness   – Frangi vesselness to bridge hollow-centre artefact
apply_directional_tophat    – multi-orientation white top-hat for elongated filaments
apply_anisotropic_diffusion – Perona-Malik diffusion to smooth halos without blurring
apply_rl_deconvolution      – Richardson-Lucy deconvolution for coccal-cluster blur
apply_phase_congruency      – log-Gabor phase congruency for low-contrast edges
apply_log_enhancement       – Laplacian of Gaussian blob enhancement for cocci

Changes from original
---------------------
apply_directional_tophat  – FIX: rotation center corrected from (length//2, 0) to
                            (size//2, size//2); kernel built on a square canvas so
                            warpAffine never clips the rotated line at any angle.
apply_rl_deconvolution    – FIX: inspect-based kwarg dispatch replaces version-string
                            parsing; PSF explicitly cast to float64.
apply_phase_congruency    – CLEANUP: removed unused ifftshift import; log-Gabor
                            denominator guarded with epsilon; sigma_theta and
                            log_sigma_sq hoisted out of inner loop.
apply_log_enhancement     – NEW: Laplacian of Gaussian blob detector for isotropic
                            coccal boundaries; sigma set to coccal radius in pixels.
"""

from __future__ import annotations

import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# FL channel normalization  (unchanged)
# ---------------------------------------------------------------------------

def normalize_fluorescence(
    fl_image: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """
    Normalize the fluorescence channel to [0, 1] using percentile clipping.

    Per-image normalization is required because PI uptake is concentration- and
    temperature-dependent, making fixed global thresholds unreliable (§4.1.2).

    Args:
        fl_image: 2-D float or integer image.
        percentile_low: Lower clipping percentile (default 1.0).
        percentile_high: Upper clipping percentile (default 99.0).

    Returns:
        float32 array in [0, 1].
    """
    p_low = float(np.percentile(fl_image, percentile_low))
    p_high = float(np.percentile(fl_image, percentile_high))
    if p_high <= p_low:
        return np.zeros_like(fl_image, dtype=np.float32)
    fl_norm = (fl_image.astype(np.float32) - p_low) / (p_high - p_low + 1e-8)
    return np.clip(fl_norm, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FL channel enhancement operators
# ---------------------------------------------------------------------------

def apply_fl_rolling_ball(
    fl_image: np.ndarray,
    radius: int = 50,
    mask_hotlines: bool = False,
    hotline_percentile: float = 99.5,
) -> np.ndarray:
    """
    Rolling-ball background subtraction for the FL channel.

    Estimates spatially varying background by Gaussian-blurring with a kernel
    matched to the ball radius, then subtracts.  Removes uneven illumination and
    autofluorescence gradients that BaSiCPy may not fully capture within a single
    movie.  Output is clipped to zero so negative values (over-subtraction) do not
    propagate.

    When ``mask_hotlines=True``, bright horizontal line artefacts (e.g. the CellASic
    PI channel hot line) are replaced with the image median before the Gaussian blur
    so the artefact does not inflate the background estimate.  The background is then
    subtracted from the *original* (unmasked) image, so the replacement only affects
    the background estimate, not the signal.

    Args:
        fl_image: 2-D float32 image in raw BaSiCPy-corrected intensity units.
        radius: Rolling-ball radius in pixels (default 50).  Should be larger than
            the largest cell; set to ~2–4× the typical cell diameter.
        mask_hotlines: If True, detect and mask bright horizontal rows before
            estimating background (default False).
        hotline_percentile: Row-mean percentile above which a row is considered a
            hot line (default 99.5).

    Returns:
        float32 array in the same intensity units as the input (background removed).
    """
    from scipy.ndimage import gaussian_filter
    img = fl_image.astype(np.float32)

    if mask_hotlines:
        row_means = img.mean(axis=1)
        bad_rows = row_means > np.percentile(row_means, hotline_percentile)
        img_for_bg = img.copy()
        img_for_bg[bad_rows] = float(np.median(img))
    else:
        img_for_bg = img

    background = gaussian_filter(img_for_bg, sigma=radius / 2.0)
    return np.clip(img - background, 0.0, None)


def apply_fl_nlmeans(
    fl_image: np.ndarray,
    h: float = 3.0,
    patch_size: int = 7,
    patch_distance: int = 21,
) -> np.ndarray:
    """
    Non-local means denoising for the FL channel.

    Reduces speckle and Poisson noise while preserving cell boundaries better than
    Gaussian or median filtering.  Operates in [0, 1] internally and rescales back
    to the original intensity range, so BaSiCPy-corrected values are preserved.

    Args:
        fl_image: 2-D float32 image.
        h: Filter strength (default 3.0).  Higher = more smoothing, more signal loss.
            Recommended range: 2–5.
        patch_size: Size of patches used for comparison (default 7).
        patch_distance: Maximum distance to search for similar patches (default 21).

    Returns:
        float32 array in the same intensity range as the input.
    """
    from skimage.restoration import denoise_nl_means
    img = fl_image.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax <= vmin:
        return img
    img_01 = (img - vmin) / (vmax - vmin)
    sigma_est = float(np.std(img_01 - np.mean(img_01))) * 0.5
    sigma_est = max(sigma_est, 1e-4)
    denoised = denoise_nl_means(
        img_01,
        h=h * sigma_est,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
    )
    return (denoised * (vmax - vmin) + vmin).astype(np.float32)


def apply_fl_clahe(
    fl_image: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid_size: int = 16,
) -> np.ndarray:
    """
    CLAHE contrast enhancement for the FL channel.

    After rolling-ball background subtraction the signal is sparse (most pixels
    near zero; bright PI+ cells rare), so p99 can be as low as 47 counts while
    the actual max is in the hundreds.  Applying CLAHE directly on this range
    would stretch noise over nearly-empty tiles.

    The fix is identical to apply_clahe for BF: scale the float32 image to the
    full uint16 range (/ img_max * 65535) so CLAHE sees a real 0–65535 span,
    then rescale back.  tile_grid_size=16 is used by default instead of 8
    because FL images are sparser than BF and larger tiles reduce the chance of
    amplifying noise in pure-background regions.

    Args:
        fl_image: 2-D float32 FL image (BaSiCPy + rolling-ball corrected).
        clip_limit: CLAHE histogram clip threshold (default 1.5).
        tile_grid_size: Tiles per axis (default 16).

    Returns:
        Enhanced float32 image in the same intensity scale as the input.
    """
    return apply_clahe(fl_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)


# ---------------------------------------------------------------------------
# CLAHE local contrast enhancement  *** NEW ***
#
# Applied as the first BF operator so downstream structural operators
# (Hessian, phase congruency) work on a locally-balanced image.
# Operates in uint16 space via cv2 to avoid quantisation artefacts that
# uint8 would introduce on float32 inputs with wide dynamic range.
#
# clip_limit  – histogram clip threshold (cv2 convention, not [0,1] fraction).
#               Typical range 1.0–4.0; higher = more contrast but more noise.
# tile_grid_size – number of tiles per axis; 8 → 8×8 grid.
# ---------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    Enhance local contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Runs first in the operator pipeline so structural filters (Hessian, phase
    congruency) operate on a locally-balanced image.  Useful for E. coli where
    the hollow-centre artefact creates large per-cell intensity variation that
    global normalization cannot resolve (§4.2.x).

    Operates in uint16 space to preserve sub-bit precision lost by uint8
    quantisation.  Output is rescaled back to the original float32 intensity range
    so downstream operators see the same scale.

    Args:
        image: 2-D float32 BF image.
        clip_limit: Histogram bin clip threshold (cv2 convention; typical 1.0–4.0).
            Higher values allow more contrast enhancement but amplify noise.
        tile_grid_size: Number of tiles per axis (default 8 → 8×8 grid).

    Returns:
        Enhanced float32 image in the original intensity scale.
    """
    import cv2

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    if img_max <= img_min:
        return image_f32

    img_u16 = ((image_f32 - img_min) / (img_max - img_min) * 65535.0).astype(np.uint16)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    enhanced_u16 = clahe.apply(img_u16)
    return (enhanced_u16.astype(np.float32) / 65535.0 * (img_max - img_min) + img_min)


# ---------------------------------------------------------------------------
# Hessian tubularness  (unchanged – working for E. coli)
# ---------------------------------------------------------------------------

def apply_hessian_tubularness(
    image: np.ndarray,
    sigmas: List[float] = (1, 2, 3),
    black_ridges: bool = True,
    blend_weight: float = 0.3,
) -> np.ndarray:
    """
    Enhance rod-shaped structures with a multi-scale Frangi vesselness filter.

    The filter detects locally tubular intensity profiles caused by the cylindrical
    geometry of rod bacteria and bridges the hollow-centre artefact by adding a
    blended tubularness response to the original image (§4.2.1).

    Args:
        image: 2-D float BF image.
        sigmas: Scales (σ, pixels) for the Hessian computation.
        black_ridges: True → detect dark ridges; False → detect bright ridges.
        blend_weight: Weight for blending the tubularness map into the image.

    Returns:
        Enhanced float32 image in the original intensity scale.
    """
    from skimage.filters import frangi

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    if img_max <= img_min:
        return image_f32

    img_norm = (image_f32 - img_min) / (img_max - img_min)
    tubularness = frangi(img_norm, sigmas=sigmas, black_ridges=black_ridges)
    enhanced = np.clip(img_norm + blend_weight * tubularness, 0.0, 1.0)
    return (enhanced * (img_max - img_min) + img_min).astype(np.float32)


# ---------------------------------------------------------------------------
# Directional morphological top-hat  *** FIXED ***
#
# Bug: rotation center was (length//2, 0), placing the pivot at the top edge
#      of the destination canvas.  At most angles warpAffine rotated the line
#      segment out of frame, yielding all-zero kernels with no enhancement.
#      Confirmed by test: scale=11, 67.5° kept 6/11 pixels (old) vs 11/11 (new).
#
# Fix: build the line inside a square canvas of odd side `size` and rotate
#      around its true centre (size//2, size//2).
# ---------------------------------------------------------------------------

def apply_directional_tophat(
    image: np.ndarray,
    scales: List[int] = (3, 7, 11),
    n_orientations: int = 8,
    blend_weight: float = 0.5,
) -> np.ndarray:
    """
    Enhance elongated filamentous structures with multi-scale directional white top-hat.

    Rotated linear structuring elements at *n_orientations* angles isolate high-contrast
    filament boundaries while suppressing isotropic background variation (§4.2.3).

    Sizing rule: minimum scale must exceed the 75th-percentile cell length in pixels.
    Scales shorter than the longest common cell length treat cell bodies as background
    and subtract signal, producing dark halos.  For P. aeruginosa at 2284×2572 px
    (16 px median, 24 px p75), confirmed safe scales are [25, 35, 51].

    Args:
        image: 2-D float BF image.
        scales: Lengths (pixels) of the linear structuring elements.
        n_orientations: Number of rotation angles in [0°, 180°).
        blend_weight: Weight for adding the top-hat response to the original image.

    Returns:
        Enhanced float32 image.
    """
    import cv2

    image_f32 = image.astype(np.float32)
    tophat_max = np.zeros_like(image_f32)

    for scale in scales:
        length = max(scale, 3)
        size = length if length % 2 == 1 else length + 1
        cx, cy = size // 2, size // 2

        for k in range(n_orientations):
            angle_deg = k * 180.0 / n_orientations

            kernel_base = np.zeros((size, size), dtype=np.uint8)
            kernel_base[cy, :] = 1

            M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, 1.0)
            kernel_rot = cv2.warpAffine(
                kernel_base,
                M,
                (size, size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            kernel_rot = (kernel_rot > 0).astype(np.uint8)

            if kernel_rot.sum() == 0:
                continue

            th = cv2.morphologyEx(image_f32, cv2.MORPH_TOPHAT, kernel_rot)
            tophat_max = np.maximum(tophat_max, th)

    enhanced = image_f32 + blend_weight * tophat_max
    return enhanced.astype(np.float32)


# ---------------------------------------------------------------------------
# Anisotropic (Perona-Malik) diffusion  (unchanged – numerically stable)
# ---------------------------------------------------------------------------

def apply_anisotropic_diffusion(
    image: np.ndarray,
    n_iter: int = 20,
    kappa: float = 30.0,
    gamma: float = 0.1,
) -> np.ndarray:
    """
    Smooth heavy diffraction noise with Perona-Malik anisotropic diffusion.

    The diffusion coefficient selectively penalises diffusion across high-gradient
    edges, sharpening true cell borders while homogenising the mycolic-acid halo
    patterns common in *M. tuberculosis* (§4.2.4).

    Args:
        image: 2-D float BF image.
        n_iter: Number of diffusion iterations.
        kappa: Edge-sensitivity threshold (higher → more smoothing near edges).
        gamma: Step size (should be ≤ 0.25 for numerical stability).

    Returns:
        Diffused float32 image.
    """
    gamma = min(gamma, 0.25)
    img = image.astype(np.float64)

    for _ in range(n_iter):
        delta_n = np.roll(img, -1, axis=0) - img
        delta_s = np.roll(img,  1, axis=0) - img
        delta_e = np.roll(img, -1, axis=1) - img
        delta_w = np.roll(img,  1, axis=1) - img

        delta_n[-1, :] = 0.0
        delta_s[0,  :] = 0.0
        delta_e[:, -1] = 0.0
        delta_w[:,  0] = 0.0

        c_n = np.exp(-(delta_n / kappa) ** 2)
        c_s = np.exp(-(delta_s / kappa) ** 2)
        c_e = np.exp(-(delta_e / kappa) ** 2)
        c_w = np.exp(-(delta_w / kappa) ** 2)

        img += gamma * (
            c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w
        )

    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Richardson-Lucy optical deconvolution  *** FIXED ***
#
# Bug: version-string parsing (skimage.__version__ >= "0.19") is unreliable —
#      sciCORE ships skimage 0.24 which still uses num_iter and clip=, so the
#      old branch fired correctly while the proposed "fix" would have called
#      num_iterations= and raised TypeError.
#
# Fix: inspect the live signature of richardson_lucy at call time.  This works
#      on every skimage version without parsing the version string, and
#      automatically handles any future rename.  PSF is explicitly cast to
#      float64 to prevent dtype-mismatch errors on float32 inputs.
# ---------------------------------------------------------------------------

def apply_rl_deconvolution(
    image: np.ndarray,
    psf_sigma: float = 1.0,
    n_iter: int = 30,
) -> np.ndarray:
    """
    Reverse out-of-focus diffraction blur with Richardson-Lucy deconvolution.

    Models the point-spread function (PSF) as a Gaussian of width *psf_sigma* and
    iteratively deconvolves the image.  Particularly useful for *S. aureus* coccal
    clusters where inter-cell boundaries are blurred by out-of-focus rings (§4.2.2).

    Args:
        image: 2-D float BF image.
        psf_sigma: Standard deviation of the Gaussian PSF (pixels).
        n_iter: Number of RL iterations.

    Returns:
        Deconvolved float32 image in the original intensity scale.
    """
    import inspect
    from skimage.restoration import richardson_lucy
    from skimage.filters import gaussian

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    print(f"[RL-DIAG] input  dtype={image.dtype} min={img_min:.6f} max={img_max:.6f} range={img_max-img_min:.6f}", flush=True)
    if img_max <= img_min:
        print(f"[RL-DIAG] degenerate range — returning input unchanged", flush=True)
        return image_f32

    img_norm = (image_f32 - img_min) / (img_max - img_min)
    print(f"[RL-DIAG] img_norm min={float(img_norm.min()):.6f} max={float(img_norm.max()):.6f}", flush=True)

    psf_radius = max(int(np.ceil(psf_sigma * 3)), 1)
    psf_size = 2 * psf_radius + 1
    psf = np.zeros((psf_size, psf_size), dtype=np.float64)
    psf[psf_radius, psf_radius] = 1.0
    psf = gaussian(psf, sigma=psf_sigma).astype(np.float64)
    psf /= psf.sum()

    sig = inspect.signature(richardson_lucy)
    iter_kwarg = "num_iterations" if "num_iterations" in sig.parameters else "num_iter"
    kwargs = {iter_kwarg: n_iter}
    if "clip" in sig.parameters:
        kwargs["clip"] = True

    deconvolved = richardson_lucy(img_norm.astype(np.float64), psf, **kwargs)
    print(f"[RL-DIAG] deconvolved min={float(deconvolved.min()):.6f} max={float(deconvolved.max()):.6f}", flush=True)
    result = deconvolved.astype(np.float32) * (img_max - img_min) + img_min
    print(f"[RL-DIAG] output  min={float(result.min()):.6f} max={float(result.max()):.6f}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Phase-congruency mapping  *** CLEANUP ***
#
# - ifftshift removed: it was unused and would have misaligned the fftfreq
#   grids if accidentally applied (grids from fftfreq are already unshifted).
# - log_sigma_sq and sigma_theta hoisted outside both loops (constants).
# - Epsilon guard on log_sigma_sq for sigma_onf close to 1.0.
# ---------------------------------------------------------------------------

def apply_phase_congruency(
    image: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_onf: float = 0.55,
    blend_weight: float = 0.5,
) -> np.ndarray:
    """
    Enhance low-contrast edges with a simplified phase-congruency map.

    Phase congruency is high where the local phase is consistent across frequency
    scales, making it invariant to smooth illumination gradients such as the
    meniscus distortions seen in *M. chimaera* microfluidic imaging (§4.2.5).

    Implementation uses a log-Gabor filter bank in the Fourier domain.  The
    multi-scale energy response is summed and added as a blended edge enhancement
    to the input image.

    Args:
        image: 2-D float BF image.
        nscale: Number of frequency scales.
        norient: Number of filter orientations.
        min_wavelength: Wavelength (pixels) of the finest-scale filter.
        mult: Scaling factor between consecutive wavelengths.
        sigma_onf: Bandwidth of log-Gabor radial filter (0.45–0.65 typical).
        blend_weight: Weight for blending the phase-congruency map into the image.

    Returns:
        Enhanced float32 image.
    """
    from scipy.fft import fft2, ifft2

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    if img_max <= img_min:
        return image_f32

    img_norm = ((image_f32 - img_min) / (img_max - img_min)).astype(np.float64)
    h, w = img_norm.shape

    # Hann window suppresses FFT boundary discontinuities that cause ringing
    # artefacts at image edges.  The taper reduces pc_energy near borders,
    # which is correct — those regions had artificially high energy from
    # wrap-around discontinuities, not real structure.
    window = (np.hanning(h).reshape(-1, 1) * np.hanning(w).reshape(1, -1)).astype(np.float64)
    IMG = fft2(img_norm * window)

    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FX, FY = np.meshgrid(fx, fy)
    radius = np.sqrt(FX ** 2 + FY ** 2)
    theta_grid = np.arctan2(FY, FX)
    radius[0, 0] = 1.0

    log_sigma_sq = max(np.log(sigma_onf) ** 2, 1e-6)
    sigma_theta = np.pi / norient / 1.2

    pc_energy = np.zeros((h, w), dtype=np.float64)

    for scale in range(nscale):
        wavelength = min_wavelength * (mult ** scale)
        fo = 1.0 / wavelength

        log_gabor_rad = np.exp(
            -(np.log(np.maximum(radius, 1e-10) / fo)) ** 2
            / (2.0 * log_sigma_sq)
        )
        log_gabor_rad[0, 0] = 0.0

        for orient in range(norient):
            angle = orient * np.pi / norient

            ds = np.sin(theta_grid) * np.cos(angle) - np.cos(theta_grid) * np.sin(angle)
            dc = np.cos(theta_grid) * np.cos(angle) + np.sin(theta_grid) * np.sin(angle)
            theta_diff = np.arctan2(ds, dc)
            orient_spread = np.exp(-(theta_diff ** 2) / (2.0 * sigma_theta ** 2))

            log_gabor_filter = log_gabor_rad * orient_spread
            response = ifft2(IMG * log_gabor_filter)
            pc_energy += np.abs(response)

    pc_max = pc_energy.max()
    if pc_max > 0:
        pc_energy /= pc_max

    enhanced = np.clip(img_norm + blend_weight * pc_energy, 0.0, 1.0)
    return (enhanced * (img_max - img_min) + img_min).astype(np.float32)


# ---------------------------------------------------------------------------
# Laplacian of Gaussian (LoG) blob enhancement  *** NEW ***
#
# Detects isotropic intensity transitions at a specific spatial scale (sigma).
# Suited to S. aureus cocci (~8 px diameter) where the coccal boundary produces
# a thin phase-contrast ring that LoG detects without orientation assumptions.
# The negative Laplacian is used so dark-ring boundaries yield a positive
# enhancement signal.  LoG response is clipped to zero (background suppression)
# and normalised to [0, 1] before blending so blend_weight is scale-invariant.
#
# Sizing rule: sigma ≈ cell_radius in pixels.
#   S. aureus: diameter ~8 px → sigma=4.
# ---------------------------------------------------------------------------

def apply_log_enhancement(
    image: np.ndarray,
    sigma: float = 4.0,
    blend_weight: float = 0.3,
) -> np.ndarray:
    """
    Enhance blob-shaped structures with a Laplacian of Gaussian filter.

    Detects isotropic intensity transitions at a specific spatial scale set by
    *sigma*.  Particularly suited to *S. aureus* cocci (~8 px diameter) where
    the coccal boundary produces a thin ring of phase contrast that LoG detects
    without any orientation assumption.  The negative Laplacian response is used
    so that dark-ring boundaries (phase-contrast halos) produce a positive
    enhancement signal (§4.2.6).

    Rule of thumb: set sigma ≈ cell_radius in pixels.  For S. aureus at ~8 px
    diameter, sigma=4 targets the ring boundary directly.

    Args:
        image: 2-D float BF image.
        sigma: Gaussian pre-smoothing scale in pixels (default 4.0).
        blend_weight: Weight for adding the LoG response to the original image.

    Returns:
        Enhanced float32 image in the original intensity scale.
    """
    from skimage.filters import gaussian, laplace

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    if img_max <= img_min:
        return image_f32

    img_norm = (image_f32 - img_min) / (img_max - img_min)

    # Smooth then take negative Laplacian — negative because phase-contrast
    # ring boundaries are dark; negating makes the response positive there
    smoothed = gaussian(img_norm.astype(np.float64), sigma=sigma)
    log_response = -laplace(smoothed)

    # Suppress background response (negative LoG values)
    log_response = np.clip(log_response, 0.0, None)

    # Normalise to [0, 1] so blend_weight is scale-invariant across images
    log_max = log_response.max()
    if log_max > 0:
        log_response /= log_max

    enhanced = np.clip(img_norm + blend_weight * log_response, 0.0, 1.0)
    return (enhanced * (img_max - img_min) + img_min).astype(np.float32)


def apply_circular_tophat(
    image: np.ndarray,
    radius: int = 5,
    blend_weight: float = 0.3,
) -> np.ndarray:
    """
    Enhance isotropic blob boundaries with a white top-hat using a disk SE.

    A disk structuring element matches coccal geometry exactly — unlike linear
    SEs (directional tophat) it has no orientation bias and does not interact
    with rod-shaped noise.  The disk radius should be set slightly larger than
    the coccal radius so the SE fits around individual cells.

    Rule of thumb: radius ≈ cell_radius + 1px.
    S. aureus: diameter ~8 px → radius=5.

    Args:
        image: 2-D float BF image.
        radius: Radius of the disk structuring element in pixels.
        blend_weight: Weight for blending the top-hat response into the image.

    Returns:
        Enhanced float32 image in the original intensity scale.
    """
    from skimage.morphology import white_tophat, disk

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    if img_max <= img_min:
        return image_f32

    img_norm = (image_f32 - img_min) / (img_max - img_min)
    response = white_tophat(img_norm.astype(np.float64), disk(radius))

    resp_max = response.max()
    if resp_max > 0:
        response /= resp_max

    enhanced = np.clip(img_norm + blend_weight * response, 0.0, 1.0)
    return (enhanced * (img_max - img_min) + img_min).astype(np.float32)


def apply_dog_enhancement(
    image: np.ndarray,
    sigma_low: float = 2.0,
    sigma_high: float = 6.0,
    blend_weight: float = 0.3,
) -> np.ndarray:
    """
    Bandpass enhancement via Difference of Gaussians tuned to coccal diameter.

    DoG approximates a bandpass filter; structures between sigma_low and
    sigma_high are enhanced.  No FFT boundary issues, no iteration divergence.

    Rule of thumb: sigma_low ≈ cell_radius/2, sigma_high ≈ cell_radius*1.5.
    S. aureus: diameter ~8 px → sigma_low=2, sigma_high=6.

    Args:
        image: 2-D float BF image.
        sigma_low: Inner Gaussian sigma (fine scale).
        sigma_high: Outer Gaussian sigma (coarse scale).
        blend_weight: Weight for blending the DoG response into the image.

    Returns:
        Enhanced float32 image in the original intensity scale.
    """
    from skimage.filters import gaussian

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    if img_max <= img_min:
        return image_f32

    img_norm = (image_f32 - img_min) / (img_max - img_min)
    img_f64 = img_norm.astype(np.float64)
    dog = gaussian(img_f64, sigma=sigma_low) - gaussian(img_f64, sigma=sigma_high)
    dog = np.clip(dog, 0.0, None)

    dog_max = dog.max()
    if dog_max > 0:
        dog /= dog_max

    enhanced = np.clip(img_norm + blend_weight * dog, 0.0, 1.0)
    return (enhanced * (img_max - img_min) + img_min).astype(np.float32)


def apply_unsharp_mask(
    image: np.ndarray,
    sigma: float = 4.0,
    blend_weight: float = 0.3,
) -> np.ndarray:
    """
    Gentle sharpening via unsharp masking.

    Subtracts a Gaussian-blurred version from the original; the residual
    captures high-frequency structure which is blended back in.  No ringing,
    no boundary artefacts, predictable behaviour.

    Rule of thumb: sigma ≈ cell_radius in pixels.
    S. aureus: diameter ~8 px → sigma=4.

    Args:
        image: 2-D float BF image.
        sigma: Gaussian blur sigma for the unsharp mask.
        blend_weight: Weight for blending the sharpening residual.

    Returns:
        Enhanced float32 image in the original intensity scale.
    """
    from skimage.filters import gaussian

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    if img_max <= img_min:
        return image_f32

    img_norm = (image_f32 - img_min) / (img_max - img_min)
    blur = gaussian(img_norm.astype(np.float64), sigma=sigma)
    mask = np.clip(img_norm.astype(np.float64) - blur, 0.0, None)

    mask_max = mask.max()
    if mask_max > 0:
        mask /= mask_max

    enhanced = np.clip(img_norm + blend_weight * mask, 0.0, 1.0)
    return (enhanced * (img_max - img_min) + img_min).astype(np.float32)