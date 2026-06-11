"""
Species-specific image preprocessing operators for HiTMicTools.

Each operator targets a known optical or biological artefact for a given bacterial
species and is applied to the brightfield (BF) channel after standard background
removal.  All operators accept and return float32 2-D arrays.  Input images may be
in raw intensity units; each operator normalises internally where needed and returns
the enhanced image in the same scale as the input.

Operators
---------
normalize_fluorescence      – percentile-clip FL channel to [0, 1]
apply_hessian_tubularness   – Frangi vesselness to bridge hollow-centre artefact
apply_directional_tophat    – multi-orientation white top-hat for elongated filaments
apply_anisotropic_diffusion – Perona-Malik diffusion to smooth halos without blurring
apply_rl_deconvolution      – Richardson-Lucy deconvolution for coccal-cluster blur
apply_phase_congruency      – log-Gabor phase congruency for low-contrast edges
"""

from __future__ import annotations

import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# FL channel normalization
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
# Hessian tubularness (Frangi vesselness)
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

    tubularness = frangi(
        img_norm,
        sigmas=sigmas,
        black_ridges=black_ridges,
    )

    enhanced = img_norm + blend_weight * tubularness
    enhanced = np.clip(enhanced, 0.0, 1.0)
    return (enhanced * (img_max - img_min) + img_min).astype(np.float32)


# ---------------------------------------------------------------------------
# Directional morphological top-hat
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
        for k in range(n_orientations):
            angle_deg = k * 180.0 / n_orientations

            # Build a horizontal line kernel then rotate it
            length = max(scale, 3)
            kernel_1d = np.ones((1, length), dtype=np.uint8)
            center = (length // 2, 0)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            # Destination size: square to accommodate rotation
            dst_size = (length, length)
            kernel_rot = cv2.warpAffine(
                kernel_1d,
                M,
                dst_size,
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
# Anisotropic (Perona-Malik) diffusion
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
    gamma = min(gamma, 0.25)  # Stability bound
    img = image.astype(np.float64)

    for _ in range(n_iter):
        # Finite-difference gradients in 4 cardinal directions
        delta_n = np.roll(img, -1, axis=0) - img
        delta_s = np.roll(img,  1, axis=0) - img
        delta_e = np.roll(img, -1, axis=1) - img
        delta_w = np.roll(img,  1, axis=1) - img

        # Fix border artefacts introduced by roll
        delta_n[-1, :] = 0.0
        delta_s[0,  :] = 0.0
        delta_e[:, -1] = 0.0
        delta_w[:,  0] = 0.0

        # Exponential conduction coefficient
        c_n = np.exp(-(delta_n / kappa) ** 2)
        c_s = np.exp(-(delta_s / kappa) ** 2)
        c_e = np.exp(-(delta_e / kappa) ** 2)
        c_w = np.exp(-(delta_w / kappa) ** 2)

        img += gamma * (
            c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w
        )

    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Richardson-Lucy optical deconvolution
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
    from skimage.restoration import richardson_lucy
    from skimage.filters import gaussian

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    if img_max <= img_min:
        return image_f32

    img_norm = (image_f32 - img_min) / (img_max - img_min)

    # Build a small Gaussian PSF
    psf_radius = max(int(np.ceil(psf_sigma * 3)), 1)
    psf_size = 2 * psf_radius + 1
    psf = np.zeros((psf_size, psf_size), dtype=np.float64)
    psf[psf_radius, psf_radius] = 1.0
    psf = gaussian(psf, sigma=psf_sigma)
    psf /= psf.sum()

    deconvolved = richardson_lucy(
        img_norm.astype(np.float64), psf, num_iter=n_iter, clip=True
    )

    return (deconvolved.astype(np.float32) * (img_max - img_min) + img_min)


# ---------------------------------------------------------------------------
# Phase-congruency mapping (log-Gabor filter bank)
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
    from scipy.fft import fft2, ifft2, ifftshift

    image_f32 = image.astype(np.float32)
    img_min = float(image_f32.min())
    img_max = float(image_f32.max())
    if img_max <= img_min:
        return image_f32

    img_norm = ((image_f32 - img_min) / (img_max - img_min)).astype(np.float64)
    h, w = img_norm.shape

    # Fourier transform of the image
    IMG = fft2(img_norm)

    # Frequency coordinate grids (centered, normalized to [−0.5, 0.5])
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FX, FY = np.meshgrid(fx, fy)
    radius = np.sqrt(FX ** 2 + FY ** 2)
    theta_grid = np.arctan2(FY, FX)

    # Avoid log(0) at DC
    radius[0, 0] = 1.0

    pc_energy = np.zeros((h, w), dtype=np.float64)

    for scale in range(nscale):
        wavelength = min_wavelength * (mult ** scale)
        fo = 1.0 / wavelength

        # Log-Gabor radial component
        log_gabor_rad = np.exp(
            -(np.log(np.maximum(radius, 1e-10) / fo)) ** 2
            / (2.0 * np.log(sigma_onf) ** 2)
        )
        log_gabor_rad[0, 0] = 0.0  # Zero DC component

        for orient in range(norient):
            angle = orient * np.pi / norient

            # Angular spread component
            ds = np.sin(theta_grid) * np.cos(angle) - np.cos(theta_grid) * np.sin(angle)
            dc = np.cos(theta_grid) * np.cos(angle) + np.sin(theta_grid) * np.sin(angle)
            theta_diff = np.arctan2(ds, dc)
            sigma_theta = np.pi / norient / 1.2
            orient_spread = np.exp(-(theta_diff ** 2) / (2.0 * sigma_theta ** 2))

            # Combined log-Gabor filter (already in frequency domain, no ifftshift needed
            # because fftfreq returns an unshifted grid)
            log_gabor_filter = log_gabor_rad * orient_spread

            # Apply filter and compute local energy
            response = ifft2(IMG * log_gabor_filter)
            pc_energy += np.abs(response)

    # Normalize phase congruency map to [0, 1]
    pc_max = pc_energy.max()
    if pc_max > 0:
        pc_energy /= pc_max

    # Blend into original
    enhanced = img_norm + blend_weight * pc_energy
    enhanced = np.clip(enhanced, 0.0, 1.0)
    return (enhanced * (img_max - img_min) + img_min).astype(np.float32)
