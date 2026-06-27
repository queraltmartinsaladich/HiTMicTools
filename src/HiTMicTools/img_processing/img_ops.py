import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Union, Tuple

from HiTMicTools.utils import (
    round_to_odd,
    unit_converter,
)


def detect_and_fix_well(
    image: np.ndarray,
    darkness_threshold_factor: float = 0.4,
    border_sample_interval: int = 100,
) -> Tuple[np.ndarray, bool]:
    """
    Detects and fixes dark well borders from the well plate in microscopy images. The algorithm works as follows:
     1- Check for dark borders using sampled border pixels. (return image if none)
     2- If dark borders are detected, it creates a mask of dark pixels and identifies connected components
     3- Checks which components touch the image borders.
     4- Finally, it replaces the dark border pixels with the mean pixel value of the non-border regions.

    Args:
        image: Input grayscale image
        darkness_threshold_factor: Factor for darkness threshold calculation
        border_sample_interval: Sampling interval for border pixel examination

    Returns:
        Tuple[np.ndarray, bool]: A tuple containing the corrected image and a boolean indicating if a border was detected.
    """

    # Quick border check using sample points with early exit for no dark border
    num_labels = 0
    image_mean = np.mean(image)
    border_pixels = np.concatenate(
        [image[0, ::50], image[-1, ::50], image[::50, 0], image[::50, -1]]
    )

    if np.min(border_pixels) > image_mean * darkness_threshold_factor:
        return image, False

    # Create dark pixel mask
    dark_mask = image < (image_mean * darkness_threshold_factor)
    if not np.any(dark_mask):
        return image, False

    # Find connected components in dark regions
    num_labels, labels = cv2.connectedComponents(dark_mask.astype(np.uint8))
    if num_labels <= 1:
        return image, False

    # Efficiently sample border pixels to detect border components
    border_components = set()

    # Check top/bottom borders
    top_samples = labels[0, ::border_sample_interval]
    bottom_samples = labels[-1, ::border_sample_interval]
    border_components.update(top_samples[top_samples > 0])
    border_components.update(bottom_samples[bottom_samples > 0])

    # Check left/right borders
    left_samples = labels[::border_sample_interval, 0]
    right_samples = labels[::border_sample_interval, -1]
    border_components.update(left_samples[left_samples > 0])
    border_components.update(right_samples[right_samples > 0])

    if not border_components:
        return image, False

    # Create mask of border components and fix image
    border_mask = np.zeros_like(dark_mask)
    for label in border_components:
        border_mask |= labels == label

    # Fix borders using non-border mean
    fixed_image = np.copy(image)
    non_border_mask = ~border_mask
    non_border_mean = np.mean(image[non_border_mask])
    fixed_image[border_mask] = non_border_mean

    return fixed_image, True


def clear_background(
    img: np.ndarray,
    sigma_r: Union[int, float],
    unit: str = "pixel",
    method: str = "divide",
    pixel_size: float = 1,
    convert_32: bool = True,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Remove background from an image using Gaussian blur.

    Args:
        img: Input 2D image
        sigma_r: Radius for Gaussian blur
        unit: Unit for sigma_r ('pixel' or physical unit)
        method: Background removal method ('subtract' or 'divide')
        pixel_size: Size of pixel in physical units
        convert_32: Convert image to float32
        clip_negative: Clip negative values to zero

    Returns:
        Background-removed image
    """
    # Input checks
    if img.ndim != 2:
        raise ValueError("Input image must be 2D")
    if convert_32:
        img = img.astype(np.float32)

    if unit == "pixel":
        pass
    else:
        sigma_r = unit_converter(sigma_r, pixel_size, to_unit="pixel")
        sigma_r = int(sigma_r)

    # Gaussian blur
    sigma_r = round_to_odd(sigma_r)
    gaussian_blur = cv2.GaussianBlur(img, (sigma_r, sigma_r), 0)

    # Background remove
    if method == "subtract":
        background_removed = cv2.subtract(img, gaussian_blur)
    elif method == "divide":
        background_removed = cv2.divide(img, gaussian_blur)
    else:
        raise ValueError("Invalid method. Choose either 'subtract' or 'divide'")

    if clip_negative:
        background_removed = np.clip(background_removed, 0, None)

    return background_removed


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the range 0-255 and convert to uint8.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Image as uint8.
    Raises:
        ValueError: If image has zero variance.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        raise ValueError("Image has zero variance; cannot normalize.")
    normalized_image = (image - min_val) / (max_val - min_val)
    scaled_image = normalized_image * 255
    return scaled_image.astype(np.uint8)


def norm_eq_hist(img: np.ndarray) -> np.ndarray:
    """
    Normalize and equalize image histogram to zero mean and unit variance.

    Args:
        img (np.ndarray): Input grayscale image

    Returns:
        np.ndarray: Normalized image with equalized histogram
    """
    img = convert_to_uint8(img)
    equalized = cv2.equalizeHist(img.astype(np.uint8))
    equalized = equalized.astype(np.float32)
    equalized = (equalized - equalized.mean()) / equalized.std()

    return equalized


def crop_black_region(img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the largest rectangle within the image that contains no black (zero) pixels.

    Args:
        img: Input grayscale image

    Returns:
        Tuple[int, int, int, int]: Coordinates (start_h, end_h, start_w, end_w) of the
        cropped region without black pixels
    """
    mask = img != 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Image is completely black.")
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return y0, y1, x0, x1


def measure_background_intensity(
    img: np.ndarray, mask: np.ndarray, target_channel: int, quantile: float = 0.10
) -> pd.DataFrame:
    """Measure background fluorescence intensity excluding foreground objects.

    Works with both NumPy and CuPy arrays (GPU-accelerated when CuPy arrays provided).

    Args:
        img (np.ndarray or cp.ndarray): Image stack [frame, slice, channel, x, y].
        mask (np.ndarray or cp.ndarray): Binary mask with objects as pixels and background as 0 [frame, slice, x, y].
        target_channel (int): Channel to measure background intensity.
        quantile (float): Quantile to compute (default 0.10 = 10th percentile).

    Returns:
        pd.DataFrame: DataFrame with background intensity (quantile) per frame.
    """
    # Detect array module (numpy or cupy)
    if hasattr(img, '__cuda_array_interface__'):
        import cupy as cp
        xp = cp
        # Use cupy's nanquantile if available, otherwise convert to numpy
        use_cupy = True
    else:
        xp = np
        use_cupy = False

    bck_intensities = []
    bck_stds = []
    frames = []
    for frame in range(img.shape[0]):
        # Ensure mask has the same number of dimensions as the image for broadcasting
        frame_mask = mask[frame, 0]
        frame_img = img[frame, 0, target_channel]

        # Apply mask to the image: set object pixels to NaN
        masked_img = xp.where(frame_mask == 0, frame_img, xp.nan)

        # Calculate the quantile and std of the background intensity
        if use_cupy:
            # CuPy doesn't have nanquantile, so convert to numpy for this operation
            masked_img_cpu = cp.asnumpy(masked_img)
            bck_intensity = float(np.nanquantile(masked_img_cpu, quantile))
            bck_std = float(np.nanstd(masked_img_cpu))
        else:
            bck_intensity = float(np.nanquantile(masked_img, quantile))
            bck_std = float(np.nanstd(masked_img))

        bck_intensities.append(bck_intensity)
        bck_stds.append(bck_std)
        frames.append(frame)

    # Create a Pandas DataFrame to store the results
    bck_fl_df = pd.DataFrame({"frame": frames, "background": bck_intensities, "bg_std": bck_stds})
    return bck_fl_df


def dynamic_resize_roi(image: torch.Tensor, min_size: int) -> torch.Tensor:
    """
    Resize a region of interest (ROI) to a uniform size using PyTorch.

    This function resizes the input image to fit within min_size while maintaining
    aspect ratio, then pads it to exactly min_size x min_size.

    Args:
        image: Input tensor of shape (Z, H, W) or (H, W)
        min_size: Target size for the output image

    Returns:
        torch.Tensor: Resized and padded image of size (min_size, min_size)
    """
    # Check if the image is 3D (Z, H, W)
    is_3d = len(image.shape) == 3
    if is_3d:
        image = image[0]

    h, w = image.shape

    if h > min_size or w > min_size:
        # Calculate scaling to maintain aspect ratio (only downscale, never upscale)
        scale = min_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = max(1, new_h)
        new_w = max(1, new_w)

        # Add batch and channel dimensions for interpolation
        image = image.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        image = F.interpolate(
            image, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        image = image.squeeze(0).squeeze(0)  # [1,1,H,W] -> [H,W]

    # Pad to target size
    pad_h = max(0, min_size - image.shape[0])
    pad_w = max(0, min_size - image.shape[1])

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = F.pad(
            image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
        )

    return image
