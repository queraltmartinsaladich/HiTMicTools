import numpy as np
import inspect
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
from templatematchingpy import StackAligner, AlignmentConfig
from basicpy import BaSiC
from .img_ops import (
    clear_background,
    norm_eq_hist,
    crop_black_region,
    detect_and_fix_well,
)
from .array_ops import (
    adjust_dimensions,
    stack_indexer,
    get_bit_depth,
)


class ImagePreprocessor:
    """
    A class for preprocessing images.

    Args:
        img (np.ndarray): Input image array.
        pixel_size (float, default=1): Pixel size of the image.
        stack_order (str, default='TSCXY'): Order of dimensions in the image stack.
        nchannels (int, default=1): Number of channels in the image.
        metadata (Optional[Dict[str, Union[float, str, int]]], default=None): Image metadata.
    """

    def __init__(
        self,
        img: np.ndarray,
        pixel_size: float = 1,
        stack_order: str = "TSCXY",
        nchannels: int = 1,
        metadata: Optional[Dict[str, Union[float, str, int]]] = None,
    ):
        """
        Standardize stack dimensions and capture metadata used throughout preprocessing.

        Args:
            img: Input image stack in any supported dimension order.
            pixel_size: Physical pixel size used for downstream measurements.
            stack_order: String describing the layout of the supplied stack.
            nchannels: Number of channels encoded in the stack.
            metadata: Optional metadata dict containing overrides for the above values.
        """
        img = adjust_dimensions(img, stack_order)
        self.img_original = img
        self.img = img
        self.frames_size = img.shape[0]
        self.slices_size = img.shape[1]
        self.channels_size = img.shape[2]

        # Image metadata
        self.bit_depth = get_bit_depth(img)
        if metadata is None:
            self.pixel_size = pixel_size
            self.stack_order = stack_order
            self.nchannels = nchannels
        else:
            self.pixel_size = metadata["pixel_size"]
            self.stack_order = metadata["stack_order"]
            self.nchannels = metadata["nchannels"]

        # Detect BaSiCPy API version for backwards compatibility
        # See: https://github.com/yuliu96/BaSiCPy vs https://github.com/yuliu96/BaSiCPy_torch
        self._basicpy_uses_is_timelapse = self._detect_basicpy_api()

    @staticmethod
    def _detect_basicpy_api() -> bool:
        """
        Detect which BaSiCPy API version is installed by inspecting the transform signature.

        This method ensures backwards compatibility between the original BaSiCPy and BaSiCPy_torch.
        The two libraries have a breaking change in the BaSiC.transform() method:
        - Original BaSiCPy uses parameter name: 'timelapse'
        - BaSiCPy_torch uses parameter name: 'is_timelapse'

        Returns:
            bool: True if using new API (is_timelapse parameter), False if using old API (timelapse parameter).
        """
        sig = inspect.signature(BaSiC.transform)
        return "is_timelapse" in sig.parameters

    def align_image(
        self,
        ref_channel: int,
        ref_slice: int,
        compres_align: float = 0,
        normalise_image: bool = True,
        crop_image: bool = True,
        alignment_config: Optional[Dict] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        reference_type: str = "static",
        subpixel: bool = False,
        upsample_factor: int = 10,
    ) -> None:
        """
        Align the image stack using a reference channel.

        Uses StackAligner (OpenCV TM_CCOEFF_NORMED, integer-pixel) by default.
        Pass ``subpixel=True`` to use DFT-upsampling phase cross-correlation
        (skimage) for sub-pixel precision — recommended for small cells or dense
        FOVs where a 1-pixel shift error is a meaningful fraction of cell size.

        After alignment ``self.frame_shifts`` is set to an (T, 2) float array of
        [dx, dy] per frame (column-first convention, matching StackAligner tmats).
        These are the translations applied to register each frame to the reference.
        For ``reference_type='dynamic'`` shifts are frame-to-frame; for
        ``reference_type='static'`` they are cumulative from the reference frame.

        Args:
            ref_channel: BF channel index used as registration reference.
            ref_slice: S-dimension slice index; also the reference frame in static mode.
            compres_align: Unused; kept for backward compatibility.
            normalise_image: Normalise each frame to its mean before matching.
            crop_image: Crop the black border introduced by translation.
            alignment_config: Config dict forwarded to StackAligner (NCC path only).
            bbox: (x, y, w, h) template crop region. Defaults to central quarter.
            reference_type: 'static' — all frames vs. one fixed frame;
                'dynamic' — each frame vs. the previous frame.
            subpixel: Use phase cross-correlation instead of NCC template matching.
            upsample_factor: DFT upsampling factor (subpixel=True only).
                Precision ≈ 1/upsample_factor pixels. Default 10 → 0.1 px.
        """
        assert 0 <= compres_align <= 1, "compress_align must be between 0 and 1"

        reference_channel = self.img[:, ref_slice, ref_channel, :, :]
        if normalise_image:
            reference_channel = reference_channel / np.mean(
                reference_channel, axis=(1, 2), keepdims=True
            )

        if bbox is None:
            height, width = reference_channel.shape[1], reference_channel.shape[2]
            box_width = width // 2
            box_height = height // 2
            x = (width - box_width) // 2
            y = (height - box_height) // 2
            bbox = (x, y, box_width, box_height)

        if subpixel:
            self._align_subpixel(
                reference_channel, ref_slice, bbox, reference_type,
                upsample_factor, crop_image,
            )
            return

        config = AlignmentConfig(**(alignment_config or {}))
        self.aligner = StackAligner(config=config)
        ref_aligned = self.aligner.register_stack(
            reference_channel,
            bbox=bbox,
            reference_slice=ref_slice,
            reference_type=reference_type,
        )
        self.tmats = self.aligner.translation_matrices
        # dx = tmats[:, 0, 2], dy = tmats[:, 1, 2] in OpenCV (column, row) convention
        self.frame_shifts = np.column_stack(
            [self.tmats[:, 0, 2], self.tmats[:, 1, 2]]
        )

        index_table_sc = set(
            (s, c)
            for t, s, c in stack_indexer(
                range(self.frames_size), range(self.slices_size), range(self.channels_size)
            )
        )
        for s, c in index_table_sc:
            self.img[:, s, c, :, :] = self.aligner.transform_stack(self.img[:, s, c, :, :])

        if crop_image:
            min_projection = np.min(ref_aligned, axis=0)
            start_h, end_h, start_w, end_w = crop_black_region(min_projection)
            self.img = self.img[:, :, :, start_h:end_h, start_w:end_w]

    def _align_subpixel(
        self,
        reference_channel: np.ndarray,
        ref_slice: int,
        bbox: Tuple[int, int, int, int],
        reference_type: str,
        upsample_factor: int,
        crop_image: bool,
    ) -> None:
        """Sub-pixel registration via DFT-upsampling phase cross-correlation."""
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import shift as nd_shift

        T = reference_channel.shape[0]
        x, y, bw, bh = bbox
        ref_frame_idx = ref_slice if ref_slice >= 0 else T + ref_slice
        ref_static = reference_channel[ref_frame_idx, y:y + bh, x:x + bw]

        raw_shifts = np.zeros((T, 2), dtype=float)  # (dy, dx) per frame, row-col
        for t in range(T):
            ref_crop = (
                ref_static if reference_type == "static"
                else reference_channel[max(0, t - 1), y:y + bh, x:x + bw]
            )
            shift_vec, _, _ = phase_cross_correlation(
                ref_crop,
                reference_channel[t, y:y + bh, x:x + bw],
                upsample_factor=upsample_factor,
            )
            raw_shifts[t] = shift_vec  # (dy, dx)

        orig_dtype = self.img.dtype
        for t in range(T):
            dy, dx = raw_shifts[t]
            if dy == 0.0 and dx == 0.0:
                continue
            for s in range(self.slices_size):
                for c in range(self.channels_size):
                    frame = self.img[t, s, c].astype(np.float32)
                    self.img[t, s, c] = nd_shift(
                        frame, [dy, dx], mode="constant", cval=0.0
                    ).astype(orig_dtype)

        # Store as (dx, dy) — column-first, matching StackAligner tmats convention
        self.frame_shifts = np.column_stack([raw_shifts[:, 1], raw_shifts[:, 0]])

        if crop_image:
            aligned_ref = np.stack([
                nd_shift(
                    reference_channel[t].astype(np.float32),
                    raw_shifts[t], mode="constant", cval=0.0,
                )
                for t in range(T)
            ])
            min_projection = np.min(aligned_ref, axis=0)
            start_h, end_h, start_w, end_w = crop_black_region(min_projection)
            self.img = self.img[:, :, :, start_h:end_h, start_w:end_w]

    def align_from_matrix(self, img: np.ndarray) -> np.ndarray:
        """
        Align a new image using a transformation matrix from the source image, using StackAligner.

        Args:
            img (np.ndarray): Input image array.

        Returns:
            np.ndarray: Aligned image array.
        """
        # Check if the input image has the correct dimensions
        if (
            img.ndim != 3
            or img.shape[0] != self.frames_size
            or img.shape[1:] != self.img.shape[-2:]
        ):
            raise ValueError(
                f"Input image must be 3D (frames, x, y) with shape ({self.frames_size}, {self.img.shape[-2]}, {self.img.shape[-1]}). "
                f"Got shape: {img.shape}"
            )

        if self.aligner is None or not self.aligner.is_registered:
            raise ValueError(
                "StackAligner has not been initialized or registration has not been performed. "
                "Please align the reference image first using align_image."
            )

        return self.aligner.transform_stack(img)

    def clear_image_background(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
        method: str,
        convert_32: bool = True,
        **kwargs,
    ) -> None:
        """
        Clear the image background using the specified method.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.
            method (str): Background removal method ('divide', 'subtract', or 'basicpy').
            convert_32 (bool, default=True): Whether to convert the image to float32 before processing.
            **kwargs: Additional keyword arguments for the background removal method.

        Returns:
            None
        """
        # Note, in order to collect the image in the self.img, I have to change type before processing
        if convert_32:
            self.img = self.img.astype(np.float32)

        # Assert that nframes, nslices, and nchannels are within valid range
        self.check_size_limit(nframes, self.frames_size, "nframes")
        self.check_size_limit(nslices, self.slices_size, "nslices")
        self.check_size_limit(nchannels, self.channels_size, "nchannels")

        if method == "divide":
            self._cv2_clear_image_background(
                nframes, nslices, nchannels, method="divide", **kwargs
            )
        elif method == "subtract":
            self._cv2_clear_image_background(
                nframes, nslices, nchannels, method="subtract", **kwargs
            )
        elif method == "basicpy":
            self._basicpy_clear_image_background(nframes, nslices, nchannels, **kwargs)
        else:
            raise ValueError(
                f"Invalid method: {method}. Choose either 'divide', 'subtract', or 'basicpy'."
            )

    def _basicpy_clear_image_background(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
        **kwargs,
    ) -> None:
        """
        Clear the image background using the basicpy method.

        This method supports both original BaSiCPy and BaSiCPy_torch libraries.
        Due to a breaking API change between these versions, the transform() method
        parameter changed from 'timelapse' to 'is_timelapse'. This implementation
        automatically detects which version is installed and uses the appropriate
        parameter name.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.
            **kwargs: Additional keyword arguments for BaSiC.

        Returns:
            None
        """
        img_to_transform = self.img[nframes, nslices, nchannels]
        if len(img_to_transform.shape) > 3:
            raise TypeError(
                f"Image to transform with basicpy must be 2D or 3D. Got shape: {img_to_transform.shape}"
            )
        elif len(img_to_transform.shape) == 3:
            is_timelapse = True
        else:
            is_timelapse = False

        basic = BaSiC(**kwargs)
        basic.fit(img_to_transform)

        # Use appropriate parameter name based on detected BaSiCPy version
        if self._basicpy_uses_is_timelapse:
            images_transformed = basic.transform(img_to_transform, is_timelapse=is_timelapse)
        else:
            images_transformed = basic.transform(img_to_transform, timelapse=is_timelapse)

        self.img[nframes, nslices, nchannels] = images_transformed

    def _cv2_clear_image_background(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
        method: str,
        **kwargs,
    ) -> None:
        """
        Clear the image background using the OpenCV method.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.
            method (str): Background removal method ('divide' or 'subtract').
            **kwargs: Additional keyword arguments for clear_background.

        Returns:
            None
        """
        index_table = stack_indexer(nframes, nslices, nchannels)
        for index in index_table:
            t, s, c = index
            self.img[t, s, c, :, :] = clear_background(
                self.img[t, s, c, :, :], method=method, **kwargs
            )

    def detect_fix_well(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
        **kwargs,
    ) -> None:
        """
        Detect and fix well borders in the image. It update the loaded image and save the border detection info for logging purposes.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.
            **kwargs: Additional keyword arguments for detect_and_fix_well.

        Returns:
            None
        """
        index_table = stack_indexer(nframes, nslices, nchannels)
        has_border_array = np.zeros(
            (self.frames_size, self.slices_size, self.channels_size), dtype=bool
        )

        for index in index_table:
            t, s, c = index
            self.img[t, s, c, :, :], has_border = detect_and_fix_well(
                self.img[t, s, c, :, :], **kwargs
            )
            has_border_array[t, s, c] = has_border
        self.borders = has_border_array

    def norm_eq_hist(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
    ) -> None:
        """
        Normalize and equalize the histogram of the image.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.

        Returns:
            None
        """
        index_table = stack_indexer(nframes, nslices, nchannels)

        for index in index_table:
            t, s, c = index
            self.img[t, s, c, :, :] = norm_eq_hist(self.img[t, s, c, :, :])

    def scale_channel(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
    ) -> None:
        """
        Scale the image by (x - mean) / sd for each slice of the target channel.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.

        Returns:
            None
        """
        index_table = stack_indexer(nframes, nslices, nchannels)

        for index in index_table:
            t, s, c = index
            slice_data = self.img[t, s, c, :, :]
            mean = np.mean(slice_data)
            std = np.std(slice_data)
            self.img[t, s, c, :, :] = (slice_data - mean) / std

    @staticmethod
    def check_size_limit(
        input: Union[int, range, List[int]], size_limit: int, name: str
    ) -> None:
        """
        Check if the input size is within the valid range.

        Args:
            input (Union[int, range, List[int]]): Input size.
            size_limit (int): Maximum allowed size.
            name (str): Name of the input parameter.

        Returns:
            None
        """
        if isinstance(input, int):
            max_value = input
        elif isinstance(input, range):
            max_value = input.stop
        elif isinstance(input, list) and all(isinstance(i, int) for i in input):
            max_value = max(input)
        else:
            raise TypeError(
                f"{name} must be an integer, a range object, or a list of integers"
            )

        assert max_value <= size_limit, f"{name} exceeds image dimensions"

    def apply_species_preprocessing(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        bf_channel: int,
        fl_channel: Optional[int],
        species_config: Dict,
    ) -> None:
        """
        Apply species-specific preprocessing operators to the BF (and optionally FL) channel.

        Operators are dispatched from *species_config*, which is typically loaded from
        ``species_defaults.yaml`` or a user-supplied override file.  Each operator entry
        must contain an ``enabled`` key; all other keys are forwarded as kwargs.

        The BF channel is enhanced in-place.  When *fl_channel* is provided and
        ``fl_normalization.enabled`` is true, the FL channel is normalised to [0, 1]
        and the result is stored both in ``self.img`` and as ``self.fl_norm`` for later
        use (e.g. union-mask construction in the segmentation step).

        Args:
            nframes: Frame indices to process.
            nslices: Slice indices to process.
            bf_channel: BF channel index inside the image stack.
            fl_channel: FL channel index, or None when FL is unavailable.
            species_config: Dict mapping operator names to parameter dicts.

        Returns:
            None  (modifies ``self.img`` in-place)
        """
        from .species_preprocessing import (
            apply_hessian_tubularness,
            apply_directional_tophat,
            apply_anisotropic_diffusion,
            apply_rl_deconvolution,
            apply_phase_congruency,
            normalize_fluorescence,
        )

        # Operator registry: name → callable
        _bf_operators = {
            "hessian_tubularness": apply_hessian_tubularness,
            "directional_tophat": apply_directional_tophat,
            "anisotropic_diffusion": apply_anisotropic_diffusion,
            "rl_deconvolution": apply_rl_deconvolution,
            "phase_congruency": apply_phase_congruency,
        }

        self.check_size_limit(nframes, self.frames_size, "nframes")
        self.check_size_limit(nslices, self.slices_size, "nslices")

        index_table = stack_indexer(nframes, nslices, [bf_channel])

        for t, s, _c in index_table:
            frame = self.img[t, s, bf_channel, :, :].astype(np.float32)

            for op_name, op_fn in _bf_operators.items():
                op_cfg = species_config.get(op_name, {})
                if not op_cfg.get("enabled", False):
                    continue
                kwargs = {k: v for k, v in op_cfg.items() if k != "enabled"}
                frame = op_fn(frame, **kwargs)

            self.img[t, s, bf_channel, :, :] = frame

        # FL channel normalisation
        if fl_channel is not None:
            fl_cfg = species_config.get("fl_normalization", {})
            if fl_cfg.get("enabled", False):
                fl_kwargs = {k: v for k, v in fl_cfg.items() if k != "enabled"}
                fl_index_table = stack_indexer(nframes, nslices, [fl_channel])
                fl_norm_stack = self.img[:, :, fl_channel, :, :].copy().astype(np.float32)
                for t, s, _c in fl_index_table:
                    norm = normalize_fluorescence(
                        self.img[t, s, fl_channel, :, :], **fl_kwargs
                    )
                    self.img[t, s, fl_channel, :, :] = norm
                    fl_norm_stack[t, s] = norm
                # Expose as a convenience attribute for the union-mask step
                self.fl_norm = fl_norm_stack

    def build_fl_norm(
        self,
        fl_channel: int,
        nframes: Union[int, range, List[int]],
        p_low: float = 1.0,
        p_high: float = 99.0,
    ) -> np.ndarray:
        """
        Build and cache a percentile-normalized FL stack for union-mask construction.

        Idempotent: if ``apply_species_preprocessing`` already populated ``self.fl_norm``,
        this method returns it without recomputing.

        Args:
            fl_channel: FL channel index in the image stack.
            nframes: Frame indices to normalize.
            p_low: Lower percentile for clipping (default 1.0).
            p_high: Upper percentile for clipping (default 99.0).

        Returns:
            np.ndarray: Normalized FL stack, shape (T, S, X, Y), float32 in [0, 1].
        """
        if hasattr(self, "fl_norm"):
            return self.fl_norm

        from .species_preprocessing import normalize_fluorescence

        fl_norm_stack = self.img[:, :, fl_channel, :, :].copy().astype(np.float32)
        index_table = stack_indexer(nframes, 0, [fl_channel])
        for t, s, _c in index_table:
            fl_norm_stack[t, s] = normalize_fluorescence(
                self.img[t, s, fl_channel, :, :],
                percentile_low=p_low,
                percentile_high=p_high,
            )
        self.fl_norm = fl_norm_stack
        return self.fl_norm
