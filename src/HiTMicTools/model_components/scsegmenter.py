from __future__ import annotations

import warnings
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

try:
    from bacdetr import BacDETRSeg as _BacDETRSeg

    class BacDETRSeg(_BacDETRSeg):
        """Concrete subclass that stubs out abstract training methods.

        BacDETR declares train() and train_from_config() as abstract, but
        HiTMicTools only needs inference.  This thin wrapper satisfies the
        ABC contract so the class can be instantiated for prediction only.
        """

        def train(self, **kwargs):
            raise NotImplementedError("Training is not supported via HiTMicTools")

        def train_from_config(self, *args, **kwargs):
            raise NotImplementedError("Training is not supported via HiTMicTools")

except ImportError:
    _BacDETRSeg = None
    BacDETRSeg = None

from HiTMicTools.model_components.base_model import BaseModel
from HiTMicTools.resource_management.sysutils import get_device


@dataclass
class SegmentationBatch:
    """Container used internally to keep track of tile metadata during inference."""

    tiles: List[torch.Tensor]
    offsets: List[Tuple[int, int]]
    valid_shapes: List[Tuple[int, int]]
    image_shape: Tuple[int, int]


class ScSegmenter(BaseModel):
    """
    Single-cell instance segmenter using BacDETR / RF-DETR detection backbone.

    This class handles high-resolution microscopy frames by tiling them into
    detector-sized crops, forwarding each crop through the detector, and merging
    the results with batched NMS to recover full-frame instance segmentations.

    Unlike the OofDetector, this model returns both bounding boxes and instance masks,
    performing simultaneous detection, segmentation, and classification of single cells.

    Attributes:
        model: BacDETRSeg model instance (handles both BacDETR and RF-DETR checkpoints)
        device: Torch device (CPU/CUDA)
        tile_size: Square tile edge length for sliding window
        overlap_ratio: Fractional overlap between adjacent tiles
        score_threshold: Minimum confidence for detections
        nms_iou: IoU threshold for non-maximum suppression
        class_dict: Mapping from class indices to class names
    """

    # Class constant for per-tile normalization
    NORMALIZATION_EPSILON = 1e-6  # Prevent division by zero in per-tile normalization
    PRIORITY_CLASS_NAMES = {1: "clump", 2: "debris", 0: "single-cell"}
    PRIORITY_CLASS_ORDER = {1: 3, 2: 2, 0: 1}
    # For our problem case it makes sense that clumps swallow everything.
    DEFAULT_MIN_MASK_AREA = 5  # Minimum mask area in pixels to keep a detection

    # Valid compile modes for torch.compile
    VALID_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune", False}

    def __init__(
        self,
        model_path: str,
        patch_size: int = 256,
        overlap_ratio: float = 0.33,
        score_threshold: float = 0.4,
        nms_iou: float = 0.4,
        clump_merge_min_overlap: int = 250,
        priority_overlap_fraction: float = 0.5,
        temporal_buffer_size: int = 8,
        batch_size: int = 128,
        mask_threshold: float = 0.5,
        min_mask_area: int = DEFAULT_MIN_MASK_AREA,
        class_dict: Optional[dict] = None,
        model_type: str = "rfdetrsegpreview",
        compile_mode: str = False,
    ) -> None:
        """
        Initialize the single-cell segmenter.

        Args:
            model_path: Filesystem path to a detector checkpoint (.pth).
            patch_size: Square tile edge length passed to the detector.
            overlap_ratio: Fractional overlap between adjacent tiles.
            score_threshold: Minimum confidence kept from per-tile detections.
            nms_iou: IoU threshold for the cross-class non-maximum suppression.
            clump_merge_min_overlap: Minimum absolute mask overlap in pixels for clump merging.
                Clumps are merged if mask overlap >= clump_merge_min_overlap pixels.
                Uses actual mask intersection (not bbox). Default: 250 pixels.
            priority_overlap_fraction: Fraction of an existing label's area that must be
                overlapped by a higher-priority class before the existing label is overwritten
                during mask stitching. Range (0, 1]. Default: 0.5.
            temporal_buffer_size: Number of frames to process in GPU memory at once.
            batch_size: Number of spatial tiles to process in parallel per batch.
            mask_threshold: Binary threshold for converting predicted masks to instance labels.
            min_mask_area: Minimum surviving mask area in pixels to keep a detection.
            class_dict: Dictionary mapping class indices to names (e.g., {0: 'single-cell', 1: 'clump'}).
                If provided, num_classes is derived from its length. If None, inferred from checkpoint.
            model_type: Stored for metadata/logging purposes. All checkpoints are
                loaded through BacDETRSeg which handles both BacDETR and RF-DETR weights.
            compile_mode (str or False): Torch compile mode. Options:
                - "default": Fast compilation, good performance
                - "reduce-overhead": Optimized for small batches, uses CUDA graphs
                - "max-autotune": Slowest compilation, best runtime performance
                - False: Disable torch.compile entirely
        """
        assert 0 <= overlap_ratio < 1, "overlap_ratio must be in [0, 1)."
        assert patch_size > 0, "patch_size must be positive."
        assert temporal_buffer_size > 0, "temporal_buffer_size must be positive."
        assert batch_size > 0, "batch_size must be positive."
        assert 0 < mask_threshold < 1, "mask_threshold must be in (0, 1)."
        assert min_mask_area >= 0, "min_mask_area must be non-negative."
        assert 0 < priority_overlap_fraction <= 1, "priority_overlap_fraction must be in (0, 1]."

        # Validate compile_mode
        if compile_mode not in self.VALID_COMPILE_MODES:
            raise ValueError(
                f"Invalid compile_mode: {compile_mode}. "
                f"Must be one of: {self.VALID_COMPILE_MODES}"
            )

        self.device = get_device()
        if self.device.type == "mps":
            warnings.warn(
                "ScSegmenter falling back to CPU because RF-DETR backbone "
                "uses ops unsupported on MPS.",
                RuntimeWarning,
            )
            self.device = torch.device("cpu")

        self.tile_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.clump_merge_min_overlap = clump_merge_min_overlap
        self.priority_overlap_fraction = priority_overlap_fraction
        self.temporal_buffer_size = temporal_buffer_size
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold
        self.min_mask_area = min_mask_area
        self.class_dict = class_dict

        self.model_type = model_type.lower()

        # Determine num_classes from class_dict or infer from checkpoint
        num_classes = None
        if class_dict:
            num_classes = len(class_dict)
        else:
            checkpoint = torch.load(
                model_path, map_location="cpu", weights_only=False
            )
            class_bias = checkpoint["model"]["class_embed.bias"]
            num_classes = class_bias.shape[0] - 1

        # BacDETRSeg handles both BacDETR (with adapter) and vanilla RF-DETR
        # checkpoints (without adapter) through the same API.
        if BacDETRSeg is None:
            raise ImportError(
                "BacDETR is required for single-cell segmentation but is not installed.\n"
                "Install it with:\n"
                "  pip install 'hitmictools[bacdetr]'\n"
                "or directly:\n"
                "  pip install 'bacdetr @ git+https://github.com/BoeckLab/BacDETRSegm.git@develop#subdirectory=bacdetr'\n\n"
                "If using an RF-DETR checkpoint, you also need:\n"
                "  pip install 'hitmictools[rfdetr]'\n"
                "or directly:\n"
                "  pip install rfdetr  # from https://github.com/roboflow/rf-detr"
            )
        self.model = BacDETRSeg(
            pretrain_weights=model_path,
            num_classes=num_classes,
            device=self.device.type,
        )
        self.model.model.model.eval()

        try:
            self.model.optimize_for_inference()
        except Exception:
            pass  # non-fatal: falls back to eager inference

        # IMPORTANT: skip torch.compile on MPS due to torch.fx symbolic tracing
        # conflicts with ThreadPoolExecutor in parallel processing.
        if compile_mode and self.device.type != "mps":
            self.model.model.model = torch.compile(
                self.model.model.model, mode=compile_mode
            )

    def predict(
        self,
        image: Union[np.ndarray, torch.Tensor],
        channel_index: int = 0,
        temporal_buffer_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        normalize_to_255: bool = True,
        score_threshold: Optional[float] = None,
        output_shape: str = "HW",
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Run 4D sliding-window inference with temporal buffering on microscopy images.

        This method efficiently processes time-series microscopy data by:
        1. Loading temporal chunks into GPU memory (controlled by temporal_buffer_size)
        2. Interleaving spatial tiles from multiple frames for maximum GPU utilization
        3. Processing tiles in batches with cross-frame batching
        4. Releasing processed frames from GPU to manage memory

        The method automatically handles both single-frame and multi-frame inputs,
        and performs channel padding (grayscale → RGB) internally.

        Args:
            image: Input array with shape:
                - [T, H, W]: Multi-frame grayscale (most common from pipelines)
                - [T, C, H, W]: Multi-frame multi-channel
                - [H, W]: Single frame grayscale
                - [C, H, W]: Single frame multi-channel
            channel_index: Channel to segment (default: 0 for grayscale)
            temporal_buffer_size: Number of frames to keep in GPU memory at once.
                If None, uses value from initialization. Larger values increase GPU
                memory usage but reduce CPU↔GPU transfers.
                Recommended: 4-8 for typical workloads, 2-4 for memory-constrained GPUs.
            batch_size: Number of spatial tiles to process in parallel.
                If None, uses value from initialization. Larger values improve GPU
                utilization but increase memory usage.
                Recommended: 16-32 for typical GPUs.
            normalize_to_255: Whether to min-max normalize the selected channel
                to [0, 1] range. Set to False if image is already preprocessed.
            score_threshold: Optional override for detection confidence threshold.
                If None, uses the threshold set during initialization.
            output_shape: Output dimension ordering for masks. Options:
                - "HW": Height × Width (default, standard image convention)
                - "WH": Width × Height (HiTMicTools TSCXY convention compatibility)
                For multi-frame output, this applies to the last two dimensions of [T, *, *]

        Returns:
            Tuple of:
                - labeled_masks: [T, H, W] or [T, W, H] array of stacked instance masks
                  (or [H, W]/[W, H] for single frame), depending on output_shape parameter
                - bboxes_list: List of [N_t, 4] bbox arrays per frame (xyxy format)
                - class_ids_list: List of [N_t] class ID arrays per frame
                - scores_list: List of [N_t] confidence score arrays per frame

        Examples:
            >>> # Multi-frame input with HW output (standard)
            >>> frames = ip.img[:, 0, 0, :, :]  # Shape: [T, H, W]
            >>> masks, bboxes, classes, scores = segmenter.predict(
            ...     frames,
            ...     channel_index=0,
            ...     temporal_buffer_size=8,
            ...     batch_size=32,
            ...     normalize_to_255=False,
            ...     output_shape="HW"  # Returns [T, H, W]
            ... )

            >>> # Multi-frame input with WH output (HiTMicTools TSCXY compatibility)
            >>> frames = ip.img[:, 0, 0, :, :]  # Shape: [T, X, Y] in TSCXY convention
            >>> masks, bboxes, classes, scores = segmenter.predict(
            ...     frames,
            ...     output_shape="WH"  # Returns [T, X, Y] matching ip.img convention
            ... )

            >>> # Single frame input (backward compatible)
            >>> frame = ip.img[0, 0, 0, :, :]  # Shape: [H, W]
            >>> mask, bboxes, classes, scores = segmenter.predict(
            ...     frame,
            ...     temporal_buffer_size=1,
            ...     batch_size=32
            ... )
        """
        # Use provided values or fall back to instance attributes
        buffer_size = temporal_buffer_size if temporal_buffer_size is not None else self.temporal_buffer_size
        batch_size = batch_size if batch_size is not None else self.batch_size
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # Validate parameters
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if buffer_size <= 0:
            raise ValueError("temporal_buffer_size must be positive.")
        if output_shape not in ["HW", "WH"]:
            raise ValueError(f"output_shape must be 'HW' or 'WH', got '{output_shape}'")

        # 1. Prepare input - convert to [T, H, W] format
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image, is_single_frame = self._reshape_input(image, channel_index)

        # 2. Process in temporal buffers
        num_frames = image.shape[0]
        all_labeled_masks = []
        all_bboxes = []
        all_class_ids = []
        all_scores = []

        effective_buffer = min(max(1, num_frames), buffer_size)

        for buffer_start in range(0, num_frames, effective_buffer):
            buffer_end = min(buffer_start + effective_buffer, num_frames)

            # Load temporal buffer to GPU
            buffer_frames = image[buffer_start:buffer_end].to(self.device)

            # Process frames in the buffer with cross-frame tile batching
            buffer_results = self._process_temporal_buffer(
                buffer_frames,
                batch_size=batch_size,
                normalize_to_255=normalize_to_255,
                score_threshold=threshold,
            )

            # Aggregate results
            for labeled_mask, bboxes, class_ids, scores in buffer_results:
                all_labeled_masks.append(labeled_mask)
                all_bboxes.append(bboxes)
                all_class_ids.append(class_ids)
                all_scores.append(scores)

            # Explicitly free buffer from GPU
            del buffer_frames
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # 3. Format and return output
        formatted_output = self._format_output(
            all_labeled_masks,
            all_bboxes,
            all_class_ids,
            all_scores,
            is_single_frame,
            output_shape,
        )
        return formatted_output

    def _reshape_input(
        self,
        image: torch.Tensor,
        channel_index: int,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Reshape input image to [T, H, W] format and detect if single frame.

        Args:
            image: Input tensor with variable dimensions
            channel_index: Channel to extract if multi-channel

        Returns:
            Tuple of:
                - image: [T, H, W] tensor ready for processing
                - is_single_frame: Boolean indicating if input was a single frame

        Raises:
            ValueError: If input dimensions are not 2D, 3D, or 4D
            IndexError: If channel_index is out of bounds
        """
        is_single_frame = False
        image = image.squeeze()

        if image.ndim == 2:
            # [H, W] → [1, 1, H, W]
            image = image.unsqueeze(0).unsqueeze(0)
            is_single_frame = True
        elif image.ndim == 3:
            # Could be [C, H, W] or [T, H, W]
            # Assume [T, H, W] (most common from pipelines)
            # If user has [C, H, W], they should use channel_index appropriately
            image = image.unsqueeze(1)  # [T, H, W] → [T, 1, H, W]
            if image.shape[0] == 1:
                is_single_frame = True
        elif image.ndim == 4:
            # [T, C, H, W] - already in correct format
            if image.shape[0] == 1:
                is_single_frame = True
        else:
            raise ValueError(
                f"Unsupported image dimensions: expected 2D, 3D, or 4D input, got shape {image.shape}."
            )

        # Validate channel index
        num_channels = image.shape[1]
        if channel_index >= num_channels:
            raise IndexError(
                f"channel_index {channel_index} out of bounds for image with {num_channels} channels."
            )

        # Extract target channel: [T, C, H, W] → [T, H, W]
        image = image[:, channel_index, :, :].to(dtype=torch.float32)

        return image, is_single_frame

    def _format_output(
        self,
        all_labeled_masks: List[np.ndarray],
        all_bboxes: List[np.ndarray],
        all_class_ids: List[np.ndarray],
        all_scores: List[np.ndarray],
        is_single_frame: bool,
        output_shape: str,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Format output masks and detections based on single/multi-frame and output_shape.

        Args:
            all_labeled_masks: List of [H, W] masks from each frame
            all_bboxes: List of bbox arrays from each frame
            all_class_ids: List of class ID arrays from each frame
            all_scores: List of score arrays from each frame
            is_single_frame: Whether input was a single frame
            output_shape: "HW" or "WH" dimension ordering

        Returns:
            Tuple of:
                - labeled_masks: Single mask [H/W, W/H] or stacked [T, H/W, W/H]
                - bboxes: Single array or list of arrays
                - class_ids: Single array or list of arrays
                - scores: Single array or list of arrays
        """
        if is_single_frame:
            mask = all_labeled_masks[0]
            if output_shape == "WH":
                mask = mask.T  # [H, W] → [W, H]
            return mask, all_bboxes[0], all_class_ids[0], all_scores[0]
        else:
            stacked_masks = np.stack(all_labeled_masks, axis=0)  # [T, H, W]
            if output_shape == "WH":
                stacked_masks = np.transpose(stacked_masks, (0, 2, 1))
            return stacked_masks, all_bboxes, all_class_ids, all_scores

    def _process_temporal_buffer(
        self,
        buffer_frames: torch.Tensor,
        batch_size: int,
        normalize_to_255: bool,
        score_threshold: float,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Process a temporal buffer of frames with cross-frame tile batching.

        This method implements the advanced batching strategy:
        1. Pre-tiles all frames in the buffer
        2. Interleaves tiles from different frames for efficient GPU batching
        3. Processes tiles from multiple frames in the same batch
        4. Demultiplexes detections back to their respective frames

        This maximizes GPU utilization by allowing RF-DETR to process
        tile_0_frame_0, tile_0_frame_1, ... tile_0_frame_N in the same batch.

        Args:
            buffer_frames: [B, H, W] tensor of frames to process
            batch_size: Number of tiles to process in parallel
            normalize_to_255: Whether to normalize frames
            score_threshold: Detection confidence threshold

        Returns:
            List of (labeled_mask, bboxes, class_ids, scores) tuples, one per frame
        """
        buffer_size = buffer_frames.shape[0]
        if buffer_size == 0:
            return []

        # 1. Prepare all frames and create tiles
        all_batches = []
        for frame_idx in range(buffer_size):
            frame_tensor = self._prepare_frame_tensor(
                buffer_frames[frame_idx],
                normalize_to_255=normalize_to_255,
            )
            batch = self._create_tiles(frame_tensor)
            all_batches.append(batch)

        # 2. Interleave tiles from all frames for better batching
        mega_tiles = []
        tile_to_frame_map = []
        max_tiles = max(len(batch.tiles) for batch in all_batches)
        for tile_idx in range(max_tiles):
            for frame_idx, batch in enumerate(all_batches):
                if tile_idx < len(batch.tiles):
                    mega_tiles.append(batch.tiles[tile_idx])
                    tile_to_frame_map.append((frame_idx, tile_idx))

        # 3. Process mega-batch with spatial batching
        all_detections = []
        use_autocast = self.device.type == "cuda"
        autocast_cm = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_autocast
            else nullcontext()
        )

        for start in range(0, len(mega_tiles), batch_size):
            batch_tiles = mega_tiles[start : start + batch_size]

            # Prepare batch with normalization and padding
            tensor_batch_list, actual_count = self._prepare_batch(
                batch_tiles, batch_size
            )

            if not tensor_batch_list:
                continue

            # Run inference with autocast
            with autocast_cm:
                predictions = self.model.predict(
                    tensor_batch_list, threshold=score_threshold
                )

            if not isinstance(predictions, list):
                predictions = [predictions]

            # Remove padding predictions
            if actual_count < len(predictions):
                predictions = predictions[:actual_count]

            all_detections.extend(predictions)

        # 4. Demultiplex detections back to frames
        frame_detections = [[] for _ in range(buffer_size)]
        for detection, (frame_idx, _) in zip(all_detections, tile_to_frame_map):
            frame_detections[frame_idx].append(detection)

        # 5. Merge detections for each frame
        results = []
        for frame_idx, detections in enumerate(frame_detections):
            batch = all_batches[frame_idx]
            labeled_mask, boxes, class_ids, scores = self._merge_detections(
                batch, detections
            )
            results.append((labeled_mask, boxes, class_ids, scores))

        return results

    def _prepare_frame_tensor(
        self,
        frame: torch.Tensor,
        normalize_to_255: bool,
    ) -> torch.Tensor:
        """
        Convert a single-channel frame to RGB tensor ready for RF-DETR.

        Args:
            frame: [H, W] tensor, single channel, float32
            normalize_to_255: Whether to apply min-max normalization

        Returns:
            [3, H, W] tensor ready for RF-DETR inference
        """
        # Ensure frame is [1, H, W] for processing
        if frame.ndim == 2:
            frame = frame.unsqueeze(0)

        # Apply min-max normalization if requested
        if normalize_to_255:
            frame = frame - frame.amin(dim=(-2, -1), keepdim=True)
            max_val = frame.amax(dim=(-2, -1), keepdim=True)
            frame = torch.where(
                max_val > 0, frame / max_val, torch.zeros_like(frame)
            )

        # Pad grayscale to RGB (replicate channel 3 times)
        frame = frame.repeat(3, 1, 1)  # [1, H, W] → [3, H, W]

        # Verify values are in [0, 1] range as required by RF-DETR
        if frame.max() > 1.0:
            frame = frame.clamp(0, 1)

        return frame

    def _create_tiles(self, image_tensor: torch.Tensor) -> SegmentationBatch:
        """Decompose full image into overlapping tiles for sliding-window inference."""
        _, height, width = image_tensor.shape
        padded_tensor = self._pad_if_needed(image_tensor)
        _, padded_h, padded_w = padded_tensor.shape

        step = max(int(self.tile_size * (1 - self.overlap_ratio)), 1)
        x_positions = self._compute_positions(padded_w, step)
        y_positions = self._compute_positions(padded_h, step)

        tiles: List[torch.Tensor] = []
        offsets: List[Tuple[int, int]] = []
        valid_shapes: List[Tuple[int, int]] = []

        for y in y_positions:
            for x in x_positions:
                crop = padded_tensor[
                    :, y : y + self.tile_size, x : x + self.tile_size
                ]
                tiles.append(crop)

                valid_h = min(self.tile_size, height - y) if y < height else 0
                valid_w = min(self.tile_size, width - x) if x < width else 0
                valid_shapes.append((max(valid_h, 0), max(valid_w, 0)))
                offsets.append((x, y))

        return SegmentationBatch(
            tiles=tiles,
            offsets=offsets,
            valid_shapes=valid_shapes,
            image_shape=(height, width),
        )

    def _pad_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pad image with mean pixel value to ensure all tiles are exactly tile_size × tile_size.

        This calculates padding needed so that the last tile position + tile_size
        doesn't exceed the padded dimensions, preventing partial tiles.
        """
        _, height, width = tensor.shape
        step = max(int(self.tile_size * (1 - self.overlap_ratio)), 1)

        # Calculate the required padded dimensions
        # We need to ensure that the last tile starting position + tile_size fits exactly
        def calc_padded_size(length: int) -> int:
            """Return the padded dimension so sliding window steps land exactly on tiles."""
            if length <= self.tile_size:
                return self.tile_size

            # Calculate number of steps needed
            num_steps = (length - self.tile_size + step - 1) // step
            # Last position is at num_steps * step
            last_position = num_steps * step
            # Required size is last_position + self.tile_size
            required_size = last_position + self.tile_size
            return required_size

        required_height = calc_padded_size(height)
        required_width = calc_padded_size(width)

        pad_bottom = required_height - height
        pad_right = required_width - width

        if pad_bottom == 0 and pad_right == 0:
            return tensor

        mean_value = tensor.mean()

        return F.pad(
            tensor,
            (0, pad_right, 0, pad_bottom),
            mode="constant",
            value=mean_value.item(),
        )

    def _compute_positions(self, length: int, step: int) -> List[int]:
        """Calculate tile starting positions along one dimension."""
        if length <= self.tile_size:
            return [0]

        positions = list(range(0, length - self.tile_size + 1, step))
        if positions[-1] != length - self.tile_size:
            positions.append(length - self.tile_size)
        return positions

    def _prepare_batch(
        self,
        tiles: List[torch.Tensor],
        target_batch_size: int,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Prepare a batch of tiles for RF-DETR inference with normalization and padding.

        This method handles three preprocessing steps:
        1. Stacks tiles into a batch tensor and transfers to device
        2. Normalizes each tile independently (per-tile min-max normalization)
        3. Pads batch to target_batch_size for consistent GPU utilization
        4. Converts to list format required by RF-DETR API

        Args:
            tiles: List of tile tensors with shape [3, H, W]
            target_batch_size: Desired batch size (will pad if needed)

        Returns:
            Tuple of:
                - List of normalized tensors ready for RF-DETR inference
                - Number of actual (non-padded) tiles in batch
        """
        if not tiles:
            return [], 0

        # Stack and transfer to device
        batch_tensor = torch.stack(tiles, dim=0).to(self.device, non_blocking=True)

        # Per-tile normalization (each tile normalized independently)
        tile_min = batch_tensor.amin(dim=(-2, -1), keepdim=True)
        tile_max = batch_tensor.amax(dim=(-2, -1), keepdim=True)
        tile_range = (tile_max - tile_min).clamp_min(self.NORMALIZATION_EPSILON)
        batch_tensor = (batch_tensor - tile_min) / tile_range

        # Pad batch to target size if needed
        actual_count = batch_tensor.shape[0]
        pad_count = max(0, target_batch_size - actual_count)

        if pad_count > 0:
            fill_value = batch_tensor.mean().item() if actual_count > 0 else 0.0
            pad_tensor = torch.full(
                (pad_count, 3, self.tile_size, self.tile_size),
                fill_value,
                dtype=batch_tensor.dtype,
                device=batch_tensor.device,
            )
            batch_tensor = torch.cat([batch_tensor, pad_tensor], dim=0)

        # Convert to list format for RF-DETR
        tensor_batch_list = [img for img in batch_tensor]

        return tensor_batch_list, actual_count

    def _merge_detections(
        self,
        batch: SegmentationBatch,
        detections: Sequence,
        return_raw_count: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
        """
        Merge per-tile detections into a full-frame labeled mask and detection arrays.

        This method implements class-specific merging strategies:
        1. Collects all bounding boxes, scores, and classes from tiles
        2. Adjusts coordinates to full-frame space
        3. Separates clump detections (class 1) from other classes
        4. Applies union-based merging to clumps (combines overlapping clumps)
        5. Applies traditional NMS to non-clump classes (suppresses duplicates)
        6. Combines results and stitches instance masks into a single labeled mask

        The union-based merging for clumps prevents fragmentation of large bacterial
        clusters that span multiple tiles, while traditional NMS is used for discrete
        objects like single cells and debris.

        Args:
            batch: Segmentation batch with tile metadata
            detections: Sequence of per-tile detection results
            return_raw_count: If True, also return the raw detection count before merging

        Returns:
            labeled_mask, boxes, class_ids, scores[, raw_count]
        """
        boxes: List[torch.Tensor] = []
        class_ids: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        areas: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        offsets_list: List[Tuple[int, int]] = []

        for det, (offset_x, offset_y), (valid_h, valid_w) in zip(
            detections, batch.offsets, batch.valid_shapes
        ):
            if det is None or len(det) == 0:
                continue

            tile_boxes = torch.from_numpy(det.xyxy)
            tile_scores = torch.from_numpy(det.confidence)
            tile_classes = torch.from_numpy(det.class_id)

            # Adjust bounding box coordinates to full-frame space
            tile_boxes[:, 0::2] += offset_x
            tile_boxes[:, 1::2] += offset_y

            # Clamp to valid region
            if valid_h > 0 and valid_w > 0:
                max_x = offset_x + valid_w
                max_y = offset_y + valid_h
                tile_boxes[:, 0::2] = tile_boxes[:, 0::2].clamp(max=max_x)
                tile_boxes[:, 1::2] = tile_boxes[:, 1::2].clamp(max=max_y)

            box_widths = (tile_boxes[:, 2] - tile_boxes[:, 0]).clamp(min=0)
            box_heights = (tile_boxes[:, 3] - tile_boxes[:, 1]).clamp(min=0)
            tile_box_areas = box_widths * box_heights
            tile_area_values = tile_box_areas

            if hasattr(det, "mask") and det.mask is not None and len(det.mask) > 0:
                tile_masks = torch.from_numpy(det.mask)
                masks.append(tile_masks)
                offsets_list.extend([(offset_x, offset_y)] * len(tile_masks))

                mask_binary = tile_masks > self.mask_threshold
                mask_areas = mask_binary.flatten(1).sum(dim=1).float()
                tile_area_values = torch.where(
                    mask_areas > 0, mask_areas, tile_box_areas.float()
                )

            boxes.append(tile_boxes)
            scores.append(tile_scores)
            class_ids.append(tile_classes)
            areas.append(tile_area_values.float())

        if not boxes:
            height, width = batch.image_shape
            result = (
                np.zeros((height, width), dtype=np.int32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )
            if return_raw_count:
                return result + (0,)
            return result

        boxes_tensor = torch.cat(boxes, dim=0).float()
        scores_tensor = torch.cat(scores, dim=0).float()
        classes_tensor = torch.cat(class_ids, dim=0).long()
        areas_tensor = torch.cat(areas, dim=0).float()

        raw_detection_count = boxes_tensor.shape[0]

        height, width = batch.image_shape
        boxes_tensor[:, 0::2] = boxes_tensor[:, 0::2].clamp(0, width)
        boxes_tensor[:, 1::2] = boxes_tensor[:, 1::2].clamp(0, height)
        # Concatenate all masks once if they exist
        masks_tensor = torch.cat(masks, dim=0) if masks else None
        offsets_tensor = (
            torch.tensor(offsets_list, dtype=torch.long)
            if masks_tensor is not None and offsets_list
            else None
        )

        # Separate clump detections (class 1) from other classes
        clump_mask = classes_tensor == 1
        non_clump_mask = ~clump_mask

        clump_indices = torch.where(clump_mask)[0]
        non_clump_indices = torch.where(non_clump_mask)[0]

        # Process clumps with union-based merging (mask overlap in global coordinates)
        if clump_indices.numel() > 0:
            clump_boxes = boxes_tensor[clump_indices]
            clump_scores = scores_tensor[clump_indices]
            clump_masks_tensor = (
                masks_tensor[clump_indices] if masks_tensor is not None else None
            )
            clump_offsets = (
                offsets_tensor[clump_indices] if offsets_tensor is not None else None
            )

            if clump_masks_tensor is not None and clump_offsets is not None:
                merged_clump_boxes, merged_clump_scores, merged_clump_masks, merged_clump_offsets = (
                    self._union_merge_clumps(
                        clump_boxes,
                        clump_scores,
                        clump_masks_tensor,
                        clump_offsets,
                    )
                )
            else:
                merged_clump_boxes = clump_boxes
                merged_clump_scores = clump_scores
                merged_clump_masks = []
                merged_clump_offsets = []
        else:
            merged_clump_boxes = torch.empty((0, 4), dtype=torch.float32)
            merged_clump_scores = torch.empty((0,), dtype=torch.float32)
            merged_clump_masks = []
            merged_clump_offsets = []

        # Process non-clumps with traditional NMS
        if non_clump_indices.numel() > 0:
            non_clump_boxes = boxes_tensor[non_clump_indices]
            non_clump_scores = scores_tensor[non_clump_indices]
            non_clump_areas = areas_tensor[non_clump_indices]

            keep_indices_nms = self._cross_class_nms(
                non_clump_boxes, non_clump_scores, non_clump_areas
            )
            kept_non_clump_indices = non_clump_indices[keep_indices_nms]
            kept_non_clump_boxes = boxes_tensor[kept_non_clump_indices]
            kept_non_clump_scores = scores_tensor[kept_non_clump_indices]
            kept_non_clump_classes = classes_tensor[kept_non_clump_indices]

            if masks_tensor is not None:
                kept_non_clump_masks = [
                    masks_tensor[i] for i in kept_non_clump_indices.tolist()
                ]
                kept_non_clump_offsets = [
                    offsets_list[i] for i in kept_non_clump_indices.tolist()
                ]
            else:
                kept_non_clump_masks = []
                kept_non_clump_offsets = []
        else:
            kept_non_clump_boxes = torch.empty((0, 4), dtype=torch.float32)
            kept_non_clump_scores = torch.empty((0,), dtype=torch.float32)
            kept_non_clump_classes = torch.empty((0,), dtype=torch.long)
            kept_non_clump_masks = []
            kept_non_clump_offsets = []

        clump_class_ids = torch.full(
            (merged_clump_boxes.shape[0],), 1, dtype=classes_tensor.dtype
        )
        boxes_tensor_final = torch.cat([merged_clump_boxes, kept_non_clump_boxes], dim=0)
        classes_tensor_final = torch.cat([clump_class_ids, kept_non_clump_classes], dim=0)
        scores_tensor_final = torch.cat([merged_clump_scores, kept_non_clump_scores], dim=0)

        boxes_np = boxes_tensor_final.numpy()
        classes_np = classes_tensor_final.numpy()
        scores_np = scores_tensor_final.numpy()

        if masks_tensor is not None:
            masks_for_stitching = []
            offsets_for_stitching = []
            if merged_clump_masks:
                masks_for_stitching.extend(merged_clump_masks)
                offsets_for_stitching.extend(merged_clump_offsets)
            if kept_non_clump_masks:
                masks_for_stitching.extend(kept_non_clump_masks)
                offsets_for_stitching.extend(kept_non_clump_offsets)

            classes_for_stitching = []
            if merged_clump_masks:
                classes_for_stitching.extend([1] * len(merged_clump_masks))
            if kept_non_clump_masks:
                classes_for_stitching.extend([int(c) for c in kept_non_clump_classes.tolist()])

            if masks_for_stitching:
                priority = self.PRIORITY_CLASS_ORDER
                order = sorted(
                    range(len(classes_for_stitching)),
                    key=lambda idx: priority.get(int(classes_for_stitching[idx]), 0),
                )
                masks_for_stitching = [masks_for_stitching[idx] for idx in order]
                offsets_for_stitching = [offsets_for_stitching[idx] for idx in order]
                classes_for_stitching = [classes_for_stitching[idx] for idx in order]
                boxes_np = boxes_np[order]
                classes_np = classes_np[order]
                scores_np = scores_np[order]

            labeled_mask, mask_to_detection_map = self._stitch_masks(
                masks_for_stitching, offsets_for_stitching, classes_for_stitching, batch.image_shape
            )

            if len(mask_to_detection_map) > 0:
                classes_np = classes_np[mask_to_detection_map]
                scores_np = scores_np[mask_to_detection_map]
                boxes_np = boxes_np[mask_to_detection_map]

                # Recalculate bboxes from surviving mask pixels and get mask areas
                # This ensures bboxes match the actual mask after priority stitching
                boxes_np, mask_areas = self._recalculate_bboxes_from_mask(
                    labeled_mask, len(mask_to_detection_map), return_areas=True
                )

                # Filter out tiny masks (< min_mask_area pixels)
                # These are fragments from partial swallowing that can't form valid polygons
                keep_mask = mask_areas >= self.min_mask_area
                if not keep_mask.all():
                    keep_indices = np.where(keep_mask)[0]
                    boxes_np = boxes_np[keep_indices]
                    classes_np = classes_np[keep_indices]
                    scores_np = scores_np[keep_indices]
                    # Relabel the mask to remove filtered labels and make consecutive
                    labeled_mask = self._relabel_mask(labeled_mask, keep_mask)
        else:
            labeled_mask = np.zeros(batch.image_shape, dtype=np.int32)

        if return_raw_count:
            return labeled_mask, boxes_np, classes_np, scores_np, raw_detection_count
        return labeled_mask, boxes_np, classes_np, scores_np

    def _cross_class_nms(
        self, boxes: torch.Tensor, scores: torch.Tensor, areas: torch.Tensor
    ) -> torch.Tensor:
        """
        Suppress overlapping detections across classes, preferring highest confidence.

        Uses torchvision.ops.nms for GPU-accelerated suppression. Area is used as a
        tiebreaker when scores are equal (larger area wins).

        Args:
            boxes: [N, 4] tensor of XYXY boxes
            scores: [N] tensor of confidence scores
            areas: [N] tensor with instance areas (mask area preferred, bbox area fallback)

        Returns:
            Indices of detections kept after suppression.
        """
        num_instances = boxes.shape[0]
        if num_instances == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        # Add tiny area-based epsilon to scores for tiebreaking (preserves original behavior)
        # Normalize areas to [0, 1e-7] range so it only affects exact ties
        if areas.numel() > 0 and areas.max() > 0:
            area_tiebreaker = (areas / areas.max()) * 1e-7
        else:
            area_tiebreaker = torch.zeros_like(scores)
        adjusted_scores = scores + area_tiebreaker

        # Use GPU-accelerated NMS from torchvision
        keep = nms(boxes, adjusted_scores, self.nms_iou)

        return keep
    def _union_merge_clumps(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        masks: Optional[torch.Tensor],
        offsets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[Tuple[int, int]]]:
        """
        Merge overlapping clump detections using mask overlap in global coordinates.

        This method uses ONLY mask overlap (no IoU criterion) and computes overlap
        in the full-frame coordinate system using tile offsets. The merged mask is
        built in the union bounding box frame and stitched using its global offset.

        Optimized with vectorized bbox intersection to avoid O(N²) Python loops.
        """
        num_clumps = boxes.shape[0]
        if num_clumps == 0:
            return boxes, scores, [], []

        if masks is None or offsets is None:
            return boxes, scores, [], []

        mask_bins = masks > self.mask_threshold
        offsets = offsets.to(torch.long)
        min_overlap = max(1, int(self.clump_merge_min_overlap))
        mask_h, mask_w = mask_bins.shape[1], mask_bins.shape[2]

        # === VECTORIZED BBOX INTERSECTION ===
        # Compute all pairwise bbox intersections at once using broadcasting
        # boxes shape: [N, 4] where each row is [x1, y1, x2, y2]
        # We need intersection between boxes[i] and boxes[j] for all i < j

        # Extract coordinates for broadcasting
        x1_all = boxes[:, 0]  # [N]
        y1_all = boxes[:, 1]  # [N]
        x2_all = boxes[:, 2]  # [N]
        y2_all = boxes[:, 3]  # [N]

        # Compute pairwise intersection coordinates using broadcasting
        # [N, 1] op [1, N] -> [N, N]
        inter_x1 = torch.maximum(x1_all.unsqueeze(1), x1_all.unsqueeze(0))  # [N, N]
        inter_y1 = torch.maximum(y1_all.unsqueeze(1), y1_all.unsqueeze(0))  # [N, N]
        inter_x2 = torch.minimum(x2_all.unsqueeze(1), x2_all.unsqueeze(0))  # [N, N]
        inter_y2 = torch.minimum(y2_all.unsqueeze(1), y2_all.unsqueeze(0))  # [N, N]

        # Compute intersection widths and heights (clamped to 0)
        inter_w = (inter_x2 - inter_x1).clamp(min=0)  # [N, N]
        inter_h = (inter_y2 - inter_y1).clamp(min=0)  # [N, N]
        inter_area = inter_w * inter_h  # [N, N]

        # === TILE ADJACENCY FILTERING ===
        # Clumps can only overlap if their tiles are adjacent (within tile_size distance)
        # This filters out pairs that cannot possibly have mask overlap
        off_x = offsets[:, 0].float()  # [N]
        off_y = offsets[:, 1].float()  # [N]

        # Compute pairwise tile distances
        tile_dist_x = torch.abs(off_x.unsqueeze(1) - off_x.unsqueeze(0))  # [N, N]
        tile_dist_y = torch.abs(off_y.unsqueeze(1) - off_y.unsqueeze(0))  # [N, N]

        # Adjacent tiles are within tile_size + some margin (masks can extend to tile edges)
        max_tile_dist = self.tile_size + mask_w  # Conservative: allows for mask extent
        adjacent_mask = (tile_dist_x <= max_tile_dist) & (tile_dist_y <= max_tile_dist)

        # Find candidate pairs: upper triangle (i < j), has bbox intersection, tiles adjacent
        # Only check upper triangle to avoid duplicate pairs
        upper_tri = torch.triu(torch.ones(num_clumps, num_clumps, dtype=torch.bool), diagonal=1)
        candidate_mask = upper_tri & (inter_area >= min_overlap) & adjacent_mask

        # Get indices of candidate pairs
        candidate_pairs = torch.nonzero(candidate_mask, as_tuple=False)  # [K, 2]

        # === UNION-FIND WITH MASK OVERLAP CHECK ===
        parent = list(range(num_clumps))

        def find(idx: int) -> int:
            root = idx
            while parent[root] != root:
                root = parent[root]
            # Path compression
            while parent[idx] != root:
                next_idx = parent[idx]
                parent[idx] = root
                idx = next_idx
            return root

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Pre-extract offsets as numpy for faster indexing in loop
        offsets_np = offsets.cpu().numpy()

        # Check mask overlap only for candidate pairs (much smaller than N²)
        for pair_idx in range(candidate_pairs.shape[0]):
            i = int(candidate_pairs[pair_idx, 0].item())
            j = int(candidate_pairs[pair_idx, 1].item())

            # Get precomputed intersection bounds (already computed above)
            gx1 = int(math.floor(inter_x1[i, j].item()))
            gy1 = int(math.floor(inter_y1[i, j].item()))
            gx2 = int(math.ceil(inter_x2[i, j].item()))
            gy2 = int(math.ceil(inter_y2[i, j].item()))

            off_ix, off_iy = int(offsets_np[i, 0]), int(offsets_np[i, 1])
            off_jx, off_jy = int(offsets_np[j, 0]), int(offsets_np[j, 1])

            # Compute local mask coordinates
            ix1 = max(gx1 - off_ix, 0)
            iy1 = max(gy1 - off_iy, 0)
            ix2 = min(gx2 - off_ix, mask_w)
            iy2 = min(gy2 - off_iy, mask_h)

            jx1 = max(gx1 - off_jx, 0)
            jy1 = max(gy1 - off_jy, 0)
            jx2 = min(gx2 - off_jx, mask_w)
            jy2 = min(gy2 - off_jy, mask_h)

            w = min(ix2 - ix1, jx2 - jx1)
            h = min(iy2 - iy1, jy2 - jy1)
            if w <= 0 or h <= 0:
                continue

            # Compute actual mask overlap
            overlap = (
                mask_bins[i, iy1:iy1 + h, ix1:ix1 + w]
                & mask_bins[j, jy1:jy1 + h, jx1:jx1 + w]
            ).sum().item()

            if overlap >= min_overlap:
                union(i, j)

        # === BUILD MERGED GROUPS ===
        groups = {}
        for idx in range(num_clumps):
            root = find(idx)
            groups.setdefault(root, []).append(idx)

        merged_entries = []
        for group in groups.values():
            group_tensor = torch.tensor(group, dtype=torch.long)
            group_boxes = boxes[group_tensor]

            # Vectorized min/max for union bbox
            x1 = int(torch.floor(group_boxes[:, 0].min()).item())
            y1 = int(torch.floor(group_boxes[:, 1].min()).item())
            x2 = int(torch.ceil(group_boxes[:, 2].max()).item())
            y2 = int(torch.ceil(group_boxes[:, 3].max()).item())

            union_h = max(1, y2 - y1)
            union_w = max(1, x2 - x1)
            union_mask = torch.zeros((union_h, union_w), dtype=torch.bool)

            for idx in group:
                off_x = int(offsets_np[idx, 0])
                off_y = int(offsets_np[idx, 1])
                mask_bin = mask_bins[idx]

                gx1 = max(x1, off_x)
                gy1 = max(y1, off_y)
                gx2 = min(x2, off_x + mask_w)
                gy2 = min(y2, off_y + mask_h)
                if gx2 <= gx1 or gy2 <= gy1:
                    continue

                ux1 = gx1 - x1
                uy1 = gy1 - y1
                ux2 = gx2 - x1
                uy2 = gy2 - y1

                mx1 = gx1 - off_x
                my1 = gy1 - off_y
                mx2 = gx2 - off_x
                my2 = gy2 - off_y

                union_mask[uy1:uy2, ux1:ux2] |= mask_bin[my1:my2, mx1:mx2]

            merged_box = torch.tensor([x1, y1, x2, y2], dtype=boxes.dtype)
            merged_score = scores[group_tensor].mean()
            merged_entries.append((merged_score.item(), merged_box, union_mask, (x1, y1)))

        merged_entries.sort(key=lambda x: x[0], reverse=True)

        merged_boxes = torch.stack([e[1] for e in merged_entries], dim=0)
        merged_scores = torch.tensor([e[0] for e in merged_entries], dtype=scores.dtype)
        merged_masks = [e[2] for e in merged_entries]
        merged_offsets = [e[3] for e in merged_entries]

        return merged_boxes, merged_scores, merged_masks, merged_offsets
    def _stitch_masks(
        self,
        masks: Sequence[torch.Tensor],
        offsets: List[Tuple[int, int]],
        classes: Sequence[int],
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stitch instance masks into a single full-frame labeled mask.

        Higher-priority classes can overwrite lower-priority labels if the overlap
        covers a sufficient fraction of the existing label. Priority order:
        clump > debris > single-cell.

        Optimized with:
        - Batch mask conversion to numpy
        - Vectorized overlap counting using bincount
        - Deferred label removal (avoid full-image scans)
        """
        height, width = image_shape
        labeled_mask = np.zeros((height, width), dtype=np.int32)

        if isinstance(masks, torch.Tensor):
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            mask_list = list(masks)

        if not mask_list:
            return labeled_mask, np.array([], dtype=np.int32)

        classes_list = list(classes)
        if len(classes_list) < len(mask_list):
            classes_list.extend([0] * (len(mask_list) - len(classes_list)))

        class_priority = self.PRIORITY_CLASS_ORDER
        threshold = self.mask_threshold
        priority_overlap_frac = self.priority_overlap_fraction

        # Pre-convert all torch masks to numpy (batch operation)
        # and pre-binarize to avoid repeated threshold comparisons
        mask_np_list = []
        for mask in mask_list:
            if mask is None:
                mask_np_list.append(None)
            elif isinstance(mask, torch.Tensor):
                mask_np_list.append(mask.cpu().numpy())
            else:
                mask_np_list.append(np.asarray(mask))

        label_class: List[int] = []
        label_area: List[int] = []
        label_detection_idx: List[int] = []
        # Track labels to remove at end (avoid full-image scans during loop)
        labels_to_remove = set()

        for detection_idx, (mask_np, (offset_x, offset_y), class_id) in enumerate(
            zip(mask_np_list, offsets, classes_list)
        ):
            if mask_np is None:
                continue

            mask_h, mask_w = mask_np.shape
            y_start = int(offset_y)
            y_end = min(y_start + mask_h, height)
            x_start = int(offset_x)
            x_end = min(x_start + mask_w, width)

            mask_crop_h = y_end - y_start
            mask_crop_w = x_end - x_start
            if mask_crop_h <= 0 or mask_crop_w <= 0:
                continue

            # Binarize mask (threshold applied once)
            binary_mask = mask_np[:mask_crop_h, :mask_crop_w] > threshold
            if not binary_mask.any():
                continue

            roi = labeled_mask[y_start:y_end, x_start:x_end]
            overlap_labels = roi[binary_mask]
            allowed_mask = binary_mask.copy()
            overwrite_mask = np.zeros_like(binary_mask, dtype=bool)

            if overlap_labels.size > 0:
                incoming_priority = class_priority.get(int(class_id), 0)

                # Use bincount for vectorized overlap counting
                # This counts how many pixels of each label are overlapped
                max_label = len(label_class)
                if max_label > 0:
                    # bincount gives counts for labels 0..max in overlap_labels
                    overlap_counts = np.bincount(overlap_labels, minlength=max_label + 1)

                    # Process labels that have overlap (skip 0 = background)
                    overlapping_labels = np.nonzero(overlap_counts[1:])[0] + 1

                    for label_id in overlapping_labels:
                        label_idx = label_id - 1
                        label_mask = (roi == label_id) & binary_mask

                        # Already-removed labels are still present in the mask until deferred cleanup.
                        # Allow incoming pixels to overwrite them now so we do not leave holes.
                        if label_area[label_idx] <= 0:
                            overwrite_mask[label_mask] = True
                            continue

                        existing_priority = class_priority.get(label_class[label_idx], 0)

                        if incoming_priority <= existing_priority:
                            allowed_mask[label_mask] = False
                            continue

                        existing_area = label_area[label_idx]
                        overlap_count = overlap_counts[label_id]
                        overlap_fraction = overlap_count / float(existing_area)

                        if overlap_fraction >= priority_overlap_frac:
                            # Mark for deferred removal (avoid full-image scan now)
                            labels_to_remove.add(int(label_id))
                            label_area[label_idx] = 0
                            overwrite_mask[label_mask] = True
                        else:
                            allowed_mask[label_mask] = False

            # Re-fetch ROI in case we need updated view
            roi = labeled_mask[y_start:y_end, x_start:x_end]
            new_pixels = allowed_mask & ((roi == 0) | overwrite_mask)

            if new_pixels.any():
                label_id = len(label_class) + 1
                roi[new_pixels] = label_id  # Direct assignment is faster than np.where
                label_class.append(int(class_id))
                label_area.append(int(new_pixels.sum()))
                label_detection_idx.append(detection_idx)

        # Batch remove dead labels at the end (single pass through labeled_mask)
        if labels_to_remove:
            # Create removal mask: True for labels to remove
            max_label = len(label_class)
            remove_mask = np.zeros(max_label + 1, dtype=bool)
            for label_id in labels_to_remove:
                if label_id <= max_label:
                    remove_mask[label_id] = True

            # Single vectorized removal
            labeled_mask[remove_mask[labeled_mask]] = 0

        # Build final mapping: compact labels and map to detection indices
        alive_mask = np.array([area > 0 for area in label_area], dtype=bool)
        if not alive_mask.all():
            old_to_new = np.zeros(len(label_area) + 1, dtype=np.int32)
            new_detection_map = []
            for idx, alive in enumerate(alive_mask):
                if alive:
                    new_id = len(new_detection_map) + 1
                    old_to_new[idx + 1] = new_id
                    new_detection_map.append(label_detection_idx[idx])
            labeled_mask = old_to_new[labeled_mask]
            mask_to_detection_map = np.array(new_detection_map, dtype=np.int32)
        else:
            mask_to_detection_map = np.array(label_detection_idx, dtype=np.int32)

        return labeled_mask, mask_to_detection_map

    def _recalculate_bboxes_from_mask(
        self,
        labeled_mask: np.ndarray,
        n_instances: int,
        return_areas: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Recalculate bounding boxes from the actual mask pixels in labeled_mask.

        After priority-based stitching, masks may be partially swallowed by higher-priority
        masks, leaving only a fragment of the original detection. This method computes
        tight bounding boxes around the surviving mask pixels to ensure bbox-mask consistency.

        Args:
            labeled_mask: [H, W] array with integer labels (0=background, 1-N=instances)
            n_instances: Number of instances (labels 1 to n_instances)
            return_areas: If True, also return mask areas for each instance

        Returns:
            If return_areas=False: [N, 4] array of bounding boxes in xyxy format
            If return_areas=True: Tuple of (boxes [N, 4], areas [N])
        """
        if n_instances == 0:
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            if return_areas:
                return empty_boxes, np.empty((0,), dtype=np.int64)
            return empty_boxes

        boxes = np.zeros((n_instances, 4), dtype=np.float32)
        areas = np.zeros((n_instances,), dtype=np.int64) if return_areas else None

        for label_id in range(1, n_instances + 1):
            # Find all pixels with this label
            ys, xs = np.where(labeled_mask == label_id)
            idx = label_id - 1

            if len(xs) == 0:
                # Label doesn't exist in mask (shouldn't happen after filtering)
                # Set to zero-size box at origin
                boxes[idx] = [0, 0, 0, 0]
                if return_areas:
                    areas[idx] = 0
            else:
                # Compute tight bounding box (xyxy format)
                x1 = float(xs.min())
                y1 = float(ys.min())
                x2 = float(xs.max() + 1)  # +1 because bbox should include the pixel
                y2 = float(ys.max() + 1)
                boxes[idx] = [x1, y1, x2, y2]
                if return_areas:
                    areas[idx] = len(xs)

        if return_areas:
            return boxes, areas
        return boxes

    def _relabel_mask(
        self,
        labeled_mask: np.ndarray,
        keep_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Relabel a labeled mask to remove filtered labels and make labels consecutive.

        Args:
            labeled_mask: [H, W] array with integer labels (0=background, 1-N=instances)
            keep_mask: [N] boolean array indicating which labels to keep

        Returns:
            Relabeled mask with consecutive labels 1, 2, ..., M where M = sum(keep_mask)
        """
        n_labels = len(keep_mask)
        # Create mapping: old_label -> new_label (0 for removed labels)
        old_to_new = np.zeros(n_labels + 1, dtype=labeled_mask.dtype)
        new_label = 1
        for old_label in range(1, n_labels + 1):
            if keep_mask[old_label - 1]:
                old_to_new[old_label] = new_label
                new_label += 1
            # else: old_to_new[old_label] remains 0, mapping to background

        # Apply relabeling
        return old_to_new[labeled_mask]
