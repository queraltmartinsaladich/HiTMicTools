# Standard library imports
import json
from typing import Tuple

# Third-party library imports
import numpy as np
import pandas as pd
from scipy.ndimage import label
from skimage.measure import regionprops_table

# Local imports
from HiTMicTools.img_processing.array_ops import adjust_dimensions
from HiTMicTools.roianalysis.roi_utils import (
    compute_label_offsets,
    apply_label_offsets,
    get_optimal_workers,
)

# Type hints


def coords_centroid(coords):
    """Return centroid coordinates as slice/y/x pandas Series."""
    centroid = np.mean(coords, axis=0)
    return pd.Series(centroid, index=["slice", "y", "x"])


def convert_to_list_and_dump(row):
    """Serialize numpy arrays of coordinates into JSON strings."""
    return json.dumps(row.tolist())


class RoiAnalyser:
    """Measure properties of probability maps or labeled masks representing regions of interest."""

    def __init__(self, image, probability_map, stack_order=("TSCXY", "TXY")):
        """
        Normalize dimensionality of image inputs and store tensors for later measurements.

        Args:
            image (np.ndarray): Raw microscopy stack.
            probability_map (np.ndarray): Map of per-pixel probabilities or masks.
            stack_order (Tuple[str, str]): Dimension order for image and mask tensors.
        """
        image = adjust_dimensions(image, stack_order[0])
        probability_map = adjust_dimensions(probability_map, stack_order[1])

        self.img = image
        self.proba_map = probability_map
        self.stack_order = stack_order

        pass

    def get(self, name: str, index=None, to_numpy: bool = True):
        """
        Retrieve stored arrays with optional indexing and NumPy conversion.

        Args:
            name: Data key to fetch ('image', 'probability', 'binary', 'labels').
            index: Optional slice/tuple/list used to index into the array.
            to_numpy: Convert GPU-backed arrays to NumPy before returning.

        Returns:
            Array view or copy depending on the backend.

        Raises:
            ValueError: If the requested data key is unknown or not available.
        """
        data_map = {
            "image": self.img,
            "probability": self.proba_map,
            "probability_map": self.proba_map,
            "binary": getattr(self, "binary_mask", None),
            "binary_mask": getattr(self, "binary_mask", None),
            "labels": getattr(self, "labeled_mask", None),
            "labeled_mask": getattr(self, "labeled_mask", None),
        }

        if name not in data_map:
            raise ValueError(
                f"Unsupported data key '{name}'. "
                "Use one of ['image', 'probability', 'probability_map', 'binary', 'binary_mask', 'labels', 'labeled_mask']."
            )

        arr = data_map[name]
        if arr is None:
            raise ValueError(f"Data '{name}' is not available on this analyser.")

        if index is not None:
            arr = arr[tuple(index) if isinstance(index, (list, tuple)) else index]

        if to_numpy and hasattr(arr, "__cuda_array_interface__"):
            arr = arr.get()

        return arr

    @classmethod
    def from_labeled_mask(
        cls,
        image: np.ndarray,
        labeled_mask: np.ndarray,
        stack_order: Tuple[str, str] = ("TSCXY", "TYX"),
    ) -> "RoiAnalyser":
        """
        Create RoiAnalyser directly from a pre-labeled mask, skipping probability-based segmentation.

        This constructor is useful when you have instance segmentation outputs (e.g., from RF-DETR-Segm)
        that already provide labeled instances, bypassing the need for probability maps and
        connected components analysis.

        Args:
            image: The original microscopy image with shape matching stack_order[0].
            labeled_mask: Pre-labeled instance mask where each unique positive integer
                represents a distinct ROI. Shape should match stack_order[1].
            stack_order: Tuple of (image_order, mask_order) dimension specifications.
                Defaults to ("TSCXY", "TYX") for time-series images with pre-labeled masks.

        Returns:
            RoiAnalyser instance ready for measurements, with labeled_mask already populated.

        Example:
            >>> # From RF-DETR-Segm output
            >>> labeled_mask, _, _, _ = sc_segmenter.predict(image)
            >>> analyser = RoiAnalyser.from_labeled_mask(image, labeled_mask)
            >>> measurements = analyser.get_roi_measurements(target_channel=1)
        """
        instance = cls.__new__(cls)

        # Adjust dimensions to expected format
        instance.img = adjust_dimensions(image, stack_order[0])
        adjusted_mask = adjust_dimensions(labeled_mask, stack_order[1])

        # Set attributes
        instance.labeled_mask = adjusted_mask
        instance.stack_order = stack_order
        instance.proba_map = None  # No probability map in this workflow

        # Derive binary mask from labeled mask
        instance.binary_mask = adjusted_mask > 0

        # Calculate total number of ROIs across all frames
        instance.total_rois = int(np.max(adjusted_mask))

        return instance

    def create_binary_mask(self, threshold=0.5):
        """
        Create a binary mask from an image stack of probabilities.

        Args:
            image_stack (np.ndarray): A 5D numpy array with shape (frames, slices, channels, height, width) containing probabilities.
            threshold (float): The threshold value to use for binarization (default: 0.5).

        Returns:
            np.ndarray: A 5D numpy array with the same shape as the input, where values above the threshold are set to 1, and values below or equal to the threshold are set to 0.
        """

        # Convert probabilities to binary values
        self.binary_mask = self.proba_map > threshold

    def clean_binmask(self, min_pixel_size=20):
        """
        Clean the binary mask by removing small ROIs and cache the labeled result.

        This method computes labels once and caches them to avoid redundant
        computation when get_labels() is called later in the workflow.

        Args:
            min_pixel_size (int): Minimum ROI size in pixels.

        Returns:
            None (updates self.binary_mask in-place)
        """
        # Check if labels already computed (from previous clean_binmask call)
        if not hasattr(self, '_cached_labeled_mask'):
            labeled, num_features = self.get_labels(return_value=True)
        else:
            labeled = self._cached_labeled_mask
            num_features = self._cached_num_features

        # Size filtering
        sizes = np.bincount(labeled.ravel())[1:]  # Exclude background (label 0)
        mask_sizes = sizes >= min_pixel_size
        label_map = np.zeros(num_features + 1, dtype=int)
        label_map[1:][mask_sizes] = np.arange(1, np.sum(mask_sizes) + 1)
        cleaned_labeled = label_map[labeled]
        cleaned_mask = cleaned_labeled > 0

        # Update binary mask
        self.binary_mask = cleaned_mask

        # Cache the cleaned labeled mask for later use by get_labels()
        self._cached_labeled_mask = cleaned_labeled
        self._cached_num_features = int(np.sum(mask_sizes))

    def get_labels(self, return_value=False, n_workers=None):
        """
        Get the labeled mask for the binary mask with vectorized label assignment.

        This method uses a two-pass algorithm that enables vectorization and
        parallelization:
        1. Label each frame independently (can be parallelized)
        2. Compute label offsets using cumsum (vectorized)
        3. Apply offsets (vectorized)

        If clean_binmask() was called previously, this method uses the cached
        labeled mask to avoid redundant computation.

        Args:
            return_value (bool): If True, return (labeled_mask, num_rois) tuple.
                Otherwise, store in instance attributes.
            n_workers (int, optional): Number of parallel workers for labeling.
                If None, uses adaptive worker selection based on data size.
                Set to 1 to disable parallelization.

        Returns:
            tuple or None: (labeled_mask, num_rois) if return_value=True, else None
        """
        # Check if we can use cached labels from clean_binmask()
        if hasattr(self, '_cached_labeled_mask'):
            labeled_mask = self._cached_labeled_mask
            num_rois = self._cached_num_features

            if return_value:
                return labeled_mask, num_rois
            else:
                self.total_rois = num_rois
                self.labeled_mask = labeled_mask
                return

        # Compute labels using optimized vectorized approach
        labeled_mask, num_rois = self._compute_labels_vectorized(n_workers)

        # Cache results for potential future use
        self._cached_labeled_mask = labeled_mask
        self._cached_num_features = num_rois

        if return_value:
            return labeled_mask, num_rois
        else:
            self.total_rois = num_rois
            self.labeled_mask = labeled_mask

    def _compute_labels_vectorized(self, n_workers=None):
        """
        Compute labels using vectorized two-pass algorithm.

        Args:
            n_workers (int, optional): Number of parallel workers. If None, auto-detect.

        Returns:
            tuple: (labeled_mask, total_rois)
        """
        T = self.binary_mask.shape[0]

        # Determine whether to use parallel processing
        if n_workers is None:
            # Auto-detect optimal worker count
            n_workers = get_optimal_workers(self.binary_mask.shape)

        # Pass 1: Label each frame independently
        if n_workers > 1 and T >= 20:
            # Use parallel processing for large datasets
            labeled_mask, num_rois = self._compute_labels_parallel(n_workers)
        else:
            # Use sequential processing for small datasets
            labeled_mask, num_rois = self._compute_labels_sequential()

        return labeled_mask, num_rois

    def _compute_labels_sequential(self):
        """
        Sequential label computation with vectorized offset application.

        Returns:
            tuple: (labeled_mask, total_rois)
        """
        T = self.binary_mask.shape[0]

        # Pass 1: Label each frame independently
        labeled_frames = []
        frame_obj_counts = np.zeros(T, dtype=np.int32)

        for i in range(T):
            labeled_frame, num = label(self.binary_mask[i])
            labeled_frames.append(labeled_frame)
            frame_obj_counts[i] = num

        # Pass 2: Compute label offsets using vectorized cumsum
        label_offsets = compute_label_offsets(frame_obj_counts, xp=np)

        # Pass 3: Apply offsets using vectorization
        labeled_mask = apply_label_offsets(labeled_frames, label_offsets, xp=np)

        total_rois = int(np.sum(frame_obj_counts))

        return labeled_mask, total_rois

    def _compute_labels_parallel(self, n_workers):
        """
        Parallel label computation using joblib.

        Args:
            n_workers (int): Number of parallel workers

        Returns:
            tuple: (labeled_mask, total_rois)
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            # Fall back to sequential if joblib not available
            return self._compute_labels_sequential()

        T = self.binary_mask.shape[0]

        # Helper function for parallel labeling
        def label_single_frame(frame_data):
            """Label a single frame."""
            frame_idx, binary_frame = frame_data
            labeled_frame, num = label(binary_frame)
            return frame_idx, labeled_frame, num

        # Prepare work items
        work_items = [(i, self.binary_mask[i]) for i in range(T)]

        # Parallel labeling (backend='loky' provides better memory handling)
        results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(label_single_frame)(item) for item in work_items
        )

        # Sort results by frame index (should already be ordered, but be explicit)
        results.sort(key=lambda x: x[0])

        # Extract labeled frames and counts
        labeled_frames = [r[1] for r in results]
        frame_obj_counts = np.array([r[2] for r in results], dtype=np.int32)

        # Vectorized offset computation
        label_offsets = compute_label_offsets(frame_obj_counts, xp=np)

        # Vectorized offset application
        labeled_mask = apply_label_offsets(labeled_frames, label_offsets, xp=np)

        total_rois = int(np.sum(frame_obj_counts))

        return labeled_mask, total_rois

    def get_roi_measurements(
        self,
        target_channel=0,
        target_slice=0,
        properties=["label", "centroid", "mean_intensity"],
        extra_properties=None,
        frame_extra_properties=None,
        n_workers=None,
    ):
        """
        Get measurements for each ROI in the labeled mask for a specific channel and all frames.

        This method supports parallel processing for improved performance on multi-core systems.
        For large datasets (>50 frames), parallel processing can provide 5-15x speedup.

        Args:
            target_channel (int): Channel index to measure
            target_slice (int): Slice index to measure
            properties (list): List of properties to measure for each ROI.
                Defaults to ['label', 'centroid', 'mean_intensity'].
            extra_properties (list, optional): Additional custom properties to measure
            n_workers (int, optional): Number of parallel workers. If None, auto-detect
                based on data size. Set to 1 to disable parallelization.

        Returns:
            pandas.DataFrame: DataFrame containing ROI measurements with columns:
                - label: ROI label (unique identifier)
                - frame: Frame number
                - slice: Slice index
                - channel: Channel index
                - [properties]: Requested measurement properties
        """
        assert self.labeled_mask is not None, (
            "Run get_labels() first to generate labeled mask"
        )

        img = self.img[:, target_slice, target_channel, :, :]
        labeled_mask = self.labeled_mask[:, target_slice, 0, :, :]
        n_frames = img.shape[0]

        # Determine whether to use parallel processing
        if n_workers is None:
            # Auto-detect optimal worker count
            n_workers = get_optimal_workers(img.shape)

        # Use parallel processing for large datasets
        if n_workers > 1 and n_frames >= 20:
            all_roi_properties_df = self._get_roi_measurements_parallel(
                img, labeled_mask, properties, extra_properties,
                target_channel, target_slice, n_workers,
                frame_extra_properties=frame_extra_properties,
            )
        else:
            all_roi_properties_df = self._get_roi_measurements_sequential(
                img, labeled_mask, properties, extra_properties,
                target_channel, target_slice,
                frame_extra_properties=frame_extra_properties,
            )

        # Process coordinates if present
        if "coords" in all_roi_properties_df.columns:
            all_roi_properties_df["coords"] = all_roi_properties_df["coords"].apply(
                convert_to_list_and_dump
            )

        # Rearrange columns
        required_cols = ["label", "frame", "slice", "channel"]
        other_cols = [
            col for col in all_roi_properties_df.columns if col not in required_cols
        ]
        cols = required_cols + other_cols
        all_roi_properties_df = all_roi_properties_df[cols]

        return all_roi_properties_df

    def _get_roi_measurements_sequential(
        self, img, labeled_mask, properties, extra_properties,
        target_channel, target_slice, frame_extra_properties=None
    ):
        """
        Sequential ROI measurement computation.

        Args:
            img: Image array (T, H, W)
            labeled_mask: Labeled mask array (T, H, W)
            properties: List of properties to measure
            extra_properties: Additional custom properties
            target_channel: Channel index
            target_slice: Slice index

        Returns:
            pandas.DataFrame: Measurements for all ROIs
        """
        all_roi_properties = []

        for frame in range(img.shape[0]):
            img_frame = img[frame]
            labeled_mask_frame = labeled_mask[frame]

            roi_properties = regionprops_table(
                labeled_mask_frame,
                intensity_image=img_frame,
                properties=properties,
                separator="_",
                extra_properties=extra_properties,
            )
            roi_properties_df = pd.DataFrame(roi_properties)
            if frame_extra_properties:
                for frame_fn in frame_extra_properties:
                    frame_df = frame_fn(labeled_mask_frame, img_frame)
                    roi_properties_df = roi_properties_df.merge(frame_df, on="label", how="left")
            roi_properties_df["frame"] = frame
            roi_properties_df["slice"] = target_channel
            roi_properties_df["channel"] = target_slice

            all_roi_properties.append(roi_properties_df)

        return pd.concat(all_roi_properties, ignore_index=True)

    def _get_roi_measurements_parallel(
        self, img, labeled_mask, properties, extra_properties,
        target_channel, target_slice, n_workers, frame_extra_properties=None
    ):
        """
        Parallel ROI measurement computation using joblib.

        Args:
            img: Image array (T, H, W)
            labeled_mask: Labeled mask array (T, H, W)
            properties: List of properties to measure
            extra_properties: Additional custom properties
            target_channel: Channel index
            target_slice: Slice index
            n_workers: Number of parallel workers

        Returns:
            pandas.DataFrame: Measurements for all ROIs
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            # Fall back to sequential if joblib not available
            return self._get_roi_measurements_sequential(
                img, labeled_mask, properties, extra_properties,
                target_channel, target_slice
            )

        def measure_single_frame(frame_data):
            """Measure ROIs in a single frame."""
            frame_idx, img_frame, labeled_mask_frame = frame_data

            roi_properties = regionprops_table(
                labeled_mask_frame,
                intensity_image=img_frame,
                properties=properties,
                separator="_",
                extra_properties=extra_properties,
            )
            roi_properties_df = pd.DataFrame(roi_properties)
            if frame_extra_properties:
                for frame_fn in frame_extra_properties:
                    frame_df = frame_fn(labeled_mask_frame, img_frame)
                    roi_properties_df = roi_properties_df.merge(frame_df, on="label", how="left")
            return frame_idx, roi_properties_df

        # Prepare work items
        work_items = [
            (i, img[i], labeled_mask[i])
            for i in range(img.shape[0])
        ]

        # Parallel measurement (backend='loky' for better memory handling)
        results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(measure_single_frame)(item) for item in work_items
        )

        # Sort results by frame index
        results.sort(key=lambda x: x[0])

        # Build DataFrames with frame metadata
        all_roi_properties = []
        for frame_idx, roi_properties_df in results:
            roi_properties_df["frame"] = frame_idx
            roi_properties_df["slice"] = target_channel
            roi_properties_df["channel"] = target_slice
            all_roi_properties.append(roi_properties_df)

        # Single concatenation
        return pd.concat(all_roi_properties, ignore_index=True)
