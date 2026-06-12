import os
import gc
import tifffile
from typing import Optional, List, Dict, Tuple
import torch
import pandas as pd
import numpy as np

# Local imports
from HiTMicTools.resource_management.memlogger import MemoryLogger
from HiTMicTools.resource_management.sysutils import (
    empty_gpu_cache,
    get_device,
    )
from HiTMicTools.resource_management.reserveresource import ReserveResource
from HiTMicTools.pipelines.base_pipeline import BasePipeline
from HiTMicTools.img_processing.img_processor import ImagePreprocessor
from HiTMicTools.img_processing.array_ops import convert_image
from HiTMicTools.img_processing.img_ops import measure_background_intensity
from HiTMicTools.img_processing.mask_ops import (
    map_predictions_to_labels_by_frame,
    apply_fl_union_mask,
)
from HiTMicTools.img_processing.morphology_corrections import (
    apply_semSeg_morphology_corrections,
)
from HiTMicTools.utils import remove_file_extension
from HiTMicTools.roianalysis import RoiAnalyser
from HiTMicTools.data_analysis.analysis_tools import (
    roi_skewness, roi_std_dev, roi_glcm_features, roi_radial_profile,
    roi_skeleton_features, roi_shape_features, roi_tubularness,
    roi_skeleton_branch_points,
)

# TODO: Currently, I can use the cupy based ROI analyser, but performance is lagging.
# I will start working with the CPU-based ROI analyser and slowly move to the GPU-based.
# if get_device() == torch.device("cuda"):
#    from HiTMicTools.roi_analyser_gpu import RoiAnalyser, roi_skewness, roi_std_dev
#    import GPUtil
#    print('using CUDA based ROI analyser')
#
#    from HiTMicTools.roi_analyser import RoiAnalyser
#    from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev
#    print('using CPU based ROI analyser')
#
# else:
#    print('using CPU based ROI analyser')
#    from HiTMicTools.roi_analyser import RoiAnalyser
#    from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev


from jetraw_tools.image_reader import ImageReader


class ASCT_singleFrame(BasePipeline):
    """
    Pipeline for automated single-cell analysis on SINGLE-FRAME images.

    Optimized for static microscopy images. Performs focus restoration,
    segmentation, classification, and fluorescence analysis without
    temporal tracking overhead.

    Architecture differences from ASCT_semSeg:
    - Native single-frame processing (no frame iteration overhead)
    - Simplified classification workflow (direct 2D processing)
    - Optional frame expansion for compatibility with shared utilities

    Attributes:
        reference_channel (int): Index of the brightfield/reference channel
        pi_channel (int): Index of the fluorescence/PI channel
        focus_correction (bool): Whether to apply focus restoration
        method (str): Background correction method ('standard', 'basicpy', or 'basicpy_fl')
        image_segmentator: Model for cell segmentation
        object_classifier: Model for classifying segmented objects
        bf_focus_restorer: Model for restoring focus in brightfield images
        fl_focus_restorer: Model for restoring focus in fluorescence images
        pi_classifier: Model for classifying PI positive/negative cells
    """

    # Models required by this pipeline
    required_models = {"bf_focus", "fl_focus", "segmentation", "cell_classifier", "pi_classification"}

    def analyse_image(
        self,
        file_i: str,
        name: str,
        export_labeled_mask: bool = True,
        export_aligned_image: bool = True,
    ) -> None:
        """Pipeline analysis for each image."""

        # 1. Read Image:
        device = get_device()
        is_cuda = device.type == "cuda"
        movie_name = remove_file_extension(name)
        img_logger = self.setup_logger(self.output_path, movie_name)
        img_logger.info(f"Start single-frame analysis for {movie_name}")
        if getattr(self, "tracking", False):
            img_logger.warning(
                "tracking=True is set but ASCT_singleFrame does not support object "
                "tracking (single-frame analysis has no temporal axis). "
                "Tracking will be skipped."
            )
        reference_channel = self.reference_channel
        pi_channel = self.pi_channel
        method = self.method

        # Read and normalize image dimensions
        img_logger.info("1 - Reading and normalizing image dimensions", show_memory=True)
        img, pixel_size, img_shape = self._read_and_normalize_image(file_i, img_logger)
        size_x, size_y = img_shape['size_x'], img_shape['size_y']
        nChannels = img_shape['nChannels']
        frame_padded = img_shape.get('frame_padded', False)
        nFrames = img.shape[0]  # Will be 2 if frame was duplicated, 1 otherwise

        img_logger.info(
            f"Normalized image shape: {img.shape} "
            f"(T={nFrames}, C={nChannels}, H={size_x}, W={size_y}), "
            f"pixel size: {pixel_size} µm"
        )
        if frame_padded:
            img_logger.info(
                f"Note: Frame duplicated from T=1 to T={nFrames} for BaSiC compatibility. "
                f"Only frame 0 will be used for final analysis."
            )

        ip = ImagePreprocessor(img, stack_order="TCXY")
        img = np.zeros((1, 1, 1, 1))  # Remove img to save memory

        # 2 Pre-process image --------------------------------------------
        img_logger.info(f"Preprocessed image shape: {ip.img.shape}")
        # No frame alignment for single-frame images

        # 2.1 Detect and fix well borders (both BF and FL channels)
        img_logger.info("2.1 - Detecting and fixing border wells")
        ip.detect_fix_well(
            nchannels=[reference_channel, pi_channel], nslices=0, nframes=range(nFrames)
        )

        # 2.2 Background removal (all frames; T=2 required for BaSiC compatibility)
        img_logger.info(
            f"2.2 - Background removal | BF mean before: {self.check_px_values(ip, reference_channel, round=3)}"
        )
        self.clear_background(
            ip,
            channel=reference_channel,
            nFrames=range(nFrames),
            method=method,
            pixel_size=pixel_size,
        )
        self.clear_background(
            ip, channel=pi_channel, nFrames=range(nFrames), method=method
        )
        img_logger.info("2.2 - Background removal completed", show_memory=True)

        # 2.3 Focus restoration (conditional)
        if getattr(self, 'focus_correction', True):
            img_logger.info(
                "2.3 - Restoring focus in the reference channel", show_memory=True
            )
            img_logger.info(
                f"Reference channel intensity before focus restoration:\n{self.check_px_values(ip, reference_channel, round=3)}"
            )
            with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
                ip.img[:, 0, reference_channel] = self.bf_focus_restorer.predict(
                    ip.img[:, 0, reference_channel],
                    rescale=False,
                    batch_size=1,
                    buffer_steps=4,
                    buffer_dim=-1,
                    sw_batch_size=1,
                )
            img_logger.info("2.3 - Restoring focus in the PI channel", show_memory=True)
            img_logger.info(
                f"PI channel intensity before focus restoration:\n{self.check_px_values(ip, pi_channel, round=3)}"
            )
            with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
                ip.img[:, 0, pi_channel] = self.fl_focus_restorer.predict(
                    ip.img[:, 0, pi_channel],
                    batch_size=1,
                    buffer_steps=4,
                    buffer_dim=-1,
                    sw_batch_size=1,
                    padding_mode="reflect",
                )
            img_logger.info(
                f"Reference channel intensity after focus restoration:\n{self.check_px_values(ip, reference_channel, round=3)}"
            )
            img_logger.info(
                f"PI channel intensity after focus restoration:\n{self.check_px_values(ip, pi_channel, round=3)}"
            )
        else:
            img_logger.info("2.3 - Focus correction disabled, skipping focus restoration", show_memory=True)

        # 2.4 Species-specific preprocessing (conditional; after focus restoration)
        species = getattr(self, "species", None)
        if species:
            img_logger.info(
                f"2.4 - Applying species-specific preprocessing for: {species}",
                show_memory=True,
            )
            species_config_path = getattr(self, "species_config_path", None)
            species_cfg = self._load_species_config(species, species_config_path)
            if species_cfg:
                ip.apply_species_preprocessing(
                    nframes=range(nFrames),
                    nslices=0,
                    bf_channel=reference_channel,
                    fl_channel=pi_channel,
                    species_config=species_cfg,
                )
                img_logger.info(
                    "2.4 - Species-specific preprocessing completed", show_memory=True
                )
        else:
            img_logger.info(
                "2.4 - No species configured; skipping species-specific preprocessing"
            )

        # 2.5 Remove original image to save mem + build fl_norm for union mask
        ip.img_original = np.zeros((1, 1, 1, 1, 1))
        ip.build_fl_norm(fl_channel=pi_channel, nframes=range(nFrames))

        # 3.1 Segment Image --------------------------------------------
        img_logger.info("3.1 - Image segmentation", show_memory=True, cuda=is_cuda)
        prob_map = self._segment_single_frame(ip, img_logger, device)

        # 3.2 Get ROIs
        img_logger.info("3.2 - Extracting ROIs", show_memory=True)
        img_analyser = RoiAnalyser(ip.img, prob_map, stack_order=("TSCXY", "TCXY"))
        fl_norm = ip.fl_norm  # capture before del — used by union mask below

        # Remove image-processor to release space
        del ip
        img_analyser.create_binary_mask()
        img_analyser.clean_binmask(min_pixel_size=20)
        img_analyser.get_labels()
        img_logger.info(f"{img_analyser.total_rois} objects found in segmentation")

        # 3.3 FL union mask — recover ghost cells visible in FL but missed by BF segmentation
        n_ghosts, ghost_records = apply_fl_union_mask(img_analyser.labeled_mask, fl_norm)
        if n_ghosts > 0:
            img_analyser.total_rois = int(img_analyser.labeled_mask.max())
            img_logger.info(f"3.3 - FL union mask: {n_ghosts} ghost cell(s) added")
        else:
            img_logger.info("3.3 - FL union mask: no ghost cells detected")
        del fl_norm

        # 3.4 Classify ROIs
        img_logger.info("3.4 - Classifying ROIs", show_memory=True, cuda=is_cuda)
        object_classes, labels = self._classify_rois_single_frame(
            img_analyser, device, img_logger
        )

        # 4 Calc. measurements --------------------------------------------
        img_logger.info("4 - Starting measurements", show_memory=True)
        fl_measurements = self._extract_measurements(
            img_analyser, object_classes, reference_channel, pi_channel, img_logger
        )

        # Override ghost cells: added by FL union mask, definitionally piPOS
        if ghost_records:
            ghost_df = pd.DataFrame(ghost_records, columns=["frame", "label"])
            ghost_df["_ghost"] = True
            fl_measurements = fl_measurements.merge(ghost_df, on=["frame", "label"], how="left")
            fl_measurements.loc[fl_measurements["_ghost"].notna(), "object_class"] = "ghost"
            fl_measurements = fl_measurements.drop(columns=["_ghost"])

        # 4.4 Morphology-based label corrections (R1 only — single frame, no division detection)
        img_logger.info("4.4 - Applying morphology corrections", show_memory=False)
        fl_measurements, morph_counts = apply_semSeg_morphology_corrections(
            fl_measurements, img_analyser.labeled_mask,
            enable_division_detection=False,
        )
        img_logger.info(
            f"4.4 - Morphology corrections: {morph_counts['interior_to_clump']} interior→clump"
        )

        counts_per_frame = fl_measurements["frame"].value_counts().sort_index()
        img_logger.info(f"4 - Object counts per frame:\n{counts_per_frame.to_string()}")
        img_logger.info("4 - Measurements completed", show_memory=True)

        # Ghost cells are definitionally piPOS — determine their indices before classifier
        ghost_mask = fl_measurements["object_class"] == "ghost"

        # 4.5 PI classification (if enabled)
        if self.pi_classifier is not None:
            img_logger.info("4.5 - Running PI classification", show_memory=True)
            non_ghost = ~ghost_mask
            if non_ghost.any():
                predictions = self.pi_classifier.predict(
                    fl_measurements.loc[non_ghost, self.pi_classifier.feature_names_in_]
                )
                fl_measurements.loc[non_ghost, "pi_class"] = predictions
            fl_measurements.loc[ghost_mask, "pi_class"] = "piPOS"
            fl_measurements["file"] = name

            # Generate summary data using the dedicated method
            d_summary = self.generate_data_summary(
                fl_measurements,
                ["file", "frame", "object_class"],  # Simplified: no temporal metadata
                img_logger,
            )
        else:
            if ghost_mask.any():
                fl_measurements["pi_class"] = "piNEG"
                fl_measurements.loc[ghost_mask, "pi_class"] = "piPOS"
            d_summary = pd.DataFrame()

        # 5. Export data --------------------------------------------
        export_path = os.path.join(self.output_path, name)
        img_logger.info(f"5 - Writing output data to {export_path}")

        fl_measurements.to_csv(export_path + "_fl.csv")
        d_summary.to_csv(export_path + "_summary.csv")

        if export_labeled_mask:
            # Build class_to_id dynamically: known classes keep a stable pixel-value
            # ordering; any unexpected class gets a new ID appended at the end so
            # exports don't silently drop labels.
            _known_classes = ["single-cell", "clump", "noise", "off-focus", "joint-cell", "ghost"]
            class_to_id = {c: i for i, c in enumerate(_known_classes)}
            _seen = set(fl_measurements["object_class"].dropna().unique())
            _unexpected = sorted(_seen - set(_known_classes))
            if _unexpected:
                img_logger.warning(
                    f"Unexpected object_class values found during export: {_unexpected}. "
                    "These will be assigned new pixel IDs beyond the standard range."
                )
                next_id = max(class_to_id.values()) + 1
                for cls in _unexpected:
                    class_to_id[cls] = next_id
                    next_id += 1
            label_slice = img_analyser.get(
                "labels", index=(slice(None), 0, 0), to_numpy=True
            )

            # Map object classes to the labeled mask
            object_class_mask = map_predictions_to_labels_by_frame(
                label_slice,
                fl_measurements,
                "object_class",
                value_map={
                    class_name: class_id + 1
                    for class_name, class_id in class_to_id.items()
                },
            )

            # If PI classifier was used, create a second channel for PI classification
            if self.pi_classifier is not None:
                # Map PI classes to the labeled mask
                pi_class_mask = map_predictions_to_labels_by_frame(
                    label_slice,
                    fl_measurements,
                    "pi_class",
                    value_map={"piPOS": 1, "piNEG": 2},
                )

                # Stack the two channels: object class and PI class
                combined_mask = np.stack([object_class_mask, pi_class_mask], axis=1)
                labs_8bit = combined_mask.astype(np.uint8)
                axes = "TCYX"
                log_msg = (
                    "Exported labeled mask with object and PI classification channels"
                )
            else:
                # If no PI classifier, just save the object classification channel
                labs_8bit = object_class_mask.astype(np.uint8)
                axes = "TYX"
                log_msg = "Exported labeled mask with object classification channel"

            # Save the labeled mask with appropriate metadata
            tifffile.imwrite(
                export_path + "_labels.tiff",
                labs_8bit,
                imagej=True,
                metadata={"axes": axes},
            )
            img_logger.info(log_msg)
        if export_aligned_image:
            # Export full 32-bit image for detailed inspection (single-frame analysis)
            image_32bit = convert_image(
                img_analyser.get("image", to_numpy=True),
                np.float32,
            )
            tifffile.imwrite(export_path + "_transformed.tiff", image_32bit, imagej=True)
            img_logger.info("Exported 32-bit transformed image for detailed inspection")

        img_logger.info(f"Analysis completed for {movie_name}", show_memory=True)
        del prob_map, img, fl_measurements, d_summary, img_analyser
        gc.collect()
        empty_gpu_cache(device)
        img_logger.info("Garbage collection completed", show_memory=True)

        self.remove_logger(img_logger)

        return name

    def clear_background(
        self,
        ip: ImagePreprocessor,
        channel: int,
        nFrames: range,
        method: str,
        pixel_size: Optional[float] = None,
    ) -> None:
        """Remove background from images using specified method.

        Args:
        ip: Image preprocessor object
        channel: Channel to process
        nFrames: Range of frames to process
        method: Background removal method ('standard', 'basicpy', or 'basicpy_fl')
        pixel_size: Physical pixel size in microns
        """
        # If using the basicpy_fl in config, reference channel is still transform with DoG
        if method == "basicpy_fl" and channel == self.reference_channel:
            method = "standard"
        elif method == "basicpy_fl" and channel == self.pi_channel:
            method = "basicpy"

        methods = {
            "standard": [
                {
                    "nframes": nFrames,
                    "nchannels": channel,
                    "nslices": 0,
                    "sigma_r": 20,
                    "method": "divide",
                }
            ],
            "basicpy": [
                {
                    "nframes": nFrames,
                    "nchannels": channel,
                    "nslices": 0,
                    "method": "basicpy",
                    "smoothness_flatfield": 5,
                    "smoothness_darkfield": 5,
                    "get_darkfield": False,
                    "sort_intensity": False,
                    "fitting_mode": "approximate",
                }
            ],
        }

        if method not in methods:
            raise ValueError(f"Invalid method: {method}")

        for params in methods[method]:
            if method == "basicpy":
                ip.clear_image_background(**params)
            else:
                ip.clear_image_background(**params, unit="um", pixel_size=pixel_size)

    def generate_data_summary(
        self,
        fl_measurements: pd.DataFrame,
        by_list: List[str],
        img_logger: MemoryLogger,
    ) -> pd.DataFrame:
        """
        Generate a summary DataFrame from fluorescence measurements with PI classification.

        This method aggregates the fluorescence measurements by file, frame, channel,
        timestamp information, and object class to create a summary of PI-positive and
        PI-negative cell counts and areas.

        Args:
            fl_measurements: DataFrame containing fluorescence measurements with 'pi_class' column.
                Must include columns: 'file', 'frame', 'channel', 'date_time', 'timestep',
                'abslag_in_s', 'object_class', 'label', 'area', and 'pi_class'.
            img_logger: Logger instance for recording progress and errors.

        Returns:
            pd.DataFrame: A summary DataFrame with aggregated counts and areas, or an empty
                DataFrame if an error occurs during the groupby operation.

        Notes:
            The summary includes the following aggregated metrics:
            - total_count: Total number of objects per group
            - pi_class_neg: Count of PI-negative objects
            - pi_class_pos: Count of PI-positive objects
            - area_pineg: Total area of PI-negative objects
            - area_pipos: Total area of PI-positive objects
            - area_total: Total area of all objects
        """
        try:
            img_logger.info(f"Group data by {by_list}")
            d_summary = (
                fl_measurements.groupby(by_list)
                .agg(
                    total_count=("label", "count"),
                    pi_class_neg=("pi_class", lambda x: (x == "piNEG").sum()),
                    pi_class_pos=("pi_class", lambda x: (x == "piPOS").sum()),
                    area_pineg=(
                        "area",
                        lambda x: x[
                            fl_measurements.loc[x.index, "pi_class"] == "piNEG"
                        ].sum(),
                    ),
                    area_pipos=(
                        "area",
                        lambda x: x[
                            fl_measurements.loc[x.index, "pi_class"] == "piPOS"
                        ].sum(),
                    ),
                    area_total=("area", "sum"),
                )
                .reset_index()
            )

            img_logger.info(
                f"Groupby operation completed successfully. Shape of d_summary: {d_summary.shape}"
            )
        except Exception as e:
            img_logger.error(f"Error during groupby operation: {str(e)}")
            img_logger.error(f"Columns in fl_measurements: {fl_measurements.columns}")
            img_logger.error(
                f"Unique values in 'pi_class': {fl_measurements['pi_class'].unique()}"
            )
            d_summary = pd.DataFrame()

        img_logger.info("d_summary created successfully", show_memory=True)

        return d_summary

    @staticmethod
    def check_px_values(ip, channel: int, round: int = None) -> np.ndarray:
        """Calculate mean pixel intensity across frames for a given channel."""
        means = np.mean(ip.img[:, 0, channel], axis=(1, 2))
        return np.round(means, round) if round is not None else means

    def _read_and_normalize_image(
        self,
        file_path: str,
        logger: MemoryLogger
    ) -> Tuple[np.ndarray, float, Dict[str, int]]:
        """
        Read image and ensure it has temporal dimension for pipeline compatibility.

        Handles various input shapes:
        - (H, W) -> (1, 2, H, W)  [single channel duplicated to 2 for focus/PI]
        - (C, H, W) -> (1, C, H, W)  [multi-channel, single frame]
        - (T, C, H, W) -> (1, C, H, W) if T != 1  [warn and take first frame]

        Returns:
            tuple: (normalized_image, pixel_size, shape_dict)
                - normalized_image: Always shape (1, C, H, W)
                - pixel_size: Physical pixel size in µm
                - shape_dict: Dict with 'nChannels', 'size_x', 'size_y'
        """
        image_reader = ImageReader(file_path, self.file_type)
        img, metadata = image_reader.read_image()

        # Handle TIFF files without OME metadata
        if metadata is None:
            logger.warning(
                "No OME metadata found (likely TIFF file). "
                "Extracting dimensions from image array. Using default pixel_size=1.0 µm."
            )
            # Extract dimensions directly from array
            if img.ndim == 2:
                size_x, size_y = img.shape
                nChannels = 1
                nFrames = 1
            elif img.ndim == 3:
                nChannels, size_x, size_y = img.shape
                nFrames = 1
            elif img.ndim == 4:
                nFrames, nChannels, size_x, size_y = img.shape
            else:
                raise ValueError(f"Unexpected image dimensions: {img.shape}")

            # Try to read pixel size from TIFF tags
            pixel_size = self._try_read_tiff_pixel_size(file_path, logger)
        else:
            # Extract from OME metadata (ND2 files)
            pixel_size = metadata.images[0].pixels.physical_size_x
            size_x = metadata.images[0].pixels.size_x
            size_y = metadata.images[0].pixels.size_y
            nChannels = metadata.images[0].pixels.size_c
            nFrames = metadata.images[0].pixels.size_t

        # Normalize dimensions
        if img.ndim == 2:
            # (H, W) -> (1, 2, H, W) - duplicate channel for focus restoration and PI
            logger.warning(
                "Input image has single channel (H, W). Duplicating to 2 channels "
                "for focus restoration and PI classification. This can cause inaccurate values."
            )
            img = img[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)
            img = np.repeat(img, 2, axis=1)  # (1, 2, H, W)
            nChannels = 2
        elif img.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            logger.info(f"Input shape: (C={img.shape[0]}, H, W) - expanding to (1, C, H, W)")
            img = img[np.newaxis, ...]
            if nChannels == 0 or nChannels != img.shape[1]:
                nChannels = img.shape[1]
                logger.warning(f"Metadata mismatch: using detected nChannels={nChannels}")
        elif img.ndim == 4:
            # Already (T, C, H, W), validate T=1
            if img.shape[0] != 1:
                logger.warning(
                    f"Expected single frame but got T={img.shape[0]}. "
                    f"Processing only first frame."
                )
                img = img[0:1, ...]
        else:
            raise ValueError(f"Unexpected image dimensions: {img.shape}")

        # Final validation and reshape
        img = img.reshape(1, nChannels, size_x, size_y)

        # CRITICAL: BaSiC background correction requires T>=2 frames to estimate background variation
        # With T=1, BaSiC fails with: "Images must be 3 or 4-dimensional array, with dimension of (T,Y,X)"
        # or produces scalar outputs that cause downstream indexing errors.
        #
        # Solution: Duplicate single frame to T=2. Since both frames are identical, BaSiC treats
        # the background as uniform, which is the correct behavior for single-frame analysis.
        # This is required for basicpy_fl method which is the primary use case.
        #
        # Future consideration: Could skip BaSiC for single frames and fall back to DoG method.

        # Track if we padded dimensions for warning messages
        frame_padded = False

        if img.shape[0] == 1:  # Single frame detected
            logger.warning(
                "Single-frame image detected (T=1). Duplicating to T=2 for BaSiC compatibility. "
                "BaSiC background correction requires multiple frames to estimate variation. "
                "Both frames will be identical, resulting in uniform background estimation."
            )
            img = np.repeat(img, 2, axis=0)  # (1, C, H, W) → (2, C, H, W)
            frame_padded = True

        shape_dict = {
            'nChannels': nChannels,
            'size_x': size_x,
            'size_y': size_y,
            'frame_padded': frame_padded,  # Track for downstream logging
        }

        return img, pixel_size, shape_dict

    @staticmethod
    def _try_read_tiff_pixel_size(file_path: str, logger: MemoryLogger) -> float:
        """
        Attempt to extract pixel size from TIFF file metadata.

        Looks for pixel size in ImageJ or OME-TIFF tags. Falls back to 1.0 µm if not found.

        Args:
            file_path: Path to TIFF file
            logger: Logger instance

        Returns:
            Pixel size in micrometers (default: 1.0)
        """
        try:
            with tifffile.TiffFile(file_path) as tif:
                # Try ImageJ metadata
                if tif.is_imagej:
                    imagej_metadata = tif.imagej_metadata
                    if imagej_metadata and 'unit' in imagej_metadata:
                        # ImageJ stores pixel size in 'XResolution' or in metadata
                        if 'XResolution' in imagej_metadata:
                            pixel_size = imagej_metadata['XResolution']
                            logger.info(f"Extracted pixel_size from ImageJ metadata: {pixel_size} µm")
                            return pixel_size

                # Try OME-TIFF metadata
                if tif.is_ome:
                    # This would require parsing XML, skip for now
                    pass

                # Try reading from tags directly
                for page in tif.pages:
                    tags = page.tags
                    # Check for resolution tags (282 = XResolution, 283 = YResolution)
                    if 'XResolution' in tags:
                        x_res = tags['XResolution'].value
                        if isinstance(x_res, tuple) and len(x_res) == 2:
                            # Resolution is stored as (numerator, denominator)
                            pixel_size = x_res[1] / x_res[0]  # Invert to get µm/pixel
                            logger.info(f"Extracted pixel_size from TIFF tags: {pixel_size:.4f} µm")
                            return pixel_size
                    break  # Only check first page

        except Exception as e:
            logger.warning(f"Could not extract pixel_size from TIFF metadata: {e}")

        # Default fallback
        logger.warning(
            "Could not read pixel_size from TIFF metadata. "
            "Falling back to 1.0 µm — calibration-dependent results will be incorrect. "
            "Set pixel_size explicitly in the pipeline config to suppress this warning."
        )
        return 1.0

    def _segment_single_frame(
        self,
        ip: ImagePreprocessor,
        logger: MemoryLogger,
        device: torch.device
    ) -> np.ndarray:
        """
        Segment single frame with optimized resource allocation.

        Extracts frame 0 directly and processes it, avoiding unnecessary
        frame iteration overhead.

        Args:
            ip: ImagePreprocessor with normalized image
            logger: Memory logger instance
            device: Torch device for computation

        Returns:
            prob_map: Probability map with shape (1, 1, H, W)
        """
        with ReserveResource(device, 4.0, logger=logger, timeout=120):
            # Extract single frame directly
            single_frame = ip.img[0, 0, self.reference_channel, :, :]
            prob_map = self.image_segmentator.predict(
                single_frame[np.newaxis, ...],  # Add batch dimension
                buffer_steps=4,
                buffer_dim=-1,
                sw_batch_size=1,
            )

        logger.info("Segmentation completed", show_memory=True, cuda=(device.type == "cuda"))

        # Ensure proper dimensions (1, 1, H, W)
        if prob_map.ndim > 3 and prob_map.shape[1] > 1:
            prob_map = np.max(prob_map, axis=1, keepdims=True)
        elif prob_map.ndim == 3:
            prob_map = np.expand_dims(prob_map, axis=1)
        elif prob_map.ndim == 2:
            prob_map = np.expand_dims(prob_map, axis=(0, 1))

        return prob_map

    def _classify_rois_single_frame(
        self,
        img_analyser: RoiAnalyser,
        device: torch.device,
        logger: MemoryLogger
    ) -> Tuple[List[str], np.ndarray]:
        """
        Classify ROIs directly without frame iteration overhead.

        Key optimization: Extracts single frame as 2D (H, W) arrays which
        the classifier expects, avoiding the frame iteration in batch_classify_rois.

        Performance: ~10-15% faster than batch_classify_rois for single frames.

        Args:
            img_analyser: RoiAnalyser instance with labels and image
            device: Torch device for computation
            logger: Memory logger instance

        Returns:
            tuple: (object_classes, labels)
                - object_classes: List of classification strings
                - labels: Array of ROI label indices
        """
        # Extract single frame directly as 2D - critical for efficiency
        # Index (0, 0, 0) = (frame_0, slice_0, channel_0)
        labeled_mask = img_analyser.get("labels", index=(0, 0, 0), to_numpy=True)
        img = img_analyser.get("image", index=(0, 0, 0), to_numpy=True)

        # Verify we got 2D arrays
        assert labeled_mask.ndim == 2, f"Expected 2D labeled_mask, got {labeled_mask.ndim}D"
        assert img.ndim == 2, f"Expected 2D image, got {img.ndim}D"

        n_rois = np.unique(labeled_mask).size - 1  # Subtract background
        logger.info(f"Classifying {n_rois} ROIs from single frame")

        with ReserveResource(device, 10.0, logger=logger, timeout=240):
            object_classes, labels = self.object_classifier.classify_rois(
                labeled_mask,  # 2D (H, W) as classifier expects
                img            # 2D (H, W) as classifier expects
            )

        logger.info(
            "Classification completed",
            show_memory=True,
            cuda=(device.type == "cuda")
        )

        return object_classes, labels

    def _extract_measurements(
        self,
        img_analyser: RoiAnalyser,
        object_classes: List[str],
        reference_channel: int,
        pi_channel: int,
        logger: MemoryLogger
    ) -> pd.DataFrame:
        """
        Extract fluorescence measurements for single frame.

        Simplified from multi-frame version: no temporal tracking needed.

        Args:
            img_analyser: RoiAnalyser instance
            object_classes: Classification results
            pi_channel: Fluorescence channel index
            logger: Memory logger instance

        Returns:
            DataFrame with measurements and classifications
        """
        logger.info("Extracting background fluorescence intensity")
        bck_fl = measure_background_intensity(
            img_analyser.get("image", to_numpy=False),
            img_analyser.get("labels", to_numpy=False),
            target_channel=pi_channel,
        )

        fl_prop = [
            "label", "centroid", "max_intensity", "min_intensity",
            "mean_intensity", "area", "major_axis_length",
            "minor_axis_length", "solidity", "orientation", "eccentricity",
            "perimeter", "equivalent_diameter_area",
            "feret_diameter_max", "moments_hu",
        ]

        logger.info("Extracting fluorescence measurements")
        fl_measurements = img_analyser.get_roi_measurements(
            target_channel=pi_channel,
            properties=fl_prop,
            extra_properties=(
                roi_skewness, roi_std_dev,
                roi_glcm_features, roi_radial_profile,
                roi_skeleton_features, roi_shape_features, roi_tubularness,
                roi_skeleton_branch_points,
            ),
        )
        fl_measurements = fl_measurements.rename(columns={
            "roi_glcm_features-0": "glcm_contrast",
            "roi_glcm_features-1": "glcm_homogeneity",
            "roi_glcm_features-2": "glcm_energy",
            "roi_glcm_features-3": "glcm_correlation",
            "roi_radial_profile-0": "radial_0",
            "roi_radial_profile-1": "radial_1",
            "roi_radial_profile-2": "radial_2",
            "roi_radial_profile-3": "radial_3",
            "roi_radial_profile-4": "radial_4",
            "roi_skeleton_features-0": "skeleton_length",
            "roi_skeleton_features-1": "mean_cell_width",
            "roi_skeleton_features-2": "skeleton_curvature",
            "roi_skeleton_features-3": "intensity_continuity",
            "roi_shape_features-0": "border_complexity",
            "roi_shape_features-1": "pole_regularity",
            "moments_hu-0": "hu_0",
            "moments_hu-1": "hu_1",
            "moments_hu-2": "hu_2",
            "moments_hu-3": "hu_3",
            "moments_hu-4": "hu_4",
            "moments_hu-5": "hu_5",
            "moments_hu-6": "hu_6",
            "roi_skeleton_branch_points": "skeleton_branch_points",
        })
        bf_meas = img_analyser.get_roi_measurements(
            target_channel=reference_channel,
            properties=["label", "centroid"],
            n_workers=1,
        )
        bf_meas = bf_meas[["frame", "label", "centroid_0", "centroid_1"]].rename(
            columns={"centroid_0": "bf_centroid_0", "centroid_1": "bf_centroid_1"}
        )
        fl_measurements = fl_measurements.merge(bf_meas, on=["frame", "label"], how="left")
        fl_measurements["centroid_offset"] = np.sqrt(
            (fl_measurements["centroid_0"] - fl_measurements["bf_centroid_0"]) ** 2
            + (fl_measurements["centroid_1"] - fl_measurements["bf_centroid_1"]) ** 2
        )
        fl_measurements["object_class"] = object_classes
        fl_measurements["frame"] = 0  # Single frame always 0

        # Merge background and calculate relative intensities
        fl_measurements = pd.merge(fl_measurements, bck_fl, on="frame", how="left")
        fl_measurements[["rel_max_intensity", "rel_min_intensity", "rel_mean_intensity"]] = \
            fl_measurements[["max_intensity", "min_intensity", "mean_intensity"]].div(
                fl_measurements["background"], axis=0
            )

        logger.info(f"Extracted measurements for {len(fl_measurements)} objects")

        return fl_measurements
