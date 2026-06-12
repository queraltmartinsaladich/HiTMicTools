import os
import gc
import tifffile
from typing import Optional, List
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
    apply_instSeg_morphology_corrections,
)
from HiTMicTools.tracking.track_events import (
    detect_division_events,
    detect_lysis_events,
    detect_filamentation_events,
    compute_fl_trajectory_features,
)
from HiTMicTools.utils import get_timestamps, remove_file_extension
from HiTMicTools.roianalysis import RoiAnalyser
from HiTMicTools.data_analysis.analysis_tools import (
    roi_skewness, roi_std_dev, roi_glcm_features, roi_radial_profile,
)

from jetraw_tools.image_reader import ImageReader


class ASCT_instSeg(BasePipeline):
    """
    Pipeline for automated single-cell tracking with single-step instance segmentation.

    This pipeline processes microscopy images to:
    1. Restore focus in both brightfield and fluorescence channels (optional)
    2. Perform single-step instance segmentation and classification using RF-DETR-Segm
    3. Track cells across time frames
    4. Analyze fluorescence intensity and other cellular properties

    This pipeline simplifies the traditional 3-step approach (segmentation → connected components → classification)
    by using an end-to-end instance segmentation model that outputs labeled masks and class predictions directly.

    Attributes:
        reference_channel (int): Index of the brightfield/reference channel
        pi_channel (int): Index of the fluorescence/PI channel
        align_frames (bool): Whether to align frames in time series
        focus_correction (bool): Whether to apply focus restoration
        method (str): Background correction method ('standard', 'basicpy', or 'basicpy_fl')
        sc_segmenter: Single-step instance segmentation model (RF-DETR-Segm)
        bf_focus_restorer: Model for restoring focus in brightfield images (optional)
        fl_focus_restorer: Model for restoring focus in fluorescence images (optional)
        pi_classifier: Model for classifying PI positive/negative cells (optional)
        cell_tracker: Cell tracking model (optional)
        class_dict: Mapping from class indices to class names
    """

    # Models required by this pipeline
    required_models = {"bf_focus", "fl_focus", "sc_segmenter", "pi_classification"}

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
        name = movie_name
        img_logger = self.setup_logger(self.output_path, movie_name)
        img_logger.info(f"Start analysis for {movie_name}")
        reference_channel = self.reference_channel
        pi_channel = self.pi_channel
        align_frames = self.align_frames
        method = self.method

        img_logger.info("1 - Reading image", show_memory=True)
        image_reader = ImageReader(file_i, self.file_type)
        img, metadata = image_reader.read_image()
        pixel_size = metadata.images[0].pixels.physical_size_x
        size_x = metadata.images[0].pixels.size_x
        size_y = metadata.images[0].pixels.size_y
        nSlices = metadata.images[0].pixels.size_z
        nChannels = metadata.images[0].pixels.size_c
        nFrames = metadata.images[0].pixels.size_t
        # 2 Pre-process image --------------------------------------------
        img_logger.info(
            f"Image shape: {img.shape}, pixel size: {pixel_size} µm. Reshaped to (frames={nFrames}, channels={nChannels}, slices={nSlices}, x={size_x}, y={size_y})"
        )
        img = img.reshape(nFrames, nChannels, size_x, size_y)
        # Subsetting for developing, testing and debugging
        # nFrames = min(nFrames, 3)
        # img = img[:nFrames]
        ip = ImagePreprocessor(img, stack_order="TCXY")
        img = np.zeros((1, 1, 1, 1))  # Remove img to save memory
        img_logger.info(f"Preprocessed image shape: {ip.img.shape}")

        # 2.1 Align frames if required
        if align_frames:
            img_logger.info("2.1 - Aligning frames in the stack", show_memory=True)
            ip.align_image(
                ref_channel=0, ref_slice=-1, crop_image=True, reference_type="dynamic"
            )
            img_logger.info("2.1 - Frame alignment completed", show_memory=True)
        # Update size x and size y after alignment and maybe crop
        size_x, size_y = ip.img.shape[-2], ip.img.shape[-1]

        # 2.2 Detect and fix well borders (both BF and FL channels)
        img_logger.info("2.2 - Detecting and fixing border wells")
        ip.detect_fix_well(
            nchannels=[reference_channel, pi_channel], nslices=0, nframes=range(nFrames)
        )

        # 2.3 Background removal
        img_logger.info(
            f"2.3 - Background removal | BF mean before: {self.check_px_values(ip, reference_channel, round=3)}"
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
        img_logger.info("2.3 - Background removal completed", show_memory=True)

        # 2.4 Focus restoration (conditional)
        if getattr(self, 'focus_correction', True):  # Default to True for backward compatibility
            img_logger.info(
                "2.4 - Restoring focus in the reference channel", show_memory=True
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
            img_logger.info("2.4 - Restoring focus in the PI channel", show_memory=True)
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
            img_logger.info("2.4 - Focus correction disabled, skipping focus restoration", show_memory=True)

        # 2.5 Species-specific preprocessing (conditional; after focus restoration so
        # operators work on sharp images and fl_norm reflects the restored FL channel)
        species = getattr(self, "species", None)
        if species:
            img_logger.info(
                f"2.5 - Applying species-specific preprocessing for: {species}",
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
                    "2.5 - Species-specific preprocessing completed", show_memory=True
                )
        else:
            img_logger.info(
                "2.5 - No species configured; skipping species-specific preprocessing"
            )

        # 2.6 Remove original image (not used after background corr) to save mem
        ip.img_original = np.zeros((1, 1, 1, 1, 1))
        # Ensure fl_norm is built for the union-mask step (idempotent if species
        # preprocessing already ran fl_normalization)
        ip.build_fl_norm(fl_channel=pi_channel, nframes=range(nFrames))

        # 3. Single-step instance segmentation + classification
        img_logger.info("3 - Running single-step instance segmentation and classification", show_memory=True, cuda=is_cuda)

        with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
            frames = ip.img[:, 0, reference_channel, :, :]
            frames = np.clip(frames, 0, 1)  # Ensure [0, 1] range

            # Run 4D instance segmentation with model-config defaults unless pipeline overrides are set
            seg_temporal_buffer_size = getattr(self, "sc_segmenter_temporal_buffer_size", None)
            seg_batch_size = getattr(self, "sc_segmenter_batch_size", None)
            stacked_labeled_masks, all_bboxes, all_class_ids, all_scores = self.sc_segmenter.predict(
                frames,
                channel_index=0,
                temporal_buffer_size=seg_temporal_buffer_size,
                batch_size=seg_batch_size,
                normalize_to_255=False,
                output_shape="HW",
            )

        img_logger.info("3 - Instance segmentation completed", show_memory=True, cuda=is_cuda)

        total_detections = sum(len(bboxes) for bboxes in all_bboxes)
        total_instances = np.sum([len(np.unique(stacked_labeled_masks[i])) - 1 for i in range(nFrames)])

        self._log_detection_summary(
            img_logger=img_logger,
            n_frames=nFrames,
            total_detections=total_detections,
            total_instances=total_instances,
            class_ids=all_class_ids,
            class_scores=all_scores,
        )

        # Create RoiAnalyser from labeled masks
        img_logger.info("3.2 - Creating ROI analyser from labeled masks", show_memory=True)

        # Both ip.img and masks follow TSCXY ordering where last two dims are (X=Width, Y=Height)
        img_analyser = RoiAnalyser.from_labeled_mask(
            ip.img,
            stacked_labeled_masks,
            stack_order=("TSCXY", "TXY")
        )
        fl_norm = ip.fl_norm  # capture before del — used by union mask below

        # Remove image-processor to release space
        del ip

        # 3.3 FL union mask — recover ghost cells visible in FL but missed by BF segmentation
        n_ghosts, ghost_records = apply_fl_union_mask(img_analyser.labeled_mask, fl_norm)
        if n_ghosts > 0:
            img_analyser.total_rois = int(img_analyser.labeled_mask.max())
            img_logger.info(f"3.3 - FL union mask: {n_ghosts} ghost cell(s) added")
        else:
            img_logger.info("3.3 - FL union mask: no ghost cells detected")
        del fl_norm

        # Build a reference table that maps (frame, label) → class name + score.
        # ScSegmenter guarantees consecutive labels 1..N per frame after NMS/relabeling,
        # so label_index+1 corresponds to the mask label.  We merge on (frame, label)
        # rather than assuming positional alignment with the regionprops DataFrame —
        # this is robust even if regionprops drops a label with zero pixels.
        detection_records = []
        for frame_idx, (frame_classes, frame_scores) in enumerate(
            zip(all_class_ids, all_scores)
        ):
            for label_idx, (cid, score) in enumerate(zip(frame_classes, frame_scores)):
                if self.class_dict:
                    class_name = self.class_dict[int(cid)]
                else:
                    class_name = f"class_{int(cid)}"
                detection_records.append(
                    (frame_idx, label_idx + 1, class_name, float(score))
                )
        detection_df = pd.DataFrame(
            detection_records,
            columns=["frame", "label", "object_class", "detection_score"],
        )

        img_logger.info(
            f"3.2 - {img_analyser.total_rois} max label per frame | "
            f"{len(detection_records)} detection records from segmenter"
        )

        # 4.1 Calc. measurements --------------------------------------------
        img_logger.info("4 - Starting measurements", show_memory=True)
        img_logger.info("4.1 - Extracting background fluorescence intensity")
        bck_fl = measure_background_intensity(
            img_analyser.get("image", to_numpy=False),
            img_analyser.get("labels", to_numpy=False),
            target_channel=pi_channel,
        )

        fl_prop = [
            "label",
            "centroid",
            "max_intensity",
            "min_intensity",
            "mean_intensity",
            "area",
            "major_axis_length",
            "minor_axis_length",
            "solidity",
            "orientation",
            "eccentricity",
        ]
        img_logger.info("4.2 - Extracting fluorescence measurements")
        fl_measurements = img_analyser.get_roi_measurements(
            target_channel=pi_channel,
            properties=fl_prop,
            extra_properties=(roi_skewness, roi_std_dev, roi_glcm_features, roi_radial_profile),
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

        # Merge class names and detection scores into measurements via (frame, label).
        n_before = len(fl_measurements)
        fl_measurements = fl_measurements.merge(
            detection_df, on=["frame", "label"], how="left"
        )
        fl_measurements["object_class"] = fl_measurements["object_class"].fillna("unknown")
        fl_measurements["detection_score"] = fl_measurements["detection_score"].fillna(0.0)

        # Override ghost cells: FL-union-mask additions have no detection record and
        # are definitionally piPOS dead cells — assign class before the unknown warning
        if ghost_records:
            ghost_df = pd.DataFrame(ghost_records, columns=["frame", "label"])
            ghost_df["_ghost"] = True
            fl_measurements = fl_measurements.merge(ghost_df, on=["frame", "label"], how="left")
            fl_measurements.loc[fl_measurements["_ghost"].notna(), "object_class"] = "ghost"
            fl_measurements = fl_measurements.drop(columns=["_ghost"])

        n_unknown = (fl_measurements["object_class"] == "unknown").sum()
        if n_unknown > 0:
            img_logger.warning(
                f"class merge: {n_unknown}/{n_before} ROIs could not be matched "
                f"to a detection — marked as 'unknown'"
            )
        else:
            img_logger.info(
                f"class merge: all {n_before} ROIs matched successfully"
            )

        img_logger.info("4.3 - Extracting time metadata")
        time_data = get_timestamps(metadata, timeformat="%Y-%m-%d %H:%M:%S")
        fl_measurements = pd.merge(fl_measurements, time_data, on="frame", how="left")
        fl_measurements = pd.merge(fl_measurements, bck_fl, on="frame", how="left")
        fl_measurements[
            ["rel_max_intensity", "rel_min_intensity", "rel_mean_intensity"]
        ] = fl_measurements[["max_intensity", "min_intensity", "mean_intensity"]].div(
            fl_measurements["background"], axis=0
        )

        # 4.4 Morphology-based label corrections
        img_logger.info("4.4 - Applying morphology corrections", show_memory=False)
        fl_measurements, morph_counts = apply_instSeg_morphology_corrections(
            fl_measurements, img_analyser.labeled_mask
        )
        img_logger.info(
            f"4.4 - Morphology corrections: "
            f"{morph_counts['filamented_to_long']} filamented→long, "
            f"{morph_counts['lysed_to_lyse']} lysed→lyse, "
            f"{morph_counts['merged_to_clump']} single-cell→clump"
        )

        # 4.5 Object tracking (if enabled)
        if self.tracking and self.cell_tracker is not None:
            img_logger.info("4.5 - Running object tracking")
            track_features = fl_prop[5:10]
            self.cell_tracker.set_features(track_features)
            try:
                fl_measurements = self.cell_tracker.track_objects(
                    fl_measurements, volume_bounds=(size_x, size_y), logger=img_logger
                )
                img_logger.info("4.5 - Object tracking completed successfully")
            except Exception as e:
                img_logger.error(f"Object tracking failed: {e}")
                # Continue without tracking

        counts_per_frame = fl_measurements["frame"].value_counts().sort_index()
        img_logger.info(f"4 - Object counts per frame:\n{counts_per_frame.to_string()}")
        img_logger.info("4 - Measurements completed", show_memory=True)

        # Ghost cells are definitionally piPOS — determine their indices before classifier
        ghost_mask = fl_measurements["object_class"] == "ghost"

        # 4.6 PI classification (if enabled)
        if self.pi_classifier is not None:
            img_logger.info("4.6 - Running PI classification", show_memory=True)
            non_ghost = ~ghost_mask
            if non_ghost.any():
                predictions = self.pi_classifier.predict(
                    fl_measurements.loc[non_ghost, self.pi_classifier.feature_names_in_]
                )
                fl_measurements.loc[non_ghost, "pi_class"] = predictions
            fl_measurements.loc[ghost_mask, "pi_class"] = "piPOS"

            # 4.7 piPOS lock-in (if tracker supports it)
            if (
                self.tracking
                and self.cell_tracker is not None
                and hasattr(self.cell_tracker, "apply_pipos_lockin")
            ):
                img_logger.info("4.7 - Applying piPOS lock-in")
                fl_measurements = self.cell_tracker.apply_pipos_lockin(
                    fl_measurements, logger=img_logger
                )

        # 4.8 Track event detection + FL trajectory (requires final pi_class)
        if self.tracking and self.cell_tracker is not None:
            img_logger.info("4.8 - Detecting track events", show_memory=False)
            fl_measurements, div_counts = detect_division_events(fl_measurements)
            fl_measurements, lys_counts = detect_lysis_events(fl_measurements)
            fl_measurements, fil_counts = detect_filamentation_events(fl_measurements)
            fl_measurements = compute_fl_trajectory_features(fl_measurements)
            img_logger.info(
                f"4.8 - Track events: "
                f"{div_counts['n_division_events']} divisions, "
                f"{lys_counts['n_lysis_events']} lysis events, "
                f"{fil_counts['n_filamentation_events']} filamentation events"
            )

        fl_measurements["file"] = name

            # Generate summary data using the dedicated method
            d_summary = self.generate_data_summary(
                fl_measurements,
                [
                    "file",
                    "frame",
                    "channel",
                    "date_time",
                    "timestep",
                    "abslag_in_s",
                    "object_class",
                ],
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
            # Export instance segmentation labels with class information
            # Create two channels: instance labels and class labels

            # Map object classes to the labeled mask
            object_classes_col = fl_measurements["object_class"].tolist()
            if self.class_dict:
                class_value_map = {
                    name: class_idx + 1
                    for class_idx, name in sorted(self.class_dict.items())
                }
            else:
                unique_class_names = sorted(set(object_classes_col))
                class_value_map = {
                    class_name: idx + 1
                    for idx, class_name in enumerate(unique_class_names)
                }

            labeled_mask_slice = img_analyser.get(
                "labels", index=(slice(None), 0, 0), to_numpy=True
            )
            object_class_mask = map_predictions_to_labels_by_frame(
                labeled_mask_slice,
                fl_measurements,
                "object_class",
                value_map=class_value_map,
            )

            # If PI classifier was used, create a second channel for PI classification
            if self.pi_classifier is not None:
                # Map PI classes to the labeled mask
                pi_class_mask = map_predictions_to_labels_by_frame(
                    labeled_mask_slice,
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
            image_8bit = convert_image(
                img_analyser.get("image", to_numpy=True),
                np.uint8,
            )
            tifffile.imwrite(export_path + "_transformed.tiff", image_8bit, imagej=True)

        img_logger.info(f"Analysis completed for {movie_name}", show_memory=True)
        del stacked_labeled_masks, img, fl_measurements, d_summary, img_analyser
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
