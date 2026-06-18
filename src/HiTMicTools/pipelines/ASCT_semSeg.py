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
    refine_masks_temporal,
)
from HiTMicTools.img_processing.morphology_corrections import (
    apply_semSeg_morphology_corrections,
)
from HiTMicTools.tracking.track_events import (
    refine_tracks,
    detect_division_events,
    reconcile_lineage,
    detect_lysis_events,
    detect_filamentation_events,
    compute_fl_trajectory_features,
)
from HiTMicTools.utils import get_timestamps, remove_file_extension
from HiTMicTools.roianalysis import RoiAnalyser
from HiTMicTools.data_analysis.analysis_tools import (
    roi_skewness, roi_std_dev, roi_glcm_features, roi_radial_profile,
    roi_skeleton_features, roi_shape_features, frame_tubularness,
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


class ASCT_semSeg(BasePipeline):
    """
    Pipeline for automated single-cell tracking with focus restoration.

    This pipeline processes microscopy images to:
    1. Restore focus in both brightfield and fluorescence channels
    2. Segment and classify cells in the images
    3. Track cells across time frames
    4. Analyze fluorescence intensity and other cellular properties

    The pipeline is designed for time-lapse microscopy data with multiple channels,
    particularly for experiments tracking PI (propidium iodide) uptake in cells.

    Attributes:
        reference_channel (int): Index of the brightfield/reference channel
        pi_channel (int): Index of the fluorescence/PI channel
        align_frames (bool): Whether to align frames in time series
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
        export_training_crops: bool = False,
        training_crop_size: int = 64,
    ) -> None:
        """Pipeline analysis for each image."""

        # 1. Read Image:
        device = get_device()
        is_cuda = device.type == "cuda"
        movie_name = remove_file_extension(name)
        name = movie_name
        # Desync analysis to avoid RAM/VRAM issues

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

        # 3.1 Segment Image --------------------------------------------
        img_logger.info("3.1 - Image segmentation", show_memory=True, cuda=is_cuda)
        with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
            prob_map = self.image_segmentator.predict(
                ip.img[:, 0, reference_channel, :, :],
                buffer_steps=4,
                buffer_dim=-1,
                sw_batch_size=1,
            )
        img_logger.info("3.1 - Segmentation completed", show_memory=True, cuda=is_cuda)

        # Get ROIs
        if prob_map.ndim > 3 and prob_map.shape[1] > 1:
            prob_map = np.max(prob_map, axis=1, keepdims=True)
        elif prob_map.ndim == 3:
            prob_map = np.expand_dims(prob_map, axis=1)
        elif prob_map.ndim == 2:
            prob_map = np.expand_dims(prob_map, axis=(0, 1))
        else:
            pass

        # 3.2 Get ROIs
        img_logger.info("3.2 - Extracting ROIs", show_memory=True)
        img_analyser = RoiAnalyser(ip.img, prob_map, stack_order=("TSCXY", "TCXY"))
        fl_norm = ip.fl_norm  # capture before del — used by union mask below
        frame_shifts = getattr(ip, "frame_shifts", None)  # capture before del

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

        # 3.3b Temporal mask refinement — split merged instances using previous-frame centroids
        if getattr(self, "tracking", False):
            gradient = 1.0 - prob_map[:, 0].astype(np.float32)  # high at boundaries
            n_splits = refine_masks_temporal(img_analyser.labeled_mask, gradient_map=gradient)
            if n_splits:
                img_analyser.total_rois = int(img_analyser.labeled_mask.max())
                img_logger.info(f"3.3b - Temporal refinement: {n_splits} region(s) split")

        # 3.4 Classify ROIs
        img_logger.info("3.4 - Classifying ROIs", show_memory=True, cuda=is_cuda)
        with ReserveResource(device, 10.0, logger=img_logger, timeout=240):
            object_classes, labels = self.batch_classify_rois(img_analyser, batch_size=1)
        img_logger.info(
            "3.4 - GPU memory status after classification",
            show_memory=True,
            cuda=is_cuda,
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
            "perimeter",
            "equivalent_diameter_area",
            "feret_diameter_max",
            "moments_hu",
        ]
        img_logger.info("4.2 - Extracting fluorescence measurements")
        fl_measurements = img_analyser.get_roi_measurements(
            target_channel=pi_channel,
            properties=fl_prop,
            extra_properties=(
                roi_skewness, roi_std_dev,
                roi_glcm_features, roi_radial_profile,
                roi_skeleton_features, roi_shape_features,
            ),
            frame_extra_properties=(frame_tubularness,),
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
            "roi_skeleton_features-4": "skeleton_branch_points",
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

        # Override ghost cells: added by FL union mask, definitionally piPOS
        if ghost_records:
            ghost_df = pd.DataFrame(ghost_records, columns=["frame", "label"])
            ghost_df["_ghost"] = True
            fl_measurements = fl_measurements.merge(ghost_df, on=["frame", "label"], how="left")
            fl_measurements.loc[fl_measurements["_ghost"].notna(), "object_class"] = "ghost"
            fl_measurements = fl_measurements.drop(columns=["_ghost"])

        img_logger.info("4.3 - Extracting time metadata")
        time_data = get_timestamps(metadata, timeformat="%Y-%m-%d %H:%M:%S")
        fl_measurements = pd.merge(fl_measurements, time_data, on="frame", how="left")
        fl_measurements = pd.merge(fl_measurements, bck_fl, on="frame", how="left")
        fl_measurements[
            ["rel_max_intensity", "rel_min_intensity", "rel_mean_intensity"]
        ] = fl_measurements[["max_intensity", "min_intensity", "mean_intensity"]].div(
            fl_measurements["background"], axis=0
        )

        if align_frames and frame_shifts is not None:
            drift_df = pd.DataFrame({
                "frame": np.arange(frame_shifts.shape[0]),
                "drift_dx": frame_shifts[:, 0],
                "drift_dy": frame_shifts[:, 1],
            })
            fl_measurements = fl_measurements.merge(drift_df, on="frame", how="left")
            img_logger.info(
                f"4.3 - Alignment drift: "
                f"dx=[{frame_shifts[:, 0].min():.1f}, {frame_shifts[:, 0].max():.1f}] px  "
                f"dy=[{frame_shifts[:, 1].min():.1f}, {frame_shifts[:, 1].max():.1f}] px"
            )

        # 4.4 Morphology-based label corrections
        img_logger.info("4.4 - Applying morphology corrections", show_memory=False)
        fl_measurements, morph_counts = apply_semSeg_morphology_corrections(
            fl_measurements
        )
        img_logger.info(
            f"4.4 - Morphology corrections: "
            f"{morph_counts['interior_to_clump']} interior→clump, "
            f"{morph_counts['division_pairs_to_joint']} cells→joint-cell"
        )

        # 4.5 Object tracking (if enabled)
        if self.tracking and self.cell_tracker is not None:
            img_logger.info("4.5 - Running object tracking")
            track_features = ["area", "major_axis_length", "minor_axis_length", "solidity", "orientation"]
            self.cell_tracker.set_features(track_features)

            cost_overrides = None
            if getattr(self, "assignment_scorer", None) is not None:
                from HiTMicTools.tracking.feature_extraction import compute_movie_stats
                _masks_for_scorer = img_analyser.labeled_mask[:, 0, 0, :, :]
                _stats = compute_movie_stats(_masks_for_scorer)
                _sorted_frames = sorted(fl_measurements["frame"].unique())
                cost_overrides = {}
                for _fi, _fval in enumerate(_sorted_frames[:-1]):
                    _fval_next = _sorted_frames[_fi + 1]
                    _cm, _lt, _lt1 = self.assignment_scorer.predict_cost_matrix(
                        _masks_for_scorer[_fval],
                        _masks_for_scorer[_fval_next],
                        _stats,
                        masks_t_prev=_masks_for_scorer[_fval - 1] if _fval > 0 else None,
                    )
                    cost_overrides[_fval] = (_cm, _lt, _lt1)
                img_logger.info(f"4.5 - Learned cost overrides computed for {len(cost_overrides)} frame pairs")

            try:
                fl_measurements = self.cell_tracker.track_objects(
                    fl_measurements, volume_bounds=(size_x, size_y),
                    logger=img_logger, pixel_size=pixel_size,
                    cost_overrides=cost_overrides,
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

            # 4.6b piPOS lock-in (if tracker supports it)
            if (
                self.tracking
                and self.cell_tracker is not None
                and hasattr(self.cell_tracker, "apply_pipos_lockin")
            ):
                img_logger.info("4.6b - Applying piPOS lock-in")
                fl_measurements = self.cell_tracker.apply_pipos_lockin(
                    fl_measurements, logger=img_logger
                )

            fl_measurements["file"] = name
            # Generate summary data using the dedicated method
            d_summary = self.generate_data_summary(
                fl_measurements,
                [
                    "file",
                    "frame",
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
            fl_measurements["file"] = name
            d_summary = pd.DataFrame()

        # 4.7 Track event detection + FL trajectory (requires final pi_class)
        if self.tracking and self.cell_tracker is not None:
            img_logger.info("4.7 - Refining tracks + detecting events", show_memory=False)
            fl_measurements, refine_counts = refine_tracks(fl_measurements)
            img_logger.info(
                f"4.7 - Track refinement: "
                f"{refine_counts['short_tracks']} short tracks, "
                f"{refine_counts['bad_frame_rows']} bad frames, "
                f"{refine_counts['class_flicker_rows']} class flicker rows"
            )
            fl_measurements, div_counts = detect_division_events(fl_measurements)
            if getattr(self, "division_classifier", None) is not None:
                masks_thw = img_analyser.labeled_mask[:, 0, 0, :, :]
                fl_measurements, rec_counts = self.division_classifier.predict_divisions(
                    fl_measurements, masks_thw)
            else:
                fl_measurements, rec_counts = reconcile_lineage(fl_measurements, **self._get_reconcile_lineage_kwargs())
            fl_measurements, lys_counts = detect_lysis_events(fl_measurements)
            fl_measurements, fil_counts = detect_filamentation_events(fl_measurements)
            fl_measurements = compute_fl_trajectory_features(fl_measurements)
            img_logger.info(
                f"4.7 - Track events: "
                f"{div_counts['n_division_events']} divisions (original), "
                f"{rec_counts['n_reconciled_divisions']} divisions (reconciled), "
                f"{lys_counts['n_lysis_events']} lysis events, "
                f"{fil_counts['n_filamentation_events']} filamentation events"
            )

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
            image_8bit = convert_image(
                img_analyser.get("image", to_numpy=True),
                np.uint8,
            )
            tifffile.imwrite(export_path + "_transformed.tiff", image_8bit, imagej=True)

        if export_training_crops:
            from HiTMicTools.data_analysis.training_data_export import TrainingDataExporter
            exporter = TrainingDataExporter(crop_size=training_crop_size)
            crop_counts = exporter.export(
                fl_measurements=fl_measurements,
                image=img_analyser.get("image", to_numpy=True),
                labeled_mask=img_analyser.get("labels", index=(slice(None), 0, 0), to_numpy=True),
                output_path=export_path + "_crops",
                species=getattr(self, "species", None),
            )
            img_logger.info(
                "5 - Training crops: {} exported — {}".format(
                    crop_counts["total"],
                    ", ".join(f"{cls}={n}" for cls, n in sorted(crop_counts["per_class"].items()))
                )
            )

        img_logger.info(f"Analysis completed for {movie_name}", show_memory=True)
        del prob_map, img, fl_measurements, d_summary, img_analyser
        gc.collect()
        empty_gpu_cache(device)
        img_logger.info("Garbage collection completed", show_memory=True)

        self.remove_logger(img_logger)

        return name


