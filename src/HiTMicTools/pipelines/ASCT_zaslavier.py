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
from HiTMicTools.img_processing.mask_ops import map_predictions_to_labels_by_frame
from HiTMicTools.utils import get_timestamps, remove_file_extension
from HiTMicTools.roianalysis import RoiAnalyser
from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev


from jetraw_tools.image_reader import ImageReader


class ASCT_zaslavier(BasePipeline):
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
    required_models = {"bf_focus", "fl_focus", "segmentation", "cell_classifier", "pi_classification", "sc_segmenter"}

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
        # Desync analysis to avoid RAM/VRAM issues
        img_logger = self.setup_logger(self.output_path, movie_name)
        img_logger.info(f"Start analysis for {movie_name}")
        reference_channel = self.reference_channel
        pi_channel = self.pi_channel
        gfp_channel = getattr(self, "gfp_channel", None)
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

        # 2.1 Remove background
        img_logger.info("2.1 - Preprocessing image", show_memory=True)
        img_logger.info(f"Preprocessed image shape: {ip.img.shape}")

        # 2.3 Align frames if required
        if align_frames:
            img_logger.info("2.1 - Aligning frames in the stack", show_memory=True)
            ip.align_image(
                ref_channel=0, ref_slice=-1, crop_image=True, reference_type="dynamic"
            )
            img_logger.info("2.1 - Frame alignment completed", show_memory=True)
        # Update size x and size y after alignment and maybe crop
        size_x, size_y = ip.img.shape[-2], ip.img.shape[-1]
        img_logger.info("2.1 - Detecting and fixing border wells")
        _channels = [reference_channel, pi_channel]
        if gfp_channel is not None:
            _channels.append(gfp_channel)
        ip.detect_fix_well(nchannels=_channels, nslices=0, nframes=range(nFrames))
        img_logger.info(
            f"Reference channel intensity before background removal:\n{self.check_px_values(ip, reference_channel, round=3)}"
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
        if gfp_channel is not None:
            self.clear_background(
                ip, channel=gfp_channel, nFrames=range(nFrames), method=method
            )

        # 2.2 Focus restoration (conditional)
        if getattr(self, 'focus_correction', True):  # Default to True for backward compatibility
            img_logger.info(
                "2.2 - Restoring focus in the reference channel", show_memory=True
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
            img_logger.info("2.2 - Restoring focus in the PI channel", show_memory=True)
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

            if gfp_channel is not None:
                img_logger.info("2.2 - Restoring focus in the GFP channel", show_memory=True)
                with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
                    ip.img[:, 0, gfp_channel] = self.fl_focus_restorer.predict(
                        ip.img[:, 0, gfp_channel],
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
            img_logger.info("2.2 - Focus correction disabled, skipping focus restoration", show_memory=True)

        # 2.4 Remove original image (not used after background corr) to save mem
        ip.img_original = np.zeros((1, 1, 1, 1, 1))

        # 3. Segmentation + Classification ------------------------------------
        use_sc_segmenter = getattr(self, "sc_segmenter", None) is not None

        if use_sc_segmenter:
            # --- ScSegmenter path (BacDETR / RF-DETR) ---
            img_logger.info("3 - Running single-step instance segmentation and classification", show_memory=True, cuda=is_cuda)

            with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
                frames = ip.img[:, 0, reference_channel, :, :]
                frames = np.clip(frames, 0, 1)  # Ensure [0, 1] range

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
            img_analyser = RoiAnalyser.from_labeled_mask(
                ip.img,
                stacked_labeled_masks,
                stack_order=("TSCXY", "TXY")
            )

            # Remove image-processor to release space
            del ip

            # Flatten class IDs and scores across frames for mapping to measurements
            object_classes = []
            for frame_idx, frame_classes in enumerate(all_class_ids):
                if self.class_dict:
                    frame_class_names = [self.class_dict[int(cid)] for cid in frame_classes]
                else:
                    frame_class_names = [f"class_{int(cid)}" for cid in frame_classes]
                object_classes.extend(frame_class_names)

            img_logger.info(f"3.2 - {img_analyser.total_rois} objects found in segmentation")

        else:
            # --- Legacy Segmentator + CellClassifier path ---
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

            # Remove image-processor to release space
            del ip
            img_analyser.create_binary_mask()
            img_analyser.clean_binmask(min_pixel_size=20)
            img_analyser.get_labels()
            img_logger.info(f"{img_analyser.total_rois} objects found in segmentation")

            # 3.3 Classify ROIs
            img_logger.info("3.3 - Classifying ROIs", show_memory=True, cuda=is_cuda)
            with ReserveResource(device, 12.0, logger=img_logger, timeout=240):
                object_classes, labels = self.batch_classify_rois(img_analyser, batch_size=4)
            img_logger.info(
                "3.3 - GPU memory status after classification",
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
        ]
        img_logger.info("4.2 - Extracting fluorescence measurements")
        fl_measurements = img_analyser.get_roi_measurements(
            target_channel=pi_channel,
            properties=fl_prop,
            extra_properties=(roi_skewness, roi_std_dev),
        )
        fl_measurements["object_class"] = object_classes

        # If measure_gfp is set, measure mean_intensity from gfp_channel and merge
        if getattr(self, "measure_gfp", False):
            gfp_channel = getattr(self, "gfp_channel", None)
            if gfp_channel is not None:
                # Measure GFP background intensity first
                img_logger.info("4.2b - Extracting GFP background intensity")
                gfp_bck = measure_background_intensity(
                    img_analyser.get("image", to_numpy=False),
                    img_analyser.get("labels", to_numpy=False),
                    target_channel=gfp_channel,
                )
                gfp_bck = gfp_bck.rename(columns={"background": "gfp_background"})

                # Measure GFP mean intensity for ROIs
                img_logger.info("4.2c - Extracting GFP mean intensity")
                gfp_measurements = img_analyser.get_roi_measurements(
                    target_channel=gfp_channel,
                    properties=["label", "mean_intensity"],
                )
                # Rename mean_intensity to mean_intensity_gfp to avoid overlap
                gfp_measurements = gfp_measurements.rename(
                    columns={"mean_intensity": "mean_intensity_gfp"}
                )

                # Merge GFP background with GFP measurements on 'frame'
                gfp_merged = pd.merge(gfp_measurements, gfp_bck, on="frame", how="left")

                # Merge the GFP results to the main fluorescence measurements on 'label'
                fl_measurements = pd.merge(
                    fl_measurements, gfp_merged, on=["frame", "label"], how="left"
                )
            else:
                img_logger.warning("measure_gfp is True but gfp_channel is not set.")

        img_logger.info("4.3 - Extracting time metadata")
        time_data = get_timestamps(metadata, timeformat="%Y-%m-%d %H:%M:%S")
        fl_measurements = pd.merge(fl_measurements, time_data, on="frame", how="left")
        fl_measurements = pd.merge(fl_measurements, bck_fl, on="frame", how="left")
        fl_measurements[
            ["rel_max_intensity", "rel_min_intensity", "rel_mean_intensity"]
        ] = fl_measurements[["max_intensity", "min_intensity", "mean_intensity"]].div(
            fl_measurements["background"], axis=0
        )

        # 4.4 Object tracking (if enabled)
        if self.tracking and self.cell_tracker is not None:
            img_logger.info("4.4 - Running object tracking")
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
                img_logger.info(f"4.4 - Learned cost overrides computed for {len(cost_overrides)} frame pairs")

            try:
                fl_measurements = self.cell_tracker.track_objects(
                    fl_measurements, volume_bounds=(size_x, size_y), logger=img_logger,
                    cost_overrides=cost_overrides, pixel_size=pixel_size,
                )
                img_logger.info("4.4 - Object tracking completed successfully")
            except Exception as e:
                img_logger.error(f"Object tracking failed: {e}")
                # Continue without tracking

        counts_per_frame = fl_measurements["frame"].value_counts().sort_index()
        img_logger.info(f"4 - Object counts per frame:\n{counts_per_frame.to_string()}")
        img_logger.info("4 - Measurements completed", show_memory=True)

        # 4.1 PI classification
        ghost_mask = fl_measurements["object_class"] == "ghost"
        if self.pi_classifier is not None:
            img_logger.info("4.4 - Running PI classification", show_memory=True)
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
            d_summary = pd.DataFrame()

        # 5. Export data --------------------------------------------
        export_path = os.path.join(self.output_path, name)
        img_logger.info(f"5 - Writing output data to {export_path}")

        fl_measurements.to_csv(export_path + "_fl.csv")
        d_summary.to_csv(export_path + "_summary.csv")

        if export_labeled_mask:
            label_slice = img_analyser.get(
                "labels", index=(slice(None), 0, 0), to_numpy=True
            )

            # Build class value map depending on segmentation backend
            if use_sc_segmenter and self.class_dict:
                class_value_map = {
                    name: class_idx + 1
                    for class_idx, name in sorted(self.class_dict.items())
                }
            else:
                # Legacy classifier path
                class_value_map = {
                    "single-cell": 1,
                    "clump": 2,
                    "noise": 3,
                    "off-focus": 4,
                    "joint-cell": 5,
                }

            # Use fl_measurements labels (works for both paths)
            label_ids = fl_measurements["label"].tolist()

            # Map object classes to the labeled mask
            object_class_mask = map_predictions_to_labels_by_frame(
                label_slice,
                fl_measurements,
                "object_class",
                value_map=class_value_map,
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

        img_logger.info(f"Analysis completed for {movie_name}", show_memory=True)
        del img, fl_measurements, d_summary, img_analyser
        try:
            del stacked_labeled_masks
        except NameError:
            pass
        try:
            del prob_map
        except NameError:
            pass
        gc.collect()
        empty_gpu_cache(device)
        img_logger.info("Garbage collection completed", show_memory=True)

        self.remove_logger(img_logger)

        return name


