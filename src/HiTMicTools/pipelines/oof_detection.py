import gc
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from HiTMicTools.img_processing.img_processor import ImagePreprocessor
from HiTMicTools.resource_management.reserveresource import ReserveResource
from HiTMicTools.resource_management.sysutils import empty_gpu_cache, get_device
from HiTMicTools.roianalysis import RoiAnalyser
from HiTMicTools.utils import remove_file_extension
from HiTMicTools.pipelines.base_pipeline import BasePipeline
from jetraw_tools.image_reader import ImageReader


class OOF_detection(BasePipeline):
    """
    Pipeline dedicated to quantifying out-of-focus detections alongside
    standard single-cell segmentation/classification on the first brightfield
    frame of a microscopy stack.
    """

    # Models required by this pipeline
    required_models = {"bf_focus", "fl_focus", "oof_detector", "segmentation", "cell_classifier"}

    def __init__(
        self,
        input_path: str,
        output_path: str,
        worklist_path: Optional[str] = None,
        file_type: str = ".nd2",
    ):
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            worklist_path=worklist_path,
            file_type=file_type,
        )
        self._oof_label_lookup: Optional[Dict[int, str]] = None

    def analyse_image(
        self,
        file_i: str,
        name: str,
        export_labeled_mask: bool = False,
        export_aligned_image: bool = False,
    ) -> str:
        """Run OOF detection and standard cell quantification on the first frame."""

        device = get_device()
        is_cuda = device.type == "cuda"
        movie_name = remove_file_extension(name)
        logger = self.setup_logger(self.output_path, movie_name)
        logger.info(f"Start OOF analysis for {movie_name}")
        if export_labeled_mask:
            logger.info(
                "export_labeled_mask flag received but not used in OOF_detection pipeline"
            )
        if export_aligned_image:
            logger.info(
                "export_aligned_image flag received but not used in OOF_detection pipeline"
            )

        if not hasattr(self, "reference_channel"):
            raise AttributeError(
                "Pipeline missing 'reference_channel'; ensure configuration is loaded."
            )
        if not hasattr(self, "method"):
            self.method = "standard"

        if self.oof_detector is None:
            raise RuntimeError("OOF detector must be loaded before running the pipeline.")
        if self.bf_focus_restorer is None:
            raise RuntimeError("Brightfield focus restorer is required for this pipeline.")
        if self.image_segmentator is None:
            raise RuntimeError("Segmentation model is required for this pipeline.")
        if self.object_classifier is None:
            raise RuntimeError("Object classifier is required for this pipeline.")

        logger.info("1 - Reading image", show_memory=True)
        image_reader = ImageReader(file_i, self.file_type)
        img, metadata = image_reader.read_image()
        # Remove after debugging
        logger.info(f"Image shape {img.shape}", show_memory=True)
        try:
            pixel_size = metadata.images[0].pixels.physical_size_x
            size_x = metadata.images[0].pixels.size_x
            size_y = metadata.images[0].pixels.size_y
            n_channels = metadata.images[0].pixels.size_c
            n_frames = metadata.images[0].pixels.size_t
        except Exception as e:
            # Fallback to "default metadata"
            pixel_size = 0.65  # Default pixel size in microns (typical for microscopy)
            size_x, size_y, n_channels, n_frames = self._infer_default_img_dims(img)
            logger.warning(
                f"Could not extract metadata from {name} ({type(e).__name__}: {e}). "
                f"Using default pixel_size={pixel_size} µm"
            )

        img = img.reshape(n_frames, n_channels, size_x, size_y)
        logger.info(
            f"Image shape: {img.shape}, pixel size: {pixel_size} µm. "
            f"Reshaped to (frames={n_frames}, channels={n_channels}, x={size_x}, y={size_y})"
        )

        # Initialize preprocessor with first frame and reference channel only
        preprocessor = ImagePreprocessor(
            img[:1, self.reference_channel : self.reference_channel + 1, :, :],
            stack_order="TCXY"
        )
        img = np.zeros((1, 1, 1, 1))  # release reader buffer

        logger.info("2 - Running OOF detector on raw brightfield frame")
        # Run OOF detection on raw, unprocessed image from preprocessor
        (
            oof_boxes,
            oof_class_ids,
            oof_scores,
        ) = self.oof_detector.predict(image=preprocessor.img, frame_index=0, channel_index=0)
        oof_indices = np.where(oof_scores >= 0.5)[0]
        oof_boxes = oof_boxes[oof_indices]
        oof_class_ids = oof_class_ids[oof_indices]
        logger.info(f"Detected {len(oof_boxes)} OOF candidates after thresholding")

        logger.info("3 - Detecting and fixing well borders", show_memory=True)
        preprocessor.detect_fix_well(nframes=range(1), nslices=0, nchannels=0)

        logger.info("4 - Clearing background on brightfield frame", show_memory=True)
        self.clear_background(
            preprocessor,
            channel=0,
            n_frames=range(1),
            method=self.method,
            pixel_size=pixel_size,
        )

        logger.info("5 - Restoring focus", show_memory=True, cuda=is_cuda)
        with ReserveResource(device, 4.0, logger=logger, timeout=120):
            preprocessor.img[:, 0, 0] = self.bf_focus_restorer.predict(
                preprocessor.img[:, 0, 0],
                rescale=False,
                batch_size=1,
                buffer_steps=2,
                buffer_dim=-1,
                sw_batch_size=1,
            )

        logger.info("6 - Segmenting restored frame", show_memory=True, cuda=is_cuda)
        with ReserveResource(device, 4.0, logger=logger, timeout=120):
            prob_map = self.image_segmentator.predict(
                preprocessor.img[:, 0, 0, :, :],
                buffer_steps=2,
                buffer_dim=-1,
                sw_batch_size=1,
            )

        if prob_map.ndim == 2:
            prob_map = np.expand_dims(prob_map, axis=(0, 1))
        elif prob_map.ndim == 3:
            prob_map = np.expand_dims(prob_map, axis=1)
        elif prob_map.ndim > 3 and prob_map.shape[1] > 1:
            prob_map = np.max(prob_map, axis=1, keepdims=True)

        logger.info("7 - Extracting ROIs", show_memory=True)
        analyser = RoiAnalyser(
            preprocessor.img,
            prob_map,
            stack_order=("TSCXY", "TCXY"),
        )
        analyser.create_binary_mask()
        analyser.clean_binmask(min_pixel_size=20)
        analyser.get_labels()
        logger.info(f"Found {analyser.total_rois} ROIs")

        logger.info("8 - Classifying ROIs", show_memory=True, cuda=is_cuda)
        with ReserveResource(device, 6.0, logger=logger, timeout=180):
            object_classes, labels = self.batch_classify_rois(analyser, batch_size=1)

        measurement = analyser.get_roi_measurements(
            target_channel=0,
            properties=["label", "centroid"],
        )

        classification_df = pd.DataFrame(
            {"label": labels, "object_class": object_classes}
        )

        standard_df = measurement.merge(classification_df, on="label", how="left")

        if not standard_df.empty:
            # Fix: Use underscore instead of hyphen for centroid column names
            standard_df = standard_df.rename(
                columns={"centroid_1": "x", "centroid_0": "y"}
            )
            standard_df = standard_df[["object_class", "x", "y"]]
            standard_df["source"] = "standard"
        else:
            standard_df = pd.DataFrame(columns=["object_class", "x", "y", "source"])

        oof_df = self._build_oof_dataframe(oof_boxes, oof_class_ids)
        combined = pd.concat([standard_df, oof_df], ignore_index=True)

        export_base = os.path.join(self.output_path, movie_name)
        combined.to_csv(export_base + "_oof_summary.csv", index=False)
        logger.info(
            f"Wrote {len(combined)} combined detections to {export_base}_oof_summary.csv"
        )

        del analyser, prob_map, measurement, standard_df, oof_df, combined, preprocessor
        gc.collect()
        empty_gpu_cache(device)
        logger.info(f"Completed OOF pipeline for {movie_name}", show_memory=True)
        self.remove_logger(logger)

        return movie_name

    def batch_classify_rois(
        self,
        analyser: RoiAnalyser,
        batch_size: int = 5,
    ) -> Tuple[List[str], List[int]]:
        labeled_mask = analyser.get(
            "labels", index=(slice(None), 0, 0), to_numpy=True
        )
        image = analyser.get("image", index=(slice(None), 0, 0), to_numpy=True)
        total_frames = labeled_mask.shape[0]

        all_classes: List[str] = []
        all_labels: List[int] = []

        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            batch_mask = labeled_mask[start:end]
            batch_img = image[start:end]
            classes, labels = self.object_classifier.classify_rois(
                batch_mask, batch_img
            )
            all_classes.extend(classes)
            all_labels.extend(labels)

        return all_classes, all_labels

    def clear_background(
        self,
        preprocessor: ImagePreprocessor,
        channel: int,
        n_frames: range,
        method: str,
        pixel_size: Optional[float] = None,
    ) -> None:
        if method == "basicpy_fl":
            method = "standard"

        configs = {
            "standard": [
                {
                    "nframes": n_frames,
                    "nchannels": channel,
                    "nslices": 0,
                    "method": "divide",
                    "sigma_r": 20,
                }
            ],
            "basicpy": [
                {
                    "nframes": n_frames,
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

        if method not in configs:
            raise ValueError(f"Unsupported background removal method: {method}")

        for params in configs[method]:
            if method == "basicpy":
                preprocessor.clear_image_background(**params)
            else:
                preprocessor.clear_image_background(
                    **params, unit="um", pixel_size=pixel_size
                )

    def _build_oof_dataframe(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
    ) -> pd.DataFrame:
        if boxes.size == 0:
            return pd.DataFrame(columns=["object_class", "x", "y", "source"])

        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0

        class_lookup = self._get_oof_label_lookup()
        class_names = [
            class_lookup.get(int(cid), f"oof_{int(cid)}") for cid in class_ids
        ]

        return pd.DataFrame(
            {
                "object_class": class_names,
                "x": centers_x,
                "y": centers_y,
                "source": "oof_detector",
            }
        )

    def _get_oof_label_lookup(self) -> Dict[int, str]:
        if self._oof_label_lookup is not None:
            return self._oof_label_lookup

        lookup: Dict[int, str] = {}
        if getattr(self, "oof_class_map", None):
            sample_key = next(iter(self.oof_class_map))
            if isinstance(sample_key, str):
                lookup = {int(v): k for k, v in self.oof_class_map.items()}
            else:
                lookup = {int(k): str(v) for k, v in self.oof_class_map.items()}

        self._oof_label_lookup = lookup
        return lookup


    def _infer_default_img_dims(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Infer (size_x, size_y, channels, frames) when metadata is missing."""
        # 3D array -> (channels, X, Y)
        if len(img.shape) == 3:
            size_x = img.shape[-2]
            size_y = img.shape[-1]
            n_channels = img.shape[0]
            n_frames = 1
        # 2D array -> (X, Y)
        elif len(img.shape) == 2:
            size_x = img.shape[0]
            size_y = img.shape[1]
            n_channels = 1
            n_frames = 1
        # 4D array -> (frame, channel, X, Y)
        elif len(img.shape) == 4:
            size_x = img.shape[-2]
            size_y = img.shape[-1]
            n_channels = img.shape[1]
            n_frames = img.shape[0]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        return (size_x, size_y, n_channels, n_frames)
