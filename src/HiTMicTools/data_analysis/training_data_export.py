"""Per-cell crop export for classifier retraining.

Extracts fixed-size BF+FL patches centered on each cell centroid, filters by
track_quality, optionally caps majority classes, and writes a compressed numpy
archive + metadata CSV.

Typical call from a pipeline at step 5 (while image data is still in memory):

    from HiTMicTools.data_analysis.training_data_export import TrainingDataExporter

    exporter = TrainingDataExporter(crop_size=64, bf_channel=0, fl_channel=1)
    counts = exporter.export(
        fl_measurements=fl_measurements,
        image=img_analyser.get("image", to_numpy=True),
        labeled_mask=img_analyser.get("labels", index=(slice(None), 0, 0), to_numpy=True),
        output_path=export_path + "_crops",
        species=self.species,
    )

Output files
------------
``{output_path}.npz``  — compressed archive: images (N, 2, H, W) float32,
                          labels (N,) str.
``{output_path}.csv``  — per-crop metadata (one row per exported crop).

Crop convention
---------------
Images stored as (2, H, W): channel 0 = BF, channel 1 = FL.  Each crop is
independently contrast-normalised to [0, 1] using its 1st/99th percentile so
the classifier sees shape and texture rather than absolute intensity.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TrainingDataExporter:
    """Export per-cell image crops and metadata for model retraining.

    Args:
        crop_size: Side length in pixels of the square output patch.  Default 64.
        bf_channel: Channel index for brightfield in the image stack.  Default 0.
        fl_channel: Channel index for fluorescence in the image stack.  Default 1.
        track_quality_filter: Only export rows where track_quality equals this
            value.  Set to None to skip the filter.  Default "ok".
        exclude_classes: object_class values to always exclude.  Default ("ghost",).
        max_cells_per_class: If set, randomly subsample each class to at most this
            many crops (class balancing).  Default None (no cap).
        random_seed: Seed for class-balancing RNG.  Default 42.
    """

    def __init__(
        self,
        crop_size: int = 64,
        bf_channel: int = 0,
        fl_channel: int = 1,
        track_quality_filter: Optional[str] = "ok",
        exclude_classes: Tuple[str, ...] = ("ghost",),
        max_cells_per_class: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        self.crop_size = crop_size
        self.bf_channel = bf_channel
        self.fl_channel = fl_channel
        self.track_quality_filter = track_quality_filter
        self.exclude_classes = set(exclude_classes)
        self.max_cells_per_class = max_cells_per_class
        self.random_seed = random_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        fl_measurements: pd.DataFrame,
        image: np.ndarray,
        labeled_mask: np.ndarray,
        output_path: str,
        species: Optional[str] = None,
    ) -> Dict[str, int]:
        """Extract crops and write NPZ + CSV.

        Args:
            fl_measurements: DataFrame with at least frame, label, centroid_0,
                centroid_1, object_class columns.  track_quality and
                object_class_raw are used when present.
            image: Image stack — shape (T, S, C, H, W) or (T, C, H, W).
                BF and FL are extracted via bf_channel and fl_channel indices
                after squeezing the S dimension (if present) to index 0.
            labeled_mask: Integer label mask — shape (T, H, W).
            output_path: Base path for output files (no extension).
                Writes ``{output_path}.npz`` and ``{output_path}.csv``.
            species: Species label written into the metadata CSV.

        Returns:
            Dict with keys: total, per_class (dict), skipped_missing_centroid,
            skipped_near_border (cells where the bounding box was partially
            clipped and any dimension of the crop was less than crop_size // 2).
        """
        df = self._filter(fl_measurements)
        if df.empty:
            warnings.warn(
                f"TrainingDataExporter: _filter produced 0 rows from {len(fl_measurements)} input rows. "
                f"Check track_quality_filter={self.track_quality_filter!r} and "
                f"exclude_classes={self.exclude_classes!r}. No output written to {output_path!r}.",
                UserWarning,
                stacklevel=2,
            )
            return {"total": 0, "per_class": {}, "skipped_missing_centroid": 0,
                    "skipped_near_border": 0}

        df = self._balance(df)

        # Normalise image shape to (T, C, H, W) — squeeze S dim if present
        img = image[:, 0] if image.ndim == 5 else image  # (T, C, H, W)

        crops, meta_rows, n_missing, n_border = self._extract_crops(df, img)

        if not crops:
            return {"total": 0, "per_class": {}, "skipped_missing_centroid": n_missing,
                    "skipped_near_border": n_border}

        images_arr = np.stack(crops, axis=0).astype(np.float32)   # (N, 2, H, W)
        labels_arr = np.array([r["object_class"] for r in meta_rows], dtype=object)

        np.savez_compressed(output_path + ".npz", images=images_arr, labels=labels_arr)

        meta_df = pd.DataFrame(meta_rows)
        if species is not None:
            meta_df.insert(0, "species", species)
        meta_df.to_csv(output_path + ".csv", index=False)

        per_class = meta_df["object_class"].value_counts().to_dict()
        return {
            "total": len(crops),
            "per_class": per_class,
            "skipped_missing_centroid": n_missing,
            "skipped_near_border": n_border,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter(self, fl: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=fl.index)
        if self.track_quality_filter and "track_quality" in fl.columns:
            mask &= fl["track_quality"] == self.track_quality_filter
        if self.exclude_classes and "object_class" in fl.columns:
            mask &= ~fl["object_class"].isin(self.exclude_classes)
        # Drop untracked rows that have no meaningful trajectory label
        if "trackid" in fl.columns:
            mask &= fl["trackid"] != -1
        return fl[mask].copy()

    def _balance(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.max_cells_per_class is None or "object_class" not in df.columns:
            return df
        rng = np.random.default_rng(self.random_seed)
        parts: List[pd.DataFrame] = []
        for _, group in df.groupby("object_class"):
            if len(group) > self.max_cells_per_class:
                idx = rng.choice(group.index, self.max_cells_per_class, replace=False)
                parts.append(group.loc[idx])
            else:
                parts.append(group)
        return pd.concat(parts).sort_index()

    def _extract_crops(
        self,
        df: pd.DataFrame,
        img: np.ndarray,  # (T, C, H, W)
    ) -> Tuple[List[np.ndarray], List[dict], int, int]:
        half = self.crop_size // 2
        H, W = img.shape[-2], img.shape[-1]
        crops: List[np.ndarray] = []
        meta_rows: List[dict] = []
        n_missing = 0
        n_border = 0

        keep_cols = [c for c in (
            "frame", "label", "trackid", "object_class", "object_class_raw",
            "track_quality", "area", "major_axis_length", "aspect_ratio",
            "centroid_0", "centroid_1",
        ) if c in df.columns]

        for row in df[keep_cols].itertuples(index=False):
            row_d = row._asdict()
            frame = int(row_d.get("frame", 0))
            cy = row_d.get("centroid_0")  # row index in image
            cx = row_d.get("centroid_1")  # col index in image

            if cy is None or cx is None or (isinstance(cy, float) and np.isnan(cy)):
                n_missing += 1
                continue

            cy, cx = int(round(float(cy))), int(round(float(cx)))

            r0, r1 = cy - half, cy + half
            c0, c1 = cx - half, cx + half

            # Detect near-border cells where crop would be <half the expected size
            actual_h = min(r1, H) - max(r0, 0)
            actual_w = min(c1, W) - max(c0, 0)
            if actual_h < half or actual_w < half:
                n_border += 1
                continue

            # Extract and pad
            patch = np.zeros((2, self.crop_size, self.crop_size), dtype=np.float32)
            pr0 = max(0, -r0)
            pc0 = max(0, -c0)
            ir0, ir1 = max(0, r0), min(H, r1)
            ic0, ic1 = max(0, c0), min(W, c1)

            bf_frame = img[frame, self.bf_channel].astype(np.float32)
            fl_frame = img[frame, self.fl_channel].astype(np.float32)

            patch[0, pr0:pr0 + (ir1 - ir0), pc0:pc0 + (ic1 - ic0)] = bf_frame[ir0:ir1, ic0:ic1]
            patch[1, pr0:pr0 + (ir1 - ir0), pc0:pc0 + (ic1 - ic0)] = fl_frame[ir0:ir1, ic0:ic1]

            patch = self._normalise_crop(patch)
            crops.append(patch)
            meta_rows.append(row_d)

        return crops, meta_rows, n_missing, n_border

    @staticmethod
    def _normalise_crop(patch: np.ndarray) -> np.ndarray:
        """Scale each channel to [0, 1] using its 1st/99th percentile."""
        out = patch.copy()
        for c in range(patch.shape[0]):
            ch = patch[c]
            lo = float(np.percentile(ch, 1))
            hi = float(np.percentile(ch, 99))
            if hi > lo:
                out[c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
            else:
                out[c] = 0.0
        return out
