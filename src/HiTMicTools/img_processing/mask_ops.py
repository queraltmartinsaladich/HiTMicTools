from typing import Any, List, Tuple, Union, Optional, Dict
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, shift as ndimage_shift
from skimage.filters import threshold_otsu
from skimage.measure import label as skimage_label, regionprops
from skimage.segmentation import watershed


def map_predictions_to_labels(
    labeled_image: np.ndarray,
    predictions: Union[List[str], List[int], np.ndarray],
    labels: Union[List[int], np.ndarray],
    value_map: Optional[Dict[str, int]] = None,
    background_value: int = 0,
) -> np.ndarray:
    """
    Maps prediction values onto a labeled image based on label IDs.

    This function creates a new image where each labeled region is assigned
    a value corresponding to its prediction class. This is useful for
    visualizing classification results directly on the segmented image.

    Args:
        labeled_image: Integer-labeled image where each object has a unique ID.
            Shape can be 2D (single image) or 3D (time series of images).
        predictions: List or array of prediction values (strings or integers)
            corresponding to each label in 'labels'.
        labels: List or array of label IDs that correspond to the predictions.
            Must be the same length as 'predictions'.
        value_map: Optional dictionary mapping string prediction values to integers.
            If None, predictions are assumed to be integers or convertible to integers.
        background_value: Value to assign to background pixels (where labeled_image == 0).
            Defaults to 0.

    Returns:
        np.ndarray: A new image with the same shape as labeled_image, where each
            labeled region is filled with its corresponding prediction value.

    Example:
        >>> # Map cell types onto a labeled image
        >>> cell_types = ["single-cell", "clump", "noise"]
        >>> label_ids = [1, 2, 3]
        >>> type_map = {"single-cell": 1, "clump": 2, "noise": 3}
        >>> type_image = map_predictions_to_labels(labeled_img, cell_types, label_ids, type_map)

        >>> # Map binary classification (e.g., PI positive/negative) onto a labeled image
        >>> pi_status = ["piPOS", "piNEG", "piPOS"]
        >>> label_ids = [1, 2, 3]
        >>> pi_map = {"piPOS": 1, "piNEG": 2}
        >>> pi_image = map_predictions_to_labels(labeled_img, pi_status, label_ids, pi_map)
    """
    # Validate inputs
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length")

    # Convert predictions to integers if a value map is provided
    if value_map is not None:
        pred_values = np.array(
            [value_map.get(pred, background_value) for pred in predictions]
        )
    else:
        # Try to convert predictions to integers directly
        try:
            pred_values = np.array([int(pred) for pred in predictions])
        except (ValueError, TypeError):
            raise ValueError(
                "Predictions must be integers or convertible to integers if no value_map is provided"
            )

    # Create a mapping from label IDs to prediction values
    label_to_pred = {label: pred for label, pred in zip(labels, pred_values)}

    # Define a vectorized function to map labels to predictions
    def map_label_to_pred(x):
        """Return the mapped prediction value for a given ROI label ID."""
        if x == 0:  # Background
            return background_value
        return label_to_pred.get(
            x, background_value
        )  # Default to background_value if label not found

    vectorized_map = np.vectorize(map_label_to_pred)

    # Apply the mapping to create the prediction image
    prediction_image = vectorized_map(labeled_image)

    return prediction_image


def map_predictions_to_labels_by_frame(
    labeled_stack: np.ndarray,
    measurements: Any,
    prediction_col: str,
    label_col: str = "label",
    frame_col: str = "frame",
    value_map: Optional[Dict[str, int]] = None,
    background_value: int = 0,
) -> np.ndarray:
    """Map per-object predictions onto a time stack with per-frame label IDs."""
    if labeled_stack.ndim != 3:
        raise ValueError("labeled_stack must have shape (T, Y, X)")

    unique_frames = sorted(measurements[frame_col].unique())
    if len(unique_frames) != labeled_stack.shape[0]:
        raise ValueError(
            f"Frame count mismatch: stack has {labeled_stack.shape[0]} frames "
            f"but measurements contain {len(unique_frames)} unique frame values"
        )

    mapped_frames = []
    for stack_idx, frame_num in enumerate(unique_frames):
        frame_measurements = measurements[measurements[frame_col] == frame_num]
        mapped_frames.append(
            map_predictions_to_labels(
                labeled_stack[stack_idx],
                frame_measurements[prediction_col].tolist(),
                frame_measurements[label_col].tolist(),
                value_map=value_map,
                background_value=background_value,
            )
        )

    return np.stack(mapped_frames, axis=0)


def apply_fl_union_mask(
    labeled_mask: np.ndarray,
    fl_norm: np.ndarray,
    min_area: int = 20,
    fl_threshold: Optional[float] = None,
    overlap_tol: float = 0.1,
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Add FL-detected ghost cells into *labeled_mask* in-place.

    Ghost cells are FL-positive connected components with negligible overlap with
    the existing BF segmentation — cells that lost phase contrast after death but
    retain PI fluorescence.  Their labels are appended starting from
    ``max_existing_label + 1`` per frame.

    Args:
        labeled_mask: Shape (T, S, C, X, Y) int array, modified in-place.
            Only the view [:, 0, 0, :, :] is used and written back.
        fl_norm: Normalized FL stack, shape (T, S, X, Y) or (T, X, Y),
            float32 in [0, 1].
        min_area: Minimum pixel area for a FL component to be added.
        fl_threshold: Fixed threshold in [0, 1].  If None, per-frame Otsu on
            the full frame is used.
        overlap_tol: Max fractional overlap of a FL component with existing
            labels before it is rejected (not a ghost).  Default 0.1 (≤10 %).

    Returns:
        Tuple of (n_added, ghost_records) where n_added is the total count of
        ghost cells added and ghost_records is a list of (frame_idx, label_id)
        for each added ghost cell — used downstream to assign object_class and
        pi_class by construction (bypassing the classifier).
    """
    # Working views: (T, X, Y)
    fl_work = fl_norm[:, 0, :, :] if fl_norm.ndim == 4 else fl_norm
    lm_work = labeled_mask[:, 0, 0, :, :]  # view — writes propagate to labeled_mask

    n_added = 0
    ghost_records: List[Tuple[int, int]] = []

    for t in range(fl_work.shape[0]):
        fl_frame = fl_work[t]
        if fl_frame.max() == 0:
            continue

        if fl_threshold is not None:
            thr = fl_threshold
        else:
            thr = threshold_otsu(fl_frame)

        fl_labeled = skimage_label(fl_frame > thr, connectivity=2)
        if fl_labeled.max() == 0:
            continue

        lm_frame = lm_work[t]
        next_label = int(lm_frame.max()) + 1

        for comp_id in range(1, int(fl_labeled.max()) + 1):
            comp_mask = fl_labeled == comp_id
            area = int(comp_mask.sum())
            if area < min_area:
                continue
            overlap = int((lm_frame[comp_mask] > 0).sum())
            if area > 0 and overlap / area > overlap_tol:
                continue
            lm_frame[comp_mask] = next_label
            ghost_records.append((t, next_label))
            next_label += 1
            n_added += 1

    return n_added, ghost_records


def refine_masks_temporal(
    labeled_mask: np.ndarray,
    gradient_map: Optional[np.ndarray] = None,
    min_seed_overlap: float = 0.3,
) -> int:
    """Split merged instances using previous-frame centroids as watershed seeds.

    For each frame t > 0, any region that contains two or more centroids from
    the previous frame (with sufficient area overlap) is re-split via watershed.
    This corrects the common failure mode where two touching cells are merged
    into a single label in one frame while they were separate in the frame before.

    The labeled_mask is modified in-place.  Frame 0 is always unchanged.
    Refinements propagate forward: the corrected mask for frame t is used as the
    prior for frame t+1.

    Args:
        labeled_mask: Shape (T, S, C, H, W) int array — only the view
            [:, 0, 0, :, :] (T, H, W) is read and written, matching the
            convention used by apply_fl_union_mask.
        gradient_map: Shape (T, H, W) float where HIGH values mark boundaries
            (e.g. 1 - prob_map[:, 0] for a UNet foreground probability map).
            If None, a distance-transform of each candidate region is used as
            the fallback elevation for watershed.
        min_seed_overlap: Fraction of a t-1 cell's pixel area that must overlap
            with the candidate t region for that centroid to count as a seed.
            Filters out centroids from cells that moved out of the region.
            Default 0.3.

    Returns:
        Total number of regions split across all frames.
    """
    lm = labeled_mask[:, 0, 0, :, :]  # view — writes propagate to labeled_mask
    T, H, W = lm.shape
    n_splits = 0

    for t in range(1, T):
        prev = lm[t - 1]
        curr = lm[t].copy()

        props_prev = regionprops(prev)
        if not props_prev:
            continue

        prev_info: Dict[int, Tuple[int, Tuple[int, int]]] = {
            p.label: (p.area, (int(round(p.centroid[0])), int(round(p.centroid[1]))))
            for p in props_prev
        }

        to_split: Dict[int, List[Tuple[int, int]]] = {}
        for curr_label in np.unique(curr):
            if curr_label == 0:
                continue
            region = curr == curr_label
            seeds: List[Tuple[int, int]] = []
            for prev_label, (prev_area, (cy, cx)) in prev_info.items():
                if cy >= H or cx >= W:
                    continue
                if not region[cy, cx]:
                    continue
                overlap_frac = float(np.sum(region & (prev == prev_label))) / prev_area
                if overlap_frac >= min_seed_overlap:
                    seeds.append((cy, cx))
            if len(seeds) >= 2:
                to_split[curr_label] = seeds

        if not to_split:
            continue

        next_label = int(curr.max()) + 1
        for curr_label, seeds in to_split.items():
            region_mask = curr == curr_label

            if gradient_map is not None:
                elev = gradient_map[t]
            else:
                dist = distance_transform_edt(region_mask)
                elev = np.where(region_mask, -dist, 0.0)

            markers = np.zeros((H, W), dtype=np.int32)
            for i, (cy, cx) in enumerate(seeds, start=1):
                markers[cy, cx] = i

            ws = watershed(elev, markers, mask=region_mask)

            curr[ws == 1] = curr_label
            for i in range(2, len(seeds) + 1):
                curr[ws == i] = next_label
                next_label += 1
            n_splits += 1

        lm[t] = curr

    return n_splits


def recover_gap_masks(
    labeled_mask: np.ndarray,
    fl_measurements: "pd.DataFrame",
    max_gap_frames: int = 2,
) -> "tuple[pd.DataFrame, int]":
    """Insert recovered masks for tracks that were bridged across short segmentation gaps.

    After HungarianTracker bridges a 1-N frame gap (via gap_bridge_frames), the
    trackid is continuous but those frames have no mask or features. This function
    fills those gaps by translating the nearest known cell mask to the linearly
    interpolated centroid and appending interpolated feature rows to fl_measurements.

    Skips gaps longer than max_gap_frames, ghost cells, and cases where the
    translated mask would overlap an existing segmented cell.

    Args:
        labeled_mask:   (T, S, C, H, W) int array — only [:, 0, 0, :, :] is used,
                        same convention as refine_masks_temporal.
        fl_measurements: Post-tracking DataFrame, must have columns:
                         trackid, frame, label, centroid_0, centroid_1.
        max_gap_frames: Maximum gap length to recover. Should match
                        gap_bridge_frames from HungarianTracker. Default 2.

    Returns:
        (updated_fl_measurements, n_recovered): DataFrame with recovered rows
        appended (object_class="recovered") and count of inserted cells.
    """
    if "trackid" not in fl_measurements.columns:
        return fl_measurements, 0

    lm = labeled_mask[:, 0, 0, :, :]  # (T, H, W) view — writes propagate in-place
    T, H, W = lm.shape

    numeric_cols = fl_measurements.select_dtypes(include=np.number).columns.tolist()
    id_cols = {"frame", "label", "trackid"}

    recovered_rows = []

    for trackid, group in fl_measurements.groupby("trackid"):
        group_sorted = group.sort_values("frame")
        frames = group_sorted["frame"].tolist()
        if len(frames) < 2:
            continue

        frame_set = set(frames)
        t_min, t_max = frames[0], frames[-1]

        for t_gap in range(t_min + 1, t_max):
            if t_gap in frame_set or t_gap >= T:
                continue

            t_before = max(f for f in frames if f < t_gap)
            t_after = min(f for f in frames if f > t_gap)
            if (t_after - t_before - 1) > max_gap_frames:
                continue

            row_before = group_sorted[group_sorted["frame"] == t_before].iloc[0]
            row_after = group_sorted[group_sorted["frame"] == t_after].iloc[0]

            # Skip ghost cells — they have dedicated handling
            if row_before.get("object_class") == "ghost":
                continue

            alpha = (t_gap - t_before) / (t_after - t_before)

            cy = row_before["centroid_0"] * (1 - alpha) + row_after["centroid_0"] * alpha
            cx = row_before["centroid_1"] * (1 - alpha) + row_after["centroid_1"] * alpha

            # Source mask: nearest neighbour in time
            src_row = row_before if alpha <= 0.5 else row_after
            src_frame = int(src_row["frame"])
            src_label = int(src_row["label"])
            src_cy = float(src_row["centroid_0"])
            src_cx = float(src_row["centroid_1"])

            src_bin = (lm[src_frame] == src_label).astype(np.float32)
            if not src_bin.any():
                continue

            dy = cy - src_cy
            dx = cx - src_cx
            shifted = ndimage_shift(src_bin, [dy, dx], mode="constant", cval=0.0) > 0.5

            if not shifted.any():
                continue
            if np.any(lm[t_gap][shifted] != 0):
                # Would overlap an existing cell — skip rather than overwrite
                continue

            new_label = int(lm[t_gap].max()) + 1
            lm[t_gap][shifted] = new_label

            # Build interpolated feature row
            new_row: dict = {}
            for col in numeric_cols:
                if col in id_cols:
                    continue
                v_before = row_before.get(col)
                v_after = row_after.get(col)
                try:
                    new_row[col] = float(v_before) * (1 - alpha) + float(v_after) * alpha
                except (TypeError, ValueError):
                    new_row[col] = v_before  # fallback: copy from t_before

            # Copy non-numeric categorical columns from nearest neighbour
            for col in fl_measurements.columns:
                if col in new_row or col in id_cols:
                    continue
                new_row[col] = src_row.get(col)

            new_row["frame"] = t_gap
            new_row["label"] = new_label
            new_row["trackid"] = trackid
            new_row["object_class"] = "recovered"

            recovered_rows.append(new_row)

    if not recovered_rows:
        return fl_measurements, 0

    recovered_df = pd.DataFrame(recovered_rows, columns=fl_measurements.columns)
    updated = (
        pd.concat([fl_measurements, recovered_df], ignore_index=True)
        .sort_values(["frame", "trackid"])
        .reset_index(drop=True)
    )
    return updated, len(recovered_rows)
