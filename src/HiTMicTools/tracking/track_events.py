"""Post-tracking event detection and FL trajectory analysis.

Called after tracking (step 4.5) and PI classification / piPOS lock-in are
complete, so all columns (trackid, pi_class, object_class, rel_mean_intensity)
are final before any of these functions run.

Public API
----------
detect_division_events     — temporal parent→daughter split detection
detect_lysis_events        — single-cell→lyse transition per track
detect_filamentation_events — persistent long-class runs per track
compute_fl_trajectory_features — per-track FL intensity stats + smoothing
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Division event detection
# ---------------------------------------------------------------------------

def detect_division_events(
    fl_measurements: pd.DataFrame,
    centroid_dist_frac: float = 1.5,
    angle_deg: float = 40.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Detect parent→daughter division events across time.

    A division event is inferred when:
      1. A track ends at frame t  (present in t, absent in t+1).
      2. Exactly two new tracks appear in frame t+1.
      3. Both new track centroids are within centroid_dist_frac × median
         major_axis_length of the parent's last centroid.
      4. The two daughters' major axes are within angle_deg of each other
         (undirected, same axis convention as morphology_corrections).

    When a division is detected, the ``division_parent_trackid`` column is set
    to the parent's trackid for both daughter rows throughout their lifetimes.
    Rows not involved in a division keep NaN.

    Args:
        fl_measurements: DataFrame with trackid, frame, centroid_0, centroid_1,
            major_axis_length, orientation columns.  Rows with trackid == -1
            (untracked) are ignored.
        centroid_dist_frac: Distance threshold multiplier (× median major axis).
        angle_deg: Maximum orientation difference between daughters.

    Returns:
        Tuple of (updated DataFrame, counts dict with key n_division_events).
    """
    fl = fl_measurements.copy()
    fl["division_parent_trackid"] = np.nan

    tracked = fl[fl["trackid"] != -1].copy()
    if tracked.empty or "trackid" not in fl.columns:
        return fl, {"n_division_events": 0}

    median_major = tracked["major_axis_length"].median()
    dist_threshold = centroid_dist_frac * median_major
    angle_threshold = np.deg2rad(angle_deg)

    frames = sorted(tracked["frame"].unique())
    n_events = 0

    for i, frame_t in enumerate(frames[:-1]):
        frame_t1 = frames[i + 1]

        ids_t = set(tracked.loc[tracked["frame"] == frame_t, "trackid"])
        ids_t1 = set(tracked.loc[tracked["frame"] == frame_t1, "trackid"])

        ended = ids_t - ids_t1        # tracks present in t but gone in t+1
        new_t1 = ids_t1 - ids_t      # tracks appearing fresh in t+1

        if not ended or len(new_t1) < 2:
            continue

        new_rows = tracked[
            (tracked["frame"] == frame_t1) & (tracked["trackid"].isin(new_t1))
        ]
        new_centroids = new_rows[["centroid_0", "centroid_1"]].values
        new_ids = new_rows["trackid"].values
        new_orientations = new_rows["orientation"].values

        for parent_id in ended:
            parent_row = tracked[
                (tracked["frame"] == frame_t) & (tracked["trackid"] == parent_id)
            ]
            if parent_row.empty:
                continue
            p_cx = float(parent_row["centroid_0"].iloc[0])
            p_cy = float(parent_row["centroid_1"].iloc[0])

            # Distance from parent last position to each new track
            dists = np.sqrt((new_centroids[:, 0] - p_cx) ** 2 + (new_centroids[:, 1] - p_cy) ** 2)
            close_mask = dists < dist_threshold
            close_idx = np.where(close_mask)[0]

            if len(close_idx) < 2:
                continue

            # Among close candidates, find a pair with similar orientation
            found = False
            for a in range(len(close_idx)):
                for b in range(a + 1, len(close_idx)):
                    ia, ib = close_idx[a], close_idx[b]
                    diff = abs(new_orientations[ia] - new_orientations[ib])
                    angle_diff = min(diff, np.pi - diff)
                    if angle_diff < angle_threshold:
                        d1_id = int(new_ids[ia])
                        d2_id = int(new_ids[ib])
                        fl.loc[fl["trackid"] == d1_id, "division_parent_trackid"] = parent_id
                        fl.loc[fl["trackid"] == d2_id, "division_parent_trackid"] = parent_id
                        n_events += 1
                        found = True
                        break
                if found:
                    break

    return fl, {"n_division_events": n_events}


# ---------------------------------------------------------------------------
# Lysis event detection
# ---------------------------------------------------------------------------

def detect_lysis_events(fl_measurements: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Flag tracks that transition from single-cell to lyse.

    For each tracked cell (trackid != -1), finds the first frame where
    object_class transitions to 'lyse' from a prior 'single-cell' label.
    Adds column ``lysis_event_frame`` (int or NaN).

    Args:
        fl_measurements: DataFrame with trackid, frame, object_class.

    Returns:
        Tuple of (updated DataFrame, counts dict with key n_lysis_events).
    """
    fl = fl_measurements.copy()
    fl["lysis_event_frame"] = np.nan

    if "lyse" not in fl.get("object_class", pd.Series(dtype=str)).values:
        return fl, {"n_lysis_events": 0}

    tracked = fl[fl["trackid"] != -1]
    if tracked.empty:
        return fl, {"n_lysis_events": 0}

    n_events = 0
    for track_id, group in tracked.groupby("trackid"):
        group_sorted = group.sort_values("frame")
        classes = group_sorted["object_class"].values
        frames = group_sorted["frame"].values

        had_single_cell = False
        for cls, frm in zip(classes, frames):
            if cls == "single-cell":
                had_single_cell = True
            elif cls == "lyse" and had_single_cell:
                fl.loc[
                    (fl["trackid"] == track_id) & (fl["frame"] >= frm),
                    "lysis_event_frame",
                ] = frm
                n_events += 1
                break

    return fl, {"n_lysis_events": n_events}


# ---------------------------------------------------------------------------
# Filamentation trajectory detection
# ---------------------------------------------------------------------------

def detect_filamentation_events(
    fl_measurements: pd.DataFrame,
    min_consecutive_frames: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Flag tracks where the 'long' class persists for ≥ min_consecutive_frames.

    Adds column ``filamentation_onset_frame`` (int or NaN): the first frame of
    the longest run of consecutive 'long' labels in the track, provided it
    meets the minimum length.

    Args:
        fl_measurements: DataFrame with trackid, frame, object_class.
        min_consecutive_frames: Minimum consecutive 'long' frames to qualify.

    Returns:
        Tuple of (updated DataFrame, counts dict with key n_filamentation_events).
    """
    fl = fl_measurements.copy()
    fl["filamentation_onset_frame"] = np.nan

    if "long" not in fl.get("object_class", pd.Series(dtype=str)).values:
        return fl, {"n_filamentation_events": 0}

    tracked = fl[fl["trackid"] != -1]
    if tracked.empty:
        return fl, {"n_filamentation_events": 0}

    n_events = 0
    for track_id, group in tracked.groupby("trackid"):
        group_sorted = group.sort_values("frame")
        classes = group_sorted["object_class"].values
        frames = group_sorted["frame"].values

        # Find runs of consecutive 'long' labels
        best_onset = None
        best_length = 0
        run_start = None
        run_length = 0

        for cls, frm in zip(classes, frames):
            if cls == "long":
                if run_start is None:
                    run_start = frm
                run_length += 1
                if run_length > best_length:
                    best_length = run_length
                    best_onset = run_start
            else:
                run_start = None
                run_length = 0

        if best_length >= min_consecutive_frames and best_onset is not None:
            fl.loc[fl["trackid"] == track_id, "filamentation_onset_frame"] = best_onset
            n_events += 1

    return fl, {"n_filamentation_events": n_events}


# ---------------------------------------------------------------------------
# FL intensity trajectory features
# ---------------------------------------------------------------------------

def compute_fl_trajectory_features(
    fl_measurements: pd.DataFrame,
    smoothing_window: int = 3,
) -> pd.DataFrame:
    """Per-track FL intensity summary statistics and smoothed trajectory.

    Adds the following columns to fl_measurements:
      fl_track_mean         — mean rel_mean_intensity over the track lifetime
      fl_track_slope        — linear slope of rel_mean_intensity vs frame (per track)
      fl_pipos_frame        — first frame where pi_class == 'piPOS' (NaN if never)
      rel_mean_intensity_smooth — rolling mean of rel_mean_intensity (window=smoothing_window)

    Untracked rows (trackid == -1) keep NaN for all summary columns; the
    smoothed column falls back to a global rolling mean.

    Args:
        fl_measurements: DataFrame with trackid, frame, rel_mean_intensity,
            pi_class columns.
        smoothing_window: Rolling window size for trajectory smoothing.

    Returns:
        Updated DataFrame with new columns.
    """
    fl = fl_measurements.copy()
    fl["fl_track_mean"] = np.nan
    fl["fl_track_slope"] = np.nan
    fl["fl_pipos_frame"] = np.nan
    fl["rel_mean_intensity_smooth"] = np.nan

    if "rel_mean_intensity" not in fl.columns:
        return fl

    tracked = fl[fl["trackid"] != -1]
    if tracked.empty:
        return fl

    for track_id, group in tracked.groupby("trackid"):
        idx = group.index
        group_sorted = group.sort_values("frame")
        frames = group_sorted["frame"].values.astype(float)
        intensities = group_sorted["rel_mean_intensity"].values.astype(float)

        # Summary stats
        track_mean = float(np.nanmean(intensities))
        if len(frames) >= 2:
            valid = ~np.isnan(intensities)
            if valid.sum() >= 2:
                slope, *_ = linregress(frames[valid], intensities[valid])
            else:
                slope = np.nan
        else:
            slope = np.nan

        fl.loc[idx, "fl_track_mean"] = track_mean
        fl.loc[idx, "fl_track_slope"] = slope

        # First piPOS frame
        if "pi_class" in fl.columns:
            pipos_rows = group_sorted[group_sorted["pi_class"] == "piPOS"]
            if not pipos_rows.empty:
                fl.loc[idx, "fl_pipos_frame"] = int(pipos_rows["frame"].min())

        # Smoothed trajectory (rolling mean within track, sorted by frame)
        smooth = (
            group_sorted["rel_mean_intensity"]
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
        )
        fl.loc[group_sorted.index, "rel_mean_intensity_smooth"] = smooth.values

    return fl
