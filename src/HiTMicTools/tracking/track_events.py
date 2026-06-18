"""Post-tracking event detection and FL trajectory analysis.

Called after tracking (step 4.5) and PI classification / piPOS lock-in are
complete, so all columns (trackid, pi_class, object_class, rel_mean_intensity)
are final before any of these functions run.

Public API
----------
refine_tracks              — short-track filter, class smoothing, area plausibility
detect_division_events     — temporal parent→daughter split detection (requires ended tracks)
reconcile_lineage          — retrospective area-halving detector for missed divisions
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
# Internal helpers
# ---------------------------------------------------------------------------

def _majority_vote_smooth(classes: np.ndarray, window: int) -> np.ndarray:
    """Per-position majority vote over a sliding window of categorical labels."""
    half = window // 2
    result = classes.copy()
    n = len(classes)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_vals = classes[start:end]
        counts: dict = {}
        for v in window_vals:
            counts[v] = counts.get(v, 0) + 1
        result[i] = max(counts, key=counts.get)
    return result


# ---------------------------------------------------------------------------
# Track quality refinement
# ---------------------------------------------------------------------------

def refine_tracks(
    fl_measurements: pd.DataFrame,
    min_track_frames: int = 3,
    class_smooth_window: int = 3,
    area_jump_frac: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Post-tracking quality refinement of fl_measurements.

    Three corrective passes over tracked cells:

    1. Short track filter: tracks shorter than min_track_frames are flagged
       as 'short_track' — likely false detections, cells entering/exiting
       the field of view mid-experiment, or segmentation fragments.

    2. Object-class smoothing: majority-vote filter (window=class_smooth_window)
       removes single-frame label flips such as single-cell→debris→single-cell.
       Original labels are preserved in object_class_raw.  Rows whose label
       changed are flagged as 'class_flicker'.

    3. Area plausibility: rows where |Δarea| / area_prev > area_jump_frac are
       flagged as 'bad_frame' — likely a segmentation split/merge error.

    Adds columns:
        object_class_raw  — original object_class before smoothing
        track_quality     — 'ok' | 'short_track' | 'bad_frame' | 'class_flicker'
                            Priority: short_track > bad_frame > class_flicker > ok.
                            Untracked rows (trackid == -1) receive 'untracked'.

    Short tracks skip smoothing and area checks.  'bad_frame' only upgrades
    rows currently 'ok' — it does not override 'short_track'.

    Args:
        fl_measurements: DataFrame with trackid, frame, object_class, area columns.
        min_track_frames: Minimum frames for a valid track.  Default 3.
        class_smooth_window: Window size for majority-vote smoothing.  Default 3.
        area_jump_frac: Fractional area-change threshold for bad_frame.  Default 0.5.

    Returns:
        Tuple of (updated DataFrame, counts dict).
        Counts keys: short_tracks (number of tracks), bad_frame_rows, class_flicker_rows.
    """
    fl = fl_measurements.copy()
    fl["object_class_raw"] = fl.get("object_class", pd.Series(dtype=str))
    fl["track_quality"] = "ok"
    fl.loc[fl["trackid"] == -1, "track_quality"] = "untracked"

    tracked_mask = fl["trackid"] != -1
    if not tracked_mask.any():
        return fl, {"short_tracks": 0, "bad_frame_rows": 0, "class_flicker_rows": 0}

    has_class = "object_class" in fl.columns
    has_area = "area" in fl.columns
    n_short_tracks = 0
    n_bad_frame_rows = 0
    n_class_flicker_rows = 0

    for track_id, group in fl[tracked_mask].groupby("trackid"):
        idx = group.index
        group_sorted = group.sort_values("frame")
        sorted_idx = group_sorted.index
        n_frames = len(group_sorted)

        # 1. Short track filter
        if n_frames < min_track_frames:
            fl.loc[idx, "track_quality"] = "short_track"
            n_short_tracks += 1
            continue

        # 2. Object-class smoothing (majority vote)
        if has_class:
            original = group_sorted["object_class"].values.copy()
            smoothed = _majority_vote_smooth(original, class_smooth_window)
            changed = smoothed != original
            if changed.any():
                fl.loc[sorted_idx[changed], "track_quality"] = "class_flicker"
                fl.loc[sorted_idx, "object_class"] = smoothed
                n_class_flicker_rows += int(changed.sum())

        # 3. Area plausibility
        if has_area:
            areas = group_sorted["area"].values.astype(float)
            prev_areas = np.concatenate([[np.nan], areas[:-1]])
            with np.errstate(invalid="ignore", divide="ignore"):
                frac_change = np.abs(areas - prev_areas) / np.maximum(prev_areas, 1.0)
            bad = np.isfinite(frac_change) & (frac_change > area_jump_frac)
            if bad.any():
                bad_idx = sorted_idx[bad]
                currently_ok = fl.loc[bad_idx, "track_quality"].values == "ok"
                fl.loc[bad_idx[currently_ok], "track_quality"] = "bad_frame"
                n_bad_frame_rows += int(bad.sum())

    return fl, {
        "short_tracks": n_short_tracks,
        "bad_frame_rows": n_bad_frame_rows,
        "class_flicker_rows": n_class_flicker_rows,
    }


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

    ghost_mask = fl.get("object_class", pd.Series(dtype=str)) == "ghost"
    tracked = fl[(fl["trackid"] != -1) & ~ghost_mask].copy()
    if tracked.empty or "trackid" not in fl.columns:
        return fl, {"n_division_events": 0}
    if "major_axis_length" not in tracked.columns:
        raise KeyError("detect_division_events requires 'major_axis_length' column — run feature extraction first")

    median_major = tracked["major_axis_length"].median()
    dist_threshold = centroid_dist_frac * median_major
    angle_threshold = np.deg2rad(angle_deg)

    frames = sorted(tracked["frame"].unique())
    if len(frames) > 1 and frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError(
            "detect_division_events requires consecutive frame numbers — "
            f"gaps detected in frames: {frames}"
        )
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
# Retrospective lineage reconciler
# ---------------------------------------------------------------------------

def reconcile_lineage(
    fl_measurements: pd.DataFrame,
    area_drop_frac: float = 0.40,
    area_conservation_tol: float = 0.30,
    centroid_dist_frac: float = 2.0,
    valid_classes: tuple = ("single-cell", "long"),
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Fill in division events missed by detect_division_events.

    The Hungarian tracker typically assigns one daughter to the parent track,
    leaving ``ended`` empty so ``detect_division_events`` never fires.  This
    function recovers those events by looking for:

      - A *continued* track X (present at both frame t and t+1) whose area
        drops by at least ``area_drop_frac`` between the two frames.
      - A *new* track Y (first appearing at t+1) whose centroid at t+1 is
        within ``centroid_dist_frac × major_axis_length`` of X's centroid at t,
        and whose area satisfies:
        ``|area(X,t+1) + area(Y,t+1) − area(X,t)| / area(X,t) ≤ area_conservation_tol``

    When a pair is found, sets ``division_parent_trackid[Y] = X.trackid``.
    Entries already set by ``detect_division_events`` are not overwritten.
    Each continued track X and each new track Y can be matched at most once
    per frame; ties are broken by centroid distance (closest first).

    Must be called after ``detect_division_events`` so that
    ``division_parent_trackid`` already exists.

    Args:
        fl_measurements: DataFrame with trackid, frame, area, centroid_0,
            centroid_1, major_axis_length, object_class columns.
        area_drop_frac: Minimum fractional area loss by the continued track
            at the division frame.  Default 0.40.
        area_conservation_tol: Maximum fractional deviation of
            ``area(X,t+1) + area(Y,t+1)`` from ``area(X,t)``.  Default 0.30.
        centroid_dist_frac: Distance threshold as a multiple of the parent's
            ``major_axis_length``.  Default 2.0.
        valid_classes: ``object_class`` values that are eligible to divide.

    Returns:
        Tuple of (updated DataFrame, counts dict with key n_reconciled_divisions).
    """
    fl = fl_measurements.copy()
    if "division_parent_trackid" not in fl.columns:
        fl["division_parent_trackid"] = np.nan

    ghost_mask = fl.get("object_class", pd.Series(dtype=str)) == "ghost"
    tracked = fl[(fl["trackid"] != -1) & ~ghost_mask].copy()

    required = {"area", "centroid_0", "centroid_1", "major_axis_length"}
    if tracked.empty or not required.issubset(tracked.columns):
        return fl, {"n_reconciled_divisions": 0}

    frames = sorted(tracked["frame"].unique())
    if len(frames) > 1 and frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError(
            "reconcile_lineage requires consecutive frame numbers — "
            f"gaps detected in frames: {frames}"
        )
    n_reconciled = 0

    for i, frame_t in enumerate(frames[:-1]):
        frame_t1 = frames[i + 1]

        rows_t  = tracked[tracked["frame"] == frame_t]
        rows_t1 = tracked[tracked["frame"] == frame_t1]

        ids_t  = set(rows_t["trackid"])
        ids_t1 = set(rows_t1["trackid"])

        continued_ids = ids_t & ids_t1
        new_ids       = ids_t1 - ids_t

        if not new_ids or not continued_ids:
            continue

        cont_t  = rows_t[rows_t["trackid"].isin(continued_ids)].set_index("trackid")
        cont_t1 = rows_t1[rows_t1["trackid"].isin(continued_ids)].set_index("trackid")

        # Continued tracks that halved in area and had a dividable class at t
        halved_ids = []
        for xid in continued_ids:
            if xid not in cont_t.index or xid not in cont_t1.index:
                continue
            if cont_t.at[xid, "object_class"] not in valid_classes:
                continue
            a_t  = float(cont_t.at[xid, "area"])
            a_t1 = float(cont_t1.at[xid, "area"])
            if a_t < 1 or np.isnan(a_t) or np.isnan(a_t1):
                continue
            if a_t1 / a_t < (1.0 - area_drop_frac):
                halved_ids.append(xid)

        if not halved_ids:
            continue

        new_rows = rows_t1[rows_t1["trackid"].isin(new_ids)]

        # Build candidate (dist, new_df_idx, new_trackid, parent_trackid) list
        candidate_pairs = []
        for _, new_row in new_rows.iterrows():
            if pd.notna(fl.at[new_row.name, "division_parent_trackid"]):
                continue  # already assigned by detect_division_events
            y_class = str(new_row.get("object_class", ""))
            if y_class not in valid_classes:
                continue
            y_cx   = float(new_row["centroid_0"])
            y_cy   = float(new_row["centroid_1"])
            y_area = float(new_row["area"])
            y_tid  = int(new_row["trackid"])

            for xid in halved_ids:
                x_cx    = float(cont_t.at[xid, "centroid_0"])
                x_cy    = float(cont_t.at[xid, "centroid_1"])
                x_major = max(float(cont_t.at[xid, "major_axis_length"]), 1.0)
                a_t     = float(cont_t.at[xid, "area"])
                a_t1    = float(cont_t1.at[xid, "area"])

                dist = np.sqrt((y_cx - x_cx) ** 2 + (y_cy - x_cy) ** 2)
                if dist > centroid_dist_frac * x_major:
                    continue

                conservation_err = abs((a_t1 + y_area) - a_t) / max(a_t, 1.0)
                if conservation_err > area_conservation_tol:
                    continue

                candidate_pairs.append((dist, new_row.name, y_tid, xid))

        # Assign greedily: closest first, no parent or daughter double-claimed
        candidate_pairs.sort(key=lambda x: x[0])
        claimed_parents: set = set()
        assigned_new: set    = set()

        for dist, y_idx, y_tid, xid in candidate_pairs:
            if xid in claimed_parents or y_tid in assigned_new:
                continue
            fl.loc[fl["trackid"] == y_tid, "division_parent_trackid"] = float(xid)
            claimed_parents.add(xid)
            assigned_new.add(y_tid)
            n_reconciled += 1

    return fl, {"n_reconciled_divisions": n_reconciled}


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

    tracked = fl[(fl["trackid"] != -1) & (fl["object_class"] != "ghost")]
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

    tracked = fl[(fl["trackid"] != -1) & (fl["object_class"] != "ghost")]
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
    pre_event_window: int = 5,
) -> pd.DataFrame:
    """Per-track FL intensity and morphology trajectory features.

    Adds the following columns to fl_measurements:
      n_frames_tracked          — frames the cell was tracked
      fl_track_mean             — mean rel_mean_intensity over track lifetime
      fl_track_std              — std of rel_mean_intensity over track lifetime
      fl_track_slope            — linear slope of rel_mean_intensity vs frame
      fl_pre_event_slope        — slope over pre_event_window frames before lysis_event_frame
      fl_pipos_frame            — first frame where pi_class == 'piPOS' (NaN if never)
      delta_fl                  — frame-to-frame change in rel_mean_intensity (NaN at track start)
      rel_mean_intensity_smooth — rolling mean of rel_mean_intensity (window=smoothing_window)
      aspect_ratio_slope        — linear slope of aspect_ratio vs frame (NaN if absent)

    n_frames_tracked and aspect_ratio_slope are computed even when
    rel_mean_intensity is absent.  All intensity columns require it.
    Untracked rows (trackid == -1) keep NaN for all columns.

    Args:
        fl_measurements: DataFrame with trackid, frame, and optionally
            rel_mean_intensity, pi_class, aspect_ratio, lysis_event_frame columns.
        smoothing_window: Rolling window for trajectory smoothing.  Default 3.
        pre_event_window: Frames before lysis_event_frame for fl_pre_event_slope.  Default 5.

    Returns:
        Updated DataFrame with new columns.
    """
    fl = fl_measurements.copy()
    for col in (
        "n_frames_tracked", "fl_track_mean", "fl_track_std", "fl_track_slope",
        "fl_pre_event_slope", "fl_pipos_frame", "delta_fl",
        "rel_mean_intensity_smooth", "aspect_ratio_slope",
    ):
        fl[col] = np.nan

    tracked = fl[fl["trackid"] != -1]
    if tracked.empty:
        return fl

    has_intensity = "rel_mean_intensity" in fl.columns
    has_ar = "aspect_ratio" in fl.columns
    has_lysis_frame = "lysis_event_frame" in fl.columns
    has_pi_class = "pi_class" in fl.columns

    for track_id, group in tracked.groupby("trackid"):
        idx = group.index
        group_sorted = group.sort_values("frame")
        track_frames = group_sorted["frame"].values.astype(float)
        n = len(track_frames)

        fl.loc[idx, "n_frames_tracked"] = n

        if has_ar:
            ar_vals = group_sorted["aspect_ratio"].values.astype(float)
            ar_valid = ~np.isnan(ar_vals)
            if ar_valid.sum() >= 2:
                ar_slope, *_ = linregress(track_frames[ar_valid], ar_vals[ar_valid])
                fl.loc[idx, "aspect_ratio_slope"] = float(ar_slope)

        if not has_intensity:
            continue

        intensities = group_sorted["rel_mean_intensity"].values.astype(float)
        valid = ~np.isnan(intensities)

        fl.loc[idx, "fl_track_mean"] = float(np.nanmean(intensities))
        fl.loc[idx, "fl_track_std"] = float(np.nanstd(intensities))

        if valid.sum() >= 2:
            slope, *_ = linregress(track_frames[valid], intensities[valid])
            fl.loc[idx, "fl_track_slope"] = float(slope)

        # Frame-to-frame intensity change (NaN at track start)
        deltas = np.concatenate([[np.nan], np.diff(intensities)])
        fl.loc[group_sorted.index, "delta_fl"] = deltas

        if has_pi_class:
            pipos_rows = group_sorted[group_sorted["pi_class"] == "piPOS"]
            if not pipos_rows.empty:
                fl.loc[idx, "fl_pipos_frame"] = int(pipos_rows["frame"].min())

        smooth = (
            group_sorted["rel_mean_intensity"]
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
        )
        fl.loc[group_sorted.index, "rel_mean_intensity_smooth"] = smooth.values

        # Pre-event slope: slope of intensity in the N frames before lysis
        if has_lysis_frame:
            lysis_vals = group_sorted["lysis_event_frame"].dropna()
            if not lysis_vals.empty:
                lysis_f = float(lysis_vals.iloc[0])
                pre_mask = (track_frames >= lysis_f - pre_event_window) & (track_frames < lysis_f)
                if pre_mask.sum() >= 2:
                    pre_valid = valid & pre_mask
                    if pre_valid.sum() >= 2:
                        pre_slope, *_ = linregress(
                            track_frames[pre_valid], intensities[pre_valid]
                        )
                        fl.loc[idx, "fl_pre_event_slope"] = float(pre_slope)

    return fl
