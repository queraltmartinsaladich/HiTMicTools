"""
Hungarian tracker v2 with gap bridging and appearance-cost support.

Compared to v1 (single-frame-back linking only):
  - Adds `gap_bridge_frames` parameter. A track whose last detection was up to
    N frames ago is still eligible for linking at the current frame. Once a
    track goes silent for longer than N, it is retired.
  - Preserves v1's class-agnostic behavior: cost matrix is built over all
    detections in scope, regardless of object_class.
  - Preserves v1's piPOS lock-in API (no changes to `apply_pipos_lockin`).
  - API-compatible with v1: `track_objects(measurements_df, ...)` returns the
    same DataFrame with `trackid` column added.

Compared to v2 (centroid-only cost):
  - Adds `feature_weights` parameter. When non-empty, appearance features
    (e.g. area, solidity) are incorporated into the cost matrix as normalised
    relative differences, scaled to pixel-cost units. This penalises mislinks
    between cells with very different morphologies (e.g. single-cell → clump).
  - `max_distance` now gates *total* cost (centroid + appearance), not just
    Euclidean distance.  For typical cells the appearance penalty is small
    (<1 px); it only materially affects cases where the morphology diverges.
  - Default weights (DEFAULT_WEIGHTS) penalise area (0.3) and solidity (0.2).
    Pass `feature_weights={}` to revert to centroid-only behaviour.

Rationale for appearance cost: on dense well images, cells near a clump are
sometimes mislinked into the clump between frames because the clump centroid
happens to be slightly closer.  Adding a relative area penalty of 0.3 adds
~6 px of virtual distance for a 3× area mismatch, typically breaking the
mislink without affecting normal same-size cell links.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class HungarianTracker:
    """Frame-to-frame optimal assignment tracker with appearance cost,
    piPOS lock-in, and gap bridging."""

    DEFAULT_WEIGHTS: Dict[str, float] = {"area": 0.3, "solidity": 0.2}

    def __init__(
        self,
        max_distance: float = 25.0,
        gap_bridge_frames: int = 2,
        feature_weights: Optional[Dict[str, float]] = None,
        max_distance_um: Optional[float] = None,
    ):
        """
        Args:
            max_distance: Maximum total linking cost in *pixels*. Used when
                ``max_distance_um`` is not set or pixel_size is unavailable.
            gap_bridge_frames: Consecutive missed frames tolerated before a
                track is retired.  0 = no bridging (v1 behaviour).
            feature_weights: Mapping of feature column name → weight.  Each
                non-zero entry adds a normalised relative-difference penalty
                (scaled to pixel units by max_distance) to the cost matrix.
                ``None`` uses DEFAULT_WEIGHTS.  Pass ``{}`` for centroid-only.
            max_distance_um: Maximum linking distance in *microns*.  When set
                and ``pixel_size`` is passed to ``track_objects``, this takes
                priority over ``max_distance`` and is converted to pixels at
                run time.  Preferred over the pixel-based threshold because it
                is objective/camera-independent (e.g. 2.0 µm suits most
                rod-shaped bacteria at 1–5 min frame intervals).
        """
        self.max_distance = max_distance
        self.max_distance_um = max_distance_um
        self.gap_bridge_frames = gap_bridge_frames
        self.feature_weights: Dict[str, float] = (
            dict(self.DEFAULT_WEIGHTS) if feature_weights is None else dict(feature_weights)
        )
        self.features: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_features(self, features: List[str]) -> None:
        """Store feature list (CellTracker API compatibility).

        For HungarianTracker the appearance cost is driven by
        ``feature_weights``; ``set_features`` just records the list for
        external inspection and future use.
        """
        self.features = list(features)

    def track_objects(
        self,
        measurements_df: pd.DataFrame,
        volume_bounds: Optional[Tuple[int, int]] = None,
        logger: Optional[logging.Logger] = None,
        pixel_size: Optional[float] = None,
        cost_overrides: Optional[Dict[int, Tuple[np.ndarray, List[int], List[int]]]] = None,
        learned_max_cost: float = 0.5,
    ) -> pd.DataFrame:
        """Assign persistent track IDs across frames.

        Args:
            measurements_df: DataFrame with columns: frame, centroid_0,
                centroid_1.  Any columns listed in ``feature_weights`` that
                are present are used for appearance cost.
            volume_bounds: Ignored (API compatibility with CellTracker).
            logger: Optional logger instance.
            pixel_size: Physical pixel size in µm/px (from image metadata).
                When provided alongside ``max_distance_um``, the µm threshold
                is converted to pixels and used instead of ``max_distance``.

        Returns:
            Input DataFrame with ``trackid`` column added (int32; -1 for
            unlinked detections).
        """
        # Resolve effective max_distance (pixels) for this image
        if self.max_distance_um is not None and pixel_size is not None and pixel_size > 0:
            effective_max_dist = self.max_distance_um / pixel_size
        else:
            effective_max_dist = self.max_distance

        df = measurements_df.copy()
        frames = sorted(df["frame"].unique())
        has_label_col = "label" in df.columns

        if len(frames) == 0:
            df["trackid"] = np.int32(-1)
            return df

        # Appearance features available in this df
        active_feats = [
            f for f, w in self.feature_weights.items()
            if w > 0 and f in df.columns
        ]
        feat_weights_arr = (
            np.array([self.feature_weights[f] for f in active_feats], dtype=float)
            if active_feats else None
        )

        if active_feats:
            _all_feat_vals = df[active_feats].values.astype(float)
            _idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

        def _feat_vals(idx: int) -> Optional[np.ndarray]:
            if not active_feats:
                return None
            return _all_feat_vals[_idx_to_pos[idx]]

        track_ids = pd.Series(np.int32(-1), index=df.index)
        next_track_id = 0
        # active_tracks[tid] = {
        #   "centroid": (c0, c1),
        #   "last_frame": int,
        #   "feat_vals": ndarray | None,
        # }
        active_tracks: dict = {}

        # ── Seed first frame ──────────────────────────────────────────
        first_mask = df["frame"] == frames[0]
        first_indices = df.index[first_mask]
        for idx in first_indices:
            c0, c1 = float(df.at[idx, "centroid_0"]), float(df.at[idx, "centroid_1"])
            track_ids[idx] = next_track_id
            active_tracks[next_track_id] = {
                "centroid": (c0, c1),
                "last_frame": frames[0],
                "feat_vals": _feat_vals(idx),
                "label": int(df.at[idx, "label"]) if has_label_col else None,
            }
            next_track_id += 1

        total_linked = 0
        total_new = len(first_indices)
        total_bridged = 0
        euclidean_dists: List[float] = []
        max_dist_used = 0.0

        # ── Link subsequent frames ────────────────────────────────────
        for fi in range(1, len(frames)):
            curr_frame = frames[fi]
            curr_mask = df["frame"] == curr_frame
            curr_indices = df.index[curr_mask]

            if len(curr_indices) == 0:
                continue

            eligible_tids = [
                tid for tid, info in active_tracks.items()
                if (curr_frame - info["last_frame"]) <= self.gap_bridge_frames
            ]

            if not eligible_tids:
                for idx in curr_indices:
                    c0, c1 = float(df.at[idx, "centroid_0"]), float(df.at[idx, "centroid_1"])
                    track_ids[idx] = next_track_id
                    active_tracks[next_track_id] = {
                        "centroid": (c0, c1),
                        "last_frame": curr_frame,
                        "feat_vals": _feat_vals(idx),
                        "label": int(df.at[idx, "label"]) if has_label_col else None,
                    }
                    next_track_id += 1
                total_new += len(curr_indices)
                continue

            prev_centroids = np.array(
                [active_tracks[tid]["centroid"] for tid in eligible_tids]
            )
            curr_centroids = df.loc[
                curr_indices, ["centroid_0", "centroid_1"]
            ].values

            euclid = cdist(prev_centroids, curr_centroids, metric="euclidean")

            if cost_overrides is not None and frames[fi - 1] in cost_overrides:
                ov_matrix, ov_labels_t, ov_labels_t1 = cost_overrides[frames[fi - 1]]
                row_map = {lbl: i for i, lbl in enumerate(ov_labels_t)}
                col_map = {lbl: j for j, lbl in enumerate(ov_labels_t1)}
                scale = effective_max_dist / max(learned_max_cost, 1e-8)
                reject = effective_max_dist * 1e6
                curr_labels = [
                    int(df.at[idx, "label"]) if has_label_col else None
                    for idx in curr_indices
                ]
                cost = euclid.copy()
                for i, tid in enumerate(eligible_tids):
                    if active_tracks[tid]["last_frame"] != frames[fi - 1]:
                        continue  # gap-bridged: keep Euclidean fallback
                    label_i = active_tracks[tid].get("label")
                    if label_i is None or label_i not in row_map:
                        continue
                    ri = row_map[label_i]
                    for j, label_j in enumerate(curr_labels):
                        if label_j is None or label_j not in col_map:
                            continue
                        cj = col_map[label_j]
                        raw = float(ov_matrix[ri, cj])
                        cost[i, j] = raw * scale if raw < learned_max_cost else reject
            elif active_feats:
                _nan_fill = np.full(len(active_feats), np.nan)
                prev_feats = np.array([
                    active_tracks[tid]["feat_vals"]
                    if active_tracks[tid]["feat_vals"] is not None
                    else _nan_fill
                    for tid in eligible_tids
                ])
                curr_feats = df.loc[curr_indices, active_feats].values.astype(float)
                cost = self._build_cost_matrix(
                    euclid, prev_feats, curr_feats, feat_weights_arr, effective_max_dist
                )
            else:
                cost = euclid

            row_ind, col_ind = linear_sum_assignment(cost)

            linked_curr: set = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= effective_max_dist:
                    tid = eligible_tids[r]
                    curr_idx = curr_indices[c]
                    c0, c1 = (
                        float(df.at[curr_idx, "centroid_0"]),
                        float(df.at[curr_idx, "centroid_1"]),
                    )
                    track_ids[curr_idx] = tid
                    gap = curr_frame - active_tracks[tid]["last_frame"]
                    if gap >= 2:
                        total_bridged += 1
                    active_tracks[tid] = {
                        "centroid": (c0, c1),
                        "last_frame": curr_frame,
                        "feat_vals": _feat_vals(curr_idx),
                        "label": int(df.at[curr_idx, "label"]) if has_label_col else None,
                    }
                    linked_curr.add(c)
                    total_linked += 1
                    ed = float(euclid[r, c])
                    euclidean_dists.append(ed)
                    if ed > max_dist_used:
                        max_dist_used = ed

            # Unmatched detections in current frame become new tracks
            for j, idx in enumerate(curr_indices):
                if j not in linked_curr:
                    c0, c1 = (
                        float(df.at[idx, "centroid_0"]),
                        float(df.at[idx, "centroid_1"]),
                    )
                    track_ids[idx] = next_track_id
                    active_tracks[next_track_id] = {
                        "centroid": (c0, c1),
                        "last_frame": curr_frame,
                        "feat_vals": _feat_vals(idx),
                        "label": int(df.at[idx, "label"]) if has_label_col else None,
                    }
                    next_track_id += 1
                    total_new += 1

            # Retire tracks silent beyond the bridge window
            cutoff = curr_frame - self.gap_bridge_frames
            active_tracks = {
                tid: info for tid, info in active_tracks.items()
                if info["last_frame"] >= cutoff
            }

        df["trackid"] = track_ids.astype(np.int32)

        if logger:
            n_tracks = int(df["trackid"].nunique())
            n_objects = len(df)
            n_frames = len(frames)
            track_lengths = df.groupby("trackid")["frame"].nunique()
            full_length_tracks = int((track_lengths == n_frames).sum())
            short_tracks = int((track_lengths == 1).sum())
            median_length = float(track_lengths.median())
            mean_dist = float(np.mean(euclidean_dists)) if euclidean_dists else 0.0
            p95_dist = float(np.percentile(euclidean_dists, 95)) if euclidean_dists else 0.0
            feat_info = (
                f"appearance features: {active_feats} "
                f"(weights: {[self.feature_weights[f] for f in active_feats]})"
                if active_feats
                else "centroid-only (feature_weights empty)"
            )
            dist_config = (
                f"max_distance={self.max_distance_um} µm → {effective_max_dist:.1f} px "
                f"(pixel_size={pixel_size} µm/px)"
                if self.max_distance_um is not None and pixel_size is not None
                else f"max_distance={effective_max_dist:.1f} px (pixel-space)"
            )
            logger.info(
                f"Hungarian tracking summary:\n"
                f"  Config: {dist_config}, "
                f"gap_bridge_frames={self.gap_bridge_frames}, {feat_info}\n"
                f"  Frames: {n_frames}, Detections: {n_objects}\n"
                f"  Tracks: {n_tracks} total, {full_length_tracks} full-length "
                f"({n_frames}f), {short_tracks} single-frame\n"
                f"  Links: {total_linked} total, {total_bridged} via gap bridging "
                f"({100.0 * total_bridged / max(total_linked, 1):.1f}%), "
                f"{total_new} new tracks\n"
                f"  Centroid distances: mean={mean_dist:.1f}px, "
                f"p95={p95_dist:.1f}px, max={max_dist_used:.1f}px "
                f"(total-cost cutoff={self.max_distance}px)"
            )

        return df

    # ------------------------------------------------------------------
    # piPOS lock-in (unchanged from v1)
    # ------------------------------------------------------------------

    def apply_pipos_lockin(
        self,
        measurements_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None,
    ) -> pd.DataFrame:
        """Enforce piPOS lock-in: once a track is piPOS, all subsequent
        frames stay piPOS."""
        if (
            "trackid" not in measurements_df.columns
            or "pi_class" not in measurements_df.columns
        ):
            if logger:
                logger.warning("piPOS lock-in skipped: missing trackid or pi_class column")
            return measurements_df

        override_count = 0
        tracks_with_lockin = 0
        tracked = measurements_df["trackid"] != -1
        n_tracked_cells = int(tracked.sum())
        n_untracked = int((~tracked).sum())
        pipos_before = int((measurements_df["pi_class"] == "piPOS").sum())

        for tid, group in measurements_df.loc[tracked].groupby("trackid"):
            pipos_frames = group.loc[group["pi_class"] == "piPOS", "frame"]
            if len(pipos_frames) == 0:
                continue
            first_pipos_frame = pipos_frames.min()
            mask = (
                (measurements_df["trackid"] == tid)
                & (measurements_df["frame"] > first_pipos_frame)
                & (measurements_df["pi_class"] != "piPOS")
            )
            n_overrides = int(mask.sum())
            if n_overrides > 0:
                tracks_with_lockin += 1
                override_count += n_overrides
            measurements_df.loc[mask, "pi_class"] = "piPOS"

        pipos_after = int((measurements_df["pi_class"] == "piPOS").sum())

        if logger:
            logger.info(
                f"piPOS lock-in summary:\n"
                f"  Tracked detections: {n_tracked_cells}, Untracked: {n_untracked}\n"
                f"  Tracks with lock-in applied: {tracks_with_lockin}\n"
                f"  Classifications overridden: {override_count}\n"
                f"  piPOS count: {pipos_before} -> {pipos_after} "
                f"(+{pipos_after - pipos_before})"
            )

        return measurements_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cost_matrix(
        euclid: np.ndarray,
        prev_feats: np.ndarray,
        curr_feats: np.ndarray,
        weights: np.ndarray,
        max_distance: float,
    ) -> np.ndarray:
        """Add normalised appearance cost to an Euclidean distance matrix.

        Appearance cost for feature f between track i and detection j:
            |prev_f_i - curr_f_j| / max(|prev_f_i|, |curr_f_j|, ε)
        This is scale-invariant and lies in [0, 1] for positive features.
        The weighted sum is multiplied by ``max_distance`` to put it in the
        same pixel units as the Euclidean component.

        Example: area weight=0.3, max_distance=25 px.  A 3× area mismatch
        (single-cell vs clump) has normalised diff ≈ 0.67, contributing
        0.67 × 0.3 × 25 ≈ 5 px of virtual distance.

        NaN feature values (missing / not observed yet) contribute 0 cost,
        so missing features never block a valid spatial link.

        Args:
            euclid:       (n_prev, n_curr) Euclidean distance matrix.
            prev_feats:   (n_prev, n_feats) last-known feature values per track.
            curr_feats:   (n_curr, n_feats) feature values in current frame.
            weights:      (n_feats,) per-feature weights.
            max_distance: Pixel scale for appearance cost (= tracker threshold).

        Returns:
            (n_prev, n_curr) total cost matrix (Euclidean + appearance).
        """
        cost = euclid.copy()
        p = prev_feats[:, np.newaxis, :]    # (n_prev, 1, n_feats)
        c = curr_feats[np.newaxis, :, :]    # (1, n_curr, n_feats)
        denom = np.maximum(np.maximum(np.abs(p), np.abs(c)), 1e-6)
        norm_diff = np.abs(p - c) / denom   # (n_prev, n_curr, n_feats) ∈ [0, 1]
        # nansum: NaN entries contribute 0, never block a link
        appearance = np.nansum(norm_diff * weights, axis=2)  # (n_prev, n_curr)
        cost += appearance * max_distance
        return cost
