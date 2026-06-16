"""
Hungarian tracker v5.

Compared with the legacy HungarianTracker, this tracker adds multi-frame gap
bridging, re-identification, division handling, and safe same-position
stitching. Defaults are tuned for immobilized cocci in ASCT-style time lapses;
other organisms should set parameters explicitly in the run config.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from HiTMicTools.tracking.hungarian_tracker import HungarianTracker


class HungarianTrackerV5:
    """Biology-constrained Hungarian tracker with re-id, divisions, and stitching."""

    def __init__(
        self,
        max_distance: float = 10.0,
        gap_bridge_frames: int = 5,
        area_weight: float = 0.3,
        shape_weight: float = 0.0,
        detect_divisions: bool = True,
        division_area_ratio_range: Tuple[float, float] = (0.30, 0.70),
        division_search_radius: float = 15.0,
        reid_enabled: bool = True,
        reid_max_gap: int = 16,
        reid_max_distance: float = 10.0,
        reid_area_tol: float = 0.30,
        reid_shape_tol: float = 1.0,
        stitch_max_distance: Optional[float] = 8.0,
    ):
        self.max_distance = max_distance
        self.gap_bridge_frames = gap_bridge_frames
        self.area_weight = area_weight
        self.shape_weight = shape_weight
        self.detect_divisions = detect_divisions
        self.division_area_ratio_range = division_area_ratio_range
        self.division_search_radius = division_search_radius
        self.reid_enabled = reid_enabled
        self.reid_max_gap = reid_max_gap
        self.reid_max_distance = reid_max_distance
        self.reid_area_tol = reid_area_tol
        self.reid_shape_tol = reid_shape_tol
        self.stitch_max_distance = stitch_max_distance

    def set_features(self, features) -> None:
        """No-op for API compatibility with CellTracker."""
        pass

    def track_objects(self, measurements_df, volume_bounds=None, logger=None):
        df = measurements_df.copy()
        if "trackid" in df.columns:
            df = df.drop(columns=["trackid"])
        if "parent_track_id" in df.columns:
            df = df.drop(columns=["parent_track_id"])
        frames = sorted(df["frame"].unique())
        if not frames:
            df["trackid"] = np.int32(-1)
            return df

        has_area = "area" in df.columns
        has_shape = ("major_axis_length" in df.columns) and ("minor_axis_length" in df.columns)

        track_ids = pd.Series(np.int32(-1), index=df.index)
        next_id = 0
        active: Dict[int, dict] = {}
        retired: Dict[int, dict] = {}
        parent_of: Dict[int, int] = {}

        def get(idx):
            c0 = float(df.at[idx, "centroid_0"])
            c1 = float(df.at[idx, "centroid_1"])
            area = float(df.at[idx, "area"]) if has_area else 1.0
            major = float(df.at[idx, "major_axis_length"]) if has_shape else 0.0
            minor = float(df.at[idx, "minor_axis_length"]) if has_shape else 0.0
            return c0, c1, area, major, minor

        for idx in df.index[df["frame"] == frames[0]]:
            c0, c1, area, major, minor = get(idx)
            track_ids[idx] = next_id
            active[next_id] = {
                "centroid": (c0, c1),
                "area": area,
                "major_axis": major,
                "minor_axis": minor,
                "last_frame": frames[0],
            }
            next_id += 1

        n_link = 0
        n_new = len(active)
        n_bridged = 0
        n_reid = 0
        n_div = 0

        for fi in range(1, len(frames)):
            current_frame = frames[fi]
            current_indices = df.index[df["frame"] == current_frame]
            if len(current_indices) == 0:
                continue

            current_centroids = df.loc[
                current_indices, ["centroid_0", "centroid_1"]
            ].values.astype(float)
            current_areas = (
                df.loc[current_indices, "area"].values.astype(float)
                if has_area
                else np.ones(len(current_indices))
            )
            current_major = (
                df.loc[current_indices, "major_axis_length"].values.astype(float)
                if has_shape
                else np.zeros(len(current_indices))
            )
            current_minor = (
                df.loc[current_indices, "minor_axis_length"].values.astype(float)
                if has_shape
                else np.zeros(len(current_indices))
            )

            eligible = [
                tid
                for tid, info in active.items()
                if (current_frame - info["last_frame"]) <= self.gap_bridge_frames
            ]
            linked = set()
            parent_pre_area = {tid: active[tid]["area"] for tid in eligible}
            parent_pre_centroid = {tid: active[tid]["centroid"] for tid in eligible}

            if eligible:
                previous_centroids = np.array([active[tid]["centroid"] for tid in eligible])
                previous_areas = np.array([active[tid]["area"] for tid in eligible])
                previous_major = np.array([active[tid]["major_axis"] for tid in eligible])
                previous_minor = np.array([active[tid]["minor_axis"] for tid in eligible])

                distance = cdist(previous_centroids, current_centroids, metric="euclidean")
                cost = distance.copy()

                if self.area_weight > 0 and has_area:
                    safe = np.maximum(
                        np.minimum(previous_areas[:, None], current_areas[None, :]),
                        1.0,
                    )
                    relative_area = np.abs(
                        previous_areas[:, None] - current_areas[None, :]
                    ) / safe
                    cost = cost + self.area_weight * self.max_distance * relative_area

                if self.shape_weight > 0 and has_shape:
                    safe_major = np.maximum(
                        np.minimum(previous_major[:, None], current_major[None, :]),
                        0.5,
                    )
                    relative_major = np.abs(
                        previous_major[:, None] - current_major[None, :]
                    ) / safe_major
                    safe_minor = np.maximum(
                        np.minimum(previous_minor[:, None], current_minor[None, :]),
                        0.5,
                    )
                    relative_minor = np.abs(
                        previous_minor[:, None] - current_minor[None, :]
                    ) / safe_minor
                    cost = cost + (
                        self.shape_weight
                        * self.max_distance
                        * 0.5
                        * (relative_major + relative_minor)
                    )

                row_ind, col_ind = linear_sum_assignment(cost)
                for row, col in zip(row_ind, col_ind):
                    if distance[row, col] <= self.max_distance:
                        tid = eligible[row]
                        idx = current_indices[col]
                        c0, c1, area, major, minor = get(idx)
                        track_ids[idx] = tid
                        if current_frame - active[tid]["last_frame"] >= 2:
                            n_bridged += 1
                        active[tid] = {
                            "centroid": (c0, c1),
                            "area": area,
                            "major_axis": major,
                            "minor_axis": minor,
                            "last_frame": current_frame,
                        }
                        linked.add(col)
                        n_link += 1

            if self.detect_divisions and has_area:
                min_ratio, max_ratio = self.division_area_ratio_range
                for col in list(linked):
                    idx = current_indices[col]
                    tid = int(track_ids[idx])
                    if tid not in parent_pre_area:
                        continue
                    parent_area = parent_pre_area[tid]
                    if parent_area <= 0:
                        continue
                    daughter1_area = float(df.at[idx, "area"])
                    if daughter1_area / parent_area > max_ratio:
                        continue
                    daughter1_centroid = (
                        float(df.at[idx, "centroid_0"]),
                        float(df.at[idx, "centroid_1"]),
                    )
                    best_col = -1
                    best_distance = np.inf
                    for other_col in range(len(current_indices)):
                        if other_col in linked:
                            continue
                        daughter2_area = float(current_areas[other_col])
                        ratio = daughter2_area / parent_area if parent_area > 0 else 0
                        if not (min_ratio <= ratio <= max_ratio):
                            continue
                        daughter2_centroid = (
                            float(current_centroids[other_col, 0]),
                            float(current_centroids[other_col, 1]),
                        )
                        daughter_distance = float(
                            np.hypot(
                                daughter1_centroid[0] - daughter2_centroid[0],
                                daughter1_centroid[1] - daughter2_centroid[1],
                            )
                        )
                        if (
                            daughter_distance < self.division_search_radius
                            and daughter_distance < best_distance
                        ):
                            best_distance = daughter_distance
                            best_col = other_col

                    if best_col >= 0:
                        idx2 = current_indices[best_col]
                        c0_d2, c1_d2, area_d2, major_d2, minor_d2 = get(idx2)
                        mother_centroid = parent_pre_centroid.get(tid)
                        if mother_centroid is not None:
                            d1_to_mother = float(
                                np.hypot(
                                    daughter1_centroid[0] - mother_centroid[0],
                                    daughter1_centroid[1] - mother_centroid[1],
                                )
                            )
                            d2_to_mother = float(
                                np.hypot(c0_d2 - mother_centroid[0], c1_d2 - mother_centroid[1])
                            )
                        else:
                            d1_to_mother = 0.0
                            d2_to_mother = 1.0

                        new_tid = next_id
                        next_id += 1

                        if d2_to_mother < d1_to_mother:
                            track_ids[idx] = new_tid
                            track_ids[idx2] = tid
                            parent_of[new_tid] = tid
                            active[tid] = {
                                "centroid": (c0_d2, c1_d2),
                                "area": area_d2,
                                "major_axis": major_d2,
                                "minor_axis": minor_d2,
                                "last_frame": current_frame,
                            }
                            active[new_tid] = {
                                "centroid": daughter1_centroid,
                                "area": daughter1_area,
                                "major_axis": (
                                    float(df.at[idx, "major_axis_length"])
                                    if has_shape
                                    else 0.0
                                ),
                                "minor_axis": (
                                    float(df.at[idx, "minor_axis_length"])
                                    if has_shape
                                    else 0.0
                                ),
                                "last_frame": current_frame,
                            }
                        else:
                            track_ids[idx2] = new_tid
                            parent_of[new_tid] = tid
                            active[new_tid] = {
                                "centroid": (c0_d2, c1_d2),
                                "area": area_d2,
                                "major_axis": major_d2,
                                "minor_axis": minor_d2,
                                "last_frame": current_frame,
                            }

                        linked.add(best_col)
                        n_div += 1

            if self.reid_enabled and retired:
                still_unlinked = [
                    col for col in range(len(current_indices)) if col not in linked
                ]
                eligible_retired = {
                    tid: info
                    for tid, info in retired.items()
                    if (current_frame - info["last_frame"]) <= self.reid_max_gap
                }
                if still_unlinked and eligible_retired:
                    retired_ids = list(eligible_retired.keys())
                    retired_centroids = np.array(
                        [eligible_retired[tid]["centroid"] for tid in retired_ids]
                    )
                    retired_areas = np.array(
                        [eligible_retired[tid]["area"] for tid in retired_ids]
                    )
                    retired_major = np.array(
                        [eligible_retired[tid]["major_axis"] for tid in retired_ids]
                    )
                    retired_minor = np.array(
                        [eligible_retired[tid]["minor_axis"] for tid in retired_ids]
                    )
                    still_centroids = current_centroids[still_unlinked]
                    still_areas = current_areas[still_unlinked]
                    still_major = current_major[still_unlinked]
                    still_minor = current_minor[still_unlinked]
                    reid_distance = cdist(retired_centroids, still_centroids, metric="euclidean")

                    safe_area = np.maximum(
                        np.minimum(retired_areas[:, None], still_areas[None, :]),
                        1.0,
                    )
                    relative_area = np.abs(
                        retired_areas[:, None] - still_areas[None, :]
                    ) / safe_area
                    if has_shape:
                        safe_major = np.maximum(
                            np.minimum(retired_major[:, None], still_major[None, :]),
                            0.5,
                        )
                        relative_major = np.abs(
                            retired_major[:, None] - still_major[None, :]
                        ) / safe_major
                        safe_minor = np.maximum(
                            np.minimum(retired_minor[:, None], still_minor[None, :]),
                            0.5,
                        )
                        relative_minor = np.abs(
                            retired_minor[:, None] - still_minor[None, :]
                        ) / safe_minor
                        relative_shape = 0.5 * (relative_major + relative_minor)
                    else:
                        relative_shape = np.zeros_like(reid_distance)

                    reid_cost = reid_distance.astype(float).copy()
                    reid_cost[reid_distance > self.reid_max_distance] = 1e9
                    reid_cost[relative_area > self.reid_area_tol] = 1e9
                    reid_cost[relative_shape > self.reid_shape_tol] = 1e9
                    if (reid_cost < 1e9).any():
                        row_ind, col_ind = linear_sum_assignment(reid_cost)
                        for row, col in zip(row_ind, col_ind):
                            if reid_cost[row, col] >= 1e9:
                                continue
                            tid = retired_ids[row]
                            idx_pos = still_unlinked[col]
                            idx = current_indices[idx_pos]
                            c0, c1, area, major, minor = get(idx)
                            track_ids[idx] = tid
                            active[tid] = {
                                "centroid": (c0, c1),
                                "area": area,
                                "major_axis": major,
                                "minor_axis": minor,
                                "last_frame": current_frame,
                            }
                            del retired[tid]
                            linked.add(idx_pos)
                            n_reid += 1

            for col, idx in enumerate(current_indices):
                if col not in linked:
                    c0, c1, area, major, minor = get(idx)
                    track_ids[idx] = next_id
                    active[next_id] = {
                        "centroid": (c0, c1),
                        "area": area,
                        "major_axis": major,
                        "minor_axis": minor,
                        "last_frame": current_frame,
                    }
                    next_id += 1
                    n_new += 1

            active_cutoff = current_frame - self.gap_bridge_frames
            silent = [
                tid for tid, info in active.items() if info["last_frame"] < active_cutoff
            ]
            for tid in silent:
                retired[tid] = active.pop(tid)

            retired_cutoff = current_frame - self.reid_max_gap
            retired = {
                tid: info
                for tid, info in retired.items()
                if info["last_frame"] >= retired_cutoff
            }

        df["trackid"] = track_ids.astype(np.int32)
        df["parent_track_id"] = df["trackid"].map(parent_of).fillna(-1).astype(np.int32)

        if logger:
            n_tracks = df["trackid"].nunique()
            logger.info(
                f"Hungarian tracking v5 summary: tracks={n_tracks}, link={n_link}, "
                f"new={n_new}, bridged={n_bridged}, reid={n_reid}, divisions={n_div}"
            )

        if self.stitch_max_distance is not None and self.stitch_max_distance > 0:
            df = self.stitch_tracks(
                df,
                max_stitch_distance=self.stitch_max_distance,
                logger=logger,
            )

        return df

    def stitch_tracks(
        self,
        measurements_df: pd.DataFrame,
        max_stitch_distance: float = 8.0,
        logger=None,
    ) -> pd.DataFrame:
        """
        Merge tracks that end and restart at nearly the same position.

        Tracks sharing any frame are never merged. This prevents collapsing
        mother and daughter detections after a division.
        """
        df = measurements_df.copy()
        if "trackid" not in df.columns:
            return df

        tracked = df[df["trackid"] != -1]
        if len(tracked) == 0:
            return df

        track_info = tracked.groupby("trackid").agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
        )
        first_rows = tracked.sort_values("frame").groupby("trackid").first()
        last_rows = tracked.sort_values("frame").groupby("trackid").last()
        track_info["first_c0"] = first_rows["centroid_0"]
        track_info["first_c1"] = first_rows["centroid_1"]
        track_info["last_c0"] = last_rows["centroid_0"]
        track_info["last_c1"] = last_rows["centroid_1"]

        track_frames = tracked.groupby("trackid")["frame"].apply(set).to_dict()
        track_info = track_info.sort_values("first_frame")
        track_ids = track_info.index.tolist()

        merge_map = {}
        merged_frames: Dict[int, set] = {}
        n_stitched = 0

        for i, tid_b in enumerate(track_ids):
            b_start = track_info.at[tid_b, "first_frame"]
            b_c0 = track_info.at[tid_b, "first_c0"]
            b_c1 = track_info.at[tid_b, "first_c1"]
            b_frames = track_frames.get(tid_b, set())

            best_a = None
            best_distance = max_stitch_distance + 1

            for j in range(i - 1, -1, -1):
                tid_a = track_ids[j]
                canonical_a = tid_a
                while canonical_a in merge_map:
                    canonical_a = merge_map[canonical_a]

                if track_info.at[tid_a, "last_frame"] >= b_start:
                    continue

                distance = float(
                    np.hypot(
                        b_c0 - track_info.at[tid_a, "last_c0"],
                        b_c1 - track_info.at[tid_a, "last_c1"],
                    )
                )
                if distance < best_distance:
                    group_frames = merged_frames.get(
                        canonical_a,
                        track_frames.get(canonical_a, set()),
                    )
                    if not group_frames.intersection(b_frames):
                        best_distance = distance
                        best_a = canonical_a

            if best_a is not None and best_distance <= max_stitch_distance:
                merge_map[tid_b] = best_a
                if best_a not in merged_frames:
                    merged_frames[best_a] = track_frames.get(best_a, set()).copy()
                merged_frames[best_a].update(b_frames)
                n_stitched += 1

        if merge_map:
            def resolve(tid):
                visited = set()
                while tid in merge_map and tid not in visited:
                    visited.add(tid)
                    tid = merge_map[tid]
                return tid

            df["trackid"] = df["trackid"].apply(
                lambda tid: resolve(tid) if tid in merge_map else tid
            ).astype(np.int32)

        if logger:
            logger.info(f"Hungarian v5 track stitching: {n_stitched} merges applied")

        return df

    def apply_pipos_lockin(self, measurements_df, logger=None):
        return HungarianTracker.apply_pipos_lockin(self, measurements_df, logger)
