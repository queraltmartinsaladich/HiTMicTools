"""
DivisionClassifier — wraps the trained division MLP and detects division events
post-tracking, replacing the reconcile_lineage() area-halving heuristic.

For each track that could be a division parent (area roughly halves between two
consecutive frames), all pairs of candidate daughters at t+1 are scored. The
highest-scoring pair above threshold is recorded as a division event.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from HiTMicTools.tracking.feature_extraction import (
    get_frame_props,
    triplet_features,
    compute_movie_stats,
)


class _MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DivisionClassifier:
    """
    Loads a trained division MLP and exposes predict_divisions().

    Args:
        model_path: Path to the .pt file saved by train_division.py.
    """

    def __init__(self, model_path: str):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.input_dim: int = checkpoint["input_dim"]
        self.feature_names: list[str] = checkpoint["feature_names"]
        self.threshold: float = checkpoint["threshold"]

        self._model = _MLP(self.input_dim)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()

    def predict_divisions(
        self,
        fl_measurements: pd.DataFrame,
        masks: np.ndarray,
        area_drop_frac: float = 0.35,
        max_daughter_dist_norm: float = 3.0,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Detect division events and fill division_parent_trackid.

        Replaces reconcile_lineage(). Scans every track for frames where the
        cell area drops by >= area_drop_frac relative to the previous frame,
        then scores all (daughter1, daughter2) pairs at that frame using the
        trained MLP. The best-scoring pair above threshold is recorded.

        Args:
            fl_measurements: Per-cell per-frame DataFrame (post-tracking).
                             Must contain: trackid, frame, area, centroid_x,
                             centroid_y, label, object_class.
            masks:           (T, H, W) uint16 labeled mask array for the movie.
            area_drop_frac:  Minimum fractional area drop to trigger a division
                             candidate check (default 0.35).
            max_daughter_dist_norm: Max daughter centroid distance from parent
                             (in units of median cell major axis) to be considered.

        Returns:
            fl_measurements: Input DataFrame with division_parent_trackid filled.
            counts:          Dict with n_reconciled_divisions.
        """
        if "division_parent_trackid" not in fl_measurements.columns:
            fl_measurements["division_parent_trackid"] = np.nan

        stats = compute_movie_stats(masks)
        med_major = stats["median_major"]
        T = masks.shape[0]

        n_found = 0
        df = fl_measurements

        # Build per-frame label→trackid reverse map
        for t in range(1, T):
            frame_t_prev = df[df["frame"] == t - 1]
            frame_t = df[df["frame"] == t]
            if frame_t_prev.empty or frame_t.empty:
                continue

            props_t_prev = get_frame_props(masks[t - 1])
            props_t = get_frame_props(masks[t])
            props_t2_prev = get_frame_props(masks[t - 2]) if t >= 2 else {}

            # label→trackid for frame t
            label_to_track_t = dict(zip(frame_t["label"].astype(int),
                                        frame_t["trackid"]))

            for _, row_prev in frame_t_prev.iterrows():
                label_prev = int(row_prev["label"])
                trackid_prev = row_prev["trackid"]
                area_prev = row_prev["area"]

                if label_prev not in props_t_prev:
                    continue

                # Check if this track already has a successor at frame t
                track_at_t = frame_t[frame_t["trackid"] == trackid_prev]
                if not track_at_t.empty:
                    # Track continues — check for area drop suggesting division
                    area_t = track_at_t.iloc[0]["area"]
                    if area_prev == 0 or (area_t / area_prev) > (1.0 - area_drop_frac):
                        continue  # No significant area drop

                # Parent lost or area-dropped — look for daughter pairs at frame t
                rp_parent = props_t_prev[label_prev]
                cy_p, cx_p = rp_parent.centroid
                mask_parent = masks[t - 1] == label_prev
                rp_parent_prev = props_t2_prev.get(label_prev)

                # Candidate daughters: cells at frame t within distance threshold
                candidates = []
                for label_t, rp_t in props_t.items():
                    cy_t, cx_t = rp_t.centroid
                    dist_norm = (np.sqrt((cy_p - cy_t)**2 + (cx_p - cx_t)**2)
                                 / (med_major + 1e-8))
                    if dist_norm <= max_daughter_dist_norm:
                        candidates.append(label_t)

                if len(candidates) < 2:
                    continue

                # Score all candidate daughter pairs
                best_score = -1.0
                best_pair = None
                feats_batch = []
                pairs_batch = []

                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        d1, d2 = candidates[i], candidates[j]
                        if d1 not in props_t or d2 not in props_t:
                            continue
                        feat = triplet_features(
                            rp_parent, mask_parent,
                            props_t[d1], masks[t] == d1,
                            props_t[d2], masks[t] == d2,
                            stats, rp_parent_prev,
                        )
                        feats_batch.append(feat)
                        pairs_batch.append((d1, d2))

                if not feats_batch:
                    continue

                feat_tensor = torch.tensor(
                    np.array(feats_batch, dtype=np.float32))
                self._model.eval()
                with torch.no_grad():
                    scores = torch.sigmoid(self._model(feat_tensor)).numpy()

                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])

                if best_score >= self.threshold:
                    best_pair = pairs_batch[best_idx]
                    d1_label, d2_label = best_pair
                    # Assign division_parent_trackid to daughter tracks
                    for d_label in (d1_label, d2_label):
                        d_track = label_to_track_t.get(d_label)
                        if d_track is not None:
                            mask_row = ((df["trackid"] == d_track) &
                                        (df["frame"] == t))
                            df.loc[mask_row, "division_parent_trackid"] = trackid_prev
                    n_found += 1

        return df, {"n_reconciled_divisions": n_found}
