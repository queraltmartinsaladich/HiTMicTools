"""
AssignmentScorer — wraps the trained assignment MLP and produces a cost matrix
for the Hungarian tracker in place of Euclidean centroid distance.

Cost = 1 - P(true assignment), so low-cost pairs are strongly preferred.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from HiTMicTools.tracking.feature_extraction import (
    compute_movie_stats,
    get_frame_props,
    pair_features,
)


class _MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AssignmentScorer:
    """
    Loads a trained assignment MLP and exposes predict_cost_matrix().

    Args:
        model_path: Path to the .pt file saved by train_assignment.py.
    """

    def __init__(self, model_path: str):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.input_dim: int = checkpoint["input_dim"]
        self.feature_names: list[str] = checkpoint["feature_names"]
        self.threshold: float = checkpoint["threshold"]

        self._model = _MLP(self.input_dim)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()

    def predict_cost_matrix(
        self,
        masks_t: np.ndarray,
        masks_t1: np.ndarray,
        stats: dict,
        masks_t_prev: Optional[np.ndarray] = None,
        max_cost: float = 0.95,
    ) -> tuple[np.ndarray, list[int], list[int]]:
        """
        Compute a learned cost matrix between cells at frame t and t+1.

        Args:
            masks_t:      (H, W) uint16 labeled mask at frame t.
            masks_t1:     (H, W) uint16 labeled mask at frame t+1.
            stats:        Movie normalisation dict from compute_movie_stats().
            masks_t_prev: (H, W) labeled mask at frame t-1 for motion extrapolation.
                          Pass None for the first frame.
            max_cost:     Pairs with cost > max_cost are treated as unlinked.

        Returns:
            cost_matrix:  (n_t, n_t1) float32 array. Entry [i, j] = cost of linking
                          cell labels_t[i] to labels_t1[j].
            labels_t:     Ordered list of cell labels in frame t (row indices).
            labels_t1:    Ordered list of cell labels in frame t+1 (col indices).
        """
        props_t = get_frame_props(masks_t)
        props_t1 = get_frame_props(masks_t1)
        props_t_prev = get_frame_props(masks_t_prev) if masks_t_prev is not None else {}

        labels_t = sorted(props_t.keys())
        labels_t1 = sorted(props_t1.keys())

        if not labels_t or not labels_t1:
            return (np.full((len(labels_t), len(labels_t1)), max_cost, dtype=np.float32),
                    labels_t, labels_t1)

        feats = []
        for li in labels_t:
            mask_i = masks_t == li
            rp_i = props_t[li]
            rp_i_prev = props_t_prev.get(li)
            row = []
            for lj in labels_t1:
                mask_j = masks_t1 == lj
                rp_j = props_t1[lj]
                f = pair_features(rp_i, mask_i, rp_j, mask_j, stats, rp_i_prev)
                row.append(f)
            feats.append(row)

        n_t, n_t1 = len(labels_t), len(labels_t1)
        feat_tensor = torch.tensor(
            np.array(feats).reshape(n_t * n_t1, self.input_dim),
            dtype=torch.float32,
        )

        with torch.no_grad():
            probs = torch.sigmoid(self._model(feat_tensor)).numpy()

        cost_matrix = (1.0 - probs.reshape(n_t, n_t1)).astype(np.float32)
        cost_matrix = np.clip(cost_matrix, 0.0, max_cost)

        return cost_matrix, labels_t, labels_t1
