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

    def __init__(self, model_path: str, device: Optional[str] = None):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.input_dim: int = checkpoint["input_dim"]
        self.feature_names: list[str] = checkpoint["feature_names"]
        self.threshold: float = checkpoint["threshold"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._model = _MLP(self.input_dim)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.to(self._device)
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

        labels_t  = sorted(props_t.keys())
        labels_t1 = sorted(props_t1.keys())
        n_t, n_t1 = len(labels_t), len(labels_t1)

        if not labels_t or not labels_t1:
            return (np.full((n_t, n_t1), max_cost, dtype=np.float32),
                    labels_t, labels_t1)

        med_major = float(stats.get("median_major", 1.0))
        H, W = masks_t.shape

        # ── Vectorised scalar features (no per-pixel work) ──────────────────
        def _prop_arr(props, labels, attr):
            return np.array([getattr(props[l], attr) for l in labels], dtype=np.float64)

        cy_t   = _prop_arr(props_t,  labels_t,  "centroid")[:, 0]
        cx_t   = _prop_arr(props_t,  labels_t,  "centroid")[:, 1]
        area_t = _prop_arr(props_t,  labels_t,  "area")
        maj_t  = _prop_arr(props_t,  labels_t,  "major_axis_length")
        min_t  = _prop_arr(props_t,  labels_t,  "minor_axis_length")
        ecc_t  = _prop_arr(props_t,  labels_t,  "eccentricity")
        sol_t  = _prop_arr(props_t,  labels_t,  "solidity")
        ori_t  = _prop_arr(props_t,  labels_t,  "orientation")

        cy_t1   = _prop_arr(props_t1, labels_t1, "centroid")[:, 0]
        cx_t1   = _prop_arr(props_t1, labels_t1, "centroid")[:, 1]
        area_t1 = _prop_arr(props_t1, labels_t1, "area")
        maj_t1  = _prop_arr(props_t1, labels_t1, "major_axis_length")
        min_t1  = _prop_arr(props_t1, labels_t1, "minor_axis_length")
        ecc_t1  = _prop_arr(props_t1, labels_t1, "eccentricity")
        sol_t1  = _prop_arr(props_t1, labels_t1, "solidity")
        ori_t1  = _prop_arr(props_t1, labels_t1, "orientation")

        # Pairwise matrices via broadcasting: shape (n_t, n_t1)
        dy   = cy_t[:, None] - cy_t1[None, :]
        dx   = cx_t[:, None] - cx_t1[None, :]
        dist = np.sqrt(dy**2 + dx**2) / med_major

        area_i    = area_t[:, None]
        delta_area = np.where(area_i > 0, (area_t1[None, :] - area_i) / area_i, 0.0)
        area_ratio = np.where(area_i > 0, area_t1[None, :] / area_i, 1.0)

        d_ori = ori_t1[None, :] - ori_t[:, None]
        d_ori = (d_ori + np.pi / 2) % np.pi - np.pi / 2  # wrap to [-π/2, π/2]

        d_ecc = ecc_t1[None, :] - ecc_t[:, None]
        d_sol = sol_t1[None, :] - sol_t[:, None]
        d_maj = (maj_t1[None, :] - maj_t[:, None]) / med_major
        d_min = (min_t1[None, :] - min_t[:, None]) / med_major

        # ── IoU matrix: O(H×W) via overlap pixel counting ───────────────────
        # Find pixels where both frames have nonzero labels.
        both    = (masks_t > 0) & (masks_t1 > 0)
        lt_pix  = masks_t[both].astype(np.intp)
        lt1_pix = masks_t1[both].astype(np.intp)

        # Vectorised label → 0-based index lookup (no Python loop over pixels).
        max_lbl = int(max(labels_t[-1], labels_t1[-1])) + 1
        lt_lookup  = np.full(max_lbl, -1, dtype=np.intp)
        lt1_lookup = np.full(max_lbl, -1, dtype=np.intp)
        for i, l in enumerate(labels_t):
            lt_lookup[l] = i
        for j, l in enumerate(labels_t1):
            lt1_lookup[l] = j

        li_arr = lt_lookup[lt_pix]
        lj_arr = lt1_lookup[lt1_pix]
        valid  = (li_arr >= 0) & (lj_arr >= 0)
        li_arr, lj_arr = li_arr[valid], lj_arr[valid]

        inter = np.bincount(li_arr * n_t1 + lj_arr,
                            minlength=n_t * n_t1).reshape(n_t, n_t1).astype(np.float64)
        union = area_t[:, None] + area_t1[None, :] - inter
        iou   = np.where(union > 0, inter / union, 0.0)

        # ── Extrapolated IoU: shift mask_i by per-cell velocity (t-1 → t) ──
        # For cells without a prev frame, extrap_iou = iou (conservative fallback).
        # For cells with velocity, only update close pairs (others stay at static iou).
        extrap_iou = iou.copy()
        iou_px_cutoff = 3.0 * med_major  # skip extrap for pairs > 3 cell lengths apart
        if props_t_prev:
            for i, li in enumerate(labels_t):
                rp_prev = props_t_prev.get(li)
                if rp_prev is None:
                    continue
                cy_prev, cx_prev = rp_prev.centroid
                # Velocity vector: where cell i is predicted to be at t+1
                shift_y = int(round(cy_t[i] - cy_prev))
                shift_x = int(round(cx_t[i] - cx_prev))
                if shift_y == 0 and shift_x == 0:
                    continue  # no motion — extrap_iou == iou already
                # Shift mask_i using roll (small boundary error is acceptable)
                mask_i = masks_t == li
                shifted = np.roll(np.roll(mask_i, shift_y, axis=0), shift_x, axis=1)
                # Predicted centroid of cell i at t+1
                cy_pred = cy_t[i] + shift_y
                cx_pred = cx_t[i] + shift_x
                close_j = np.where(
                    (np.abs(cy_pred - cy_t1) < iou_px_cutoff) &
                    (np.abs(cx_pred - cx_t1) < iou_px_cutoff)
                )[0]
                for j in close_j:
                    mask_j = masks_t1 == labels_t1[j]
                    inter_s = np.logical_and(shifted, mask_j).sum()
                    union_s = np.logical_or(shifted, mask_j).sum()
                    extrap_iou[i, j] = float(inter_s / union_s) if union_s > 0 else 0.0

        # ── Assemble feature tensor (n_t × n_t1 × 10) ────────────────────────
        feats = np.stack([
            dist, delta_area, area_ratio, iou,
            d_ecc, d_sol, d_ori, d_maj, d_min,
            extrap_iou,
        ], axis=-1).astype(np.float32)   # (n_t, n_t1, 10)

        feat_tensor = torch.tensor(
            feats.reshape(n_t * n_t1, self.input_dim),
            dtype=torch.float32,
        ).to(self._device)

        with torch.no_grad():
            probs = torch.sigmoid(self._model(feat_tensor)).cpu().numpy()

        cost_matrix = (1.0 - probs.reshape(n_t, n_t1)).astype(np.float32)
        cost_matrix = np.clip(cost_matrix, 0.0, max_cost)

        return cost_matrix, labels_t, labels_t1
