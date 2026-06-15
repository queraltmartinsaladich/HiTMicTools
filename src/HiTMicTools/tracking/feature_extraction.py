"""
Per-cell, per-pair, and per-triplet feature extraction for learned tracking.

All features are normalised so they are species- and scale-agnostic:
  - areas divided by median cell area in the movie
  - lengths divided by median major axis length in the movie
  - centroids divided by frame (H, W)
  - distances divided by median major axis length

This module operates on labeled mask arrays (T, H, W) and lineage dicts.
No raw pixel values are used — features come entirely from regionprops.
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Optional
from skimage.measure import regionprops


# ── per-cell descriptor (8 features) ─────────────────────────────────────────

CELL_FEATURE_NAMES = [
    "area_norm",
    "major_axis_norm",
    "minor_axis_norm",
    "eccentricity",
    "solidity",
    "orientation",
    "centroid_y_norm",
    "centroid_x_norm",
]

# ── per-pair descriptor (10 features) ────────────────────────────────────────

PAIR_FEATURE_NAMES = [
    "centroid_dist_norm",
    "delta_area",
    "area_ratio",
    "iou",
    "delta_eccentricity",
    "delta_solidity",
    "delta_orientation",
    "delta_major_axis_norm",
    "delta_minor_axis_norm",
    "extrapolated_overlap",
]

# ── per-triplet descriptor (8 features) ──────────────────────────────────────

TRIPLET_FEATURE_NAMES = [
    "area_conservation",
    "area_symmetry",
    "combined_iou",
    "dist_daughter1_norm",
    "dist_daughter2_norm",
    "angle_between_daughters",
    "parent_elongation",
    "parent_area_growth",
]


# ── normalisation statistics ──────────────────────────────────────────────────

def compute_movie_stats(masks: np.ndarray) -> dict:
    """Compute per-movie normalisation constants from all non-background regions."""
    areas, majors = [], []
    for t in range(masks.shape[0]):
        for rp in regionprops(masks[t].astype(np.int32)):
            areas.append(rp.area)
            majors.append(rp.major_axis_length if rp.major_axis_length > 0 else 1.0)
    if not areas:
        return {"median_area": 1.0, "median_major": 1.0,
                "H": masks.shape[1], "W": masks.shape[2]}
    return {
        "median_area": float(np.median(areas)),
        "median_major": float(np.median(majors)),
        "H": masks.shape[1],
        "W": masks.shape[2],
    }


# ── per-frame regionprops cache ───────────────────────────────────────────────

def get_frame_props(mask_frame: np.ndarray) -> dict[int, object]:
    """Return {label: regionprops_object} for one frame (excludes background 0)."""
    return {rp.label: rp for rp in regionprops(mask_frame.astype(np.int32))}


# ── mask IoU helpers ──────────────────────────────────────────────────────────

def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def _shifted_iou(mask_t: np.ndarray, mask_t1: np.ndarray,
                 cy_t: float, cx_t: float,
                 cy_t1: float, cx_t1: float) -> float:
    """IoU after shifting mask_t to predicted position at t+1 via centroid delta."""
    dy = int(round(cy_t1 - cy_t))
    dx = int(round(cx_t1 - cx_t))
    shifted = np.roll(np.roll(mask_t, dy, axis=0), dx, axis=1)
    return _mask_iou(shifted, mask_t1)


# ── cell feature vector ───────────────────────────────────────────────────────

def cell_features(rp, stats: dict) -> np.ndarray:
    """8-dim normalised feature vector for one cell."""
    H, W = stats["H"], stats["W"]
    med_area = stats["median_area"]
    med_major = stats["median_major"]
    cy, cx = rp.centroid
    return np.array([
        rp.area / med_area,
        rp.major_axis_length / med_major,
        rp.minor_axis_length / med_major,
        rp.eccentricity,
        rp.solidity,
        rp.orientation,          # radians, [-π/2, π/2]
        cy / H,
        cx / W,
    ], dtype=np.float32)


# ── pair feature vector ───────────────────────────────────────────────────────

def pair_features(
    rp_i, mask_i: np.ndarray,
    rp_j, mask_j: np.ndarray,
    stats: dict,
    rp_i_prev=None,
) -> np.ndarray:
    """
    10-dim feature vector for candidate assignment (cell_i @ t) → (cell_j @ t+1).

    rp_i_prev: regionprops of cell_i at t-1, used for motion extrapolation.
               Pass None if t == 0.
    """
    med_major = stats["median_major"]
    cy_i, cx_i = rp_i.centroid
    cy_j, cx_j = rp_j.centroid

    dist = np.sqrt((cy_i - cy_j) ** 2 + (cx_i - cx_j) ** 2)

    iou = _mask_iou(mask_i, mask_j)

    # Extrapolated overlap: shift mask_i by velocity estimated from t-1 → t
    if rp_i_prev is not None:
        cy_prev, cx_prev = rp_i_prev.centroid
        cy_pred = cy_i + (cy_i - cy_prev)
        cx_pred = cx_i + (cx_i - cx_prev)
        extrap_iou = _shifted_iou(mask_i, mask_j, cy_i, cx_i, cy_pred, cx_pred)
    else:
        extrap_iou = iou  # no motion info — fall back to static IoU

    area_i = rp_i.area
    area_j = rp_j.area
    area_ratio = area_j / area_i if area_i > 0 else 1.0
    delta_area = (area_j - area_i) / area_i if area_i > 0 else 0.0

    delta_orient = rp_j.orientation - rp_i.orientation
    # wrap to [-π/2, π/2]
    while delta_orient > np.pi / 2:
        delta_orient -= np.pi
    while delta_orient < -np.pi / 2:
        delta_orient += np.pi

    return np.array([
        dist / med_major,
        delta_area,
        area_ratio,
        iou,
        rp_j.eccentricity - rp_i.eccentricity,
        rp_j.solidity - rp_i.solidity,
        delta_orient,
        (rp_j.major_axis_length - rp_i.major_axis_length) / med_major,
        (rp_j.minor_axis_length - rp_i.minor_axis_length) / med_major,
        extrap_iou,
    ], dtype=np.float32)


# ── triplet feature vector ────────────────────────────────────────────────────

def triplet_features(
    rp_parent, mask_parent: np.ndarray,
    rp_d1, mask_d1: np.ndarray,
    rp_d2, mask_d2: np.ndarray,
    stats: dict,
    rp_parent_prev=None,
) -> np.ndarray:
    """
    8-dim feature vector for candidate division (parent @ t) → (d1, d2 @ t+1).
    """
    med_major = stats["median_major"]
    cy_p, cx_p = rp_parent.centroid
    cy_d1, cx_d1 = rp_d1.centroid
    cy_d2, cx_d2 = rp_d2.centroid

    area_p = rp_parent.area
    area_d1 = rp_d1.area
    area_d2 = rp_d2.area

    area_conservation = (area_d1 + area_d2) / area_p if area_p > 0 else 1.0
    area_symmetry = (min(area_d1, area_d2) / max(area_d1, area_d2)
                     if max(area_d1, area_d2) > 0 else 1.0)

    combined_mask = np.logical_or(mask_d1, mask_d2)
    combined_iou = _mask_iou(mask_parent, combined_mask)

    dist_d1 = np.sqrt((cy_p - cy_d1) ** 2 + (cx_p - cx_d1) ** 2) / med_major
    dist_d2 = np.sqrt((cy_p - cy_d2) ** 2 + (cx_p - cx_d2) ** 2) / med_major

    # Angle between daughters at parent centroid (should be ~π for rods)
    v1 = np.array([cy_d1 - cy_p, cx_d1 - cx_p])
    v2 = np.array([cy_d2 - cy_p, cx_d2 - cx_p])
    cos_angle = (np.dot(v1, v2) /
                 (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    elongation = (rp_parent.major_axis_length / rp_parent.minor_axis_length
                  if rp_parent.minor_axis_length > 0 else 1.0)

    # Parent area growth from previous frame (proxy for pre-division elongation)
    if rp_parent_prev is not None:
        area_growth = (area_p - rp_parent_prev.area) / rp_parent_prev.area
    else:
        area_growth = 0.0

    return np.array([
        area_conservation,
        area_symmetry,
        combined_iou,
        dist_d1,
        dist_d2,
        angle,
        elongation,
        float(area_growth),
    ], dtype=np.float32)


# ── movie-level extraction ────────────────────────────────────────────────────

def extract_from_movie(
    masks: np.ndarray,
    lineage: dict[str, dict[str, int]],
    max_dist_norm: float = 5.0,
) -> dict:
    """
    Extract all assignment pair and division triplet examples from one movie.

    Args:
        masks: (T, H, W) uint16 labeled mask array.
        lineage: {str(t): {str(daughter_label): parent_label_at_t_minus_1}}
                 Frame keys start at "1" (frame 0 has no parents).
        max_dist_norm: Maximum centroid distance (in units of median_major) for
                       a pair to be considered a candidate. Filters out clearly
                       impossible assignments.

    Returns dict with keys:
        pair_feats:   (N, 10) float32
        pair_labels:  (N,)    int8  — 1=true assignment, 0=negative
        triplet_feats:(M, 8)  float32
        triplet_labels:(M,)   int8  — 1=true division, 0=negative
    """
    stats = compute_movie_stats(masks)
    T = masks.shape[0]

    pair_feats, pair_labels = [], []
    triplet_feats, triplet_labels = [], []

    # Cache regionprops per frame so we don't recompute
    props_cache: dict[int, dict[int, object]] = {}

    def _props(t: int) -> dict[int, object]:
        if t not in props_cache:
            props_cache[t] = get_frame_props(masks[t])
        return props_cache[t]

    for t in range(T - 1):
        t1 = t + 1
        lin_t1 = lineage.get(str(t1), {})  # {str(daughter_label): parent_label}

        props_t = _props(t)
        props_t1 = _props(t1)
        props_t_prev = _props(t - 1) if t > 0 else {}

        if not props_t or not props_t1:
            continue

        # Build ground-truth maps for this frame pair
        # gt_assign[label_j] = parent_label  (for non-division continuations)
        # divisions: parent_label → [daughter_label1, daughter_label2]
        parent_to_daughters: dict[int, list[int]] = {}
        for str_j, par in lin_t1.items():
            j = int(str_j)
            parent_to_daughters.setdefault(par, []).append(j)

        # Cells that divided: parent appears twice in lin_t1
        dividing_parents = {p: ds for p, ds in parent_to_daughters.items()
                            if len(ds) >= 2 and p in props_t}
        # Cells that continued (no division)
        continuing: dict[int, int] = {}  # label_j → parent_label
        for str_j, par in lin_t1.items():
            j = int(str_j)
            if par not in dividing_parents and j in props_t1:
                continuing[j] = par

        # ── assignment pairs ──────────────────────────────────────────────────
        for label_i, rp_i in props_t.items():
            mask_i = masks[t] == label_i
            rp_i_prev = props_t_prev.get(label_i)
            cy_i, cx_i = rp_i.centroid

            for label_j, rp_j in props_t1.items():
                cy_j, cx_j = rp_j.centroid
                dist_norm = (np.sqrt((cy_i - cy_j) ** 2 + (cx_i - cx_j) ** 2)
                             / stats["median_major"])
                if dist_norm > max_dist_norm:
                    continue

                mask_j = masks[t1] == label_j
                feat = pair_features(rp_i, mask_i, rp_j, mask_j, stats, rp_i_prev)

                # Positive: label_j's parent is label_i AND label_i didn't divide
                is_positive = (
                    label_i not in dividing_parents
                    and continuing.get(label_j) == label_i
                )
                pair_feats.append(feat)
                pair_labels.append(1 if is_positive else 0)

        # ── division triplets ─────────────────────────────────────────────────
        for parent_label, daughters in dividing_parents.items():
            rp_parent = props_t[parent_label]
            mask_parent = masks[t] == parent_label
            rp_parent_prev = props_t_prev.get(parent_label)

            # Positive triplet: true daughters
            if len(daughters) >= 2:
                d1, d2 = daughters[0], daughters[1]
                if d1 in props_t1 and d2 in props_t1:
                    feat = triplet_features(
                        rp_parent, mask_parent,
                        props_t1[d1], masks[t1] == d1,
                        props_t1[d2], masks[t1] == d2,
                        stats, rp_parent_prev,
                    )
                    triplet_feats.append(feat)
                    triplet_labels.append(1)

            # Negative triplets: random pairs of non-daughter cells at t+1
            non_daughters = [l for l in props_t1 if l not in daughters]
            rng = np.random.default_rng(seed=t * 1000 + parent_label)
            for _ in range(min(3, len(non_daughters) * (len(non_daughters) - 1) // 2)):
                neg_pair = rng.choice(non_daughters, size=2, replace=False)
                n1, n2 = int(neg_pair[0]), int(neg_pair[1])
                feat = triplet_features(
                    rp_parent, mask_parent,
                    props_t1[n1], masks[t1] == n1,
                    props_t1[n2], masks[t1] == n2,
                    stats, rp_parent_prev,
                )
                triplet_feats.append(feat)
                triplet_labels.append(0)

    return {
        "pair_feats": np.array(pair_feats, dtype=np.float32) if pair_feats
                      else np.empty((0, len(PAIR_FEATURE_NAMES)), dtype=np.float32),
        "pair_labels": np.array(pair_labels, dtype=np.int8) if pair_labels
                       else np.empty(0, dtype=np.int8),
        "triplet_feats": np.array(triplet_feats, dtype=np.float32) if triplet_feats
                         else np.empty((0, len(TRIPLET_FEATURE_NAMES)), dtype=np.float32),
        "triplet_labels": np.array(triplet_labels, dtype=np.int8) if triplet_labels
                          else np.empty(0, dtype=np.int8),
    }
