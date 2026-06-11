"""Rule-based morphology corrections for post-classification refinement.

Applied after the primary segmentation model assigns object_class labels.
Detects morphological phenotypes that the model misses or mis-labels and
overrides the class accordingly.

Currently implemented for the instSeg / cellasic vocabulary
    (single-cell, clump, debris).

Three rules, applied in priority order (debris > clump):

  R1 — spaghetti → debris
        single-cell with extreme elongation (high AR) AND irregular boundary
        (low solidity).  Normal stressed/elongated rods are elongated but
        smooth; spaghetti forms are bent/tangled.

  R2 — lysis / explosion → debris
        single-cell or clump with very low solidity AND area above the
        noise floor.  Fragmented, burst cells have highly irregular masks.

  R3 — merged detection → clump
        single-cell whose mask skeleton has branch points (dumbbell shape
        from two touching cells merged into one detection) AND area above
        the expected single-cell size.  Requires no intensity image —
        purely geometric from the labeled mask.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.ndimage import convolve as nd_convolve


# 3×3 kernel counting the 8-connected neighbours of each skeleton pixel
_BRANCH_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _branch_point_count(binary_mask: np.ndarray) -> int:
    """Count skeleton branch points in a 2-D binary mask."""
    if binary_mask.sum() < 5:
        return 0
    skel = skeletonize(binary_mask)
    if not skel.any():
        return 0
    neighbors = nd_convolve(
        skel.astype(np.int32), _BRANCH_KERNEL, mode="constant", cval=0
    )
    # A branch point has ≥ 3 skeleton neighbours
    return int((skel & (neighbors >= 3)).sum())


def _add_aspect_ratio(fl_measurements: pd.DataFrame) -> pd.DataFrame:
    """Add aspect_ratio = major_axis_length / minor_axis_length (clipped)."""
    fl = fl_measurements.copy()
    minor = fl["minor_axis_length"].clip(lower=1e-6)
    fl["aspect_ratio"] = fl["major_axis_length"] / minor
    return fl


def _add_skeleton_features(
    fl_measurements: pd.DataFrame,
    labeled_mask: np.ndarray,
) -> pd.DataFrame:
    """Add skeleton_branch_points column.

    Iterates over frames for efficiency; within each frame iterates over
    ROI labels and computes the branch-point count from the binary mask.
    Requires only the labeled mask — no intensity image.

    Args:
        fl_measurements: DataFrame with 'frame' and 'label' columns.
        labeled_mask: Shape (T, S, C, X, Y) — the TSCXY mask array.
    """
    lm_work = labeled_mask[:, 0, 0, :, :] if labeled_mask.ndim == 5 else labeled_mask

    bp_series = pd.Series(0, index=fl_measurements.index, dtype=np.int32)

    for frame_idx, frame_group in fl_measurements.groupby("frame"):
        lm_frame = lm_work[int(frame_idx)]
        for row_idx, row in frame_group.iterrows():
            bp_series.at[row_idx] = _branch_point_count(
                lm_frame == int(row["label"])
            )

    fl = fl_measurements.copy()
    fl["skeleton_branch_points"] = bp_series
    return fl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_instSeg_morphology_corrections(
    fl_measurements: pd.DataFrame,
    labeled_mask: np.ndarray,
    spaghetti_ar: float = 8.0,
    spaghetti_solidity: float = 0.72,
    lysis_solidity: float = 0.45,
    lysis_area_frac: float = 0.4,
    clump_branch_min: int = 1,
    clump_area_frac: float = 1.8,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Rule-based morphology corrections for instSeg / cellasic pipelines.

    Adds two derived feature columns to the returned DataFrame:
        aspect_ratio            = major_axis_length / minor_axis_length
        skeleton_branch_points  = number of skeleton branch points in the ROI mask

    These columns are kept for downstream analysis (morphology feature step).

    Args:
        fl_measurements: DataFrame containing at minimum object_class, area,
            major_axis_length, minor_axis_length, solidity, frame, label.
        labeled_mask: (T, S, C, X, Y) integer mask array from img_analyser.
        spaghetti_ar: Aspect-ratio threshold for R1.  Default 8.0.
        spaghetti_solidity: Solidity ceiling for R1.  Default 0.72.
        lysis_solidity: Solidity ceiling for R2.  Default 0.45.
        lysis_area_frac: Area floor (× median) for R2.  Default 0.4.
        clump_branch_min: Min branch-point count for R3.  Default 1.
        clump_area_frac: Area floor (× median) for R3.  Default 1.8.

    Returns:
        Tuple of (updated DataFrame, correction counts dict).
        Counts keys: spaghetti_to_debris, lysed_to_debris, merged_to_clump.
    """
    fl = _add_aspect_ratio(fl_measurements)
    fl = _add_skeleton_features(fl, labeled_mask)

    # Reference area from single-cell labels only — clumps inflate the median
    sc_areas = fl.loc[fl["object_class"] == "single-cell", "area"]
    median_area = sc_areas.median() if len(sc_areas) > 0 else fl["area"].median()

    # R1: spaghetti → debris
    # High AR catches extreme elongation; low solidity ensures the shape is
    # bent/tangled rather than a straight stressed rod (which stays single-cell).
    r1 = (
        (fl["object_class"] == "single-cell")
        & (fl["aspect_ratio"] > spaghetti_ar)
        & (fl["solidity"] < spaghetti_solidity)
    )

    # R2: lysis / explosion → debris
    # Very low solidity (fragmented boundary) + non-trivial area (not just noise).
    r2 = (
        fl["object_class"].isin(["single-cell", "clump"])
        & (fl["solidity"] < lysis_solidity)
        & (fl["area"] > lysis_area_frac * median_area)
    )

    # R3: merged detection → clump
    # Skeleton branch points flag a multi-cell cluster (the skeleton forks at
    # cell junctions).  The solidity floor (> 0.70) excludes fragmented masses
    # that also have many branch points but should be debris not clump.
    # Note: end-to-end two-cell linear merges have 0 branch points and are
    # NOT caught by this rule — the model's own clump class handles them.
    r3 = (
        (fl["object_class"] == "single-cell")
        & (fl["skeleton_branch_points"] >= clump_branch_min)
        & (fl["area"] > clump_area_frac * median_area)
        & (fl["solidity"] > 0.70)
    )

    fl.loc[r3, "object_class"] = "clump"           # applied first
    fl.loc[r1 | r2, "object_class"] = "debris"     # overrides clump if overlap

    counts: Dict[str, int] = {
        "spaghetti_to_debris": int(r1.sum()),
        "lysed_to_debris": int((r2 & ~r1).sum()),
        "merged_to_clump": int((r3 & ~r1 & ~r2).sum()),
    }
    return fl, counts
