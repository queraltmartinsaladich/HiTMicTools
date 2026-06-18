"""Rule-based morphology corrections for post-classification refinement.

Applied after the primary segmentation model assigns object_class labels.
Detects morphological phenotypes that the model misses or mis-labels and
overrides the class accordingly.

Currently implemented for the instSeg / cellasic vocabulary
    (single-cell, clump, debris).

Three rules applied after the primary segmentation model assigns labels.
The model's ``debris`` class is a catch-all; R1/R2 can rescue biologically
meaningful cells that were mis-labelled as debris.  Only truly artifact-
shaped detections (wrong size/shape for both rules) remain as debris.

  R1 — filamentation → long
        single-cell or debris with extreme elongation (high AR) AND
        irregular boundary (low solidity).  Normal stressed/elongated rods
        stay single-cell; spaghetti forms (bent/tangled, growth without
        separation) become ``long``.  Biologically meaningful — may or may
        not be dying; passes through the PI classifier normally.

  R2 — lysis / explosion → lyse
        single-cell, clump, or debris with very low solidity AND area above
        the noise floor.  Ruptured cells have highly irregular masks.  PI
        signal may have diffused post-lysis so they go through the classifier.

  R3 — merged detection → clump
        single-cell whose mask skeleton has branch points (dumbbell shape
        from two touching cells merged into one detection) AND area above
        the expected single-cell size.  Requires no intensity image —
        purely geometric from the labeled mask.
"""
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_aspect_ratio(fl_measurements: pd.DataFrame) -> pd.DataFrame:
    """Add aspect_ratio = major_axis_length / minor_axis_length (clipped)."""
    fl = fl_measurements.copy()
    minor = fl["minor_axis_length"].clip(lower=1e-6)
    fl["aspect_ratio"] = fl["major_axis_length"] / minor
    return fl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_instSeg_morphology_corrections(
    fl_measurements: pd.DataFrame,
    spaghetti_ar: float = 5.0,
    spaghetti_solidity: float = 0.72,
    lysis_solidity: float = 0.45,
    lysis_area_frac: float = 0.4,
    clump_branch_min: int = 1,
    clump_area_frac: float = 1.8,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Rule-based morphology corrections for instSeg / cellasic pipelines.

    Adds aspect_ratio (major_axis_length / minor_axis_length) to the returned
    DataFrame.  Requires skeleton_branch_points to be pre-computed by
    roi_skeleton_features in extra_properties (step 4.2) — raises ValueError
    if the column is absent.

    Args:
        fl_measurements: DataFrame containing at minimum object_class, area,
            major_axis_length, minor_axis_length, solidity, frame, label,
            skeleton_branch_points.
        spaghetti_ar: Aspect-ratio threshold for R1.  Default 5.0.
        spaghetti_solidity: Solidity ceiling for R1.  Default 0.72.
        lysis_solidity: Solidity ceiling for R2.  Default 0.45.
        lysis_area_frac: Area floor (× median) for R2.  Default 0.4.
        clump_branch_min: Min branch-point count for R3.  Default 1.
        clump_area_frac: Area floor (× median) for R3.  Default 1.8.

    Returns:
        Tuple of (updated DataFrame, correction counts dict).
        Counts keys: filamented_to_long, lysed_to_lyse, merged_to_clump.
    """
    fl = _add_aspect_ratio(fl_measurements)

    if "skeleton_branch_points" not in fl.columns:
        raise ValueError(
            "skeleton_branch_points column is missing from fl_measurements. "
            "Run feature extraction (get_roi_measurements with roi_skeleton_branch_points "
            "in extra_properties) before calling apply_instSeg_morphology_corrections."
        )

    # Reference area from single-cell labels only — clumps inflate the median
    sc_areas = fl.loc[fl["object_class"] == "single-cell", "area"]
    median_area = sc_areas.median() if len(sc_areas) > 0 else fl["area"].median()

    # R1: filamentation → long
    # High AR catches extreme elongation; low solidity ensures the shape is
    # bent/tangled rather than a straight stressed rod (which stays single-cell).
    # Also fires on model-classified debris that matches spaghetti geometry.
    r1 = (
        fl["object_class"].isin(["single-cell", "debris"])
        & (fl["aspect_ratio"] > spaghetti_ar)
        & (fl["solidity"] < spaghetti_solidity)
    )

    # R2: lysis / explosion → lyse
    # Very low solidity (fragmented boundary) + non-trivial area (not just noise).
    # Also fires on model-classified debris that matches explosion geometry.
    r2 = (
        fl["object_class"].isin(["single-cell", "clump", "debris"])
        & (fl["solidity"] < lysis_solidity)
        & (fl["area"] > lysis_area_frac * median_area)
    )

    # R3: merged detection → clump
    # Skeleton branch points flag a multi-cell cluster (the skeleton forks at
    # cell junctions).  The solidity floor (> 0.70) excludes fragmented masses
    # that also have many branch points but should be lyse not clump.
    # Note: end-to-end two-cell linear merges have 0 branch points and are
    # NOT caught by this rule — the model's own clump class handles them.
    r3 = (
        (fl["object_class"] == "single-cell")
        & (fl["skeleton_branch_points"] >= clump_branch_min)
        & (fl["area"] > clump_area_frac * median_area)
        & (fl["solidity"] > 0.70)
    )

    fl.loc[r3, "object_class"] = "clump"    # applied first
    fl.loc[r1, "object_class"] = "long"     # filamented cells
    fl.loc[r2 & ~r1, "object_class"] = "lyse"  # lysed/exploded cells

    counts: Dict[str, int] = {
        "filamented_to_long": int(r1.sum()),
        "lysed_to_lyse": int((r2 & ~r1).sum()),
        "merged_to_clump": int((r3 & ~r1 & ~r2).sum()),
    }
    return fl, counts


def apply_semSeg_morphology_corrections(
    fl_measurements: pd.DataFrame,
    area_clump_frac: float = 2.5,
    clump_solidity_max: float = 0.78,
    division_dist_frac: float = 1.0,
    division_angle_deg: float = 40.0,
    enable_division_detection: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Rule-based morphology corrections for semSeg pipeline (mycobacteria).

    Two rules targeting the specific failure modes of UNet + watershed on
    dense mycobacterial colonies:

    R1 — interior clump artifact → clump
         single-cell with area > area_clump_frac × median AND at least one of:
           • skeleton branch points ≥ 1 (complex internal shape)
           • solidity < clump_solidity_max (irregular boundary from merged cells)
         Catches interior cells of dense clumps where phase-contrast boundaries
         are invisible and the watershed merges multiple cells into one ROI.

    R2 — proximate aligned pair → joint-cell  (division detection)
         Pairs of single-cell ROIs in the same frame where:
           • centroid distance < division_dist_frac × median_major_axis_length
           • |orientation difference| < division_angle_deg  (undirected angle)
         Both ROIs are flagged as joint-cell.  Skipped when
         enable_division_detection=False (e.g., singleFrame pipeline).

    Adds aspect_ratio (major_axis_length / minor_axis_length) to the returned
    DataFrame.  Requires skeleton_branch_points to be pre-computed by
    roi_skeleton_features in extra_properties (step 4.2) — raises ValueError
    if the column is absent.

    Args:
        fl_measurements: DataFrame with object_class, area, major_axis_length,
            minor_axis_length, solidity, orientation, centroid_0, centroid_1,
            frame, label, skeleton_branch_points columns.
        area_clump_frac: Area threshold multiplier for R1.  Default 2.5.
        clump_solidity_max: Solidity ceiling for R1.  Default 0.78.
        division_dist_frac: Centroid distance threshold (× median major axis)
            for R2.  Default 1.0.
        division_angle_deg: Maximum orientation difference in degrees for R2.
            Default 40.0.
        enable_division_detection: Set False to skip R2 (singleFrame pipeline).

    Returns:
        Tuple of (updated DataFrame, correction counts dict).
        Counts keys: interior_to_clump, division_pairs_to_joint.
    """
    fl = _add_aspect_ratio(fl_measurements)

    if "skeleton_branch_points" not in fl.columns:
        raise ValueError(
            "skeleton_branch_points column is missing from fl_measurements. "
            "Run feature extraction (get_roi_measurements with roi_skeleton_branch_points "
            "in extra_properties) before calling apply_semSeg_morphology_corrections."
        )

    sc_mask = fl["object_class"] == "single-cell"
    sc_rows = fl.loc[sc_mask]
    median_area = sc_rows["area"].median() if len(sc_rows) > 0 else fl["area"].median()
    median_major = (
        sc_rows["major_axis_length"].median()
        if len(sc_rows) > 0
        else fl["major_axis_length"].median()
    )

    # R1: interior artifact → clump
    r1 = (
        sc_mask
        & (fl["area"] > area_clump_frac * median_area)
        & (
            (fl["skeleton_branch_points"] >= 1)
            | (fl["solidity"] < clump_solidity_max)
        )
    )
    fl.loc[r1, "object_class"] = "clump"

    # R2: proximate aligned pair → joint-cell (division detection)
    joint_indices: set = set()

    if enable_division_detection:
        if fl["frame"].nunique() == 1:
            warnings.warn(
                "apply_semSeg_morphology_corrections: division detection (R2) requires "
                "more than one frame but only 1 frame found — R2 skipped. "
                "If this is unexpected, check that your data has multiple time points.",
                RuntimeWarning,
                stacklevel=2,
            )
    if enable_division_detection and fl["frame"].nunique() > 1:
        dist_threshold = division_dist_frac * median_major
        angle_threshold = np.deg2rad(division_angle_deg)

        cx_col = "centroid_0" if "centroid_0" in fl.columns else None
        cy_col = "centroid_1" if "centroid_1" in fl.columns else None

        if cx_col is not None:
            for _, frame_group in fl.groupby("frame"):
                sc_frame = frame_group[frame_group["object_class"] == "single-cell"]
                if len(sc_frame) < 2:
                    continue

                centroids = sc_frame[[cx_col, cy_col]].values
                orientations = sc_frame["orientation"].values
                indices = sc_frame.index.tolist()

                dists = cdist(centroids, centroids)

                for i in range(len(sc_frame)):
                    for j in range(i + 1, len(sc_frame)):
                        if dists[i, j] >= dist_threshold:
                            continue
                        # Undirected angle between major axes in [-π/2, π/2]
                        diff = abs(orientations[i] - orientations[j])
                        angle_diff = min(diff, np.pi - diff)
                        if angle_diff < angle_threshold:
                            joint_indices.add(indices[i])
                            joint_indices.add(indices[j])

    if joint_indices:
        fl.loc[fl.index.isin(joint_indices), "object_class"] = "joint-cell"

    counts: Dict[str, int] = {
        "interior_to_clump": int(r1.sum()),
        "division_pairs_to_joint": len(joint_indices),
    }
    return fl, counts
