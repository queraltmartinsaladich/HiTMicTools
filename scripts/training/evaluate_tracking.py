#!/usr/bin/env python3
"""
evaluate_tracking.py — compute tracking metrics on held-out movies.

Compares HungarianTracker predictions (optionally with a trained AssignmentScorer)
against ground-truth tracks derived from SyMBac_2 label continuity + lineage.json.

Metrics reported:
  Link F1 / precision / recall  — primary: how well are frame-to-frame links made
  ID switches / MOTA            — how often does a track flip identity
  IDF1                          — global identity-based accuracy
  MT / PT / ML                  — mostly-tracked / partial / mostly-lost GT tracks

Since the tracker runs on the GT masks, detection accuracy is perfect by
construction. All errors are linking errors (missed links, wrong links, ID
switches), which is exactly what the AssignmentScorer is trained to improve.

Usage:
    # Test split written by build_dataset.py (recommended):
    python evaluate_tracking.py \\
        --split_file /path/to/dataset/split.json --split test

    # All movies in a folder (quick sanity check):
    python evaluate_tracking.py --data_dir /path/to/movies

    # With a trained assignment scorer:
    python evaluate_tracking.py \\
        --split_file /path/to/dataset/split.json --split test \\
        --model_path /path/to/assignment_scorer.pt

    # Compare baseline vs. learned side-by-side:
    python evaluate_tracking.py --split_file split.json --split test \\
        --output baseline.json
    python evaluate_tracking.py --split_file split.json --split test \\
        --model_path assignment_scorer.pt --output learned.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from skimage.measure import regionprops

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from HiTMicTools.tracking.hungarian_tracker import HungarianTracker


# ── data loading ──────────────────────────────────────────────────────────────

def load_movie(movie_dir: str):
    masks = np.load(os.path.join(movie_dir, "masks.npz"))["data"]
    with open(os.path.join(movie_dir, "lineage.json")) as f:
        lineage = json.load(f)
    return masks, lineage


def extract_measurements(masks: np.ndarray) -> pd.DataFrame:
    """Extract per-cell rows from mask array for tracker input."""
    rows = []
    for t, frame in enumerate(masks):
        for rp in regionprops(frame):
            cy, cx = rp.centroid
            rows.append({
                "frame": t,
                "label": rp.label,
                "centroid_0": cy,
                "centroid_1": cx,
                "area": rp.area,
                "solidity": rp.solidity,
                "major_axis_length": rp.major_axis_length,
                "minor_axis_length": rp.minor_axis_length,
                "orientation": rp.orientation,
            })
    if not rows:
        return pd.DataFrame(
            columns=["frame", "label", "centroid_0", "centroid_1",
                     "area", "solidity", "major_axis_length",
                     "minor_axis_length", "orientation"]
        )
    return pd.DataFrame(rows)


# ── ground-truth track assignment ─────────────────────────────────────────────

def build_gt_tracks(masks: np.ndarray, lineage: dict):
    """
    Assign GT track IDs from label continuity + recorded division events.

    A cell at (t, label) continues its existing track when:
      - the same label appears at (t-1), AND
      - it is NOT listed as a daughter in lineage[str(t)].

    Division daughters and genuinely new labels start a fresh track ID.
    Division links are intentionally excluded from gt_links so that a tracker
    which links across a division is penalised (the split is the DivisionClassifier's
    job, not the assignment model's).

    Returns:
        gt_det_to_track : dict  (frame, label) -> gt_track_id
        gt_links        : set   of (t, label_t, t+1, label_t1) tuples
    """
    T = masks.shape[0]
    gt_det_to_track: dict = {}
    next_tid = 0

    for lbl in np.unique(masks[0]):
        if lbl == 0:
            continue
        gt_det_to_track[(0, int(lbl))] = next_tid
        next_tid += 1

    gt_links: set = set()

    for t in range(1, T):
        daughters = {}
        if str(t) in lineage:
            for d_str, p in lineage[str(t)].items():
                daughters[int(d_str)] = int(p)

        prev_labels = {int(l) for l in np.unique(masks[t - 1]) if l > 0}

        for lbl in np.unique(masks[t]):
            if lbl == 0:
                continue
            lbl = int(lbl)

            if lbl not in daughters and lbl in prev_labels:
                parent_tid = gt_det_to_track.get((t - 1, lbl))
                if parent_tid is not None:
                    gt_det_to_track[(t, lbl)] = parent_tid
                    gt_links.add((t - 1, lbl, t, lbl))
                    continue

            # New cell: daughter from a division or a genuinely new label
            gt_det_to_track[(t, lbl)] = next_tid
            next_tid += 1

    return gt_det_to_track, gt_links


# ── tracker execution ─────────────────────────────────────────────────────────

def run_tracker(masks, df, tracker, assignment_scorer=None, learned_max_cost=0.5):
    """
    Run tracker and extract predicted tracks + links.

    Returns:
        pred_det_to_track : dict  (frame, label) -> pred_track_id
        pred_links        : set   of (t, label_t, t+1, label_t1)
    """
    cost_overrides = None
    if assignment_scorer is not None:
        from HiTMicTools.tracking.feature_extraction import compute_movie_stats
        stats = compute_movie_stats(masks)
        sorted_frames = sorted(df["frame"].unique())
        cost_overrides = {}
        for fi, fval in enumerate(sorted_frames[:-1]):
            fval_next = sorted_frames[fi + 1]
            cm, lt, lt1 = assignment_scorer.predict_cost_matrix(
                masks[fval], masks[fval_next], stats,
                masks_t_prev=masks[fval - 1] if fval > 0 else None,
            )
            cost_overrides[fval] = (cm, lt, lt1)

    result_df = tracker.track_objects(
        df.copy(),
        cost_overrides=cost_overrides,
        learned_max_cost=learned_max_cost,
    )

    pred_det_to_track = {
        (int(r["frame"]), int(r["label"])): int(r["trackid"])
        for _, r in result_df.iterrows()
    }

    # Build predicted link set: same trackid in consecutive frames
    pred_links: set = set()
    sorted_frames = sorted(result_df["frame"].unique())
    for fi in range(len(sorted_frames) - 1):
        f1, f2 = sorted_frames[fi], sorted_frames[fi + 1]
        df1 = result_df[result_df["frame"] == f1]
        df2 = result_df[result_df["frame"] == f2]
        # Build trackid → label maps (trackid is unique per frame in valid output)
        tid_lbl_1 = dict(zip(df1["trackid"].astype(int), df1["label"].astype(int)))
        tid_lbl_2 = dict(zip(df2["trackid"].astype(int), df2["label"].astype(int)))
        for tid in set(tid_lbl_1) & set(tid_lbl_2):
            if tid < 0:
                continue
            pred_links.add((f1, tid_lbl_1[tid], f2, tid_lbl_2[tid]))

    return pred_det_to_track, pred_links


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(gt_det_to_track, gt_links, pred_det_to_track, pred_links):
    """
    Compute per-movie tracking metrics.

    Detection accuracy is perfect by construction (tracker runs on GT masks).
    All error sources are linking errors.
    """
    # ── link-level ────────────────────────────────────────────────────────────
    tp = len(gt_links & pred_links)
    fp = len(pred_links - gt_links)
    fn = len(gt_links - pred_links)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    # ── ID switches ───────────────────────────────────────────────────────────
    # An ID switch occurs when the predicted track ID assigned to a GT cell
    # changes from one frame to the next (excluding frames where it was -1).
    gt_track_frames: dict = {}
    for (frame, lbl), gt_tid in gt_det_to_track.items():
        gt_track_frames.setdefault(gt_tid, {})[frame] = \
            pred_det_to_track.get((frame, lbl), -1)

    idsw = 0
    for _, frame_map in gt_track_frames.items():
        prev = None
        for f in sorted(frame_map):
            curr = frame_map[f]
            if prev is not None and curr >= 0 and curr != prev:
                idsw += 1
            if curr >= 0:
                prev = curr

    n_gt = len(gt_det_to_track)
    mota = 1.0 - idsw / max(n_gt, 1)

    # ── GT track lengths and per-track best overlap ───────────────────────────
    gt_lengths: dict = {}
    pred_lengths: dict = {}
    overlap: dict = {}  # (pred_tid, gt_tid) -> count

    for (frame, lbl), gt_tid in gt_det_to_track.items():
        gt_lengths[gt_tid] = gt_lengths.get(gt_tid, 0) + 1
        pred_tid = pred_det_to_track.get((frame, lbl), -1)
        if pred_tid >= 0:
            pred_lengths[pred_tid] = pred_lengths.get(pred_tid, 0) + 1
            key = (pred_tid, gt_tid)
            overlap[key] = overlap.get(key, 0) + 1

    # Greedy one-to-one matching: best pred per GT, then best GT per pred
    pred_to_best_gt: dict = {}
    for (pred_tid, gt_tid), cnt in overlap.items():
        if pred_tid not in pred_to_best_gt or cnt > pred_to_best_gt[pred_tid][1]:
            pred_to_best_gt[pred_tid] = (gt_tid, cnt)

    gt_to_best_pred: dict = {}
    for pred_tid, (gt_tid, cnt) in pred_to_best_gt.items():
        if gt_tid not in gt_to_best_pred or cnt > gt_to_best_pred[gt_tid][1]:
            gt_to_best_pred[gt_tid] = (pred_tid, cnt)

    # ── IDF1 ─────────────────────────────────────────────────────────────────
    tp_id = fp_id = fn_id = 0
    for gt_tid, (pred_tid, cnt) in gt_to_best_pred.items():
        tp_id += cnt
        fp_id += pred_lengths.get(pred_tid, 0) - cnt
        fn_id += gt_lengths[gt_tid] - cnt
    for gt_tid, gl in gt_lengths.items():
        if gt_tid not in gt_to_best_pred:
            fn_id += gl

    idf1 = 2 * tp_id / max(2 * tp_id + fp_id + fn_id, 1)

    # ── MT / PT / ML ─────────────────────────────────────────────────────────
    mt = pt = ml = 0
    for gt_tid, gl in gt_lengths.items():
        best_ov = gt_to_best_pred.get(gt_tid, (None, 0))[1]
        frac = best_ov / max(gl, 1)
        if frac > 0.8:
            mt += 1
        elif frac >= 0.2:
            pt += 1
        else:
            ml += 1

    n_gt_tracks = len(gt_lengths)

    return {
        "n_gt_detections":   n_gt,
        "n_gt_tracks":       n_gt_tracks,
        "tp_links":          tp,
        "fp_links":          fp,
        "fn_links":          fn,
        "link_precision":    prec,
        "link_recall":       rec,
        "link_f1":           f1,
        "id_switches":       idsw,
        "mota":              mota,
        "idf1":              idf1,
        "mostly_tracked":    mt,
        "partially_tracked": pt,
        "mostly_lost":       ml,
        "mt_frac":           mt / max(n_gt_tracks, 1),
        "ml_frac":           ml / max(n_gt_tracks, 1),
    }


def aggregate(metrics_list):
    """
    Micro-average count-based metrics (sum counts, recompute rates).
    Macro-average IDF1 (mean across movies — global IDF1 needs cross-movie matching).
    """
    if not metrics_list:
        return {}

    count_keys = {
        "n_gt_detections", "n_gt_tracks", "tp_links", "fp_links", "fn_links",
        "id_switches", "mostly_tracked", "partially_tracked", "mostly_lost",
    }

    agg = {}
    for k in metrics_list[0]:
        vals = [m[k] for m in metrics_list]
        agg[k] = sum(vals) if k in count_keys else float(np.mean(vals))

    # Recompute rate metrics from aggregate counts (micro-average)
    tp, fp, fn = agg["tp_links"], agg["fp_links"], agg["fn_links"]
    agg["link_precision"] = tp / max(tp + fp, 1)
    agg["link_recall"]    = tp / max(tp + fn, 1)
    p, r = agg["link_precision"], agg["link_recall"]
    agg["link_f1"]  = 2 * p * r / max(p + r, 1e-9)
    agg["mota"]     = 1.0 - agg["id_switches"] / max(agg["n_gt_detections"], 1)
    agg["mt_frac"]  = agg["mostly_tracked"] / max(agg["n_gt_tracks"], 1)
    agg["ml_frac"]  = agg["mostly_lost"]    / max(agg["n_gt_tracks"], 1)
    return agg


def _print_table(name: str, m: dict) -> None:
    w = 58
    print(f"\n{'─' * w}")
    print(f"  {name}")
    print(f"{'─' * w}")
    print(f"  GT tracks      : {m['n_gt_tracks']:5d}    GT detections : {m['n_gt_detections']:7d}")
    print(f"  TP/FP/FN links : {m['tp_links']:5d} / {m['fp_links']:5d} / {m['fn_links']:5d}")
    print(f"  Link F1        : {m['link_f1']:.4f}   "
          f"(P={m['link_precision']:.4f}, R={m['link_recall']:.4f})")
    print(f"  ID switches    : {m['id_switches']:5d}")
    print(f"  MOTA           : {m['mota']:.4f}")
    print(f"  IDF1           : {m['idf1']:.4f}")
    print(f"  MT / PT / ML   : {m['mostly_tracked']} / {m['partially_tracked']} / {m['mostly_lost']}"
          f"   ({100*m['mt_frac']:.1f}% MT, {100*m['ml_frac']:.1f}% ML)")
    print(f"{'─' * w}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hungarian tracker (± AssignmentScorer) on synthetic movies."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split_file", metavar="PATH",
                        help="split.json written by build_dataset.py")
    source.add_argument("--data_dir", metavar="PATH",
                        help="folder containing movie_XXX sub-folders (evaluates all)")

    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "all"],
                        help="which split to evaluate (only with --split_file; default: test)")
    parser.add_argument("--model_path", metavar="PATH", default=None,
                        help="path to assignment_scorer.pt — omit for Euclidean baseline")
    parser.add_argument("--max_distance", type=float, default=25.0,
                        help="max linking distance in pixels (default 25.0)")
    parser.add_argument("--gap_bridge_frames", type=int, default=2,
                        help="gap bridging window (default 2)")
    parser.add_argument("--learned_max_cost", type=float, default=0.5,
                        help="reject learned-cost links above this value (default 0.5 = P<50%%)")
    parser.add_argument("--verbose", action="store_true",
                        help="print a full metric table for every movie")
    parser.add_argument("--output", metavar="PATH", default=None,
                        help="write aggregate + per-movie results to this JSON file")
    args = parser.parse_args()

    # ── resolve movie list ────────────────────────────────────────────────────
    if args.split_file:
        with open(args.split_file) as f:
            split_data = json.load(f)
        if args.split == "all":
            movie_dirs = (split_data["train"] + split_data["val"]
                          + split_data["test"])
        else:
            movie_dirs = split_data[args.split]
        print(f"Split [{args.split}]: {len(movie_dirs)} movies  "
              f"(from {args.split_file})")
    else:
        movie_dirs = sorted([
            os.path.join(args.data_dir, d)
            for d in os.listdir(args.data_dir)
            if d.startswith("movie_")
            and os.path.isdir(os.path.join(args.data_dir, d))
        ])
        print(f"Data dir: {args.data_dir}  ({len(movie_dirs)} movies found)")

    if not movie_dirs:
        print("ERROR: no movies found.")
        sys.exit(1)

    # ── load scorer (optional) ────────────────────────────────────────────────
    assignment_scorer = None
    if args.model_path:
        from HiTMicTools.tracking.assignment_scorer import AssignmentScorer
        assignment_scorer = AssignmentScorer(args.model_path)
        print(f"AssignmentScorer : {args.model_path}")
        print(f"learned_max_cost : {args.learned_max_cost}")
    else:
        print("Mode             : Euclidean + appearance baseline (no model_path)")

    # ── build tracker ─────────────────────────────────────────────────────────
    tracker = HungarianTracker(
        max_distance=args.max_distance,
        gap_bridge_frames=args.gap_bridge_frames,
    )
    tracker.set_features(
        ["area", "major_axis_length", "minor_axis_length", "solidity", "orientation"]
    )
    print(f"Tracker          : max_distance={args.max_distance} px, "
          f"gap_bridge_frames={args.gap_bridge_frames}")
    print()

    # ── evaluate ──────────────────────────────────────────────────────────────
    all_metrics = []
    per_movie_records = []

    for movie_dir in movie_dirs:
        name = os.path.basename(movie_dir)

        try:
            masks, lineage = load_movie(movie_dir)
        except Exception as e:
            print(f"  SKIP {name}: load error — {e}")
            continue

        df = extract_measurements(masks)
        if df.empty:
            print(f"  SKIP {name}: no detections in masks")
            continue

        gt_det_to_track, gt_links = build_gt_tracks(masks, lineage)

        if not gt_links:
            print(f"  SKIP {name}: no GT links (movie too short or no persistent cells)")
            continue

        try:
            pred_det_to_track, pred_links = run_tracker(
                masks, df, tracker,
                assignment_scorer=assignment_scorer,
                learned_max_cost=args.learned_max_cost,
            )
        except Exception as e:
            print(f"  SKIP {name}: tracker error — {e}")
            continue

        m = compute_metrics(gt_det_to_track, gt_links, pred_det_to_track, pred_links)
        all_metrics.append(m)
        per_movie_records.append({"movie": movie_dir, **m})

        if args.verbose:
            _print_table(name, m)
        else:
            print(
                f"  {name:20s}  "
                f"LinkF1={m['link_f1']:.3f}  "
                f"MOTA={m['mota']:.3f}  "
                f"IDF1={m['idf1']:.3f}  "
                f"MT={100*m['mt_frac']:.0f}%  "
                f"IDSW={m['id_switches']}"
            )

    if not all_metrics:
        print("\nNo movies evaluated successfully.")
        sys.exit(1)

    agg = aggregate(all_metrics)
    _print_table(f"AGGREGATE  ({len(all_metrics)} / {len(movie_dirs)} movies)", agg)

    if args.output:
        out_data = {
            "config": {
                "model_path":        args.model_path,
                "max_distance":      args.max_distance,
                "gap_bridge_frames": args.gap_bridge_frames,
                "learned_max_cost":  args.learned_max_cost,
                "split":             args.split,
                "n_movies":          len(all_metrics),
            },
            "aggregate": agg,
            "per_movie": per_movie_records,
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
