#!/usr/bin/env python3
"""
Build HDF5 training datasets from SyMBac_2 synthetic movies.

Reads masks.npz + lineage.json from each movie folder, extracts assignment-pair
and division-triplet features, and writes four HDF5 files:

  <output_dir>/
    train_assignments.h5    # 80% of movies
    val_assignments.h5      # 10% of movies
    test_assignments.h5     # 10% of movies
    train_divisions.h5
    val_divisions.h5
    test_divisions.h5
    split.json              # records which movies went to train/val/test

Usage:
    python build_dataset.py --data_dir /path/to/synthetic_data \
                            --output_dir /path/to/dataset \
                            --seed 42
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

# Allow running from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from HiTMicTools.tracking.feature_extraction import (
    extract_from_movie,
    PAIR_FEATURE_NAMES,
    TRIPLET_FEATURE_NAMES,
)


def load_movie(movie_dir: str):
    masks = np.load(os.path.join(movie_dir, "masks.npz"))["data"]
    with open(os.path.join(movie_dir, "lineage.json")) as f:
        lineage = json.load(f)
    return masks, lineage


def write_h5(path: str, pair_feats, pair_labels, triplet_feats, triplet_labels):
    with h5py.File(path, "w") as f:
        f.create_dataset("pair_features", data=pair_feats, compression="gzip")
        f.create_dataset("pair_labels", data=pair_labels, compression="gzip")
        f.create_dataset("triplet_features", data=triplet_feats, compression="gzip")
        f.create_dataset("triplet_labels", data=triplet_labels, compression="gzip")
        f.attrs["pair_feature_names"] = PAIR_FEATURE_NAMES
        f.attrs["triplet_feature_names"] = TRIPLET_FEATURE_NAMES
        f.attrs["n_pair_features"] = len(PAIR_FEATURE_NAMES)
        f.attrs["n_triplet_features"] = len(TRIPLET_FEATURE_NAMES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Root folder containing movie_000/, movie_001/, ...")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write the HDF5 files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_dist_norm", type=float, default=5.0,
                        help="Max centroid distance (× median cell length) to consider a pair")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Discover movies
    movie_dirs = sorted([
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if d.startswith("movie_") and os.path.isdir(os.path.join(args.data_dir, d))
    ])
    if not movie_dirs:
        print(f"ERROR: No movie_XXX folders found in {args.data_dir}")
        sys.exit(1)

    n = len(movie_dirs)
    print(f"Found {n} movies")

    # Movie-level split (never split on frames — that leaks temporal context)
    indices = rng.permutation(n)
    n_val = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.10))
    test_idx = set(indices[:n_test].tolist())
    val_idx = set(indices[n_test:n_test + n_val].tolist())
    train_idx = set(indices[n_test + n_val:].tolist())

    split_record = {
        "train": [movie_dirs[i] for i in sorted(train_idx)],
        "val": [movie_dirs[i] for i in sorted(val_idx)],
        "test": [movie_dirs[i] for i in sorted(test_idx)],
    }
    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump(split_record, f, indent=2)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    buckets = {"train": train_idx, "val": val_idx, "test": test_idx}
    accum = {
        split: {"pair_feats": [], "pair_labels": [],
                "triplet_feats": [], "triplet_labels": []}
        for split in buckets
    }

    for idx, movie_dir in enumerate(movie_dirs):
        split = next(s for s, ids in buckets.items() if idx in ids)

        try:
            masks, lineage = load_movie(movie_dir)
        except Exception as e:
            print(f"  SKIP {os.path.basename(movie_dir)}: {e}")
            continue

        result = extract_from_movie(masks, lineage, max_dist_norm=args.max_dist_norm)

        n_pos_pairs = int(result["pair_labels"].sum())
        n_neg_pairs = int((result["pair_labels"] == 0).sum())
        n_pos_div = int(result["triplet_labels"].sum())
        n_neg_div = int((result["triplet_labels"] == 0).sum())

        print(
            f"  [{split:5s}] {os.path.basename(movie_dir)}: "
            f"pairs {n_pos_pairs}+ / {n_neg_pairs}- | "
            f"divisions {n_pos_div}+ / {n_neg_div}-"
        )

        acc = accum[split]
        acc["pair_feats"].append(result["pair_feats"])
        acc["pair_labels"].append(result["pair_labels"])
        acc["triplet_feats"].append(result["triplet_feats"])
        acc["triplet_labels"].append(result["triplet_labels"])

    # Write HDF5 files and print summary
    print("\n--- Dataset summary ---")
    for split, acc in accum.items():
        pf = np.concatenate(acc["pair_feats"], axis=0) if acc["pair_feats"] else np.empty((0, 10), dtype=np.float32)
        pl = np.concatenate(acc["pair_labels"], axis=0) if acc["pair_labels"] else np.empty(0, dtype=np.int8)
        tf = np.concatenate(acc["triplet_feats"], axis=0) if acc["triplet_feats"] else np.empty((0, 8), dtype=np.float32)
        tl = np.concatenate(acc["triplet_labels"], axis=0) if acc["triplet_labels"] else np.empty(0, dtype=np.int8)

        out_path = os.path.join(args.output_dir, f"{split}_assignments.h5")
        write_h5(out_path, pf, pl, tf, tl)

        print(
            f"  {split}: {len(pl):6d} pairs ({pl.sum()} pos / {(pl==0).sum()} neg)  |  "
            f"{len(tl):5d} triplets ({tl.sum()} pos / {(tl==0).sum()} neg)"
        )
        print(f"    → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
