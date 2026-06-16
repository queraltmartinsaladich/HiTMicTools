"""Quick verification test for HungarianTracker."""

import numpy as np
import pandas as pd
from HiTMicTools.tracking.hungarian_tracker import HungarianTracker
from HiTMicTools.tracking.hungarian_tracker_v5 import HungarianTrackerV5


def test_tracking_and_lockin():
    # 5 cells, 3 frames, slight movement (2px jitter)
    np.random.seed(42)
    records = []
    base_positions = np.array(
        [[100, 100], [200, 200], [300, 300], [400, 400], [500, 500]]
    )
    for frame in range(3):
        for i, (y, x) in enumerate(base_positions):
            noise = np.random.randn(2) * 2
            records.append(
                {
                    "frame": frame,
                    "label": i + 1,
                    "centroid_0": y + noise[0],
                    "centroid_1": x + noise[1],
                    "pi_class": "piNEG",
                }
            )

    df = pd.DataFrame(records)

    # Cell 2 dies in frame 1, flickers back to piNEG in frame 2
    df.loc[(df["frame"] == 1) & (df["label"] == 2), "pi_class"] = "piPOS"
    df.loc[(df["frame"] == 2) & (df["label"] == 2), "pi_class"] = "piNEG"

    tracker = HungarianTracker(max_distance=25.0)
    df = tracker.track_objects(df)

    # All 5 cells should get consistent track IDs across frames
    print("Track IDs per frame:")
    for f in range(3):
        ids = df.loc[df["frame"] == f, "trackid"].tolist()
        print(f"  Frame {f}: {ids}")

    # Each cell should have exactly 1 unique trackid across all 3 frames
    for label in range(1, 6):
        track_ids = df.loc[df["label"] == label, "trackid"].unique()
        assert len(track_ids) == 1, f"Cell {label} has multiple track IDs: {track_ids}"
    print("PASS: All cells have consistent track IDs")

    # Test lock-in
    df = tracker.apply_pipos_lockin(df)
    cell2_track = df.loc[(df["frame"] == 0) & (df["label"] == 2), "trackid"].iloc[0]
    cell2_classes = df.loc[df["trackid"] == cell2_track, "pi_class"].tolist()
    assert cell2_classes == [
        "piNEG",
        "piPOS",
        "piPOS",
    ], f"Lock-in failed: {cell2_classes}"
    print(f"PASS: Cell 2 lock-in correct: {cell2_classes}")

    # Other cells should remain piNEG
    for label in [1, 3, 4, 5]:
        tid = df.loc[(df["frame"] == 0) & (df["label"] == label), "trackid"].iloc[0]
        classes = df.loc[df["trackid"] == tid, "pi_class"].tolist()
        assert all(c == "piNEG" for c in classes), f"Cell {label} should be all piNEG"
    print("PASS: Other cells remain piNEG")

    print("\nAll tests passed!")


def test_v5_stitches_temporally_disjoint_tracks():
    df = pd.DataFrame(
        [
            {"frame": 0, "centroid_0": 10.0, "centroid_1": 10.0, "trackid": 1},
            {"frame": 2, "centroid_0": 11.0, "centroid_1": 10.5, "trackid": 2},
            {"frame": 2, "centroid_0": 10.8, "centroid_1": 10.2, "trackid": 3},
        ]
    )

    tracker = HungarianTrackerV5()
    stitched = tracker.stitch_tracks(df, max_stitch_distance=8.0)

    assert stitched.loc[stitched["frame"] == 0, "trackid"].iloc[0] == 1
    assert stitched.loc[stitched["trackid"] == 1, "frame"].tolist() == [0, 2]
    assert stitched.groupby(["trackid", "frame"]).size().max() == 1


def test_v5_track_objects_auto_stitches_before_lockin():
    df = pd.DataFrame(
        [
            {
                "frame": 0,
                "centroid_0": 10.0,
                "centroid_1": 10.0,
                "area": 50.0,
                "pi_class": "piNEG",
            },
            {
                "frame": 2,
                "centroid_0": 11.0,
                "centroid_1": 10.5,
                "area": 50.0,
                "pi_class": "piPOS",
            },
            {
                "frame": 20,
                "centroid_0": 11.5,
                "centroid_1": 10.8,
                "area": 50.0,
                "pi_class": "piNEG",
            },
        ]
    )

    tracker = HungarianTrackerV5(reid_max_gap=1, stitch_max_distance=8.0)
    tracked = tracker.track_objects(df)

    assert tracked["trackid"].nunique() == 1
    locked = tracker.apply_pipos_lockin(tracked)
    assert locked["pi_class"].tolist() == ["piNEG", "piPOS", "piPOS"]


if __name__ == "__main__":
    test_tracking_and_lockin()
