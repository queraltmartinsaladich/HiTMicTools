"""
DiagnosticWriter — step-by-step pipeline QC image export.

Activated by setting ``diagnostic_mode: true`` in the pipeline config.
Saves representative frames at each pipeline stage and generates a static
HTML report in  <output_path>/_diagnostics/<movie_name>/report.html.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Array utility helpers
# ---------------------------------------------------------------------------

def _to_numpy(arr) -> np.ndarray:
    """Convert CuPy / PyTorch tensor to numpy without copy when possible."""
    if arr is None:
        return None
    if hasattr(arr, "get"):            # CuPy ndarray
        return arr.get()
    if hasattr(arr, "cpu") and hasattr(arr, "numpy"):  # PyTorch tensor
        return arr.cpu().numpy()
    return np.asarray(arr)


def _frame_2d(arr: np.ndarray, t: int, s: int = 0, c: int = 0) -> np.ndarray:
    """Extract a single (H, W) frame from arrays with ndim 2–5."""
    if arr.ndim == 5:   # TSCXY
        return arr[t, s, c]
    if arr.ndim == 4:   # TSXY / TCXY
        return arr[t, s]
    if arr.ndim == 3:   # TXY
        return arr[t]
    if arr.ndim == 2:   # XY — same frame every call
        return arr
    raise ValueError(f"Cannot extract 2D frame from ndim={arr.ndim}")


def _mask_txy(mask_nd: np.ndarray) -> np.ndarray:
    """Reduce a mask array of any supported shape to (T, H, W)."""
    if mask_nd.ndim == 5:
        return mask_nd[:, 0, 0, :, :]
    if mask_nd.ndim == 4:
        return mask_nd[:, 0, :, :]
    if mask_nd.ndim == 3:
        return mask_nd
    raise ValueError(f"Unexpected mask ndim={mask_nd.ndim}")


def _norm_gray(frame: np.ndarray, plo: float = 1.0, phi: float = 99.0) -> np.ndarray:
    """Percentile-clip and return as float32 in [0, 1]."""
    lo, hi = np.percentile(frame, [plo, phi])
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.float32)
    return np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def _norm_gray_fl(frame: np.ndarray) -> np.ndarray:
    """FL-tuned normalization: p1→p99.9, preserving bright sparse signal on dark BG."""
    return _norm_gray(frame, plo=1.0, phi=99.9)


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Pure-Python HSV → RGB (avoids matplotlib import at module level)."""
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


def _label_hue_rgb(label_id: int, sat: float = 0.75, val: float = 0.92) -> Tuple[float, float, float]:
    """Deterministic, perceptually-spread color for an integer label."""
    hue = (label_id * 0.618033988749895) % 1.0
    return _hsv_to_rgb(hue, sat, val)


# ---------------------------------------------------------------------------
# DiagnosticWriter
# ---------------------------------------------------------------------------

class DiagnosticWriter:
    """
    Saves representative frames at key pipeline stages for visual QC.

    Each ``save_*`` call writes PNG files to a ``_diagnostics/<movie>/``
    subfolder and records a step entry.  Call ``generate_report()`` at the
    end of the pipeline to create a self-contained HTML index.

    Typical usage in a pipeline::

        diag = DiagnosticWriter(output_path, movie_name, nFrames) \\
               if getattr(self, "diagnostic_mode", False) else None

        if diag:
            diag.save_image_frames("01_raw", "1 — Raw image", ip.img,
                                   [("BF", reference_channel), ("FL", pi_channel)])
        ...
        if diag:
            diag.generate_report()
    """

    # Class-name → RGBA overlay color for instance segmentation
    CLASS_RGBA: Dict[str, Tuple[float, float, float, float]] = {
        "single-cell": (0.30, 0.60, 1.00, 0.50),
        "clump":       (1.00, 0.55, 0.00, 0.75),
        "debris":      (1.00, 0.20, 0.20, 0.65),
        "ghost":       (1.00, 0.00, 1.00, 0.80),
        "recovered":   (0.00, 1.00, 1.00, 0.80),
        "unknown":     (0.80, 0.80, 0.80, 0.50),
    }

    def __init__(
        self,
        output_dir: str,
        movie_name: str,
        n_frames: int,
        n_probe_frames: int = 5,
    ) -> None:
        self.root = Path(output_dir) / "_diagnostics" / movie_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.movie_name = movie_name
        self.probe_frames: List[int] = self._pick_frames(n_frames, n_probe_frames)
        self._steps: List[dict] = []

    # ------------------------------------------------------------------
    # Frame selection
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_frames(n: int, k: int) -> List[int]:
        if n <= k:
            return list(range(n))
        idxs = np.linspace(0, n - 1, k, dtype=int).tolist()
        return list(dict.fromkeys(idxs))   # deduplicate, preserve order

    # ------------------------------------------------------------------
    # Low-level PNG writers
    # ------------------------------------------------------------------

    def _write_gray(self, frame: np.ndarray, path: Path, fl: bool = False) -> None:
        import tifffile
        from PIL import Image as _PILImage

        # Primary: full-resolution float32 TIFF — lossless, original intensity scale
        tifffile.imwrite(
            str(path.with_suffix(".tiff")),
            frame.astype(np.float32),
            compression="deflate",
        )
        # Secondary: small uint8 PNG for report.html thumbnails (browsers cannot display TIFF)
        normed = _norm_gray_fl(frame) if fl else _norm_gray(frame)
        u8 = (normed * 255).astype(np.uint8)
        thumb = _PILImage.fromarray(u8, mode="L")
        if max(thumb.size) > 200:
            r = 200 / max(thumb.size)
            thumb = thumb.resize((int(thumb.width * r), int(thumb.height * r)),
                                 _PILImage.LANCZOS)
        thumb.save(str(path.with_suffix(".png")))

    def _write_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        path: Path,
        label_rgba: Optional[Dict[int, Tuple]] = None,
        default_alpha: float = 0.45,
        fl_background: bool = False,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        normed = _norm_gray_fl(image) if fl_background else _norm_gray(image)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.axis("off")
        ax.imshow(normed, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest", aspect="equal")

        if mask is not None and mask.max() > 0:
            H, W = mask.shape
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            if label_rgba:
                for lbl, color in label_rgba.items():
                    px = mask == lbl
                    if px.any():
                        rgba[px] = color
            else:
                for lbl in np.unique(mask[mask > 0]):
                    r, g, b = _label_hue_rgb(int(lbl))
                    rgba[mask == lbl] = (r, g, b, default_alpha)
            ax.imshow(rgba, interpolation="nearest", aspect="equal")

        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Public save methods
    # ------------------------------------------------------------------

    def save_image_frames(
        self,
        step_id: str,
        step_label: str,
        image_tscxy,
        channel_specs: List[Tuple[str, int]],
        slice_idx: int = 0,
    ) -> None:
        """Save grayscale probe frames for one or more image channels."""
        img = _to_numpy(image_tscxy)
        entries = []
        for ch_name, ch_idx in channel_specs:
            is_fl = ch_name.upper() == "FL"
            files = []
            for f in self.probe_frames:
                frame = _frame_2d(img, f, slice_idx, ch_idx)
                fname = f"{step_id}_{ch_name}_f{f:03d}.png"
                self._write_gray(frame, self.root / fname, fl=is_fl)
                files.append(fname)
            entries.append({"channel": ch_name, "files": files})
        self._steps.append({"step_id": step_id, "label": step_label, "entries": entries})

    def _write_class_json(
        self,
        mask: np.ndarray,
        all_class_ids_frame: Optional[list],
        class_dict: Optional[Dict[int, str]],
        path: Path,
    ) -> None:
        """JSON sidecar: normalized centroid lists per class, for the interactive viewer."""
        import json
        from skimage.measure import regionprops

        H, W = mask.shape
        label_to_class: Dict[int, str] = {}
        if all_class_ids_frame is not None and class_dict is not None:
            for idx, cid in enumerate(all_class_ids_frame):
                label_to_class[idx + 1] = class_dict.get(int(cid), "unknown")

        by_class: Dict[str, list] = {}
        for prop in regionprops(mask):
            cls = label_to_class.get(prop.label, "unknown")
            cy, cx = prop.centroid
            by_class.setdefault(cls, []).append([round(cx / W, 4), round(cy / H, 4)])

        colors = {
            cls: [round(v, 3) for v in self.CLASS_RGBA.get(cls, self.CLASS_RGBA["unknown"])]
            for cls in by_class
        }
        with open(path, "w") as fh:
            json.dump({
                "cells": by_class,
                "colors": colors,
                "label_to_class": {str(k): v for k, v in label_to_class.items()},
            }, fh, separators=(",", ":"))

    def save_segmentation_overlay(
        self,
        step_id: str,
        step_label: str,
        image_tscxy,
        labeled_mask,
        all_class_ids: Optional[List] = None,
        class_dict: Optional[Dict[int, str]] = None,
        image_channel: int = 0,
        image_slice: int = 0,
        fl_background: bool = False,
    ) -> None:
        """Save image + mask overlay, cells colored by detected class.

        If ``all_class_ids`` / ``class_dict`` are not provided every cell is
        assigned a random hue (same as ``save_mask_overlay``).
        Set ``fl_background=True`` when the background channel is fluorescence
        (applies p1→p99.9 normalization instead of the default p1→p99).
        Also writes a ``{step_id}_classes_f*.json`` sidecar with per-class
        centroid data for interactive class filtering in gen_diag_viewer.py.
        """
        img = _to_numpy(image_tscxy)
        mask = _mask_txy(_to_numpy(labeled_mask))

        files = []
        for f in self.probe_frames:
            frame_img  = _frame_2d(img, f, image_slice, image_channel)
            frame_mask = mask[f] if f < mask.shape[0] else mask[-1]

            label_rgba: Optional[Dict[int, Tuple]] = None
            cids_frame = all_class_ids[f] if (all_class_ids is not None and f < len(all_class_ids)) else None
            if cids_frame is not None and class_dict is not None:
                label_rgba = {}
                for idx, cid in enumerate(cids_frame):
                    cls = class_dict.get(int(cid), "unknown")
                    label_rgba[idx + 1] = self.CLASS_RGBA.get(cls, self.CLASS_RGBA["unknown"])

            fname = f"{step_id}_overlay_f{f:03d}.png"
            self._write_overlay(frame_img, frame_mask, self.root / fname,
                                 label_rgba=label_rgba, fl_background=fl_background)
            files.append(fname)

            # uint16 instance label TIFF — exact pixel labels, full resolution
            import tifffile as _tifffile
            _tifffile.imwrite(
                str(self.root / f"{step_id}_labels_f{f:03d}.tiff"),
                frame_mask.astype(np.uint16),
                compression="deflate",
            )

            # JSON sidecar for interactive class filter in viewer
            self._write_class_json(
                frame_mask, cids_frame, class_dict,
                self.root / f"{step_id}_classes_f{f:03d}.json",
            )

        self._steps.append({
            "step_id": step_id,
            "label": step_label,
            "entries": [{"channel": "overlay (class)", "files": files}],
        })

    def save_mask_overlay(
        self,
        step_id: str,
        step_label: str,
        image_tscxy,
        labeled_mask,
        image_channel: int = 0,
        image_slice: int = 0,
        label_rgba: Optional[Dict[int, Tuple]] = None,
        alpha: float = 0.45,
    ) -> None:
        """Save BF + mask overlay with random per-label hue colors."""
        img  = _to_numpy(image_tscxy)
        mask = _mask_txy(_to_numpy(labeled_mask))

        files = []
        for f in self.probe_frames:
            frame_img  = _frame_2d(img, f, image_slice, image_channel)
            frame_mask = mask[f] if f < mask.shape[0] else mask[-1]
            fname = f"{step_id}_overlay_f{f:03d}.png"
            self._write_overlay(frame_img, frame_mask, self.root / fname,
                                 label_rgba=label_rgba, default_alpha=alpha)
            files.append(fname)

            # uint16 instance label TIFF — exact pixel labels, full resolution
            import tifffile as _tifffile
            _tifffile.imwrite(
                str(self.root / f"{step_id}_labels_f{f:03d}.tiff"),
                frame_mask.astype(np.uint16),
                compression="deflate",
            )

        self._steps.append({
            "step_id": step_id,
            "label": step_label,
            "entries": [{"channel": "overlay", "files": files}],
        })

    def save_tracking_overlay(
        self,
        step_id: str,
        step_label: str,
        image_tscxy,
        labeled_mask,
        fl_measurements,
        image_channel: int = 0,
        image_slice: int = 0,
    ) -> None:
        """Save overlay colored by track ID; ghost=magenta, recovered=cyan."""
        img  = _to_numpy(image_tscxy)
        mask = _mask_txy(_to_numpy(labeled_mask))

        files = []
        for f in self.probe_frames:
            frame_img  = _frame_2d(img, f, image_slice, image_channel)
            frame_mask = mask[f] if f < mask.shape[0] else mask[-1]

            frame_df = fl_measurements[fl_measurements["frame"] == f]
            label_rgba: Dict[int, Tuple] = {}
            for _, row in frame_df.iterrows():
                lbl = int(row["label"])
                obj_cls = str(row.get("object_class", ""))
                if obj_cls == "ghost":
                    label_rgba[lbl] = (1.0, 0.0, 1.0, 0.75)
                elif obj_cls == "recovered":
                    label_rgba[lbl] = (0.0, 1.0, 1.0, 0.80)
                else:
                    tid = int(row.get("trackid", lbl))
                    r, g, b = _label_hue_rgb(tid, sat=0.75, val=0.92)
                    label_rgba[lbl] = (r, g, b, 0.55)

            fname = f"{step_id}_tracking_f{f:03d}.png"
            self._write_overlay(frame_img, frame_mask, self.root / fname,
                                 label_rgba=label_rgba or None)
            files.append(fname)

        self._steps.append({
            "step_id": step_id,
            "label": step_label,
            "entries": [{"channel": "tracking", "files": files}],
        })

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(self) -> Path:
        """Write a static HTML QC report; return its path."""
        css = """
    body { font-family: 'Courier New', monospace; background: #0d0d0d; color: #d8d8d8;
           margin: 0; padding: 28px 32px; }
    h1   { color: #7ec8e3; font-size: 1.35em; margin: 0 0 4px; }
    .meta { color: #555; font-size: 0.78em; margin-bottom: 36px; }
    .step { background: #141414; border: 1px solid #252525; border-radius: 6px;
            padding: 18px 20px; margin-bottom: 32px; }
    .step h2 { color: #a8d8a8; font-size: 1.05em; margin: 0 0 14px;
               border-bottom: 1px solid #252525; padding-bottom: 8px; }
    .ch-label { color: #666; font-size: 0.72em; text-transform: uppercase;
                letter-spacing: 1px; margin: 12px 0 6px; }
    .frames { display: flex; gap: 10px; flex-wrap: wrap; }
    .thumb  { display: flex; flex-direction: column; align-items: center; }
    .thumb img { width: 200px; height: 200px; object-fit: contain;
                 background: #000; border: 1px solid #222; display: block; }
    .thumb .fl  { color: #444; font-size: 0.68em; margin-top: 3px; }
    a { color: inherit; text-decoration: none; }
    a:hover img { border-color: #7ec8e3; }"""

        lines = [
            "<!DOCTYPE html>", "<html lang='en'>", "<head>",
            f"  <title>QC — {self.movie_name}</title>",
            "  <meta charset='UTF-8'>",
            f"  <style>{css}</style>",
            "</head>", "<body>",
            f"  <h1>Pipeline QC &mdash; {self.movie_name}</h1>",
            f"  <p class='meta'>Probe frames: {self.probe_frames}"
            f"  &nbsp;|&nbsp; {len(self._steps)} steps recorded"
            f"  &nbsp;|&nbsp; {self.root}</p>",
        ]

        for step in self._steps:
            lines += [
                "  <div class='step'>",
                f"    <h2>{step['label']}</h2>",
            ]
            for entry in step["entries"]:
                lines.append(f"    <p class='ch-label'>{entry['channel']}</p>")
                lines.append("    <div class='frames'>")
                for i, fname in enumerate(entry["files"]):
                    f_idx = self.probe_frames[i] if i < len(self.probe_frames) else "?"
                    lines += [
                        "      <div class='thumb'>",
                        f"        <a href='{fname}' target='_blank'>",
                        f"          <img src='{fname}' alt='f{f_idx}' loading='lazy'>",
                        "        </a>",
                        f"        <span class='fl'>frame {f_idx}</span>",
                        "      </div>",
                    ]
                lines.append("    </div>")
            lines.append("  </div>")

        lines += ["</body>", "</html>"]
        report_path = self.root / "report.html"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path
