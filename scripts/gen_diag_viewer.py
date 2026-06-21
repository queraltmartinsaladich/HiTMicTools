#!/usr/bin/env python3
"""
gen_diag_viewer.py — HTML viewer for HiTMicTools diagnostic output.

Reads the PNG files written by DiagnosticWriter (diagnostic_mode: true) and
produces a self-contained HTML page with a 6-panel layout per probe frame.

Usage
-----
    python3 scripts/gen_diag_viewer.py <diag_dir> [--out PATH]

    <diag_dir>   Path to _diagnostics/<movie_name>/ directory.
    --out        Output HTML path.  Default: <diag_dir>/viewer.html

Examples
--------
    # Single movie
    python3 scripts/gen_diag_viewer.py \\
        /scicore/projects/rinfsci/Queralt/EColi/e011_ecoli_results_diag/_diagnostics/20260416_SCZ_e011_M10_p01

    # Explicit output path
    python3 scripts/gen_diag_viewer.py \\
        /path/to/_diagnostics/my_movie \\
        --out ~/viewers/my_movie_diag.html
"""
import argparse
import base64
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Panel definitions — ordered as they should appear in the viewer.
# Each entry: (column_label, glob_pattern_suffix)
# The pattern suffix is appended after the step_id prefix, e.g.:
#   step_id="01_raw", suffix="_BF_f{frame}.png" → "01_raw_BF_f000.png"
# ---------------------------------------------------------------------------
PANEL_DEFS = [
    ("BF raw",       "01_raw_BF_f{frame}.png"),
    ("BF prep",      "02_preprocessed_BF_f{frame}.png"),
    ("Seg overlay",  "03_segmentation_overlay_f{frame}.png"),
    ("FL raw",       "01_raw_FL_f{frame}.png"),
    ("FL prep",      "02_preprocessed_FL_f{frame}.png"),
    ("Mask refined", "04_mask_corrected_overlay_f{frame}.png"),
]

LEGEND = [
    ("BF raw",       "Brightfield before any correction"),
    ("BF prep",      "After align + BaSiCPy + focus + species preprocessing"),
    ("Seg overlay",  "RF-DETR instance segmentation (colored by class)"),
    ("FL raw",       "FL channel before correction"),
    ("FL prep",      "After BaSiCPy + focus restoration + fl_normalization"),
    ("Mask refined", "After FL union mask + temporal consistency (refine_masks_temporal)"),
]


def load_b64(path: Path) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def discover_probe_frames(diag_dir: Path) -> list:
    """Find probe frame indices from 01_raw_BF_f*.png filenames."""
    frames = sorted(
        int(p.stem.split("_f")[1])
        for p in diag_dir.glob("01_raw_BF_f*.png")
    )
    if not frames:
        # Fall back to any available stage
        for pat in ("02_preprocessed_BF_f*.png", "03_segmentation_overlay_f*.png"):
            frames = sorted(
                int(p.stem.rsplit("_f", 1)[1])
                for p in diag_dir.glob(pat)
            )
            if frames:
                break
    return frames


def img_tag(path: Path, alt: str = "") -> str:
    if not path.exists():
        return f'<div class="missing">—</div>'
    b64 = load_b64(path)
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'alt="{alt}" title="{path.name}" loading="lazy">'
    )


def build_html(diag_dir: Path, movie_name: str, probe_frames: list) -> str:
    # Check which panels are actually present for at least one frame
    active_panels = []
    for label, pattern in PANEL_DEFS:
        present = any(
            (diag_dir / pattern.format(frame=f"{f:03d}")).exists()
            for f in probe_frames
        )
        active_panels.append((label, pattern, present))

    n_cols = sum(1 for _, _, present in active_panels if present)

    # Build frame rows
    frame_rows = []
    for f in probe_frames:
        fs = f"{f:03d}"
        cells = []
        for label, pattern, present in active_panels:
            if not present:
                continue
            path = diag_dir / pattern.format(frame=fs)
            cells.append(
                f'<div class="cell">'
                f'<div class="cell-label">{label}</div>'
                f'{img_tag(path, label)}'
                f'</div>'
            )
        frame_rows.append(
            f'<div class="frame-row" data-frame="{f}">'
            f'<div class="frame-hdr">Frame {f}</div>'
            f'<div class="panels" style="grid-template-columns:repeat({n_cols},1fr)">'
            + "".join(cells) +
            f'</div></div>'
        )

    # Legend — only for active panels
    active_legend = [
        (lbl, desc) for lbl, desc in LEGEND
        if any(lbl == label and present for label, _, present in active_panels)
    ]
    legend_html = "".join(
        f'<div class="leg"><b>{lbl}</b> — {desc}</div>'
        for lbl, desc in active_legend
    )

    missing_note = ""
    if any(not present for _, _, present in active_panels):
        missing = [label for label, _, present in active_panels if not present]
        missing_note = (
            f'<div class="warn">⚠ Not found in diagnostic output: '
            f'{", ".join(missing)}</div>'
        )

    return f"""<title>Diagnostic Viewer — {movie_name}</title>
<style>
  body {{ background:#111; color:#ccc; font-family:system-ui,sans-serif;
         margin:0; padding:14px; box-sizing:border-box; }}
  h2 {{ color:#eee; margin:0 0 3px; font-size:1.05rem; }}
  .meta {{ color:#666; font-size:0.75rem; margin-bottom:14px; }}
  .legend {{ display:flex; gap:9px; flex-wrap:wrap; margin-bottom:13px; }}
  .leg {{ background:#1b1b1b; border:1px solid #2e2e2e; border-radius:5px;
          padding:4px 9px; font-size:0.71rem; }}
  .leg b {{ color:#aaa; }}
  .warn {{ background:#2a1515; border:1px solid #5a2020; border-radius:5px;
           padding:6px 10px; font-size:0.75rem; color:#cc7777;
           margin-bottom:10px; }}
  .frame-row {{ margin-bottom:14px; border:1px solid #222; border-radius:7px;
                overflow:hidden; }}
  .frame-hdr {{ background:#181818; padding:6px 11px; font-size:0.78rem;
                color:#777; border-bottom:1px solid #222; }}
  .panels {{ display:grid; gap:3px; padding:3px; background:#111; }}
  .cell {{ text-align:center; min-width:0; }}
  .cell-label {{ font-size:0.62rem; color:#555; padding:2px 0 2px; }}
  .cell img {{ width:100%; display:block; border-radius:3px; cursor:zoom-in; }}
  .missing {{ color:#444; font-size:0.7rem; padding:30px 4px; background:#181818;
              border-radius:3px; }}
  #lightbox {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.9);
               z-index:100; align-items:center; justify-content:center;
               flex-direction:column; }}
  #lightbox.open {{ display:flex; }}
  #lightbox img {{ max-width:92vw; max-height:86vh; border-radius:6px; }}
  #lb-label {{ color:#999; font-size:0.75rem; margin-top:8px; }}
  #lightbox button {{ position:fixed; top:12px; right:16px; background:#2a2a2a;
                      color:#ddd; border:1px solid #444; padding:4px 13px;
                      border-radius:4px; cursor:pointer; font-size:0.95rem; }}
</style>

<h2>Diagnostic Viewer — {movie_name}</h2>
<div class="meta">
  {len(probe_frames)} probe frames · {n_cols} panels ·
  source: {diag_dir}
</div>

<div class="legend">{legend_html}</div>
{missing_note}
{"".join(frame_rows)}

<div id="lightbox">
  <button onclick="closeLB()">✕</button>
  <img id="lbimg" src="" alt="">
  <div id="lb-label"></div>
</div>

<script>
document.querySelectorAll('.cell img').forEach(img => {{
  img.addEventListener('click', () => {{
    const label = img.closest('.cell').querySelector('.cell-label').textContent;
    const fhdr  = img.closest('.frame-row').querySelector('.frame-hdr').textContent;
    document.getElementById('lbimg').src = img.src;
    document.getElementById('lb-label').textContent = fhdr + '  ·  ' + label;
    document.getElementById('lightbox').classList.add('open');
  }});
}});
function closeLB() {{ document.getElementById('lightbox').classList.remove('open'); }}
document.getElementById('lightbox').addEventListener('click', e => {{
  if (e.target === document.getElementById('lightbox')) closeLB();
}});
</script>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML viewer from HiTMicTools diagnostic output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "diag_dir",
        help="Path to _diagnostics/<movie_name>/ directory",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Output HTML path (default: <diag_dir>/viewer.html)",
    )
    args = parser.parse_args()

    diag_dir = Path(args.diag_dir).resolve()
    if not diag_dir.exists():
        print(f"ERROR: directory not found: {diag_dir}", file=sys.stderr)
        print("Has the diagnostic job finished? Check: squeue -u $USER", file=sys.stderr)
        sys.exit(1)

    movie_name = diag_dir.name
    out_path = Path(args.out) if args.out else diag_dir / "viewer.html"

    probe_frames = discover_probe_frames(diag_dir)
    if not probe_frames:
        print(f"ERROR: no probe-frame PNGs found in {diag_dir}", file=sys.stderr)
        print("Expected files like 01_raw_BF_f000.png", file=sys.stderr)
        sys.exit(1)

    print(f"Movie   : {movie_name}")
    print(f"Frames  : {probe_frames}")
    print(f"Building viewer...", flush=True)

    html = build_html(diag_dir, movie_name, probe_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)

    size_kb = out_path.stat().st_size / 1024
    print(f"Written : {out_path}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
