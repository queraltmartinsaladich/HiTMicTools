#!/usr/bin/env python3
"""Split an input folder into N worklist chunks for SLURM array jobs.

Each chunk is a plain text file listing one absolute file path per line.
The submit_array.sh script passes SLURM_ARRAY_TASK_ID as the chunk index.

Usage
-----
    python scripts/scicore/split_worklist.py \\
        --input_folder /scicore/data/images \\
        --output_dir   /scicore/data/worklists \\
        --n_chunks     16 \\
        --file_type    nd2

Output
------
    {output_dir}/chunk_00.txt
    {output_dir}/chunk_01.txt
    ...
    {output_dir}/chunk_N-1.txt

Prints the total number of chunks (useful for --array=0-N in sbatch).
"""
import argparse
import glob
import math
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_folder", required=True, help="Folder containing image files")
    parser.add_argument("--output_dir", required=True, help="Where to write chunk files")
    parser.add_argument("--n_chunks", type=int, default=16, help="Number of chunks (= SLURM array size)")
    parser.add_argument("--file_type", default="nd2", help="File extension to glob (without dot)")
    args = parser.parse_args()

    pattern = os.path.join(args.input_folder, f"**/*.{args.file_type}")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pattern_flat = os.path.join(args.input_folder, f"*.{args.file_type}")
        files = sorted(glob.glob(pattern_flat))

    if not files:
        print(f"ERROR: No .{args.file_type} files found in {args.input_folder}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    n_chunks = min(args.n_chunks, len(files))
    chunk_size = math.ceil(len(files) / n_chunks)

    for i in range(n_chunks):
        chunk = files[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue
        chunk_path = os.path.join(args.output_dir, f"chunk_{i:02d}.txt")
        with open(chunk_path, "w") as fh:
            fh.write("\n".join(chunk) + "\n")

    print(f"Split {len(files)} files into {n_chunks} chunks → {args.output_dir}")
    print(f"SLURM array range: --array=0-{n_chunks - 1}")


if __name__ == "__main__":
    main()
