"""
Convert a single PAAWS participant's accelerometer file into RelCon pre-training format.

Produces two dataset versions:
  v1_80hz/   — 256-sample windows @ 80 Hz (3.2 s), no resampling
  v2_100hz/  — 205-sample windows @ 80 Hz (2.56 s), resampled to 256 samples @ 100 Hz

Output structure (same for both versions):
  {out_dir}/{split}/subject_{ID}/hour_{i}/ts_{j}.npy
  Each .npy: float32 array of shape (256, 3)

Output directory resolution (first match wins):
  --output_dir_v1 / --output_dir_v2   per-version explicit paths
  --output_dir DIR                     → DIR_v1_80hz  and  DIR_v2_100hz
  (default)                            → relcon/data/datasets/v1_80hz
                                          relcon/data/datasets/v2_100hz

Usage:
  python process_paaws_pretrain.py \
      --input /path/to/participant.csv \
      --ID P001 \
      --train

  python process_paaws_pretrain.py \
      --input /path/to/participant.csv \
      --ID P001 \
      --val \
      --output_dir /data/paaws_pretrain
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.signal import resample

# ── Constants ──────────────────────────────────────────────────────────────────
SRC_HZ = 80

# Version 1: 3.2 s window @ 80 Hz — no resampling
V1_WINDOW = 256          # 256 / 80 = 3.2 s

# Version 2: 2.56 s window @ 80 Hz → resample to 100 Hz
V2_WINDOW = 205          # floor(2.56 * 80) = 204.8 → 205 samples
V2_OUT    = 256          # 2.56 s * 100 Hz = 256 samples

MIN_SEGS_PER_HOUR = 20   # RelCon requires withinuser_cands segments per hour

# Standard ActiGraph accelerometer column names for PAAWS
ACCEL_COLS = ["Accelerometer X", "Accelerometer Y", "Accelerometer Z"]

DEFAULT_V1 = Path("relcon/data/datasets/v1_80hz")
DEFAULT_V2 = Path("relcon/data/datasets/v2_100hz")


# ── Data reader (adapted from read_accelerometer_data.py) ─────────────────────

def read_data(file: str) -> pd.DataFrame:
    """
    Read an ActiGraph CSV file and return a DataFrame with a Timestamp column.
    Handles the standard ActiGraph metadata header (10 rows).
    """
    sampling_rate = 1

    with open(file) as f:
        line = f.readline()
        parsed = line.split()
        for i in range(len(parsed)):
            if parsed[i] == "Hz":
                sampling_rate = int(parsed[i - 1])
                break
        f.readline()
        start_time = f.readline().split()[-1]
        start_date = f.readline().split()[-1]

    start = datetime.strptime(start_date + " " + start_time, "%m/%d/%Y %H:%M:%S")
    step  = timedelta(seconds=1 / sampling_rate)

    df = pd.read_csv(file, skiprows=10, header=0)
    df["Timestamp"] = [start + i * step for i in range(len(df))]
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def _segment(data: np.ndarray, window: int) -> np.ndarray:
    """
    Slice (T, 3) into non-overlapping windows of length `window`.
    Returns (N, window, 3). Drops the trailing incomplete window.
    """
    n = len(data) // window
    return data[: n * window].reshape(n, window, 3)


def _save(windows: np.ndarray, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(windows):
        np.save(folder / f"ts_{i}.npy", w)


# ── Core processing ────────────────────────────────────────────────────────────

def process(
    csv_path: Path,
    subject_id: str,
    split: str,
    out_v1: Path,
    out_v2: Path,
) -> None:
    df = read_data(str(csv_path))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if not all(c in df.columns for c in ACCEL_COLS):
        print(f"Error: expected columns {ACCEL_COLS}, found: {df.columns.tolist()}")
        sys.exit(1)

    accel    = df[ACCEL_COLS].to_numpy(dtype=np.float32)  # (T, 3)
    hour_bin = df["Timestamp"].dt.floor("1h")
    warnings = []

    for hour_idx, (_, hour_df) in enumerate(df.groupby(hour_bin, sort=True)):
        hour_accel = accel[hour_df.index.to_numpy()]       # (T_hour, 3)

        # ── v1: 3.2 s, 256 samples @ 80 Hz, no resampling ─────────────────
        v1_wins = _segment(hour_accel, V1_WINDOW).astype(np.float32)
        _save(v1_wins, out_v1 / split / f"subject_{subject_id}" / f"hour_{hour_idx}")

        # ── v2: 2.56 s (205 samples @ 80 Hz) → resample to 256 @ 100 Hz ───
        v2_src  = _segment(hour_accel, V2_WINDOW)
        v2_wins = resample(v2_src, V2_OUT, axis=1).astype(np.float32) if len(v2_src) > 0 else v2_src
        _save(v2_wins, out_v2 / split / f"subject_{subject_id}" / f"hour_{hour_idx}")

        for label, n in [("v1_80hz", len(v1_wins)), ("v2_100hz", len(v2_wins))]:
            if n < MIN_SEGS_PER_HOUR:
                warnings.append(
                    f"  [{label}] hour_{hour_idx}: {n} segments < {MIN_SEGS_PER_HOUR} required"
                )

    print(f"Done — subject_{subject_id} ({split})")
    print(f"  v1 → {out_v1 / split / f'subject_{subject_id}'}")
    print(f"  v2 → {out_v2 / split / f'subject_{subject_id}'}")

    if warnings:
        print(f"\nWarning: {len(warnings)} hour folder(s) below {MIN_SEGS_PER_HOUR} segments:")
        for w in warnings:
            print(w)
        print("Consider lowering withinuser_cands in the RelCon config.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a single PAAWS participant CSV to RelCon pre-training format."
    )
    parser.add_argument("--input", required=True,
                        help="Path to the participant's ActiGraph CSV file.")
    parser.add_argument("--ID", required=True,
                        help="Unique participant ID (used in subject_<ID> folder name).")

    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--train", action="store_true")
    split_group.add_argument("--val",   action="store_true")
    split_group.add_argument("--test",  action="store_true")

    parser.add_argument("--output_dir",    default=None,
                        help="Base output path. Creates <output_dir>_v1_80hz and <output_dir>_v2_100hz.")
    parser.add_argument("--output_dir_v1", default=None,
                        help="Explicit output directory for v1_80hz. Overrides --output_dir.")
    parser.add_argument("--output_dir_v2", default=None,
                        help="Explicit output directory for v2_100hz. Overrides --output_dir.")
    args = parser.parse_args()

    # ── Resolve split ──────────────────────────────────────────────────────
    split = "train" if args.train else "val" if args.val else "test"

    # ── Resolve output directories ─────────────────────────────────────────
    if args.output_dir_v1:
        out_v1 = Path(args.output_dir_v1)
    elif args.output_dir:
        out_v1 = Path(f"{args.output_dir}_v1_80hz")
    else:
        out_v1 = DEFAULT_V1

    if args.output_dir_v2:
        out_v2 = Path(args.output_dir_v2)
    elif args.output_dir:
        out_v2 = Path(f"{args.output_dir}_v2_100hz")
    else:
        out_v2 = DEFAULT_V2

    process(
        csv_path   = Path(args.input),
        subject_id = args.ID,
        split      = split,
        out_v1     = out_v1,
        out_v2     = out_v2,
    )


if __name__ == "__main__":
    main()
