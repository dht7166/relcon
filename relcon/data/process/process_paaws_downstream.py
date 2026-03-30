"""
Convert PAAWS labeled accelerometer data into RelCon downstream evaluation format.

Supports both PAAWS_FreeLiving and PAAWS_SimFL_Lab datasets.
Expected folder structure:
  DS_ID/
  ├── accel/  DS_ID-{condition}-{location}.csv
  └── label/  DS_ID-{condition}-label.csv

Produces per-fold numpy files in --output_dir:
  cv{fold}_train_X.npy   shape: (N, 3, --window_size), float32  — channels first
  cv{fold}_train_y.npy   shape: (N,),                  int64    — 0-indexed class IDs
  cv{fold}_val_X.npy / cv{fold}_val_y.npy
  cv{fold}_test_X.npy / cv{fold}_test_y.npy

Window duration = window_size / sampling_rate seconds.
Source Hz is read from each file's header. If source Hz differs from --sampling_rate,
raw windows are resampled to --window_size output samples.
Label = majority vote per window. CV splits are subject-wise.

Usage:
  # 3.2 s @ 80 Hz — no resampling (matches pre-training v1)
  python process_paaws_downstream.py \
      --dataset_dir /projects/annemarie/PAAWS_backup/for_release/PAAWS_FreeLiving \
      --location    LeftWrist \
      --output_dir  relcon/data/datasets/paaws_downstream/freeliving_leftwrist \
      --n_folds     5 \
      --mapping     coarse_6 \
      --window_size 256 \
      --sampling_rate 80

  # 2.56 s @ 100 Hz — resamples from 205 raw samples to 256 (matches pre-training v2)
  python process_paaws_downstream.py \
      ... \
      --window_size 256 \
      --sampling_rate 100
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ACCEL_COLS = ["Accelerometer X", "Accelerometer Y", "Accelerometer Z"]

# Exclude these from windowing
SKIP_LABELS = {
    "PA_Type_Video_Unavailable/Indecipherable",
    "PA_Type_Too_Complex",
    "PA_Type_Other",
    "Synchronizing_Sensors",
    "Sensor_Sync_Stand_Lab",
    "Sensor_Sync_Walk_Lab",
    None,
}

# ── Activity label mapping schemes ─────────────────────────────────────────────
# Edit / extend these to match your analysis goals.
MAPPING_SCHEMES = {
    # 6-class coarse map (works for both FreeLiving and SimFL_Lab)
    "coarse_6": {
        # Lying
        "Lying_Still":                  0,
        "Lying_With_Movement":          0,
        "Lying_On_Back_Lab":            0,
        "Lying_On_Left_Side_Lab":       0,
        "Lying_On_Right_Side_Lab":      0,
        "Lying_On_Stomach_Lab":         0,
        # Sitting
        "Sitting_Still":                1,
        "Sitting_With_Movement":        1,
        "Sit_Recline_Talk_Lab":         1,
        "Sit_Recline_Web_Browse_Lab":   1,
        "Sit_Typing_Lab":               1,
        "Sit_Writing_Lab":              1,
        # Standing
        "Standing_Still":               2,
        "Standing_With_Movement":       2,
        "Stand_Conversation_Lab":       2,
        "Stand_Shelf_Load_Lab":         2,
        # Walking
        "Walking":                      3,
        "Walking_Slow":                 3,
        "Walking_Fast":                 3,
        "Walking_Down_Stairs":          3,
        "Walking_Up_Stairs":            3,
        # Household
        "Folding_Clothes":              4,
        "Ironing":                      4,
        "Sweeping":                     4,
        "Vacuuming":                    4,
        "Kneeling_With_Movement":       4,
        "Puttering_Around":             4,
        "Organizing_Shelf/Cabinet":     4,
        "Chopping_Food_Lab":            4,
        "Watering_Plants":              4,
        "Washing_Hands":                4,
        "Brushing_Teeth":               4,
        "Brushing/Combing/Tying_Hair":  4,
        "Loading/Unloading_Washing_Machine/Dryer": 4,
        "Putting_Clothes_Away":         4,
        # Exercise
        "Cycling_Active_Pedaling_Stationary_Bike": 5,
        "Playing_Frisbee":              5,
        "Ab_Crunches_Lab":              5,
        "Arm_Curls_Lab":                5,
        "Push_Up_Lab":                  5,
        "Machine_Chest_Press_Lab":      5,
        "Machine_Leg_Press_Lab":        5,
    },
}


# ── Data reader ────────────────────────────────────────────────────────────────

def read_actigraph(file: str):
    """Read an ActiGraph CSV and return (DataFrame with Timestamp column, source_hz)."""
    source_hz = 1
    with open(file) as f:
        line = f.readline()
        parsed = line.split()
        for i in range(len(parsed)):
            if parsed[i] == "Hz":
                source_hz = int(parsed[i - 1])
                break
        f.readline()
        start_time = f.readline().split()[-1]
        start_date = f.readline().split()[-1]

    start = datetime.strptime(start_date + " " + start_time, "%m/%d/%Y %H:%M:%S")
    step  = timedelta(seconds=1 / source_hz)

    df = pd.read_csv(file, skiprows=10, header=0)
    df["Timestamp"] = [start + i * step for i in range(len(df))]
    return df, source_hz


def add_labels(actigraph: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """Merge activity labels onto actigraph data by timestamp intervals."""
    actigraph = actigraph.copy()
    actigraph["Activity"] = None

    data_start = label_df["START_TIME"].iloc[0]
    data_end   = label_df["STOP_TIME"].iloc[-1]
    actigraph.loc[actigraph["Timestamp"] < data_start, "Activity"] = "Before_Data_Collection"
    actigraph.loc[actigraph["Timestamp"] > data_end,   "Activity"] = "After_Data_Collection"

    for _, row in label_df.iterrows():
        mask = (actigraph["Timestamp"] >= row["START_TIME"]) & \
               (actigraph["Timestamp"] <= row["STOP_TIME"])
        actigraph.loc[mask, "Activity"] = row["PA_TYPE"]

    return actigraph


# ── Windowing ─────────────────────────────────────────────────────────────────

def window_subject(df: pd.DataFrame, activity_to_id: dict,
                   window_size: int, target_hz: int, source_hz: int):
    """
    Slide a non-overlapping window over one subject's data.
    Window duration = window_size / target_hz seconds.
    source_window = round(window_size / target_hz * source_hz) raw samples are taken per window.
    If source_hz != target_hz, each raw window is resampled to window_size samples.
    Returns X (N, 3, window_size) float32 and y (N,) int64.
    Drops windows whose majority label is not in activity_to_id.
    """
    source_window = round(window_size / target_hz * source_hz)
    do_resample   = source_hz != target_hz

    accel  = df[ACCEL_COLS].to_numpy(dtype=np.float32)
    labels = df["Activity"].to_numpy()

    n_windows = len(accel) // source_window
    X_list, y_list = [], []

    for i in range(n_windows):
        sl  = slice(i * source_window, (i + 1) * source_window)
        win = accel[sl]                                       # (source_window, 3)
        act = Counter(labels[sl]).most_common(1)[0][0]

        if act not in activity_to_id:
            continue

        if do_resample:
            win = resample(win, window_size, axis=0)          # (window_size, 3)

        X_list.append(win.T.astype(np.float32))               # (3, window_size) channels first
        y_list.append(activity_to_id[act])

    if not X_list:
        return np.empty((0, 3, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


# ── Subject-wise K-fold CV ─────────────────────────────────────────────────────

def make_folds(subject_ids, n_folds: int, seed: int):
    """
    Rotate test subjects across folds. Each fold's remaining subjects are
    split 80/20 into train/val.
    Returns list of {"train": [...], "val": [...], "test": [...]} dicts.
    """
    ids    = np.array(subject_ids)
    ids    = np.random.default_rng(seed).permutation(ids)
    chunks = np.array_split(ids, n_folds)
    folds  = []

    for i in range(n_folds):
        test_ids  = chunks[i].tolist()
        trainval  = [s for j, c in enumerate(chunks) if j != i for s in c]
        train_ids, val_ids = train_test_split(trainval, test_size=0.2, random_state=seed)
        folds.append({"train": train_ids, "val": val_ids, "test": test_ids})

    return folds


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert PAAWS labeled data to RelCon downstream eval format."
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to PAAWS_FreeLiving or PAAWS_SimFL_Lab.")
    parser.add_argument("--location",    required=True,
                        help="Sensor location to use, e.g. LeftWrist, RightWrist, RightWaist.")
    parser.add_argument("--output_dir",  required=True,
                        help="Output directory for cv{fold}_{split}_X/y.npy files.")
    parser.add_argument("--n_folds",  type=int, default=5)
    parser.add_argument("--mapping",  default="coarse_6",
                        choices=list(MAPPING_SCHEMES.keys()),
                        help="Activity label mapping scheme (default: coarse_6).")
    parser.add_argument("--window_size",   type=int, default=256,
                        help="Number of output samples per window (default: 256).")
    parser.add_argument("--sampling_rate", type=int, default=80,
                        help="Target sampling rate in Hz. Raw windows are resampled if the file's source Hz differs (default: 80).")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    dataset_dir    = Path(args.dataset_dir)
    activity_to_id = MAPPING_SCHEMES[args.mapping]
    out            = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Detect condition tag ("Free" or "Lab") from directory name
    condition = "Free" if "FreeLiving" in dataset_dir.name else "Lab"

    # ── Discover subjects ──────────────────────────────────────────────────
    subject_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("DS_"))

    window_dur_s = args.window_size / args.sampling_rate
    print(f"Dataset:        {dataset_dir.name}  (condition='{condition}')")
    print(f"Location:       {args.location}")
    print(f"Window:         {args.window_size} samples @ {args.sampling_rate} Hz  ({window_dur_s:.3f} s)")
    print(f"Mapping:        {args.mapping}  ({len(activity_to_id)} classes)")
    print(f"Subjects:       {len(subject_dirs)} found")

    # ── Load and window all subjects ───────────────────────────────────────
    subject_X, subject_y = {}, {}

    for subj_dir in tqdm(subject_dirs, desc="Loading"):
        sid = subj_dir.name

        # Locate accel file for this location
        accel_matches = list((subj_dir / "accel").glob(f"*{condition}-{args.location}.csv"))
        label_matches = list((subj_dir / "label").glob(f"*{condition}-label.csv"))

        if not accel_matches:
            print(f"  [{sid}] No accel file for location '{args.location}' — skipping.")
            continue
        if not label_matches:
            print(f"  [{sid}] No label file — skipping.")
            continue

        df, source_hz = read_actigraph(str(accel_matches[0]))
        label_df = pd.read_csv(label_matches[0], parse_dates=["START_TIME", "STOP_TIME"])
        df       = add_labels(df, label_df)

        X, y = window_subject(df, activity_to_id, args.window_size, args.sampling_rate, source_hz)
        if len(X) == 0:
            print(f"  [{sid}] 0 usable windows — skipping.")
            continue

        subject_X[sid] = X
        subject_y[sid] = y

    valid_ids = list(subject_X.keys())
    print(f"\n{len(valid_ids)} subjects with usable windows.")

    # ── Build folds and save ───────────────────────────────────────────────
    folds = make_folds(valid_ids, n_folds=args.n_folds, seed=args.seed)

    for fold_idx, fold in enumerate(folds):
        for split in ["train", "val", "test"]:
            ids_in_split = [s for s in fold[split] if s in subject_X]
            if not ids_in_split:
                print(f"  Warning: fold {fold_idx} {split} is empty.")
                continue

            X_split = np.concatenate([subject_X[s] for s in ids_in_split])
            y_split = np.concatenate([subject_y[s] for s in ids_in_split])

            np.save(out / f"cv{fold_idx}_{split}_X.npy", X_split)
            np.save(out / f"cv{fold_idx}_{split}_y.npy", y_split)

            print(f"  fold {fold_idx} {split:5s}: {X_split.shape}  classes={sorted(np.unique(y_split).tolist())}")

    id_to_name = {v: k for k, v in activity_to_id.items()}
    print(f"\nClass legend: { {i: id_to_name[i] for i in sorted(set(id_to_name))} }")
    print(f"\nDone. Wire into RelCon config with:")
    print(f"  SupervisedDataConfig(")
    print(f'    data_folder="{out}",')
    print(f'    X_annotates=[""],')
    print(f'    y_annotate="",')
    print(f"  )")
    print(f"  cv_splits={args.n_folds}")


if __name__ == "__main__":
    main()
