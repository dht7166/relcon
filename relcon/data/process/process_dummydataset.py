"""
This function constructs a dummy dataset that mimics our original pre-training dataset.
Due to privacy concerns and IRB restrictions, we will be unable to release our pre-training
data. Therefore, if you would like to pre-train the RelCon model on
your own dataset, please pre-process the data so that the numpy time-series files follows
the below structure.

NOTE: This script uses the large-scale data format (per-subject .npy + _meta.pkl),
designed for use with large datasets such as NHANES or PAAWS. Each subject's windows
are stored as a single stacked array rather than individual per-window files.

relcon/
└── data/
    └── datasets/
        └── dummydataset/
            ├── train/
            │   ├── subject_0.npy              shape: (total_windows, 256, 3)
            │   ├── subject_0_meta.pkl         {hour_id: (start_row, end_row)}
            │   └── ...
            ├── val/
            │   ├── subject_32.npy
            │   ├── subject_32_meta.pkl
            │   └── ...
            └── test/
                ├── subject_48.npy
                ├── subject_48_meta.pkl
                └── ...

"""

import pickle
import numpy as np
import os
from tqdm import tqdm

PATH = "relcon/data/datasets/dummydataset"
NUM_SUBJECTS = 64
NUM_HOURS_PER_SUBJECT = 2
NUM_TS_PER_HOUR = 25

TIMELEN = 256  # 2.56 long sequence sampled at 100 Hz
CHANNELS = 3  # 3 axis accerlometry


def main():
    os.makedirs(PATH, exist_ok=True)
    for subject_id in tqdm(range(NUM_SUBJECTS)):
        # construct parent folder for train/val/test
        if subject_id < NUM_SUBJECTS // 2:
            TYPE = "train"
        elif subject_id < 3 * NUM_SUBJECTS // 4:
            TYPE = "val"
        else:
            TYPE = "test"
        typepath = os.path.join(PATH, TYPE)
        os.makedirs(typepath, exist_ok=True)

        # Generate all windows for this subject across all hours
        windows_all = []
        meta = {}
        row = 0
        for hour_id in range(NUM_HOURS_PER_SUBJECT):
            hour_windows = np.random.normal(size=(NUM_TS_PER_HOUR, TIMELEN, CHANNELS)).astype(np.float32)
            meta[hour_id] = (row, row + NUM_TS_PER_HOUR)
            windows_all.append(hour_windows)
            row += NUM_TS_PER_HOUR

        stacked = np.concatenate(windows_all, axis=0)
        np.save(os.path.join(typepath, f"subject_{subject_id}.npy"), stacked)
        with open(os.path.join(typepath, f"subject_{subject_id}_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)


if __name__ == "__main__":
    main()
