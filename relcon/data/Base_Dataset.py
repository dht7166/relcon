import pickle
import torch
import pathlib
import numpy as np

##### Dataset Configs #####
class Base_DatasetConfig:
    def __init__(
        self,
        data_folder: str,
    ):
        self.data_folder = data_folder
        self.type = None


class SupervisedDataConfig(Base_DatasetConfig):
    def __init__(self, X_annotates: list = [""], y_annotate: str = "", **kwargs):
        super().__init__(**kwargs)

        self.X_annotates = X_annotates
        self.y_annotate = y_annotate

        self.type = "supervised"


class SSLDataConfig(Base_DatasetConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type = "ssl"


##### Dataset Classes #####
class Base_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super.__init__()


class OnTheFly_FolderNpyDataset(Base_Dataset):
    def __init__(self, path: str):
        "Initialization"
        self.path = path
        split_dir = pathlib.Path(path)
        meta_files = sorted(split_dir.glob("subject_*_meta.pkl"))

        self.subject_meta = {}
        for mf in meta_files:
            npy_path = str(mf.parent / (mf.stem.replace("_meta", "") + ".npy"))
            with open(mf, "rb") as f:
                self.subject_meta[npy_path] = pickle.load(f)

        # hours_list: one entry per (subject, hour) — compact, O(subjects × hours)
        self.hours_list = [
            (npy_path, hour_id, start, end)
            for npy_path, meta in sorted(self.subject_meta.items())
            for hour_id, (start, end) in sorted(meta.items())
        ]
        # cum_lengths[i] = total windows up to and including hours_list[i]
        self.cum_lengths = np.cumsum([end - start for _, _, start, end in self.hours_list])
        self.length = int(self.cum_lengths[-1]) if len(self.cum_lengths) else 0

    def __len__(self):
        "Denotes the total number of samples"
        return self.length

    def __getitem__(self, idx):
        "Generates one sample of data"
        # Find which hour entry this index belongs to
        i = int(np.searchsorted(self.cum_lengths, idx, side="right"))
        npy_path, hour_id, start, _ = self.hours_list[i]
        offset = idx - (int(self.cum_lengths[i - 1]) if i > 0 else 0)
        row_idx = start + offset

        arr = np.load(npy_path, mmap_mode="r")
        signal = arr[row_idx].astype(np.float32).copy()

        return {"signal": signal, "filepath": npy_path, "row_idx": row_idx, "hour_id": hour_id}
