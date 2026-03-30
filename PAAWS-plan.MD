# PAAWS RelCon Pipeline

Full pipeline from raw PAAWS accelerometer data to a pre-trained RelCon model and downstream evaluation.

---

## Step 0 — Create PAAWS configs

### MotifDist config
Add to `relcon/experiments/configs/MotifDist_expconfigs.py`:

```python
allmotifdist_expconfigs["paaws_motifdist"] = MotifDist_ModelConfig(
    data_config=SSLDataConfig(
        data_folder="relcon/data/datasets/paaws_pretrain_v1_80hz",  # or _v2_100hz; must match Step 1 --output_dir
    ),
    net_config=Base_NetConfig(
        net_folder="CrossAttn",
        net_file="CrossAttn_Net",
        params={
            "query_dimsize": 3,
            "key_dimsize": 3,
            "kernel_size": 15,
            "embed_dim": 64,
            "double_receptivefield": 2,
        },
    ),
    epochs=10,
    lr=0.001,
    batch_size=64,
    save_epochfreq=10,
)
```

### RelCon config
Add to `relcon/experiments/configs/RelCon_expconfigs.py`:

```python
allrelcon_expconfigs["paaws_relcon"] = RelCon_ModelConfig(
    withinuser_cands=20,
    motifdist_expconfig_key="paaws_motifdist",   # must match key above
    data_config=SSLDataConfig(
        data_folder="relcon/data/datasets/paaws_pretrain_v1_80hz",  # must match MotifDist data_folder
    ),
    net_config=Base_NetConfig(
        net_folder="ResNet1D",
        net_file="ResNet1D_Net",
        params={
            "init_conv": {"in_channels": 3, "base_filters": 64, "kernel_size": 7, "stride": 2, "padding": 0},
            "init_maxpool": {"kernel_size": 3, "stride": 2},
            "blocks": [
                {"n": 3, "in_channels": 64, "base_filters": 64, "kernel_size": 3, "stride": 1},
                {"n": 4, "in_channels": 64, "base_filters": 128, "kernel_size": 4, "stride": 1},
                {"n": 6, "in_channels": 128, "base_filters": 256, "kernel_size": 4, "stride": 1},
            ],
            "finalpool": "avg",
        },
    ),
    epochs=100,
    lr=0.000001,
    batch_size=64,
    save_epochfreq=10,
    eval_configs=[
        Base_EvalConfig(
            name="PAAWS FreeLiving LeftWrist | Linear Probe",
            model_folder="Classify",
            model_file="linear_probe",     # or "MLP_probe" / "MLP_finetune"
            cv_splits=5,
            data_config=SupervisedDataConfig(
                data_folder="relcon/data/datasets/paaws_downstream/freeliving_leftwrist",
            ),
        ),
    ],
)
```

---

## Step 1 — Preprocess PAAWS pre-training data

Run once per participant, specifying their train/val split assignment.

```bash
python relcon/data/process/process_actigraph_pretrain.py \
    --input <INPUT_CSV> \
    --ID <SUBJECT_ID> \
    --train \
    --output_dir relcon/data/datasets/paaws_motifdist
# replace --train with --val for validation participants
```

The script produces **two** output directories from a single `--output_dir <BASE>`:
- `<BASE>_v1_80hz/` — 256-sample windows @ 80 Hz (3.2 s), no resampling
- `<BASE>_v2_100hz/` — 205-sample windows @ 80 Hz resampled to 256 samples @ 100 Hz

Both share the same structure:
```
<BASE>_{v1_80hz,v2_100hz}/{train,val}/subject_<SUBJECT_ID>/hour_<i>/ts_<j>.npy
```

- Each subject/hour folder must have ≥ 20 segments (`withinuser_cands=20`); script warns if not met
- `test/` split is supported but not used during pre-training

---

## Step 2 — Train MotifDist

```bash
python run_exp.py --config paaws_motifdist
```

Output: `relcon/experiments/out/paaws_motifdist/checkpoint_best.pkl`

---

## Step 3 — Pre-train RelCon

```bash
python run_exp.py --config paaws_relcon
```

Output: `relcon/experiments/out/paaws_relcon/checkpoint_best.pkl`

Downstream evaluation is skipped because `--run_eval` is not passed.

---

## Step 4 — Preprocess PAAWS labeled (downstream) data

```bash
python relcon/data/process/process_paaws_downstream.py \
    --dataset_dir <PAAWS_FREELIVING_DIR> \
    --location LeftWrist \
    --output_dir relcon/data/datasets/paaws_downstream/freeliving_leftwrist \
    --n_folds 5 \
    --mapping coarse_6
```

- Output: `cv{i}_{train,val,test}_X.npy` + `cv{i}_{train,val,test}_y.npy` per fold
- `--mapping coarse_6`: 6-class activity mapping (Lying, Sitting, Standing, Walking, Light activity, Vigorous exercise)
- `--location`: sensor placement (e.g. LeftWrist, RightWrist, RightWaist)

---

## Step 5 — Downstream evaluation

```bash
python run_exp.py --config paaws_relcon --run_eval
```

Pre-training is skipped (checkpoint already exists). Evaluation runs all `eval_configs` entries.

Results saved to: `relcon/experiments/out/paaws_relcon/paaws_relcon_easy_paste.csv`

---

## Key files

| File | Purpose |
|------|---------|
| `relcon/data/process/process_paaws_pretrain.py` | Raw → pre-training `.npy` windows |
| `relcon/data/process/process_paaws_downstream.py` | Raw → labeled CV fold `.npy` files |
| `relcon/experiments/configs/MotifDist_expconfigs.py` | Add `paaws_motifdist` config here |
| `relcon/experiments/configs/RelCon_expconfigs.py` | Add `paaws_relcon` config here |
| `run_exp.py` | Unified entry point for training and evaluation |
