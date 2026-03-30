# PAAWS RelCon Pipeline

Full pipeline from raw PAAWS accelerometer data to a pre-trained RelCon model and downstream evaluation.

---

## Step 0 — Create PAAWS configs

### MotifDist config
Add to `relcon/experiments/configs/MotifDist_expconfigs.py`:

```python
allmotifdist_expconfigs["paaws_motifdist"] = MotifDist_ModelConfig(
    data_config=SSLDataConfig(
        data_folder="relcon/data/datasets/paaws_pretrain",
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
        data_folder="relcon/data/datasets/paaws_pretrain",
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
    eval_configs=[],   # keep empty until Step 5
)
```

---

## Step 1 — Preprocess PAAWS pre-training data

Run once per participant, specifying their train/val split assignment.

```bash
python relcon/data/process/process_paaws_pretrain.py \
    --input /path/to/participant.csv \
    --ID <participant_id> \
    --train \          # or --val
    --output_dir relcon/data/datasets/paaws_pretrain
```

- Output: `relcon/data/datasets/paaws_pretrain/{train,val}/subject_{ID}/hour_{i}/ts_{j}.npy`
- Each subject/hour folder must have ≥ 20 segments (`withinuser_cands=20`); script warns if not met
- test/ folder is not needed and is never used during pre-training

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

Downstream evaluation is skipped because `eval_configs=[]` in the config.

---

## Step 4 — Preprocess PAAWS labeled (downstream) data

```bash
python relcon/data/process/process_paaws_downstream.py \
    --dataset_dir /path/to/PAAWS_FreeLiving \
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

Add `Base_EvalConfig` entries to `eval_configs` in the `paaws_relcon` config in
`relcon/experiments/configs/RelCon_expconfigs.py`, then re-run Step 3:

```python
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
]
```

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
