"""
Example: generate RelCon embeddings from a raw ActiGraph CSV file.

Reads an ActiGraph CSV, windows the accelerometer signal, and runs each window
through the frozen pre-trained backbone to produce a (N, embed_dim) embedding array.

A sidecar <output>_timestamps.npy is also saved with the start timestamp of each
window, so embeddings can be aligned back to wall-clock time.

Usage:
  python generate_embeddings.py \
      --config      paaws_relcon_80hz \
      --input       /path/to/DS_001-Free-LeftAnkle.csv \
      --output      embeddings/DS_001_LeftAnkle.npy \
      --window_size 256 \
      --sampling_rate 80

  # override checkpoint explicitly
  python generate_embeddings.py \
      --config      paaws_relcon_80hz \
      --checkpoint  relcon/experiments/out/paaws_relcon_80hz/checkpoint_best.pkl \
      --input       /path/to/participant.csv \
      --output      embeddings/participant.npy
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from tqdm import tqdm

ACCEL_COLS = ["Accelerometer X", "Accelerometer Y", "Accelerometer Z"]


# ── Data loading ───────────────────────────────────────────────────────────────

def read_actigraph(path: str):
    """
    Parse an ActiGraph CSV header and return (DataFrame with Timestamp, source_hz).
    """
    source_hz = 1
    with open(path) as f:
        line = f.readline()
        tokens = line.split()
        for i, tok in enumerate(tokens):
            if tok == "Hz":
                source_hz = int(tokens[i - 1])
                break
        f.readline()
        start_time = f.readline().split()[-1]
        start_date = f.readline().split()[-1]

    start = datetime.strptime(start_date + " " + start_time, "%m/%d/%Y %H:%M:%S")
    step  = timedelta(seconds=1 / source_hz)

    df = pd.read_csv(path, skiprows=10, header=0)
    df["Timestamp"] = [start + i * step for i in range(len(df))]
    return df, source_hz


# ── Windowing ──────────────────────────────────────────────────────────────────

def make_windows(df: pd.DataFrame, window_size: int, target_hz: int, source_hz: int):
    """
    Slice the signal into non-overlapping windows.

    If source_hz != target_hz, each window is resampled to window_size samples.

    Returns:
      X          (N, 3, window_size) float32  — channels first, ready for the net
      timestamps (N,) datetime64              — start timestamp of each window
    """
    source_window = round(window_size / target_hz * source_hz)
    do_resample   = source_hz != target_hz

    accel  = df[ACCEL_COLS].to_numpy(dtype=np.float32)  # (T, 3)
    times  = df["Timestamp"].to_numpy()

    n = len(accel) // source_window
    X_list, ts_list = [], []

    for i in range(n):
        sl  = slice(i * source_window, (i + 1) * source_window)
        win = accel[sl]                                   # (source_window, 3)

        if do_resample:
            win = resample(win, window_size, axis=0).astype(np.float32)

        X_list.append(win.T)                              # (3, window_size)
        ts_list.append(times[i * source_window])

    X          = np.stack(X_list).astype(np.float32)     # (N, 3, window_size)
    timestamps = np.array(ts_list)
    return X, timestamps


# ── Model ──────────────────────────────────────────────────────────────────────

def build_net(config_key: str) -> torch.nn.Module:
    from relcon.experiments.configs.RelCon_expconfigs import allrelcon_expconfigs
    from relcon.utils.imports import import_net
    return import_net(allrelcon_expconfigs[config_key].net_config)


def load_checkpoint(net: torch.nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state["net"])
    print(f"Loaded: {ckpt_path}")
    return net


@torch.no_grad()
def embed(net: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """X: (N, 3, W)  →  returns (N, embed_dim)"""
    net.eval()
    chunks = []
    for i in tqdm(range(0, len(X), batch_size), desc="Embedding", unit="batch"):
        batch = torch.from_numpy(X[i : i + batch_size]).to(device)
        chunks.append(net(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate RelCon embeddings from an ActiGraph CSV.")
    parser.add_argument("--config",       required=True,
                        help="RelCon config key, e.g. paaws_relcon_80hz.")
    parser.add_argument("--input",        required=True,
                        help="Path to an ActiGraph CSV file.")
    parser.add_argument("--output",       required=True,
                        help="Output path for embeddings .npy (shape: N x embed_dim).")
    parser.add_argument("--window_size",  type=int, default=256,
                        help="Number of output samples per window (default: 256).")
    parser.add_argument("--sampling_rate", type=int, default=80,
                        help="Target Hz. Windows are resampled if source Hz differs (default: 80).")
    parser.add_argument("--checkpoint",   default=None,
                        help="Checkpoint path. Defaults to relcon/experiments/out/<config>/checkpoint_best.pkl.")
    parser.add_argument("--batch_size",   type=int, default=256)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or os.path.join(
        "relcon/experiments/out", args.config, "checkpoint_best.pkl"
    )
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # ── Net ───────────────────────────────────────────────────────────────────
    net = build_net(args.config).to(device)
    net = load_checkpoint(net, ckpt_path, device)

    # ── Read + window ─────────────────────────────────────────────────────────
    print(f"Reading {args.input}")
    df, source_hz = read_actigraph(args.input)
    print(f"  Source Hz: {source_hz}   rows: {len(df):,}")

    X, timestamps = make_windows(df, args.window_size, args.sampling_rate, source_hz)
    print(f"  Windows: {len(X):,}  shape: {X.shape}  (window_size={args.window_size}, target_hz={args.sampling_rate})")

    # ── Embed ─────────────────────────────────────────────────────────────────
    embeddings = embed(net, X, batch_size=args.batch_size, device=device)
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)
    print(f"Saved embeddings  → {out}")

    ts_out = out.with_name(out.stem + "_timestamps.npy")
    np.save(ts_out, timestamps)
    print(f"Saved timestamps  → {ts_out}")


if __name__ == "__main__":
    main()
