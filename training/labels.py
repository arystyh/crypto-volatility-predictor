"""Label untuk training — tipis di atas app.ml.target (single source of truth).

Target utama (lihat app/ml/target.py & docs/target_and_ledger.md):
    y_t = 1 jika log-Parkinson vol H menit ke depan > Z * sd di atas baseline 30 hari (masa lalu saja)

Target hazard (kapan spike MULAI):
    state_{t+k} = (trail_logvol_{t+k} - mu_t) / sd_t > Z      (baseline dibekukan di t)
    onset_t     = k pertama di 1..H dengan state_{t+k} = 1, bila state_t = 0
Karena trail_{t+H} = fwd_t, maka y_t = 1  =>  onset_t <= H (untuk t yang belum dalam spike). Kebalikannya
tidak selalu: spike bisa mulai dan selesai sebelum menit ke-H (onset <= H tapi y = 0), sehingga
P(onset <= 15) >= P(y = 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.target import CANDLE_MS, TargetSpec, compute_targets

SPEC = TargetSpec()
HORIZON = SPEC.horizon
Z_THRESHOLD = SPEC.z_threshold
Z_WINDOW = SPEC.baseline_window


def contiguous_segments(df: pd.DataFrame, min_len: int = 1) -> list[pd.DataFrame]:
    """Pecah data jadi segmen candle 1m yang kontinu (target menolak data berlubang)."""
    t = df["open_time"].to_numpy(dtype=np.int64)
    breaks = np.flatnonzero(np.diff(t) != CANDLE_MS) + 1
    parts = np.split(np.arange(len(df)), breaks)
    return [df.iloc[p] for p in parts if len(p) >= min_len]


def make_labels(candles: pd.DataFrame, spec: TargetSpec = SPEC) -> pd.DataFrame:
    """Kolom: label, z, baseline_mu, baseline_sd, trail_logvol, fwd_logvol, onset (1..H, 0 = tidak ada,
    NaN = sudah dalam spike / tidak terdefinisi), fwd_ret (untuk backtest)."""
    tg = compute_targets(candles, spec)
    H = spec.horizon
    mu, sd = tg["baseline_mu"], tg["baseline_sd"].where(tg["baseline_sd"] > 0)
    trail = tg["trail_logvol"]
    z_now = (trail - mu) / sd
    zk = np.column_stack([((trail.shift(-k) - mu) / sd).to_numpy() for k in range(1, H + 1)])
    above = zk > spec.z_threshold
    first = np.where(above.any(axis=1), above.argmax(axis=1) + 1, 0).astype(float)
    onset = pd.Series(first, index=candles.index)
    onset[tg["label"].isna() | (z_now > spec.z_threshold) | z_now.isna()] = np.nan
    close = candles["close"].astype(float)
    tg["onset"] = onset
    tg["in_spike_now"] = (z_now > spec.z_threshold).astype(float).where(z_now.notna())
    tg["fwd_ret"] = np.log(close.shift(-H) / close)
    return tg
