"""Dataset training lewat JALUR KODE YANG SAMA dengan live (roadmap 2.4).

Live (app/engine.py) per candle close:
    compute_features(window 300 candle)  ->  EWMNormalizer.transform(baris terakhir, update=True)
Offline (di sini) per segmen kontinu:
    compute_features(segmen)              ->  EWMNormalizer.transform_frame(...)  (rekursi yang sama)
Label dari app.ml.target (lihat training/labels.py). tests/test_parity.py membuktikan kedua jalur
memberi fitur yang sama; tests/test_purged_cv.py membuktikan tidak ada kebocoran antar fold.

Lookback fitur terpanjang = LONG_WINDOW (240 candle) -> jarak purge+embargo antar fold =
TargetSpec.min_embargo(LONG_WINDOW) = H + 240.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.features import LONG_WINDOW, compute_features
from app.ml.target import TargetSpec
from app.normalize import HALFLIFE_MIN, MODEL_FEATURES, EWMNormalizer
from training.cv import PurgedGroupTimeSeriesSplit
from training.labels import SPEC, contiguous_segments, make_labels

log = logging.getLogger(__name__)

FEATURE_LOOKBACK = LONG_WINDOW


def embargo_rows(spec: TargetSpec = SPEC, lookback: int = FEATURE_LOOKBACK) -> int:
    return spec.min_embargo(lookback)


def segment_features(df: pd.DataFrame, halflife: float = HALFLIFE_MIN) -> pd.DataFrame:
    """Fitur model untuk satu segmen candle kontinu (urut waktu)."""
    df = df.reset_index(drop=True)
    feats = compute_features(df)
    norm = EWMNormalizer(halflife=halflife).transform_frame(feats, df["open_time"])
    return pd.concat([feats, norm], axis=1)


def build_dataset(candles: dict[str, pd.DataFrame], spec: TargetSpec = SPEC,
                  halflife: float = HALFLIFE_MIN, keep_unlabeled: bool = False, extra=None) -> pd.DataFrame:
    """`extra(segment_df) -> DataFrame` menambah kolom per segmen (mis. input baseline HAR/GARCH)."""
    frames = []
    for symbol, full in candles.items():
        full = full.sort_values("open_time").reset_index(drop=True)
        segments = contiguous_segments(full, min_len=spec.min_baseline + spec.horizon + 1)
        if len(segments) > 1:
            log.info("%s: %d segmen kontinu (ada candle hilang di data sumber)", symbol, len(segments))
        for seg_id, df in enumerate(segments):
            df = df.reset_index(drop=True)
            feats = segment_features(df, halflife)
            labels = make_labels(df, spec)
            cols = [c for c in ("open_time", "open", "high", "low", "close", "volume") if c in df]
            parts = [df[cols], feats, labels] + ([extra(df)] if extra is not None else [])
            frame = pd.concat(parts, axis=1)
            frame["symbol"] = symbol
            frame["segment"] = f"{symbol}:{seg_id}"
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    if not keep_unlabeled:
        data = data.dropna(subset=MODEL_FEATURES + ["label"])
    data = data.sort_values(["open_time", "symbol"]).reset_index(drop=True)
    if not keep_unlabeled:
        data["label"] = data["label"].astype(int)
    return data


def make_splitter(n_splits: int = 5, spec: TargetSpec = SPEC, lookback: int = FEATURE_LOOKBACK
                  ) -> PurgedGroupTimeSeriesSplit:
    """Walk-forward CV dengan jarak = TargetSpec.min_embargo(lookback) grup waktu (menit)."""
    return PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge=spec.horizon, embargo=lookback)


def holdout_split(data: pd.DataFrame, frac: float = 0.2, spec: TargetSpec = SPEC,
                  lookback: int = FEATURE_LOOKBACK) -> tuple[np.ndarray, np.ndarray]:
    """Index (dev, holdout): holdout = `frac` waktu terakhir; dev berhenti `min_embargo` menit sebelumnya."""
    times = np.sort(data["open_time"].unique())
    cut = int(len(times) * (1 - frac))
    gap = spec.min_embargo(lookback)
    dev_end = times[max(cut - gap, 1)]
    dev = np.flatnonzero(data["open_time"].to_numpy() < dev_end)
    hold = np.flatnonzero(data["open_time"].to_numpy() >= times[cut])
    return dev, hold


def windows_overlap(train_times: np.ndarray, test_times: np.ndarray, lookback: int, horizon: int,
                    step_ms: int = 60_000) -> bool:
    """True bila jendela [t - lookback + 1, t + horizon] (candle) sebuah baris train beririsan dengan test."""
    if len(train_times) == 0 or len(test_times) == 0:
        return False
    tr = np.sort(np.asarray(train_times))
    test_times = np.asarray(test_times)
    lo = test_times - (lookback - 1) * step_ms  # awal jendela tiap baris test
    hi = test_times + horizon * step_ms          # akhir jendela tiap baris test
    tr_lo = tr - (lookback - 1) * step_ms
    tr_hi = tr + horizon * step_ms               # monoton naik karena tr terurut
    k = np.searchsorted(tr_lo, hi, side="right")  # jumlah train yang mulai sebelum akhir jendela test
    has = k > 0
    return bool((tr_hi[np.maximum(k - 1, 0)][has] >= lo[has]).any())
