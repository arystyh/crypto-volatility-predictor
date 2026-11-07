"""Fallback rule-based (safety net) bila model ML error / timeout / tidak ter-load.

Aturan murni berbasis threshold, tanpa ML, agar notifikasi Telegram tetap jalan:

    SPIKE jika  (rv_ratio_15_240 >= RV_RATIO  ATAU  atr_ratio_14_60 >= ATR_RATIO
                 ATAU  parkinson_ratio_15_240 >= PARKINSON_RATIO)
           DAN  volume_surge_5 >= VOLUME_SURGE         (konfirmasi volume, hindari fakeout)

Mengembalikan "probabilitas" 1.0 (spike) atau 0.0 supaya bisa diproses oleh logika
threshold yang sama.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FallbackRule:
    rv_ratio: float = 2.0
    atr_ratio: float = 1.8
    parkinson_ratio: float = 2.0
    volume_surge: float = 1.5

    def triggered(self, f: pd.Series | dict) -> bool:
        vol_expanding = (
            f["rv_ratio_15_240"] >= self.rv_ratio
            or f["atr_ratio_14_60"] >= self.atr_ratio
            or f["parkinson_ratio_15_240"] >= self.parkinson_ratio
        )
        return bool(vol_expanding and f["volume_surge_5"] >= self.volume_surge)

    def score(self, f: pd.Series | dict) -> float:
        return 1.0 if self.triggered(f) else 0.0

    def score_frame(self, df: pd.DataFrame) -> pd.Series:
        vol_expanding = (
            (df["rv_ratio_15_240"] >= self.rv_ratio)
            | (df["atr_ratio_14_60"] >= self.atr_ratio)
            | (df["parkinson_ratio_15_240"] >= self.parkinson_ratio)
        )
        return (vol_expanding & (df["volume_surge_5"] >= self.volume_surge)).astype(float)


def _parkinson_np(high: np.ndarray, low: np.ndarray) -> float:
    hl2 = np.log(high / low) ** 2
    return float(np.sqrt(hl2.mean() / (4.0 * np.log(2.0))))


def heuristic_from_candles(candles: pd.DataFrame, rule: FallbackRule | None = None) -> tuple[float, dict]:
    """Heuristic engine murni NumPy langsung dari OHLCV mentah (tanpa feature pipeline pandas).

    Dipakai saat circuit breaker OPEN: tetap jalan walau feature engine/model bermasalah.
    SPIKE jika Parkinson(15m)/Parkinson(240m) >= rule.parkinson_ratio DAN volume surge 5m >= rule.volume_surge.
    """
    rule = rule or FallbackRule()
    h = candles["high"].to_numpy(dtype=float)
    l = candles["low"].to_numpy(dtype=float)
    v = candles["volume"].to_numpy(dtype=float)
    if len(h) < 30:
        return 0.0, {"reason": "data kurang"}
    long_w = min(240, len(h))
    pk_short = _parkinson_np(h[-15:], l[-15:])
    pk_long = _parkinson_np(h[-long_w:], l[-long_w:])
    ratio = pk_short / (pk_long + 1e-12)
    surge = v[-5:].mean() / (v[-20:].mean() + 1e-12)
    spike = ratio >= rule.parkinson_ratio and surge >= rule.volume_surge
    return (1.0 if spike else 0.0), {"parkinson_ratio_15_240": ratio, "volume_surge_5": surge, "parkinson_15": pk_short}
