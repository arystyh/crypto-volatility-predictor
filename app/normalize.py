"""Normalisasi online EWM (pengganti StandardScaler statis), half-life bisa dikonfigurasi.

Rekursi yang SAMA dipakai saat training (berurutan per segmen kontinu, training/dataset.py) dan
saat live (state per simbol di Redis `norm:<SYMBOL>`), dibuktikan oleh tests/test_parity.py.

    a_t    = max(alpha, 1 / (n + 1))     # awal: rata-rata kumulatif (Welford), lalu EWM murni
    z_t    = (x_t - mean_{t-1}) / sqrt(var_{t-1})      # dihitung SEBELUM update (kausal)
    diff   = x_t - mean_{t-1}
    mean_t = mean_{t-1} + a_t * diff
    var_t  = (1 - a_t) * (var_{t-1} + a_t * diff^2)     # EW variance (West, 1979)

alpha = 1 - 0.5 ** (1 / half_life). Default half-life 1440 menit (1 hari, EWM_HALFLIFE_MIN).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from app.features import FEATURE_COLUMNS

NORM_SOURCES = ["rv_15", "parkinson_15", "volume_surge_5", "vpin_50", "rv_down_15"]
NORM_FEATURES = [f"{c}_ewmz" for c in NORM_SOURCES]
MODEL_FEATURES = FEATURE_COLUMNS + NORM_FEATURES

HALFLIFE_MIN = 1440
WARMUP = 30
Z_CLIP = 10.0


def alpha_from_halflife(halflife: float) -> float:
    return 1.0 - math.exp(math.log(0.5) / halflife)


class EWMNormalizer:
    def __init__(self, halflife: float = HALFLIFE_MIN, state: dict | None = None):
        self.halflife = float(halflife)
        self.alpha = alpha_from_halflife(halflife)
        # state: {feature: [mean, var, n]}, plus "_last_time" dan "_halflife"
        self.state: dict = state or {}
        if self.state.get("_halflife") not in (None, self.halflife):
            raise ValueError(f"state EWM dibuat dengan half-life {self.state['_halflife']}, bukan {self.halflife}")
        self.state["_halflife"] = self.halflife

    @property
    def last_time(self) -> int:
        return int(self.state.get("_last_time", -1))

    @property
    def n(self) -> int:
        return int(self.state.get(NORM_SOURCES[0], [0, 0, 0])[2])

    def transform(self, row: dict | pd.Series, update: bool, open_time: int | None = None) -> dict:
        out = {}
        for src in NORM_SOURCES:
            x = float(row[src])
            mean, var, n = self.state.get(src, [x, 0.0, 0])
            z = 0.0 if n < WARMUP or var <= 0 else (x - mean) / math.sqrt(var)
            out[f"{src}_ewmz"] = float(np.clip(z, -Z_CLIP, Z_CLIP))
            if update and math.isfinite(x):
                a = max(self.alpha, 1.0 / (n + 1))
                diff = x - mean
                self.state[src] = [mean + a * diff, (1 - a) * (var + a * diff * diff), n + 1]
        if update and open_time is not None:
            self.state["_last_time"] = int(open_time)
        return out

    def transform_frame(self, feats: pd.DataFrame, open_times: pd.Series | None = None) -> pd.DataFrame:
        """Proses berurutan (training / warm-up). Baris dengan NaN tidak meng-update state."""
        rows = []
        times = open_times.values if open_times is not None else [None] * len(feats)
        values = feats[NORM_SOURCES].values
        for i in range(len(feats)):
            r = dict(zip(NORM_SOURCES, values[i]))
            if any(not math.isfinite(v) for v in r.values()):
                rows.append({f: np.nan for f in NORM_FEATURES})
                continue
            rows.append(self.transform(r, update=True, open_time=times[i]))
        return pd.DataFrame(rows, index=feats.index, columns=NORM_FEATURES)

    def dumps(self) -> str:
        return json.dumps(self.state)

    @classmethod
    def loads(cls, raw: str | None, halflife: float = HALFLIFE_MIN) -> "EWMNormalizer":
        return cls(halflife=halflife, state=json.loads(raw) if raw else None)
