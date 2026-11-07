"""Model hazard diskrit: KAPAN spike mulai, per menit, bukan sekadar ya/tidak.

    h_k(x_t)   = P(onset = t+k | belum onset sampai t+k-1, x_t),   k = 1..H
    S(k)       = prod_{j<=k} (1 - h_j)                              survival
    P(onset<=H)= 1 - S(H)
    median onset = k terkecil dengan S(k) <= 0.5 (jika ada)

Dilatih sebagai klasifikasi "person-period": setiap waktu keputusan t diekspansi menjadi
baris (x_t, k) untuk k = 1..min(onset, H) dengan event = 1 hanya pada k = onset.
Model: XGBoost (ONNX bila tersedia) dengan fitur tambahan `k`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def expand_person_period(X: np.ndarray, onset: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """X: (n, f); onset: 0 = tidak ada spike dalam H, 1..H = menit onset. Return (X_pp, y_pp)."""
    onset = onset.astype(int)
    last = np.where(onset > 0, onset, horizon)
    reps = last
    idx = np.repeat(np.arange(len(X)), reps)
    k = np.concatenate([np.arange(1, r + 1) for r in reps]) if len(reps) else np.array([], int)
    y = ((onset[idx] > 0) & (k == onset[idx])).astype(int)
    return np.column_stack([X[idx], k.astype(np.float32)]), y


@dataclass
class HazardCurve:
    hazard: list[float]
    p_onset_within_h: float
    median_onset_min: int | None
    most_likely_min: int
    cumulative: list[float] | None = None   # P(onset <= k), k = 1..H; monoton naik by construction

    def p_within(self, k: int) -> float:
        """P(spike mulai dalam <= k menit) = 1 - prod_{j<=k} (1 - h_j). Selalu monoton dalam k."""
        return float(self.cumulative[min(k, len(self.cumulative)) - 1])

    def to_dict(self) -> dict:
        return {"hazard": [round(h, 5) for h in self.hazard], "p_onset_within_h": round(self.p_onset_within_h, 4),
                "p_within": {k: round(self.p_within(k), 4) for k in (5, 10, 15) if k <= len(self.hazard)},
                "median_onset_min": self.median_onset_min, "most_likely_min": self.most_likely_min}


def curve_from_hazard(h: np.ndarray) -> HazardCurve:
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    surv = np.cumprod(1.0 - h)
    pdf = h * np.r_[1.0, surv[:-1]]
    below = np.flatnonzero(surv <= 0.5)
    return HazardCurve(
        hazard=h.tolist(),
        p_onset_within_h=float(1.0 - surv[-1]),
        median_onset_min=int(below[0] + 1) if below.size else None,
        most_likely_min=int(pdf.argmax() + 1),
        cumulative=(1.0 - surv).tolist(),
    )


class HazardModel:
    def __init__(self, path: str | Path, features: list[str]):
        self.path = Path(path)
        meta_path = self.path.with_name("hazard_metadata.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.horizon = int(meta.get("horizon", 15))
        self.features = meta.get("features", features)
        self.backend = None
        onnx_path = self.path.with_suffix(".onnx")
        if onnx_path.exists():
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                self._sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
                self._input = self._sess.get_inputs()[0].name
                self.backend = "onnx"
            except ImportError:
                pass
        if self.backend is None:
            import xgboost as xgb

            self._booster = xgb.Booster()
            self._booster.load_model(str(self.path))
            self.backend = "xgboost"

    def predict_hazard(self, features: pd.Series) -> np.ndarray:
        return self.predict_hazard_matrix(features[self.features].to_numpy(dtype=np.float32)[None, :])[0]

    def predict_hazard_matrix(self, X: np.ndarray) -> np.ndarray:
        """Hazard h_1..h_H untuk banyak baris sekaligus: X (n, fitur) -> (n, H). Dipakai batch (insiden)."""
        X = np.asarray(X, dtype=np.float32)
        n = len(X)
        Z = np.column_stack([np.repeat(X, self.horizon, axis=0),
                             np.tile(np.arange(1, self.horizon + 1, dtype=np.float32), n)])
        if self.backend == "onnx":
            out = self._sess.run(None, {self._input: Z})[1]
            if isinstance(out, list) and out and isinstance(out[0], dict):
                h = np.array([d[1] for d in out], dtype=float)
            else:
                h = np.asarray(out, dtype=float)[:, 1]
        else:
            import xgboost as xgb

            h = self._booster.predict(xgb.DMatrix(Z))
        return h.reshape(n, self.horizon)

    def curve(self, features: pd.Series) -> HazardCurve:
        return curve_from_hazard(self.predict_hazard(features))
