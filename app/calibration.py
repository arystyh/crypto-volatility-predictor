"""Kalibrasi isotonic (roadmap 4.2).

Dilatih pada fold validasi (bagian akhir periode development, tidak pernah holdout) dan disimpan
sebagai fungsi piecewise-linear (titik x -> y) di models/calibration.json, sehingga live cukup
`np.interp` tanpa scikit-learn. Probabilitas yang dikalibrasi inilah yang dibandingkan dengan
threshold alert dan dicatat di ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class IsotonicCalibrator:
    def __init__(self, x: list[float], y: list[float], meta: dict | None = None):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.meta = meta or {}

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return cls(iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist())

    def __call__(self, p):
        return np.clip(np.interp(p, self.x, self.y), 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"x": self.x.tolist(), "y": self.y.tolist(), "meta": self.meta}))

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator | None":
        p = Path(path)
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return cls(d["x"], d["y"], d.get("meta"))


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Reliability diagram berbasis kuantil (bin berisi jumlah sampel sama)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            out.append({"p_mean": float(p[m].mean()), "freq": float(y[m].mean()), "n": int(m.sum())})
    return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
