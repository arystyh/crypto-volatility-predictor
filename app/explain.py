"""Kontributor utama sebuah prediksi (roadmap 4.4): kontribusi bawaan XGBoost (`pred_contribs`,
TreeSHAP exact untuk tree ensemble) pada skala log-odds.

Booster native dimuat MALAS (hanya saat ada alert), karena jalur inference normal memakai ONNX.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.reason import PHRASES

log = logging.getLogger(__name__)

LABELS = {
    "rv_ratio_15_240": "rasio volatilitas 15m/4j", "rv_ratio_5_60": "rasio volatilitas 5m/1j",
    "parkinson_ratio_15_240": "rasio range high-low 15m/4j", "atr_ratio_14_60": "rasio ATR 14/60",
    "volume_surge": "lonjakan volume", "volume_surge_5": "lonjakan volume 5m", "vpin_50": "VPIN (toksisitas order)",
    "ofi_15": "order flow imbalance 15m", "downside_share_15": "porsi volatilitas turun",
    "rv_15_ewmz": "volatilitas vs norma adaptif", "volume_surge_5_ewmz": "volume vs norma adaptif",
    "parkinson_15_ewmz": "range high-low vs norma adaptif", "log_ret_15": "return 15m",
}


class Contributions:
    def __init__(self, model_path: str | Path, features: list[str]):
        self.model_path = Path(model_path)
        self.features = features
        self._booster = None

    def _load(self):
        if self._booster is None:
            import xgboost as xgb

            self._booster = xgb.Booster()
            self._booster.load_model(str(self.model_path))
        return self._booster

    def top(self, features: pd.Series | dict, k: int = 3) -> list[dict]:
        import xgboost as xgb

        x = np.asarray([[float(features[f]) for f in self.features]], dtype=np.float32)
        contrib = self._load().predict(xgb.DMatrix(x, feature_names=self.features), pred_contribs=True)[0]
        pairs = sorted(zip(self.features, contrib[:-1]), key=lambda t: -t[1])
        out = []
        for name, c in pairs[:k]:
            if c <= 0:
                break
            v = float(features[name])
            hi, lo = PHRASES.get(name, (None, None))
            out.append({"feature": name, "value": v, "contribution": float(c),
                        "label": LABELS.get(name, name)})
        return out


def format_contrib(c: dict) -> str:
    v = c["value"]
    val = f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return f"{c['label']} = {val} (+{c['contribution']:.2f} log-odds)"
