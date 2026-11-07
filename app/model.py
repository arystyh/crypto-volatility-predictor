"""Model inference.

Champion: XGBoost. Default dijalankan via ONNX Runtime (models/xgb_volatility.onnx) dengan
input array NumPy float32 1 baris — tanpa meng-import xgboost sama sekali, sehingga RAM &
latensi container jauh lebih kecil. Bila file ONNX/onnxruntime tidak ada, otomatis memakai
XGBoost native (INFERENCE_BACKEND=xgboost untuk memaksa).

Challenger (shadow): LightGBM (models/challenger_lgbm.txt), hanya dicatat, tidak memicu alert.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _load_metadata(model_path: Path) -> dict:
    meta_path = model_path.with_name("metadata.json")
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}


def _as_matrix(features: pd.Series | pd.DataFrame, names: list[str]) -> np.ndarray:
    if isinstance(features, pd.Series):
        return np.asarray([features[names].to_numpy(dtype=np.float32)], dtype=np.float32)
    return features[names].to_numpy(dtype=np.float32)


class VolatilityModel:
    def __init__(self, model_path: str | Path, backend: str = "auto"):
        self.model_path = Path(model_path)
        if not self.model_path.exists() and not self.model_path.with_suffix(".onnx").exists():
            raise FileNotFoundError(f"Model tidak ditemukan di {self.model_path}. Jalankan: python -m training.train")
        self.metadata = _load_metadata(self.model_path)
        if "features" not in self.metadata:
            raise ValueError("metadata.json tidak berisi daftar fitur; latih ulang model")
        self.features: list[str] = self.metadata["features"]
        self.backend = None
        onnx_path = self.model_path.with_suffix(".onnx")
        if backend in ("auto", "onnx") and onnx_path.exists():
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self._sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
                self._input = self._sess.get_inputs()[0].name
                self.backend = "onnx"
            except ImportError:
                log.warning("onnxruntime tidak terpasang, memakai XGBoost native")
        if self.backend is None:
            import xgboost as xgb

            self._booster = xgb.Booster()
            self._booster.load_model(str(self.model_path))
            self.backend = "xgboost"
        log.info("Model champion dimuat (%s), %d fitur", self.backend, len(self.features))

    def predict_proba(self, features: pd.Series | pd.DataFrame) -> np.ndarray:
        X = _as_matrix(features, self.features)
        if self.backend == "onnx":
            out = self._sess.run(None, {self._input: X})[1]
            if isinstance(out, list) and out and isinstance(out[0], dict):  # ZipMap
                return np.array([d[1] for d in out], dtype=float)
            return np.asarray(out, dtype=float)[:, 1]
        import xgboost as xgb

        return self._booster.predict(xgb.DMatrix(X, feature_names=self.features))

    def predict_one(self, features: pd.Series) -> float:
        return float(self.predict_proba(features)[0])


class ChallengerModel:
    """LightGBM challenger untuk shadow deployment."""

    name = "lightgbm"

    def __init__(self, path: str | Path, features: list[str]):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=str(path))
        self.features = features

    def predict_one(self, features: pd.Series) -> float:
        return float(self.booster.predict(_as_matrix(features, self.features))[0])
