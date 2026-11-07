"""Baseline sederhana pembanding XGBoost (roadmap 2.5). Semua dikonversi ke P(spike) dengan cara
yang SAMA: ramalan log-vol 15 menit ke depan -> z_hat = (ramalan - mu_t) / sd_t (baseline target
yang dibekukan) -> regresi logistik 1 fitur (dilatih di periode development).

    persistence  ramalan = log-Parkinson vol 15 menit terakhir (volatilitas sekarang akan berlanjut)
    HAR-RV       fwd_logvol ~ b0 + b1 lv_15m + b2 lv_1h + b3 lv_1d     (Corsi 2009, versi log)
    GARCH(1,1)   variance return 1m (paket `arch`, fit di development), rata-rata variance 15 langkah ke
                 depan, dipetakan ke skala log-Parkinson dengan regresi linier di development
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.ml.target import LN2X4, VAR_FLOOR

log = logging.getLogger(__name__)
BASELINES = ("persistence", "har_rv", "garch_1_1")


def baseline_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Kolom tambahan per segmen kontinu: log-vol trailing 1 jam & 1 hari, return 1m."""
    pk = pd.Series(np.log(df["high"].astype(float) / df["low"].astype(float)) ** 2 / LN2X4, index=df.index)
    out = pd.DataFrame(index=df.index)
    for name, w in (("lv_60", 60), ("lv_1440", 1440)):
        out[name] = 0.5 * np.log(pk.rolling(w, min_periods=w).mean().clip(lower=VAR_FLOOR))
    out["ret_1m"] = np.log(df["close"].astype(float)).diff()
    return out


def _garch_fit(returns: np.ndarray) -> dict:
    from arch import arch_model

    r = returns[np.isfinite(returns)] * 1000.0  # skala agar optimizer stabil
    res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
    p = res.params
    return {"mu": float(p["mu"]), "omega": float(p["omega"]), "alpha": float(p["alpha[1]"]),
            "beta": float(p["beta[1]"]), "scale": 1000.0}


def _garch_forecast(returns: np.ndarray, prm: dict, horizon: int) -> np.ndarray:
    """Rata-rata variance H langkah ke depan pada tiap t (informasi s.d. t saja)."""
    r = returns * prm["scale"] - prm["mu"]
    w, a, b = prm["omega"], prm["alpha"], prm["beta"]
    pers = a + b
    uncond = w / max(1 - pers, 1e-6)
    s2 = np.empty(len(r))
    prev = uncond
    for t in range(len(r)):
        e2 = r[t] ** 2 if np.isfinite(r[t]) else prev
        prev = w + a * e2 + b * prev   # sigma^2_{t+1 | t}
        s2[t] = prev
    k = np.arange(horizon)
    mult = (pers ** k).mean()
    return (uncond + mult * (s2 - uncond)) / prm["scale"] ** 2


class BaselineSuite:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.garch: dict[str, dict] = {}
        self.har = LinearRegression()
        self.garch_map = LinearRegression()
        self.clf: dict[str, LogisticRegression] = {}

    def _garch_lv(self, data: pd.DataFrame) -> np.ndarray:
        out = np.full(len(data), np.nan)
        for seg, g in data.groupby("segment"):
            g = g.sort_values("open_time")
            prm = self.garch.get(g["symbol"].iloc[0])
            if prm is None:
                continue
            var = _garch_forecast(g["ret_1m"].to_numpy(), prm, self.horizon)
            out[data.index.get_indexer(g.index)] = 0.5 * np.log(np.clip(var, VAR_FLOOR, None))
        return out

    def forecasts(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        X = data[["trail_logvol", "lv_60", "lv_1440"]].to_numpy()
        g = self._garch_lv(data)
        return {"persistence": data["trail_logvol"].to_numpy(),
                "har_rv": self.har.predict(X),
                "garch_1_1": self.garch_map.predict(g.reshape(-1, 1)) if np.isfinite(g).all() else
                np.where(np.isfinite(g), self.garch_map.predict(np.nan_to_num(g).reshape(-1, 1)), np.nan)}

    def _z(self, data: pd.DataFrame, f: np.ndarray) -> np.ndarray:
        return ((f - data["baseline_mu"].to_numpy()) / data["baseline_sd"].to_numpy()).reshape(-1, 1)

    def fit(self, dev: pd.DataFrame) -> "BaselineSuite":
        dev = dev.dropna(subset=["lv_60", "lv_1440", "ret_1m"])
        for sym, g in dev.groupby("symbol"):
            self.garch[sym] = _garch_fit(g.sort_values("open_time")["ret_1m"].to_numpy())
            log.info("GARCH(1,1) %s: %s", sym, {k: round(v, 5) for k, v in self.garch[sym].items()})
        self.har.fit(dev[["trail_logvol", "lv_60", "lv_1440"]].to_numpy(), dev["fwd_logvol"].to_numpy())
        g = self._garch_lv(dev)
        ok = np.isfinite(g)
        self.garch_map.fit(g[ok].reshape(-1, 1), dev["fwd_logvol"].to_numpy()[ok])
        for name, f in self.forecasts(dev).items():
            z = self._z(dev, f)
            ok = np.isfinite(z[:, 0])
            self.clf[name] = LogisticRegression().fit(z[ok], dev["label"].to_numpy()[ok])
        return self

    def predict(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for name, f in self.forecasts(data).items():
            z = self._z(data, f)
            p = np.full(len(data), np.nan)
            ok = np.isfinite(z[:, 0])
            p[ok] = self.clf[name].predict_proba(z[ok])[:, 1]
            out[name] = p
        return out
