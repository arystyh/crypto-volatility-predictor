"""Volatility Regime Classifier: Gaussian Hidden Markov Model 2-state (implementasi numpy murni).

State 0 = "Low Volatility / Compression", state 1 = "High Volatility / Expansion".
Observasi: x_t = log(realized vol 15 menit).

- Fit (Baum-Welch / EM) hanya dilakukan saat training -> models/hmm_regime.json
- Inference memakai FORWARD FILTERING (hanya data <= t, tanpa smoothing) sehingga
  aman dari look-ahead bias dan bisa dihitung real-time.

Gate untuk XGBoost (HMM_GATE):
    transition : panggil XGBoost hanya jika terjadi transisi Low -> High dalam
                 `gate_window` menit terakhir (sesuai desain)
    high       : panggil XGBoost selama rezim High
    off        : selalu panggil XGBoost
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OBS_FEATURE = "rv_15"
_EPS = 1e-12
LOW, HIGH = 0, 1
REGIME_NAMES = {LOW: "low_vol_compression", HIGH: "high_vol_expansion"}


def observations(features: pd.DataFrame | pd.Series) -> np.ndarray:
    col = features[OBS_FEATURE]
    return np.log(np.asarray(col, dtype=float) + _EPS)


def _gauss(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Likelihood N(x | mu_k, var_k) -> shape (T, K)."""
    x = x[:, None]
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


@dataclass
class GaussianHMM:
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    # ------------------------------------------------------------ fit
    @classmethod
    def fit(cls, x: np.ndarray, n_iter: int = 50, tol: float = 1e-4) -> "GaussianHMM":
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        lo, hi = np.quantile(x, [0.3, 0.85])
        model = cls(
            startprob=np.array([0.8, 0.2]),
            transmat=np.array([[0.99, 0.01], [0.03, 0.97]]),
            means=np.array([lo, hi]),
            variances=np.array([x.var(), x.var()]) / 2,
        )
        prev = -np.inf
        for _ in range(n_iter):
            ll, gamma, xi_sum = model._e_step(x)
            model._m_step(x, gamma, xi_sum)
            if abs(ll - prev) < tol * abs(ll):
                break
            prev = ll
        # pastikan state 1 = volatilitas tinggi
        if model.means[0] > model.means[1]:
            order = [1, 0]
            model = cls(model.startprob[order], model.transmat[np.ix_(order, order)],
                        model.means[order], model.variances[order])
        return model

    def _e_step(self, x):
        T, K = len(x), 2
        B = _gauss(x, self.means, self.variances) + 1e-300
        alpha = np.empty((T, K))
        scale = np.empty(T)
        alpha[0] = self.startprob * B[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        A = self.transmat
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t]
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / scale[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi_sum = np.zeros((K, K))
        for t in range(T - 1):
            xi = alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :] / scale[t + 1]
            xi_sum += xi
        return float(np.log(scale).sum()), gamma, xi_sum

    def _m_step(self, x, gamma, xi_sum):
        self.startprob = gamma[0] / gamma[0].sum()
        self.transmat = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        self.means = (gamma * x[:, None]).sum(axis=0) / w
        self.variances = np.maximum((gamma * (x[:, None] - self.means) ** 2).sum(axis=0) / w, 1e-6)

    # ------------------------------------------------------------ inference
    def filter(self, x: np.ndarray) -> np.ndarray:
        """P(state=High | x_1..x_t) untuk tiap t (forward filtering, online)."""
        x = np.asarray(x, dtype=float)
        B = _gauss(np.nan_to_num(x, nan=float(self.means.mean())), self.means, self.variances) + 1e-300
        B[~np.isfinite(x)] = 1.0  # observasi hilang -> tidak informatif
        p = np.empty(len(x))
        a = self.startprob.copy()
        for t in range(len(x)):
            a = (a @ self.transmat if t else a) * B[t]
            a /= a.sum()
            p[t] = a[HIGH]
        return p

    # ------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "startprob": self.startprob.tolist(),
            "transmat": self.transmat.tolist(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
            "observation": f"log({OBS_FEATURE})",
            "states": REGIME_NAMES,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "GaussianHMM":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["startprob"]), np.array(d["transmat"]), np.array(d["means"]), np.array(d["variances"]))


def gate_mask(p_high: np.ndarray, mode: str = "transition", window: int = 30, threshold: float = 0.5) -> np.ndarray:
    """True = XGBoost dipanggil. Untuk setiap t, hanya memakai p_high[<= t]."""
    high = p_high >= threshold
    if mode == "off":
        return np.ones_like(high, dtype=bool)
    if mode == "high":
        return high
    # transition: saat ini High DAN dalam `window` menit terakhir pernah Low
    low_recent = pd.Series(~high).rolling(window + 1, min_periods=1).max().astype(bool).values
    return high & low_recent


@dataclass
class RegimeState:
    p_high: float
    regime: str
    gate_open: bool
    transition: bool


def current_regime(hmm: GaussianHMM, features: pd.DataFrame, mode: str, window: int) -> RegimeState:
    """Regime untuk baris terakhir `features` (DataFrame fitur beberapa jam terakhir)."""
    p = hmm.filter(observations(features))
    gates = gate_mask(p, mode, window)
    trans = gate_mask(p, "transition", window)
    return RegimeState(
        p_high=float(p[-1]),
        regime=REGIME_NAMES[HIGH if p[-1] >= 0.5 else LOW],
        gate_open=bool(gates[-1]),
        transition=bool(trans[-1]),
    )
