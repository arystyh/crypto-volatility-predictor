"""Proses Hawkes multivariat (kernel eksponensial) untuk transaksi besar & likuidasi.

Dua jenis event (D = 2):
    0 = transaksi besar (aggTrade dengan notional >= LARGE_TRADE_USD)
    1 = likuidasi (Binance Futures `<symbol>@forceOrder`; stream ini SNAPSHOT 1 detik -> sampel, bukan total)

    lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij * exp(-beta * (t - t_k^j))

    alpha_ij : lonjakan intensitas tipe i oleh satu event tipe j
               alpha_01 = eksitasi silang likuidasi -> transaksi besar (cascade ke pasar)
    beta     : laju peluruhan bersama (1/detik)
    branching ratio = radius spektral matriks alpha / beta  (< 1 agar stasioner; -> 1 = mendekati kritis)

Intensitas di-update O(1) per event (rekursi eksponensial, simpan sum per sumber).
Parameter di-fit OFFLINE dengan MLE (scripts/fit_hawkes.py, mingguan) dan disimpan bersama
versi model (models/hawkes_params.json); live hanya memakai parameter itu.

Fitur per menit (live): hawkes_trade_intensity, hawkes_liq_intensity, hawkes_excitation
(lambda_trade / mu_trade), hawkes_cross_liq_to_trade (bagian lambda_trade yang dipicu likuidasi),
hawkes_branching.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

TRADE, LIQ = 0, 1


@dataclass
class HawkesParams:
    mu: list[float] = field(default_factory=lambda: [0.05, 0.01])
    alpha: list[list[float]] = field(default_factory=lambda: [[0.3, 0.4], [0.05, 0.3]])
    beta: float = 1.0
    model_version: str | None = None
    fitted_at: str | None = None
    n_events: list[int] | None = None

    @property
    def dim(self) -> int:
        return len(self.mu)

    @property
    def branching(self) -> float:
        a = np.asarray(self.alpha, dtype=float) / self.beta
        return float(np.max(np.abs(np.linalg.eigvals(a)))) if self.beta > 0 else math.nan

    def to_dict(self) -> dict:
        return {"mu": list(self.mu), "alpha": [list(r) for r in self.alpha], "beta": self.beta,
                "branching": self.branching, "model_version": self.model_version, "fitted_at": self.fitted_at,
                "n_events": self.n_events}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HawkesParams":
        d = json.loads(Path(path).read_text())
        return cls(mu=d["mu"], alpha=d["alpha"], beta=d["beta"], model_version=d.get("model_version"),
                   fitted_at=d.get("fitted_at"), n_events=d.get("n_events"))


class OnlineHawkes:
    """Intensitas O(1) per event. `_src[i][j]` = eksitasi tipe i yang berasal dari event tipe j."""

    def __init__(self, params: HawkesParams | None = None):
        self.params = params or HawkesParams()
        d = self.params.dim
        self._src = np.zeros((d, d))
        self._last: float | None = None
        self.counts = [0] * d

    def _decay_to(self, t: float) -> None:
        if self._last is not None and t > self._last:
            self._src *= math.exp(-self.params.beta * (t - self._last))
        self._last = t if self._last is None else max(self._last, t)

    def add_event(self, t: float, kind: int = TRADE) -> None:
        self._decay_to(t)
        self._src[:, kind] += np.asarray(self.params.alpha, dtype=float)[:, kind]
        self.counts[kind] += 1

    def intensity(self, t: float, kind: int = TRADE) -> float:
        self._decay_to(t)
        return float(self.params.mu[kind] + self._src[kind].sum())

    def cross(self, t: float, target: int = TRADE, source: int = LIQ) -> float:
        self._decay_to(t)
        return float(self._src[target, source])

    def features(self, t: float) -> dict:
        lam_t = self.intensity(t, TRADE)
        out = {"hawkes_trade_intensity": round(lam_t, 6),
               "hawkes_excitation": round(lam_t / self.params.mu[TRADE], 4) if self.params.mu[TRADE] > 0 else None,
               "hawkes_branching": round(self.params.branching, 4)}
        if self.params.dim > 1:
            lam_l = self.intensity(t, LIQ)
            out.update({"hawkes_liq_intensity": round(lam_l, 6),
                        "hawkes_liq_excitation": round(lam_l / self.params.mu[LIQ], 4) if self.params.mu[LIQ] > 0 else None,
                        "hawkes_cross_liq_to_trade": round(self.cross(t, TRADE, LIQ), 6)})
        return out


# ---------------------------------------------------------------------------- MLE
def _unpack(theta: np.ndarray, d: int):
    x = np.exp(theta)
    return x[:d], x[d:d + d * d].reshape(d, d), x[-1]


def neg_loglik(theta: np.ndarray, t: np.ndarray, kind: np.ndarray, T: float, d: int) -> float:
    """Negative log-likelihood Hawkes multivariat eksponensial (beta bersama), O(n * d)."""
    mu, alpha, beta = _unpack(theta, d)
    if np.max(np.abs(np.linalg.eigvals(alpha / beta))) >= 0.999:
        return 1e12
    A = np.zeros(d)  # sum_{t_k^j < t} exp(-beta (t - t_k^j)) per sumber j
    ll, prev = 0.0, None
    for ti, ki in zip(t, kind):
        if prev is not None:
            A *= math.exp(-beta * (ti - prev))
        lam = mu[ki] + alpha[ki] @ A
        if lam <= 0:
            return 1e12
        ll += math.log(lam)
        A[ki] += 1.0
        prev = ti
    tail = np.array([np.sum(1.0 - np.exp(-beta * (T - t[kind == j]))) for j in range(d)])
    comp = mu.sum() * T + (alpha.sum(axis=0) / beta) @ tail
    return -(ll - comp)


def fit_hawkes(times, kinds=None, horizon: float | None = None, dim: int | None = None) -> HawkesParams:
    """MLE dari event (detik, terurut) dengan label tipe. `kinds=None` -> univariat."""
    t = np.asarray(times, dtype=float)
    kind = np.zeros(len(t), dtype=int) if kinds is None else np.asarray(kinds, dtype=int)
    order = np.argsort(t, kind="stable")
    t, kind = t[order], kind[order]
    d = dim or int(kind.max() + 1 if len(kind) else 1)
    if len(t) < 20:
        return HawkesParams(mu=[0.01] * d, alpha=[[0.1] * d for _ in range(d)], beta=1.0)
    t = t - t[0]
    T = horizon if horizon is not None else t[-1] + 1e-9
    rate = np.array([max((kind == j).sum(), 1) / max(T, 1.0) for j in range(d)])
    x0 = np.log(np.r_[rate * 0.5, np.full(d * d, 0.2), 1.0])
    res = minimize(neg_loglik, x0, args=(t, kind, T, d), method="L-BFGS-B",
                   options={"maxiter": 500})
    if not res.success:
        res = minimize(neg_loglik, res.x, args=(t, kind, T, d), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4})
    mu, alpha, beta = _unpack(res.x, d)
    return HawkesParams(mu=mu.tolist(), alpha=alpha.tolist(), beta=float(beta),
                        n_events=[int((kind == j).sum()) for j in range(d)])


def simulate_hawkes(params: HawkesParams, T: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulasi Ogata thinning multivariat. Return (times, kinds)."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(params.mu, dtype=float)
    alpha = np.asarray(params.alpha, dtype=float)
    d = len(mu)
    t, src = 0.0, np.zeros((d, d))
    times, kinds = [], []
    while True:
        lam_bar = mu.sum() + src.sum()
        w = rng.exponential(1.0 / lam_bar)
        t += w
        if t >= T:
            break
        src *= math.exp(-params.beta * w)
        lam = mu + src.sum(axis=1)
        u = rng.random() * lam_bar
        if u <= lam.sum():
            k = int(np.searchsorted(np.cumsum(lam), u))
            times.append(t)
            kinds.append(k)
            src[:, k] += alpha[:, k]
    return np.array(times), np.array(kinds, dtype=int)
