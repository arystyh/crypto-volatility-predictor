"""Fitur lead-lag perpetual -> spot (roadmap 2.3).

Hipotesis: harga perpetual futures bergerak beberapa detik lebih dulu daripada spot (leverage &
likuiditas lebih besar). Fitur dihitung saat candle close dari buffer harga transaksi:

    perp_ret_{1,5,15}s, spot_ret_{1,5,15}s = ln(p(t) / p(t - d))
    perp_spot_gap_5s   = perp_ret_5s - spot_ret_5s       (perp sudah bergerak, spot belum)
    basis              = mark / spot - 1
    basis_chg_1m       = basis(t) - basis(t - 60 s)

Fitur ini HANYA LIVE dan TIDAK masuk model sampai `scripts/leadlag_report.py` mengukur lag yang
nyata dari rekaman tick (cross-correlation return 1 detik). Bila tidak ada lag yang signifikan,
fitur dihapus dari model (lihat reports/leadlag.md).
"""
from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np

HORIZONS_S = (1, 5, 15)


class PriceBuffer:
    """Harga terakhir per waktu (ms). `price_at(t)` = harga transaksi terakhir pada/sebelum t."""

    def __init__(self, keep_ms: int = 120_000):
        self.keep_ms = keep_ms
        self.t: deque[int] = deque()
        self.p: deque[float] = deque()

    def add(self, t_ms: int, price: float) -> None:
        if self.t and t_ms < self.t[-1]:
            t_ms = self.t[-1]  # jaga monoton (urutan event antar koneksi bisa sedikit tertukar)
        self.t.append(int(t_ms))
        self.p.append(float(price))
        while self.t and self.t[0] < t_ms - self.keep_ms:
            self.t.popleft()
            self.p.popleft()

    def price_at(self, t_ms: int) -> float | None:
        i = bisect.bisect_right(self.t, t_ms) - 1
        return self.p[i] if i >= 0 else None


def _ret(buf: PriceBuffer, t: int, d_s: int) -> float | None:
    a, b = buf.price_at(t - d_s * 1000), buf.price_at(t)
    if a is None or b is None or a <= 0:
        return None
    return math.log(b / a)


class LeadLagState:
    def __init__(self):
        self.spot = PriceBuffer()
        self.perp = PriceBuffer()
        self.basis_hist: deque[tuple[int, float]] = deque(maxlen=600)
        self.mark: float | None = None

    def add_spot(self, t_ms: int, price: float) -> None:
        self.spot.add(t_ms, price)

    def add_perp(self, t_ms: int, price: float) -> None:
        self.perp.add(t_ms, price)

    def add_mark(self, t_ms: int, mark: float) -> None:
        self.mark = mark
        s = self.spot.price_at(t_ms)
        if s:
            self.basis_hist.append((t_ms, mark / s - 1.0))

    def features(self, t_ms: int) -> dict:
        out: dict = {}
        for d in HORIZONS_S:
            out[f"perp_ret_{d}s"] = _ret(self.perp, t_ms, d)
            out[f"spot_ret_{d}s"] = _ret(self.spot, t_ms, d)
        if out["perp_ret_5s"] is not None and out["spot_ret_5s"] is not None:
            out["perp_spot_gap_5s"] = out["perp_ret_5s"] - out["spot_ret_5s"]
        if self.basis_hist:
            ts = [x[0] for x in self.basis_hist]
            i_now = bisect.bisect_right(ts, t_ms) - 1
            i_ago = bisect.bisect_right(ts, t_ms - 60_000) - 1
            if i_now >= 0:
                out["basis"] = self.basis_hist[i_now][1]
                if i_ago >= 0:
                    out["basis_chg_1m"] = self.basis_hist[i_now][1] - self.basis_hist[i_ago][1]
        return {k: (None if v is None else round(v, 8)) for k, v in out.items()}


def lagged_xcorr(spot_t: np.ndarray, spot_p: np.ndarray, perp_t: np.ndarray, perp_p: np.ndarray,
                 max_lag_s: int = 10, step_ms: int = 1000) -> dict:
    """Cross-correlation return 1 detik: corr(perp_ret(t), spot_ret(t + lag)). lag > 0 = perp memimpin."""
    t0 = max(spot_t.min(), perp_t.min())
    t1 = min(spot_t.max(), perp_t.max())
    grid = np.arange(t0 + step_ms, t1, step_ms)
    if len(grid) < 100:
        raise ValueError("data terlalu sedikit untuk cross-correlation")

    def resample(t, p):
        idx = np.searchsorted(t, grid, side="right") - 1
        return np.log(p[np.clip(idx, 0, None)])

    rs = np.diff(resample(spot_t, spot_p))
    rp = np.diff(resample(perp_t, perp_p))
    lags = list(range(-max_lag_s, max_lag_s + 1))
    corr = {}
    for L in lags:
        if L >= 0:
            a, b = rp[: len(rp) - L], rs[L:]
        else:
            a, b = rp[-L:], rs[: len(rs) + L]
        corr[L] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    best = max(corr, key=lambda k: corr[k])
    n = len(rs)
    se = 1.0 / math.sqrt(n)  # ~ standar error korelasi di bawah H0
    return {"n_seconds": n, "corr_by_lag": corr, "best_lag_s": best, "best_corr": corr[best],
            "corr_at_0": corr[0], "significant": best > 0 and corr[best] - max(corr[0], 0) > 3 * se}
