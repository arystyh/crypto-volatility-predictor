"""Generator data candle 1m sintetis (fallback bila Binance tidak bisa diakses saat training).

Model: GARCH(1,1) + regime switching (tenang/bergejolak) + pola musiman intraday
(jam buka pasar US lebih volatil) + volume yang berkorelasi dengan volatilitas.
Data ini HANYA untuk demo/CI. Untuk model produksi, latih ulang dengan data Binance:
    python -m training.train --source binance --days 90
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_candles(
    symbol: str = "BTCUSDT",
    days: int = 60,
    start_price: float = 60_000.0,
    seed: int = 7,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 1440
    open_time = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    hour = open_time.hour.values + open_time.minute.values / 60

    # Musiman intraday: puncak di sekitar pembukaan US (13:30-16 UTC), sepi di 3-6 UTC
    season = 1.0 + 0.6 * np.exp(-((hour - 14.5) ** 2) / 3.0) - 0.25 * np.exp(-((hour - 4.5) ** 2) / 4.0)

    # Regime switching Markov: 0=tenang, 1=bergejolak
    p_up, p_down = 1 / 2500, 1 / 180
    regime = np.zeros(n, dtype=np.int8)
    for t in range(1, n):
        prev = regime[t - 1]
        regime[t] = (1 if rng.random() < p_up else 0) if prev == 0 else (0 if rng.random() < p_down else 1)
    regime_mult = np.where(regime == 1, 2.8, 1.0)

    # GARCH(1,1) pada inovasi standar
    omega, alpha, beta = 0.02, 0.08, 0.90
    base_sigma = 0.0006
    z = rng.standard_t(df=4, size=n) / np.sqrt(2.0)  # fat tails, var=1
    h = np.empty(n)
    eps = np.empty(n)
    h[0] = 1.0
    for t in range(n):
        if t > 0:
            h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        eps[t] = np.sqrt(h[t]) * z[t]
    sigma = base_sigma * season * regime_mult
    ret = sigma * eps

    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 0.6, size=(2, n))) * sigma * np.sqrt(h)
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])

    vol_level = (sigma * np.sqrt(h)) / base_sigma
    volume = 50 * vol_level**1.2 * rng.lognormal(0, 0.35, n)

    # Aliran order: porsi beli agresif condong ke arah return, makin "toxic" saat volatil
    buy_share = np.clip(0.5 + 0.35 * np.tanh(ret / (sigma + 1e-12)) * (0.6 + 0.4 * (regime == 1))
                        + rng.normal(0, 0.08, n), 0.02, 0.98)
    taker_buy = volume * buy_share

    ms = open_time.as_unit("ms").asi8 if hasattr(open_time, "as_unit") else open_time.asi8 // 1_000_000
    return pd.DataFrame(
        {
            "symbol": symbol,
            "open_time": ms,
            "close_time": ms + 59_999,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trades": (volume * 20).astype(int),
            "taker_buy_volume": taker_buy,
        }
    )
