"""Feature engine: sinyal statistik & mikrostruktur (bukan indikator lagging).

Semua fungsi bekerja di atas DataFrame candle 1-menit dengan kolom
open_time (ms), open, high, low, close, volume — urut waktu naik.
Fungsi yang SAMA dipakai di training dan inference real-time sehingga tidak ada
training/serving skew. Semua fitur hanya memakai data <= t (tanpa look-ahead).

Kelompok fitur:
- Log return (1m, 5m, 15m)
- Realized volatility multi-timeframe: std log return 5m, 15m, 1h, 4h
- Semi-variance: upside vs downside volatility (pump vs dump punya risiko berbeda)
- Rasio volatilitas pendek/panjang (kompresi/akumulasi energi sebelum breakout)
- Parkinson & Garman-Klass estimator (memakai High/Low/Open, lebih efisien dari close-to-close)
- ATR (Average True Range) dinormalisasi harga
- Volume surge = Volume_t / SMA(Volume, 20) + interaksi volume x return
- Stationary Wavelet Transform (Haar a trous) KAUSAL: denoising tanpa artefak batas kanan
- Mikrostruktur: VPIN (bulk volume classification dari taker-buy volume) & order-flow imbalance
- Fitur kalender siklikal: sin/cos jam dan hari dalam minggu
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    # return
    "log_ret_1",
    "log_ret_5",
    "log_ret_15",
    # realized volatility multi-timeframe
    "rv_5",
    "rv_15",
    "rv_60",
    "rv_240",
    "rv_ratio_5_60",
    "rv_ratio_15_60",
    "rv_ratio_15_240",
    "rv_ratio_60_240",
    # semi-variance: volatilitas dua arah
    "rv_down_15",
    "rv_up_15",
    "rv_down_60",
    "rv_up_60",
    "downside_share_15",
    "downside_share_60",
    # wavelet (SWT kausal) denoising
    "swt_rv_15",
    "swt_noise_ratio_15",
    "swt_trend_15",
    # mikrostruktur: VPIN & order-flow imbalance (dari taker buy volume)
    "vpin_50",
    "vpin_ratio_15_240",
    "ofi_5",
    "ofi_15",
    # range-based estimators
    "parkinson_15",
    "parkinson_60",
    "parkinson_ratio_15_240",
    "garman_klass_15",
    "garman_klass_60",
    "gk_ratio_15_240",
    # ATR
    "atr_14_pct",
    "atr_ratio_14_60",
    # volume
    "volume_surge",
    "volume_surge_5",
    "volume_z_60",
    "signed_volume_surge",
    # kalender siklikal
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

LONG_WINDOW = 240                 # 4 jam
MIN_CANDLES = LONG_WINDOW + 20    # candle minimum untuk 1 vektor fitur lengkap

_EPS = 1e-12
_PARKINSON_K = 1.0 / (4.0 * math.log(2.0))
_GK_K = 2.0 * math.log(2.0) - 1.0


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    # Rata-rata TR berjendela tetap (bukan EWM Wilder): lookback terbatas -> fitur di window live 300
    # candle identik dengan fitur offline atas seluruh histori (tests/test_parity.py menangkap selisih
    # 1.4% pada atr_ratio_14_60 saat masih memakai Wilder dengan memori tak terbatas).
    return true_range.rolling(period, min_periods=period).mean()


def realized_vol(log_ret: pd.Series, window: int) -> pd.Series:
    return log_ret.rolling(window).std()


def semi_vol(log_ret: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Semi-variance: sigma_down^2 = 1/N * sum(min(0, r - mean)^2), sigma_up^2 analog dengan max.
    Mean dihitung per window yang sama (hanya data <= t). Vektorisasi via sliding_window_view."""
    r = log_ret.to_numpy(dtype=float)
    down = np.full(r.shape, np.nan)
    up = np.full(r.shape, np.nan)
    if len(r) >= window:
        win = np.lib.stride_tricks.sliding_window_view(r, window)  # (T-w+1, w)
        centered = win - win.mean(axis=1, keepdims=True)
        down[window - 1:] = np.sqrt((np.minimum(centered, 0) ** 2).mean(axis=1))
        up[window - 1:] = np.sqrt((np.maximum(centered, 0) ** 2).mean(axis=1))
    return pd.Series(down, index=log_ret.index), pd.Series(up, index=log_ret.index)


def causal_swt_denoise(x: pd.Series, levels: int = 3, mad_window: int = 240) -> pd.Series:
    """Stationary Wavelet Transform (Haar, algoritma a trous) dengan filter KAUSAL.

    approx_j(t) = (approx_{j-1}(t) + approx_{j-1}(t - 2^(j-1))) / 2 ;  detail_j = approx_{j-1} - approx_j
    Rekonstruksi: x = approx_J + sum(detail_j). Detail di-soft-threshold (universal threshold,
    sigma dari rolling MAD kausal). Karena setiap koefisien hanya memakai sampel <= t, tidak
    ada boundary distortion di titik paling kanan (berbeda dengan SWT/DWT biasa yang simetris).
    """
    s = x.fillna(0.0)
    approx = s
    denoised = None
    k = math.sqrt(2.0 * math.log(mad_window))
    for j in range(levels):
        smooth = (approx + approx.shift(2**j)) / 2.0
        detail = approx - smooth
        sigma = detail.abs().rolling(mad_window, min_periods=30).median() / 0.6745
        thr = sigma * k
        kept = np.sign(detail) * (detail.abs() - thr).clip(lower=0)
        denoised = kept if denoised is None else denoised + kept
        approx = smooth
    out = approx + denoised
    out[x.isna()] = np.nan
    return out


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """sigma^2 = 1/(4 ln2) * mean( ln(H/L)^2 )"""
    hl2 = np.log(high / low) ** 2
    return np.sqrt(_PARKINSON_K * hl2.rolling(window).mean())


def garman_klass_vol(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, window: int) -> pd.Series:
    """sigma^2 = mean( 0.5 ln(H/L)^2 - (2 ln2 - 1) ln(C/O)^2 )"""
    term = 0.5 * np.log(h / l) ** 2 - _GK_K * np.log(c / o) ** 2
    return np.sqrt(term.rolling(window).mean().clip(lower=0))


def compute_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Hitung semua fitur. Baris awal yang belum cukup history berisi NaN."""
    df = candles[["open", "high", "low", "close", "volume"]].astype(float).reset_index(drop=True)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    log_ret = np.log(c / c.shift(1))

    out = pd.DataFrame(index=df.index)
    out["log_ret_1"] = log_ret
    out["log_ret_5"] = log_ret.rolling(5).sum()
    out["log_ret_15"] = log_ret.rolling(15).sum()

    for w in (5, 15, 60, 240):
        out[f"rv_{w}"] = realized_vol(log_ret, w)
    out["rv_ratio_5_60"] = out["rv_5"] / (out["rv_60"] + _EPS)
    out["rv_ratio_15_60"] = out["rv_15"] / (out["rv_60"] + _EPS)
    out["rv_ratio_15_240"] = out["rv_15"] / (out["rv_240"] + _EPS)
    out["rv_ratio_60_240"] = out["rv_60"] / (out["rv_240"] + _EPS)

    for w in (15, 60):
        down, up = semi_vol(log_ret, w)
        out[f"rv_down_{w}"] = down
        out[f"rv_up_{w}"] = up
        out[f"downside_share_{w}"] = down**2 / (down**2 + up**2 + _EPS)

    clean = causal_swt_denoise(log_ret)
    out["swt_rv_15"] = clean.rolling(15).std()
    noise = (log_ret - clean) ** 2
    out["swt_noise_ratio_15"] = noise.rolling(15).sum() / ((log_ret**2).rolling(15).sum() + _EPS)
    out["swt_trend_15"] = clean.rolling(15).sum()

    # VPIN: |V_buy - V_sell| / V, dirata-rata per bar (bulk classification dari taker-buy volume)
    buy = candles["taker_buy_volume"].astype(float).values if "taker_buy_volume" in candles else v.values / 2
    buy = pd.Series(buy, index=df.index).fillna(v / 2)
    imbalance = buy - (v - buy)
    out["vpin_50"] = imbalance.abs().rolling(50).sum() / (v.rolling(50).sum() + _EPS)
    vpin15 = imbalance.abs().rolling(15).sum() / (v.rolling(15).sum() + _EPS)
    out["vpin_ratio_15_240"] = vpin15 / (imbalance.abs().rolling(240).sum() / (v.rolling(240).sum() + _EPS) + _EPS)
    out["ofi_5"] = imbalance.rolling(5).sum() / (v.rolling(5).sum() + _EPS)
    out["ofi_15"] = imbalance.rolling(15).sum() / (v.rolling(15).sum() + _EPS)

    out["parkinson_15"] = parkinson_vol(h, l, 15)
    out["parkinson_60"] = parkinson_vol(h, l, 60)
    out["parkinson_ratio_15_240"] = out["parkinson_15"] / (parkinson_vol(h, l, 240) + _EPS)
    out["garman_klass_15"] = garman_klass_vol(o, h, l, c, 15)
    out["garman_klass_60"] = garman_klass_vol(o, h, l, c, 60)
    out["gk_ratio_15_240"] = out["garman_klass_15"] / (garman_klass_vol(o, h, l, c, 240) + _EPS)

    out["atr_14_pct"] = _atr(df, 14) / c
    out["atr_ratio_14_60"] = _atr(df, 14) / (_atr(df, 60) + _EPS)

    sma20 = v.rolling(20).mean()
    out["volume_surge"] = v / (sma20 + _EPS)
    out["volume_surge_5"] = v.rolling(5).mean() / (sma20 + _EPS)
    out["volume_z_60"] = (v - v.rolling(60).mean()) / (v.rolling(60).std() + _EPS)
    # Volatilitas tanpa volume sering fakeout -> interaksi arah return x volume surge
    out["signed_volume_surge"] = np.sign(out["log_ret_5"]) * out["volume_surge_5"]

    if "open_time" in candles.columns:
        ts = pd.to_datetime(candles["open_time"].astype("int64").values, unit="ms", utc=True)
        hour = ts.hour + ts.minute / 60.0
        dow = ts.day_of_week
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    else:
        out[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out.index = candles.index
    return out[FEATURE_COLUMNS]


def latest_feature_vector(candles: pd.DataFrame) -> pd.Series:
    """Fitur untuk candle terakhir (yang dipakai untuk inference)."""
    if len(candles) < MIN_CANDLES:
        raise ValueError(f"Butuh minimal {MIN_CANDLES} candle 1m, baru ada {len(candles)}")
    feats = compute_features(candles.iloc[-(MIN_CANDLES + 10):])
    row = feats.iloc[-1]
    if row.isna().any():
        missing = row[row.isna()].index.tolist()
        raise ValueError(f"Fitur belum lengkap: {missing}")
    return row
