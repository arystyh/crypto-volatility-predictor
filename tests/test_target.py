import math

import numpy as np
import pandas as pd
import pytest

from app.ml.target import (CANDLE_MS, OnlineBaseline, TargetSpec, compute_targets,
                           forward_logvol)

SPEC = TargetSpec(horizon=15, baseline_window=600, min_baseline=200, z_threshold=2.0)


def synthetic_candles(n=3000, seed=0, spike_at=None):
    rng = np.random.default_rng(seed)
    vol = np.full(n, 0.0008)
    if spike_at is not None:
        vol[spike_at:spike_at + 15] = 0.006
    ret = rng.normal(0, vol)
    close = 30_000 * np.exp(np.cumsum(ret))
    rng_hl = np.abs(rng.normal(0, vol)) + 1e-6
    high = close * np.exp(rng_hl)
    low = close * np.exp(-np.abs(rng.normal(0, vol)) - 1e-6)
    return pd.DataFrame({"open_time": np.arange(n, dtype=np.int64) * CANDLE_MS,
                         "high": high, "low": low})


def test_online_matches_offline():
    df = synthetic_candles()
    off = compute_targets(df, SPEC)
    ob = OnlineBaseline(SPEC)
    ob.RESYNC_EVERY = 700  # exercise the resync path inside the test
    mus, sds = [], []
    for r in df.itertuples():
        ob.update(r.high, r.low, r.open_time)
        mus.append(ob.mu)
        sds.append(ob.sd)
    np.testing.assert_allclose(mus, off["baseline_mu"], rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(sds, off["baseline_sd"], rtol=1e-7, equal_nan=True)


def test_baseline_is_causal():
    df = synthetic_candles()
    t = 2000
    base = compute_targets(df, SPEC)
    tampered = df.copy()
    tampered.loc[t + 1:, "high"] *= 1.05  # change everything after t
    after = compute_targets(tampered, SPEC)
    assert base.loc[t, "baseline_mu"] == after.loc[t, "baseline_mu"]
    assert base.loc[t, "baseline_sd"] == after.loc[t, "baseline_sd"]


def test_label_only_sees_next_h_candles():
    df = synthetic_candles()
    t = 2000
    base = compute_targets(df, SPEC)
    tampered = df.copy()
    tampered.loc[t + SPEC.horizon + 1:, "high"] *= 1.05
    after = compute_targets(tampered, SPEC)
    assert base.loc[t, "z"] == after.loc[t, "z"]


def test_spike_is_labelled_and_forward_matches_helper():
    df = synthetic_candles(spike_at=2500)
    out = compute_targets(df, SPEC)
    t = 2499  # decision right before the spike window
    assert out.loc[t, "label"] == 1.0
    win = df.iloc[t + 1:t + 1 + SPEC.horizon]
    assert math.isclose(out.loc[t, "fwd_logvol"],
                        forward_logvol(win["high"].to_numpy(), win["low"].to_numpy()),
                        rel_tol=1e-12)
    assert out["label"].iloc[-SPEC.horizon:].isna().all()  # future unknown


def test_gap_is_rejected():
    df = synthetic_candles(n=100).drop(index=50).reset_index(drop=True)
    with pytest.raises(ValueError, match="gap-filler"):
        compute_targets(df, SPEC)
    ob = OnlineBaseline(SPEC)
    with pytest.raises(ValueError):
        for r in df.itertuples():
            ob.update(r.high, r.low, r.open_time)


def test_state_roundtrip():
    df = synthetic_candles(n=1200)
    a = OnlineBaseline(SPEC)
    for r in df.iloc[:900].itertuples():
        a.update(r.high, r.low, r.open_time)
    b = OnlineBaseline.from_state(a.to_state())
    for r in df.iloc[900:].itertuples():
        a.update(r.high, r.low, r.open_time)
        b.update(r.high, r.low, r.open_time)
    assert math.isclose(a.mu, b.mu, rel_tol=1e-9) and math.isclose(a.sd, b.sd, rel_tol=1e-7)


def test_min_embargo():
    assert TargetSpec(horizon=15).min_embargo(feature_lookback=60) == 75
