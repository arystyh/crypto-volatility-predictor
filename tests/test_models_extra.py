"""Conformal ACI (4.1), kalibrasi isotonic (4.2), hazard monoton (2.6), lead-lag (2.3)."""
import json

import numpy as np

from app.calibration import IsotonicCalibrator, brier
from app.conformal import AdaptiveConformalThreshold, false_alarm_rate, simulate_aci
from app.hazard import curve_from_hazard
from app.services import simticks
from ingestion.leadlag import LeadLagState, lagged_xcorr
from training.synthetic import generate_candles


def _stream(n, shift_at=None, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    logit = np.where(y == 1, 1.0, -2.5) + rng.normal(0, 1.2, n)
    if shift_at is not None:
        logit[shift_at:] += shift  # model jadi terlalu percaya diri setelah pergantian rezim
    return 1 / (1 + np.exp(-logit)), y


def test_aci_keeps_false_alarm_share_near_target_after_regime_shift():
    p, y = _stream(60_000, shift_at=30_000, shift=1.5)
    tau0 = 0.6
    taus = simulate_aci(p, y, delay=15, tau0=tau0, alpha=0.2, gamma=0.01, mode="fdr")
    late = slice(40_000, None)
    static = false_alarm_rate(p[late], y[late], tau0)
    aci = false_alarm_rate(p[late], y[late], taus[late])
    assert static > 0.35                 # threshold statis rusak setelah pergeseran
    assert abs(aci - 0.2) < 0.05         # ACI kembali ke target
    assert taus[-1] > tau0


def test_aci_state_roundtrip_and_fpr_mode():
    a = AdaptiveConformalThreshold(tau=0.5, alpha=0.2, gamma=0.05, mode="fdr")
    assert a.update(0.4, 0) == 0.5            # bukan alert: tidak ada update (mode fdr)
    assert a.update(0.7, 0) > 0.5             # alert palsu -> tau naik
    b = AdaptiveConformalThreshold.loads(a.dumps())
    assert b.tau == a.tau and b.n_errors == 1
    f = AdaptiveConformalThreshold(tau=0.5, alpha=0.1, gamma=0.05, mode="fpr")
    assert f.update(0.9, 1) == 0.5            # mode fpr: outcome positif diabaikan


def test_isotonic_improves_brier(tmp_path):
    p, y = _stream(20_000, seed=3)
    raw = np.clip(p * 2.5, 0, 1)  # model mentah overconfident
    cal = IsotonicCalibrator.fit(raw[:10_000], y[:10_000])
    assert brier(cal(raw[10_000:]), y[10_000:]) < brier(raw[10_000:], y[10_000:])
    cal.save(tmp_path / "c.json")
    again = IsotonicCalibrator.load(tmp_path / "c.json")
    assert np.allclose(again(raw[:100]), cal(raw[:100]))
    assert np.all(np.diff(cal(np.linspace(0, 1, 50))) >= 0)


def test_hazard_cumulative_is_monotone():
    rng = np.random.default_rng(1)
    for _ in range(200):
        c = curve_from_hazard(rng.random(15) * rng.random())
        pw = [c.p_within(k) for k in range(1, 16)]
        assert np.all(np.diff(pw) >= 0) and abs(pw[-1] - c.p_onset_within_h) < 1e-12
        d = c.to_dict()["p_within"]
        assert d[5] <= d[10] <= d[15]


def test_leadlag_xcorr_recovers_perp_lead():
    c = generate_candles("BTCUSDT", days=1, seed=4).iloc[100:160]
    df = simticks.generate(c, seed=4, perp_lead_ms=2000, trades_per_min=300)
    spot = df[df["stream"] == "aggTrade"]["payload"].map(json.loads)
    perp = df[df["stream"] == "fut_aggTrade"]["payload"].map(json.loads)
    res = lagged_xcorr(np.array([d["T"] for d in spot]), np.array([float(d["p"]) for d in spot]),
                       np.array([d["T"] for d in perp]), np.array([float(d["p"]) for d in perp]))
    assert res["best_lag_s"] == 2 and res["significant"]
    st = LeadLagState()
    for d in perp:
        st.add_perp(d["T"], float(d["p"]))
    for d in spot:
        st.add_spot(d["T"], float(d["p"]))
    f = st.features(int(c["open_time"].iloc[-1]))
    assert f["perp_ret_5s"] is not None and f["spot_ret_15s"] is not None
