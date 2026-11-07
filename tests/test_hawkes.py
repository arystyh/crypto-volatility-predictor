"""Roadmap 2.2: simulasi sintetis -> MLE memulihkan parameter; rekursi online O(1) = definisi."""
import math

import numpy as np

from app.hawkes import LIQ, TRADE, HawkesParams, OnlineHawkes, fit_hawkes, simulate_hawkes

TRUE = HawkesParams(mu=[0.20, 0.05], alpha=[[0.6, 0.9], [0.1, 0.5]], beta=2.0)


def test_mle_recovers_bivariate_parameters():
    t, k = simulate_hawkes(TRUE, T=30_000, seed=7)
    assert len(t) > 5_000
    fit = fit_hawkes(t, k, horizon=30_000)
    np.testing.assert_allclose(fit.mu, TRUE.mu, rtol=0.2)
    np.testing.assert_allclose(fit.beta, TRUE.beta, rtol=0.2)
    np.testing.assert_allclose(fit.alpha, TRUE.alpha, atol=0.12)
    assert abs(fit.branching - TRUE.branching) < 0.08
    # eksitasi silang likuidasi -> transaksi besar terdeteksi lebih besar dari arah sebaliknya
    assert fit.alpha[TRADE][LIQ] > 3 * fit.alpha[LIQ][TRADE]


def test_online_intensity_matches_definition():
    hk = OnlineHawkes(TRUE)
    events = [(1.0, TRADE), (1.5, LIQ), (2.0, TRADE), (4.0, LIQ)]
    for t, k in events:
        hk.add_event(t, k)
    t_now = 5.0
    a = np.asarray(TRUE.alpha)
    for i in (TRADE, LIQ):
        expect = TRUE.mu[i] + sum(a[i, k] * math.exp(-TRUE.beta * (t_now - t)) for t, k in events)
        assert abs(hk.intensity(t_now, i) - expect) < 1e-12
    cross = sum(a[TRADE, LIQ] * math.exp(-TRUE.beta * (t_now - t)) for t, k in events if k == LIQ)
    assert abs(hk.cross(t_now, TRADE, LIQ) - cross) < 1e-12
    f = hk.features(t_now)
    assert set(f) >= {"hawkes_trade_intensity", "hawkes_liq_intensity", "hawkes_cross_liq_to_trade", "hawkes_branching"}


def test_params_roundtrip(tmp_path):
    p = HawkesParams(mu=[0.1, 0.02], alpha=[[0.2, 0.3], [0.0, 0.1]], beta=1.5, model_version="v1")
    p.save(tmp_path / "h.json")
    q = HawkesParams.load(tmp_path / "h.json")
    assert q.alpha == p.alpha and q.model_version == "v1" and abs(q.branching - p.branching) < 1e-12
