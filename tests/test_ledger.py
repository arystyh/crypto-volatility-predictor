import sqlite3

import numpy as np
import pandas as pd
import pytest

from app.ml.target import CANDLE_MS, forward_logvol
from app.services.ledger import LateLogError, Ledger, Prediction
from app.services.scorer import average_precision, live_metrics, score_pending


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def pred(ts, prob=0.7, model="champion_xgboost", mu=-7.0, sd=0.5):
    return Prediction("BTCUSDT", ts, 15, model, prob, 0.5, mu, sd, 2.0)


@pytest.fixture
def ledger(tmp_path):
    clock = FakeClock()
    lg = Ledger(str(tmp_path / "l.db"), clock=clock)
    lg.clock = clock
    yield lg
    lg.close()


def test_append_only(ledger):
    seq = ledger.log_prediction(pred(0))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.conn.execute("UPDATE predictions SET prob = 0.99 WHERE seq = ?", (seq,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.conn.execute("DELETE FROM predictions")


def test_late_logging_rejected(ledger):
    ledger.clock.t = 10 * 60_000
    with pytest.raises(LateLogError):
        ledger.log_prediction(pred(0))


def test_duplicate_rejected(ledger):
    ledger.log_prediction(pred(0))
    with pytest.raises(sqlite3.IntegrityError):
        ledger.log_prediction(pred(0))
    ledger.log_prediction(pred(0, model="challenger_lgbm"))  # shadow model, same time: fine


def test_tampering_detected(ledger):
    for i in range(5):
        ledger.clock.t = i * CANDLE_MS
        ledger.log_prediction(pred(i * CANDLE_MS))
    assert ledger.verify() == (True, None)
    # an attacker with file access drops the trigger and edits history
    ledger.conn.execute("DROP TRIGGER predictions_no_update")
    ledger.conn.execute("UPDATE predictions SET prob = 0.01 WHERE seq = 3")
    assert ledger.verify() == (False, 3)


def candles_with_spike(n=60, spike_start=20):
    rng = np.random.default_rng(1)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 4e-4, n)))
    width = np.full(n, 3e-4)
    width[spike_start:spike_start + 15] = 6e-3
    return pd.DataFrame({"open_time": np.arange(n) * CANDLE_MS,
                         "high": close * (1 + width), "low": close * (1 - width)})


def test_scoring_waits_for_maturity_and_labels_correctly(ledger):
    c = candles_with_spike()
    calm_mu = forward_logvol(c["high"][:15].to_numpy(), c["low"][:15].to_numpy())
    ledger.clock.t = 20 * CANDLE_MS
    seq = ledger.log_prediction(pred(20 * CANDLE_MS, mu=calm_mu, sd=0.3))

    # only 10 forward candles closed: not mature
    r = score_pending(ledger, c.iloc[:30], "BTCUSDT")
    assert r.scored == 0 and r.not_mature == 1

    r = score_pending(ledger, c, "BTCUSDT")
    assert r.scored == 1
    row = ledger.conn.execute("SELECT * FROM outcomes WHERE seq = ?", (seq,)).fetchone()
    assert row["label"] == 1 and row["z"] > 2
    assert score_pending(ledger, c, "BTCUSDT").scored == 0  # idempotent


def test_gap_in_forward_window_is_not_scored(ledger):
    c = candles_with_spike().drop(index=25)
    ledger.clock.t = 20 * CANDLE_MS
    ledger.log_prediction(pred(20 * CANDLE_MS))
    r = score_pending(ledger, c, "BTCUSDT")
    assert r.scored == 0 and r.missing_candles == 1


def test_average_precision_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500), 2)  # rounding creates ties
    assert np.isclose(average_precision(y, p), sk.average_precision_score(y, p))


def test_live_metrics_end_to_end(ledger):
    c = candles_with_spike(n=80, spike_start=40)
    calm_mu = forward_logvol(c["high"][:15].to_numpy(), c["low"][:15].to_numpy())
    for t in range(20, 60):
        ledger.clock.t = t * CANDLE_MS
        prob = 0.9 if 30 <= t <= 40 else 0.1  # alerts start 10 min before the spike
        ledger.log_prediction(pred(t * CANDLE_MS, prob=prob, mu=calm_mu, sd=0.3))
    score_pending(ledger, c, "BTCUSDT")
    m = live_metrics(ledger, "champion_xgboost")
    assert m.n == 40 and m.positives > 0
    assert m.episodes == 1 and m.episodes_caught == 1
    # first labelled decision s: spike visible to a naive monitor at s + 15;
    # first alert at t = 30, so the lead is (s + 15) - 30 minutes
    s = ledger.scored("champion_xgboost")
    first_pos = min(r["decision_ts"] for r in s if r["label"] == 1) // CANDLE_MS
    assert m.median_lead_min == first_pos + 15 - 30 > 0
