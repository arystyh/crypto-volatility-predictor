"""Breaker DEGRADED: model gagal -> fallback heuristic Parkinson, tetap tercatat di ledger dengan status DEGRADED."""
import asyncio
import sqlite3

from app.engine import CHAMPION, FALLBACK
from app.services import simticks
from app.services.replay import replay
from tests.helpers import SMALL_SPEC, settings
from training.synthetic import generate_candles


def test_model_failure_degrades_to_fallback(tmp_path):
    c = generate_candles("BTCUSDT", days=1, seed=3)
    hist, live = c.iloc[:700], c.iloc[700:712]

    def boom(_):
        raise RuntimeError("model rusak")

    res = asyncio.run(replay(simticks.generate(live, seed=1), settings(alert_prob_threshold=0.01), tmp_path / "l.db",
                             hist, spec=SMALL_SPEC, engine_hook=lambda e: setattr(e.model, "predict_one", boom)))
    assert res.ledger_ok and res.breaker["decisions"]["DEGRADED"] == res.decisions > 0
    assert all(r.status == "DEGRADED" and r.mode == "fallback" for r in res.results)
    rows = sqlite3.connect(tmp_path / "l.db").execute("SELECT model_version, status FROM predictions").fetchall()
    assert ("fallback_parkinson", "DEGRADED") in rows and not any(m == CHAMPION for m, _ in rows)
    assert FALLBACK == "fallback_parkinson"
