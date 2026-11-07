"""Roadmap 3.2: dua replay atas rekaman yang sama -> ledger byte-identik (hash head sama)."""
import asyncio

from app.services import simticks
from app.services.recorder import TickRecorder, load
from app.services.replay import replay
from tests.helpers import SMALL_SPEC, alerts_only, settings
from training.synthetic import generate_candles


def test_two_replays_give_identical_ledger(tmp_path):
    c = generate_candles("BTCUSDT", days=1, seed=3)
    hist, live = c.iloc[:700], c.iloc[700:730]
    rec = TickRecorder(tmp_path / "ticks")
    simticks.to_recorder(simticks.generate(live, seed=1), rec)
    records = load(tmp_path / "ticks")  # lewat Parquet, seperti produksi
    s = settings(conformal_enabled=True, alert_prob_threshold=0.01, alert_cooldown_minutes=5)
    a = asyncio.run(replay(records, s, tmp_path / "a.db", hist, spec=SMALL_SPEC))
    b = asyncio.run(replay(records, s, tmp_path / "b.db", hist, spec=SMALL_SPEC))
    assert a.ledger_rows > 30 and a.ledger_ok and b.ledger_ok
    assert a.ledger_head == b.ledger_head and a.ledger_rows == b.ledger_rows
    assert (tmp_path / "a.db").exists()
    assert [t for _, t in a.alerts] == [t for _, t in b.alerts]
    assert a.decisions == 30
    # rekaman lengkap: tidak ada keputusan STALE, alert keluar
    assert a.breaker["decisions"]["STALE"] == 0
    assert alerts_only(a.alerts)
