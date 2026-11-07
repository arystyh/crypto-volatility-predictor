"""Circuit breaker, arah volatilitas, HMM regime, bot Telegram, format alert."""
import asyncio

import numpy as np
import pandas as pd

from app.core.circuit_breaker import DEGRADED, OK, STALE, CircuitBreaker
from app.engine import volatility_direction
from app.regime import GaussianHMM, gate_mask
from app.telegram import format_alert
from app.telegram_bot import TelegramBot


def test_breaker_three_states_per_stream():
    clock = {"t": 100_000}
    br = CircuitBreaker(("kline", "depth"), stale_sec=5, recovery_sec=30, clock=lambda: clock["t"])
    fresh = {"kline": {"event_ms": 99_500}, "depth": {"event_ms": 99_900}}
    assert br.evaluate(fresh).state == OK
    stale = {"kline": {"event_ms": 99_500}, "depth": {"event_ms": 94_000}}  # depth 6 detik
    d = br.evaluate(stale)
    assert d.state == STALE and "depth" in d.reason and not d.allow_alert
    assert br.evaluate({"kline": {"event_ms": 99_500}}).state == STALE  # stream wajib belum pernah ada
    recovering = {"kline": {"event_ms": 99_900, "outage_end_ms": 90_000}, "depth": {"event_ms": 99_900}}
    assert br.evaluate(recovering).state == STALE  # baru pulih 10 detik
    clock["t"] = 125_000
    assert br.evaluate({"kline": {"event_ms": 124_900, "outage_end_ms": 90_000},
                        "depth": {"event_ms": 124_950}}).state == OK
    br.model_failed("timeout")
    assert br.evaluate({"kline": {"event_ms": 124_900}, "depth": {"event_ms": 124_950}}).state == DEGRADED
    sent = []

    async def note(t):
        sent.append(t)

    br.notify = note
    asyncio.run(br.record(br.evaluate(stale)))
    assert br.state == STALE and "STALE" in sent[-1]


def test_direction_and_alert_text():
    assert volatility_direction({"downside_share_15": 0.8}) == "downside"
    assert volatility_direction({"downside_share_15": 0.2}) == "upside"
    text = format_alert("BTCUSDT", 0.9, 60000, 0.8, {"downside_share_15": 0.8}, direction="downside")
    assert "Risk of Dump" in text


def test_hmm_recovers_regimes():
    rng = np.random.default_rng(0)
    states = np.repeat([0, 1, 0, 1, 0], 400)
    x = np.where(states == 1, rng.normal(-6, 0.3, states.size), rng.normal(-8, 0.3, states.size))
    hmm = GaussianHMM.fit(x)
    assert hmm.means[1] > hmm.means[0]
    p = hmm.filter(x)
    assert ((p >= 0.5) == states.astype(bool)).mean() > 0.97
    gates = gate_mask(p, "transition", window=30)
    assert gates[400:430].any() and not gates[500:800].any()  # hanya sesaat setelah transisi
    assert gate_mask(p, "off").all()


class _Conformal:
    alpha = 0.2


class _Engine:
    threshold = 0.8
    conformal = _Conformal()

    async def set_threshold(self, v):
        self.threshold = v

    async def set_alpha(self, v):
        self.conformal.alpha = v


def test_bot_allowed_chat_ids_and_validation(caplog):
    """Roadmap 1.6: command pengubah state hanya dari ALLOWED_CHAT_IDS; validasi 0 < p < 1."""
    eng = _Engine()
    bot = TelegramBot("t", {"42", "7"}, eng, lambda: {"status": "ok", "threshold": eng.threshold,
                                                       "circuit_breaker": {"state": "OK"}})
    replies = []

    async def capture(chat_id, text):
        replies.append((chat_id, text))

    bot.send = capture

    def run(text, chat="42"):
        return asyncio.run(bot.handle_update({"message": {"text": text, "chat": {"id": chat}}}))

    assert "0.850" in run("/setthreshold 0.85") and eng.threshold == 0.85
    for bad in ("2", "1", "0", "-0.1", "1.0"):
        assert "❌" in run(f"/setthreshold {bad}") and eng.threshold == 0.85
    assert "❌" in run("/setthreshold abc")
    with caplog.at_level("WARNING"):
        r = run("/setthreshold 0.5", chat="999")
    assert "⛔" in r and eng.threshold == 0.85 and bot.rejected == 1
    assert any("DITOLAK" in m and "999" in m for m in caplog.messages)
    assert "⛔" in run("/setalpha 0.1", chat="999") and eng.conformal.alpha == 0.2
    assert "0.100" in run("/setalpha 0.1", chat="7") and eng.conformal.alpha == 0.1
    assert "Status" in run("/status", chat="999")  # read-only boleh dari mana saja


def test_alert_path_has_no_z_threshold_wording():
    """Roadmap 1.4: jalur alert hanya bicara threshold PROBABILITAS."""
    text = format_alert("BTCUSDT", 0.9, 60000, 0.42, {"downside_share_15": 0.5},
                        hazard={"p_within": {5: 0.2, 10: 0.5, 15: 0.7}}, contributors=["a", "b", "c"],
                        status="OK", ledger={"seq": 12, "row_hash": "ab" * 32, "url": "https://x/ledger/row/12"})
    low = text.lower()
    assert "z >" not in low and "z-threshold" not in low and "z_threshold" not in low
    assert "≤5m" in text and "seq 12" in text and "Status sistem" in text


def test_numpy_parkinson_heuristic():
    from app.fallback import heuristic_from_candles

    n = 260
    df = pd.DataFrame({"high": np.full(n, 101.0), "low": np.full(n, 100.0), "volume": np.full(n, 10.0)})
    assert heuristic_from_candles(df)[0] == 0.0
    df.loc[n - 15:, "high"] = 106.0  # range 6x lebih lebar
    df.loc[n - 5:, "volume"] = 40.0
    prob, info = heuristic_from_candles(df)
    assert prob == 1.0 and info["parkinson_ratio_15_240"] > 2
