"""Harness bersama: engine lengkap di atas fakeredis + ledger sementara + jam simulasi."""
from __future__ import annotations

import fakeredis.aioredis

from app.config import Settings
from app.ml.target import TargetSpec
from app.services.ledger import Ledger
from app.services.replay import RecordingNotifier, SimClock, build_engine

# baseline kecil agar siap dari ~300 candle (default produksi: 30 hari, minimal 7 hari)
SMALL_SPEC = TargetSpec(horizon=15, baseline_window=600, min_baseline=300)


def settings(**kw) -> Settings:
    base = dict(hmm_gate="off", conformal_enabled=False, alert_prob_threshold=0.0001, alert_cooldown_minutes=0,
                stream_block_ms=50, stream_claim_idle_ms=0, database_url="sqlite://", symbols="BTCUSDT")
    base.update(kw)
    return Settings(**base)


def make_engine(tmp_path, s: Settings | None = None, clock=None, redis=None, name="ledger.db"):
    s = s or settings()
    clock = clock or SimClock(0)
    redis = redis or fakeredis.aioredis.FakeRedis(decode_responses=True)
    notifier = RecordingNotifier()
    notifier.clock = clock
    ledger = Ledger(str(tmp_path / name), max_log_lag_ms=s.ledger_max_log_lag_ms, clock=clock)
    eng = build_engine(s, redis, ledger, clock, SMALL_SPEC, notifier)
    return eng, redis, ledger, notifier, clock


def alerts_only(sent):
    return [t for _, t in sent if "ALERT" in t]
