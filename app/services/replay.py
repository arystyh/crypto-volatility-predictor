"""Replay deterministik dari rekaman tick (roadmap 3.2).

Semua stream rekaman digabung dan diurutkan (event_time, rank stream tetap, seq terima) lalu
dimasukkan ke IngestionWorker lewat antarmuka YANG SAMA dengan WebSocket (`handle_message`).
Jam simulasi (= recv_time rekaman, dibuat monoton) menggerakkan semuanya: staleness circuit
breaker, cooldown alert, `logged_at` ledger. Engine memproses setiap candle close langsung dari
Redis Stream (fakeredis) pada jam yang sama. Tidak ada jam dinding, I/O jaringan, atau thread
yang memengaruhi hasil, sehingga dua replay atas data yang sama menghasilkan ledger yang
byte-identik (hash head sama) - diuji di tests/test_replay.py.

    python -m app.services.replay --ticks data/ticks --history data/history.csv --ledger /tmp/replay.db
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.cache import get_candles, store_candles
from app.config import Settings
from app.ml.target import OnlineBaseline, TargetSpec
from app.services.ledger import Ledger

log = logging.getLogger(__name__)
FUTURES_STREAMS = {"fut_aggTrade", "markPrice", "forceOrder"}


class SimClock:
    def __init__(self, start_ms: int):
        self.now = int(start_ms)

    def __call__(self) -> int:
        return self.now

    def advance_to(self, t_ms: int) -> None:
        self.now = max(self.now, int(t_ms))


class RecordingNotifier:
    enabled = True

    def __init__(self):
        self.sent: list[tuple[int, str]] = []
        self.clock = None

    async def send(self, text: str):
        self.sent.append((self.clock() if self.clock else 0, text))
        return True, None


@dataclass
class ReplayResult:
    ledger_head: str
    ledger_rows: int
    ledger_ok: bool
    decisions: int
    alerts: list = field(default_factory=list)
    breaker: dict = field(default_factory=dict)
    results: list = field(default_factory=list)
    worker: object = None
    cache: object = None


def build_engine(settings: Settings, redis, ledger, clock, spec: TargetSpec, notifier, models_dir: str = "models"):
    from app.calibration import IsotonicCalibrator
    from app.core.circuit_breaker import CircuitBreaker
    from app.engine import InferenceEngine
    from app.explain import Contributions
    from app.hazard import HazardModel
    from app.model import ChallengerModel, VolatilityModel
    from app.reason import ReasonBuilder
    from app.regime import GaussianHMM
    from app.services.baseline import BaselineManager

    md = Path(models_dir)
    model = VolatilityModel(md / "xgb_volatility.json", backend="onnx")
    breaker = CircuitBreaker(settings.required_streams, settings.stream_stale_sec, settings.breaker_recovery_sec,
                             clock=clock, notify=notifier.send)
    return InferenceEngine(
        settings, redis, None, model, notifier, ledger=ledger,
        baselines=BaselineManager(redis, settings.binance_rest_url, spec=spec, bootstrap=False),
        breaker=breaker,
        challenger=ChallengerModel(md / "challenger_lgbm.txt", model.features) if (md / "challenger_lgbm.txt").exists() else None,
        hazard=HazardModel(md / "hazard_xgb.json", model.features) if (md / "hazard_xgb.json").exists() else None,
        hmm=GaussianHMM.load(md / "hmm_regime.json") if settings.hmm_gate != "off" else None,
        calibrator=IsotonicCalibrator.load(md / "calibration.json"),
        reasons=ReasonBuilder(md / "feature_quantiles.json"), contributions=Contributions(md / "xgb_volatility.json",
                                                                                          model.features),
        clock=clock, spec=spec)


async def replay(records: pd.DataFrame, settings: Settings, ledger_path: str | Path, history: pd.DataFrame,
                 spec: TargetSpec = TargetSpec(), models_dir: str = "models", engine_hook=None,
                 worker_hook=None, before_event=None) -> ReplayResult:
    """`worker_hook(worker)` dan `before_event(record, worker)` (async) dipakai chaos test untuk
    menyuntikkan gangguan (reconnect, REST palsu) tanpa mengubah jalur kode produksi."""
    import fakeredis.aioredis

    from app.services.recorder import STREAM_RANK
    from ingestion.worker import IngestionWorker

    records = records.copy()
    records["rank"] = records["stream"].map(STREAM_RANK).fillna(99).astype(int)
    records = records.sort_values(["event_time", "rank", "seq"], kind="mergesort").reset_index(drop=True)
    clock = SimClock(int(records["recv_time"].iloc[0]) if len(records) else 0)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    Path(ledger_path).unlink(missing_ok=True)
    ledger = Ledger(str(ledger_path), max_log_lag_ms=settings.ledger_max_log_lag_ms, clock=clock)
    notifier = RecordingNotifier()
    notifier.clock = clock
    engine = build_engine(settings, redis, ledger, clock, spec, notifier, models_dir)
    if engine_hook:
        engine_hook(engine)
    worker = IngestionWorker(settings, redis=redis, clock=clock, live=False)
    if worker_hook:
        worker_hook(worker)

    # histori sebelum rekaman: cache candle + baseline target (pengganti bootstrap REST)
    symbols = sorted(records["symbol"].unique()) if len(records) else settings.symbol_list
    hist = history.sort_values("open_time")
    for sym in symbols:
        h = hist[hist.get("symbol", sym) == sym] if "symbol" in hist else hist
        rows = [{**r, "symbol": sym} for r in h.to_dict("records")]
        await store_candles(redis, rows, settings.cache_size)
        b = OnlineBaseline(spec)
        engine.baselines._feed(b, h)
        engine.baselines.baselines[sym] = b
        engine.baselines._loaded.add(sym)
        if rows:
            worker.last_open[sym] = int(rows[-1]["open_time"])

    stream = settings.stream_key
    last_id = "0-0"
    results = []
    for r in records.itertuples():
        clock.advance_to(r.recv_time)
        if before_event is not None:
            await before_event(r, worker)
        data = json.loads(r.payload)
        if r.stream == "depthSnapshot":
            await worker.handle_snapshot(r.symbol, data, clock())
        elif r.stream == "openInterest":
            await worker.handle_open_interest(r.symbol, int(data["time"]), float(data["openInterest"]))
        elif r.stream in FUTURES_STREAMS or r.stream == "fut_aggTrade":
            await worker.handle_message({"stream": f"{r.symbol.lower()}@{r.stream}", "data": data}, source="futures",
                                        recv_ms=clock())
        else:
            name = {"depth": "depth@100ms", "kline": "kline_1m"}.get(r.stream, r.stream)
            await worker.handle_message({"stream": f"{r.symbol.lower()}@{name}", "data": data}, recv_ms=clock())
        if r.stream == "kline" and data.get("k", {}).get("x"):
            resp = await redis.xread({stream: last_id}, count=100)
            for _, entries in resp or []:
                for msg_id, fields in entries:
                    last_id = msg_id
                    res = await engine.process(fields["data"])
                    if res is not None:
                        results.append(res)
    ok, _ = ledger.verify()
    out = ReplayResult(ledger.head(), ledger.count(), ok, len(results), notifier.sent, engine.breaker.snapshot(), results)
    out.worker, out.cache = worker, await get_candles(redis, symbols[0]) if symbols else None
    ledger.close()
    await redis.aclose()
    return out


def main(argv=None) -> None:
    from app.services import recorder

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", required=True, help="Folder rekaman TickRecorder")
    ap.add_argument("--history", required=True, help="CSV candle 1m sebelum rekaman (>= 7 hari untuk baseline)")
    ap.add_argument("--ledger", default="data/replay_ledger.db")
    ap.add_argument("--models", default="models")
    args = ap.parse_args(argv)
    logging.basicConfig(level="WARNING")
    records = recorder.load(args.ticks)
    res = asyncio.run(replay(records, Settings(), args.ledger, pd.read_csv(args.history), models_dir=args.models))
    print(json.dumps({"ledger_head": res.ledger_head, "rows": res.ledger_rows, "ledger_ok": res.ledger_ok,
                      "decisions": res.decisions, "alerts": len(res.alerts), "breaker": res.breaker["decisions"]},
                     indent=2))


if __name__ == "__main__":
    main()
