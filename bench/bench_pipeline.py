"""Benchmark latency pipeline (roadmap 3.5): p50/p95/p99 per tahap, diukur, bukan diklaim.

Memutar rekaman tick sintetis lewat jalur produksi (IngestionWorker.handle_message ->
Redis Stream -> InferenceEngine.process) dan mengukur waktu dinding (time.perf_counter):

    ingest_<stream>  parse JSON + update state per pesan WebSocket (depth, aggTrade, kline, ...)
    features         fitur 240 menit + EWM z
    model            champion ONNX + isotonic
    shadow / hazard  challenger LightGBM, hazard XGBoost
    ledger           tulis baris ledger (SHA-256 + SQLite, fsync WAL)
    processing       engine terima event -> selesai
    close_to_decision  pesan kline close diterima -> keputusan selesai (ingest + Redis + engine)

Redis = fakeredis in-process (tanpa jaringan): di produksi tambahkan RTT Redis (~0.2-1 ms lokal,
~5-20 ms Upstash). Telegram tidak diukur (HTTP eksternal).

    python -m bench.bench_pipeline --minutes 120
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

STAGE_ORDER = ["features", "model", "shadow", "hazard", "ledger", "processing"]


def pct(arr) -> dict:
    a = np.asarray(arr, float)
    return {"n": int(a.size), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}


async def run(minutes: int, seed: int = 0) -> dict:
    import fakeredis.aioredis

    from app.cache import store_candles
    from app.config import Settings
    from app.ml.target import TargetSpec
    from app.ml.target import OnlineBaseline
    from app.services import simticks
    from app.services.ledger import Ledger
    from app.services.recorder import STREAM_RANK
    from app.services.replay import FUTURES_STREAMS, RecordingNotifier, SimClock, build_engine
    from ingestion.worker import IngestionWorker
    from training.synthetic import generate_candles

    c = generate_candles("BTCUSDT", days=2, seed=seed)
    hist, live = c.iloc[:1000], c.iloc[1000:1000 + minutes]
    rec = simticks.generate(live, seed=seed)
    rec["rank"] = rec["stream"].map(STREAM_RANK).fillna(99).astype(int)
    rec = rec.sort_values(["event_time", "rank", "seq"], kind="mergesort").reset_index(drop=True)
    payloads = [json.loads(p) for p in rec["payload"]]

    # baseline target kecil agar siap dari 1000 candle histori; tidak memengaruhi latency
    SMALL_SPEC = TargetSpec(horizon=15, baseline_window=600, min_baseline=300)
    s = Settings(hmm_gate="off", conformal_enabled=False, alert_prob_threshold=0.5, alert_cooldown_minutes=15,
                 database_url="sqlite://", symbols="BTCUSDT")
    clock = SimClock(int(rec["recv_time"].iloc[0]))
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    tmp = tempfile.mkdtemp()
    ledger = Ledger(os.path.join(tmp, "bench.db"), max_log_lag_ms=s.ledger_max_log_lag_ms, clock=clock)
    notifier = RecordingNotifier()
    notifier.clock = clock
    engine = build_engine(s, redis, ledger, clock, SMALL_SPEC, notifier)
    worker = IngestionWorker(s, redis=redis, clock=clock, live=False)
    rows = [{**r, "symbol": "BTCUSDT"} for r in hist.to_dict("records")]
    await store_candles(redis, rows, s.cache_size)
    b = OnlineBaseline(SMALL_SPEC)
    engine.baselines._feed(b, hist)
    engine.baselines.baselines["BTCUSDT"] = b
    engine.baselines._loaded.add("BTCUSDT")
    worker.last_open["BTCUSDT"] = int(rows[-1]["open_time"])

    ingest = defaultdict(list)
    e2e = []
    last_id = "0-0"
    for r, data in zip(rec.itertuples(), payloads):
        clock.advance_to(r.recv_time)
        t0 = time.perf_counter()
        if r.stream == "depthSnapshot":
            await worker.handle_snapshot(r.symbol, data, clock())
        elif r.stream == "openInterest":
            await worker.handle_open_interest(r.symbol, int(data["time"]), float(data["openInterest"]))
        elif r.stream in FUTURES_STREAMS:
            await worker.handle_message({"stream": f"btcusdt@{r.stream}", "data": data}, source="futures",
                                        recv_ms=clock())
        else:
            name = {"depth": "depth@100ms", "kline": "kline_1m"}.get(r.stream, r.stream)
            await worker.handle_message({"stream": f"btcusdt@{name}", "data": data}, recv_ms=clock())
        t1 = time.perf_counter()
        is_close = r.stream == "kline" and data["k"]["x"]
        ingest["kline_close" if is_close else r.stream].append((t1 - t0) * 1000)
        if is_close:
            resp = await redis.xread({s.stream_key: last_id}, count=10)
            for _, entries in resp or []:
                for msg_id, fields in entries:
                    last_id = msg_id
                    await engine.process(fields["data"])
            e2e.append((time.perf_counter() - t0) * 1000)
    eng = {k: list(v) for k, v in engine.metrics.samples.items() if k in STAGE_ORDER}
    ledger.close()
    await redis.aclose()
    # 5 keputusan pertama = pemanasan (lazy load booster, cache numpy)
    out = {"ingest": {k: pct(v) for k, v in sorted(ingest.items())},
           "engine": {k: pct(eng[k][5:]) for k in STAGE_ORDER if len(eng.get(k, [])) > 5},
           "close_to_decision": pct(e2e[5:]),
           "messages": int(len(rec)), "decisions": len(e2e), "minutes": minutes,
           "machine": {"python": platform.python_version(), "cpu": _cpu(), "cores": os.cpu_count(),
                       "platform": platform.platform(terse=True)}}
    return out


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "?"


LABELS = {"features": "Fitur (240 mnt + EWM z)", "model": "Champion ONNX + isotonic", "shadow": "Challenger LightGBM",
          "hazard": "Hazard XGBoost (P≤5/10/15)", "ledger": "Tulis ledger (SHA-256 + SQLite)",
          "processing": "Engine total per candle"}


def markdown(res: dict) -> str:
    f = lambda v: f"{v:.3f}" if v < 1 else f"{v:.2f}"  # noqa: E731
    rows = [f"| {LABELS[k]} | {f(v['p50_ms'])} | {f(v['p95_ms'])} | {f(v['p99_ms'])} | {v['n']} |"
            for k, v in res["engine"].items()]
    c = res["close_to_decision"]
    rows.append(f"| **Kline close diterima → keputusan selesai** | **{f(c['p50_ms'])}** | **{f(c['p95_ms'])}** | "
                f"**{f(c['p99_ms'])}** | {c['n']} |")
    ing = [f"| `{k}` | {f(v['p50_ms'])} | {f(v['p95_ms'])} | {f(v['p99_ms'])} | {v['n']:,} |"
           for k, v in res["ingest"].items()]
    m = res["machine"]
    return f"""# Benchmark latency pipeline

Diukur dengan `python -m bench.bench_pipeline --minutes {res['minutes']}` · {res['messages']:,} pesan WebSocket sintetis,
{res['decisions']} keputusan · {m['cpu']} ({m['cores']} core) · Python {m['python']}.
Redis in-process (fakeredis); di produksi tambahkan RTT Redis. Telegram tidak termasuk.

## Per keputusan (tiap candle 1 menit close)

| Tahap | p50 (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|
{chr(10).join(rows)}

## Per pesan WebSocket (ingestion)

| Stream | p50 (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|
{chr(10).join(ing)}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--out", default="reports/benchmark.md")
    args = ap.parse_args(argv)
    import logging

    logging.basicConfig(level="ERROR")
    res = asyncio.run(run(args.minutes))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(res))
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
