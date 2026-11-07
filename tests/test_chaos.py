"""Roadmap 3.4 - chaos test. Alert TIDAK PERNAH keluar dari data basi, celah terisi, pesan yang
belum di-ACK dilanjutkan (XAUTOCLAIM). Threshold dipaksa sangat rendah: setiap keputusan OK akan
alert, jadi satu-satunya penahan adalah circuit breaker."""
import asyncio
import json

import fakeredis.aioredis
import pandas as pd
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache import store_candles
from app.core.circuit_breaker import STALE
from app.engine import GROUP
from app.services import simticks
from app.services.replay import replay
from tests.helpers import SMALL_SPEC, make_engine, settings
from training.synthetic import generate_candles

C = generate_candles("BTCUSDT", days=1, seed=5)
HIST, LIVE = C.iloc[:700], C.iloc[700:725]
T0 = int(LIVE["open_time"].iloc[0])


def _rest_patch(worker):
    """REST palsu: kline dari data sumber; snapshot & aggTrade diisi lewat rekaman."""
    async def fake_fetch(symbol, start, end, limit):
        rows = C[(C["open_time"] >= (start or 0)) & (C["open_time"] <= (end or 1e18))]
        rows = rows[rows["open_time"] + 60_000 <= worker.clock()]
        return [{**r, "symbol": symbol} for r in rows.to_dict("records")][-limit:]

    async def no_snapshot(symbol):
        return None

    async def no_fill(symbol, from_id, to_id, futures=False):
        (worker.fut.incomplete if futures else worker.spot_incomplete).add(
            (symbol, worker.clock() // 60_000 * 60_000))
        return 0

    worker.live = True
    worker.fill_agg_gap = no_fill
    worker._fetch = fake_fetch
    worker.fetch_snapshot = no_snapshot
    worker.heartbeat = _noop_hb


async def _noop_hb(recv):
    return None


def _check_no_stale_alert(res):
    stale = [r for r in res.results if r.status == STALE]
    assert stale, "skenario harus memicu STALE"
    assert all(not r.alerted for r in stale)
    assert all(r.held_reason and r.held_reason.startswith("STALE") for r in stale if r.is_spike)
    ok = [r for r in res.results if r.status == "OK"]
    assert ok and all(r.alerted for r in ok if r.is_spike)
    return stale


def test_ws_down_30s_holds_alerts_and_fills_gap(tmp_path):
    close_5, close_15 = T0 + 5 * 60_000, T0 + 15 * 60_000
    swallow = (close_5 - 10_000, close_5 + 20_000)   # 30 dtk mati, menelan kline close -> candle hilang
    late = (close_15 - 31_000, close_15 - 1_000)     # 30 dtk mati, pulih 1 dtk sebelum candle close
    recs = simticks.generate(LIVE, seed=1, drop_windows=[swallow, late], snapshot_every_ms=60_000)
    res = asyncio.run(replay(recs, settings(), tmp_path / "l.db", HIST, spec=SMALL_SPEC, worker_hook=_rest_patch))
    stale = _check_no_stale_alert(res)
    # yang STALE hanya keputusan dalam masa pemulihan 30 dtk setelah feed kembali
    assert [r.candle_time + 60_000 for r in stale] == [close_15]
    # celah candle diisi dari REST: cache kontinu, ledger tetap sah
    t = res.cache["open_time"].astype("int64").diff().dropna()
    assert (t == 60_000).all()
    assert res.worker.candles_gap_filled >= 1 and res.ledger_ok


def test_daily_reconnect_resyncs_book_without_losing_data(tmp_path):
    """Binance memutus koneksi tiap 24 jam: reconnect cepat (2 dtk), book wajib resync,
    celah aggTradeId diisi REST, tidak ada candle hilang."""
    full = simticks.generate(LIVE, seed=2, snapshot_every_ms=30_000)
    cut = (T0 + 5 * 60_000 + 20_000, T0 + 5 * 60_000 + 22_000)
    recs = full[~((full["recv_time"] >= cut[0]) & (full["recv_time"] < cut[1]))]
    lost = full[(full["recv_time"] >= cut[0]) & (full["recv_time"] < cut[1]) & (full["stream"] == "aggTrade")]
    state = {"reconnected": False, "filled": []}

    def hook(worker):
        _rest_patch(worker)

        async def fill(symbol, from_id, to_id, futures=False):
            if futures:  # celah aggTrade futures: tandai menit tidak lengkap
                worker.fut.incomplete.add((symbol, worker.clock() // 60_000 * 60_000))
                return 0
            got = [json.loads(p) for p in lost["payload"]]
            got = [d for d in got if from_id <= d["a"] <= to_id]
            for d in got:
                worker._add_spot_trade(d)
            state["filled"].append((from_id, to_id, len(got)))
            return len(got)

        worker.fill_agg_gap = fill

    async def before(r, worker):
        if not state["reconnected"] and r.recv_time >= cut[1]:
            state["reconnected"] = True
            await worker.recover()          # sama dengan IngestionWorker.run() saat reconnect
            for book in worker.books.values():
                book.invalidate("reconnect")
                book.last_update_id = None
                book.buffer.clear()

    res = asyncio.run(replay(recs, settings(), tmp_path / "l.db", HIST, spec=SMALL_SPEC, worker_hook=hook,
                             before_event=before))
    assert state["reconnected"] and res.worker.agg_gaps == 2  # spot + futures
    assert state["filled"] and state["filled"][0][2] == len(lost)
    assert res.decisions == len(LIVE) and res.ledger_ok
    assert all(r.micro["agg_complete"] for r in res.results)
    # book invalid -> OBI kosong (bukan angka karangan) sampai snapshot berikutnya
    for r in res.results:
        assert (r.micro["obi"] is None) == (not r.micro["book_valid"])


def test_missing_depth_updates_never_produce_obi_from_invalid_book(tmp_path):
    recs = simticks.generate(LIVE, seed=3, skip_depth_every=45, snapshot_every_ms=20_000)
    res = asyncio.run(replay(recs, settings(), tmp_path / "l.db", HIST, spec=SMALL_SPEC))
    book = res.worker.books["BTCUSDT"]
    assert book.gaps > 0 and book.resyncs > 0
    invalid = [r for r in res.results if not r.micro["book_valid"]]
    valid = [r for r in res.results if r.micro["book_valid"]]
    assert invalid and valid
    assert all(r.micro["obi"] is None for r in invalid)
    assert all(r.micro["obi"] is not None for r in valid)


def _payload(row, t_ingest, state=None):
    from app.cache import publish_payload

    return publish_payload({**row, "symbol": "BTCUSDT", "ingested_at": t_ingest,
                            "stream_state": state or {k: {"event_ms": t_ingest} for k in ("kline", "aggTrade", "depth")}})


def _seed(redis, s, rows_hist):
    async def go():
        await store_candles(redis, rows_hist, s.cache_size)
    return go()


@pytest.mark.parametrize("failures", [2])
def test_redis_restart_engine_reconnects_and_resumes(tmp_path, failures, monkeypatch):
    s = settings()
    eng, redis, ledger, notifier, clock = make_engine(tmp_path, s)
    rows = [{**r, "symbol": "BTCUSDT"} for r in C.iloc[:720].to_dict("records")]

    async def scenario():
        await _seed(redis, s, rows[:700])
        await redis.xgroup_create(s.stream_key, GROUP, id="0", mkstream=True)
        for r in rows[700:705]:
            clock.advance_to(r["close_time"] + 50)
            await redis.zadd(f"candles:BTCUSDT", {json.dumps(r): r["open_time"]})
            await redis.xadd(s.stream_key, {"data": _payload(r, clock())})
        real = redis.xreadgroup
        calls = {"n": 0}

        async def flaky(*a, **kw):  # Redis restart: koneksi putus beberapa kali
            calls["n"] += 1
            if calls["n"] <= failures:
                raise RedisConnectionError("Connection reset by peer")
            return await real(*a, **kw)

        monkeypatch.setattr(redis, "xreadgroup", flaky)
        sleeps = []
        real_sleep = asyncio.sleep

        async def fast_sleep(t):
            sleeps.append(t)
            await real_sleep(0)

        monkeypatch.setattr("app.engine.asyncio.sleep", fast_sleep)
        task = asyncio.create_task(eng.run())
        for _ in range(400):
            await real_sleep(0.01)
            if eng.processed >= 5:
                break
        eng.running = False
        task.cancel()
        pending = (await redis.xpending(s.stream_key, GROUP))["pending"]
        return calls["n"], pending, [x for x in sleeps if x >= 1]

    n, pending, backoffs = asyncio.run(scenario())
    assert eng.processed == 5 and pending == 0
    assert backoffs[:2] == [1, 2]  # exponential backoff saat Redis putus
    assert ledger.verify()[0]


def test_api_restart_unacked_message_is_autoclaimed_once(tmp_path):
    """API mati setelah mencatat ke ledger tapi sebelum XACK: consumer baru mengambil alih via
    XAUTOCLAIM, memproses ulang, ledger tidak dobel, alert tidak dobel."""
    s = settings(alert_cooldown_minutes=15)
    eng, redis, ledger, notifier, clock = make_engine(tmp_path, s)
    rows = [{**r, "symbol": "BTCUSDT"} for r in C.iloc[:702].to_dict("records")]

    async def scenario():
        await _seed(redis, s, rows[:701])
        await redis.xgroup_create(s.stream_key, GROUP, id="0", mkstream=True)
        r = rows[701]
        await redis.zadd("candles:BTCUSDT", {json.dumps(r): r["open_time"]})
        clock.advance_to(r["close_time"] + 50)
        await redis.xadd(s.stream_key, {"data": _payload(r, clock())})
        # consumer lama membaca & memproses, lalu crash sebelum XACK
        resp = await redis.xreadgroup(GROUP, "engine-old", {s.stream_key: ">"}, count=10)
        await eng.process(resp[0][1][0][1]["data"])
        rows_before = ledger.count()
        alerts_before = len(notifier.sent)
        assert (await redis.xpending(s.stream_key, GROUP))["pending"] == 1
        eng.processed = 0
        task = asyncio.create_task(eng.run())  # consumer baru "engine-1"
        for _ in range(200):
            await asyncio.sleep(0.01)
            if eng.processed:
                break
        eng.running = False
        task.cancel()
        return rows_before, alerts_before, (await redis.xpending(s.stream_key, GROUP))["pending"]

    rows_before, alerts_before, pending = asyncio.run(scenario())
    assert eng.processed == 1 and pending == 0
    assert ledger.count() == rows_before > 0          # baris duplikat ditolak (idempoten)
    assert len(notifier.sent) == alerts_before        # cooldown: tidak ada alert kedua
    assert ledger.verify()[0]
