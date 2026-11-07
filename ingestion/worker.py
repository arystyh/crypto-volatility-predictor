"""Data Ingestion Worker (proses latar terpisah dari API).

Spot (data-stream.binance.vision, satu combined stream per simbol):
    @kline_1m      OHLCV + taker-buy volume (VPIN). Candle close -> cache Redis + XADD crypto_stream
    @aggTrade      trade flow per menit, transaksi besar -> Hawkes; kontinuitas aggTradeId dicek,
                   lubang diisi dari REST /api/v3/aggTrades?fromId=
    @depth@100ms   order book LOKAL (snapshot REST /api/v3/depth?limit=1000 + diff, lihat
                   ingestion/orderbook.py). OBI 5 level dihitung saat candle close, None bila book invalid.
Futures (fstream.binance.com, opsional; lihat ingestion/futures_feed.py):
    @forceOrder, @markPrice@1s, @aggTrade, open interest REST tiap menit.

Setiap candle yang di-publish membawa:
    micro         agregat mikrostruktur menit itu (spot, order book, futures, Hawkes, lead-lag)
    stream_state  waktu event terakhir & akhir putus terakhir per stream -> circuit breaker di engine
                  menentukan OK / STALE secara deterministik saat keputusan dibuat.

Semua waktu memakai `clock()` yang bisa disuntik, sehingga worker yang sama dipakai oleh replay
deterministik (app/services/replay.py) dengan jam simulasi. Pesan mentah opsional direkam ke
Parquet (RECORD_TICKS=true, app/services/recorder.py).

Jalankan:  python -m ingestion.worker
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from collections import defaultdict
from pathlib import Path

import httpx
import websockets

from app.cache import get_candles, get_redis, store_candles
from app.config import Settings, get_settings
from app.core.circuit_breaker import STREAMS_KEY
from app.hawkes import LIQ, TRADE, HawkesParams, OnlineHawkes
from ingestion.binance import parse_rest_kline, parse_ws_kline
from ingestion.futures_feed import FuturesAggregator, FuturesFeed, parse_combined
from ingestion.leadlag import LeadLagState
from ingestion.orderbook import LocalOrderBook

log = logging.getLogger("ingestion")


def now_ms() -> int:
    return int(time.time() * 1000)


def _empty_spot() -> dict:
    return {"buy": 0.0, "sell": 0.0, "trades": 0, "large_notional": 0.0, "notional": 0.0, "large_n": 0}


def stream_url(base: str, symbols: list[str], interval: str, aggtrade: bool, depth: bool) -> str:
    streams = []
    for s in symbols:
        s = s.lower()
        streams.append(f"{s}@kline_{interval}")
        if aggtrade:
            streams.append(f"{s}@aggTrade")
        if depth:
            streams.append(f"{s}@depth@100ms")
    return f"{base.rstrip('/')}/stream?streams={'/'.join(streams)}"


def stream_name(stream: str, data: dict, source: str) -> str | None:
    e = data.get("e")
    if e == "kline":
        return "kline"
    if e == "aggTrade":
        return "fut_aggTrade" if source == "futures" else "aggTrade"
    if e == "depthUpdate" or "@depth" in stream:
        return "depth"
    if e == "markPriceUpdate":
        return "markPrice"
    if e == "forceOrder":
        return "forceOrder"
    return None


class IngestionWorker:
    def __init__(self, settings: Settings, redis=None, clock=now_ms, recorder=None, live: bool = True):
        self.settings = settings
        self.redis = redis if redis is not None else get_redis(settings.redis_url)
        self.clock = clock
        self.live = live
        self.symbols = settings.symbol_list
        self.recorder = recorder
        if recorder is None and live and settings.record_ticks:
            from app.services.recorder import TickRecorder

            self.recorder = TickRecorder(settings.record_dir)
        self.books = {s: LocalOrderBook(s) for s in self.symbols}
        self.spot: dict[tuple[str, int], dict] = defaultdict(_empty_spot)
        self.spot_incomplete: set[tuple[str, int]] = set()
        self.fut = FuturesAggregator()
        hp = Path(settings.hawkes_path)
        params = HawkesParams.load(hp) if hp.exists() else HawkesParams()
        self.hawkes = {s: OnlineHawkes(params) for s in self.symbols}
        self.leadlag = {s: LeadLagState() for s in self.symbols}
        self.streams: dict[str, dict] = {}
        self.last_agg_id: dict[tuple[str, str], int] = {}
        self._snapshot_tasks: dict[str, asyncio.Task] = {}
        self._bg: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self.candles_published = 0
        self.candles_gap_filled = 0
        self.agg_gaps = 0
        self.last_open: dict[str, int] = {}
        self.last_message_ts: float | None = None
        self._last_hb_write = 0
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ util
    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http

    def _record(self, stream: str, symbol: str, data: dict, event_time: int, recv: int) -> None:
        if self.recorder is not None:
            self.recorder.record(stream, symbol, data, event_time, recv)

    def _touch(self, name: str, event_ms: int, recv: int) -> None:
        st = self.streams.setdefault(name, {"event_ms": None, "recv_ms": None, "outage_end_ms": None})
        prev = st["event_ms"]
        if prev is not None and event_ms - prev > self.settings.stream_stale_sec * 1000:
            st["outage_end_ms"] = recv  # stream sempat diam lebih lama dari ambang: tandai baru pulih
        st["event_ms"] = max(event_ms, prev or 0)
        st["recv_ms"] = recv

    def stream_state(self) -> dict:
        return {k: dict(v) for k, v in self.streams.items()}

    async def heartbeat(self, recv: int) -> None:
        self.last_message_ts = recv / 1000
        if not self.live:
            return
        if recv - self._last_hb_write >= self.settings.heartbeat_interval_sec * 1000:
            self._last_hb_write = recv
            await self.redis.set(STREAMS_KEY, json.dumps({"streams": self.stream_state(), "_reported_ms": recv}),
                                 ex=600)

    # ------------------------------------------------------------------ REST recovery kline
    async def _fetch(self, symbol: str, start: int | None, end: int | None, limit: int) -> list[dict]:
        params = {"symbol": symbol, "interval": self.settings.interval, "limit": min(limit, 1000)}
        if start is not None:
            params["startTime"] = start
        if end is not None:
            params["endTime"] = end
        resp = await self.http().get(f"{self.settings.binance_rest_url.rstrip('/')}/api/v3/klines", params=params)
        resp.raise_for_status()
        now = self.clock()
        return [parse_rest_kline(symbol, r) for r in resp.json() if int(r[6]) < now]  # hanya yang sudah close

    def _record_rest_kline(self, c: dict) -> None:
        k = {"t": c["open_time"], "T": c["close_time"], "s": c["symbol"], "i": "1m", "o": str(c["open"]),
             "h": str(c["high"]), "l": str(c["low"]), "c": str(c["close"]), "v": str(c["volume"]),
             "n": c["trades"], "V": str(c["taker_buy_volume"]), "x": True}
        self._record("kline", c["symbol"], {"e": "kline", "E": c["close_time"], "s": c["symbol"], "k": k,
                                             "rest": True}, c["close_time"], self.clock())

    async def fill_gap(self, symbol: str, start: int, end: int) -> int:
        """Isi candle [start, end) dari REST. Tidak di-XADD (data lama, hanya untuk fitur)."""
        filled, cursor = 0, start
        while cursor < end:
            rows = await self._fetch(symbol, cursor, end - 1, 1000)
            if not rows:
                break
            for c in rows:
                self._record_rest_kline(c)
            await store_candles(self.redis, rows, self.settings.cache_size)
            filled += len(rows)
            cursor = rows[-1]["open_time"] + 60_000
            if len(rows) < 1000:
                break
        if filled:
            self.candles_gap_filled += filled
            log.warning("Gap-fill %s: %d candle dipulihkan dari REST", symbol, filled)
        return filled

    async def recover(self) -> None:
        """Sebelum (re)connect: isi celah sejak candle terakhir yang diketahui, atau backfill awal."""
        s = self.settings
        for symbol in self.symbols:
            try:
                if symbol not in self.last_open:
                    cached = await get_candles(self.redis, symbol, limit=1)
                    if not cached.empty:
                        self.last_open[symbol] = int(cached.iloc[-1]["open_time"])
                now = self.clock()
                last = self.last_open.get(symbol)
                if last is None or now - last > s.cache_size * 60_000:
                    rows = await self._fetch(symbol, None, None, s.cache_size + 1)
                    for c in rows:
                        self._record_rest_kline(c)
                    await store_candles(self.redis, rows, s.cache_size)
                    log.info("Backfill %s: %d candle", symbol, len(rows))
                else:
                    await self.fill_gap(symbol, last + 60_000, now)
                cached = await get_candles(self.redis, symbol, limit=1)
                if not cached.empty:
                    self.last_open[symbol] = int(cached.iloc[-1]["open_time"])
            except Exception as exc:  # noqa: BLE001
                log.warning("Recovery %s gagal: %s", symbol, exc)

    # ------------------------------------------------------------------ order book
    async def fetch_snapshot(self, symbol: str) -> None:
        try:
            r = await self.http().get(f"{self.settings.binance_rest_url.rstrip('/')}/api/v3/depth",
                                      params={"symbol": symbol, "limit": 1000})
            r.raise_for_status()
            await self.handle_snapshot(symbol, r.json(), self.clock())
        except Exception as exc:  # noqa: BLE001
            log.warning("Snapshot depth %s gagal: %s", symbol, exc)
        finally:
            self._snapshot_tasks.pop(symbol, None)

    def _need_snapshot(self, symbol: str) -> None:
        if self.live and symbol not in self._snapshot_tasks:
            # beri jeda singkat agar event pertama sempat di-buffer sebelum snapshot diambil
            async def later():
                await asyncio.sleep(0.5)
                await self.fetch_snapshot(symbol)

            self._snapshot_tasks[symbol] = asyncio.create_task(later())

    async def handle_snapshot(self, symbol: str, snap: dict, recv: int) -> None:
        self._record("depthSnapshot", symbol, snap, recv, recv)
        ok = self.books[symbol].apply_snapshot(snap)
        log.info("Order book %s snapshot lastUpdateId=%s valid=%s", symbol, snap.get("lastUpdateId"), ok)
        if not ok:
            self._need_snapshot(symbol)

    # ------------------------------------------------------------------ aggTrade
    def _add_spot_trade(self, d: dict) -> None:
        sym, t = d["s"], int(d["T"])
        qty, price = float(d["q"]), float(d["p"])
        b = self.spot[(sym, t // 60_000 * 60_000)]
        b["sell" if d["m"] else "buy"] += qty  # m=True: pembeli maker -> agresor penjual
        b["trades"] += 1
        notional = qty * price
        b["notional"] += notional
        if notional >= self.settings.large_trade_usd:
            b["large_notional"] += notional
            b["large_n"] += 1
            if sym in self.hawkes:
                self.hawkes[sym].add_event(t / 1000, TRADE)
        if sym in self.leadlag:
            self.leadlag[sym].add_spot(t, price)

    async def fill_agg_gap(self, symbol: str, from_id: int, to_id: int, futures: bool = False) -> int:
        base = self.settings.binance_futures_rest_url if futures else self.settings.binance_rest_url
        path = "/fapi/v1/aggTrades" if futures else "/api/v3/aggTrades"
        name = "fut_aggTrade" if futures else "aggTrade"
        got, cursor = 0, from_id
        try:
            while cursor <= to_id:
                r = await self.http().get(f"{base.rstrip('/')}{path}",
                                          params={"symbol": symbol, "fromId": cursor, "limit": 1000})
                r.raise_for_status()
                rows = [x for x in r.json() if x["a"] <= to_id]
                if not rows:
                    break
                for x in rows:
                    d = {"e": "aggTrade", "E": x["T"], "s": symbol, **x, "rest": True}
                    self._record(name, symbol, d, int(x["T"]), self.clock())
                    if futures:
                        self.fut.add_trade(d)
                    else:
                        self._add_spot_trade(d)
                got += len(rows)
                cursor = rows[-1]["a"] + 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Isi celah %s %s gagal: %s", name, symbol, exc)
        if got < to_id - from_id + 1:
            minute = self.clock() // 60_000 * 60_000
            (self.fut.incomplete if futures else self.spot_incomplete).add((symbol, minute))
        return got

    def _check_agg(self, name: str, d: dict) -> None:
        key = (name, d["s"])
        a = int(d["a"])
        last = self.last_agg_id.get(key)
        if last is not None and a > last + 1:
            self.agg_gaps += 1
            log.warning("%s %s: aggTradeId lompat %d -> %d (%d hilang)", name, d["s"], last, a, a - last - 1)
            if self.live:
                self._spawn(self.fill_agg_gap(d["s"], last + 1, a - 1, futures=name == "fut_aggTrade"))
            else:
                self.spot_incomplete.add((d["s"], int(d["T"]) // 60_000 * 60_000))
        if last is None or a > last:
            self.last_agg_id[key] = a

    # ------------------------------------------------------------------ dispatcher
    async def handle_message(self, raw: str | dict, source: str = "spot", recv_ms: int | None = None) -> None:
        recv = self.clock() if recv_ms is None else recv_ms
        stream, data = parse_combined(raw)
        name = stream_name(stream, data, source)
        if name is None:
            return
        sym = (data.get("s") or data.get("o", {}).get("s") or stream.split("@")[0]).upper()
        event_ms = int(data.get("E") or data.get("T") or recv)
        if name != "forceOrder":  # forceOrder hanya muncul saat ada likuidasi: bukan ukuran kesegaran
            self._touch(name, event_ms, recv)
        self._record(name, sym, data, event_ms, recv)
        await self.heartbeat(recv)

        if name == "aggTrade":
            if not data.get("rest"):
                self._check_agg(name, data)
            self._add_spot_trade(data)
        elif name == "depth":
            book = self.books.get(sym)
            if book is not None and (not book.on_event(data) or book.last_update_id is None):
                self._need_snapshot(sym)
        elif name == "fut_aggTrade":
            if not data.get("rest"):
                self._check_agg(name, data)
            self.fut.add_trade(data)
            if sym in self.leadlag:
                self.leadlag[sym].add_perp(int(data["T"]), float(data["p"]))
        elif name == "markPrice":
            self.fut.add_mark(data)
            if sym in self.leadlag:
                self.leadlag[sym].add_mark(event_ms, float(data["p"]))
        elif name == "forceOrder":
            s, t, _ = self.fut.add_liquidation(data)
            if s in self.hawkes:
                self.hawkes[s].add_event(t / 1000, LIQ)
        elif name == "kline":
            await self._on_kline({"stream": stream, "data": data}, recv, rest=bool(data.get("rest")))

    async def handle_open_interest(self, symbol: str, t_ms: int, oi: float) -> None:
        self._record("openInterest", symbol, {"s": symbol, "time": t_ms, "openInterest": oi}, t_ms, self.clock())
        self.fut.add_open_interest(symbol, t_ms, oi)

    def micro_for(self, symbol: str, minute: int) -> dict:
        b = self.spot.pop((symbol, minute), None) or _empty_spot()
        for key in [k for k in self.spot if k[0] == symbol and k[1] < minute - 5 * 60_000]:
            self.spot.pop(key, None)
        vol = b["buy"] + b["sell"]
        book = self.books.get(symbol)
        obi = book.obi(5) if book is not None else None
        close_ms = minute + 60_000
        out = {
            "agg_trades": b["trades"],
            "agg_buy_qty": round(b["buy"], 8),
            "agg_sell_qty": round(b["sell"], 8),
            "trade_imbalance": round((b["buy"] - b["sell"]) / vol, 6) if vol else 0.0,
            "large_trade_share": round(b["large_notional"] / b["notional"], 6) if b["notional"] else 0.0,
            "large_trades": b["large_n"],
            "agg_complete": (symbol, minute) not in self.spot_incomplete,
            "obi": None if obi is None else round(obi, 6),
            "book_valid": bool(book and book.valid),
        }
        self.spot_incomplete.discard((symbol, minute))
        if self.settings.enable_futures:
            out.update(self.fut.pop(symbol, minute))
            out.update(self.leadlag[symbol].features(close_ms))
        if symbol in self.hawkes:
            out.update(self.hawkes[symbol].features(close_ms / 1000))
        return out

    async def _on_kline(self, msg: dict, recv: int, rest: bool = False) -> None:
        candle = parse_ws_kline(msg)
        if candle is None or not candle["is_closed"]:
            return  # hanya candle close yang diteruskan (hemat kuota Redis)
        symbol, open_time = candle["symbol"], candle["open_time"]
        if rest:  # candle hasil gap-fill REST (rekaman/replay): simpan saja, tidak memicu keputusan
            await store_candles(self.redis, [candle], self.settings.cache_size)
            self.last_open[symbol] = max(open_time, self.last_open.get(symbol, 0))
            return
        last = self.last_open.get(symbol)
        if last is not None and open_time - last > 60_000 and self.live:  # ada candle yang terlewat
            try:
                await self.fill_gap(symbol, last + 60_000, open_time)
            except Exception as exc:  # noqa: BLE001
                log.warning("Gap-fill %s gagal: %s", symbol, exc)
        candle["ingested_at"] = recv
        candle["micro"] = self.micro_for(symbol, open_time)
        candle["stream_state"] = self.stream_state()
        await store_candles(self.redis, [candle], self.settings.cache_size,
                            stream=self.settings.stream_key, maxlen=self.settings.stream_maxlen)
        self.last_open[symbol] = max(open_time, last or 0)
        self.candles_published += 1
        m = candle["micro"]
        log.info("%s close=%.4f vol=%.2f trades=%s obi=%s -> %s", symbol, candle["close"], candle["volume"],
                 m.get("agg_trades"), m.get("obi"), self.settings.stream_key)

    # ------------------------------------------------------------------ koneksi
    async def run(self) -> None:
        s = self.settings
        url = stream_url(s.binance_ws_url, self.symbols, s.interval, s.enable_aggtrade, s.enable_depth)
        backoff = 1
        while not self._stop.is_set():
            await self.recover()
            for book in self.books.values():  # book lama tidak bisa dipercaya setelah putus
                book.invalidate("reconnect")
                book.last_update_id = None
                book.buffer.clear()
            try:
                log.info("Connect WebSocket: %s", url)
                # Binance memutus koneksi tiap 24 jam; ping server dibalas otomatis oleh library websockets.
                async with websockets.connect(url, ping_interval=None, max_queue=8192, open_timeout=20,
                                              max_size=2**22) as ws:
                    backoff = 1
                    while not self._stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        except asyncio.TimeoutError:
                            log.warning("Tidak ada data 60 detik, reconnect")
                            break
                        await self.handle_message(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - koneksi putus (1006), DNS, redis, dll.
                log.warning("Koneksi terputus (%s). Reconnect dalam %ss", exc, backoff)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60)

    async def run_all(self) -> None:
        tasks = [asyncio.create_task(self.run())]
        if self.settings.enable_futures:
            feed = FuturesFeed(self, self.settings.binance_futures_ws_url, self.settings.binance_futures_rest_url,
                               self.symbols)
            tasks.append(asyncio.create_task(feed.run(self._stop)))
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            if self.recorder is not None:
                self.recorder.flush()
            if self._http is not None:
                await self._http.aclose()

    def stop(self) -> None:
        self._stop.set()


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = IngestionWorker(settings)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.stop)
        except NotImplementedError:  # Windows
            pass
    log.info("Ingestion worker start: %s (aggTrade=%s, depth=%s, futures=%s, record=%s)", settings.symbol_list,
             settings.enable_aggtrade, settings.enable_depth, settings.enable_futures, settings.record_ticks)
    task = asyncio.create_task(worker.run_all())
    await worker._stop.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await worker.redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
