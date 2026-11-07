"""Feed Binance USDⓈ-M Futures (roadmap 2.1).

Stream per simbol (satu koneksi combined stream ke fstream.binance.com):
    <symbol>@forceOrder   likuidasi. PERHATIAN: Binance hanya mengirim SNAPSHOT likuidasi terbesar
                          per 1 detik, jadi ini SAMPEL, bukan total. Fitur diberi nama *_sampled.
    <symbol>@markPrice@1s mark price, index price, funding rate -> basis & funding
    <symbol>@aggTrade     transaksi perpetual -> trade flow & lead-lag perp->spot
REST (tiap menit):
    /fapi/v1/openInterest            open interest terkini
    /futures/data/openInterestHist   isi celah OI setelah koneksi putus (granularitas 5 menit)
    /fapi/v1/aggTrades?fromId=       isi celah aggTrade (cek kontinuitas aggTradeId, sama dengan kline)

Agregasi per candle 1m: jumlah likuidasi, notional long vs short, ukuran terbesar, trade flow
perpetual, OI dan perubahannya, funding rate, basis.

Catatan: fstream.binance.com diblokir dari IP Amerika Serikat (HTTP 451). Bila koneksi gagal,
fitur futures kosong dan stream ini tidak termasuk stream wajib circuit breaker.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

import httpx
import websockets

log = logging.getLogger("ingestion.futures")


def _empty_minute() -> dict:
    return {"liq_n": 0, "liq_long_notional": 0.0, "liq_short_notional": 0.0, "liq_max_notional": 0.0,
            "buy": 0.0, "sell": 0.0, "trades": 0, "notional": 0.0}


class FuturesAggregator:
    def __init__(self):
        self.minutes: dict[tuple[str, int], dict] = defaultdict(_empty_minute)
        self.mark: dict[str, dict] = {}
        self.oi: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.incomplete: set[tuple[str, int]] = set()  # menit dengan celah yang tidak bisa diisi

    def add_liquidation(self, data: dict) -> tuple[str, int, float]:
        o = data.get("o", data)
        sym, t = o["s"], int(o.get("T") or data.get("E"))
        qty = float(o.get("z") or o.get("q") or 0)
        price = float(o.get("ap") or o.get("p") or 0)
        notional = qty * price
        b = self.minutes[(sym, t // 60_000 * 60_000)]
        b["liq_n"] += 1
        # order SELL paksa = posisi LONG yang dilikuidasi
        b["liq_long_notional" if o.get("S") == "SELL" else "liq_short_notional"] += notional
        b["liq_max_notional"] = max(b["liq_max_notional"], notional)
        return sym, t, notional

    def add_trade(self, msg: dict) -> None:
        sym, t = msg["s"], int(msg["T"])
        qty, price = float(msg["q"]), float(msg["p"])
        b = self.minutes[(sym, t // 60_000 * 60_000)]
        b["sell" if msg["m"] else "buy"] += qty
        b["trades"] += 1
        b["notional"] += qty * price

    def add_mark(self, msg: dict) -> None:
        self.mark[msg["s"]] = {"mark": float(msg["p"]), "index": float(msg.get("i") or 0) or None,
                               "funding_rate": float(msg.get("r") or 0), "event_ms": int(msg["E"])}

    def add_open_interest(self, symbol: str, t_ms: int, oi: float) -> None:
        hist = self.oi[symbol]
        if hist and t_ms <= hist[-1][0]:
            return
        hist.append((t_ms, oi))
        del hist[:-200]

    def _oi_at(self, symbol: str, t_ms: int) -> float | None:
        best = None
        for t, v in self.oi.get(symbol, []):
            if t <= t_ms:
                best = v
        return best

    def pop(self, symbol: str, minute: int) -> dict:
        b = self.minutes.pop((symbol, minute), None) or _empty_minute()
        for key in [k for k in self.minutes if k[0] == symbol and k[1] < minute - 5 * 60_000]:
            self.minutes.pop(key, None)
        end = minute + 60_000
        liq = b["liq_long_notional"] + b["liq_short_notional"]
        vol = b["buy"] + b["sell"]
        oi_now, oi_prev = self._oi_at(symbol, end), self._oi_at(symbol, end - 15 * 60_000)
        m = self.mark.get(symbol, {})
        out = {
            "liq_count_1m_sampled": b["liq_n"],
            "liq_long_notional_1m_sampled": round(b["liq_long_notional"], 2),
            "liq_short_notional_1m_sampled": round(b["liq_short_notional"], 2),
            "liq_max_notional_1m_sampled": round(b["liq_max_notional"], 2),
            "liq_long_share_1m": round(b["liq_long_notional"] / liq, 4) if liq else None,
            "perp_trades_1m": b["trades"],
            "perp_trade_imbalance": round((b["buy"] - b["sell"]) / vol, 6) if vol else None,
            "open_interest": oi_now,
            "oi_chg_15m": round(oi_now / oi_prev - 1, 6) if oi_now and oi_prev else None,
            "funding_rate": m.get("funding_rate"),
            "futures_complete": (symbol, minute) not in self.incomplete,
        }
        self.incomplete.discard((symbol, minute))
        return out


class FuturesFeed:
    """Koneksi WebSocket futures + polling open interest. Pesan diteruskan ke IngestionWorker."""

    def __init__(self, worker, ws_url: str, rest_url: str, symbols: list[str]):
        self.worker = worker
        self.ws_url = ws_url.rstrip("/")
        self.rest_url = rest_url.rstrip("/")
        self.symbols = symbols
        self.last_oi_ms: dict[str, int] = {}

    def url(self) -> str:
        streams = []
        for s in self.symbols:
            s = s.lower()
            streams += [f"{s}@forceOrder", f"{s}@markPrice@1s", f"{s}@aggTrade"]
        return f"{self.ws_url}/stream?streams={'/'.join(streams)}"

    async def poll_open_interest(self, client: httpx.AsyncClient) -> None:
        for sym in self.symbols:
            try:
                last = self.last_oi_ms.get(sym)
                now = self.worker.clock()
                if last is not None and now - last > 6 * 60_000:  # celah: isi dari histori 5 menit
                    r = await client.get(f"{self.rest_url}/futures/data/openInterestHist",
                                         params={"symbol": sym, "period": "5m", "startTime": last, "limit": 500})
                    r.raise_for_status()
                    for row in r.json():
                        await self.worker.handle_open_interest(sym, int(row["timestamp"]),
                                                               float(row["sumOpenInterest"]))
                r = await client.get(f"{self.rest_url}/fapi/v1/openInterest", params={"symbol": sym})
                r.raise_for_status()
                d = r.json()
                await self.worker.handle_open_interest(sym, int(d["time"]), float(d["openInterest"]))
                self.last_oi_ms[sym] = int(d["time"])
            except Exception as exc:  # noqa: BLE001
                log.warning("Open interest %s gagal: %s", sym, exc)

    async def run_oi(self, stop: asyncio.Event) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            while not stop.is_set():
                await self.poll_open_interest(client)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=60)
                except asyncio.TimeoutError:
                    pass

    async def run(self, stop: asyncio.Event) -> None:
        backoff = 5
        oi_task = asyncio.create_task(self.run_oi(stop))
        try:
            while not stop.is_set():
                try:
                    log.info("Connect futures stream: %s", self.url())
                    async with websockets.connect(self.url(), ping_interval=None, open_timeout=20,
                                                  max_size=2**22) as ws:
                        backoff = 5
                        while not stop.is_set():
                            raw = await asyncio.wait_for(ws.recv(), timeout=60)
                            await self.worker.handle_message(raw, source="futures")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - mis. HTTP 451 (region diblokir)
                    log.warning("Futures stream terputus (%s). Coba lagi dalam %ss", exc, backoff)
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=backoff)
                    except asyncio.TimeoutError:
                        pass
                    backoff = min(backoff * 2, 600)
        finally:
            oi_task.cancel()


def parse_combined(raw: str | dict) -> tuple[str, dict]:
    msg = json.loads(raw) if isinstance(raw, str) else raw
    return msg.get("stream", ""), msg.get("data", msg)
