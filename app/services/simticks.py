"""Generator tick sintetis yang KONSISTEN dengan candle 1m (untuk test replay, chaos, parity, benchmark).

Menghasilkan rekaman berformat sama dengan TickRecorder (stream, symbol, event_time, recv_time,
payload) untuk: kline (update tiap 2 detik + close), aggTrade (id berurutan), depth diff (U/u
berurutan) + depthSnapshot, markPrice@1s, futures aggTrade (perp MEMIMPIN spot `perp_lead_ms`),
forceOrder (jarang).

Opsi gangguan untuk chaos test:
    drop_windows      [(start_ms, end_ms)] semua event di jendela ini hilang (WebSocket mati)
    skip_depth_every  buang satu event depth tiap N event (update hilang -> book harus resync)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

TICK = 0.01


class _Book:
    def __init__(self, mid: float, rng: np.random.Generator, levels: int = 20, spacing: float = 0.5):
        self.rng = rng
        self.spacing = spacing
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.recenter(mid, levels)

    def _px(self, x: float) -> float:
        return round(round(x / TICK) * TICK, 2)

    def recenter(self, mid: float, levels: int = 20) -> tuple[list, list]:
        """Geser book ke mid baru. Return diff (bids, asks) yang diperlukan."""
        bb, ba = self._px(mid - self.spacing / 2), self._px(mid + self.spacing / 2)
        db, da = [], []
        for p in [p for p in self.bids if p > bb]:
            del self.bids[p]
            db.append((p, 0.0))
        for p in [p for p in self.asks if p < ba]:
            del self.asks[p]
            da.append((p, 0.0))
        for i in range(levels):
            pb, pa = self._px(bb - i * self.spacing), self._px(ba + i * self.spacing)
            if pb not in self.bids:
                self.bids[pb] = round(float(self.rng.exponential(0.5)) + 0.001, 5)
                db.append((pb, self.bids[pb]))
            if pa not in self.asks:
                self.asks[pa] = round(float(self.rng.exponential(0.5)) + 0.001, 5)
                da.append((pa, self.asks[pa]))
        return db, da

    def churn(self) -> tuple[list, list]:
        db, da = [], []
        for side, out in ((self.bids, db), (self.asks, da)):
            if not side:
                continue
            keys = sorted(side)
            for _ in range(int(self.rng.integers(1, 4))):
                p = keys[int(self.rng.integers(0, len(keys)))]
                q = 0.0 if self.rng.random() < 0.1 and len(side) > 10 else round(float(self.rng.exponential(0.5)) + 0.001, 5)
                if q == 0.0:
                    side.pop(p, None)
                else:
                    side[p] = q
                out.append((p, q))
        return db, da


def _path(c, t0: int):
    """Harga intra-menit: open -> high/low -> low/high -> close (piecewise linear)."""
    up_first = c.close >= c.open
    pts = [c.open, c.low if up_first else c.high, c.high if up_first else c.low, c.close]
    xs = [0, 15_000, 45_000, 60_000]

    def f(t: int) -> float:
        return float(np.interp(t - t0, xs, pts))
    return f


def generate(candles: pd.DataFrame, symbol: str = "BTCUSDT", seed: int = 0, depth_ms: int = 1000,
             trades_per_min: int = 200, perp_lead_ms: int = 2000, futures: bool = True,
             drop_windows: list[tuple[int, int]] | None = None, skip_depth_every: int | None = None,
             snapshot_every_ms: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    candles = candles.reset_index(drop=True)
    sym_l = symbol.lower()
    rows: list[tuple] = []
    seq = 0
    t_start = int(candles["open_time"].iloc[0])
    book = _Book(float(candles["open"].iloc[0]), rng)
    update_id = 1000
    agg_id, fut_id = 5_000_000, 9_000_000
    paths = {int(r.open_time): _path(r, int(r.open_time)) for r in candles.itertuples()}
    t_end = int(candles["open_time"].iloc[-1]) + 60_000

    def price_at(t: int) -> float:
        m = (t // 60_000) * 60_000
        m = min(max(m, t_start), t_end - 60_000)
        return paths[m](t)

    def emit(stream: str, event_time: int, payload: dict, lat: int | None = None):
        nonlocal seq
        seq += 1
        recv = event_time + (lat if lat is not None else int(rng.integers(20, 80)))
        rows.append((stream, symbol, int(event_time), int(recv), seq, payload))

    def snapshot(t: int):
        emit("depthSnapshot", t, {"lastUpdateId": update_id,
                                   "bids": [[f"{p:.2f}", f"{q}"] for p, q in sorted(book.bids.items(), reverse=True)],
                                   "asks": [[f"{p:.2f}", f"{q}"] for p, q in sorted(book.asks.items())]}, lat=0)

    snapshot(t_start)
    n_depth = 0
    for c in candles.itertuples():
        t0 = int(c.open_time)
        f = paths[t0]
        # aggTrade spot
        times = np.sort(rng.integers(1, 59_990, size=trades_per_min)) + t0
        buy_share = c.taker_buy_volume / c.volume if c.volume > 0 else 0.5
        qty_total = c.volume
        for t in times:
            agg_id += 1
            q = max(qty_total / trades_per_min * float(rng.exponential(1.0)), 1e-5)
            if rng.random() < 0.05:
                q = max(q, 150_000 / f(int(t)))  # transaksi besar
            emit("aggTrade", int(t), {"e": "aggTrade", "E": int(t) + 1, "s": symbol, "a": agg_id, "p": f"{f(int(t)):.2f}",
                                      "q": f"{q:.6f}", "f": agg_id * 3, "l": agg_id * 3 + 1, "T": int(t),
                                      "m": bool(rng.random() > buy_share)})
        # depth diff
        for t in range(t0 + depth_ms // 2, t0 + 60_000, depth_ms):
            db, da = book.recenter(price_at(t))
            cb, ca = book.churn()
            n_depth += 1
            U = update_id + 1
            update_id = U + int(rng.integers(0, 3))
            if skip_depth_every and n_depth % skip_depth_every == 0:
                continue  # update hilang di jaringan
            emit("depth", t, {"e": "depthUpdate", "E": t, "s": symbol, "U": U, "u": update_id,
                              "b": [[f"{p:.2f}", f"{q}"] for p, q in db + cb],
                              "a": [[f"{p:.2f}", f"{q}"] for p, q in da + ca]})
            if snapshot_every_ms and (t - t_start) % snapshot_every_ms < depth_ms:
                snapshot(t + 1)
        # kline update tiap 2 detik + close
        for k in range(2_000, 60_000, 2_000):
            t = t0 + k
            emit("kline", t, {"e": "kline", "E": t, "s": symbol, "k": {
                "t": t0, "T": t0 + 59_999, "s": symbol, "i": "1m", "o": f"{c.open}", "h": f"{max(c.open, f(t)):.2f}",
                "l": f"{min(c.open, f(t)):.2f}", "c": f"{f(t):.2f}", "v": f"{c.volume * k / 60_000:.6f}", "n": 1,
                "V": f"{c.taker_buy_volume * k / 60_000:.6f}", "x": False}})
        t = t0 + 60_000 + 3
        emit("kline", t, {"e": "kline", "E": t, "s": symbol, "k": {
            "t": t0, "T": t0 + 59_999, "s": symbol, "i": "1m", "o": f"{c.open}", "h": f"{c.high}", "l": f"{c.low}",
            "c": f"{c.close}", "v": f"{c.volume}", "n": int(getattr(c, "trades", 100) or 100),
            "V": f"{c.taker_buy_volume}", "x": True}})
        if futures:
            for t in range(t0 + 500, t0 + 60_000, 1000):
                emit("markPrice", t, {"e": "markPriceUpdate", "E": t, "s": symbol,
                                      "p": f"{price_at(t + perp_lead_ms) * 1.0002:.2f}", "i": f"{price_at(t):.2f}",
                                      "r": "0.00010000", "T": t0 + 8 * 3_600_000})
            for t in np.sort(rng.integers(1, 59_990, size=trades_per_min)) + t0:
                fut_id += 1
                emit("fut_aggTrade", int(t), {"e": "aggTrade", "E": int(t) + 1, "s": symbol, "a": fut_id,
                                              "p": f"{price_at(int(t) + perp_lead_ms):.2f}",
                                              "q": f"{float(rng.exponential(0.5)):.4f}", "f": fut_id, "l": fut_id,
                                              "T": int(t), "m": bool(rng.random() > 0.5)})
            if rng.random() < 0.15:
                t = t0 + int(rng.integers(1, 59_000))
                emit("forceOrder", t, {"e": "forceOrder", "E": t, "o": {
                    "s": symbol, "S": "SELL" if rng.random() < 0.5 else "BUY", "o": "LIMIT", "f": "IOC",
                    "q": f"{float(rng.exponential(2)):.3f}", "p": f"{f(t):.2f}", "ap": f"{f(t):.2f}", "X": "FILLED",
                    "l": "0.1", "z": f"{float(rng.exponential(2)):.3f}", "T": t}})
            emit("openInterest", t0 + 30_000, {"s": symbol, "time": t0 + 30_000,
                                                 "openInterest": float(80_000 + rng.normal(0, 100))}, lat=0)

    df = pd.DataFrame(rows, columns=["stream", "symbol", "event_time", "recv_time", "seq", "payload"])
    for a, b in drop_windows or []:
        df = df[~((df["recv_time"] >= a) & (df["recv_time"] < b))]
    df["payload"] = df["payload"].map(lambda d: json.dumps(d, separators=(",", ":")))
    return df.reset_index(drop=True)


def to_recorder(df: pd.DataFrame, recorder) -> None:
    for r in df.itertuples():
        recorder.record(r.stream, r.symbol, json.loads(r.payload), r.event_time, r.recv_time)
    recorder.flush()
