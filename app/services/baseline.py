"""Manajer baseline target online per simbol (app.ml.target.OnlineBaseline), state di Redis.

Ledger butuh (mu, sd) baseline 30 hari yang berlaku saat keputusan dibuat. State ini:
- diperbarui setiap candle close (harus kontinu; gap diisi dari cache dulu),
- disimpan ke Redis SETIAP candle (= tiap 1 menit) secara inkremental agar hemat kuota Upstash:
      baseline:<SYM>:trail   Redis LIST nilai trail_logvol (RPUSH nilai baru + LTRIM ke W)
      baseline:<SYM>:meta    JSON {spec, pk (15 nilai terakhir), last_open_time}
  keduanya dalam satu transaksi MULTI, jadi tidak pernah setengah tertulis,
- dimuat lagi dari Redis saat start (OnlineBaseline.from_state),
- bila tidak ada state (atau tertinggal lebih jauh dari cache): di-bootstrap di background dari
  Binance REST (30 hari candle 1m), sehingga container baru tidak perlu warm-up 7 hari.
  Data histori yang sama diteruskan ke `on_history` (warm-up normalisasi EWM di engine).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math

import pandas as pd

from app.ml.target import CANDLE_MS, OnlineBaseline, TargetSpec

log = logging.getLogger(__name__)


def trail_key(symbol: str) -> str:
    return f"baseline:{symbol}:trail"


def meta_key(symbol: str) -> str:
    return f"baseline:{symbol}:meta"


def _enc(v: float) -> str:
    return "nan" if math.isnan(v) else repr(float(v))


class BaselineManager:
    def __init__(self, redis, rest_url: str, spec: TargetSpec = TargetSpec(), bootstrap: bool = True,
                 on_history=None, clock=None):
        self.redis = redis
        self.rest_url = rest_url
        self.spec = spec
        self.bootstrap_enabled = bootstrap
        self.on_history = on_history  # async (symbol, DataFrame) -> None
        self.baselines: dict[str, OnlineBaseline] = {}
        self._bootstrapping: set[str] = set()
        self._loaded: set[str] = set()
        self._dirty: set[str] = set()  # penulisan Redis sebelumnya gagal -> tulis ulang penuh
        self.last_error: dict[str, str] = {}
        self.saves = 0
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ persistence (Redis)
    async def load(self, symbol: str) -> OnlineBaseline | None:
        try:
            meta = await self.redis.get(meta_key(symbol))
            if not meta:
                return None
            m = json.loads(meta)
            trail = await self.redis.lrange(trail_key(symbol), 0, -1)
            state = {"spec": m["spec"], "pk": m["pk"], "last_open_time": m["last_open_time"],
                     "trail": [None if v == "nan" else float(v) for v in trail]}
            if TargetSpec(**state["spec"]) != self.spec:
                log.warning("State baseline %s dibuat dengan spec lain, diabaikan", symbol)
                return None
            return OnlineBaseline.from_state(state)
        except Exception as exc:  # noqa: BLE001
            log.warning("State baseline %s tidak bisa dimuat (%s), diabaikan", symbol, exc)
            return None

    def _meta(self, b: OnlineBaseline) -> str:
        st = b.to_state()
        return json.dumps({"spec": st["spec"], "pk": st["pk"], "last_open_time": st["last_open_time"]})

    async def persist(self, symbol: str, b: OnlineBaseline, n_new: int | None = None) -> None:
        """Tulis `n_new` nilai trail terakhir (inkremental) atau seluruh state bila None / terlalu banyak."""
        pipe = self.redis.pipeline(transaction=True)
        trail = list(b._trail)
        if n_new is None or n_new >= len(trail):
            pipe.delete(trail_key(symbol))
            for i in range(0, len(trail), 5000):
                pipe.rpush(trail_key(symbol), *[_enc(v) for v in trail[i:i + 5000]])
        elif n_new > 0:
            pipe.rpush(trail_key(symbol), *[_enc(v) for v in trail[-n_new:]])
            pipe.ltrim(trail_key(symbol), -self.spec.baseline_window, -1)
        pipe.set(meta_key(symbol), self._meta(b))
        await pipe.execute()
        self.saves += 1

    def status(self) -> dict:
        return {s: {"ready": b.ready, "last_open_time": b.last_open_time, "bootstrapping": s in self._bootstrapping,
                    "mu": None if not b.ready else round(b.mu, 5), "sd": None if not b.ready else round(b.sd, 5),
                    "error": self.last_error.get(s)}
                for s, b in self.baselines.items()}

    # ------------------------------------------------------------------ bootstrap
    def _download(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        from ingestion.binance import download_history

        rows = download_history(self.rest_url, symbol, "1m", start_ms, end_ms)
        return pd.DataFrame(rows).drop_duplicates("open_time").sort_values("open_time")

    @staticmethod
    def _feed(b: OnlineBaseline, df: pd.DataFrame) -> int:
        n = 0
        for ot, h, l in zip(df["open_time"].astype("int64"), df["high"], df["low"]):
            if b.last_open_time is not None and ot <= b.last_open_time:
                continue
            if b.last_open_time is not None and ot != b.last_open_time + CANDLE_MS:
                raise ValueError(f"gap: {b.last_open_time} -> {ot}")
            b.update(float(h), float(l), int(ot))
            n += 1
        return n

    async def bootstrap(self, symbol: str, until_open_time: int) -> tuple[OnlineBaseline, pd.DataFrame]:
        """Bangun baseline dari REST sampai `until_open_time` (inklusif)."""
        start = until_open_time - (self.spec.baseline_window + self.spec.horizon) * CANDLE_MS
        log.info("Bootstrap baseline %s dari REST (%d hari)...", symbol, self.spec.baseline_window // 1440)
        df = await asyncio.to_thread(self._download, symbol, start, until_open_time + CANDLE_MS - 1)
        b = OnlineBaseline(self.spec)
        # data sumber bisa berlubang (maintenance Binance): mulai dari segmen kontinu terakhir
        t = df["open_time"].astype("int64").to_numpy()
        breaks = [i for i in range(1, len(t)) if t[i] - t[i - 1] != CANDLE_MS]
        if breaks:
            df = df.iloc[breaks[-1]:]
        await asyncio.to_thread(self._feed, b, df)
        log.info("Baseline %s siap=%s (n=%d candle)", symbol, b.ready, len(df))
        return b, df

    async def _bootstrap_task(self, symbol: str, until_open_time: int) -> None:
        try:
            b, df = await self.bootstrap(symbol, until_open_time)
            self.baselines[symbol] = b
            await self.persist(symbol, b)
            if self.on_history is not None:
                await self.on_history(symbol, df)
        except Exception as exc:  # noqa: BLE001 - mis. REST tidak bisa diakses
            self.last_error[symbol] = f"bootstrap gagal: {exc}"
            log.warning("Bootstrap baseline %s gagal: %s", symbol, exc)
        finally:
            self._bootstrapping.discard(symbol)

    # ------------------------------------------------------------------ update
    async def advance(self, symbol: str, candles: pd.DataFrame) -> OnlineBaseline | None:
        """Majukan baseline sampai candle terakhir di `candles` (cache kontinu). None bila belum siap."""
        if symbol in self._bootstrapping or candles.empty:
            return None
        last_ot = int(candles["open_time"].iloc[-1])
        b = self.baselines.get(symbol)
        if b is None and symbol not in self._loaded:
            self._loaded.add(symbol)
            b = await self.load(symbol)
        try:
            if b is not None and b.last_open_time is not None and b.last_open_time >= last_ot:
                self.baselines[symbol] = b
                return b if b.ready else None
            first_ot = int(candles["open_time"].iloc[0])
            behind = b is None or b.last_open_time is None or b.last_open_time + CANDLE_MS < first_ot
            full = False
            if behind:
                if self.bootstrap_enabled:
                    # bootstrap REST bisa belasan detik: jalankan di background, keputusan ini lewati ledger
                    self._bootstrapping.add(symbol)
                    task = asyncio.create_task(self._bootstrap_task(symbol, last_ot))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                    return None
                b = OnlineBaseline(self.spec)  # tertinggal & tidak bisa bootstrap: mulai ulang dari cache
                full = True
            n = self._feed(b, candles)
        except Exception as exc:  # noqa: BLE001
            self.last_error[symbol] = str(exc)
            log.warning("Baseline %s gagal maju (%s)", symbol, exc)
            return None
        self.baselines[symbol] = b
        try:
            await self.persist(symbol, b, None if full or symbol in self._dirty else n)
            self._dirty.discard(symbol)
        except Exception as exc:  # noqa: BLE001 - Redis sementara bermasalah: state di memori tetap jalan
            self._dirty.add(symbol)
            self.last_error[symbol] = f"simpan ke Redis gagal: {exc}"
        return b if b.ready else None
