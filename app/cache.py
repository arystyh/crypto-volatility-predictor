"""Redis: cache N candle terakhir per simbol (sorted set by open_time) + Redis Stream.

Jumlah command per candle dibuat minimal (ZADD + ZREMRANGEBYRANK + XADD dalam
satu pipeline) agar hemat kuota Upstash free tier.
"""
from __future__ import annotations

import json

import pandas as pd
import redis.asyncio as aioredis

CANDLE_FIELDS = ["open_time", "close_time", "open", "high", "low", "close", "volume", "trades", "taker_buy_volume"]


def get_redis(url: str) -> aioredis.Redis:
    return aioredis.from_url(url, decode_responses=True, health_check_interval=30)


def cache_key(symbol: str) -> str:
    return f"candles:{symbol.upper()}"


def serialize_candle(candle: dict) -> str:
    """Serialisasi kanonik: candle yang sama (dari REST maupun WebSocket) -> string
    yang sama, sehingga ZADD otomatis dedupe."""
    clean = {
        "symbol": candle["symbol"].upper(),
        "open_time": int(candle["open_time"]),
        "close_time": int(candle["close_time"]),
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
        "volume": float(candle["volume"]),
        "trades": int(candle.get("trades", 0)),
        "taker_buy_volume": float(candle.get("taker_buy_volume", float(candle["volume"]) / 2)),
    }
    return json.dumps(clean, separators=(",", ":"), sort_keys=True)


def publish_payload(candle: dict) -> str:
    """Payload stream = candle kanonik + waktu diterima ingestion + mikrostruktur live + state stream (breaker)."""
    data = json.loads(serialize_candle(candle))
    data["ingested_at"] = candle.get("ingested_at")
    if candle.get("micro"):
        data["micro"] = candle["micro"]
    if candle.get("stream_state"):
        data["stream_state"] = candle["stream_state"]
    return json.dumps(data, separators=(",", ":"))


async def store_candles(
    r: aioredis.Redis,
    candles: list[dict],
    cache_size: int,
    stream: str | None = None,
    maxlen: int = 5000,
) -> None:
    """Simpan candle ke cache; jika `stream` diisi, XADD juga ke Redis Stream (durable queue)."""
    if not candles:
        return
    pipe = r.pipeline(transaction=False)
    by_symbol: dict[str, list[dict]] = {}
    for c in candles:
        by_symbol.setdefault(c["symbol"].upper(), []).append(c)
    for symbol, items in by_symbol.items():
        key = cache_key(symbol)
        pipe.zadd(key, {serialize_candle(c): int(c["open_time"]) for c in items})
        pipe.zremrangebyrank(key, 0, -(cache_size + 1))
        if stream:
            for c in items:
                pipe.xadd(stream, {"data": publish_payload(c)}, maxlen=maxlen, approximate=True)
    await pipe.execute()


async def get_candles(r: aioredis.Redis, symbol: str, limit: int | None = None) -> pd.DataFrame:
    start = -limit if limit else 0
    raw = await r.zrange(cache_key(symbol), start, -1)
    return candles_to_frame([json.loads(x) for x in raw])


def candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=CANDLE_FIELDS)
    df = pd.DataFrame(candles)
    df = df.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    return df
