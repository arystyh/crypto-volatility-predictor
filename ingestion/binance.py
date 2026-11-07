"""Helper Binance: parsing kline WebSocket & REST, plus download history (untuk training/backfill)."""
from __future__ import annotations

import time

import httpx


def parse_ws_kline(msg: dict) -> dict | None:
    """Ubah event kline WebSocket menjadi payload JSON sederhana.
    Mendukung format combined stream ({"stream":..,"data":..}) maupun raw."""
    data = msg.get("data", msg)
    if data.get("e") != "kline":
        return None
    k = data["k"]
    return {
        "symbol": k["s"],
        "open_time": int(k["t"]),
        "close_time": int(k["T"]),
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"]),
        "trades": int(k["n"]),
        "taker_buy_volume": float(k["V"]),  # volume taker beli (agresor beli) -> VPIN
        "is_closed": bool(k["x"]),
    }


def parse_rest_kline(symbol: str, row: list) -> dict:
    return {
        "symbol": symbol.upper(),
        "open_time": int(row[0]),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "close_time": int(row[6]),
        "trades": int(row[8]),
        "taker_buy_volume": float(row[9]),
    }


def download_history(
    rest_url: str, symbol: str, interval: str, start_ms: int, end_ms: int, timeout: float = 20.0
) -> list[dict]:
    """Download kline historis (paginasi 1000/req) secara sinkron, untuk training."""
    out: list[dict] = []
    cursor = start_ms
    with httpx.Client(timeout=timeout) as client:
        while cursor < end_ms:
            resp = client.get(
                f"{rest_url.rstrip('/')}/api/v3/klines",
                params={
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            out.extend(parse_rest_kline(symbol, r) for r in rows)
            cursor = int(rows[-1][0]) + 1
            time.sleep(0.1)  # sopan terhadap rate limit
    return out
