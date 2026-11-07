"""Roadmap 3.3: order book lokal kita vs hftbacktest (implementasi referensi) - best bid/ask dan
5 level teratas harus identik di setiap titik pengecekan.

Default: 15 menit depth diff sintetis @100ms (~9 ribu event). Untuk 24 jam data NYATA hasil
recorder: PARITY_TICKS_DIR=data/ticks pytest tests/test_orderbook_parity.py
"""
import json
import os

import numpy as np
import pytest

hbt_mod = pytest.importorskip("hftbacktest")
from hftbacktest import (BUY_EVENT, DEPTH_EVENT, EXCH_EVENT, LOCAL_EVENT, SELL_EVENT,  # noqa: E402
                         BacktestAsset, HashMapMarketDepthBacktest, event_dtype)

from app.services import simticks  # noqa: E402
from app.services.recorder import load  # noqa: E402
from ingestion.orderbook import LocalOrderBook  # noqa: E402
from training.synthetic import generate_candles  # noqa: E402

TICK = 0.01


def _records():
    root = os.environ.get("PARITY_TICKS_DIR")
    if root:
        df = load(root, streams=("depthSnapshot", "depth"))
        return df, float(os.environ.get("PARITY_TICK_SIZE", TICK))
    c = generate_candles("BTCUSDT", days=1, seed=8).iloc[200:215]
    df = simticks.generate(c, seed=8, depth_ms=100, trades_per_min=1, futures=False)
    return df[df["stream"].isin(["depthSnapshot", "depth"])].reset_index(drop=True), TICK


def _hbt_events(applied: list[tuple[int, dict, bool]]) -> np.ndarray:
    ev = []
    flag = DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT
    for ts, d, is_snap in applied:
        for side, key in ((BUY_EVENT, "bids" if is_snap else "b"), (SELL_EVENT, "asks" if is_snap else "a")):
            for p, q in d[key]:
                ev.append((flag | side, ts * 1_000_000, ts * 1_000_000, float(p), float(q), 0, 0, 0.0))
    return np.array(ev, dtype=event_dtype)


def test_local_book_matches_hftbacktest():
    df, tick = _records()
    first_snap = df.index[df["stream"] == "depthSnapshot"][0]
    df = df.loc[first_snap:]
    book = LocalOrderBook("BTCUSDT")
    applied, checkpoints = [], []
    for r in df.itertuples():
        d = json.loads(r.payload)
        if r.stream == "depthSnapshot":
            if applied:
                break  # resnapshot: cukup bandingkan segmen kontinu pertama
            book.apply_snapshot(d)
            applied.append((r.event_time, d, True))
            continue
        before = book.last_update_id
        ok = book.on_event(d)
        if not ok:
            break
        if book.last_update_id != before:  # event benar-benar diterapkan (bukan duplikat)
            applied.append((r.event_time, d, False))
            if len(applied) % 10 == 0:
                bids, asks = book.top(5)
                checkpoints.append((r.event_time, bids, asks))
    assert len(checkpoints) > 100

    events = _hbt_events(applied)  # referensi harus tetap hidup: hftbacktest memegang pointer ke array ini
    asset = (BacktestAsset().data([events]).linear_asset(1.0).constant_order_latency(0, 0)
             .risk_adverse_queue_model().no_partial_fill_exchange().trading_value_fee_model(0.0, 0.0)
             .tick_size(tick).lot_size(1e-8).last_trades_capacity(0))
    hbt = HashMapMarketDepthBacktest([asset])
    now = applied[0][0] * 1_000_000
    mismatches = 0
    for ts, bids, asks in checkpoints:
        target = ts * 1_000_000 + 1
        hbt.elapse(target - now)
        now = target
        depth = hbt.depth(0)
        assert abs(depth.best_bid - bids[0][0]) < tick / 2, (ts, depth.best_bid, bids[0])
        assert abs(depth.best_ask - asks[0][0]) < tick / 2, (ts, depth.best_ask, asks[0])
        for p, q in bids:
            mismatches += abs(depth.bid_qty_at_tick(int(round(p / tick))) - q) > 1e-9
        for p, q in asks:
            mismatches += abs(depth.ask_qty_at_tick(int(round(p / tick))) - q) > 1e-9
    assert mismatches == 0
    del hbt, events
