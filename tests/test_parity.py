"""Roadmap 1.5: offline (training/dataset.py) == online (engine, candle demi candle) untuk SEMUA
fitur model, termasuk fitur skala EWM (half-life 1 hari), dan state EWM selamat lewat Redis."""
import asyncio
import json

import numpy as np

from app.cache import store_candles
from app.normalize import MODEL_FEATURES, NORM_FEATURES, EWMNormalizer
from tests.helpers import make_engine, settings
from training.dataset import segment_features
from training.synthetic import generate_candles


def test_offline_online_feature_parity(tmp_path):
    c = generate_candles("BTCUSDT", days=1, seed=11).reset_index(drop=True)
    offline = segment_features(c, halflife=1440)
    s = settings(cache_size=400)
    eng, redis, ledger, notifier, clock = make_engine(tmp_path, s)
    start, n = 700, 40

    async def run():
        # state awal = hasil warm-up histori (seperti BaselineManager.on_history di produksi)
        rows = [{**r, "symbol": "BTCUSDT"} for r in c.to_dict("records")]
        await store_candles(redis, rows[:start], s.cache_size)
        warm = EWMNormalizer(halflife=1440)
        warm.transform_frame(offline.iloc[:start][[f.replace("_ewmz", "") for f in NORM_FEATURES]]
                             .set_axis(warm_cols := [f.replace("_ewmz", "") for f in NORM_FEATURES], axis=1),
                             c["open_time"].iloc[:start])
        await redis.set("norm:BTCUSDT", warm.dumps())
        out = []
        for i in range(start, start + n):
            await store_candles(redis, rows[i:i + 1], s.cache_size)
            clock.advance_to(rows[i]["close_time"] + 50)
            res = await eng.evaluate("BTCUSDT", stream_state={k: {"event_ms": clock()} for k in s.required_streams})
            out.append(res.features)
            if i == start + n // 2:  # restart di tengah: state EWM dari Redis
                eng.normalizers.clear()
        return out

    online = asyncio.run(run())
    off = offline.iloc[start:start + n][MODEL_FEATURES].to_numpy()
    on = np.array([[f[k] for k in MODEL_FEATURES] for f in online])
    rel = np.abs(on - off) / np.maximum(np.abs(off), 1e-9)
    worst = MODEL_FEATURES[int(np.unravel_index(rel.argmax(), rel.shape)[1])]
    assert rel.max() < 1e-6, f"fitur {worst} beda {rel.max():.2e}"
    st = json.loads(asyncio.run(redis.get("norm:BTCUSDT")))
    assert st["_halflife"] == 1440 and st["_last_time"] == int(c["open_time"].iloc[start + n - 1])


def test_normalizer_halflife_is_configurable_and_guarded():
    a = EWMNormalizer(halflife=1440)
    assert abs((1 - a.alpha) ** 1440 - 0.5) < 1e-9
    try:
        EWMNormalizer.loads(a.dumps(), halflife=240)
    except ValueError as exc:
        assert "half-life" in str(exc)
    else:
        raise AssertionError("state half-life lain harus ditolak")
