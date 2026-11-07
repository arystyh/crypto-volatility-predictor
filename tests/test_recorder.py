"""Roadmap 3.1: rekaman Parquet per stream per jam dapat dibaca kembali tanpa lubang."""
import json

from app.services import simticks
from app.services.recorder import TickRecorder, check_continuity, load
from training.synthetic import generate_candles


def _record(tmp_path, **kw):
    c = generate_candles("BTCUSDT", days=1, seed=2).iloc[100:220]  # 2 jam
    rec = TickRecorder(tmp_path, flush_rows=5_000)
    simticks.to_recorder(simticks.generate(c, seed=4, trades_per_min=50, **kw), rec)
    return load(tmp_path)


def test_roundtrip_hourly_partitions_and_no_gaps(tmp_path):
    df = _record(tmp_path)
    hours = {p.parent.name for p in tmp_path.glob("aggTrade/*/*/*.parquet")}
    assert len(hours) >= 2  # dipartisi per jam
    assert {"kline", "aggTrade", "depth", "depthSnapshot", "markPrice", "fut_aggTrade"} <= set(df["stream"])
    assert (df["recv_time"] >= df["event_time"]).all()
    rep = check_continuity(df)
    assert rep["ok"], rep
    assert rep["streams"]["kline:BTCUSDT"]["closed_candles"] == 120
    assert rep["span_hours"] >= 1.9
    # payload mentah utuh
    first = json.loads(df[df["stream"] == "aggTrade"]["payload"].iloc[0])
    assert {"a", "p", "q", "T", "m"} <= set(first)


def test_detects_missing_trades_and_depth_updates(tmp_path):
    df = _record(tmp_path, skip_depth_every=97)
    agg = df[df["stream"] == "aggTrade"]
    df = df.drop(agg.index[500:503])  # 3 aggTrade hilang
    rep = check_continuity(df)
    assert not rep["ok"]
    assert rep["streams"]["aggTrade:BTCUSDT"]["gaps"][0]["missing"] == 3
    assert rep["streams"]["depth:BTCUSDT"]["gaps"]
