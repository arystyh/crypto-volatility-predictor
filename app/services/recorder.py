"""Perekam tick mentah ke Parquet (roadmap 3.1).

Semua pesan mentah dari WebSocket/REST disimpan apa adanya, plus `event_time` (waktu bursa) dan
`recv_time` (jam lokal saat diterima), dipartisi per stream per jam:

    <root>/<stream>/<YYYY-MM-DD>/<HH>/part-<seq_awal>.parquet

Stream: kline, aggTrade, depth (diff), depthSnapshot (REST), fut_aggTrade, forceOrder, markPrice,
openInterest. Kolom: stream, symbol, event_time, recv_time, seq (urutan terima global), id_first,
id_last (aggTradeId / U..u / lastUpdateId), payload (JSON mentah).

`check_continuity()` membuktikan rekaman lengkap: aggTradeId berurutan tanpa lubang, depth
`U == u_sebelumnya + 1` (kecuali tepat setelah snapshot), dan candle 1m tanpa lompatan.
Dipakai juga oleh replay deterministik (app/services/replay.py).
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

STREAMS = ("kline", "aggTrade", "depthSnapshot", "depth", "fut_aggTrade", "markPrice", "forceOrder", "openInterest")
# tie-break tetap untuk event dengan event_time sama (replay deterministik):
# snapshot sebelum diff, trade sebelum kline close (kline close memicu keputusan).
STREAM_RANK = {"depthSnapshot": 0, "depth": 1, "aggTrade": 2, "fut_aggTrade": 3, "markPrice": 4,
               "forceOrder": 5, "openInterest": 6, "kline": 7}

SCHEMA = pa.schema([
    ("stream", pa.string()), ("symbol", pa.string()), ("event_time", pa.int64()), ("recv_time", pa.int64()),
    ("seq", pa.int64()), ("id_first", pa.int64()), ("id_last", pa.int64()), ("payload", pa.string()),
])


def ids_of(stream: str, data: dict) -> tuple[int | None, int | None]:
    if stream in ("aggTrade", "fut_aggTrade"):
        return int(data["a"]), int(data["a"])
    if stream == "depth":
        return int(data["U"]), int(data["u"])
    if stream == "depthSnapshot":
        return int(data["lastUpdateId"]), int(data["lastUpdateId"])
    return None, None


class TickRecorder:
    def __init__(self, root: str | Path, flush_rows: int = 20_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.flush_rows = flush_rows
        self.buf: dict[tuple[str, str], list[dict]] = defaultdict(list)  # (stream, hour) -> rows
        self.seq = 0
        self.rows_written = 0
        self.lock = threading.Lock()

    @staticmethod
    def _hour(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d/%H")

    def record(self, stream: str, symbol: str, data: dict, event_time: int, recv_time: int) -> None:
        id_first, id_last = ids_of(stream, data)
        with self.lock:
            self.seq += 1
            key = (stream, self._hour(event_time))
            self.buf[key].append({"stream": stream, "symbol": symbol, "event_time": int(event_time),
                                  "recv_time": int(recv_time), "seq": self.seq, "id_first": id_first,
                                  "id_last": id_last, "payload": json.dumps(data, separators=(",", ":"))})
            full = len(self.buf[key]) >= self.flush_rows
            # partisi jam sebelumnya sudah tidak akan bertambah: tulis
            old = [k for k in self.buf if k[0] == stream and k[1] < key[1]]
        for k in old:
            self._flush_key(k)
        if full:
            self._flush_key(key)

    def _flush_key(self, key: tuple[str, str]) -> None:
        with self.lock:
            rows = self.buf.pop(key, [])
        if not rows:
            return
        stream, hour = key
        d = self.root / stream / hour
        d.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_table(table, d / f"part-{rows[0]['seq']:012d}.parquet", compression="zstd")
        self.rows_written += len(rows)

    def flush(self) -> None:
        for k in list(self.buf):
            self._flush_key(k)

    close = flush


def load(root: str | Path, streams: tuple[str, ...] | None = None, start_ms: int | None = None,
         end_ms: int | None = None, symbol: str | None = None) -> pd.DataFrame:
    """Baca rekaman, urut (event_time, rank stream, seq) = urutan replay."""
    root = Path(root)
    files = [f for s in (streams or STREAMS) for f in sorted((root / s).glob("*/*/*.parquet"))]
    if not files:
        return pd.DataFrame(columns=[f.name for f in SCHEMA])
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    if start_ms is not None:
        df = df[df["event_time"] >= start_ms]
    if end_ms is not None:
        df = df[df["event_time"] < end_ms]
    if symbol is not None:
        df = df[df["symbol"] == symbol]
    df["rank"] = df["stream"].map(STREAM_RANK).fillna(99).astype(int)
    return df.sort_values(["event_time", "rank", "seq"], kind="mergesort").reset_index(drop=True)


def check_continuity(df: pd.DataFrame) -> dict:
    """Laporan kelengkapan rekaman per stream & simbol. `ok` False bila ada lubang."""
    report: dict = {"ok": True, "streams": {}}
    if df.empty:
        return {"ok": False, "streams": {}, "error": "rekaman kosong"}
    span_h = (df["event_time"].max() - df["event_time"].min()) / 3_600_000
    report["span_hours"] = round(float(span_h), 3)
    for (stream, sym), g in df.sort_values("seq").groupby(["stream", "symbol"]):
        info: dict = {"rows": int(len(g)), "gaps": []}
        if stream in ("aggTrade", "fut_aggTrade"):
            ids = g.sort_values("id_first")["id_first"].to_numpy()
            jumps = (ids[1:] - ids[:-1])
            for i in (jumps > 1).nonzero()[0][:20]:
                info["gaps"].append({"after_id": int(ids[i]), "missing": int(jumps[i] - 1)})
            info["duplicates"] = int((jumps == 0).sum())
        elif stream == "depth":
            snaps = df[(df["stream"] == "depthSnapshot") & (df["symbol"] == sym)]["seq"].to_numpy()
            prev_u, last_seq = None, None
            for seq, U, u in g[["seq", "id_first", "id_last"]].itertuples(index=False):
                resync = last_seq is not None and ((snaps > last_seq) & (snaps < seq)).any()
                if prev_u is not None and not resync and U != prev_u + 1 and u > prev_u:
                    info["gaps"].append({"after_u": int(prev_u), "next_U": int(U)})
                prev_u, last_seq = max(u, prev_u or 0), seq
            info["resnapshots"] = int(len(snaps))
        elif stream == "kline":
            closed = [json.loads(p) for p in g["payload"]]
            ot = sorted({k["k"]["t"] for k in closed if k.get("k", {}).get("x")})
            for a, b in zip(ot, ot[1:]):
                if b - a != 60_000:
                    info["gaps"].append({"after_open_time": a, "missing": (b - a) // 60_000 - 1})
            info["closed_candles"] = len(ot)
        if info["gaps"]:
            report["ok"] = False
        report["streams"][f"{stream}:{sym}"] = info
    return report
