"""Latency & memory profiler ringan (tanpa dependency tambahan).

Tahap yang diukur per candle:
    exchange_lag  : candle close di Binance -> diterima ingestion (WebSocket)
    queue         : diterima ingestion -> diterima engine (Redis pub/sub)
    features      : hitung fitur + regime HMM
    model         : inference XGBoost (atau fallback)
    alert         : kirim Telegram (hanya bila alert)
    processing    : engine terima event -> selesai (fitur + model + alert + log DB)
    end_to_end    : diterima WebSocket -> selesai (termasuk notifikasi terkirim)
"""
from __future__ import annotations

import os
try:
    import resource  # tidak ada di Windows
except ImportError:  # pragma: no cover
    resource = None
import sys
import time
from collections import defaultdict, deque

import numpy as np

STAGES = ["exchange_lag", "queue", "features", "model", "shadow", "hazard", "ledger", "alert", "db", "processing", "end_to_end"]


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def _peak_rss_mb() -> float:
    if resource is None:
        return _rss_mb() or 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


class LatencyTracker:
    def __init__(self, maxlen: int = 5000):
        self.samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.counters: dict[str, int] = defaultdict(int)
        self.started = time.time()

    def observe(self, stage: str, ms: float) -> None:
        if ms is not None and ms >= 0:
            self.samples[stage].append(float(ms))

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def summary(self) -> dict:
        stages = {}
        for name in STAGES:
            data = self.samples.get(name)
            if not data:
                continue
            arr = np.fromiter(data, dtype=float)
            stages[name] = {
                "count": int(arr.size),
                "avg_ms": round(float(arr.mean()), 2),
                "p50_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "p99_ms": round(float(np.percentile(arr, 99)), 2),
                "max_ms": round(float(arr.max()), 2),
            }
        return {
            "uptime_sec": round(time.time() - self.started),
            "latency": stages,
            "counters": dict(self.counters),
            "memory": {"rss_mb": _rss_mb(), "peak_rss_mb": round(_peak_rss_mb(), 1), "pid": os.getpid()},
        }

    def prometheus(self) -> str:
        """Format teks Prometheus untuk scraping (opsional)."""
        s = self.summary()
        lines = []
        for stage, v in s["latency"].items():
            for q in ("p50", "p95", "p99"):
                lines.append(f'cvp_latency_ms{{stage="{stage}",quantile="{q}"}} {v[q + "_ms"]}')
            lines.append(f'cvp_latency_ms_count{{stage="{stage}"}} {v["count"]}')
        for k, v in s["counters"].items():
            lines.append(f'cvp_{k}_total {v}')
        if s["memory"]["rss_mb"] is not None:
            lines.append(f'cvp_memory_rss_mb {s["memory"]["rss_mb"]:.1f}')
        return "\n".join(lines) + "\n"
