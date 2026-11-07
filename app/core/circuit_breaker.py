"""Circuit breaker tiga status, dievaluasi PER KEPUTUSAN (deterministik, bisa di-replay).

    OK        data segar, model sehat                       -> prediksi normal, alert boleh
    DEGRADED  model error / timeout (data masih segar)       -> skor dari heuristic Parkinson (fallback),
                                                               alert boleh, ledger diberi status DEGRADED
    STALE     data basi pada salah satu stream wajib          -> SEMUA alert DITAHAN; prediksi tetap
                                                               dicatat ke ledger dengan status STALE

Kesegaran dihitung per stream:  staleness = now - event_time  (event_time = waktu event di bursa).
Selain itu, bila sebuah stream baru saja putus (jeda antar event > STALE_SEC), status tetap STALE
sampai stream itu segar terus selama RECOVERY_SEC (setelah reconnect, order book sedang resync dan
data mikrostruktur belum lengkap).

Input datang dari ingestion sebagai `stream_state` per candle:
    {"kline": {"event_ms": ..., "outage_end_ms": ...}, "aggTrade": {...}, "depth": {...}, ...}
Di proses terpisah, ingestion juga menulis state yang sama ke Redis (`ingestion:streams`) tiap
HEARTBEAT_INTERVAL_SEC, dipakai loop latar untuk /health dan notifikasi saat tidak ada candle sama
sekali (WebSocket mati total).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

OK, DEGRADED, STALE = "OK", "DEGRADED", "STALE"
STATE_CODE = {OK: 0, DEGRADED: 1, STALE: 2}
STREAMS_KEY = "ingestion:streams"


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class BreakerDecision:
    state: str
    reason: str | None = None
    ages_sec: dict = field(default_factory=dict)

    @property
    def allow_alert(self) -> bool:
        return self.state != STALE


class CircuitBreaker:
    def __init__(
        self,
        required_streams: tuple[str, ...] = ("kline",),
        stale_sec: float = 5.0,
        recovery_sec: float = 30.0,
        clock: Callable[[], int] = now_ms,
        notify: Callable[[str], Awaitable] | None = None,
        redis=None,
        check_sec: float = 5.0,
        heartbeat_interval_sec: float = 5.0,
        local_state: Callable[[], dict] | None = None,
    ):
        self.required = tuple(required_streams)
        self.stale_ms = stale_sec * 1000
        self.recovery_ms = recovery_sec * 1000
        self.clock = clock
        self.notify = notify
        self.redis = redis
        self.check_sec = check_sec
        self.report_slack_ms = heartbeat_interval_sec * 1000
        self.local_state = local_state  # ingestion di proses yang sama
        self.state = OK
        self.reason: str | None = None
        self.since = clock()
        self.counts = {OK: 0, DEGRADED: 0, STALE: 0}
        self.transitions = 0
        self.model_error: str | None = None
        self.last_ages: dict = {}
        self._running = False

    # ------------------------------------------------------------------ input
    def model_failed(self, error: str) -> None:
        self.model_error = error

    def model_ok(self) -> None:
        self.model_error = None

    # ------------------------------------------------------------------ evaluasi murni
    def evaluate(self, stream_state: dict | None, now: int | None = None, report_lag_ms: float = 0.0) -> BreakerDecision:
        """Status untuk keputusan pada waktu `now` (ms). Tidak ada efek samping selain cache state."""
        now = self.clock() if now is None else now
        stream_state = stream_state or {}
        ages, problems = {}, []
        for name in self.required:
            st = stream_state.get(name)
            if not st or st.get("event_ms") is None:
                problems.append(f"stream {name}: belum ada event")
                continue
            age = now - st["event_ms"] - report_lag_ms
            ages[name] = round(age / 1000, 3)
            if age > self.stale_ms:
                problems.append(f"stream {name} basi {age / 1000:.1f}s (> {self.stale_ms / 1000:.0f}s)")
            elif st.get("outage_end_ms") and now - st["outage_end_ms"] < self.recovery_ms:
                problems.append(f"stream {name} baru pulih {(now - st['outage_end_ms']) / 1000:.0f}s lalu "
                                f"(tunggu {self.recovery_ms / 1000:.0f}s)")
        for name, st in stream_state.items():  # stream opsional: hanya dilaporkan
            if name not in ages and st and st.get("event_ms") is not None:
                ages[name] = round((now - st["event_ms"] - report_lag_ms) / 1000, 3)
        self.last_ages = ages
        if problems:
            return BreakerDecision(STALE, "; ".join(problems), ages)
        if self.model_error:
            return BreakerDecision(DEGRADED, f"model gagal: {self.model_error}", ages)
        return BreakerDecision(OK, None, ages)

    async def record(self, d: BreakerDecision, decision: bool = True) -> None:
        """Simpan status keputusan & kirim notifikasi Telegram saat status berubah."""
        if decision:
            self.counts[d.state] += 1
        if d.state == self.state:
            self.reason = d.reason
            return
        prev, dur = self.state, (self.clock() - self.since) / 1000
        self.state, self.reason, self.since = d.state, d.reason, self.clock()
        self.transitions += 1
        log.warning("BREAKER %s -> %s: %s", prev, d.state, d.reason)
        if d.state == STALE:
            text = ("🚨 <b>STALE: data basi, alert ditahan</b>\n"
                    f"{d.reason}\nPrediksi tetap dicatat di ledger dengan status STALE.")
        elif d.state == DEGRADED:
            text = f"⚠️ <b>DEGRADED</b>: {d.reason}\nSkor sementara dari heuristic Parkinson (fallback)."
        else:
            text = f"✅ <b>Normal kembali</b> setelah {dur:.0f} detik dalam status {prev}."
        if self.notify:
            try:
                await self.notify(text)
            except Exception:  # noqa: BLE001
                log.exception("Gagal kirim notifikasi breaker")

    def snapshot(self) -> dict:
        return {"state": self.state, "status": self.state, "reason": self.reason, "since_ms": self.since,
                "stream_age_sec": self.last_ages, "stale_sec": self.stale_ms / 1000,
                "recovery_sec": self.recovery_ms / 1000, "required_streams": list(self.required),
                "decisions": dict(self.counts), "transitions": self.transitions, "model_error": self.model_error}

    # ------------------------------------------------------------------ loop latar (status saat tidak ada candle)
    async def _current_stream_state(self) -> tuple[dict, float]:
        if self.local_state is not None:
            return self.local_state(), 0.0
        raw = await self.redis.get(STREAMS_KEY)
        if not raw:
            return {}, 0.0
        data = json.loads(raw)
        # state di Redis ditulis tiap heartbeat_interval: kurangi jeda pelaporan agar tidak false STALE
        lag = max(0.0, min(self.clock() - data.get("_reported_ms", self.clock()), self.report_slack_ms))
        return data.get("streams", {}), lag

    async def check(self) -> BreakerDecision:
        try:
            st, lag = await self._current_stream_state()
            d = self.evaluate(st, report_lag_ms=lag)
        except Exception as exc:  # noqa: BLE001 - Redis mati = tidak bisa memastikan data segar
            d = BreakerDecision(STALE, f"Redis tidak bisa diakses ({exc})")
        await self.record(d, decision=False)
        return d

    async def run(self, startup_grace_sec: float = 30.0) -> None:
        self._running = True
        await asyncio.sleep(startup_grace_sec if self.local_state is None else 0)
        while self._running:
            await self.check()
            await asyncio.sleep(self.check_sec)

    def stop(self) -> None:
        self._running = False
