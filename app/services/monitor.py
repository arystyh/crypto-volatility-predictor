"""Performance monitor berbasis forecast ledger (pengganti drift KS-test).

Setiap MONITOR_INTERVAL_SEC:
 1. Verifikasi hash chain ledger.
 2. Skor prediksi yang sudah "matang" (H candle ke depan sudah close) memakai candle di cache
    -> outcome ditulis append-only (app.services.scorer.score_pending).
 3. Outcome champion yang baru (status != STALE) -> update threshold adaptive conformal (umpan balik
    tertunda, memakai threshold yang tersimpan di baris ledger).
 4. Hitung metrik live 7 hari (PR-AUC, lift vs base rate, Brier, lead time) champion, challenger, hazard,
    tulis badge shields.io ke DATA_DIR/badges.
 5. Degradasi: bila lift PR-AUC champion < PERF_MIN_LIFT (dengan cukup positif), kirim Telegram
    "performa turun, retrain" (maks 1x/24 jam) dan opsional jalankan RETRAIN_COMMAND.

Ini mengukur hal yang benar-benar penting (apakah alert masih berguna), bukan sekadar
apakah distribusi fitur bergeser.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import shlex
import time
from dataclasses import asdict
from pathlib import Path

from app.cache import get_candles
from app.services.ledger import Ledger
from app.services.scorer import badge_json, live_metrics, score_pending

log = logging.getLogger(__name__)

CHAMPION = "champion_xgboost"
CHALLENGER = "challenger_lgbm"
HAZARD = "hazard_xgb"
MODELS = (CHAMPION, CHALLENGER, HAZARD)


def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


class PerformanceMonitor:
    def __init__(self, settings, ledger: Ledger, redis, engine=None, notifier=None):
        self.s = settings
        self.ledger = ledger
        self.redis = redis
        self.engine = engine
        self.notifier = notifier
        self.badge_dir = Path(settings.data_dir) / "badges"
        self.badge_dir.mkdir(parents=True, exist_ok=True)
        self.last: dict = {}
        self._last_degraded_alert = 0.0
        self._last_scored_at = int(time.time() * 1000)
        self._stop = asyncio.Event()

    def metrics(self, days: float | None = None, symbol: str | None = None) -> dict:
        since = int(time.time() * 1000) - int((days or self.s.perf_window_days) * 86_400_000)
        return {m: _clean(asdict(live_metrics(self.ledger, m, since_ts=since, symbol=symbol,
                                               lookback_min=self.s.perf_lookback_min)))
                for m in MODELS}

    def _new_champion_outcomes(self) -> list[tuple[float, float, int]]:
        rows = self.ledger.query(
            "SELECT p.prob, p.threshold, o.label, o.scored_at FROM predictions p JOIN outcomes o ON o.seq = p.seq "
            "WHERE p.model_version = ? AND o.scored_at > ? AND p.status != 'STALE' ORDER BY p.decision_ts",
            (CHAMPION, self._last_scored_at))
        if rows:
            self._last_scored_at = max(r["scored_at"] for r in rows)
        return [(r["prob"], r["threshold"], r["label"]) for r in rows]

    async def check_once(self) -> dict:
        ok, bad = await asyncio.to_thread(self.ledger.verify)
        if not ok:
            log.error("LEDGER TAMPERED: hash chain putus di seq %s", bad)
            if self.notifier:
                await self.notifier.send(f"🛑 <b>Ledger dimanipulasi</b>: hash chain putus di seq {bad}")
        scored = 0
        for sym in self.s.symbol_list:
            candles = await get_candles(self.redis, sym)
            if candles.empty:
                continue
            rep = await asyncio.to_thread(score_pending, self.ledger, candles[["open_time", "high", "low"]], sym)
            scored += rep.scored
        # umpan balik conformal
        if self.engine is not None and self.engine.conformal is not None:
            for prob, thr, label in await asyncio.to_thread(self._new_champion_outcomes):
                self.engine.conformal.update(prob, label, thr)
            await self.engine.save_conformal()
        metrics = await asyncio.to_thread(self.metrics)
        for name, m in metrics.items():
            badge = badge_json(live_metrics(self.ledger, name, since_ts=int(time.time() * 1000)
                                            - int(self.s.perf_window_days * 86_400_000)),
                               f"{self.s.perf_window_days:g}d")
            (self.badge_dir / f"{name}.json").write_text(json.dumps(badge))
            (self.badge_dir / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))
        (self.badge_dir / "ledger_head.txt").write_text(self.ledger.head() + "\n")

        champ = metrics[CHAMPION]
        degraded = (champ["positives"] >= self.s.perf_min_positives and champ["pr_auc_lift"] is not None
                    and champ["pr_auc_lift"] < self.s.perf_min_lift)
        self.last = {"checked_at": int(time.time()), "ledger_ok": ok, "first_bad_seq": bad, "scored_now": scored,
                     "degraded": degraded, "metrics": metrics}
        if degraded and time.time() - self._last_degraded_alert > 86_400:
            self._last_degraded_alert = time.time()
            text = ("📉 <b>Performa model turun</b>\n"
                    f"PR-AUC live {champ['pr_auc']:.2f} = {champ['pr_auc_lift']:.1f}x base rate "
                    f"(minimum {self.s.perf_min_lift}x), n={champ['n']}, {champ['positives']} spike.\n"
                    + ("🔁 Retrain otomatis dijalankan." if self.s.retrain_on_degradation else "Pertimbangkan retrain."))
            if self.notifier:
                await self.notifier.send(text)
            if self.s.retrain_on_degradation:
                proc = await asyncio.create_subprocess_exec(*shlex.split(self.s.retrain_command))
                log.warning("Retrain selesai, exit code %s", await proc.wait())
        return self.last

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.check_once()
            except Exception:  # noqa: BLE001
                log.exception("Performance check gagal")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.s.monitor_interval_sec)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()
