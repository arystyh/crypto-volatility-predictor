"""Inference engine: Redis Stream (consumer group) -> fitur -> model -> forecast ledger -> alert.

Per candle close (satu "keputusan"):
 1. Circuit breaker dievaluasi dari `stream_state` payload ingestion:
        STALE     data basi        -> alert DITAHAN, prediksi tetap dicatat (status STALE)
        DEGRADED  model gagal      -> skor heuristic Parkinson, dicatat sebagai `fallback_parkinson`
        OK                         -> normal
 2. OnlineBaseline (mu, sd target 30 hari) dimajukan; state ke Redis tiap candle.
 3. Fitur (app.features) + normalisasi EWM online (state Redis) -> jalur kode sama dengan training.
 4. Champion XGBoost (ONNX) -> kalibrasi isotonic -> gate rezim HMM; challenger LightGBM (shadow);
    model hazard -> P(spike mulai <= 5/10/15 menit).
 5. Threshold alert = adaptive conformal (ACI) atau statis dari kurva PR validasi.
 6. Ledger: champion, challenger, hazard (P15) dicatat SEBELUM outcome ada (hash chain).
 7. Alert Telegram bila prob >= threshold, status != STALE, dan tidak sedang cooldown.

Semua waktu memakai `clock()` yang bisa disuntik -> replay deterministik (app/services/replay.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field

import pandas as pd
from redis.exceptions import ResponseError

from app.cache import candles_to_frame, get_candles
from app.conformal import AdaptiveConformalThreshold
from app.config import Settings
from app.core.circuit_breaker import DEGRADED, OK, STALE, BreakerDecision, CircuitBreaker
from app.fallback import FallbackRule, heuristic_from_candles
from app.features import FEATURE_COLUMNS, MIN_CANDLES, compute_features
from app.metrics import LatencyTracker
from app.ml.target import TargetSpec
from app.normalize import EWMNormalizer
from app.regime import current_regime
from app.services.ledger import LateLogError, Prediction
from app.telegram import format_alert

log = logging.getLogger(__name__)

THRESHOLD_KEY = "config:alert_threshold"
CONFORMAL_KEY = "config:conformal"
FEATURE_WINDOW = 300       # candle yang dipakai menghitung fitur (>= MIN_CANDLES)
GROUP = "inference"
CHAMPION, CHALLENGER, HAZARD, FALLBACK = "champion_xgboost", "challenger_lgbm", "hazard_xgb", "fallback_parkinson"


def now_ms() -> int:
    return int(time.time() * 1000)


def cooldown_key(symbol: str) -> str:
    return f"alert:cooldown:{symbol}"


def norm_key(symbol: str) -> str:
    return f"norm:{symbol}"


@dataclass
class PredictionResult:
    symbol: str
    candle_time: int
    close: float
    probability: float
    threshold: float
    is_spike: bool
    alerted: bool
    status: str = OK
    mode: str = "model"          # model | fallback | gated
    direction: str = "two_way"   # downside | upside | two_way
    regime: str | None = None
    regime_p_high: float | None = None
    raw_probability: float | None = None
    challenger_probability: float | None = None
    hazard: dict | None = None
    contributors: list[dict] | None = None
    breaker_reason: str | None = None
    ledger_seq: int | None = None
    ledger_note: str | None = None
    micro: dict | None = None
    alert_error: str | None = None
    held_reason: str | None = None
    features: dict = field(default_factory=dict)


def volatility_direction(features: dict, hi: float = 0.6, lo: float = 0.4) -> str:
    share = features.get("downside_share_15", 0.5)
    if share >= hi:
        return "downside"
    if share <= lo:
        return "upside"
    return "two_way"


def contiguous_tail(candles: pd.DataFrame) -> pd.DataFrame:
    """Bagian akhir cache yang kontinu (fitur & baseline butuh candle 1m tanpa lubang)."""
    t = candles["open_time"].astype("int64").to_numpy()
    if len(t) < 2:
        return candles
    breaks = (t[1:] - t[:-1]) != 60_000
    if not breaks.any():
        return candles
    last_break = int(breaks.nonzero()[0][-1]) + 1
    return candles.iloc[last_break:].reset_index(drop=True)


class InferenceEngine:
    def __init__(self, settings: Settings, redis, db, model, notifier, *, ledger=None, baselines=None,
                 breaker: CircuitBreaker | None = None, challenger=None, hazard=None, hmm=None, calibrator=None,
                 reasons=None, contributions=None, metrics: LatencyTracker | None = None, clock=now_ms,
                 fallback: FallbackRule | None = None, spec: TargetSpec = TargetSpec()):
        self.settings = settings
        self.redis = redis
        self.db = db
        self.model = model
        self.notifier = notifier
        self.ledger = ledger
        self.baselines = baselines
        self.breaker = breaker or CircuitBreaker(settings.required_streams, settings.stream_stale_sec,
                                                 settings.breaker_recovery_sec, clock=clock)
        self.challenger = challenger
        self.hazard = hazard
        self.hmm = hmm
        self.calibrator = calibrator
        self.reasons = reasons
        self.contributions = contributions
        self.metrics = metrics or LatencyTracker()
        self.clock = clock
        self.fallback = fallback or FallbackRule()
        self.spec = spec
        meta = model.metadata if model is not None else {}
        self.model_version = meta.get("version")
        self.halflife = float(meta.get("ewm_halflife_min") or settings.ewm_halflife_min)
        if meta.get("ewm_halflife_min") and float(meta["ewm_halflife_min"]) != settings.ewm_halflife_min:
            log.error("EWM_HALFLIFE_MIN=%s berbeda dengan model (%s): memakai nilai model agar tidak skew",
                      settings.ewm_halflife_min, meta["ewm_halflife_min"])
        self.static_threshold = float(settings.alert_prob_threshold or meta.get("alert_prob_threshold") or 0.5)
        self.conformal: AdaptiveConformalThreshold | None = None
        if settings.conformal_enabled:
            self.conformal = AdaptiveConformalThreshold(
                tau=self.static_threshold, alpha=settings.conformal_alpha, gamma=settings.conformal_gamma,
                tau_min=settings.conformal_tau_min, tau_max=settings.conformal_tau_max, mode=settings.conformal_mode)
        self.normalizers: dict[str, EWMNormalizer] = {}
        self.running = False
        self.processed = 0
        self.fallback_count = 0
        self.last_error: str | None = None
        self.last_model_error: str | None = None
        self.last_results: dict[str, PredictionResult] = {}

    # ------------------------------------------------------------------ threshold runtime
    @property
    def threshold(self) -> float:
        return self.conformal.tau if self.conformal is not None else self.static_threshold

    @property
    def threshold_kind(self) -> str:
        return f"adaptif ACI, target alarm palsu {self.conformal.alpha:.0%}" if self.conformal else "statis"

    async def load_state(self) -> None:
        try:
            raw = await self.redis.get(THRESHOLD_KEY)
            if raw is not None:
                self.static_threshold = float(raw)
            if self.conformal is not None:
                saved = await self.redis.get(CONFORMAL_KEY)
                if saved:
                    c = AdaptiveConformalThreshold.loads(saved)
                    # target & laju dari config menang; tau hasil adaptasi dipertahankan
                    c.alpha, c.gamma, c.mode = self.conformal.alpha, self.conformal.gamma, self.conformal.mode
                    c.tau_min, c.tau_max = self.conformal.tau_min, self.conformal.tau_max
                    self.conformal = c
        except Exception as exc:  # noqa: BLE001
            log.warning("Gagal baca state threshold dari Redis: %s", exc)

    load_threshold = load_state

    async def save_conformal(self) -> None:
        if self.conformal is not None:
            await self.redis.set(CONFORMAL_KEY, self.conformal.dumps())

    async def set_threshold(self, value: float) -> None:
        if not 0.0 < float(value) < 1.0:
            raise ValueError("Threshold harus di antara 0 dan 1 (eksklusif)")
        self.static_threshold = float(value)
        await self.redis.set(THRESHOLD_KEY, self.static_threshold)
        if self.conformal is not None:  # override manual: ACI lanjut beradaptasi dari nilai ini
            self.conformal.tau = float(min(max(value, self.conformal.tau_min), self.conformal.tau_max))
            await self.save_conformal()
        log.warning("Threshold alert diubah menjadi %.3f", value)

    async def set_alpha(self, value: float) -> None:
        if not 0.0 < float(value) < 1.0:
            raise ValueError("Alpha harus di antara 0 dan 1 (eksklusif)")
        if self.conformal is None:
            raise ValueError("Adaptive conformal tidak aktif (CONFORMAL_ENABLED=false)")
        self.conformal.alpha = float(value)
        await self.save_conformal()

    # ------------------------------------------------------------------ normalisasi
    async def _normalizer(self, symbol: str, feats_df: pd.DataFrame, open_times: pd.Series) -> EWMNormalizer:
        if symbol in self.normalizers:
            return self.normalizers[symbol]
        raw = None
        try:
            raw = await self.redis.get(norm_key(symbol))
        except Exception:  # noqa: BLE001
            pass
        try:
            norm = EWMNormalizer.loads(raw, halflife=self.halflife)
        except ValueError as exc:  # state dari half-life lain
            log.warning("%s; normalizer diinisialisasi ulang", exc)
            raw, norm = None, EWMNormalizer(halflife=self.halflife)
        if not raw:  # warm-up sementara dari cache; diganti histori 30 hari setelah bootstrap baseline
            norm.transform_frame(feats_df.iloc[:-1], open_times.iloc[:-1])
        self.normalizers[symbol] = norm
        return norm

    async def warm_normalizer(self, symbol: str, history: pd.DataFrame) -> None:
        """Dipanggil BaselineManager setelah bootstrap REST: hangatkan EWM dari histori panjang + cache."""
        cache = await get_candles(self.redis, symbol)
        df = candles_to_frame(pd.concat([history, cache], ignore_index=True).to_dict("records"))
        df = contiguous_tail(df)
        cur = self.normalizers.get(symbol)
        upto = cur.last_time if cur is not None and cur.last_time > 0 else int(df["open_time"].iloc[-1])
        df = df[df["open_time"] <= upto].reset_index(drop=True)
        if len(df) < MIN_CANDLES:
            return

        def build():
            feats = compute_features(df)
            norm = EWMNormalizer(halflife=self.halflife)
            norm.transform_frame(feats, df["open_time"])
            return norm

        norm = await asyncio.to_thread(build)
        self.normalizers[symbol] = norm
        await self.redis.set(norm_key(symbol), norm.dumps())
        log.info("Normalizer %s dihangatkan dari %d candle histori", symbol, len(df))

    # ------------------------------------------------------------------ komponen
    def _compute(self, candles: pd.DataFrame):
        feats_df = compute_features(candles)
        regime = None
        if self.hmm is not None:
            valid = feats_df.dropna(subset=["rv_15"])
            if not valid.empty:
                regime = current_regime(self.hmm, valid, self.settings.hmm_gate, self.settings.hmm_gate_window)
        return feats_df, regime

    async def _champion(self, features: pd.Series) -> float:
        prob = await asyncio.wait_for(asyncio.to_thread(self.model.predict_one, features),
                                      timeout=self.settings.model_timeout_sec)
        if self.calibrator is not None:
            prob = float(self.calibrator(prob))
        return float(prob)

    async def _cooldown_acquire(self, symbol: str) -> bool:
        """Anti-spam berbasis jam engine (deterministik saat replay). True = boleh kirim."""
        ttl_ms = int(self.settings.alert_cooldown_minutes * 60_000)
        if ttl_ms <= 0:
            return True
        now = self.clock()
        try:
            until = await self.redis.get(cooldown_key(symbol))
            if until is not None and now < int(until):
                return False
            await self.redis.set(cooldown_key(symbol), now + ttl_ms, px=ttl_ms)
            return True
        except Exception as exc:  # noqa: BLE001 - Redis bermasalah: jangan diam total
            log.warning("Cooldown Redis gagal (%s); alert tetap dikirim", exc)
            return True

    def _ledger_log(self, symbol: str, decision_ts: int, version: str, prob: float, threshold: float,
                    baseline, status: str, conformal: bool = False) -> tuple[int | None, str | None]:
        c = self.conformal if conformal else None
        p = Prediction(symbol, decision_ts, self.spec.horizon, version, float(min(max(prob, 0.0), 1.0)),
                       float(threshold), float(baseline.mu), float(baseline.sd), self.spec.z_threshold,
                       status=status, conformal_alpha=c.alpha if c else None, conformal_gamma=c.gamma if c else None)
        try:
            return self.ledger.log_prediction(p), None
        except sqlite3.IntegrityError:  # sudah tercatat (pesan stream diproses ulang setelah restart)
            return None, "duplikat"
        except LateLogError as exc:      # backlog lama: tidak boleh dicatat setelah fakta
            self.metrics.inc("ledger_late_rejected")
            return None, f"terlambat: {exc}"

    # ------------------------------------------------------------------ core
    async def evaluate(self, symbol: str, candles: pd.DataFrame | None = None, source: str = "stream",
                       log_to_db: bool = True, allow_alert: bool = True, micro: dict | None = None,
                       stream_state: dict | None = None) -> PredictionResult:
        symbol = symbol.upper()
        is_stream = source == "stream"
        if candles is None:
            candles = await get_candles(self.redis, symbol, limit=FEATURE_WINDOW if not is_stream else None)
        candles = contiguous_tail(candles)
        if len(candles) < MIN_CANDLES:
            raise ValueError(f"Butuh minimal {MIN_CANDLES} candle 1m kontinu, baru ada {len(candles)}")
        window = candles.iloc[-FEATURE_WINDOW:].reset_index(drop=True)
        last = window.iloc[-1]
        open_time = int(last["open_time"])
        decision_ts = open_time + 60_000  # = close time candle terakhir = awal jendela forward
        t0 = time.perf_counter()

        # 1) baseline target (frozen mu, sd untuk ledger)
        baseline = await self.baselines.advance(symbol, candles) if (is_stream and self.baselines) else None

        # 2) fitur
        feats: dict = {}
        regime = None
        features = None
        try:
            feats_df, regime = await asyncio.to_thread(self._compute, window)
            base = feats_df.iloc[-1]
            if base[FEATURE_COLUMNS].isna().any():
                raise ValueError(f"Fitur belum lengkap: {base[base.isna()].index.tolist()}")
            norm = await self._normalizer(symbol, feats_df, window["open_time"])
            update = is_stream and open_time > norm.last_time
            z = norm.transform(base, update=update, open_time=open_time)
            if update:
                await self.redis.set(norm_key(symbol), norm.dumps())
            features = pd.concat([base, pd.Series(z)])
            feats = {k: float(v) for k, v in features.items()}
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - feature engine gagal -> perlakukan sebagai model gagal
            log.exception("Feature engine gagal: %s", exc)
            self.breaker.model_failed(f"feature engine: {exc}")
        t1 = time.perf_counter()
        self.metrics.observe("features", (t1 - t0) * 1000)

        # 3) champion (atau fallback), lalu status breaker (hanya jalur stream; prediksi manual
        #    tidak memengaruhi status sistem)
        raw_prob = None
        mode = "model"
        if features is not None and self.model is not None:
            try:
                raw_prob = await self._champion(features)
                self.breaker.model_ok()
                self.metrics.inc("model_calls")
            except Exception as exc:  # noqa: BLE001 - termasuk TimeoutError
                self.last_model_error = f"{type(exc).__name__}: {exc}"
                self.breaker.model_failed(self.last_model_error)
                log.error("Model gagal (%s) -> DEGRADED, fallback Parkinson", self.last_model_error)
        elif self.model is None:
            self.breaker.model_failed("model tidak dimuat")
        if is_stream:
            decision = self.breaker.evaluate(stream_state, now=self.clock())
            await self.breaker.record(decision)
        else:
            decision = BreakerDecision(OK if raw_prob is not None else DEGRADED)
        status = decision.state

        if raw_prob is None:
            prob, _ = heuristic_from_candles(window, self.fallback)
            mode = "fallback"
            self.fallback_count += 1
            self.metrics.inc("fallback_predictions")
        elif regime is not None and not regime.gate_open:
            prob, mode = 0.0, "gated"  # gate rezim HMM: champion tidak boleh memicu alert
            self.metrics.inc("hmm_gated")
        else:
            prob = raw_prob
        t2 = time.perf_counter()
        self.metrics.observe("model", (t2 - t1) * 1000)

        # 4) shadow challenger & hazard
        challenger_prob = hazard = None
        if features is not None and self.challenger is not None:
            try:
                challenger_prob = await asyncio.to_thread(self.challenger.predict_one, features)
            except Exception as exc:  # noqa: BLE001
                log.warning("Challenger gagal: %s", exc)
        t3 = time.perf_counter()
        self.metrics.observe("shadow", (t3 - t2) * 1000)
        if features is not None and self.hazard is not None:
            try:
                hazard = (await asyncio.to_thread(self.hazard.curve, features)).to_dict()
            except Exception as exc:  # noqa: BLE001
                log.warning("Hazard gagal: %s", exc)
        self.metrics.observe("hazard", (time.perf_counter() - t3) * 1000)

        threshold = self.threshold
        result = PredictionResult(
            symbol=symbol, candle_time=open_time, close=float(last["close"]), probability=float(prob),
            threshold=threshold, is_spike=prob >= threshold, alerted=False, status=status, mode=mode,
            direction=volatility_direction(feats), regime=regime.regime if regime else None,
            regime_p_high=regime.p_high if regime else None, raw_probability=raw_prob,
            challenger_probability=challenger_prob, hazard=hazard, breaker_reason=decision.reason,
            micro=micro, features=feats,
        )

        # 5) ledger SEBELUM alert (outcome belum ada; link baris ikut di pesan)
        ledger_row = None
        if is_stream and self.ledger is not None:
            tl = time.perf_counter()
            if baseline is None:
                result.ledger_note = "baseline belum siap (warm-up/bootstrap)"
            else:
                version = CHAMPION if mode != "fallback" else FALLBACK
                seq, note = self._ledger_log(symbol, decision_ts, version, prob, threshold, baseline, status,
                                             conformal=version == CHAMPION)
                result.ledger_seq, result.ledger_note = seq, note
                if challenger_prob is not None:
                    self._ledger_log(symbol, decision_ts, CHALLENGER, challenger_prob, self.static_threshold,
                                     baseline, status)
                if hazard is not None:
                    self._ledger_log(symbol, decision_ts, HAZARD, hazard["p_within"].get(15, hazard["p_onset_within_h"]),
                                     self.static_threshold, baseline, status)
                if seq is not None:
                    ledger_row = self.ledger.get(seq)
            self.metrics.observe("ledger", (time.perf_counter() - tl) * 1000)

        # 6) alert
        if result.is_spike and allow_alert:
            if status == STALE:
                result.held_reason = f"STALE: {decision.reason}"
                self.metrics.inc("alerts_held_stale")
                log.warning("%s prob=%.3f >= threshold tapi data basi: alert DITAHAN (%s)", symbol, prob,
                            decision.reason)
            elif not await self._cooldown_acquire(symbol):
                result.held_reason = "cooldown"
                self.metrics.inc("alerts_suppressed_cooldown")
            else:
                await self._send_alert(result, features, ledger_row)

        if log_to_db and self.db is not None:
            td = time.perf_counter()
            await asyncio.to_thread(
                self.db.log_prediction, symbol=symbol, candle_time=open_time, close=result.close,
                probability=result.probability, threshold=threshold, alerted=result.alerted,
                features={**feats, **{f"live_{k}": v for k, v in (micro or {}).items()
                                      if isinstance(v, (int, float)) and not isinstance(v, bool)}},
                source=source, mode=mode, direction=result.direction, regime_p_high=result.regime_p_high,
                challenger_probability=challenger_prob, status=status, ledger_seq=result.ledger_seq)
            self.metrics.observe("db", (time.perf_counter() - td) * 1000)
        self.last_results[symbol] = result
        return result

    async def _send_alert(self, result: PredictionResult, features: pd.Series | None, ledger_row: dict | None):
        contributors = []
        if features is not None and self.contributions is not None and result.mode == "model":
            try:
                result.contributors = await asyncio.to_thread(self.contributions.top, features, 3)
                from app.explain import format_contrib

                contributors = [format_contrib(c) for c in result.contributors]
            except Exception as exc:  # noqa: BLE001
                log.warning("Kontribusi fitur gagal: %s", exc)
        reasons = self.reasons.build(result.features, None, result.regime, result.direction, result.micro,
                                     result.mode) if self.reasons else None
        ledger = None
        if ledger_row:
            base = self.settings.public_base_url.rstrip("/")
            ledger = {"seq": ledger_row["seq"], "row_hash": ledger_row["row_hash"],
                      "url": f"{base}/ledger/row/{ledger_row['seq']}" if base else None}
        text = format_alert(result.symbol, result.probability, result.close, result.threshold, result.features,
                            result.mode, result.direction, result.regime, reasons, result.hazard, contributors,
                            result.status, self.threshold_kind, ledger)
        ta = time.perf_counter()
        sent, error = await self.notifier.send(text)
        self.metrics.observe("alert", (time.perf_counter() - ta) * 1000)
        if self.db is not None:
            self.db.log_alert(result.symbol, result.probability, result.close, text, sent, error, result.mode,
                              result.direction)
        result.alerted = True
        result.alert_error = error
        self.metrics.inc("alerts")
        log.warning("ALERT %s prob=%.3f mode=%s status=%s sent=%s %s", result.symbol, result.probability,
                    result.mode, result.status, sent, error or "")

    # ------------------------------------------------------------------ jalur stream
    async def process(self, data: str | dict) -> PredictionResult | None:
        """Satu pesan stream (candle close dari ingestion)."""
        t0 = time.perf_counter()
        candle = json.loads(data) if isinstance(data, str) else data
        now = self.clock()
        ingested = candle.get("ingested_at") or candle["close_time"]
        self.metrics.observe("queue", now - ingested)
        if candle.get("ingested_at") and 0 <= ingested - candle["close_time"] < 3_600_000:
            self.metrics.observe("exchange_lag", ingested - candle["close_time"])
        try:
            res = await self.evaluate(candle["symbol"], micro=candle.get("micro"),
                                      stream_state=candle.get("stream_state"))
        except ValueError as exc:  # history belum cukup, dsb.
            log.info("Lewati prediksi: %s", exc)
            return None
        self.processed += 1
        self.metrics.inc("predictions")
        self.metrics.observe("processing", (time.perf_counter() - t0) * 1000)
        self.metrics.observe("end_to_end", self.clock() - ingested)
        log.info("%s close=%.4f prob=%.3f thr=%.3f [%s, %s, %s] seq=%s%s", res.symbol, res.close, res.probability,
                 res.threshold, res.status, res.mode, res.regime, res.ledger_seq, " ALERT" if res.alerted else "")
        return res

    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(self.settings.stream_key, GROUP, id="$", mkstream=True)
            log.info("Consumer group '%s' dibuat di stream '%s'", GROUP, self.settings.stream_key)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def run(self) -> None:
        """Loop consumer dengan auto-reconnect. Pesan pending (belum di-ACK) diproses ulang dulu."""
        self.running = True
        await self.load_state()
        s = self.settings
        backoff = 1
        while self.running:
            try:
                await self.ensure_group()
                # 1) ambil alih pesan milik consumer mati yang menggantung > 60 detik
                try:
                    await self.redis.xautoclaim(s.stream_key, GROUP, s.stream_consumer, min_idle_time=s.stream_claim_idle_ms,
                                                start_id="0-0", count=100)
                except ResponseError:
                    pass
                # 2) proses pending milik consumer ini (mis. API restart di tengah proses), lalu pesan baru
                last_id = "0"
                while self.running:
                    resp = await self.redis.xreadgroup(GROUP, s.stream_consumer, {s.stream_key: last_id},
                                                       count=50, block=None if last_id == "0" else s.stream_block_ms)
                    entries = resp[0][1] if resp else []
                    if last_id == "0" and not entries:
                        last_id = ">"
                        log.info("Engine membaca stream '%s' sebagai '%s'", s.stream_key, s.stream_consumer)
                        continue
                    if not entries:
                        await asyncio.sleep(0)
                        continue
                    for msg_id, fields in entries:
                        try:
                            await self.process(fields.get("data", "{}"))
                        except Exception as exc:  # noqa: BLE001 - satu pesan rusak tidak boleh menghentikan loop
                            self.last_error = str(exc)
                            self.metrics.inc("errors")
                            log.exception("Gagal memproses event %s", msg_id)
                        await self.redis.xack(s.stream_key, GROUP, msg_id)
                backoff = 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - Redis putus dll.: jaga loop tetap hidup
                self.last_error = str(exc)
                log.exception("Engine error, reconnect dalam %ss", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
