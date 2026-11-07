"""FastAPI service: engine prediksi, forecast ledger publik, health/metrics, bot Telegram.

Saat start (RUN_ENGINE=true) service ini menjalankan:
    InferenceEngine        consumer Redis Stream -> fitur -> model -> ledger -> alert
    CircuitBreaker loop    status OK/DEGRADED/STALE untuk /health saat feed mati total
    PerformanceMonitor     tiap 5 menit: verifikasi hash chain, skor ledger, update ACI, badge
    TelegramBot            polling / webhook
    IngestionWorker        opsional (RUN_INGESTION=true, mode hosting gratis 1 service)

Halaman publik read-only: / (ledger explorer, verifikasi hash chain di browser).

Jalankan:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from datetime import timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.cache import candles_to_frame, get_candles, get_redis
from app.calibration import IsotonicCalibrator
from app.config import get_settings
from app.core.circuit_breaker import STATE_CODE, CircuitBreaker
from app.db import Database
from app.engine import InferenceEngine
from app.features import MIN_CANDLES
from app.metrics import LatencyTracker
from app.model import ChallengerModel, VolatilityModel
from app.reason import ReasonBuilder
from app.regime import GaussianHMM
from app.services.baseline import BaselineManager
from app.services.ledger import Ledger
from app.services.monitor import PerformanceMonitor
from app.telegram import TelegramNotifier
from app.telegram_bot import TelegramBot

log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"
REPORTS = Path("reports")


class State:
    engine: InferenceEngine
    ledger: Ledger
    monitor: PerformanceMonitor | None = None
    breaker: CircuitBreaker
    tasks: list[asyncio.Task]
    ingestion = None
    bot: TelegramBot | None = None
    model_error: str | None = None
    started_at: float = time.time()


state = State()


def _load_optional(name: str, fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - komponen opsional: tetap hidup tanpa
        log.warning("%s tidak dimuat: %s", name, exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    redis = getattr(app.state, "redis_override", None) or get_redis(s.redis_url)
    db = Database(s.database_url)
    Path(s.ledger_path).parent.mkdir(parents=True, exist_ok=True)
    state.ledger = Ledger(s.ledger_path, max_log_lag_ms=s.ledger_max_log_lag_ms)
    try:
        model = VolatilityModel(s.model_path, backend=s.inference_backend)
        state.model_error = None
    except Exception as exc:  # noqa: BLE001 - tetap hidup dengan fallback (status DEGRADED)
        model, state.model_error = None, str(exc)
        log.error("Model gagal dimuat (%s): semua keputusan DEGRADED (fallback Parkinson)", exc)
    hmm = _load_optional("HMM regime", lambda: GaussianHMM.load(s.hmm_path)) if s.hmm_gate != "off" else None
    calibrator = _load_optional("Kalibrasi isotonic", lambda: IsotonicCalibrator.load(s.calibration_path))
    challenger = hazard = contributions = None
    if model is not None:
        if s.enable_challenger:
            challenger = _load_optional("Challenger LightGBM", lambda: ChallengerModel(s.challenger_path, model.features))
        from app.hazard import HazardModel
        from app.explain import Contributions

        if Path(s.hazard_path).exists():
            hazard = _load_optional("Model hazard", lambda: HazardModel(s.hazard_path, model.features))
        contributions = Contributions(s.model_path, model.features)
    notifier = TelegramNotifier(s.telegram_bot_token, s.telegram_chat_id)
    state.started_at = time.time()
    state.tasks, state.ingestion, state.bot, state.monitor = [], None, None, None
    if s.run_ingestion:
        from ingestion.worker import IngestionWorker

        state.ingestion = IngestionWorker(s, redis=redis)
    state.breaker = CircuitBreaker(
        s.required_streams, s.stream_stale_sec, s.breaker_recovery_sec,
        notify=notifier.send if notifier.enabled else None, redis=redis, check_sec=s.breaker_check_sec,
        heartbeat_interval_sec=s.heartbeat_interval_sec,
        local_state=state.ingestion.stream_state if state.ingestion else None)
    baselines = BaselineManager(redis, s.binance_rest_url, bootstrap=s.baseline_bootstrap)
    state.engine = InferenceEngine(
        s, redis, db, model, notifier, ledger=state.ledger, baselines=baselines, breaker=state.breaker,
        challenger=challenger, hazard=hazard, hmm=hmm, calibrator=calibrator,
        reasons=ReasonBuilder(s.feature_quantiles_path), contributions=contributions, metrics=LatencyTracker())
    baselines.on_history = state.engine.warm_normalizer
    await state.engine.load_state()

    if s.run_engine:
        state.tasks.append(asyncio.create_task(state.engine.run(), name="engine"))
        state.tasks.append(asyncio.create_task(state.breaker.run(), name="breaker"))
    if s.run_monitor:
        state.monitor = PerformanceMonitor(s, state.ledger, redis, engine=state.engine,
                                           notifier=notifier if notifier.enabled else None)
        state.tasks.append(asyncio.create_task(state.monitor.run(), name="monitor"))
    if state.ingestion:
        state.tasks.append(asyncio.create_task(state.ingestion.run_all(), name="ingestion"))
    if notifier.enabled and s.telegram_mode in ("polling", "webhook"):
        state.bot = TelegramBot(s.telegram_bot_token, s.allowed_chats, state.engine, status_snapshot)
        if s.telegram_mode == "polling":
            state.tasks.append(asyncio.create_task(state.bot.run_polling(), name="telegram-bot"))
        elif s.telegram_webhook_url:
            try:
                await state.bot.set_webhook(s.telegram_webhook_url, s.telegram_webhook_secret)
            except Exception as exc:  # noqa: BLE001
                log.warning("setWebhook gagal: %s", exc)
    if not notifier.enabled:
        log.warning("Telegram belum dikonfigurasi: alert hanya dicatat")

    yield

    state.engine.running = False
    state.breaker.stop()
    if state.monitor:
        state.monitor.stop()
    if state.bot:
        state.bot.stop()
    if state.ingestion:
        state.ingestion.stop()
    for t in state.tasks:
        t.cancel()
    await asyncio.gather(*state.tasks, return_exceptions=True)
    state.ledger.close()
    await redis.aclose()


app = FastAPI(
    title="Crypto Volatility Predictor",
    version="3.0.0",
    description="Prediksi lonjakan volatilitas BTCUSDT 15 menit ke depan, dengan forecast ledger publik.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


# ------------------------------------------------------------------ halaman publik
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/ledger", response_class=HTMLResponse, include_in_schema=False)
async def explorer():
    return FileResponse(STATIC / "explorer.html")


@app.get("/ledger/row/{seq}", response_class=HTMLResponse, include_in_schema=False)
async def explorer_row(seq: int):
    return FileResponse(STATIC / "explorer.html")


@app.get("/incidents", include_in_schema=False)
async def incidents_index():
    idx = REPORTS / "incidents" / "index.json"
    return JSONResponse(json.loads(idx.read_text()) if idx.exists() else {"incidents": []})


@app.get("/incidents/{name}", include_in_schema=False)
async def incident_file(name: str):
    p = (REPORTS / "incidents" / name).resolve()
    if not p.is_file() or (REPORTS / "incidents").resolve() not in p.parents:
        raise HTTPException(404, "tidak ada")
    return FileResponse(p)


# ------------------------------------------------------------------ ledger API (read-only)
@app.get("/ledger/head")
async def ledger_head():
    return {"head": state.ledger.head(), "rows": state.ledger.count()}


@app.get("/ledger/verify")
async def ledger_verify():
    ok, bad = await asyncio.to_thread(state.ledger.verify)
    return {"ok": ok, "first_bad_seq": bad, "head": state.ledger.head(), "rows": state.ledger.count()}


@app.get("/ledger/rows")
async def ledger_rows(after_seq: int = Query(0, ge=0), limit: int = Query(5000, ge=1, le=20000)):
    """Baris mentah berurutan untuk verifikasi hash chain independen (browser / CI / skrip)."""
    rows = state.ledger.export(after_seq, limit)
    return {"rows": rows, "next_after_seq": rows[-1]["seq"] if rows else None, "head": state.ledger.head()}


@app.get("/ledger/recent")
async def ledger_recent(limit: int = Query(100, ge=1, le=2000), model: str | None = None):
    rows = state.ledger.recent(limit * (3 if model else 1))
    if model:
        rows = [r for r in rows if r["model_version"] == model][:limit]
    return {"rows": _clean(rows)}


@app.get("/ledger/rows/{seq}")
async def ledger_row(seq: int):
    row = state.ledger.get(seq)
    if row is None:
        raise HTTPException(404, "baris ledger tidak ada")
    siblings = state.ledger.query("SELECT seq, model_version, prob FROM predictions WHERE symbol = ? AND decision_ts = ?",
                                  (row["symbol"], row["decision_ts"]))
    return _clean({"row": row, "same_decision": [dict(r) for r in siblings]})


@app.get("/performance")
async def performance(days: float = Query(7, gt=0, le=90)):
    m = state.monitor
    metrics = await asyncio.to_thread(m.metrics, days) if m else {}
    c = state.engine.conformal
    return _clean({"window_days": days, "metrics": metrics, "last_check": m.last if m else None,
                   "threshold": state.engine.threshold,
                   "conformal": None if c is None else {"tau": c.tau, "alpha": c.alpha, "gamma": c.gamma, "mode": c.mode,
                                                        "updates": c.n_updates, "realized_rate": c.realized_rate}})


@app.get("/badges/{name}.json", include_in_schema=False)
async def badge(name: str):
    p = Path(get_settings().data_dir) / "badges" / f"{name}.json"
    if not p.exists() or "/" in name:
        raise HTTPException(404, "badge belum ada")
    return JSONResponse(json.loads(p.read_text()))


# ------------------------------------------------------------------ status
@app.get("/health")
async def health():
    s = get_settings()
    eng = state.engine
    meta = eng.model.metadata if eng.model else {}
    checks: dict = {"model": {
        "ok": eng.model is not None, "version": meta.get("version"), "data_source": meta.get("data_source"),
        "backend": eng.model.backend if eng.model else None, "load_error": state.model_error,
        "last_inference_error": eng.last_model_error, "fallback_predictions": eng.fallback_count,
        "calibrated": eng.calibrator is not None, "challenger": eng.challenger is not None,
        "hazard": eng.hazard is not None}}
    try:
        await eng.redis.ping()
        checks["redis"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = {"ok": False, "error": str(exc)}
    try:
        eng.db.ping()
        checks["database"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        checks["database"] = {"ok": False, "error": str(exc)}
    br = state.breaker.snapshot()
    checks["circuit_breaker"] = {"ok": br["state"] == "OK", **br}
    checks["engine"] = {"ok": (not s.run_engine) or any(t.get_name() == "engine" and not t.done() for t in state.tasks),
                        "processed": eng.processed, "last_error": eng.last_error}
    checks["ledger"] = {"ok": True, "rows": state.ledger.count(), "head": state.ledger.head(),
                        "last_verify_ok": (state.monitor.last or {}).get("ledger_ok") if state.monitor else None}
    checks["baseline"] = {"ok": True, **(eng.baselines.status() if eng.baselines else {})}
    checks["telegram"] = {"ok": eng.notifier.enabled, "bot_mode": s.telegram_mode if state.bot else "off"}
    status = "ok" if all(checks[k]["ok"] for k in ("redis", "database", "engine", "model")) else "degraded"
    return _clean({"status": status, "breaker_state": br["state"], "uptime_sec": round(time.time() - state.started_at),
                   "checks": checks})


def status_snapshot() -> dict:
    """Ringkasan status (dipakai /status dan bot Telegram)."""
    s = get_settings()
    eng = state.engine
    lat = eng.metrics.summary()["latency"]
    meta = eng.model.metadata if eng.model else {}
    br = state.breaker.snapshot()
    summary = eng.metrics.summary()
    return {
        "status": "ok" if br["state"] == "OK" else br["state"].lower(),
        "circuit_breaker": br,
        "model_version": meta.get("version"),
        "threshold": eng.threshold,
        "hmm_gate": s.hmm_gate if eng.hmm else "off",
        "latency": {"processing_p95_ms": lat.get("processing", {}).get("p95_ms"),
                    "end_to_end_p95_ms": lat.get("end_to_end", {}).get("p95_ms")},
        "latest": {k: {"probability": v.probability, "status": v.status, "mode": v.mode, "direction": v.direction,
                       "regime": v.regime} for k, v in eng.last_results.items()},
        "ledger_rows": state.ledger.count(),
        "uptime_sec": summary["uptime_sec"],
        "rss_mb": round(summary["memory"]["rss_mb"], 1) if summary["memory"]["rss_mb"] else None,
    }


@app.get("/status")
async def status():
    return _clean(status_snapshot())


@app.get("/metrics")
async def metrics(format: str = Query("json", pattern="^(json|prometheus)$")):
    """Latency p50/p95/p99 per tahap + memory + status breaker."""
    eng = state.engine
    br = state.breaker.snapshot()
    if format == "prometheus":
        extra = (f"\n# TYPE cvp_breaker_state gauge\ncvp_breaker_state {STATE_CODE[br['state']]}\n"
                 + "".join(f'cvp_breaker_decisions_total{{state="{k}"}} {v}\n' for k, v in br["decisions"].items())
                 + f"cvp_alert_threshold {eng.threshold}\n")
        return PlainTextResponse(eng.metrics.prometheus() + extra)
    out = eng.metrics.summary()
    out["circuit_breaker"] = br
    out["threshold"] = eng.threshold
    return _clean(out)


# ------------------------------------------------------------------ admin
class ThresholdIn(BaseModel):
    threshold: float = Field(..., gt=0, lt=1)


def _require_admin(token: str | None) -> None:
    expected = get_settings().admin_api_token
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="butuh header X-Admin-Token (ADMIN_API_TOKEN)")


@app.get("/config/threshold")
async def get_threshold():
    c = state.engine.conformal
    return {"threshold": state.engine.threshold, "static_threshold": state.engine.static_threshold,
            "conformal": None if c is None else {"alpha": c.alpha, "gamma": c.gamma, "mode": c.mode}}


@app.put("/config/threshold")
async def put_threshold(body: ThresholdIn, x_admin_token: str | None = Header(None)):
    _require_admin(x_admin_token)
    await state.engine.set_threshold(body.threshold)
    return {"threshold": state.engine.threshold}


@app.post("/telegram/webhook", include_in_schema=False)
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str | None = Header(None)):
    s = get_settings()
    if s.telegram_webhook_secret and x_telegram_bot_api_secret_token != s.telegram_webhook_secret:
        raise HTTPException(status_code=403, detail="secret token salah")
    if state.bot is None:
        raise HTTPException(status_code=503, detail="bot Telegram tidak aktif")
    reply = await state.bot.handle_update(await request.json())
    return {"ok": True, "replied": reply is not None}


# ------------------------------------------------------------------ prediksi manual & data
class CandleIn(BaseModel):
    open_time: int = Field(..., description="Waktu buka candle (epoch ms, UTC)")
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_volume: float | None = None


class PredictRequest(BaseModel):
    symbol: str = Field("BTCUSDT", examples=["BTCUSDT"])
    candles: list[CandleIn] | None = Field(None, description=f"Opsional, minimal {MIN_CANDLES} candle 1m kontinu. "
                                                             "Kosong = pakai cache Redis.")


@app.post("/predict")
async def predict(req: PredictRequest):
    """Prediksi manual. Tidak dicatat ke ledger dan tidak mengirim alert."""
    candles = None
    if req.candles:
        rows = [{**c.model_dump(), "symbol": req.symbol.upper(), "close_time": c.open_time + 59_999}
                for c in req.candles]
        for r in rows:
            if r["taker_buy_volume"] is None:
                r["taker_buy_volume"] = r["volume"] / 2
        candles = candles_to_frame(rows)
    try:
        res = await state.engine.evaluate(req.symbol, candles=candles, source="manual", log_to_db=False,
                                          allow_alert=False)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out = {k: getattr(res, k) for k in ("symbol", "candle_time", "close", "probability", "threshold", "is_spike",
                                         "status", "mode", "direction", "regime", "regime_p_high",
                                         "challenger_probability", "hazard", "features")}
    return _clean(out)


@app.get("/candles/{symbol}")
async def candles(symbol: str, limit: int = Query(200, ge=1, le=1000)):
    df = await get_candles(state.engine.redis, symbol, limit=limit)
    return {"symbol": symbol.upper(), "candles": df.to_dict(orient="records")}


@app.get("/alerts")
async def get_alerts(symbol: str | None = None, limit: int = Query(100, ge=1, le=1000)):
    rows = state.engine.db.recent_alerts(symbol.upper() if symbol else None, limit)
    for r in rows:
        ts = r.get("created_at")
        if ts is not None:
            r["created_at"] = (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).isoformat()
    return {"alerts": rows}


@app.get("/features/live")
async def live_features(hours: float = Query(168, gt=0, le=24 * 30), limit: int = Query(20000, le=100000)):
    """Fitur live (untuk PSI drift opsional di CI mingguan)."""
    from datetime import datetime, timedelta

    rows = state.engine.db.feature_rows_since(datetime.now(timezone.utc) - timedelta(hours=hours), limit)
    return {"rows": [r.get("features") or {} for r in rows]}


@app.get("/model")
async def model_info():
    eng = state.engine
    out: dict = {"loaded": eng.model is not None, "metadata": eng.model.metadata if eng.model else None}
    report = REPORTS / "evaluation.json"
    if report.exists():
        ev = json.loads(report.read_text())
        out["evaluation"] = {k: ev.get(k) for k in ("cv", "baselines", "calibration", "hazard_vs_binary", "ablation")}
        out["evaluation"]["holdout_ranking"] = ev.get("holdout", {}).get("ranking")
    return _clean(out)


@app.get("/micro/{symbol}")
async def micro(symbol: str):
    """Mikrostruktur live menit terakhir (trade flow, OBI order book lokal, futures, Hawkes, lead-lag)."""
    res = state.engine.last_results.get(symbol.upper())
    return _clean({"symbol": symbol.upper(), "candle_time": res.candle_time if res else None,
                   "micro": res.micro if res else None})
