"""Logging prediksi & notifikasi. SQLite secara default; cukup ganti DATABASE_URL
ke postgresql+psycopg://... untuk Supabase Postgres."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    BigInteger,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("symbol", String(20), nullable=False, index=True),
    Column("candle_time", BigInteger, nullable=False),  # open_time candle (ms)
    Column("close", Float),
    Column("probability", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("alerted", Boolean, nullable=False, default=False),
    Column("volatility", Float),  # rv_15 (realized vol 15m)
    Column("atr_pct", Float),     # atr_14_pct
    Column("source", String(10), nullable=False, default="stream"),
    Column("mode", String(10), nullable=False, default="model"),  # model | fallback | gated
    Column("direction", String(10)),  # downside | upside | two_way (semi-variance)
    Column("regime_p_high", Float),   # P(rezim High) dari HMM
    Column("challenger_probability", Float),  # shadow LightGBM
    Column("status", String(10)),             # OK | DEGRADED | STALE (circuit breaker)
    Column("ledger_seq", Integer),            # baris forecast ledger (bila tercatat)
    Column("features", JSON),
)

alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("symbol", String(20), nullable=False, index=True),
    Column("probability", Float, nullable=False),
    Column("close", Float),
    Column("message", Text),
    Column("sent", Boolean, nullable=False, default=False),
    Column("error", Text),
    Column("mode", String(10), nullable=False, default="model"),
    Column("direction", String(10)),
)

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    def __init__(self, url: str):
        if url.startswith("sqlite:///"):
            path = url.replace("sqlite:///", "", 1)
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        metadata.create_all(self.engine)

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(select(1))
        return True

    # ---------- writes ----------
    def log_prediction(
        self,
        symbol: str,
        candle_time: int,
        close: float,
        probability: float,
        threshold: float,
        alerted: bool,
        features: dict,
        source: str = "stream",
        mode: str = "model",
        direction: str | None = None,
        regime_p_high: float | None = None,
        challenger_probability: float | None = None,
        status: str | None = None,
        ledger_seq: int | None = None,
    ) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(
                predictions.insert().values(
                    created_at=utcnow(),
                    symbol=symbol,
                    candle_time=int(candle_time),
                    close=close,
                    probability=probability,
                    threshold=threshold,
                    alerted=alerted,
                    volatility=features.get("rv_15"),
                    atr_pct=features.get("atr_14_pct"),
                    source=source,
                    mode=mode,
                    direction=direction,
                    regime_p_high=regime_p_high,
                    challenger_probability=challenger_probability,
                    status=status,
                    ledger_seq=ledger_seq,
                    features=json.loads(json.dumps(features, default=float)),
                )
            )
            return int(res.inserted_primary_key[0])

    def log_alert(
        self, symbol: str, probability: float, close: float, message: str, sent: bool, error: str | None,
        mode: str = "model", direction: str | None = None,
    ) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(
                alerts.insert().values(
                    created_at=utcnow(),
                    symbol=symbol,
                    probability=probability,
                    close=close,
                    message=message,
                    sent=sent,
                    error=error,
                    mode=mode,
                    direction=direction,
                )
            )
            return int(res.inserted_primary_key[0])

    # ---------- reads ----------
    def last_alert_time(self, symbol: str) -> datetime | None:
        with self.engine.connect() as conn:
            value = conn.execute(
                select(func.max(alerts.c.created_at)).where(alerts.c.symbol == symbol)
            ).scalar()
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value

    def recent_predictions(self, symbol: str | None = None, limit: int = 200, hours: float | None = None) -> list[dict]:
        q = select(predictions).order_by(predictions.c.id.desc()).limit(limit)
        if symbol:
            q = q.where(predictions.c.symbol == symbol)
        if hours:
            q = q.where(predictions.c.created_at >= utcnow() - timedelta(hours=hours))
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(q)]

    def recent_alerts(self, symbol: str | None = None, limit: int = 100) -> list[dict]:
        q = select(alerts).order_by(alerts.c.id.desc()).limit(limit)
        if symbol:
            q = q.where(alerts.c.symbol == symbol)
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(q)]

    def feature_rows_since(self, since: datetime, limit: int = 50_000) -> list[dict]:
        q = (select(predictions.c.features).where(predictions.c.created_at >= since)
             .where(predictions.c.source == "stream").order_by(predictions.c.id.desc()).limit(limit))
        with self.engine.connect() as conn:
            return [r[0] for r in conn.execute(q) if r[0]]

    def count_since(self, table: Table, since: datetime, symbol: str | None = None) -> int:
        q = select(func.count()).select_from(table).where(table.c.created_at >= since)
        if symbol:
            q = q.where(table.c.symbol == symbol)
        with self.engine.connect() as conn:
            return int(conn.execute(q).scalar() or 0)
