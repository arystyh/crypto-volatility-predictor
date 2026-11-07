"""Forecast ledger: every prediction is recorded before its outcome exists.

Guarantees
----------
1. Append-only. SQLite triggers reject UPDATE and DELETE on both tables, so a
   bad prediction cannot be quietly edited or removed.
2. Tamper-evident. Each prediction row stores ``row_hash = sha256(prev_hash ||
   canonical_json(row))``. Editing any historical row (for example after
   dropping the triggers) breaks every hash after it, and ``verify()`` reports
   the first broken row. Publish ``head()`` somewhere public on a schedule
   (the weekly CI commit is enough) and the history before that point is
   anchored by git.
3. No late logging. ``log_prediction`` refuses a row whose wall-clock
   ``logged_at`` is more than ``max_log_lag_ms`` after its decision time, so
   a prediction cannot be written after the outcome is visible.
4. Frozen baseline. The Z-score baseline (mu, sd) in force at decision time is
   stored with the prediction and used for scoring, so a later change to the
   labelling window cannot rewrite old results.

Champion and challenger both log here under different ``model_version`` values,
which gives shadow-mode evaluation on identical decision points for free.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    decision_ts   INTEGER NOT NULL,   -- ms; close time of the last candle used = open time of the first forward candle
    horizon_min   INTEGER NOT NULL,
    model_version TEXT    NOT NULL,
    prob          REAL    NOT NULL CHECK (prob >= 0 AND prob <= 1),
    threshold     REAL    NOT NULL,
    alerted       INTEGER NOT NULL CHECK (alerted IN (0, 1)),
    baseline_mu   REAL    NOT NULL,
    baseline_sd   REAL    NOT NULL CHECK (baseline_sd > 0),
    z_threshold   REAL    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'OK' CHECK (status IN ('OK', 'DEGRADED', 'STALE')),
    conformal_alpha REAL,             -- target false alarm rate ACI saat keputusan (NULL = threshold statis)
    conformal_gamma REAL,             -- laju adaptasi ACI saat keputusan
    logged_at     INTEGER NOT NULL,
    prev_hash     TEXT    NOT NULL,
    row_hash      TEXT    NOT NULL UNIQUE,
    UNIQUE (symbol, decision_ts, model_version)
);
CREATE TABLE IF NOT EXISTS outcomes (
    seq         INTEGER PRIMARY KEY REFERENCES predictions(seq),
    fwd_logvol  REAL    NOT NULL,
    z           REAL    NOT NULL,
    label       INTEGER NOT NULL CHECK (label IN (0, 1)),
    scored_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(model_version, decision_ts);
CREATE TRIGGER IF NOT EXISTS predictions_no_update BEFORE UPDATE ON predictions
    BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS predictions_no_delete BEFORE DELETE ON predictions
    BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS outcomes_no_update BEFORE UPDATE ON outcomes
    BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS outcomes_no_delete BEFORE DELETE ON outcomes
    BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
"""

HASHED_FIELDS = (
    "symbol", "decision_ts", "horizon_min", "model_version", "prob", "threshold",
    "alerted", "baseline_mu", "baseline_sd", "z_threshold", "logged_at",
    "status", "conformal_alpha", "conformal_gamma",
)


class LateLogError(ValueError):
    """The prediction was logged too long after its decision time."""


def _canonical(row: dict) -> str:
    # repr() of a float round-trips exactly, so the hash is reproducible.
    payload = {k: (repr(row.get(k)) if isinstance(row.get(k), float) else row.get(k)) for k in HASHED_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def row_hash(prev_hash: str, row: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(row)).encode()).hexdigest()


@dataclass(frozen=True)
class Prediction:
    symbol: str
    decision_ts: int
    horizon_min: int
    model_version: str
    prob: float
    threshold: float
    baseline_mu: float
    baseline_sd: float
    z_threshold: float
    status: str = "OK"                    # OK | DEGRADED (fallback) | STALE (data basi: alert DITAHAN)
    conformal_alpha: float | None = None
    conformal_gamma: float | None = None

    @property
    def alerted(self) -> int:
        """Sinyal model (prob >= threshold). Saat status STALE sinyal ini TIDAK dikirim ke Telegram."""
        return int(self.prob >= self.threshold)


class Ledger:
    def __init__(self, path: str = "ledger.db", max_log_lag_ms: int = 60_000, clock=None):
        # check_same_thread=False + lock: dipakai dari event loop dan asyncio.to_thread
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)  # explicit transactions
        self.lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.max_log_lag_ms = max_log_lag_ms
        self._clock = clock or (lambda: int(time.time() * 1000))

    def close(self) -> None:
        self.conn.close()

    # -- writing -----------------------------------------------------------------
    def log_prediction(self, p: Prediction) -> int:
        logged_at = self._clock()
        if logged_at - p.decision_ts > self.max_log_lag_ms:
            raise LateLogError(
                f"logged {logged_at - p.decision_ts} ms after decision time "
                f"(limit {self.max_log_lag_ms} ms)"
            )
        if not (math.isfinite(p.baseline_mu) and p.baseline_sd > 0):
            raise ValueError("baseline not ready; do not log predictions during warm-up")

        row = {
            "symbol": p.symbol, "decision_ts": int(p.decision_ts),
            "horizon_min": int(p.horizon_min), "model_version": p.model_version,
            "prob": float(p.prob), "threshold": float(p.threshold),
            "alerted": p.alerted, "baseline_mu": float(p.baseline_mu),
            "baseline_sd": float(p.baseline_sd), "z_threshold": float(p.z_threshold),
            "logged_at": int(logged_at), "status": p.status,
            "conformal_alpha": None if p.conformal_alpha is None else float(p.conformal_alpha),
            "conformal_gamma": None if p.conformal_gamma is None else float(p.conformal_gamma),
        }
        with self.lock:
            return self._insert(row)

    def _insert(self, row: dict) -> int:
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")  # serialise writers so the chain stays linear
        try:
            prev = self._head()
            row["prev_hash"] = prev
            row["row_hash"] = row_hash(prev, row)
            cols = ", ".join(row)
            marks = ", ".join("?" for _ in row)
            cur.execute(f"INSERT INTO predictions ({cols}) VALUES ({marks})", tuple(row.values()))
            seq = cur.lastrowid
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        return seq

    def record_outcome(self, seq: int, fwd_logvol: float, z: float, label: int) -> None:
        with self.lock:
            self._record_outcome(seq, fwd_logvol, z, label)

    def _record_outcome(self, seq: int, fwd_logvol: float, z: float, label: int) -> None:
        self.conn.execute(
            "INSERT INTO outcomes (seq, fwd_logvol, z, label, scored_at) VALUES (?, ?, ?, ?, ?)",
            (seq, float(fwd_logvol), float(z), int(label), self._clock()),
        )

    # -- reading -----------------------------------------------------------------
    def head(self) -> str:
        with self.lock:
            return self._head()

    def _head(self) -> str:
        r = self.conn.execute("SELECT row_hash FROM predictions ORDER BY seq DESC LIMIT 1").fetchone()
        return r["row_hash"] if r else GENESIS

    def verify(self) -> tuple[bool, int | None]:
        """Recompute the hash chain. Returns (ok, first_bad_seq)."""
        with self.lock:
            rows = [dict(r) for r in self.conn.execute("SELECT * FROM predictions ORDER BY seq").fetchall()]
        return verify_rows(rows)


    def unscored(self, symbol: str | None = None) -> list[sqlite3.Row]:
        q = ("SELECT p.* FROM predictions p LEFT JOIN outcomes o ON o.seq = p.seq "
             "WHERE o.seq IS NULL")
        args: tuple = ()
        if symbol is not None:
            q += " AND p.symbol = ?"
            args = (symbol,)
        with self.lock:
            return self.conn.execute(q + " ORDER BY p.decision_ts", args).fetchall()

    def scored(self, model_version: str, since_ts: int = 0, symbol: str | None = None) -> list[sqlite3.Row]:
        q = ("SELECT p.*, o.fwd_logvol, o.z, o.label FROM predictions p "
             "JOIN outcomes o ON o.seq = p.seq "
             "WHERE p.model_version = ? AND p.decision_ts >= ?")
        args: list = [model_version, since_ts]
        if symbol is not None:
            q += " AND p.symbol = ?"
            args.append(symbol)
        with self.lock:
            return self.conn.execute(q + " ORDER BY p.decision_ts", args).fetchall()

    def recent(self, limit: int = 100, symbol: str | None = None) -> list[dict]:
        """Baris ledger terbaru beserta outcome (bila sudah diskor), untuk halaman ledger."""
        q = ("SELECT p.*, o.fwd_logvol, o.z, o.label, o.scored_at FROM predictions p "
             "LEFT JOIN outcomes o ON o.seq = p.seq")
        args: tuple = ()
        if symbol is not None:
            q += " WHERE p.symbol = ?"
            args = (symbol,)
        with self.lock:
            return [dict(r) for r in self.conn.execute(q + " ORDER BY p.seq DESC LIMIT ?", (*args, limit))]

    def get(self, seq: int) -> dict | None:
        with self.lock:
            r = self.conn.execute(
                "SELECT p.*, o.fwd_logvol, o.z, o.label, o.scored_at FROM predictions p "
                "LEFT JOIN outcomes o ON o.seq = p.seq WHERE p.seq = ?", (seq,)).fetchone()
        return dict(r) if r else None

    def export(self, after_seq: int = 0, limit: int = 10_000) -> list[dict]:
        """Baris mentah (urut seq) untuk verifikasi hash chain secara independen."""
        with self.lock:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM predictions WHERE seq > ? ORDER BY seq LIMIT ?", (after_seq, limit))]

    def count(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()


def verify_rows(rows, prev: str = GENESIS) -> tuple[bool, int | None]:
    """Verifikasi rantai hash dari daftar baris (dict) berurutan seq. Dipakai juga oleh CI & explorer."""
    for row in rows:
        if row["prev_hash"] != prev or row_hash(prev, row) != row["row_hash"]:
            return False, row["seq"]
        prev = row["row_hash"]
    return True, None
