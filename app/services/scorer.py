"""Score matured predictions and compute live metrics.

A prediction made at ``decision_ts`` with horizon H matures once the H candles
with open times in [decision_ts, decision_ts + H min) have all closed. The
scorer computes their log Parkinson volatility, turns it into a Z-score with
the baseline frozen in the ledger row, and appends the outcome.

Run it on a schedule (every few minutes, or from the weekly CI job); it is
idempotent because outcomes are keyed by prediction ``seq``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from app.ml.target import CANDLE_MS, forward_logvol
from app.services.ledger import Ledger


@dataclass
class ScoreReport:
    scored: int = 0
    not_mature: int = 0
    missing_candles: int = 0


def score_pending(ledger: Ledger, candles: pd.DataFrame, symbol: str) -> ScoreReport:
    """``candles`` needs open_time (ms), high, low for ``symbol``, closed bars only."""
    report = ScoreReport()
    c = candles.set_index("open_time").sort_index()
    if c.empty:
        return report
    last_closed_end = int(c.index[-1]) + CANDLE_MS

    for p in ledger.unscored(symbol):
        start = p["decision_ts"]
        end = start + p["horizon_min"] * CANDLE_MS
        if end > last_closed_end:
            report.not_mature += 1
            continue
        win = c.loc[start:end - CANDLE_MS]
        if len(win) != p["horizon_min"]:
            # A gap in the forward window: leave unscored until the gap-filler repairs it.
            report.missing_candles += 1
            continue
        fwd = forward_logvol(win["high"].to_numpy(), win["low"].to_numpy())
        z = (fwd - p["baseline_mu"]) / p["baseline_sd"]
        ledger.record_outcome(p["seq"], fwd, z, int(z > p["z_threshold"]))
        report.scored += 1
    return report


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, as scikit-learn computes it)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    # Collapse tied scores so every threshold is evaluated once.
    last_of_tie = np.r_[np.diff(p) != 0, True]
    tp = np.cumsum(y)[last_of_tie]
    fp = np.cumsum(1 - y)[last_of_tie]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))


@dataclass
class LiveMetrics:
    model_version: str
    n: int
    positives: int
    base_rate: float
    alerts: int
    precision: float
    recall: float
    pr_auc: float
    pr_auc_lift: float          # PR-AUC / base rate; 1.0 = no better than random
    brier: float
    episodes: int               # distinct volatility spikes in the window
    episodes_caught: int        # spikes with at least one alert before or during them
    median_lead_min: float      # minutes the first alert beat a naive current-vol monitor


def spike_episodes(ts: np.ndarray, label: np.ndarray, gap_ms: int = CANDLE_MS):
    """Group consecutive positive decision times into episodes [(start, end), ...]."""
    eps = []
    start = prev = None
    for t, l in zip(ts, label):
        if l == 1:
            if start is None or t - prev > gap_ms:
                if start is not None:
                    eps.append((start, prev))
                start = t
            prev = t
    if start is not None:
        eps.append((start, prev))
    return eps


def live_metrics(ledger: Ledger, model_version: str, since_ts: int = 0,
                 symbol: str | None = None, lookback_min: int = 30) -> LiveMetrics:
    rows = ledger.scored(model_version, since_ts, symbol)
    if not rows:
        return LiveMetrics(model_version, 0, 0, math.nan, 0, math.nan, math.nan,
                           math.nan, math.nan, math.nan, 0, 0, math.nan)
    ts = np.array([r["decision_ts"] for r in rows])
    y = np.array([r["label"] for r in rows])
    prob = np.array([r["prob"] for r in rows])
    alert = np.array([r["alerted"] for r in rows])

    tp = int(np.sum((alert == 1) & (y == 1)))
    base = float(y.mean())
    ap = average_precision(y, prob)

    # Lead time is measured against a naive monitor that only watches *current*
    # volatility. The first decision labelled 1 at time s has a forward window
    # ending at s + H; that is the earliest moment the trailing H-minute
    # volatility itself crosses the Z threshold. Lead = that moment minus the
    # first alert, i.e. how many minutes earlier than "just watching the chart"
    # the system warned.
    horizon_ms = rows[0]["horizon_min"] * CANDLE_MS
    leads, caught = [], 0
    eps = spike_episodes(ts, y)
    for start, end in eps:
        visible_at = start + horizon_ms
        mask = (alert == 1) & (ts >= start - lookback_min * CANDLE_MS) & (ts <= end)
        if mask.any():
            caught += 1
            leads.append((visible_at - ts[mask].min()) / CANDLE_MS)

    return LiveMetrics(
        model_version=model_version,
        n=len(rows),
        positives=int(y.sum()),
        base_rate=base,
        alerts=int(alert.sum()),
        precision=tp / alert.sum() if alert.sum() else math.nan,
        recall=tp / y.sum() if y.sum() else math.nan,
        pr_auc=ap,
        pr_auc_lift=ap / base if base > 0 else math.nan,
        brier=float(np.mean((prob - y) ** 2)),
        episodes=len(eps),
        episodes_caught=caught,
        median_lead_min=float(np.median(leads)) if leads else math.nan,
    )


def badge_json(m: LiveMetrics, window_label: str = "7d") -> dict:
    """shields.io endpoint badge: https://img.shields.io/endpoint?url=<raw url of this json>"""
    if m.n == 0 or math.isnan(m.pr_auc):
        return {"schemaVersion": 1, "label": f"live PR-AUC ({window_label})",
                "message": "no data", "color": "lightgrey"}
    color = "brightgreen" if m.pr_auc_lift >= 3 else "yellow" if m.pr_auc_lift >= 1.5 else "red"
    return {
        "schemaVersion": 1,
        "label": f"live PR-AUC ({window_label})",
        "message": f"{m.pr_auc:.2f} ({m.pr_auc_lift:.1f}x base) n={m.n}",
        "color": color,
    }


def metrics_to_json(m: LiveMetrics) -> str:
    d = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in asdict(m).items()}
    return json.dumps(d, indent=2)
