# Target definition and forecast ledger

## What the model predicts

> **"In the next 15 minutes, will BTCUSDT's Parkinson volatility be more than
> 2 standard deviations above its own 30-day history (in log terms)?"**

The full definition, with formulas, is in `app/ml/target.py`. Key properties:

| Property | How it is guaranteed |
|---|---|
| Baseline (mu, sd) uses only the past | `tests/test_target.py::test_baseline_is_causal` changes every candle after `t` and checks the baseline at `t` is bit-identical |
| Label looks exactly H candles ahead, no further | `test_label_only_sees_next_h_candles` |
| Training and serving compute the same thing | `test_online_matches_offline` compares the pandas path with the streaming `OnlineBaseline` row by row |
| Gaps never produce silent wrong labels | both paths raise on non-contiguous candles; the gap-filler must run first |
| CV embargo is derived, not guessed | `TargetSpec.min_embargo(lookback) = H + lookback` |

`OnlineBaseline.to_state()` / `from_state()` serialise the 30-day state so a
restarted container resumes without a 7-day warm-up.

## Forecast ledger

Every decision writes one row per model (`champion_xgboost`, `challenger_lgbm`,
`hazard_xgb` with prob = P(onset ≤ 15m), or `fallback_parkinson` when DEGRADED) to
`ledger.db` **at decision time**, before its outcome exists, and **before** any alert is sent:

```python
from app.services.ledger import Ledger, Prediction

ledger = Ledger("ledger.db")
ledger.log_prediction(Prediction(
    symbol="BTCUSDT",
    decision_ts=candle_close_ms,          # = open time of the first forward candle
    horizon_min=spec.horizon,
    model_version="champion_xgboost",
    prob=p, threshold=alert_threshold,
    baseline_mu=baseline.mu, baseline_sd=baseline.sd,  # frozen for scoring
    z_threshold=spec.z_threshold,
))
```

Guarantees: append-only (SQLite triggers), SHA-256 hash chain
(`ledger.verify()` names the first tampered row), late writes rejected
(`max_log_lag_ms`), and scoring uses the baseline stored in the row.

Hashed fields: `symbol, decision_ts, horizon_min, model_version, prob, threshold,
alerted, baseline_mu, baseline_sd, z_threshold, logged_at, status, conformal_alpha,
conformal_gamma`. `row_hash = sha256(prev_hash + json.dumps(fields, sort_keys=True,
separators=(",", ":")))` where every float is first replaced by its Python `repr()`
string, so the hash is reproducible in any language.

* `status` is the circuit breaker state at decision time. `alerted` is the model's
  signal (prob ≥ threshold); with `status = STALE` the alert was **held**, not sent.
* `threshold` is the threshold actually in force (static PR-curve threshold, or the
  adaptive conformal threshold), and `conformal_alpha` / `conformal_gamma` record
  the ACI target and step size used (NULL = static threshold).

## Verifying it yourself

Three independent paths, all recomputing the chain from raw rows:

1. **Browser**: open `/` (ledger explorer) and press *Verifikasi di browser*. The page
   pulls `/ledger/rows` and hashes with SubtleCrypto (pure-JS fallback on plain HTTP).
   `tests/test_explorer.py` runs the same JS in Node and checks it against Python.
2. **Terminal**: `python -m scripts.verify_ledger --url https://<host> --anchor badges/ledger_head.txt`.
3. **Weekly CI** (`.github/workflows/drift_check.yml`): runs step 2, fails red if the
   chain is broken *or* if last week's committed head is no longer in the chain (history
   rewritten), then commits the new head and badges.

## Scoring and the live badge

In the service, `PerformanceMonitor` scores matured rows every 5 minutes from the
Redis candle cache. Offline / one-off:

```bash
python -m scripts.score_ledger --db ledger.db --symbol BTCUSDT
```

This verifies the chain (exit 1 if tampered), scores matured predictions
against Binance candles, and writes to `badges/`:

* `<model>.json`: a shields.io endpoint badge, used in the README as
  `![live](https://img.shields.io/endpoint?url=<raw-url>/badges/champion_xgboost.json)`
* `<model>_metrics.json`: precision, recall, PR-AUC, **PR-AUC lift over the base
  rate**, Brier score, spike episodes caught, and **median lead time**
* `ledger_head.txt`: the current chain head. Commit it on a schedule; git
  history then timestamps the ledger, so past predictions cannot be rewritten
  without the change showing up.

### Lead time

Lead time answers "did the alert help?". For each spike episode, the moment
a naive monitor watching only current 15-minute volatility would see the
spike is `first labelled decision + H`. Lead = that moment minus the first
alert. Positive means the system warned earlier than simply watching the
chart; this is the number to put in the README, next to PR-AUC lift.
