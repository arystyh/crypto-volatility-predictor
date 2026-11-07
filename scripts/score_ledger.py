"""Score matured predictions, verify the hash chain, and write the live badge.

Examples
--------
    # live: pull recent closed candles from Binance REST
    python -m scripts.score_ledger --db ledger.db --symbol BTCUSDT

    # offline / CI: use a CSV with open_time,high,low columns
    python -m scripts.score_ledger --db ledger.db --symbol BTCUSDT --candles candles.csv

Exit code is 1 if the ledger fails verification, so CI goes red on tampering.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from app.services.ledger import Ledger
from app.services.scorer import badge_json, live_metrics, metrics_to_json, score_pending

BINANCE_KLINES = "https://data-api.binance.vision/api/v3/klines"  # market-data only, tidak diblokir dari AS


def fetch_closed_klines(symbol: str, minutes: int = 1000) -> pd.DataFrame:
    """Most recent closed 1m candles (max 1000 per Binance request)."""
    qs = urllib.parse.urlencode({"symbol": symbol, "interval": "1m", "limit": min(minutes, 1000)})
    with urllib.request.urlopen(f"{BINANCE_KLINES}?{qs}", timeout=10) as r:
        raw = json.load(r)
    df = pd.DataFrame(raw).iloc[:, [0, 2, 3, 6]]
    df.columns = ["open_time", "high", "low", "close_time"]
    df = df.astype({"open_time": "int64", "high": "float64", "low": "float64", "close_time": "int64"})
    now_ms = int(time.time() * 1000)
    return df[df["close_time"] < now_ms][["open_time", "high", "low"]]  # drop the still-open bar


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="ledger.db")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--candles", help="CSV with open_time,high,low (skips the Binance call)")
    ap.add_argument("--models", nargs="+", default=["champion_xgboost", "challenger_lgbm", "hazard_xgb"])
    ap.add_argument("--window-days", type=float, default=7.0)
    ap.add_argument("--out", default="badges")
    args = ap.parse_args(argv)

    ledger = Ledger(args.db)
    ok, bad = ledger.verify()
    if not ok:
        print(f"LEDGER TAMPERED: hash chain breaks at seq {bad}", file=sys.stderr)
        return 1

    candles = pd.read_csv(args.candles) if args.candles else fetch_closed_klines(args.symbol)
    report = score_pending(ledger, candles, args.symbol)
    print(f"scored={report.scored} not_mature={report.not_mature} "
          f"missing_candles={report.missing_candles}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    since = int(time.time() * 1000) - int(args.window_days * 86_400_000)
    for model in args.models:
        m = live_metrics(ledger, model, since_ts=since, symbol=args.symbol)
        (out / f"{model}.json").write_text(json.dumps(badge_json(m, f"{args.window_days:g}d")))
        (out / f"{model}_metrics.json").write_text(metrics_to_json(m))
        print(metrics_to_json(m))

    head = ledger.head()
    (out / "ledger_head.txt").write_text(head + "\n")
    print(f"ledger head: {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
