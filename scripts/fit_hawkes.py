"""Fit parameter Hawkes (transaksi besar + likuidasi) secara offline, mingguan (roadmap 2.2).

Live hanya MEMAKAI parameter ini (models/hawkes_params.json); fit tidak pernah berjalan di jalur
prediksi.

    python -m scripts.fit_hawkes --ticks data/ticks                 # rekaman TickRecorder
    python -m scripts.fit_hawkes --agg-trades BTCUSDT-aggTrades-2024-08-05.zip [--liquidations liq.zip]
    python -m scripts.fit_hawkes --synthetic                        # demo: simulasi lalu fit ulang

Event: 0 = aggTrade dengan notional >= LARGE_TRADE_USD, 1 = forceOrder (sampel snapshot 1 detik).
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings
from app.hawkes import LIQ, TRADE, HawkesParams, fit_hawkes, simulate_hawkes

log = logging.getLogger("fit_hawkes")


def events_from_ticks(root: str, large_usd: float, symbol: str = "BTCUSDT") -> tuple[np.ndarray, np.ndarray]:
    from app.services.recorder import load

    df = load(root, streams=["aggTrade", "forceOrder"], symbol=symbol)
    t, k = [], []
    for r in df.itertuples():
        d = json.loads(r.payload)
        if r.stream == "aggTrade":
            if float(d["p"]) * float(d["q"]) >= large_usd:
                t.append(d["T"] / 1000)
                k.append(TRADE)
        else:
            t.append(d["o"]["T"] / 1000)
            k.append(LIQ)
    return np.asarray(t, float), np.asarray(k, int)


def events_from_archive(agg_path: str, liq_path: str | None, large_usd: float) -> tuple[np.ndarray, np.ndarray]:
    """Arsip data.binance.vision (zip/csv). aggTrades: kolom 1=price 2=qty 5=time."""
    tr = pd.read_csv(agg_path, header=None)
    if not str(tr.iloc[0, 0]).isdigit():
        tr = tr.iloc[1:].astype(float)
    ts = tr[5].astype(float).to_numpy()
    ts = np.where(ts > 1e14, ts / 1e6, ts / 1e3)
    sel = (tr[1].astype(float) * tr[2].astype(float)).to_numpy() >= large_usd
    t, k = [ts[sel]], [np.full(sel.sum(), TRADE)]
    if liq_path:
        lq = pd.read_csv(liq_path, header=None)
        if not str(lq.iloc[0, 0]).isdigit():
            lq = lq.iloc[1:]
        t.append(lq[0].astype(float).to_numpy() / 1000)
        k.append(np.full(len(lq), LIQ))
    return np.concatenate(t), np.concatenate(k).astype(int)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticks")
    src.add_argument("--agg-trades")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--liquidations")
    ap.add_argument("--out", default="models/hawkes_params.json")
    ap.add_argument("--max-events", type=int, default=200_000, help="ambil N event terakhir (MLE O(N))")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    s = Settings()
    if args.synthetic:
        truth = HawkesParams(mu=[0.05, 0.01], alpha=[[0.4, 0.5], [0.05, 0.3]], beta=1.2)
        t, k = simulate_hawkes(truth, 20_000, seed=0)
        log.info("Simulasi dari %s", truth.to_dict())
    elif args.ticks:
        t, k = events_from_ticks(args.ticks, s.large_trade_usd)
    else:
        t, k = events_from_archive(args.agg_trades, args.liquidations, s.large_trade_usd)
    order = np.argsort(t, kind="mergesort")
    t, k = t[order][-args.max_events:], k[order][-args.max_events:]
    log.info("%d event (transaksi besar %d, likuidasi %d), rentang %.1f jam", len(t), (k == TRADE).sum(),
             (k == LIQ).sum(), (t[-1] - t[0]) / 3600 if len(t) else 0)
    p = fit_hawkes(t, k, dim=2)
    p.fitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.model_version = "synthetic" if args.synthetic else Path(args.ticks or args.agg_trades).name
    if p.branching >= 1:
        log.warning("Branching ratio %.3f >= 1 (non-stasioner): periksa data sebelum dipakai live", p.branching)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    p.save(args.out)
    print(json.dumps(p.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
