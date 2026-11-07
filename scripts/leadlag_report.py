"""Laporan lead-lag perp -> spot (roadmap 2.3): berapa detik futures perpetual memimpin spot.

    python -m scripts.leadlag_report --ticks data/ticks                    # rekaman nyata
    python -m scripts.leadlag_report --synthetic                           # demo (perp memimpin 2 detik)

Output: reports/leadlag.md + reports/leadlag.json. Fitur lead-lag tetap LIVE-ONLY (tidak masuk
model) sampai laporan dari data nyata menunjukkan lag > 0 yang signifikan DAN ablation membuktikan
fitur ini menambah PR-AUC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ingestion.leadlag import lagged_xcorr


def prices(df, stream: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["stream"] == stream]
    d = [json.loads(p) for p in sub["payload"]]
    t = np.array([x["T"] for x in d], dtype=float)
    p = np.array([float(x["p"]) for x in d])
    o = np.argsort(t, kind="mergesort")
    return t[o], p[o]


def write(res: dict, source: str, out_md: Path) -> None:
    rows = "\n".join(f"| {L:+d} | {c:+.3f} |{' ← terbaik' if L == res['best_lag_s'] else ''}"
                     for L, c in sorted(res["corr_by_lag"].items()))
    verdict = (f"Perp **memimpin spot {res['best_lag_s']} detik** (korelasi {res['best_corr']:.3f} vs "
               f"{res['corr_at_0']:.3f} pada lag 0), signifikan (> 3 standar error)."
               if res["significant"] else
               "Tidak ada lead perp→spot yang signifikan (lag terbaik ≤ 0 atau selisih korelasi < 3 standar error).")
    warn = ("\n> ⚠️ Data **sintetis**: generator sengaja membuat perp memimpin 2 detik. Ini menguji bahwa "
            "pengukurannya bekerja, BUKAN temuan pasar. Jalankan `--ticks` pada rekaman ≥ 24 jam.\n"
            if source == "synthetic" else "")
    out_md.write_text(f"""# Lead-lag perp → spot (BTCUSDT)

Sumber: {source} · {res['n_seconds']:,} detik return 1 detik (harga transaksi terakhir per detik).
{warn}
{verdict}

corr(return perp(t), return spot(t + lag)); lag > 0 berarti perp bergerak lebih dulu.

| Lag (detik) | Korelasi |
|---|---|
{rows}

Status fitur `perp_lead_ret_*`, `basis_bps`: **live-only**, tidak dipakai model sampai terbukti di data nyata.
""")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticks")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-lag", type=int, default=10)
    ap.add_argument("--out", default="reports/leadlag.md")
    args = ap.parse_args(argv)
    if args.synthetic:
        from app.services import simticks
        from training.synthetic import generate_candles

        df = simticks.generate(generate_candles("BTCUSDT", days=1, seed=4).iloc[:180], seed=2, depth_ms=5000,
                               trades_per_min=300, perp_lead_ms=2000)
        source = "synthetic"
    else:
        from app.services.recorder import load

        df = load(args.ticks, streams=["aggTrade", "fut_aggTrade"])
        source = f"rekaman {args.ticks}"
    res = lagged_xcorr(*prices(df, "aggTrade"), *prices(df, "fut_aggTrade"), max_lag_s=args.max_lag)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(res, source, out)
    out.with_suffix(".json").write_text(json.dumps({**res, "source": source}, indent=2))
    print(json.dumps({k: res[k] for k in ("best_lag_s", "best_corr", "corr_at_0", "significant")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
