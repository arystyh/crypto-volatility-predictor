"""PSI drift fitur live vs data train (opsional di CI mingguan, roadmap 1.7).

PSI = sum_b (a_b - e_b) * ln(a_b / e_b), bin = desil data train (models/feature_bins.json).
Aturan praktis: < 0.1 stabil, 0.1-0.2 bergeser, > 0.2 drift signifikan.

    python -m scripts.psi_drift --url https://<host> --threshold 0.2 --out badges/psi.json
    python -m scripts.psi_drift --live features.json            # {"rows": [{fitur: nilai}, ...]}

Exit code 2 bila ada fitur dengan PSI > threshold (CI menandai, tidak memblokir verifikasi ledger).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4


def reference_bins(train: pd.DataFrame, n_bins: int = 10) -> dict:
    out = {}
    for col in train.columns:
        x = train[col].dropna().to_numpy(float)
        if len(x) < n_bins * 10:
            continue
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
        out[col] = {"edges": edges.tolist(), "expected": (counts / counts.sum()).tolist()}
    return out


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(bins: dict, live: pd.DataFrame, threshold: float = 0.2, min_rows: int = 200) -> dict:
    feats = {}
    for col, ref in bins.items():
        if col not in live:
            continue
        x = pd.to_numeric(live[col], errors="coerce").dropna().to_numpy(float)
        if len(x) < min_rows:
            continue
        edges = np.asarray(ref["edges"])
        counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
        feats[col] = round(psi(ref["expected"], counts / counts.sum()), 4)
    drifted = sorted((k for k, v in feats.items() if v > threshold), key=lambda k: -feats[k])
    return {"threshold": threshold, "n_live": int(len(live)), "features": feats, "drifted": drifted,
            "max_psi": max(feats.values()) if feats else None}


def badge(rep: dict) -> dict:
    if rep["max_psi"] is None:
        return {"schemaVersion": 1, "label": "feature drift (PSI)", "message": "no data", "color": "lightgrey"}
    n = len(rep["drifted"])
    return {"schemaVersion": 1, "label": "feature drift (PSI)",
            "message": f"{n} fitur > {rep['threshold']}" if n else f"stabil (max {rep['max_psi']:.2f})",
            "color": "red" if n else "brightgreen"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="base URL service (GET /features/live)")
    src.add_argument("--live", help="file JSON {'rows': [...]}")
    ap.add_argument("--bins", default="models/feature_bins.json")
    ap.add_argument("--hours", type=float, default=168)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--out", default="badges/psi.json")
    args = ap.parse_args(argv)
    if args.url:
        import httpx

        rows = httpx.get(f"{args.url.rstrip('/')}/features/live", params={"hours": args.hours}, timeout=60).json()["rows"]
    else:
        rows = json.loads(Path(args.live).read_text())["rows"]
    rep = psi_report(json.loads(Path(args.bins).read_text()), pd.DataFrame(rows), args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(badge(rep)))
    out.with_name("psi_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("n_live", "max_psi", "drifted")}))
    return 2 if rep["drifted"] else 0


if __name__ == "__main__":
    sys.exit(main())
