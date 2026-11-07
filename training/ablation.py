"""Ablation per kelompok fitur (roadmap 2.7): latih ulang tanpa satu kelompok, ukur ΔPR-AUC holdout
dan Δ median lead time (definisi sama dengan scorer live).

Kelompok mikrostruktur live (OBI, Hawkes, lead-lag, funding/OI) belum punya histori training -
Binance tidak menyediakan histori order book/likuidasi publik. Kolomnya hanya ada setelah
recorder Parquet (app/services/recorder.py) mengumpulkan data; sampai itu baris tabel = "n/a".

Jalankan: python -m training.train ... (ablation ikut dihitung, --no-ablation untuk melewati)
"""
from __future__ import annotations

import numpy as np

GROUPS: dict[str, list[str]] = {
    "VPIN": ["vpin_50", "vpin_ratio_15_240", "vpin_50_ewmz"],
    "SWT (wavelet)": ["swt_rv_15", "swt_noise_ratio_15", "swt_trend_15"],
    "OBI (order book)": ["obi_5"],
    "Hawkes": ["hawkes_trade_intensity", "hawkes_excitation", "hawkes_cross_liq_to_trade", "hawkes_branching"],
    "Lead-lag perp→spot": ["perp_ret_5s", "perp_spot_gap_5s", "basis", "basis_chg_1m"],
    "Funding/OI": ["funding_rate", "oi_chg_15m"],
}
NOISE_PR_AUC = 0.005  # perubahan lebih kecil dari ini dianggap tidak berarti


def run_ablation(features: list[str], fit_eval, full: dict) -> list[dict]:
    """fit_eval(feature_list) -> {"pr_auc": .., "median_lead_min": ..}. `full` = hasil semua fitur."""
    rows = []
    for name, cols in GROUPS.items():
        present = [c for c in cols if c in features]
        if not present:
            rows.append({"group": name, "features": cols, "status": "n/a (fitur live-only, belum ada histori)"})
            continue
        res = fit_eval([f for f in features if f not in present])
        d_auc = res["pr_auc"] - full["pr_auc"]
        d_lead = (None if res["median_lead_min"] is None or full["median_lead_min"] is None
                  else res["median_lead_min"] - full["median_lead_min"])
        useless = d_auc >= -NOISE_PR_AUC and (d_lead is None or d_lead >= 0)
        rows.append({"group": name, "features": present, "pr_auc_without": res["pr_auc"],
                     "delta_pr_auc": d_auc, "delta_median_lead_min": d_lead,
                     "status": "tidak berkontribusi -> kandidat dibuang" if useless else "berkontribusi"})
    return rows


def removable(rows: list[dict]) -> list[str]:
    return [f for r in rows if r.get("status", "").startswith("tidak") for f in r["features"]]


__all__ = ["GROUPS", "run_ablation", "removable", "np"]
