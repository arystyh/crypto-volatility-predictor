"""Training pipeline: champion XGBoost + kalibrasi + threshold PR + hazard + baseline + ablation.

Contoh:
    python -m training.train --source binance --symbols BTCUSDT --days 90
    python -m training.train --source csv --csv data/klines.csv
    python -m training.train --source synthetic --days 60      # demo/CI tanpa internet

Langkah (semua periode dipisah dengan jarak purge+embargo = TargetSpec.min_embargo(240) = H + 240):
 1. Candle 1m -> training/dataset.py (jalur kode sama dengan live) -> fitur + label (Z > 2)
 2. Split waktu:  [ train | val ] | holdout      (val = 20% akhir development)
 3. Purged walk-forward CV pada development (laporan)
 4. Champion XGBoost dilatih di `train`; kalibrasi isotonic + threshold PROBABILITAS alert dipilih
    dari kurva precision-recall di `val` (recall maks. dengan precision >= --min-precision)
 5. Holdout: PR-AUC (utama), Brier sebelum/sesudah kalibrasi, reliability diagram, lead time,
    hazard P(<=5/10/15) vs model biner, baseline persistence/HAR-RV/GARCH, challenger LightGBM,
    adaptive conformal vs threshold statis (termasuk skenario pergantian rezim), ablation
 6. Artefak -> models/, laporan -> reports/evaluation.{json,md} + reports/figures/
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from app.calibration import IsotonicCalibrator, brier, reliability_table
from app.conformal import false_alarm_rate, simulate_aci
from app.features import LONG_WINDOW
from app.hazard import expand_person_period
from app.normalize import HALFLIFE_MIN, MODEL_FEATURES
from app.regime import GaussianHMM, gate_mask, observations
from training.ablation import GROUPS, removable, run_ablation
from training.baselines import BASELINES, BaselineSuite, baseline_inputs
from training.dataset import build_dataset, holdout_split, make_splitter
from training.evaluate import (
    backtest_notifications,
    baseline_costs,
    classification_report,
    episode_lead_times,
    pick_pr_threshold,
    ranking_metrics,
    threshold_sweep,
)
from training.labels import HORIZON, SPEC, Z_THRESHOLD, Z_WINDOW

log = logging.getLogger("train")

# Regularisasi: tree dangkal, split konservatif, subsampling baris & kolom.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["aucpr", "logloss"],
    "max_depth": 4,
    "min_child_weight": 5,
    "gamma": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "n_jobs": -1,
}
N_ESTIMATORS = 400
GAP = SPEC.min_embargo(LONG_WINDOW)  # = H + lookback terpanjang


# ---------------------------------------------------------------- data
def load_candles(args) -> dict[str, pd.DataFrame]:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out: dict[str, pd.DataFrame] = {}
    if args.source == "synthetic":
        from training.synthetic import generate_candles

        prices = {"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0, "SOLUSDT": 150.0}
        for i, s in enumerate(symbols):
            out[s] = generate_candles(s, days=args.days, start_price=prices.get(s, 100.0), seed=args.seed + i)
    elif args.source == "csv":
        df = pd.read_csv(args.csv)
        if "symbol" not in df:
            df["symbol"] = symbols[0]
        for s, g in df.groupby("symbol"):
            out[str(s).upper()] = g.sort_values("open_time").reset_index(drop=True)
    else:
        from ingestion.binance import download_history

        end = int(time.time() * 1000)
        start = end - args.days * 86_400_000
        for s in symbols:
            log.info("Download %s %d hari dari %s ...", s, args.days, args.rest_url)
            rows = download_history(args.rest_url, s, "1m", start, end)
            out[s] = pd.DataFrame(rows).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
            if args.save_csv:
                Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
                out[s].to_csv(Path(args.save_csv).with_suffix(f".{s}.csv"), index=False)
    return out


# ---------------------------------------------------------------- model
def make_model(spw: float, n_estimators: int = N_ESTIMATORS) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(n_estimators=n_estimators, scale_pos_weight=spw, **XGB_PARAMS)


def run_cv(dev: pd.DataFrame, n_splits: int, spw: float) -> list[dict]:
    results = []
    X, y = dev[MODEL_FEATURES].values, dev["label"].values
    for fold, (tr, te) in enumerate(make_splitter(n_splits).split(dev["open_time"].values), start=1):
        model = make_model(spw)
        model.fit(X[tr], y[tr], verbose=False)
        m = ranking_metrics(y[te], model.predict_proba(X[te])[:, 1])
        m.update(fold=fold, n_train=int(len(tr)), n_test=int(len(te)))
        results.append(m)
        log.info("CV fold %d: PR-AUC=%s base=%.4f", fold, _fmt(m["pr_auc"]), m["base_rate"])
    return results


def export_onnx(model: xgb.XGBClassifier, n_features: int):
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        return convert_xgboost(model, initial_types=[("input", FloatTensorType([None, n_features]))], target_opset=15)
    except Exception as exc:  # noqa: BLE001
        log.warning("Ekspor ONNX gagal (%s); API akan memakai XGBoost native", exc)
        return None


def onnx_check(onnx_model, X: np.ndarray, p_ref: np.ndarray) -> dict | None:
    if onnx_model is None:
        return None
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": X})
    probs = np.asarray(out[1])[:, 1] if not isinstance(out[1][0], dict) else np.array([d[1] for d in out[1]])
    return {"parity_max_abs_diff": float(np.abs(probs - p_ref).max())}


def hazard_matrix(clf, X: np.ndarray) -> np.ndarray:
    n = len(X)
    Xpp = np.column_stack([np.repeat(X, HORIZON, axis=0), np.tile(np.arange(1, HORIZON + 1, dtype=np.float32), n)])
    return clf.predict_proba(Xpp)[:, 1].reshape(n, HORIZON)


def cumulative(h: np.ndarray) -> np.ndarray:
    """P(onset <= k) = 1 - prod_{j<=k}(1 - h_j): monoton naik di k by construction."""
    return 1.0 - np.cumprod(1.0 - h, axis=1)


def train_hazard(train: pd.DataFrame, val: pd.DataFrame, hold: pd.DataFrame, binary: dict, args) -> dict:
    """Model hazard diskrit (person-period) -> P(spike mulai <= 5/10/15 menit)."""
    elig = train[train["onset"].notna()].iloc[:: args.hazard_stride]
    Xd, yd = expand_person_period(elig[MODEL_FEATURES].to_numpy(np.float32), elig["onset"].to_numpy(), HORIZON)
    clf = xgb.XGBClassifier(n_estimators=300, **XGB_PARAMS)
    clf.fit(Xd, yd, verbose=False)
    onnx_model = export_onnx(clf, len(MODEL_FEATURES) + 1)

    cum_val = cumulative(hazard_matrix(clf, val[MODEL_FEATURES].to_numpy(np.float32)))
    cum = cumulative(hazard_matrix(clf, hold[MODEL_FEATURES].to_numpy(np.float32)))
    assert np.all(np.diff(cum, axis=1) >= -1e-12), "P(<=k) harus monoton"
    y = hold["label"].to_numpy()
    p15 = cum[:, HORIZON - 1]
    thr = pick_pr_threshold(val["label"].to_numpy(), cum_val[:, HORIZON - 1], args.min_precision)
    lead = episode_lead_times(hold, p15 >= thr["threshold"], HORIZON)
    rk = ranking_metrics(y, p15)
    # timing: menit onset vs perkiraan (hanya baris dengan onset)
    on = hold["onset"].to_numpy()
    pos = np.isfinite(on) & (on > 0)
    h = np.diff(np.c_[np.zeros(len(cum)), cum], axis=1)
    expected = (h * np.arange(1, HORIZON + 1)).sum(axis=1) / np.clip(h.sum(axis=1), 1e-12, None)
    res = {
        "_model": clf, "_onnx": onnx_model,
        "n_person_period_rows": int(len(yd)),
        "p_within": {str(k): ranking_metrics(y, cum[:, k - 1]) for k in (5, 10, 15)},
        "p15_pr_auc": rk["pr_auc"], "binary_pr_auc": binary["pr_auc"],
        "threshold": thr, "lead": lead, "binary_lead": binary["lead"],
        "onset_mae_min": float(np.abs(expected[pos] - on[pos]).mean()) if pos.any() else None,
        "naive_mid_horizon_mae_min": float(np.abs((HORIZON + 1) / 2 - on[pos]).mean()) if pos.any() else None,
        "monotone": True,
    }
    bl, hl = binary["lead"]["median_lead_min"], lead["median_lead_min"]
    res["criteria"] = {
        "pr_auc_not_worse": rk["pr_auc"] is not None and rk["pr_auc"] >= binary["pr_auc"] - 0.005,
        "lead_time_better": hl is not None and bl is not None and hl > bl,
    }
    res["_p15"] = p15
    return res


def conformal_eval(hold: pd.DataFrame, p: np.ndarray, tau0: float, args) -> dict:
    """Threshold statis vs ACI (umpan balik tertunda H menit), metrik = porsi alert palsu (FDR)."""
    order = np.lexsort((hold["symbol"].to_numpy(), hold["open_time"].to_numpy()))
    y = hold["label"].to_numpy()[order]
    pr = p[order]
    delay = HORIZON * hold["symbol"].nunique()

    def scenario(probs: np.ndarray) -> dict:
        taus = simulate_aci(probs, y, delay, tau0, args.conformal_alpha, args.conformal_gamma, mode="fdr")
        chunks = []
        for i, idx in enumerate(np.array_split(np.arange(len(y)), 4), start=1):
            chunks.append({"quarter": i,
                           "false_alarm_static": false_alarm_rate(probs[idx], y[idx], tau0),
                           "false_alarm_aci": false_alarm_rate(probs[idx], y[idx], taus[idx]),
                           "alerts_static": int((probs[idx] >= tau0).sum()),
                           "alerts_aci": int((probs[idx] >= taus[idx]).sum()),
                           "recall_static": float((probs[idx][y[idx] == 1] >= tau0).mean()) if (y[idx] == 1).any() else None,
                           "recall_aci": float((probs[idx][y[idx] == 1] >= taus[idx][y[idx] == 1]).mean())
                           if (y[idx] == 1).any() else None,
                           "tau_end": float(taus[idx][-1])})
        return {"false_alarm_static": false_alarm_rate(probs, y, tau0),
                "false_alarm_aci": false_alarm_rate(probs, y, taus), "tau_final": float(taus[-1]),
                "quarters": chunks}

    # Skenario pergantian rezim: mulai paruh kedua holdout model menjadi terlalu percaya diri
    # (logit + shift), seperti saat distribusi pasar bergeser setelah training.
    shifted = pr.copy()
    half = len(pr) // 2
    logit = np.log(np.clip(shifted[half:], 1e-6, 1 - 1e-6) / np.clip(1 - shifted[half:], 1e-6, None))
    shifted[half:] = 1 / (1 + np.exp(-(logit + args.regime_shift_logit)))
    return {"target_alpha": args.conformal_alpha, "gamma": args.conformal_gamma, "tau0": tau0, "mode": "fdr",
            "natural": scenario(pr), "regime_shift": {**scenario(shifted), "logit_shift": args.regime_shift_logit}}


def reliability_png(raw: list[dict], cal: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="kalibrasi sempurna")
    ax.plot([r["p_mean"] for r in raw], [r["freq"] for r in raw], "o-", color="#c2410c", label="sebelum (mentah)")
    ax.plot([r["p_mean"] for r in cal], [r["freq"] for r in cal], "s-", color="#1d4ed8", label="sesudah isotonic")
    ax.set_xlabel("Probabilitas prediksi (rata-rata bin)")
    ax.set_ylabel("Frekuensi spike aktual")
    ax.set_title("Reliability diagram (holdout)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------- main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["binance", "csv", "synthetic"], default="binance")
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=90, help="Minimal ~45 hari (label butuh 7-30 hari history)")
    ap.add_argument("--csv", help="Path CSV (kolom: open_time, open, high, low, close, volume[, symbol])")
    ap.add_argument("--save-csv", help="Simpan data Binance yang di-download ke path ini")
    ap.add_argument("--rest-url", default="https://data-api.binance.vision")
    ap.add_argument("--min-precision", type=float, default=0.5, help="Syarat precision saat memilih threshold dari kurva PR")
    ap.add_argument("--ewm-halflife", type=float, default=HALFLIFE_MIN)
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=5.0)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.2, help="Porsi akhir development untuk kalibrasi & threshold")
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spw-mode", choices=["sqrt", "full", "none"], default="sqrt")
    ap.add_argument("--no-challenger", action="store_true")
    ap.add_argument("--no-hazard", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--apply-ablation", action="store_true", help="Buang kelompok fitur yang tidak berkontribusi")
    ap.add_argument("--hazard-stride", type=int, default=3)
    ap.add_argument("--conformal-alpha", type=float, default=0.20, help="Target porsi alert palsu (FDR)")
    ap.add_argument("--conformal-gamma", type=float, default=0.01)
    ap.add_argument("--regime-shift-logit", type=float, default=1.5)
    ap.add_argument("--gate-window", type=int, default=30)
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rng_seed = args.seed

    candles = load_candles(args)
    data = build_dataset(candles, halflife=args.ewm_halflife, extra=baseline_inputs)
    if data.empty:
        raise SystemExit("Dataset kosong: butuh lebih banyak hari data (label butuh >= 7 hari history).")
    log.info("Dataset: %d baris, %d fitur, base rate %.3f%%", len(data), len(MODEL_FEATURES), data["label"].mean() * 100)

    dev_idx, hold_idx = holdout_split(data, args.holdout_frac)
    dev_all = data.iloc[dev_idx].reset_index(drop=True)
    tr_idx, val_idx = holdout_split(dev_all, args.val_frac)

    # HMM regime: fit di train saja, filter online di semua data
    train_sorted = dev_all.iloc[tr_idx].sort_values(["symbol", "open_time"])
    hmm = GaussianHMM.fit(observations(train_sorted))
    data["p_high"] = np.nan
    for _, idx in data.groupby("symbol").groups.items():
        sub = data.loc[idx].sort_values("open_time")
        data.loc[sub.index, "p_high"] = hmm.filter(observations(sub))
    dev_all = data.iloc[dev_idx].reset_index(drop=True)
    train = dev_all.iloc[tr_idx].reset_index(drop=True)
    val = dev_all.iloc[val_idx].reset_index(drop=True)
    hold = data.iloc[hold_idx].reset_index(drop=True)

    pos = train["label"].sum()
    ratio = (len(train) - pos) / max(pos, 1)
    spw = {"full": ratio, "sqrt": float(np.sqrt(ratio)), "none": 1.0}[args.spw_mode]
    log.info("Train=%d Val=%d Holdout=%d scale_pos_weight=%.2f", len(train), len(val), len(hold), spw)

    folds = run_cv(dev_all, args.cv_splits, spw)

    # ---- champion
    def fit_eval(features: list[str], return_model: bool = False):
        m = make_model(spw)
        m.fit(train[features].values, train["label"].values, verbose=False)
        pv = m.predict_proba(val[features].values)[:, 1]
        ph = m.predict_proba(hold[features].values)[:, 1]
        thr = pick_pr_threshold(val["label"].values, pv, args.min_precision)
        res = {"pr_auc": ranking_metrics(hold["label"].values, ph)["pr_auc"],
               "median_lead_min": episode_lead_times(hold, ph >= thr["threshold"], HORIZON)["median_lead_min"]}
        return (res, m, pv, ph) if return_model else res

    full_res, model, p_val_raw, p_hold_raw = fit_eval(MODEL_FEATURES, return_model=True)
    features = list(MODEL_FEATURES)
    ablation = None
    if not args.no_ablation:
        ablation = run_ablation(features, fit_eval, full_res)
        drop = removable(ablation)
        if args.apply_ablation and drop:
            features = [f for f in features if f not in drop]
            log.warning("Ablation: membuang %s", drop)
            full_res, model, p_val_raw, p_hold_raw = fit_eval(features, return_model=True)

    calibrator = IsotonicCalibrator.fit(p_val_raw, val["label"].values)
    p_val = calibrator(p_val_raw)
    p_hold = calibrator(p_hold_raw)
    thr = pick_pr_threshold(val["label"].values, p_val, args.min_precision)
    calibrator.meta = {"fitted_on": "validation fold", "n": int(len(val))}
    tau = thr["threshold"]
    y_hold = hold["label"].values
    log.info("Threshold alert dari kurva PR validasi: %.4f (%s)", tau, thr["rule"])

    onnx_model = export_onnx(model, len(features))
    model.get_booster().feature_names = features
    lead = episode_lead_times(hold, p_hold >= tau, HORIZON)
    ranking = ranking_metrics(y_hold, p_hold)
    calibration = {
        "brier_before": brier(p_hold_raw, y_hold), "brier_after": brier(p_hold, y_hold),
        "reliability_before": reliability_table(p_hold_raw, y_hold), "reliability_after": reliability_table(p_hold, y_hold),
        "figure": "figures/reliability.png",
    }
    reliability_png(calibration["reliability_before"], calibration["reliability_after"],
                    Path(args.report_dir) / "figures" / "reliability.png")

    # ---- gate HMM (informasi, sama seperti live)
    gate = {}
    for mode in ("off", "high"):
        g = np.zeros(len(hold), dtype=bool)
        for _, sub in hold.groupby("symbol"):
            sub = sub.sort_values("open_time")
            g[sub.index] = gate_mask(sub["p_high"].values, mode, args.gate_window)
        pg = np.where(g, p_hold, 0.0)
        rep = classification_report(y_hold, pg, tau, args.cost_fp, args.cost_fn)
        gate[mode] = {"model_calls_pct": float(g.mean() * 100), "precision": rep["precision"], "recall": rep["recall"],
                      "lead": episode_lead_times(hold, pg >= tau, HORIZON)}

    # ---- pembanding: baseline sederhana, challenger, hazard
    comparison = [{"model": "XGBoost (champion, isotonic)", "pr_auc": ranking["pr_auc"], "roc_auc": ranking["roc_auc"],
                   "brier": ranking["brier"], "threshold": tau, "lead": lead}]
    baselines = None
    if not args.no_baselines:
        suite = BaselineSuite(HORIZON).fit(train)
        pv, ph = suite.predict(val), suite.predict(hold)
        baselines = {}
        for name in BASELINES:
            ok_v, ok_h = np.isfinite(pv[name]), np.isfinite(ph[name])
            t = pick_pr_threshold(val["label"].values[ok_v], pv[name][ok_v], args.min_precision)
            rk = ranking_metrics(y_hold[ok_h], ph[name][ok_h])
            baselines[name] = {**rk, "threshold": t["threshold"], "n": int(ok_h.sum()),
                               "lead": episode_lead_times(hold[ok_h], ph[name][ok_h] >= t["threshold"], HORIZON)}
            comparison.append({"model": {"persistence": "Persistence (vol sekarang berlanjut)", "har_rv": "HAR-RV (15m/1j/1h)",
                                         "garch_1_1": "GARCH(1,1)"}[name], **{k: rk[k] for k in ("pr_auc", "roc_auc", "brier")},
                               "threshold": t["threshold"], "lead": baselines[name]["lead"]})
        baselines["garch_params"] = suite.garch

    challenger = None
    if not args.no_challenger:
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, learning_rate=0.05, num_leaves=15, max_depth=4,
                                 min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 reg_lambda=2.0, scale_pos_weight=spw, verbose=-1, random_state=rng_seed)
        clf.fit(train[features].values, train["label"].values)
        pc = clf.predict_proba(hold[features].values)[:, 1]
        tc = pick_pr_threshold(val["label"].values, clf.predict_proba(val[features].values)[:, 1], args.min_precision)
        challenger = {"model": "LightGBM", "_model": clf, "ranking": ranking_metrics(y_hold, pc),
                      "threshold": tc["threshold"], "lead": episode_lead_times(hold, pc >= tc["threshold"], HORIZON)}
        comparison.append({"model": "LightGBM (challenger)", **{k: challenger["ranking"][k] for k in ("pr_auc", "roc_auc", "brier")},
                           "threshold": tc["threshold"], "lead": challenger["lead"]})

    hazard = None
    if not args.no_hazard:
        hazard = train_hazard(train, val, hold, {"pr_auc": ranking["pr_auc"], "lead": lead}, args)
        comparison.append({"model": "Hazard P(onset ≤ 15m)", "pr_auc": hazard["p15_pr_auc"],
                           "roc_auc": hazard["p_within"]["15"]["roc_auc"], "brier": hazard["p_within"]["15"]["brier"],
                           "threshold": hazard["threshold"]["threshold"], "lead": hazard["lead"]})
        hazard.pop("_p15")

    sweep = threshold_sweep(y_hold, p_hold, args.cost_fp, args.cost_fn)
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": args.source,
        "symbols": list(candles.keys()),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_holdout": int(len(hold)),
        "features": features,
        "params": {**XGB_PARAMS, "n_estimators": N_ESTIMATORS, "scale_pos_weight": spw,
                   "ewm_halflife_min": args.ewm_halflife},
        "split": {"gap_rows": GAP, "rule": f"TargetSpec.min_embargo({LONG_WINDOW}) = H + {LONG_WINDOW}"},
        "cv": {"method": f"Purged walk-forward, {args.cv_splits} fold, jarak {GAP} menit",
               "folds": folds,
               "mean_pr_auc": float(np.nanmean([f["pr_auc"] or np.nan for f in folds])) if folds else None,
               "mean_roc_auc": float(np.nanmean([f["roc_auc"] or np.nan for f in folds])) if folds else None},
        "threshold": thr,
        "calibration": calibration,
        "holdout": {
            "ranking": ranking,
            "lead": lead,
            "at_threshold": classification_report(y_hold, p_hold, tau, args.cost_fp, args.cost_fn),
            "baseline_costs": baseline_costs(y_hold, args.cost_fp, args.cost_fn),
            "threshold_sweep": sweep.to_dict(orient="records"),
            "backtest": backtest_notifications(hold[["symbol", "open_time", "close"]].assign(prob=p_hold), tau,
                                               horizon=HORIZON),
            "hmm_gate": gate,
        },
        "comparison": comparison,
        "baselines": baselines,
        "challenger": challenger,
        "hazard_vs_binary": hazard,
        "conformal": conformal_eval(hold, p_hold, tau, args),
        "ablation": ablation,
        "onnx": onnx_check(onnx_model, hold[features].values[:2000].astype(np.float32), p_hold_raw[:2000]),
        "feature_importance_gain": dict(sorted(
            ((k, float(v)) for k, v in model.get_booster().get_score(importance_type="gain").items()),
            key=lambda kv: -kv[1])),
    }

    # ---- artefak
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M") + f"-{args.source}"
    model.get_booster().save_model(out / "xgb_volatility.json")
    if onnx_model is not None:
        (out / "xgb_volatility.onnx").write_bytes(onnx_model.SerializeToString())
    calibrator.meta.update({"model_version": version})
    calibrator.save(out / "calibration.json")
    if challenger:
        challenger.pop("_model").booster_.save_model(str(out / "challenger_lgbm.txt"))
    if hazard:
        hazard.pop("_model").get_booster().save_model(out / "hazard_xgb.json")
        hz_onnx = hazard.pop("_onnx")
        if hz_onnx is not None:
            (out / "hazard_xgb.onnx").write_bytes(hz_onnx.SerializeToString())
        (out / "hazard_metadata.json").write_text(json.dumps(
            {"features": features, "horizon": HORIZON, "extra_feature": "k (menit 1..H)",
             "outputs": "P(onset <= k) = 1 - prod(1 - h_j), k = 5/10/15"}, indent=2))
    hawkes_path = out / "hawkes_params.json"
    if not hawkes_path.exists():
        from app.hawkes import HawkesParams

        HawkesParams(model_version="default (belum di-fit; jalankan scripts/fit_hawkes.py)").save(hawkes_path)
    metadata = {
        "version": version,
        "features": features,
        "horizon_min": HORIZON,
        "label": f"log-Parkinson vol {HORIZON}m ke depan, Z > {Z_THRESHOLD} vs baseline {Z_WINDOW // 1440} hari (app.ml.target)",
        "target_spec": {"horizon": SPEC.horizon, "baseline_window": SPEC.baseline_window,
                        "min_baseline": SPEC.min_baseline, "z_threshold": SPEC.z_threshold},
        "alert_prob_threshold": tau,
        "alert_prob_threshold_rule": thr["rule"] + " (fold validasi, setelah kalibrasi isotonic)",
        "calibration": "calibration.json",
        "ewm_halflife_min": args.ewm_halflife,
        "trained_at": report["trained_at"],
        "data_source": args.source,
        "symbols": report["symbols"],
        "holdout_pr_auc": ranking["pr_auc"],
        "holdout_base_rate": ranking["base_rate"],
        "holdout_median_lead_min": lead["median_lead_min"],
        "scale_pos_weight": spw,
        "challenger": "challenger_lgbm.txt" if challenger else None,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    hmm.save(out / "hmm_regime.json")
    qs = train[features].quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    (out / "feature_quantiles.json").write_text(json.dumps(
        {f: {str(q): float(qs.loc[q, f]) for q in qs.index} for f in features}, indent=2))
    # referensi PSI drift (scripts/psi_drift.py): 10 bin desil dari data train
    from scripts.psi_drift import reference_bins

    (out / "feature_bins.json").write_text(json.dumps(reference_bins(train[features]), indent=2))

    rep = Path(args.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "evaluation.json").write_text(json.dumps(report, indent=2, default=float))
    write_markdown(report, rep / "EVALUATION.md")
    log.info("Selesai. Holdout PR-AUC=%s (base %.4f), median lead %s menit. Model -> %s",
             _fmt(ranking["pr_auc"]), ranking["base_rate"], lead["median_lead_min"], out)
    return report


# ---------------------------------------------------------------- report
def _lead(d: dict | None) -> str:
    if not d or d.get("median_lead_min") is None:
        return "n/a"
    return f"{d['median_lead_min']:.1f} mnt ({d['caught']}/{d['episodes']} episode)"


def comparison_table(report: dict) -> list[str]:
    base = report["holdout"]["ranking"]["base_rate"]
    lines = ["| Model | PR-AUC | Lift vs acak | ROC-AUC | Brier | Median lead time @ threshold val |",
             "|---|---|---|---|---|---|"]
    for r in sorted(report["comparison"], key=lambda r: -(r["pr_auc"] or 0)):
        lines.append(f"| {r['model']} | {_fmt(r['pr_auc'], 3)} | {(r['pr_auc'] or 0) / base:.1f}x | "
                     f"{_fmt(r['roc_auc'], 3)} | {_fmt(r['brier'], 4)} | {_lead(r['lead'])} |")
    return lines


def ablation_table(report: dict) -> list[str]:
    if not report.get("ablation"):
        return ["(ablation dilewati)"]
    lines = ["| Kelompok fitur dibuang | ΔPR-AUC | Δ median lead (menit) | Kesimpulan |", "|---|---|---|---|"]
    for r in report["ablation"]:
        if "delta_pr_auc" not in r:
            lines.append(f"| {r['group']} | n/a | n/a | {r['status']} |")
        else:
            dl = r["delta_median_lead_min"]
            lines.append(f"| {r['group']} | {r['delta_pr_auc']:+.4f} | {'n/a' if dl is None else f'{dl:+.1f}'} | {r['status']} |")
    return lines


def write_markdown(report: dict, path: Path) -> None:
    h = report["holdout"]
    cal = report["calibration"]
    thr = report["threshold"]
    cm = h["at_threshold"]["confusion_matrix"]
    hz = report.get("hazard_vs_binary")
    cf = report["conformal"]
    lines = [
        "# Evaluasi model volatility spike",
        "",
        f"- Dilatih {report['trained_at']} · sumber data **{report['data_source']}** · simbol {', '.join(report['symbols'])}",
        f"- Label: log-Parkinson vol {HORIZON} menit ke depan > {Z_THRESHOLD} sd di atas baseline {Z_WINDOW // 1440} hari "
        "(hanya masa lalu, `app/ml/target.py`). Threshold label ini TERPISAH dari threshold probabilitas alert.",
        f"- Split waktu train / val / holdout = {report['n_train']:,} / {report['n_val']:,} / {report['n_holdout']:,} baris, "
        f"jarak antar bagian {report['split']['gap_rows']} menit ({report['split']['rule']}).",
        "",
        "## Ringkasan holdout",
        "",
        f"PR-AUC **{_fmt(h['ranking']['pr_auc'], 3)}** vs base rate {h['ranking']['base_rate']:.3f} "
        f"(lift {h['ranking']['pr_auc'] / h['ranking']['base_rate']:.1f}x) · median lead time **{_lead(h['lead'])}** "
        f"· ROC-AUC {_fmt(h['ranking']['roc_auc'], 3)}",
        "",
        f"Threshold alert **{thr['threshold']:.4f}**, dipilih di fold validasi: {thr['rule']} "
        f"(val precision {thr['precision']:.2f}, recall {thr['recall']:.2f}). Di holdout: precision "
        f"{h['at_threshold']['precision']:.2f}, recall {h['at_threshold']['recall']:.2f} "
        f"(TP {cm['tp']}, FP {cm['fp']}, FN {cm['fn']}, TN {cm['tn']:,}).",
        "",
        "## Perbandingan dengan baseline (roadmap 2.5)",
        "",
        "Semua model memakai holdout yang sama; threshold masing-masing dipilih di fold validasi dengan aturan yang sama.",
        "",
        *comparison_table(report),
        "",
        "## Kalibrasi isotonic (roadmap 4.2)",
        "",
        f"Brier score holdout: sebelum **{cal['brier_before']:.5f}** → sesudah **{cal['brier_after']:.5f}**. "
        "Kalibrator dilatih di fold validasi (bukan holdout).",
        "",
        f"![Reliability diagram]({cal['figure']})",
        "",
    ]
    if hz:
        c = hz["criteria"]
        lines += [
            "## Hazard: P(spike mulai ≤ 5/10/15 menit) (roadmap 2.6)",
            "",
            "| Horizon | PR-AUC | Base rate |",
            "|---|---|---|",
            *[f"| ≤{k} menit | {_fmt(v['pr_auc'], 3)} | {v['base_rate']:.4f} |" for k, v in hz["p_within"].items()],
            "",
            "P(≤k) = 1 − Π(1 − h_j) sehingga monoton naik di k secara konstruksi (diuji).",
            "",
            f"- PR-AUC P(≤15) {_fmt(hz['p15_pr_auc'], 3)} vs model biner {_fmt(hz['binary_pr_auc'], 3)} → "
            f"{'tidak lebih buruk ✅' if c['pr_auc_not_worse'] else 'LEBIH BURUK ❌'}",
            f"- Median lead time hazard {_lead(hz['lead'])} vs biner {_lead(hz['binary_lead'])} → "
            f"{'lebih baik ✅' if c['lead_time_better'] else 'tidak lebih baik ❌'}",
            f"- MAE menit onset {_fmt(hz['onset_mae_min'], 2)} vs tebakan naif tengah horizon {_fmt(hz['naive_mid_horizon_mae_min'], 2)}",
            "",
        ]
    lines += [
        f"## Adaptive conformal vs threshold statis (roadmap 4.1, target alarm palsu {cf['target_alpha']:.0%})",
        "",
        "Metrik: porsi alert yang ternyata bukan spike. Umpan balik tertunda H menit seperti live.",
        "",
        "| Skenario | Kuartal | Alarm palsu statis | Alarm palsu ACI | Alert statis | Alert ACI | Recall statis | Recall ACI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sc, label in (("natural", "Holdout apa adanya"), ("regime_shift", f"Pergantian rezim (logit +{cf['regime_shift']['logit_shift']} di paruh 2)")):
        for q in cf[sc]["quarters"]:
            lines.append(f"| {label} | Q{q['quarter']} | {_fmt(q['false_alarm_static'], 2)} | {_fmt(q['false_alarm_aci'], 2)} | "
                         f"{q['alerts_static']} | {q['alerts_aci']} | {_fmt(q['recall_static'], 2)} | {_fmt(q['recall_aci'], 2)} |")
        lines.append(f"| {label} | **Total** | **{_fmt(cf[sc]['false_alarm_static'], 2)}** | **{_fmt(cf[sc]['false_alarm_aci'], 2)}** | | | | |")
    lines += [
        "",
        "## Ablation kelompok fitur (roadmap 2.7)",
        "",
        *ablation_table(report),
        "",
        "## Purged walk-forward CV",
        "",
        "| Fold | Train | Test | PR-AUC | Base rate |",
        "|---|---|---|---|---|",
        *[f"| {f['fold']} | {f['n_train']:,} | {f['n_test']:,} | {_fmt(f['pr_auc'], 3)} | {f['base_rate']:.4f} |"
          for f in report["cv"]["folds"]],
        f"| **Rata-rata** | | | **{_fmt(report['cv']['mean_pr_auc'], 3)}** | |",
        "",
        "## Gate rezim HMM",
        "",
        "| Gate | Model dipanggil | Precision | Recall | Median lead |",
        "|---|---|---|---|---|",
        *[f"| {k} | {v['model_calls_pct']:.0f}% | {v['precision']:.2f} | {v['recall']:.2f} | {_lead(v['lead'])} |"
          for k, v in h["hmm_gate"].items()],
        "",
    ]
    if report["data_source"] == "synthetic":
        lines += ["> ⚠️ Data **sintetis** (lingkungan build tidak bisa mengakses Binance). Angka di atas membuktikan "
                  "pipeline berjalan, BUKAN performa pasar nyata. Latih ulang dengan `--source binance`.", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
