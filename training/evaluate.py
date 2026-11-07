"""Evaluasi berorientasi utilitas: confusion matrix, cost matrix, dan backtest notifikasi."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

# Biaya default: alert palsu (FP) = 1 unit (waktu/perhatian trader terbuang),
# lonjakan luput (FN) = 5 unit (risiko kerugian tanpa proteksi).
DEFAULT_COST_FP = 1.0
DEFAULT_COST_FN = 5.0


def classification_report(y: np.ndarray, p: np.ndarray, threshold: float,
                          cost_fp: float = DEFAULT_COST_FP, cost_fn: float = DEFAULT_COST_FN) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    total_cost = fp * cost_fp + fn * cost_fn
    return {
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "alert_rate": float(pred.mean()),
        "cost_matrix": {"fp": cost_fp, "fn": cost_fn, "tp": 0.0, "tn": 0.0},
        "total_cost": float(total_cost),
        "cost_per_1000": float(total_cost / len(y) * 1000),
    }


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None, "brier": None, "base_rate": float(y.mean())}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(y.mean()),
    }


def threshold_sweep(y: np.ndarray, p: np.ndarray, cost_fp: float = DEFAULT_COST_FP,
                    cost_fn: float = DEFAULT_COST_FN) -> pd.DataFrame:
    rows = [classification_report(y, p, t, cost_fp, cost_fn) for t in np.round(np.arange(0.05, 1.0, 0.05), 2)]
    flat = [{"threshold": r["threshold"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
             "alert_rate": r["alert_rate"], "cost_per_1000": r["cost_per_1000"], **r["confusion_matrix"]} for r in rows]
    return pd.DataFrame(flat)


def baseline_costs(y: np.ndarray, cost_fp: float = DEFAULT_COST_FP, cost_fn: float = DEFAULT_COST_FN) -> dict:
    y = np.asarray(y).astype(int)
    n = len(y)
    return {
        "never_alert_cost_per_1000": float(y.sum() * cost_fn / n * 1000),
        "always_alert_cost_per_1000": float((n - y.sum()) * cost_fp / n * 1000),
    }


def backtest_notifications(
    df: pd.DataFrame,
    threshold: float,
    horizon: int = 15,
    cooldown: int = 15,
    loss_quantile: float = 0.02,
    seed: int = 0,
) -> dict:
    """Simulasi: trader memegang posisi long terus-menerus. Setiap kali notifikasi
    Telegram berbunyi (prob >= threshold, dengan cooldown seperti di produksi),
    trader keluar posisi (manajemen risiko) selama `horizon` menit berikutnya.

    "Kerugian besar" = window 30 menit (tidak overlap) dengan return <= kuantil
    `loss_quantile` (default 2% terburuk) per simbol.

    Dibandingkan dengan baseline acak: jumlah alert yang sama, waktu acak.

    df wajib punya kolom: symbol, open_time, close, prob.
    """
    rng = np.random.default_rng(seed)
    per_symbol = []
    agg = {"loss_total": 0.0, "loss_avoided": 0.0, "loss_avoided_random": 0.0,
           "episodes": 0, "episodes_protected": 0, "episodes_protected_random": 0,
           "alerts": 0, "minutes": 0, "minutes_flat": 0}

    for symbol, g in df.sort_values("open_time").groupby("symbol"):
        g = g.reset_index(drop=True)
        n = len(g)
        ret_next = np.log(g["close"].shift(-1) / g["close"]).fillna(0).values  # return menit t -> t+1

        # alert dengan cooldown
        alert_idx, last = [], -10**9
        for i, pr in enumerate(g["prob"].values):
            if pr >= threshold and i - last >= cooldown:
                alert_idx.append(i)
                last = i
        random_idx = np.sort(rng.choice(n, size=min(len(alert_idx), n), replace=False)) if alert_idx else np.array([], int)

        def flat_mask(idxs):
            m = np.zeros(n, dtype=bool)
            for i in idxs:
                m[i: i + horizon] = True  # keluar posisi untuk return t..t+horizon-1
            return m

        flat = flat_mask(alert_idx)
        flat_rand = flat_mask(random_idx)

        # episode kerugian besar: window tidak overlap
        starts = np.arange(0, n - horizon, horizon)
        win_ret = np.array([ret_next[s: s + horizon].sum() for s in starts])
        if len(win_ret) == 0:
            continue
        cutoff = np.quantile(win_ret, loss_quantile)
        bad = starts[win_ret <= cutoff]
        loss_total = loss_avoided = loss_avoided_rand = 0.0
        prot = prot_rand = 0
        for s in bad:
            r = ret_next[s: s + horizon]
            neg = np.minimum(r, 0)
            loss_total += -neg.sum()
            loss_avoided += -(neg * flat[s: s + horizon]).sum()
            loss_avoided_rand += -(neg * flat_rand[s: s + horizon]).sum()
            prot += flat[s: s + horizon].mean() >= 0.5
            prot_rand += flat_rand[s: s + horizon].mean() >= 0.5

        per_symbol.append({
            "symbol": symbol, "alerts": len(alert_idx), "loss_episodes": int(len(bad)),
            "episodes_protected": int(prot),
            "loss_avoided_pct": float(loss_avoided / loss_total * 100) if loss_total else 0.0,
            "loss_avoided_random_pct": float(loss_avoided_rand / loss_total * 100) if loss_total else 0.0,
            "time_out_of_market_pct": float(flat.mean() * 100),
        })
        agg["loss_total"] += loss_total
        agg["loss_avoided"] += loss_avoided
        agg["loss_avoided_random"] += loss_avoided_rand
        agg["episodes"] += int(len(bad))
        agg["episodes_protected"] += int(prot)
        agg["episodes_protected_random"] += int(prot_rand)
        agg["alerts"] += len(alert_idx)
        agg["minutes"] += n
        agg["minutes_flat"] += int(flat.sum())

    lt = agg["loss_total"] or 1.0
    return {
        "threshold": threshold,
        "horizon_min": horizon,
        "cooldown_min": cooldown,
        "loss_quantile": loss_quantile,
        "alerts": agg["alerts"],
        "alerts_per_day": float(agg["alerts"] / max(agg["minutes"], 1) * 1440 / max(df["symbol"].nunique(), 1)),
        "large_loss_episodes": agg["episodes"],
        "episodes_protected": agg["episodes_protected"],
        "episodes_protected_pct": float(agg["episodes_protected"] / max(agg["episodes"], 1) * 100),
        "episodes_protected_random_pct": float(agg["episodes_protected_random"] / max(agg["episodes"], 1) * 100),
        "large_loss_avoided_pct": float(agg["loss_avoided"] / lt * 100),
        "large_loss_avoided_random_pct": float(agg["loss_avoided_random"] / lt * 100),
        "time_out_of_market_pct": float(agg["minutes_flat"] / max(agg["minutes"], 1) * 100),
        "per_symbol": per_symbol,
    }


def pick_pr_threshold(y: np.ndarray, p: np.ndarray, min_precision: float = 0.5) -> dict:
    """Threshold PROBABILITAS dari kurva precision-recall (validasi): recall maksimum dengan
    precision >= min_precision. Bila tidak ada yang memenuhi, pakai threshold F1 maksimum."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(np.asarray(y).astype(int), p)
    prec, rec = prec[:-1], rec[:-1]
    ok = prec >= min_precision
    if ok.any():
        i = int(np.flatnonzero(ok)[np.argmax(rec[ok])])
        rule = f"recall maksimum dengan precision >= {min_precision}"
    else:
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        i = int(np.argmax(f1))
        rule = f"F1 maksimum (tidak ada titik dengan precision >= {min_precision})"
    return {"threshold": float(thr[i]), "precision": float(prec[i]), "recall": float(rec[i]), "rule": rule,
            "min_precision": min_precision}


def episode_lead_times(frame: pd.DataFrame, alert: np.ndarray, horizon: int, lookback_min: int = 30) -> dict:
    """Lead time per episode spike, definisi SAMA dengan scorer ledger live (app/services/scorer.py):
    lead = (awal episode + H) - alert pertama dalam [awal - lookback, akhir episode]."""
    leads, caught, n_eps = [], 0, 0
    f = frame.assign(_alert=np.asarray(alert).astype(int)).sort_values(["symbol", "open_time"])
    for _, g in f.groupby("symbol"):
        ts = g["open_time"].to_numpy()
        y = g["label"].to_numpy()
        al = g["_alert"].to_numpy()
        pos = np.flatnonzero(y == 1)
        if not len(pos):
            continue
        breaks = np.flatnonzero(np.diff(ts[pos]) > 60_000)
        starts = np.r_[pos[0], pos[breaks + 1]]
        ends = np.r_[pos[breaks], pos[-1]]
        for s, e in zip(starts, ends):
            n_eps += 1
            m = (al == 1) & (ts >= ts[s] - lookback_min * 60_000) & (ts <= ts[e])
            if m.any():
                caught += 1
                leads.append((ts[s] + horizon * 60_000 - ts[m].min()) / 60_000)
    return {"episodes": n_eps, "caught": caught, "catch_rate": caught / n_eps if n_eps else None,
            "median_lead_min": float(np.median(leads)) if leads else None,
            "mean_lead_min": float(np.mean(leads)) if leads else None}
