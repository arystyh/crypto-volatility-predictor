import numpy as np
import pandas as pd

from training.cv import PurgedGroupTimeSeriesSplit
from training.evaluate import backtest_notifications, classification_report
from training.labels import make_labels


def test_purged_split_has_gap_and_no_future():
    groups = np.repeat(np.arange(1000), 3)  # 3 simbol per menit
    splitter = PurgedGroupTimeSeriesSplit(n_splits=4, purge=30, embargo=20)
    folds = list(splitter.split(groups))
    assert len(folds) == 4
    for tr, te in folds:
        assert groups[tr].max() < groups[te].min() - 50  # purge + embargo
        assert len(set(groups[tr]) & set(groups[te])) == 0


def test_zscore_label_uses_future_vol(candles):
    from app.ml.target import TargetSpec

    spec = TargetSpec(horizon=15, baseline_window=1440, min_baseline=240)
    lab = make_labels(candles, spec)
    valid = lab.dropna(subset=["label"])
    assert set(valid["label"].unique()) <= {0.0, 1.0}
    assert lab["label"].iloc[-15:].isna().all()  # tidak ada label tanpa masa depan
    assert (valid["label"] == (valid["z"] > 2)).all()
    # onset (hazard) konsisten dengan label: y = 1 => onset dalam 1..H (bila belum dalam spike)
    ok = valid["onset"].notna()
    pos = ok & (valid["label"] == 1)
    assert (valid.loc[pos, "onset"] > 0).all()  # label 1 => spike mulai dalam 1..H


def test_pr_threshold_selection():
    from training.evaluate import pick_pr_threshold

    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.05
    p = np.clip(y * 0.5 + rng.random(5000) * 0.6, 0, 1)
    t = pick_pr_threshold(y, p, 0.5)
    alert = p >= t["threshold"]
    assert (y[alert].mean()) >= 0.5 and t["rule"].startswith("recall maksimum")


def test_classification_report_costs():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.95])
    r = classification_report(y, p, 0.8, cost_fp=1, cost_fn=5)
    assert r["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert r["total_cost"] == 6


def test_backtest_perfect_alerts_avoid_losses():
    n = 600
    close = np.full(n, 100.0)
    close[300:330] = np.linspace(100, 80, 30)  # crash
    close[330:] = 80
    prob = np.zeros(n)
    prob[299] = 0.99  # alert tepat sebelum crash
    df = pd.DataFrame({"symbol": "X", "open_time": np.arange(n), "close": close, "prob": prob})
    bt = backtest_notifications(df, 0.8, horizon=30, loss_quantile=0.05)
    assert bt["alerts"] == 1
    assert bt["large_loss_avoided_pct"] > 90
