"""Roadmap 2.4: purged CV tanpa kebocoran. Embargo = TargetSpec.min_embargo(lookback terpanjang)."""
import numpy as np
import pandas as pd

from app.features import LONG_WINDOW
from training.dataset import embargo_rows, holdout_split, make_splitter, windows_overlap
from training.labels import SPEC


def _times(n, start=1_700_000_000_000, holes=()):
    t = start + np.arange(n) * 60_000
    for a, b in holes:  # buang rentang -> data berlubang seperti maintenance Binance
        t = t[(t < start + a * 60_000) | (t >= start + b * 60_000)]
    return t


def test_embargo_is_derived_from_target_spec():
    assert embargo_rows() == SPEC.horizon + LONG_WINDOW == SPEC.min_embargo(LONG_WINDOW)


def test_overlap_detector_works():
    t = _times(1000)
    assert windows_overlap(t[:500], t[500:], LONG_WINDOW, SPEC.horizon)        # tanpa jarak: bocor
    gap = SPEC.min_embargo(LONG_WINDOW)
    assert not windows_overlap(t[:500], t[500 + gap:], LONG_WINDOW, SPEC.horizon)
    assert windows_overlap(t[:500], t[500 + gap - 2:], LONG_WINDOW, SPEC.horizon)


def test_no_train_window_overlaps_any_test_fold_multi_symbol_with_holes():
    t = _times(12_000, holes=[(3000, 3100), (7000, 7005)])
    groups = np.concatenate([t, t])  # dua simbol, menit yang sama
    splitter = make_splitter(n_splits=5)
    folds = list(splitter.split(groups))
    assert len(folds) == 5
    for tr, te in folds:
        assert groups[tr].max() < groups[te].min()  # walk-forward: train selalu di masa lalu
        assert not windows_overlap(groups[tr], groups[te], LONG_WINDOW, SPEC.horizon)
        # semua simbol pada menit yang sama berada di sisi yang sama
        assert not set(groups[tr]) & set(groups[te])


def test_holdout_split_has_no_overlap():
    t = _times(20_000)
    data = pd.DataFrame({"open_time": np.concatenate([t, t]), "symbol": ["A"] * len(t) + ["B"] * len(t)})
    dev, hold = holdout_split(data, 0.2)
    assert len(dev) and len(hold)
    assert not windows_overlap(data["open_time"].to_numpy()[dev], data["open_time"].to_numpy()[hold],
                               LONG_WINDOW, SPEC.horizon)
