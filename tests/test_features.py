import math

import numpy as np
import pandas as pd
import pytest

from app.features import FEATURE_COLUMNS, MIN_CANDLES, compute_features, latest_feature_vector, parkinson_vol


def test_feature_columns_complete(candles):
    feats = compute_features(candles)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert not feats.iloc[MIN_CANDLES:].isna().any().any()


def test_no_lookahead(candles):
    """Fitur di waktu t tidak boleh berubah jika data setelah t diubah."""
    t = 1000
    base = compute_features(candles).iloc[t]
    mutated = candles.copy()
    mutated.loc[t + 1:, ["open", "high", "low", "close"]] *= 3
    mutated.loc[t + 1:, "volume"] *= 10
    after = compute_features(mutated).iloc[t]
    pd.testing.assert_series_equal(base, after)


def test_parkinson_formula():
    h = pd.Series([101.0] * 20)
    l = pd.Series([99.0] * 20)
    expected = math.sqrt(math.log(101 / 99) ** 2 / (4 * math.log(2)))
    assert parkinson_vol(h, l, 15).iloc[-1] == pytest.approx(expected)


def test_cyclical_hour(candles):
    feats = compute_features(candles)
    ts = pd.to_datetime(candles["open_time"], unit="ms", utc=True)
    midnight = feats[(ts.dt.hour == 0) & (ts.dt.minute == 0)].iloc[0]
    assert midnight["hour_sin"] == pytest.approx(0, abs=1e-9)
    assert midnight["hour_cos"] == pytest.approx(1)
    assert np.allclose(feats["hour_sin"] ** 2 + feats["hour_cos"] ** 2, 1)


def test_latest_vector_needs_history(candles):
    with pytest.raises(ValueError):
        latest_feature_vector(candles.iloc[: MIN_CANDLES - 1])
    row = latest_feature_vector(candles.iloc[: MIN_CANDLES + 5])
    assert set(row.index) == set(FEATURE_COLUMNS)
