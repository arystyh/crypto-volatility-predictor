"""Target definition for the volatility-spike classifier.

This module is the single source of truth for what the model predicts. Training
(offline, pandas) and serving (online, one candle at a time) both import it, and
``tests/test_target.py`` checks that the two paths agree to floating-point
precision.

Definition
----------
Inputs are contiguous, closed 1-minute candles. Row ``t`` is the candle that
has just closed; the decision is made at its close time.

    pk_i     = ln(high_i / low_i)^2 / (4 ln 2)          per-candle Parkinson variance
    lv(a..b) = 0.5 * ln( mean(pk_a .. pk_b) )           log Parkinson volatility

    trail_t  = lv(t-H+1 .. t)                           known at time t
    fwd_t    = lv(t+1 .. t+H)                           the thing we forecast
    mu_t, sd_t = mean / std(ddof=0) of trail_s for s in (t-W, t]   past-only baseline
    z_t      = (fwd_t - mu_t) / sd_t
    y_t      = 1 if z_t > Z else 0

Why these choices
-----------------
* Log volatility: raw volatility is right-skewed, so a Z-score on raw values
  fires mostly on the skew. Log volatility is close to symmetric.
* The baseline uses *trailing* windows that ended at or before ``t``. Nothing
  after the decision time leaks into ``mu_t`` / ``sd_t``. Only ``fwd_t`` looks
  ahead, by design: it is the label.
* ``mu_t`` and ``sd_t`` are frozen at decision time and stored in the
  prediction ledger, so a prediction is later scored against exactly the
  baseline that existed when it was made.

Defaults: H = 15 minutes, W = 30 days (43,200 candles), Z = 2.0, and no label
until 7 days of baseline exist.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

LN2X4 = 4.0 * math.log(2.0)
VAR_FLOOR = 1e-18  # a run of zero-range candles would otherwise give ln(0)
CANDLE_MS = 60_000


@dataclass(frozen=True)
class TargetSpec:
    horizon: int = 15             # H: forward window in candles (minutes)
    baseline_window: int = 43_200  # W: 30 days of 1m candles
    min_baseline: int = 10_080    # no label until 7 days of trailing values exist
    z_threshold: float = 2.0      # Z

    def min_embargo(self, feature_lookback: int) -> int:
        """Smallest purge/embargo gap (in rows) for cross-validation.

        A training row at t uses features from (t - lookback, t] and a label from
        (t, t + H]. Two rows can share information if they are closer than
        H + lookback, so the gap between train and test folds must be at least that.
        """
        return self.horizon + feature_lookback


# ----------------------------------------------------------------------------
# Offline (vectorised) path
# ----------------------------------------------------------------------------

def _check_contiguous(df: pd.DataFrame) -> None:
    if "open_time" not in df.columns:
        return
    steps = np.diff(df["open_time"].to_numpy(dtype=np.int64))
    if steps.size and not np.all(steps == CANDLE_MS):
        bad = int(np.argmax(steps != CANDLE_MS))
        raise ValueError(
            f"candles are not contiguous 1m bars (gap after row {bad}); "
            "run the REST gap-filler before computing targets"
        )


def candle_pk_var(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if np.any(low <= 0) or np.any(high < low):
        raise ValueError("invalid candle: need 0 < low <= high")
    return np.log(high / low) ** 2 / LN2X4


def trailing_logvol(df: pd.DataFrame, spec: TargetSpec) -> pd.Series:
    pk = pd.Series(candle_pk_var(df["high"], df["low"]), index=df.index)
    var = pk.rolling(spec.horizon, min_periods=spec.horizon).mean()
    return 0.5 * np.log(var.clip(lower=VAR_FLOOR))


def compute_targets(df: pd.DataFrame, spec: TargetSpec = TargetSpec()) -> pd.DataFrame:
    """Return trail, fwd, mu, sd, z and label for every row.

    ``label`` is NaN where it is undefined: during baseline warm-up and for the
    last H rows, whose future is not yet known.
    """
    _check_contiguous(df)
    trail = trailing_logvol(df, spec)
    fwd = trail.shift(-spec.horizon)
    roll = trail.rolling(spec.baseline_window, min_periods=spec.min_baseline)
    mu = roll.mean()
    sd = roll.std(ddof=0)
    z = (fwd - mu) / sd.where(sd > 0)
    label = (z > spec.z_threshold).astype(float).where(z.notna())
    return pd.DataFrame(
        {"trail_logvol": trail, "fwd_logvol": fwd, "baseline_mu": mu,
         "baseline_sd": sd, "z": z, "label": label},
        index=df.index,
    )


def forward_logvol(high: np.ndarray, low: np.ndarray) -> float:
    """Log Parkinson volatility of exactly the H forward candles."""
    var = float(np.mean(candle_pk_var(high, low)))
    return 0.5 * math.log(max(var, VAR_FLOOR))


# ----------------------------------------------------------------------------
# Online (streaming) path
# ----------------------------------------------------------------------------

class OnlineBaseline:
    """Streaming equivalent of the trailing / baseline part of compute_targets.

    Feed it each closed candle in order. After every update, ``mu`` and ``sd``
    equal the offline ``baseline_mu`` / ``baseline_sd`` for that row. The state
    is a few deques of floats, so it can be serialised to Redis with
    ``to_state`` / ``from_state`` and survive a container restart.
    """

    RESYNC_EVERY = 10_000  # recompute running sums exactly to stop float drift

    def __init__(self, spec: TargetSpec = TargetSpec()):
        self.spec = spec
        self._pk: deque[float] = deque(maxlen=spec.horizon)
        self._pk_sum = 0.0
        self._trail: deque[float] = deque(maxlen=spec.baseline_window)  # NaN during warm-up
        self._ref: float | None = None  # shift for numerically stable variance
        self._s1 = 0.0
        self._s2 = 0.0
        self._n = 0
        self._since_resync = 0
        self.last_open_time: int | None = None
        self.trail: float = math.nan
        self.mu: float = math.nan
        self.sd: float = math.nan

    def update(self, high: float, low: float, open_time: int | None = None) -> None:
        if open_time is not None:
            if self.last_open_time is not None and open_time != self.last_open_time + CANDLE_MS:
                raise ValueError(
                    f"non-contiguous candle: expected {self.last_open_time + CANDLE_MS}, "
                    f"got {open_time}; gap-fill before updating"
                )
            self.last_open_time = open_time

        pk = float(candle_pk_var(np.array([high]), np.array([low]))[0])
        if len(self._pk) == self._pk.maxlen:
            self._pk_sum -= self._pk[0]
        self._pk.append(pk)
        self._pk_sum += pk

        if len(self._pk) < self.spec.horizon:
            trail = math.nan
        else:
            var = max(self._pk_sum / self.spec.horizon, 0.0)
            trail = 0.5 * math.log(max(var, VAR_FLOOR))
        self.trail = trail

        if len(self._trail) == self._trail.maxlen:
            self._remove(self._trail[0])
        self._trail.append(trail)
        self._add(trail)

        self._since_resync += 1
        if self._since_resync >= self.RESYNC_EVERY:
            self._resync()

        if self._n >= self.spec.min_baseline:
            mean_shifted = self._s1 / self._n
            var = max(self._s2 / self._n - mean_shifted ** 2, 0.0)
            self.mu = self._ref + mean_shifted
            self.sd = math.sqrt(var)
        else:
            self.mu = self.sd = math.nan

    def z_of(self, fwd_logvol: float) -> float:
        if not (self.sd > 0):
            return math.nan
        return (fwd_logvol - self.mu) / self.sd

    @property
    def ready(self) -> bool:
        return self._n >= self.spec.min_baseline and self.sd > 0

    # -- running sums over non-NaN values, shifted by a reference value --------
    def _add(self, x: float) -> None:
        if math.isnan(x):
            return
        if self._ref is None:
            self._ref = x
        d = x - self._ref
        self._s1 += d
        self._s2 += d * d
        self._n += 1

    def _remove(self, x: float) -> None:
        if math.isnan(x):
            return
        d = x - self._ref
        self._s1 -= d
        self._s2 -= d * d
        self._n -= 1

    def _resync(self) -> None:
        self._pk_sum = math.fsum(self._pk)
        vals = [v for v in self._trail if not math.isnan(v)]
        self._ref = vals[0] if vals else None
        self._s1 = math.fsum(v - self._ref for v in vals) if vals else 0.0
        self._s2 = math.fsum((v - self._ref) ** 2 for v in vals) if vals else 0.0
        self._n = len(vals)
        self._since_resync = 0

    # -- persistence -------------------------------------------------------------
    def to_state(self) -> dict:
        return {
            "spec": asdict(self.spec),
            "pk": list(self._pk),
            "trail": [None if math.isnan(v) else v for v in self._trail],
            "last_open_time": self.last_open_time,
        }

    @classmethod
    def from_state(cls, state: dict) -> "OnlineBaseline":
        obj = cls(TargetSpec(**state["spec"]))
        obj._pk.extend(state["pk"])
        obj._trail.extend(math.nan if v is None else v for v in state["trail"])
        obj.last_open_time = state["last_open_time"]
        obj._resync()
        if obj._trail:
            obj.trail = obj._trail[-1]
        if obj._n >= obj.spec.min_baseline:
            mean_shifted = obj._s1 / obj._n
            obj.mu = obj._ref + mean_shifted
            obj.sd = math.sqrt(max(obj._s2 / obj._n - mean_shifted ** 2, 0.0))
        return obj
