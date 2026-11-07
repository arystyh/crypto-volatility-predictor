"""Roadmap 4.3: halaman insiden dibangun ulang dengan satu perintah (di sini: data sintetis kecil)."""
import json

import numpy as np

from scripts import build_incident as bi
from tests.helpers import SMALL_SPEC
from training.synthetic import generate_candles


def test_build_incident_page(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "SPEC", SMALL_SPEC)
    c = generate_candles("BTCUSDT", days=2, seed=5)
    models = bi.Models("models")
    start, end = int(c["open_time"].iloc[1500]), int(c["open_time"].iloc[2400])
    ev = bi.synthetic_events(c, 1)
    s = bi.build("t1", "Uji", "catatan", c, ev, start, end, models, 15, "synthetic", "python -m x", out_dir=tmp_path)
    page = (tmp_path / "t1.html").read_text()
    for needle in ("DATA SINTETIS", "Lead time", "data:image/png;base64,", "P(≤5m)", "python -m x"):
        assert needle in page
    assert s["episodes"] >= s["caught"] and s["hawkes_events"] == len(ev[0]) > 0
    for a in s["alert_table"]:  # alert hanya bila P >= threshold, dan hazard kumulatif monoton
        assert a["prob"] >= models.threshold and a["p_le_5"] <= a["p_le_10"] <= a["p_le_15"]
    ts = [a["decision_ts"] for a in s["alert_table"]]
    assert all(np.diff(ts) >= 15 * 60_000)  # cooldown
    bi.update_index([s], tmp_path)
    idx = json.loads((tmp_path / "index.json").read_text())
    assert idx["incidents"][0]["page"] == "t1.html" and idx["incidents"][0]["synthetic"] is True


def test_hawkes_minute_intensity_rises_after_burst():
    p = bi.HawkesParams()
    t = np.r_[np.linspace(100, 101, 20), [500.0]]
    k = np.zeros(len(t), dtype=int)
    df = bi.hawkes_per_minute(t, k, np.array([60.0, 101.5, 400.0]), p)
    assert df["hawkes_trade"].iloc[1] > 10 * df["hawkes_trade"].iloc[0]
    assert abs(df["hawkes_trade"].iloc[2] - p.mu[0]) < 1e-6  # sudah meluruh
