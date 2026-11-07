"""Skrip CI mingguan (roadmap 1.7): verifikasi ledger independen + anchor git, PSI drift."""
import json
import sqlite3

import numpy as np
import pandas as pd

from app.services.ledger import Ledger, Prediction
from scripts import psi_drift, verify_ledger


def _ledger(path, n=5):
    led = Ledger(str(path), max_log_lag_ms=10**15, clock=lambda: 10**6)
    for i in range(n):
        led.log_prediction(Prediction("BTCUSDT", i * 60_000, 15, "champion_xgboost", 0.1 * (i % 9), 0.07,
                                      -7.0, 1.0, 2.0))
    return led


def test_verify_ledger_ok_and_anchor(tmp_path):
    led = _ledger(tmp_path / "l.db")
    mid = led.export()[2]["row_hash"]
    led.close()
    (tmp_path / "anchor.txt").write_text(mid + "\n")
    out = tmp_path / "badges"
    assert verify_ledger.main(["--db", str(tmp_path / "l.db"), "--anchor", str(tmp_path / "anchor.txt"),
                               "--out", str(out)]) == 0
    assert (out / "ledger_head.txt").read_text().strip() != mid  # head maju
    assert json.loads((out / "ledger.json").read_text())["color"] == "brightgreen"
    (tmp_path / "anchor.txt").write_text("ab" * 32)  # head yang tidak pernah ada -> riwayat ditulis ulang
    assert verify_ledger.main(["--db", str(tmp_path / "l.db"), "--anchor", str(tmp_path / "anchor.txt")]) == 1


def test_verify_ledger_detects_tampering(tmp_path):
    _ledger(tmp_path / "l.db").close()
    conn = sqlite3.connect(tmp_path / "l.db")
    conn.execute("DROP TRIGGER predictions_no_update")
    conn.execute("UPDATE predictions SET prob = 0.99 WHERE seq = 3")
    conn.commit()
    conn.close()
    out = tmp_path / "b"
    assert verify_ledger.main(["--db", str(tmp_path / "l.db"), "--out", str(out)]) == 1
    assert json.loads((out / "ledger.json").read_text())["message"] == "BROKEN"
    assert not (out / "ledger_head.txt").exists()  # head rusak tidak dipublikasikan


def test_psi_flags_shift():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(0, 1, 20_000), "b": rng.normal(0, 1, 20_000)})
    bins = psi_drift.reference_bins(train)
    live = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(1.0, 1, 3000)})
    rep = psi_drift.psi_report(bins, live, threshold=0.2)
    assert rep["features"]["a"] < 0.05 and rep["features"]["b"] > 0.2 and rep["drifted"] == ["b"]
    assert psi_drift.badge(rep)["color"] == "red"
