"""Roadmap 4.5: verifikasi hash chain di browser HARUS menghasilkan hash yang sama dengan Python.

app/static/ledger-verify.js dijalankan di Node (mesin JS yang sama dengan browser) atas baris
ledger yang sudah melewati JSON, termasuk float yang repr()-nya rumit.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.ledger import GENESIS, Ledger, Prediction, row_hash

JS = Path(__file__).resolve().parents[1] / "app" / "static" / "ledger-verify.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node tidak terpasang")

RUNNER = """
const L = require(process.argv[1]);
const inp = JSON.parse(require("fs").readFileSync(0, "utf8"));
(async () => {
  const out = {repr: inp.floats.map(L.pyRepr), js: [], subtle: []};
  let prev = L.GENESIS;
  for (const r of inp.rows) {
    out.js.push(L.sha256js(new TextEncoder().encode(prev + L.canonical(r))));
    out.subtle.push(await L.rowHash(prev, r));
    prev = r.row_hash;
  }
  console.log(JSON.stringify(out));
})();
"""

FLOATS = [0.1, 0.5, 1.0, 123.0, 1e-05, 1.5e-05, 0.0001, 0.00012345, 1e16, 1.2345e16, 9999999999999998.0,
          123456789.123, 5e-324, 1.7976931348623157e308, -0.0, 0.0, -2.5e-07, 0.07142857142857142, 2.0 / 3]


def _node(payload: dict) -> dict:
    out = subprocess.run(["node", "-e", RUNNER, str(JS)], input=json.dumps(payload), capture_output=True, text=True,
                         check=True, timeout=30)
    return json.loads(out.stdout)


def test_js_matches_python_hash(tmp_path):
    led = Ledger(str(tmp_path / "l.db"), max_log_lag_ms=10**15, clock=lambda: 1_700_000_000_123)
    probs = [0.1, 1e-05, 0.99999, 1.0, 0.0, 2.0 / 3, 0.0001]
    for i, p in enumerate(probs):
        led.log_prediction(Prediction("BTCUSDT", 1_700_000_000_000 + i, 15, "champion_xgboost", p, 0.0714,
                                      -7.25, 1.0, 2.0, status=("OK", "STALE", "DEGRADED")[i % 3],
                                      conformal_alpha=0.2 if i % 2 else None, conformal_gamma=1e-05 if i % 2 else None))
    rows = json.loads(json.dumps(led.export()))  # seperti /ledger/rows
    res = _node({"floats": FLOATS, "rows": rows})
    assert res["repr"] == [repr(x) for x in FLOATS]
    want = [r["row_hash"] for r in rows]
    assert res["js"] == want and res["subtle"] == want
    assert want[0] == row_hash(GENESIS, rows[0])


def test_js_detects_tampering(tmp_path):
    led = Ledger(str(tmp_path / "l.db"), max_log_lag_ms=10**15, clock=lambda: 10)
    for i in range(3):
        led.log_prediction(Prediction("BTCUSDT", i, 15, "champion_xgboost", 0.3, 0.5, 0.0, 1.0, 2.0))
    rows = json.loads(json.dumps(led.export()))
    rows[1]["prob"] = 0.9
    res = _node({"floats": [], "rows": rows})
    assert res["js"][0] == rows[0]["row_hash"] and res["js"][1] != rows[1]["row_hash"]
