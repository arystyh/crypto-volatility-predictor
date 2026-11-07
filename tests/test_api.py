"""API publik: ledger read-only, explorer, health/metrics, admin threshold, prediksi manual.

Ledger diisi lewat replay deterministik (champion + challenger + hazard per keputusan) lalu
diskor, kemudian service FastAPI dibuka di atas file ledger tersebut.
"""
import asyncio
import json

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from app import main as main_mod
from app.cache import store_candles
from app.config import get_settings
from app.engine import CHALLENGER, CHAMPION, HAZARD
from app.services import simticks
from app.services.ledger import Ledger, verify_rows
from app.services.replay import replay
from app.services.scorer import score_pending
from tests.helpers import SMALL_SPEC, settings
from training.synthetic import generate_candles


@pytest.fixture(scope="module")
def ledger_file(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    c = generate_candles("BTCUSDT", days=1, seed=3)
    hist, live = c.iloc[:700], c.iloc[700:730]
    res = asyncio.run(replay(simticks.generate(live, seed=1), settings(alert_prob_threshold=0.01), tmp / "l.db", hist,
                             spec=SMALL_SPEC))
    led = Ledger(str(tmp / "l.db"), max_log_lag_ms=10**15)
    rep = score_pending(led, c.iloc[:760][["open_time", "high", "low"]], "BTCUSDT")
    led.close()
    assert rep.scored > 0
    return tmp / "l.db", res, c


@pytest.fixture(scope="module")
def client(ledger_file, tmp_path_factory):
    path, _, c = ledger_file
    tmp = tmp_path_factory.mktemp("srv")
    mp = pytest.MonkeyPatch()
    for k, v in {"DATABASE_URL": f"sqlite:///{tmp}/t.db", "LEDGER_PATH": str(path), "DATA_DIR": str(tmp),
                 "RUN_ENGINE": "false", "RUN_MONITOR": "false", "RUN_INGESTION": "false", "HMM_GATE": "off",
                 "BASELINE_BOOTSTRAP": "false", "ADMIN_API_TOKEN": "rahasia", "TELEGRAM_BOT_TOKEN": "",
                 "SYMBOLS": "BTCUSDT"}.items():
        mp.setenv(k, v)
    get_settings.cache_clear()
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    main_mod.app.state.redis_override = redis
    with TestClient(main_mod.app) as cl:
        cl.portal.call(store_candles, redis, c.iloc[:400].to_dict(orient="records"), 1000)
        yield cl
    main_mod.app.state.redis_override = None
    mp.undo()
    get_settings.cache_clear()


def test_health_and_metrics(client):
    h = client.get("/health").json()
    assert h["checks"]["redis"]["ok"] and h["checks"]["model"]["ok"]
    assert h["breaker_state"] in ("OK", "DEGRADED", "STALE")
    assert h["checks"]["ledger"]["rows"] > 0
    prom = client.get("/metrics", params={"format": "prometheus"}).text
    assert "cvp_breaker_state" in prom and "cvp_alert_threshold" in prom


def test_ledger_rows_verify_independently(client, ledger_file):
    _, res, _ = ledger_file
    head = client.get("/ledger/head").json()
    assert head == {"head": res.ledger_head, "rows": res.ledger_rows}
    assert client.get("/ledger/verify").json()["ok"] is True
    rows, after = [], 0
    while True:  # paging seperti explorer / scripts/verify_ledger.py
        page = client.get("/ledger/rows", params={"after_seq": after, "limit": 17}).json()
        if not page["rows"]:
            break
        rows += page["rows"]
        after = page["next_after_seq"]
    assert len(rows) == res.ledger_rows
    assert verify_rows(rows) == (True, None)  # setelah round-trip JSON
    rows[5]["prob"] = min(1.0, rows[5]["prob"] + 0.01)
    assert verify_rows(rows) == (False, rows[5]["seq"])


def test_each_decision_has_champion_challenger_hazard(client):
    rows = client.get("/ledger/recent", params={"limit": 500}).json()["rows"]
    models = {r["model_version"] for r in rows}
    assert {CHAMPION, CHALLENGER, HAZARD} <= models
    champ = next(r for r in rows if r["model_version"] == CHAMPION)
    assert champ["status"] == "OK" and champ["conformal_alpha"] is None
    detail = client.get(f"/ledger/rows/{champ['seq']}").json()
    assert {s["model_version"] for s in detail["same_decision"]} == {CHAMPION, CHALLENGER, HAZARD}
    scored = [r for r in rows if r["label"] is not None]
    assert scored and all(r["z"] is not None for r in scored)
    only = client.get("/ledger/recent", params={"model": HAZARD, "limit": 5}).json()["rows"]
    assert only and all(r["model_version"] == HAZARD for r in only)
    assert client.get("/ledger/rows/999999").status_code == 404


def test_explorer_pages(client):
    for url in ("/", "/ledger", "/ledger/row/3"):
        r = client.get(url)
        assert r.status_code == 200 and "Forecast ledger" in r.text
    js = client.get("/static/ledger-verify.js")
    assert js.status_code == 200 and "pyRepr" in js.text
    assert "incidents" in client.get("/incidents").json()
    assert client.get("/incidents/..%2F..%2Fapp%2Fconfig.py").status_code == 404


def test_admin_threshold_requires_token(client):
    assert client.put("/config/threshold", json={"threshold": 0.3}).status_code == 403
    assert client.put("/config/threshold", json={"threshold": 0.3}, headers={"X-Admin-Token": "salah"}).status_code == 403
    assert client.put("/config/threshold", json={"threshold": 1.5}, headers={"X-Admin-Token": "rahasia"}).status_code == 422
    r = client.put("/config/threshold", json={"threshold": 0.3}, headers={"X-Admin-Token": "rahasia"})
    assert r.status_code == 200 and client.get("/config/threshold").json()["static_threshold"] == 0.3


def test_manual_predict_does_not_touch_ledger(client):
    before = client.get("/ledger/head").json()
    r = client.post("/predict", json={"symbol": "BTCUSDT"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability"] <= 1 and body["status"] in ("OK", "DEGRADED")
    assert client.get("/ledger/head").json() == before


def test_performance_and_model_endpoints(client):
    p = client.get("/performance").json()
    assert "threshold" in p and "metrics" in p
    m = client.get("/model").json()
    assert m["loaded"] and m["metadata"]["alert_prob_threshold"] > 0
    assert json.loads(json.dumps(client.get("/status").json()))["circuit_breaker"]["state"]
