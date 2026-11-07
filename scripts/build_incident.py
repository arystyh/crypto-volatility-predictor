"""Halaman insiden (roadmap 4.3): satu perintah membangun ulang laporan per peristiwa pasar.

Tiap halaman memuat harga, RV 15 menit, intensitas Hawkes (transaksi besar & likuidasi),
P(spike) champion + P(onset <= 5/10/15m) hazard, waktu alert, episode spike sebenarnya, dan
lead time. Semua angka dihitung dengan jalur fitur/model yang SAMA dengan produksi
(compute_features -> EWM z -> XGBoost -> isotonic -> threshold dari metadata model -> cooldown).

    python -m scripts.build_incident --event all                  # 5 peristiwa nyata (butuh akses Binance)
    python -m scripts.build_incident --event ftx
    python -m scripts.build_incident --start 2024-08-04 --end 2024-08-06 --name custom
    python -m scripts.build_incident --demo                       # 3 contoh dari data SINTETIS (offline)

Output: reports/incidents/<nama>.html + <nama>.json, dan reports/incidents/index.json (dipakai
halaman explorer /incidents). Peristiwa yang tidak bisa diunduh dilewati dengan pesan jelas.

Catatan:
- Hawkes butuh daftar transaksi. Untuk peristiwa nyata diambil dari arsip harian publik
  data.binance.vision (aggTrades spot; likuidasi dari liquidationSnapshot futures bila arsip
  tanggal itu ada). Untuk demo sintetis, waktu event disimulasikan dan diberi label jelas.
- Gate rezim HMM dan threshold adaptif (ACI) tidak dipakai di sini: threshold statis dari
  metadata model, supaya halaman bisa direproduksi tanpa state live.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.calibration import IsotonicCalibrator
from app.config import Settings
from app.hawkes import LIQ, TRADE, HawkesParams, OnlineHawkes
from app.hazard import HazardModel
from training.labels import SPEC
from app.model import VolatilityModel
from training.dataset import segment_features
from training.evaluate import episode_lead_times
from training.labels import make_labels

log = logging.getLogger("incident")
OUT = Path("reports/incidents")
WARMUP_DAYS = 31  # baseline target 30 hari + warm-up EWM
VISION = "https://data.binance.vision/data"


@dataclass(frozen=True)
class Event:
    name: str
    title: str
    start: str
    end: str
    note: str


EVENTS = {
    "covid-2020-03-12": Event("covid-2020-03-12", "Black Thursday COVID", "2020-03-11", "2020-03-14",
                              "BTC turun ~40% dalam sehari, likuidasi masif BitMEX/Binance."),
    "china-ban-2021-05-19": Event("china-ban-2021-05-19", "Larangan kripto China", "2021-05-18", "2021-05-21",
                                  "Pengumuman larangan layanan kripto oleh asosiasi keuangan China."),
    "luna-2022-05": Event("luna-2022-05", "Kolaps LUNA/UST", "2022-05-08", "2022-05-13",
                          "Depeg UST, penjualan cadangan BTC Luna Foundation Guard."),
    "ftx-2022-11-08": Event("ftx-2022-11-08", "Kolaps FTX", "2022-11-07", "2022-11-11",
                            "Bank run FTX setelah Binance batal mengakuisisi."),
    "yen-carry-2024-08-05": Event("yen-carry-2024-08-05", "Unwind yen carry trade", "2024-08-04", "2024-08-06",
                                  "Aksi jual aset berisiko global setelah BoJ menaikkan suku bunga."),
}
ALIASES = {"covid": "covid-2020-03-12", "china": "china-ban-2021-05-19", "luna": "luna-2022-05",
           "ftx": "ftx-2022-11-08", "yen-carry": "yen-carry-2024-08-05"}


def _ms(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)


# ---------------------------------------------------------------------------- data
def binance_candles(symbol: str, start_ms: int, end_ms: int, rest_url: str) -> pd.DataFrame:
    from ingestion.binance import download_history

    rows = download_history(rest_url, symbol, "1m", start_ms - WARMUP_DAYS * 86_400_000, end_ms)
    return pd.DataFrame(rows)


def _vision_csv(url: str) -> pd.DataFrame | None:
    import httpx

    r = httpx.get(url, timeout=120, follow_redirects=True)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, header=None)


def binance_events(symbol: str, start_ms: int, end_ms: int, large_usd: float) -> tuple[np.ndarray, np.ndarray]:
    """(waktu detik, jenis) transaksi besar spot + likuidasi futures dari arsip harian publik."""
    times, kinds = [], []
    day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    last = datetime.fromtimestamp((end_ms - 1) / 1000, timezone.utc).date()
    while day <= last:
        d = day.isoformat()
        tr = _vision_csv(f"{VISION}/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{d}.zip")
        if tr is not None:
            if not str(tr.iloc[0, 0]).isdigit():  # header di arsip baru
                tr = tr.iloc[1:].astype(float)
            notional = tr[1].astype(float) * tr[2].astype(float)
            ts = tr[5].astype(float)
            ts = np.where(ts > 1e14, ts / 1e6, ts / 1e3)  # arsip 2025+: mikrodetik
            sel = notional.to_numpy() >= large_usd
            times.append(ts[sel])
            kinds.append(np.full(sel.sum(), TRADE))
        liq = _vision_csv(f"{VISION}/futures/um/daily/liquidationSnapshot/{symbol}/{symbol}-liquidationSnapshot-{d}.zip")
        if liq is not None:
            if not str(liq.iloc[0, 0]).isdigit():
                liq = liq.iloc[1:]
            ts = liq[0].astype(float).to_numpy() / 1e3
            times.append(ts)
            kinds.append(np.full(len(ts), LIQ))
        day += timedelta(days=1)
    if not times:
        return np.array([]), np.array([], dtype=int)
    t, k = np.concatenate(times), np.concatenate(kinds).astype(int)
    order = np.argsort(t, kind="mergesort")
    return t[order], k[order]


def synthetic_events(candles: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """SIMULASI (demo): transaksi besar ~ Poisson sebanding volume, likuidasi ~ kuadrat range candle."""
    rng = np.random.default_rng(seed)
    vol = candles["volume"].to_numpy(float)
    rng_ = ((candles["high"] - candles["low"]) / candles["close"]).to_numpy(float)
    lam_t = 0.6 * vol / np.median(vol)
    lam_l = 0.05 * (rng_ / np.median(rng_)) ** 2
    t, k = [], []
    for t0, a, b in zip(candles["open_time"].to_numpy() / 1000, lam_t, lam_l):
        for kind, lam in ((TRADE, a), (LIQ, b)):
            n = rng.poisson(min(lam, 200))
            t.append(t0 + np.sort(rng.uniform(0, 60, n)))
            k.append(np.full(n, kind))
    t, k = np.concatenate(t), np.concatenate(k).astype(int)
    order = np.argsort(t, kind="mergesort")
    return t[order], k[order]


def hawkes_per_minute(times: np.ndarray, kinds: np.ndarray, minute_close_s: np.ndarray,
                      params: HawkesParams) -> pd.DataFrame:
    """Intensitas di akhir tiap menit (sama dengan fitur live yang dibaca saat candle close)."""
    h = OnlineHawkes(params)
    out = np.full((len(minute_close_s), 2), np.nan)
    i = 0
    for j, t_end in enumerate(minute_close_s):
        while i < len(times) and times[i] < t_end:
            h.add_event(float(times[i]), int(kinds[i]))
            i += 1
        out[j] = (h.intensity(t_end, TRADE), h.intensity(t_end, LIQ) if params.dim > 1 else np.nan)
    return pd.DataFrame(out, columns=["hawkes_trade", "hawkes_liq"])


# ---------------------------------------------------------------------------- model
class Models:
    def __init__(self, models_dir: str = "models"):
        md = Path(models_dir)
        self.model = VolatilityModel(md / "xgb_volatility.json", backend="auto")
        self.meta = self.model.metadata
        self.cal = IsotonicCalibrator.load(md / "calibration.json") if (md / "calibration.json").exists() else None
        self.hazard = HazardModel(md / "hazard_xgb.json", self.model.features) if (md / "hazard_xgb.json").exists() else None
        hp = md / "hawkes_params.json"
        self.hawkes = HawkesParams.load(hp) if hp.exists() else HawkesParams()
        self.threshold = float(self.meta.get("alert_prob_threshold") or 0.5)

    def score(self, candles: pd.DataFrame) -> pd.DataFrame:
        c = candles.sort_values("open_time").reset_index(drop=True)
        feats = segment_features(c, self.meta.get("ewm_halflife_min", 1440))
        labels = make_labels(c, SPEC)
        frame = pd.concat([c[["open_time", "open", "high", "low", "close", "volume"]], feats, labels], axis=1)
        ok = frame[self.model.features].notna().all(axis=1).to_numpy()
        prob = np.full(len(frame), np.nan)
        X = frame.loc[ok, self.model.features]
        if len(X):
            p = self.model.predict_proba(X)
            prob[ok] = self.cal(p) if self.cal is not None else p
        frame["prob"] = prob
        if self.hazard is not None and len(X):
            cum = 1 - np.cumprod(1 - self.hazard.predict_hazard_matrix(X.to_numpy(np.float32)), axis=1)
            for k in (5, 10, 15):
                frame[f"p_le_{k}"] = np.nan
                frame.loc[ok, f"p_le_{k}"] = cum[:, k - 1]
        r = np.log(frame["close"]).diff()
        frame["rv_15_ann"] = np.sqrt((r ** 2).rolling(15).sum() * 525_600 / 15) * 100
        return frame


def alerts_with_cooldown(prob: np.ndarray, threshold: float, cooldown_min: int) -> np.ndarray:
    out = np.zeros(len(prob), dtype=int)
    last = -10**9
    for i, p in enumerate(prob):
        if np.isfinite(p) and p >= threshold and i - last >= cooldown_min:
            out[i], last = 1, i
    return out


# ---------------------------------------------------------------------------- laporan
def _figure(win: pd.DataFrame, threshold: float, title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    t = pd.to_datetime(win["open_time"] + 60_000, unit="ms", utc=True)  # waktu keputusan = close candle
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    spikes = win["label"].fillna(0).to_numpy() == 1
    for a in ax:
        a.fill_between(t, 0, 1, where=spikes, transform=a.get_xaxis_transform(), color="#f2b8b5", alpha=0.45,
                       linewidth=0, label="spike sebenarnya (label)" if a is ax[0] else None)
        for x in t[win["alert"].to_numpy() == 1]:
            a.axvline(x, color="#b42318", linewidth=0.8, alpha=0.8)
        a.grid(alpha=0.25)
    ax[0].plot(t, win["close"], color="#1d1d1b", linewidth=0.9)
    ax[0].set_ylabel("Harga (USDT)")
    ax[0].plot([], [], color="#b42318", linewidth=0.8, label="alert")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title(title, loc="left", fontsize=12)
    ax[1].plot(t, win["rv_15_ann"], color="#2f5bd3", linewidth=0.9)
    ax[1].set_ylabel("RV 15m\n(% tahunan)")
    ax[2].plot(t, win["hawkes_trade"], color="#a15c00", linewidth=0.9, label="transaksi besar")
    if win["hawkes_liq"].notna().any():
        ax[2].plot(t, win["hawkes_liq"], color="#7a3cc3", linewidth=0.9, label="likuidasi")
    ax[2].set_ylabel("Hawkes λ\n(event/detik)")
    ax[2].set_yscale("symlog", linthresh=0.05)
    ax[2].legend(loc="upper left", fontsize=8)
    ax[3].plot(t, win["prob"], color="#1a7f4b", linewidth=0.9, label="P(spike 15m)")
    if "p_le_5" in win:
        ax[3].plot(t, win["p_le_5"], color="#6b6b66", linewidth=0.7, alpha=0.8, label="P(onset ≤ 5m)")
    ax[3].axhline(threshold, color="#b42318", linestyle="--", linewidth=0.8, label=f"threshold {threshold:.3f}")
    ax[3].set_ylim(0, 1)
    ax[3].set_ylabel("Probabilitas")
    ax[3].legend(loc="upper left", fontsize=8)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


PAGE = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>
:root{{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--warn:#a15c00}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#a3a29b;--line:#33322e;--warn:#f5b041}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 48px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px;overflow-x:auto}}
img{{max-width:100%;height:auto;background:#fff;border-radius:6px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:4px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;font-size:.9rem}}
th{{color:var(--muted)}}.muted{{color:var(--muted)}}.warn{{color:var(--warn);font-weight:600}}
code{{font-size:.85rem}}a{{color:#2f5bd3}}
</style></head><body><main>
<p><a href="/">← ledger explorer</a></p>
<h1>{title}</h1>
<p class="muted">{window} UTC · {symbol} · model <code>{version}</code> · data {source}</p>
{banner}
<section><b>{summary}</b><p class="muted" style="margin:6px 0 0">{note}</p></section>
<section><img alt="Harga, RV, Hawkes dan P(spike) selama peristiwa" src="data:image/png;base64,{img}"></section>
<section><h2>Episode spike</h2><table><tr><th>Mulai (UTC)</th><th>Selesai</th><th>Alert pertama</th><th>Lead time</th></tr>{episodes}</table>
<p class="muted">Lead time = (awal episode + {h} menit) − alert pertama, definisi sama dengan skor ledger live: berapa menit
lebih awal dari "melihat volatilitas di chart". Alert dihitung dalam {lookback} menit sebelum awal episode sampai akhir episode.</p></section>
<section><h2>Alert ({n_alerts})</h2><table><tr><th>Waktu keputusan (UTC)</th><th>P(spike 15m)</th><th>P(≤5m)</th><th>P(≤10m)</th><th>P(≤15m)</th><th>Harga</th><th>Hasil</th></tr>{alerts}</table>
<p class="muted">Threshold {threshold:.4f} dari kurva PR validasi (metadata model), cooldown {cooldown} menit, tanpa gate HMM/ACI.</p></section>
<p class="muted">Bangun ulang: <code>{command}</code></p>
</main></body></html>"""


def _fmt_t(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def build(name: str, title: str, note: str, candles: pd.DataFrame, events: tuple[np.ndarray, np.ndarray],
          start_ms: int, end_ms: int, models: Models, cooldown: int, source: str, command: str,
          symbol: str = "BTCUSDT", out_dir: Path = OUT) -> dict:
    frame = models.score(candles)
    close_s = (frame["open_time"].to_numpy() + 60_000) / 1000
    frame = pd.concat([frame, hawkes_per_minute(*events, close_s, models.hawkes)], axis=1)
    win = frame[(frame["open_time"] >= start_ms) & (frame["open_time"] < end_ms)].reset_index(drop=True)
    if win["prob"].notna().sum() == 0:
        raise RuntimeError("tidak ada baris dengan fitur lengkap di jendela peristiwa")
    win["alert"] = alerts_with_cooldown(win["prob"].to_numpy(), models.threshold, cooldown)
    lab = win.dropna(subset=["label"]).assign(symbol=symbol)
    lead = episode_lead_times(lab, lab["alert"].to_numpy(), SPEC.horizon)

    # tabel episode
    ep_rows = []
    pos = lab.index[lab["label"] == 1].to_numpy()
    if len(pos):
        ts = lab["open_time"].to_numpy()
        lpos = np.flatnonzero(lab["label"].to_numpy() == 1)
        br = np.flatnonzero(np.diff(ts[lpos]) > 60_000)
        for s, e in zip(np.r_[lpos[0], lpos[br + 1]], np.r_[lpos[br], lpos[-1]]):
            m = (lab["alert"].to_numpy() == 1) & (ts >= ts[s] - 30 * 60_000) & (ts <= ts[e])
            first = ts[m].min() if m.any() else None
            lead_m = (ts[s] + SPEC.horizon * 60_000 - first) / 60_000 if first is not None else None
            ep_rows.append({"start": int(ts[s]), "end": int(ts[e]), "first_alert": None if first is None else int(first),
                            "lead_min": lead_m})
    alerts = win[win["alert"] == 1]
    alert_rows = [{"decision_ts": int(r.open_time) + 60_000, "prob": float(r.prob),
                   **{f"p_le_{k}": (float(getattr(r, f"p_le_{k}")) if hasattr(r, f"p_le_{k}") else None) for k in (5, 10, 15)},
                   "close": float(r.close), "label": None if pd.isna(r.label) else int(r.label)} for r in alerts.itertuples()]
    synthetic = source == "synthetic"
    summary = {
        "name": name, "title": title, "date": _fmt_t(start_ms)[:10], "page": f"{name}.html", "synthetic": synthetic,
        "window": [start_ms, end_ms], "model_version": models.meta.get("version"), "threshold": models.threshold,
        "alerts": len(alert_rows), "episodes": lead["episodes"], "caught": lead["caught"],
        "lead_min": lead["median_lead_min"], "max_rv_15_ann": float(win["rv_15_ann"].max()),
        "max_prob": float(win["prob"].max()), "hawkes_events": int(len(events[0])), "command": command,
        "episode_table": ep_rows, "alert_table": alert_rows,
    }
    pct = lambda x: "–" if x is None or not np.isfinite(x) else f"{x * 100:.1f}%"  # noqa: E731
    ep_html = "".join(
        f"<tr><td>{_fmt_t(e['start'])}</td><td>{_fmt_t(e['end'])}</td>"
        f"<td>{_fmt_t(e['first_alert']) if e['first_alert'] else '<span class=muted>terlewat</span>'}</td>"
        f"<td>{'' if e['lead_min'] is None else f'{e['lead_min']:.0f} mnt'}</td></tr>" for e in ep_rows) \
        or "<tr><td colspan=4 class=muted>Tidak ada spike berlabel di jendela ini.</td></tr>"
    al_html = "".join(
        f"<tr><td>{_fmt_t(a['decision_ts'])}</td><td>{pct(a['prob'])}</td><td>{pct(a['p_le_5'])}</td>"
        f"<td>{pct(a['p_le_10'])}</td><td>{pct(a['p_le_15'])}</td><td>{a['close']:,.0f}</td>"
        f"<td>{'spike' if a['label'] == 1 else 'normal' if a['label'] == 0 else '–'}</td></tr>" for a in alert_rows) \
        or "<tr><td colspan=7 class=muted>Tidak ada alert.</td></tr>"
    lead_txt = (f"median lead time {lead['median_lead_min']:.0f} menit" if lead["median_lead_min"] is not None
                else "tidak ada lead time (tidak ada episode yang didahului alert)")
    text = (f"{len(alert_rows)} alert · {lead['caught']}/{lead['episodes']} episode spike didahului alert · {lead_txt}")
    banner = ('<section class="warn">⚠️ DATA SINTETIS (demo pipeline). Harga, transaksi dan likuidasi disimulasikan; '
              'ini BUKAN rekaman peristiwa nyata. Bangun halaman peristiwa nyata dengan '
              '<code>python -m scripts.build_incident --event all</code> di mesin yang bisa mengakses Binance.</section>'
              if synthetic else "")
    page = PAGE.format(
        title=html.escape(title), window=f"{_fmt_t(start_ms)} – {_fmt_t(end_ms)}", symbol=symbol,
        version=html.escape(str(models.meta.get("version"))), source="SINTETIS" if synthetic else "Binance",
        banner=banner, summary=html.escape(text), note=html.escape(note),
        img=_figure(win, models.threshold, title), episodes=ep_html, h=SPEC.horizon, lookback=30,
        n_alerts=len(alert_rows), alerts=al_html, threshold=models.threshold, cooldown=cooldown,
        command=html.escape(command))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.html").write_text(page)
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def update_index(entries: list[dict], out_dir: Path = OUT) -> None:
    idx_path = out_dir / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"incidents": []}
    by_name = {i["name"]: i for i in idx["incidents"]}
    keys = ("name", "title", "date", "page", "synthetic", "alerts", "episodes", "caught", "lead_min", "model_version")
    for e in entries:
        by_name[e["name"]] = {k: e.get(k) for k in keys}
    idx["incidents"] = sorted(by_name.values(), key=lambda i: (i["synthetic"], i["date"]))
    idx_path.write_text(json.dumps(idx, indent=2))


def demo(models: Models, cooldown: int, n: int = 3, seed: int = 11) -> list[dict]:
    """3 peristiwa dari data sintetis seed BERBEDA dengan data latih (out-of-sample)."""
    from training.synthetic import generate_candles

    c = generate_candles("BTCUSDT", days=45, seed=seed)
    labels = make_labels(c, SPEC)
    z = labels["z"].to_numpy()
    tail = np.arange(len(c)) >= len(c) - 14 * 1440
    picked: list[int] = []
    for i in np.argsort(-np.nan_to_num(np.where(tail, z, -np.inf), nan=-np.inf)):
        if not np.isfinite(z[i]) or len(picked) == n:
            break
        if all(abs(i - j) > 1440 for j in picked) and i < len(c) - 12 * 60:
            picked.append(int(i))
    events = synthetic_events(c, seed)
    out = []
    for k, i in enumerate(sorted(picked), 1):
        t_peak = int(c["open_time"].iloc[i])
        start, end = t_peak - 6 * 3_600_000, t_peak + 4 * 3_600_000
        lo = int(np.searchsorted(c["open_time"].to_numpy(), start - (SPEC.baseline_window + 300) * 60_000))
        sub = c.iloc[lo: int(np.searchsorted(c["open_time"].to_numpy(), end)) + SPEC.horizon]
        name = f"demo-sintetis-{k}"
        out.append(build(name, f"Demo sintetis #{k}: lonjakan volatilitas {_fmt_t(t_peak)}",
                         "Contoh pipeline pada data sintetis (rezim tenang→bergejolak).", sub, events, start, end,
                         models, cooldown, "synthetic", "python -m scripts.build_incident --demo"))
        log.info("%s: %s", name, {k2: out[-1][k2] for k2 in ("alerts", "episodes", "caught", "lead_min")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", help=f"all | {' | '.join(list(ALIASES) + list(EVENTS))}")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--name")
    ap.add_argument("--demo", action="store_true", help="contoh dari data sintetis (offline)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--models", default="models")
    ap.add_argument("--no-trades", action="store_true", help="lewati unduhan aggTrades (panel Hawkes kosong)")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    s = Settings()
    models = Models(args.models)
    built: list[dict] = []
    if args.demo:
        built += demo(models, s.alert_cooldown_minutes)
    todo: list[Event] = []
    if args.event == "all":
        todo = list(EVENTS.values())
    elif args.event:
        todo = [EVENTS[ALIASES.get(args.event, args.event)]]
    elif args.start and args.end:
        todo = [Event(args.name or f"custom-{args.start}", args.name or f"{args.start} – {args.end}", args.start,
                      args.end, "Jendela kustom.")]
    failed = []
    for ev in todo:
        start, end = _ms(ev.start), _ms(ev.end) + 86_400_000
        try:
            log.info("%s: unduh candle %s + %d hari warm-up...", ev.name, ev.start, WARMUP_DAYS)
            candles = binance_candles(args.symbol, start, end + SPEC.horizon * 60_000, s.binance_rest_url)
            events = (np.array([]), np.array([], dtype=int)) if args.no_trades else \
                binance_events(args.symbol, start, end, s.large_trade_usd)
            cmd = f"python -m scripts.build_incident --event {ev.name}"
            built.append(build(ev.name, ev.title, ev.note, candles, events, start, end, models,
                               s.alert_cooldown_minutes, "binance", cmd, args.symbol))
            log.info("%s selesai: %s", ev.name, {k: built[-1][k] for k in ("alerts", "episodes", "caught", "lead_min")})
        except Exception as exc:  # noqa: BLE001 - lanjutkan peristiwa lain
            failed.append(ev.name)
            log.error("%s GAGAL: %s", ev.name, exc)
    if built:
        update_index(built)
    if not built and not failed:
        ap.print_help()
    return 1 if failed and not built else 0


if __name__ == "__main__":
    raise SystemExit(main())
