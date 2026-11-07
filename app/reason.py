"""Alasan alert yang bisa dibaca manusia (dikirim di pesan Telegram & disimpan di log).

Metode: setiap fitur dibandingkan dengan distribusinya di data training
(models/feature_quantiles.json). Fitur yang berada di ekor ekstrem (>= p95 atau <= p5)
dan masuk daftar fitur yang bisa dijelaskan, diurutkan dari yang paling ekstrem.
Ini penjelasan deskriptif ("apa yang tidak biasa sekarang"), bukan atribusi SHAP.
Ditambah konteks: kurva hazard (kapan spike diperkirakan mulai), rezim HMM, arah
semi-variance, dan aliran likuidasi (Hawkes).
"""
from __future__ import annotations

import json
from pathlib import Path

# fitur -> (frasa saat tinggi, frasa saat rendah)
PHRASES: dict[str, tuple[str, str | None]] = {
    "rv_ratio_15_240": ("volatilitas 15m {v:.1f}x di atas rata-rata 4 jam", "volatilitas terkompresi ({v:.2f}x vs 4 jam)"),
    "rv_ratio_5_60": ("volatilitas 5m melonjak {v:.1f}x vs 1 jam", None),
    "parkinson_ratio_15_240": ("range high-low 15m {v:.1f}x lebih lebar dari biasanya", None),
    "atr_ratio_14_60": ("ATR naik {v:.1f}x", None),
    "volume_surge": ("volume {v:.1f}x SMA-20", None),
    "volume_surge_5": ("volume 5m {v:.1f}x SMA-20", None),
    "vpin_50": ("VPIN tinggi ({v:.2f}): aliran order satu arah/toxic", None),
    "vpin_ratio_15_240": ("toksisitas order flow naik {v:.1f}x", None),
    "ofi_15": ("tekanan beli agresif (OFI {v:+.2f})", "tekanan jual agresif (OFI {v:+.2f})"),
    "downside_share_15": ("volatilitas didominasi pergerakan turun ({v:.0%})", "volatilitas didominasi pergerakan naik"),
    "swt_noise_ratio_15": ("pergerakan acak/whipsaw tinggi", "pergerakan terarah (noise rendah)"),
    "rv_15_ewmz": ("volatilitas {v:+.1f} sd di atas norma adaptif", None),
    "volume_surge_5_ewmz": ("volume {v:+.1f} sd di atas norma adaptif", None),
    "log_ret_15": ("harga naik tajam 15m ({v:+.2%})", "harga turun tajam 15m ({v:+.2%})"),
}


class ReasonBuilder:
    def __init__(self, quantiles_path: str | Path = "models/feature_quantiles.json"):
        p = Path(quantiles_path)
        self.q: dict = json.loads(p.read_text()) if p.exists() else {}

    def _extremeness(self, f: str, v: float) -> tuple[float, str] | None:
        q = self.q.get(f)
        if not q:
            return None
        p5, p50, p95 = q["0.05"], q["0.5"], q["0.95"]
        p1, p99 = q.get("0.01", p5), q.get("0.99", p95)
        if v >= p95 and p95 > p50:
            return 1.0 + (v - p95) / max(p99 - p95, 1e-12), "high"
        if v <= p5 and p5 < p50:
            return 1.0 + (p5 - v) / max(p5 - p1, 1e-12), "low"
        return None

    def feature_reasons(self, features: dict, top: int = 3) -> list[str]:
        scored = []
        for f, (hi, lo) in PHRASES.items():
            if f not in features:
                continue
            ext = self._extremeness(f, features[f])
            if not ext:
                continue
            score, side = ext
            phrase = hi if side == "high" else lo
            if phrase:
                scored.append((score, phrase.format(v=features[f])))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top]]

    def build(self, features: dict, hazard: dict | None = None, regime: str | None = None,
              direction: str | None = None, micro: dict | None = None, mode: str = "model") -> list[str]:
        reasons = self.feature_reasons(features)
        if hazard and hazard.get("median_onset_min"):
            reasons.append(f"model hazard: spike paling mungkin mulai ~{hazard['most_likely_min']} menit lagi "
                           f"(median {hazard['median_onset_min']} menit)")
        elif hazard:
            reasons.append(f"model hazard: P(spike mulai ≤15 menit) {hazard['p_onset_within_h']:.0%}, "
                           f"puncak di menit ke-{hazard['most_likely_min']}")
        if regime == "high_vol_expansion":
            reasons.append("HMM: pasar dalam rezim High Volatility / Expansion")
        m = micro or {}
        if (m.get("hawkes_excitation") or 0) >= 3:
            side = ""
            if m.get("liq_long_share_1m") is not None:
                side = " (mayoritas posisi long)" if m["liq_long_share_1m"] >= 0.6 else (
                    " (mayoritas posisi short)" if m["liq_long_share_1m"] <= 0.4 else "")
            reasons.append(f"likuidasi beruntun: intensitas Hawkes {m['hawkes_excitation']:.1f}x normal, "
                           f"branching {m.get('hawkes_branching', 0):.2f}{side}")
        if mode == "fallback":
            reasons.insert(0, "model ML gagal: keputusan dari heuristic Parkinson (fallback)")
        if not reasons:
            reasons.append("kombinasi fitur di atas threshold model (tidak ada satu fitur yang ekstrem)")
        return reasons
