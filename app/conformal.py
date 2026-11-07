"""Adaptive Conformal alert threshold (Adaptive Conformal Inference, Gibbs & Candès 2021).

Threshold PROBABILITAS alert diadaptasi dari outcome ledger agar tingkat alarm palsu jangka
panjang tetap di target ALPHA walau rezim pasar dan kalibrasi model bergeser.

Mode "fdr" (default): err dihitung pada ALERT saja ->  porsi alert yang ternyata bukan spike.
    err_t     = 1 - label_t        bila prob_t >= tau_t (alert), selain itu tidak ada update
Mode "fpr": err dihitung pada outcome NEGATIF saja -> porsi menit non-spike yang memicu alert.
    err_t     = 1[prob_t >= tau_t] bila label_t = 0

    tau_{t+1} = clip(tau_t + gamma * (err_t - alpha), tau_min, tau_max)

Terlalu banyak alarm palsu -> tau naik; terlalu sedikit -> tau turun. Untuk T update berturut-turut,
|mean(err) - alpha| <= (tau_max - tau_min + gamma) / (gamma * T) tanpa asumsi distribusi.
Catatan mode fdr: bila tau begitu tinggi hingga tidak ada alert, tidak ada update (alarm palsu = 0,
target terpenuhi secara trivial); tau_max < 1 menjaga sistem tidak diam selamanya.

Umpan balik TERTUNDA: outcome diketahui H menit setelah keputusan; update dipanggil monitor ledger
saat prediksi matang, memakai tau yang berlaku SAAT keputusan (tersimpan di baris ledger).
alpha dan gamma juga disimpan di setiap baris ledger champion.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class AdaptiveConformalThreshold:
    tau: float = 0.50
    alpha: float = 0.20
    gamma: float = 0.01
    tau_min: float = 0.05
    tau_max: float = 0.95
    mode: str = "fdr"
    n_updates: int = 0
    n_errors: int = 0

    def update(self, prob: float, label: int, tau_at_decision: float | None = None) -> float:
        thr = self.tau if tau_at_decision is None else tau_at_decision
        alerted = prob >= thr
        if self.mode == "fdr":
            if not alerted:
                return self.tau
            err = 1.0 - float(int(label))
        else:
            if int(label) != 0:
                return self.tau
            err = 1.0 if alerted else 0.0
        self.n_updates += 1
        self.n_errors += int(err)
        self.tau = float(np.clip(self.tau + self.gamma * (err - self.alpha), self.tau_min, self.tau_max))
        return self.tau

    @property
    def realized_rate(self) -> float | None:
        return self.n_errors / self.n_updates if self.n_updates else None

    def dumps(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def loads(cls, raw: str | None, **defaults) -> "AdaptiveConformalThreshold":
        if not raw:
            return cls(**defaults)
        data = json.loads(raw)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def simulate_aci(prob: np.ndarray, label: np.ndarray, delay: int, tau0: float, alpha: float, gamma: float,
                 tau_min: float = 0.05, tau_max: float = 0.95, mode: str = "fdr") -> np.ndarray:
    """Simulasi ACI dengan umpan balik tertunda `delay` langkah. Return tau yang dipakai di tiap langkah."""
    aci = AdaptiveConformalThreshold(tau=tau0, alpha=alpha, gamma=gamma, tau_min=tau_min, tau_max=tau_max, mode=mode)
    taus = np.empty(len(prob))
    for t in range(len(prob)):
        taus[t] = aci.tau
        m = t - delay
        if m >= 0:
            aci.update(prob[m], label[m], taus[m])
    return taus


def false_alarm_rate(prob: np.ndarray, label: np.ndarray, taus: np.ndarray | float, mode: str = "fdr") -> float | None:
    alert = prob >= taus
    if mode == "fdr":
        return float((label[alert] == 0).mean()) if alert.any() else None
    neg = label == 0
    return float(alert[neg].mean()) if neg.any() else None
