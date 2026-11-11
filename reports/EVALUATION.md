# Evaluasi model volatility spike

- Dilatih 2026-11-11T12:38:55+00:00 · sumber data **binance** · simbol BTCUSDT
- Label: log-Parkinson vol 15 menit ke depan > 2.0 sd di atas baseline 30 hari (hanya masa lalu, `app/ml/target.py`). Threshold label ini TERPISAH dari threshold probabilitas alert.
- Split waktu train / val / holdout = 76,015 / 19,068 / 23,899 baris, jarak antar bagian 255 menit (TargetSpec.min_embargo(240) = H + 240).

## Ringkasan holdout

PR-AUC **0.254** vs base rate 0.037 (lift 6.8x) · median lead time **-49.0 mnt (1/46 episode)** · ROC-AUC 0.828

Threshold alert **0.5000**, dipilih di fold validasi: recall maksimum dengan precision >= 0.5 (val precision 0.50, recall 0.00). Di holdout: precision 1.00, recall 0.00 (TP 1, FP 0, FN 885, TN 23,013).

## Perbandingan dengan baseline

Semua model memakai holdout yang sama; threshold masing-masing dipilih di fold validasi dengan aturan yang sama.

| Model | PR-AUC | Lift vs acak | ROC-AUC | Brier | Median lead time @ threshold val |
|---|---|---|---|---|---|
| Persistence (vol sekarang berlanjut) | 0.496 | 13.4x | 0.895 | 0.0259 | 27.0 mnt (31/46 episode) |
| GARCH(1,1) | 0.479 | 12.9x | 0.897 | 0.0264 | -7.0 mnt (6/46 episode) |
| HAR-RV (15m/1j/1h) | 0.451 | 12.2x | 0.893 | 0.0262 | 35.5 mnt (30/46 episode) |
| Hazard P(onset ≤ 15m) | 0.380 | 10.2x | 0.864 | 0.0310 | 25.0 mnt (25/46 episode) |
| LightGBM (challenger) | 0.285 | 7.7x | 0.815 | 0.0313 | 17.0 mnt (2/46 episode) |
| XGBoost (champion, isotonic) | 0.254 | 6.8x | 0.828 | 0.0327 | -49.0 mnt (1/46 episode) |

## Kalibrasi isotonic

Brier score holdout: sebelum **0.03058** → sesudah **0.03265**. Kalibrator dilatih di fold validasi (bukan holdout).

![Reliability diagram](figures/reliability.png)

## Hazard: P(spike mulai ≤ 5/10/15 menit)

| Horizon | PR-AUC | Base rate |
|---|---|---|
| ≤5 menit | 0.442 | 0.0371 |
| ≤10 menit | 0.410 | 0.0371 |
| ≤15 menit | 0.380 | 0.0371 |

P(≤k) = 1 − Π(1 − h_j) sehingga monoton naik di k secara konstruksi (diuji).

- PR-AUC P(≤15) 0.380 vs model biner 0.254 → tidak lebih buruk
- Median lead time hazard 25.0 mnt (25/46 episode) vs biner -49.0 mnt (1/46 episode) → lebih baik
- MAE menit onset 3.39 vs tebakan naif tengah horizon 3.76

## Adaptive conformal vs threshold statis

Metrik: porsi alert yang ternyata bukan spike. Umpan balik tertunda H menit seperti live.

| Skenario | Kuartal | Alarm palsu statis | Alarm palsu ACI | Alert statis | Alert ACI | Recall statis | Recall ACI |
|---|---|---|---|---|---|---|---|
| Holdout apa adanya | Q1 | n/a | n/a | 0 | 0 | 0.00 | 0.00 |
| Holdout apa adanya | Q2 | n/a | n/a | 0 | 0 | 0.00 | 0.00 |
| Holdout apa adanya | Q3 | 0.00 | 0.00 | 1 | 1 | 0.00 | 0.00 |
| Holdout apa adanya | Q4 | n/a | n/a | 0 | 0 | 0.00 | 0.00 |
| Holdout apa adanya | **Total** | **0.00** | **0.00** | | | | |
| Pergantian rezim (logit +1.5 di paruh 2) | Q1 | n/a | n/a | 0 | 0 | 0.00 | 0.00 |
| Pergantian rezim (logit +1.5 di paruh 2) | Q2 | n/a | n/a | 0 | 0 | 0.00 | 0.00 |
| Pergantian rezim (logit +1.5 di paruh 2) | Q3 | 0.04 | 0.07 | 54 | 61 | 0.14 | 0.16 |
| Pergantian rezim (logit +1.5 di paruh 2) | Q4 | 0.22 | 0.20 | 9 | 10 | 0.03 | 0.04 |
| Pergantian rezim (logit +1.5 di paruh 2) | **Total** | **0.06** | **0.08** | | | | |

## Ablation kelompok fitur

| Kelompok fitur dibuang | ΔPR-AUC | Δ median lead (menit) | Kesimpulan |
|---|---|---|---|
| VPIN | +0.0191 | +48.0 | tidak berkontribusi -> kandidat dibuang |
| SWT (wavelet) | +0.0253 | +44.0 | tidak berkontribusi -> kandidat dibuang |
| OBI (order book) | n/a | n/a | n/a (fitur live-only, belum ada histori) |
| Hawkes | n/a | n/a | n/a (fitur live-only, belum ada histori) |
| Lead-lag perp→spot | n/a | n/a | n/a (fitur live-only, belum ada histori) |
| Funding/OI | n/a | n/a | n/a (fitur live-only, belum ada histori) |

## Purged walk-forward CV

| Fold | Train | Test | PR-AUC | Base rate |
|---|---|---|---|---|
| 1 | 15,634 | 15,889 | 0.004 | 0.0035 |
| 2 | 31,523 | 15,889 | 0.075 | 0.0158 |
| 3 | 47,412 | 15,889 | 0.197 | 0.0198 |
| 4 | 63,301 | 15,889 | 0.115 | 0.0062 |
| 5 | 79,190 | 15,893 | 0.130 | 0.0120 |
| **Rata-rata** | | | **0.104** | |

## Gate rezim HMM

| Gate | Model dipanggil | Precision | Recall | Median lead |
|---|---|---|---|---|
| off | 100% | 1.00 | 0.00 | -49.0 mnt (1/46 episode) |
| high | 62% | 1.00 | 0.00 | -49.0 mnt (1/46 episode) |
