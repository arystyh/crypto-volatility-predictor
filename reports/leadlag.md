# Lead-lag perp → spot (BTCUSDT)

Sumber: synthetic · 10,798 detik return 1 detik (harga transaksi terakhir per detik).

> ⚠️ Data **sintetis**: generator sengaja membuat perp memimpin 2 detik. Ini menguji bahwa pengukurannya bekerja, BUKAN temuan pasar. Jalankan `--ticks` pada rekaman ≥ 24 jam.

Perp **memimpin spot 2 detik** (korelasi 0.925 vs 0.816 pada lag 0), signifikan (> 3 standar error).

corr(return perp(t), return spot(t + lag)); lag > 0 berarti perp bergerak lebih dulu.

| Lag (detik) | Korelasi |
|---|---|
| -10 | +0.134 |
| -9 | +0.203 |
| -8 | +0.270 |
| -7 | +0.337 |
| -6 | +0.407 |
| -5 | +0.476 |
| -4 | +0.540 |
| -3 | +0.608 |
| -2 | +0.678 |
| -1 | +0.744 |
| +0 | +0.816 |
| +1 | +0.881 |
| +2 | +0.925 | ← terbaik
| +3 | +0.881 |
| +4 | +0.814 |
| +5 | +0.745 |
| +6 | +0.678 |
| +7 | +0.611 |
| +8 | +0.540 |
| +9 | +0.475 |
| +10 | +0.404 |

Status fitur `perp_lead_ret_*`, `basis_bps`: **live-only**, tidak dipakai model sampai terbukti di data nyata.
