# Benchmark latency pipeline

Diukur dengan `python -m bench.bench_pipeline --minutes 120` · 66,140 pesan WebSocket sintetis,
120 keputusan · AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD (16 core) · Python 3.12.10.
Redis in-process (fakeredis); di produksi tambahkan RTT Redis. Telegram tidak termasuk.

## Per keputusan (tiap candle 1 menit close)

| Tahap | p50 (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|
| Fitur (240 mnt + EWM z) | 42.84 | 56.49 | 61.25 | 115 |
| Champion ONNX + isotonic | 1.46 | 2.10 | 2.81 | 115 |
| Challenger LightGBM | 1.27 | 2.02 | 3.16 | 115 |
| Hazard XGBoost (P≤5/10/15) | 1.31 | 2.18 | 2.52 | 115 |
| Tulis ledger (SHA-256 + SQLite) | 11.79 | 12.97 | 13.46 | 115 |
| Engine total per candle | 66.42 | 83.85 | 90.65 | 115 |
| **Kline close diterima → keputusan selesai** | **71.51** | **87.28** | **97.21** | 115 |

## Per pesan WebSocket (ingestion)

| Stream | p50 (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|
| `aggTrade` | 0.009 | 0.023 | 0.052 | 24,000 |
| `depth` | 0.291 | 0.445 | 0.790 | 7,200 |
| `depthSnapshot` | 0.066 | 0.066 | 0.066 | 1 |
| `forceOrder` | 0.028 | 0.070 | 0.092 | 19 |
| `fut_aggTrade` | 0.009 | 0.015 | 0.026 | 24,000 |
| `kline` | 0.011 | 0.018 | 0.026 | 3,480 |
| `kline_close` | 4.85 | 6.05 | 7.24 | 120 |
| `markPrice` | 0.011 | 0.025 | 0.042 | 7,200 |
| `openInterest` | 0.008 | 0.012 | 0.013 | 120 |
