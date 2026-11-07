# Benchmark latency pipeline

Diukur dengan `python -m bench.bench_pipeline --minutes 120` · 66,140 pesan WebSocket sintetis,
120 keputusan · Intel(R) Xeon(R) Processor @ 2.10GHz (4 core) · Python 3.13.18.
Redis in-process (fakeredis); di produksi tambahkan RTT Redis. Telegram tidak termasuk.

## Per keputusan (tiap candle 1 menit close)

| Tahap | p50 (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|
| Fitur (240 mnt + EWM z) | 27.23 | 37.10 | 43.04 | 115 |
| Champion ONNX + isotonic | 1.20 | 1.42 | 1.52 | 115 |
| Challenger LightGBM | 1.02 | 1.26 | 2.17 | 115 |
| Hazard XGBoost (P≤5/10/15) | 0.950 | 1.23 | 1.29 | 115 |
| Tulis ledger (SHA-256 + SQLite) | 2.27 | 3.14 | 5.92 | 115 |
| Engine total per candle | 36.92 | 48.30 | 53.62 | 115 |
| **Kline close diterima → keputusan selesai** | **39.42** | **52.66** | **68.14** | 115 |

## Per pesan WebSocket (ingestion)

| Stream | p50 (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|
| `aggTrade` | 0.004 | 0.016 | 0.043 | 24,000 |
| `depth` | 0.103 | 0.299 | 0.400 | 7,200 |
| `depthSnapshot` | 0.043 | 0.043 | 0.043 | 1 |
| `forceOrder` | 0.027 | 0.045 | 0.045 | 19 |
| `fut_aggTrade` | 0.004 | 0.008 | 0.020 | 24,000 |
| `kline` | 0.006 | 0.011 | 0.019 | 3,480 |
| `kline_close` | 2.97 | 4.55 | 5.82 | 120 |
| `markPrice` | 0.005 | 0.019 | 0.039 | 7,200 |
| `openInterest` | 0.006 | 0.008 | 0.011 | 120 |
