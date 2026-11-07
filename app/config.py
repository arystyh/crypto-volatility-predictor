"""Konfigurasi terpusat, dibaca dari environment variable / file .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # --- Redis (lokal: redis://service-redis:6379/0, Upstash: rediss://default:<pwd>@<host>:6379) ---
    redis_url: str = "redis://localhost:6379/0"
    stream_key: str = "crypto_stream"     # Redis Stream (XADD + consumer group)
    stream_maxlen: int = 5000             # panjang maksimum stream (XADD MAXLEN ~)
    stream_consumer: str = "engine-1"     # nama consumer (tetap, agar pending bisa diambil lagi setelah restart)
    stream_block_ms: int = 15000          # XREADGROUP BLOCK (lebih besar = lebih hemat kuota Upstash)
    stream_claim_idle_ms: int = 60000     # XAUTOCLAIM pesan consumer lain yang menggantung lebih lama dari ini
    cache_size: int = 400                 # jumlah candle terakhir per simbol (fitur butuh >= 260)

    # --- Binance ---
    # Hanya BTCUSDT sampai performanya terbukti di ledger (roadmap: "sengaja tidak dikerjakan").
    symbols: str = "BTCUSDT"
    interval: str = "1m"
    enable_aggtrade: bool = True      # @aggTrade -> trade flow, transaksi besar (Hawkes)
    enable_depth: bool = True         # @depth@100ms + snapshot REST -> order book lokal -> OBI 5 level
    enable_futures: bool = True       # forceOrder, markPrice@1s, futures aggTrade, open interest
    large_trade_usd: float = 100_000  # ambang "transaksi besar" untuk Hawkes
    # Endpoint market-data-only (binance.vision) tidak diblokir di region AS (Render/Railway).
    binance_ws_url: str = "wss://data-stream.binance.vision"
    binance_rest_url: str = "https://data-api.binance.vision"
    binance_futures_ws_url: str = "wss://fstream.binance.com"     # diblokir dari IP AS (HTTP 451)
    binance_futures_rest_url: str = "https://fapi.binance.com"
    record_ticks: bool = False        # rekam semua pesan mentah ke Parquet (replay & parity test)
    record_dir: str = "data/ticks"

    # --- Storage ---
    database_url: str = "sqlite:///./data/volatility.db"

    # --- Model & keputusan ---
    model_path: str = "models/xgb_volatility.json"
    # Threshold PROBABILITAS alert. Kosong = dipilih dari kurva precision-recall validasi
    # (models/metadata.json -> alert_prob_threshold). Bukan threshold label (Z > 2).
    alert_prob_threshold: float | None = None
    alert_cooldown_minutes: int = 15
    model_timeout_sec: float = 2.0    # lewat batas ini -> DEGRADED, skor dari heuristic Parkinson
    inference_backend: str = "auto"   # auto (ONNX bila ada) | onnx | xgboost
    calibration_path: str = "models/calibration.json"   # isotonic (roadmap 4.2)
    challenger_path: str = "models/challenger_lgbm.txt"
    enable_challenger: bool = True    # shadow deployment LightGBM (dicatat di ledger, tidak memicu alert)
    hazard_path: str = "models/hazard_xgb.json"
    hawkes_path: str = "models/hawkes_params.json"
    feature_quantiles_path: str = "models/feature_quantiles.json"
    hmm_path: str = "models/hmm_regime.json"
    hmm_gate: str = "high"            # off | high | transition (lihat app/regime.py & reports/EVALUATION.md)
    hmm_gate_window: int = 30         # menit, untuk mode transition
    ewm_halflife_min: int = 1440      # half-life normalisasi EWM online (1 hari); wajib sama dengan saat training

    # --- Circuit breaker: OK / DEGRADED (model gagal) / STALE (data basi, alert ditahan) ---
    stream_stale_sec: float = 5.0         # now - event_time > ini pada stream wajib -> STALE
    breaker_recovery_sec: float = 30.0    # setelah stream putus, tetap STALE sampai segar selama ini
    breaker_streams: str = "kline,aggTrade,depth"  # stream wajib (yang dinonaktifkan otomatis dilewati)
    breaker_check_sec: float = 5.0        # loop latar untuk /health & notifikasi saat feed mati total
    heartbeat_interval_sec: float = 5.0   # ingestion menulis state stream ke Redis tiap N detik

    # --- Target, ledger & performa live ---
    data_dir: str = "data"
    ledger_path: str = "data/ledger.db"
    ledger_max_log_lag_ms: int = 60_000
    baseline_bootstrap: bool = True       # bootstrap baseline 30 hari dari Binance REST saat start
    baseline_save_sec: float = 60.0       # state OnlineBaseline -> Redis tiap N detik
    monitor_interval_sec: float = 300     # scorer ledger tiap 5 menit
    perf_window_days: float = 7
    perf_lookback_min: int = 30
    perf_min_positives: int = 20
    perf_min_lift: float = 1.5            # PR-AUC live / base rate di bawah ini = performa turun
    retrain_on_degradation: bool = False
    retrain_command: str = "python -m training.train --source binance --days 90"
    public_base_url: str = ""             # untuk link baris ledger di pesan Telegram

    # --- Adaptive conformal threshold (roadmap 4.1) ---
    conformal_enabled: bool = True
    conformal_mode: str = "fdr"           # fdr: target porsi alert palsu | fpr: target alarm pada menit non-spike
    conformal_alpha: float = 0.20         # target long-run false alarm rate
    conformal_gamma: float = 0.01         # laju adaptasi
    conformal_tau_min: float = 0.05
    conformal_tau_max: float = 0.95

    # --- Telegram ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_mode: str = "polling"      # polling | webhook | off  (bot dua arah)
    telegram_webhook_url: str = ""      # contoh: https://api-kamu.onrender.com/telegram/webhook
    telegram_webhook_secret: str = ""
    # Chat yang boleh menjalankan command pengubah state (/setthreshold, /setalpha). Kosong = TELEGRAM_CHAT_ID.
    allowed_chat_ids: str = ""
    admin_api_token: str = ""           # header X-Admin-Token untuk PUT /config/threshold (kosong = nonaktif)

    # --- Proses di dalam service API ---
    run_engine: bool = True      # subscriber Redis -> fitur -> model -> alert
    run_ingestion: bool = False  # jalankan ingestion di proses API (mode hosting gratis 1 service)
    run_monitor: bool = True     # scorer ledger + performa live + conformal di proses API

    log_level: str = "INFO"

    @property
    def allowed_chats(self) -> set[str]:
        ids = {x.strip() for x in self.allowed_chat_ids.split(",") if x.strip()}
        return ids or ({str(self.telegram_chat_id)} if self.telegram_chat_id else set())

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]

    @property
    def required_streams(self) -> tuple[str, ...]:
        enabled = {"kline": True, "aggTrade": self.enable_aggtrade, "depth": self.enable_depth,
                   "markPrice": self.enable_futures, "fut_aggTrade": self.enable_futures}
        return tuple(s.strip() for s in self.breaker_streams.split(",") if s.strip() and enabled.get(s.strip(), True))


@lru_cache
def get_settings() -> Settings:
    return Settings()
