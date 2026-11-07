"""Bot Telegram dua arah (interactive).

Command (baca saja, semua chat):
    /status              -> status breaker (OK/DEGRADED/STALE), latency p95, versi model, threshold, prob terakhir
    /threshold           -> lihat threshold aktif (+ target alarm palsu ACI)
    /help                -> daftar command
Command pengubah state (HANYA chat di ALLOWED_CHAT_IDS; chat lain ditolak & dicatat di log):
    /setthreshold 0.85   -> threshold probabilitas alert, 0 < p < 1 (disimpan di Redis)
    /setalpha 0.2        -> target false alarm rate adaptive conformal, 0 < a < 1

Mode (TELEGRAM_MODE):
    polling : long-polling getUpdates (cocok untuk lokal / docker compose, tanpa URL publik)
    webhook : Telegram memanggil POST /telegram/webhook (cocok untuk Render/Railway)
    off     : bot hanya mengirim notifikasi (satu arah)

Keamanan: whitelist ALLOWED_CHAT_IDS (koma); kosong = hanya TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

log = logging.getLogger(__name__)

HELP = (
    "🤖 <b>Crypto Volatility Bot</b>\n"
    "/status — status sistem, latency, versi model\n"
    "/threshold — lihat threshold alert\n"
    "/setthreshold 0.85 — ubah threshold probabilitas alert (0 < p < 1)\n"
    "/setalpha 0.2 — ubah target alarm palsu adaptive conformal (0 < a < 1)\n"
    "/help — bantuan"
)


class TelegramBot:
    STATE_CHANGING = {"/setthreshold", "/setalpha"}

    def __init__(self, token: str, allowed_chat_ids: set[str] | str, engine, status_provider: Callable[[], dict]):
        self.token = token
        if isinstance(allowed_chat_ids, str):
            allowed_chat_ids = {x.strip() for x in allowed_chat_ids.split(",") if x.strip()}
        self.allowed = {str(x) for x in allowed_chat_ids}
        self.rejected = 0
        self.engine = engine
        self.status_provider = status_provider
        self.api = f"https://api.telegram.org/bot{token}"
        self._offset = 0
        self._running = False

    # ------------------------------------------------------------ commands
    async def handle_update(self, update: dict) -> str | None:
        msg = update.get("message") or update.get("edited_message") or {}
        text = (msg.get("text") or "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if not text.startswith("/"):
            return None
        reply = await self.handle_command(text, chat_id)
        if reply:
            await self.send(chat_id, reply)
        return reply

    async def handle_command(self, text: str, chat_id: str | None = None) -> str:
        parts = text.split()
        cmd = parts[0].split("@")[0].lower()
        if cmd in self.STATE_CHANGING and (chat_id is None or str(chat_id) not in self.allowed):
            self.rejected += 1
            log.warning("DITOLAK: %s dari chat %s (tidak ada di ALLOWED_CHAT_IDS)", cmd, chat_id)
            return "⛔ Command ini hanya untuk admin (chat tidak ada di ALLOWED_CHAT_IDS). Percobaan dicatat."
        if cmd in ("/start", "/help"):
            return HELP
        if cmd == "/threshold":
            c = getattr(self.engine, "conformal", None)
            extra = f" · adaptif ACI, target alarm palsu {c.alpha:.0%}" if c else " · statis"
            return f"🎚️ Threshold alert aktif: <b>{self.engine.threshold:.3f}</b>{extra}"
        if cmd in self.STATE_CHANGING:
            name = "threshold" if cmd == "/setthreshold" else "alpha"
            if len(parts) < 2:
                return f"Format: {cmd} 0.{'85' if name == 'threshold' else '2'}"
            try:
                value = float(parts[1].replace(",", "."))
            except ValueError:
                return f"❌ Nilai harus angka, contoh: {cmd} 0.5"
            if not 0.0 < value < 1.0:
                return f"❌ {name} harus di antara 0 dan 1 (eksklusif), bukan {value}"
            try:
                if name == "threshold":
                    old = self.engine.threshold
                    await self.engine.set_threshold(value)
                else:
                    old = self.engine.conformal.alpha if getattr(self.engine, "conformal", None) else float("nan")
                    await self.engine.set_alpha(value)
            except ValueError as exc:
                return f"❌ {exc}"
            log.warning("%s diubah %.3f -> %.3f oleh chat %s", name, old, value, chat_id)
            return f"✅ {name} diubah {old:.3f} → <b>{value:.3f}</b> (berlaku mulai candle berikutnya)"
        if cmd == "/status":
            return format_status(self.status_provider())
        return "Command tidak dikenal. " + HELP

    # ------------------------------------------------------------ transport
    async def send(self, chat_id: str, text: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(f"{self.api}/sendMessage",
                                  json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
        except httpx.HTTPError as exc:
            log.warning("Gagal balas Telegram: %s", exc)

    async def set_webhook(self, url: str, secret: str) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{self.api}/setWebhook",
                                  json={"url": url, "secret_token": secret or None, "allowed_updates": ["message"]})
            log.info("setWebhook: %s", r.text[:200])

    async def run_polling(self) -> None:
        self._running = True
        async with httpx.AsyncClient(timeout=40) as client:
            try:
                await client.post(f"{self.api}/deleteWebhook")
            except httpx.HTTPError:
                pass
            backoff = 1
            while self._running:
                try:
                    r = await client.get(f"{self.api}/getUpdates",
                                         params={"timeout": 30, "offset": self._offset, "allowed_updates": '["message"]'})
                    data = r.json()
                    if not data.get("ok"):
                        raise RuntimeError(data.get("description", "getUpdates gagal"))
                    for upd in data.get("result", []):
                        self._offset = upd["update_id"] + 1
                        await self.handle_update(upd)
                    backoff = 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("Polling Telegram error: %s (retry %ss)", exc, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)

    def stop(self) -> None:
        self._running = False


def format_status(s: dict) -> str:
    icon = "🟢" if s.get("status") == "ok" else "🟠"
    breaker = s.get("circuit_breaker", {})
    lat = s.get("latency", {})
    lines = [
        f"{icon} <b>Status: {s.get('status', '?').upper()}</b>",
        f"Breaker: <b>{breaker.get('state', '?')}</b>" + (f" — {breaker.get('reason')}" if breaker.get("reason") else ""),
        f"Model: {s.get('model_version') or 'TIDAK AKTIF (fallback)'}",
        f"Threshold: {s.get('threshold', 0):.2f} · HMM gate: {s.get('hmm_gate')}",
    ]
    if lat:
        lines.append(
            f"Latency processing p95: {lat.get('processing_p95_ms', '–')} ms · end-to-end p95: {lat.get('end_to_end_p95_ms', '–')} ms"
        )
    for sym, r in (s.get("latest") or {}).items():
        lines.append(f"• {sym}: prob {r['probability']:.0%} [{r.get('status', '?')}, {r['mode']}, {r['direction']}]")
    lines.append(f"Uptime: {s.get('uptime_sec', 0) // 60} menit · RAM: {s.get('rss_mb') or '–'} MB")
    return "\n".join(lines)
