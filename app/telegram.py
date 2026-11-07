"""Kirim pesan PUSH via Telegram Bot API (REST call sederhana)."""
from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, timeout: float = 10.0):
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(self, text: str) -> tuple[bool, str | None]:
        """Return (terkirim, pesan_error)."""
        if not self.enabled:
            return False, "Telegram belum dikonfigurasi (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID kosong)"
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for attempt in range(2):
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 429 and attempt == 0:
                        # Rate limit Telegram: hormati retry_after (maks 30 dtk), lalu coba sekali lagi
                        retry = resp.json().get("parameters", {}).get("retry_after", 5)
                        log.warning("Telegram 429, retry dalam %ss", retry)
                        await asyncio.sleep(min(float(retry), 30.0))
                        continue
                    break
            if resp.status_code == 200 and resp.json().get("ok"):
                return True, None
            return False, f"HTTP {resp.status_code}: {resp.text[:300]}"
        except httpx.HTTPError as exc:  # jaringan putus, timeout, dll.
            log.warning("Gagal kirim Telegram: %s", exc)
            return False, str(exc)


DIRECTION_HEADLINE = {
    "downside": "🔻 <b>ALERT: High Downside Volatility Spike Detected (Risk of Dump)</b>",
    "upside": "🔺 <b>ALERT: High Upside Volatility Spike Detected (Pump Risk / FOMO)</b>",
    "two_way": "⚠️ <b>ALERT: Two-Way Volatility Spike Detected (Whipsaw)</b>",
}


def format_alert(symbol: str, probability: float, close: float, threshold: float, features: dict,
                 mode: str = "model", direction: str = "two_way", regime: str | None = None,
                 reasons: list[str] | None = None, hazard: dict | None = None, contributors: list[str] | None = None,
                 status: str = "OK", threshold_kind: str = "statis", ledger: dict | None = None) -> str:
    """Pesan alert: P(spike <=5/10/15 menit), kontributor utama, status sistem, link baris ledger."""
    lines = [DIRECTION_HEADLINE.get(direction, DIRECTION_HEADLINE["two_way"]),
             f"<b>{symbol}</b> · harga <code>{close:,.4f}</code>"]
    if hazard and hazard.get("p_within"):
        pw = hazard["p_within"]
        lines.append("Peluang spike mulai: " + " · ".join(f"≤{k}m <b>{float(v):.0%}</b>" for k, v in pw.items()))
    if mode == "model":
        lines.append(f"Probabilitas model utama (15m): <b>{probability:.0%}</b> "
                     f"(threshold alert {threshold:.2f}, {threshold_kind})")
    else:
        lines.append("⚙️ <i>Model ML gagal: skor dari heuristic Parkinson (fallback)</i>")
    if contributors:
        lines.append("<b>Kontributor utama:</b>")
        lines += [f"• {c}" for c in contributors[:3]]
    if reasons:
        lines.append("<b>Konteks:</b>")
        lines += [f"• {r}" for r in reasons[:3]]
    lines.append(f"Status sistem: <b>{status}</b>" + (f" · rezim {regime}" if regime else ""))
    if ledger:
        ref = f"seq {ledger['seq']} · hash <code>{ledger['row_hash'][:12]}</code>"
        lines.append(f"Ledger: <a href=\"{ledger['url']}\">{ref}</a>" if ledger.get("url") else f"Ledger: {ref}")
    return "\n".join(lines)
