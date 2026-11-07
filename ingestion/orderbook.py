"""Order book lokal dari diff-depth stream Binance Spot (`<symbol>@depth@100ms`).

Aturan sinkronisasi (dokumentasi Binance "How to manage a local order book correctly"):

1. Buka stream dan BUFFER event sebelum snapshot ada.
2. Ambil snapshot REST `/api/v3/depth?symbol=X&limit=1000` -> `lastUpdateId`.
3. Buang event dengan `u <= lastUpdateId` (sudah termasuk di snapshot).
4. Event pertama yang dipakai WAJIB memenuhi `U <= lastUpdateId + 1 <= u`.
   Kalau tidak (snapshot terlalu tua / event hilang), book INVALID -> ambil snapshot lagi.
5. Setelah itu setiap event wajib `U == u_sebelumnya + 1`. Kalau tidak, ada update yang hilang:
   book INVALID -> snapshot ulang. Event duplikat/lama (`u <= u_sebelumnya`) diabaikan.
6. qty = 0 berarti level dihapus.

OBI (order book imbalance) dihitung dari 5 level teratas saat candle close, dan TIDAK PERNAH
dihitung dari book yang invalid (return None -> fitur kosong, bukan angka karangan).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class BookInvalid(Exception):
    pass


@dataclass
class LocalOrderBook:
    symbol: str
    max_buffer: int = 5000
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_update_id: int | None = None   # u terakhir yang sudah diterapkan
    valid: bool = False
    synced_first: bool = False
    buffer: list[dict] = field(default_factory=list)
    resyncs: int = 0
    gaps: int = 0
    duplicates: int = 0
    invalid_reason: str | None = "belum ada snapshot"

    # ------------------------------------------------------------------ input
    def on_event(self, ev: dict) -> bool:
        """Proses satu event depthUpdate. Return False bila book menjadi invalid (perlu snapshot ulang)."""
        if self.last_update_id is None:  # belum ada snapshot: buffer
            self.buffer.append(ev)
            if len(self.buffer) > self.max_buffer:
                self.buffer.pop(0)
            return True
        if not self.valid:
            self.buffer.append(ev)  # menunggu snapshot ulang
            if len(self.buffer) > self.max_buffer:
                self.buffer.pop(0)
            return False
        return self._apply(ev)

    def apply_snapshot(self, snap: dict) -> bool:
        """Pasang snapshot REST lalu terapkan event yang di-buffer. Return True bila book valid."""
        self.bids = {float(p): float(q) for p, q in snap["bids"] if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in snap["asks"] if float(q) > 0}
        self.last_update_id = int(snap["lastUpdateId"])
        self.valid, self.synced_first = True, False
        self.invalid_reason = None
        pending, self.buffer = self.buffer, []
        for ev in pending:
            if not self._apply(ev):
                return False
        return self.valid

    def invalidate(self, reason: str) -> None:
        if self.valid:
            log.warning("Order book %s INVALID: %s", self.symbol, reason)
        self.valid = False
        self.invalid_reason = reason
        self.resyncs += 1

    # ------------------------------------------------------------------ core
    def _apply(self, ev: dict) -> bool:
        U, u = int(ev["U"]), int(ev["u"])
        if u <= self.last_update_id:  # sudah ada di snapshot / duplikat
            self.duplicates += 1
            return True
        if not self.synced_first:
            if not (U <= self.last_update_id + 1 <= u):
                self.gaps += 1
                self.invalidate(f"event pertama tidak menyambung snapshot (U={U}, u={u}, "
                                f"lastUpdateId={self.last_update_id})")
                return False
            self.synced_first = True
        elif U != self.last_update_id + 1:
            self.gaps += 1
            self.invalidate(f"update hilang: U={U} != u_sebelumnya+1={self.last_update_id + 1}")
            return False
        for side, levels in ((self.bids, ev.get("b", [])), (self.asks, ev.get("a", []))):
            for p, q in levels:
                p, q = float(p), float(q)
                if q == 0:
                    side.pop(p, None)
                else:
                    side[p] = q
        self.last_update_id = u
        if self.bids and self.asks and max(self.bids) >= min(self.asks):
            self.invalidate("book bersilangan (best bid >= best ask)")
            return False
        return True

    # ------------------------------------------------------------------ output
    def top(self, n: int = 5) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bids = sorted(self.bids.items(), key=lambda x: -x[0])[:n]
        asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return bids, asks

    def best(self) -> tuple[float | None, float | None]:
        return (max(self.bids) if self.bids else None, min(self.asks) if self.asks else None)

    def obi(self, levels: int = 5) -> float | None:
        """(sum bid qty - sum ask qty) / total pada `levels` level teratas. None bila book invalid."""
        if not self.valid or not self.synced_first and self.buffer:
            return None
        bids, asks = self.top(levels)
        b, a = sum(q for _, q in bids), sum(q for _, q in asks)
        if b + a <= 0:
            return None
        return (b - a) / (b + a)

    def status(self) -> dict:
        bb, ba = self.best()
        return {"valid": self.valid, "last_update_id": self.last_update_id, "best_bid": bb, "best_ask": ba,
                "resyncs": self.resyncs, "gaps": self.gaps, "duplicates": self.duplicates,
                "invalid_reason": self.invalid_reason, "buffered": len(self.buffer)}
