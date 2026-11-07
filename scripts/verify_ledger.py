"""Verifikasi independen forecast ledger + publikasi head (roadmap 1.7, dipakai drift_check.yml).

Menghitung ulang SELURUH hash chain dari data mentah (tanpa mempercayai /ledger/verify server),
lalu memastikan head yang pernah di-commit ke git (badges/ledger_head.txt) masih ada di rantai.
Kalau tidak ada, riwayat sebelum titik itu sudah ditulis ulang.

    python -m scripts.verify_ledger --url https://<host> --anchor badges/ledger_head.txt --out badges
    python -m scripts.verify_ledger --db data/ledger.db

Exit code: 0 OK, 1 hash chain putus / head lama hilang (CI merah), 3 service tidak terjangkau.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from app.services.ledger import GENESIS, row_hash

MODELS = ("champion_xgboost", "challenger_lgbm", "hazard_xgb")


def rows_from_url(url: str, page: int = 5000):
    import httpx

    after = 0
    with httpx.Client(timeout=60) as c:
        while True:
            r = c.get(f"{url.rstrip('/')}/ledger/rows", params={"after_seq": after, "limit": page})
            r.raise_for_status()
            d = r.json()
            if not d["rows"]:
                return
            yield from d["rows"]
            after = d["next_after_seq"]


def rows_from_db(path: str):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    for r in conn.execute("SELECT * FROM predictions ORDER BY seq"):
        yield dict(r)
    conn.close()


def verify(rows, anchor: str | None = None) -> dict:
    prev, n, seen_anchor = GENESIS, 0, anchor in (None, GENESIS)
    for row in rows:
        if row["prev_hash"] != prev or row_hash(prev, row) != row["row_hash"]:
            return {"ok": False, "first_bad_seq": row["seq"], "rows_checked": n, "head": prev,
                    "anchor_found": seen_anchor}
        prev = row["row_hash"]
        n += 1
        seen_anchor = seen_anchor or prev == anchor
    return {"ok": seen_anchor, "first_bad_seq": None, "rows_checked": n, "head": prev, "anchor_found": seen_anchor}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--db")
    ap.add_argument("--anchor", help="file berisi head lama (mis. badges/ledger_head.txt dari git)")
    ap.add_argument("--out", help="folder badge: tulis ledger_head.txt + badge ledger + salin badge model dari service")
    args = ap.parse_args(argv)
    anchor = None
    if args.anchor and Path(args.anchor).exists():
        anchor = Path(args.anchor).read_text().strip() or None
    try:
        res = verify(rows_from_url(args.url) if args.url else rows_from_db(args.db), anchor)
    except Exception as exc:  # noqa: BLE001
        print(f"Tidak bisa membaca ledger: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(res))
    if res["first_bad_seq"] is not None:
        print(f"LEDGER TAMPERED: hash chain putus di seq {res['first_bad_seq']}", file=sys.stderr)
    elif not res["anchor_found"]:
        print(f"LEDGER REWRITTEN: head lama {anchor[:16]}… tidak ada di rantai sekarang", file=sys.stderr)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ledger.json").write_text(json.dumps({
            "schemaVersion": 1, "label": "ledger hash chain",
            "message": f"verified · {res['rows_checked']} rows" if res["ok"] else "BROKEN",
            "color": "brightgreen" if res["ok"] else "red"}))
        if res["ok"]:
            (out / "ledger_head.txt").write_text(res["head"] + "\n")
        if args.url:
            import httpx

            for m in MODELS:
                for suffix in ("", "_metrics"):
                    r = httpx.get(f"{args.url.rstrip('/')}/badges/{m}{suffix}.json", timeout=30)
                    if r.status_code == 200:
                        (out / f"{m}{suffix}.json").write_text(json.dumps(r.json(), indent=2 if suffix else None))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
