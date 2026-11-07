"""Roadmap 1.2: order book lokal dengan urutan event sintetis (benar, update hilang, duplikat)."""
from ingestion.orderbook import LocalOrderBook

SNAP = {"lastUpdateId": 100, "bids": [["100.0", "1"], ["99.0", "2"], ["98.0", "3"]],
        "asks": [["101.0", "1"], ["102.0", "2"], ["103.0", "3"]]}


def ev(U, u, b=(), a=()):
    return {"e": "depthUpdate", "U": U, "u": u, "b": [list(x) for x in b], "a": [list(x) for x in a]}


def test_correct_sequence_with_buffered_events():
    book = LocalOrderBook("BTCUSDT")
    book.on_event(ev(95, 99, b=[("97.0", "9")]))         # u <= lastUpdateId: dibuang
    book.on_event(ev(100, 102, b=[("100.0", "5")]))      # U <= 101 <= u: event pertama yang sah
    book.on_event(ev(103, 103, a=[("101.0", "0")]))      # hapus level
    assert book.obi() is None  # belum ada snapshot
    assert book.apply_snapshot(SNAP)
    assert book.valid and book.last_update_id == 103
    assert book.bids[100.0] == 5 and 97.0 not in book.bids
    assert book.best() == (100.0, 102.0)
    assert book.on_event(ev(104, 105, b=[("99.5", "1")]))
    assert book.best() == (100.0, 102.0) and book.bids[99.5] == 1
    obi = book.obi(5)
    b = 5 + 1 + 2 + 3
    a = 2 + 3
    assert abs(obi - (b - a) / (b + a)) < 1e-12


def test_missing_update_invalidates_and_resnapshot_recovers():
    book = LocalOrderBook("BTCUSDT")
    book.apply_snapshot(SNAP)
    assert book.on_event(ev(101, 101))
    assert not book.on_event(ev(103, 104))  # 102 hilang
    assert not book.valid and book.gaps == 1
    assert book.obi() is None  # OBI tidak pernah dari book invalid
    book.on_event(ev(105, 106, b=[("100.0", "7")]))  # di-buffer sambil menunggu snapshot
    assert book.apply_snapshot({**SNAP, "lastUpdateId": 105})
    assert book.valid and book.bids[100.0] == 7 and book.last_update_id == 106
    assert book.obi() is not None


def test_first_event_must_bridge_snapshot():
    book = LocalOrderBook("BTCUSDT")
    book.on_event(ev(110, 112))  # snapshot (100) terlalu tua: U=110 > 101
    assert not book.apply_snapshot(SNAP)
    assert not book.valid and book.obi() is None


def test_duplicates_are_ignored():
    book = LocalOrderBook("BTCUSDT")
    book.apply_snapshot(SNAP)
    assert book.on_event(ev(101, 102, b=[("100.0", "4")]))
    assert book.on_event(ev(101, 102, b=[("100.0", "999")]))  # duplikat persis
    assert book.on_event(ev(102, 102))  # sebagian lama
    assert book.bids[100.0] == 4 and book.duplicates == 2 and book.valid
    assert book.on_event(ev(103, 103))


def test_crossed_book_is_invalid():
    book = LocalOrderBook("BTCUSDT")
    book.apply_snapshot(SNAP)
    assert not book.on_event(ev(101, 101, b=[("101.5", "1")]))
    assert not book.valid
