from db.normalize import looks_like_text, normalize_key
from db.templates import fill, templatize


def test_normalize_key_unifies_ocr_variants():
    variants = ["星臨者.さま..？", "星臨者..さま...?", "星臨者・.さま...?", "星臨者さま……？"]
    keys = {normalize_key(v) for v in variants}
    assert keys == {"星臨者…さま…？", "星臨者さま…？"}  # dots inside vs none: two keys, both stable
    assert normalize_key("私たちは『願った』からまた会えた。") == normalize_key(
        "私たちは「願った」からまた会えた。"
    )
    assert normalize_key("ソザン力さま") == normalize_key("ソザンカさま")
    assert normalize_key("×1.0") == "×1.0"  # dot inside a number survives
    assert normalize_key("帰つた") == normalize_key("帰った")


def test_looks_like_text_rejects_hud_noise():
    assert not looks_like_text("78")
    assert not looks_like_text("ー")
    assert not looks_like_text("」8")
    assert looks_like_text("星臨者さま！")
    assert looks_like_text("AUTO")


def test_templates_roundtrip():
    t = templatize("はじめまして、村長さん。私はWren.", "Wren")
    assert "{player}" in t.key and t.player == "Wren"
    t2 = templatize("残り30秒で12体", None)
    assert t2.key.count("{n}") == 2 and t2.values == ["30", "12"]
    assert fill("{n} seconds left, {n} enemies, {player}", t2.values, "Dave") == (
        "30 seconds left, 12 enemies, Dave"
    )


def test_reading_a_database_never_writes_it(tmp_path):
    import hashlib

    import pytest

    from db.store import Line, Store

    path = tmp_path / "game.sqlite"
    with Store(path) as s:
        s.upsert_line(Line("あ", "あ", None, "ui", False, "A", "human", "fixture", None, 1, None))
    before = hashlib.md5(path.read_bytes()).hexdigest()
    with Store(path, readonly=True) as s:
        assert [ln.key for ln in s.iter_lines()] == ["あ"]
    assert hashlib.md5(path.read_bytes()).hexdigest() == before
    with pytest.raises(FileNotFoundError):
        Store(tmp_path / "missing.sqlite", readonly=True)
    # a failure halfway leaves the database as it was
    with pytest.raises(RuntimeError):
        with Store(path) as s:
            s.upsert_line(Line("い", "い", None, "ui", False, "I", "human", "x", None, 2, None))
            raise RuntimeError("halfway")
    with Store(path, readonly=True) as s:
        assert [ln.key for ln in s.iter_lines()] == ["あ"]
