"""A game's database folder: the databases in the player's language are used (the newest
first, live translations last), a database in another language and any other file are
named and left alone; a file the player has is moved in once it passed the checks; files
from before databases were kept per language are moved in place."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from app.config import GameEntry, Settings
from db.store import Line, Store


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "data"))
    from app.library import Library

    return Library(Settings())


def make_db(path: Path, lines: dict[str, str | None], lang: str | None = "en",
            source: str = "batch") -> Path:
    with Store(path) as store:
        if lang:
            store.set_info(game="test-game", source_lang="ja", target_lang=lang)
        for i, (key, tr) in enumerate(lines.items()):
            store.upsert_line(Line(key=key, source_text=key, speaker=None, screen_type="ui",
                                   is_template=False, translation=tr, tier="machine",
                                   source=source, chapter=None, seq=i, game_version=None))
    return path


def names(files) -> list[str]:  # noqa: ANN001
    return [f.path.name for f in files]


def test_every_database_in_the_folder_counts_and_other_files_are_named(lib) -> None:
    g = lib.add_game("Some Game")
    d = lib.db_dir(g)
    make_db(d / "community.sqlite", {"こんにちは": "Hello", "さようなら": "Goodbye"})
    (d / "notes.txt").write_text("not a database", encoding="utf-8")
    (d / "broken.sqlite").write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)  # a cut download
    (d / "half.sqlite.download").write_bytes(b"in progress")  # (a download under way)
    (d / "community.sqlite-journal").write_bytes(b"")  # (SQLite's own)
    files = lib.databases(g).scan()
    assert [f.path.name for f in files if f.ok] == ["community.sqlite"]
    bad = {f.path.name: f.why for f in files if not f.ok}
    assert set(bad) == {"notes.txt", "broken.sqlite"}
    assert "not a database" in bad["notes.txt"]
    info = lib.info(g)
    assert info["files"] == 1 and info["translated"] == 2
    assert lib.db_paths(g) == [d / "community.sqlite"]


def test_only_the_databases_in_the_players_language_are_used(lib) -> None:
    from db.lookup import Lookup

    g = lib.add_game("Some Game")
    d = lib.db_dir(g)
    make_db(d / "game.en.sqlite", {"はい": "Yes"}, lang="en")
    make_db(d / "game.es-419.sqlite", {"はい": "Sí"}, lang="es-419")
    folder = lib.databases(g)
    assert folder.languages() == ["en", "es-419"]
    assert names(lib.db_files(g)) == ["game.en.sqlite"]  # English, the default
    es_file = next(f for f in folder.scan() if f.lang == "es-419")
    assert "Español" in folder.why_unused(es_file, "en")
    g.target_lang = "es"  # Spanish: the Latin American database serves it
    assert names(lib.db_files(g)) == ["game.es-419.sqlite"]
    assert folder.why_unused(es_file, "es") == ""
    assert Lookup.from_dbs(lib.db_paths(g)).match("はい").translation == "Sí"
    g.target_lang = "de"  # nothing in German: live translation only
    assert lib.db_files(g) == [] and lib.info(g)["files"] == 0


def test_a_database_that_says_no_language_is_english(lib) -> None:
    g = lib.add_game("Some Game")
    make_db(lib.db_dir(g) / "old.sqlite", {"はい": "Yes"}, lang=None)
    assert names(lib.db_files(g)) == ["old.sqlite"]
    g.target_lang = "es-419"
    assert lib.db_files(g) == []


def test_the_newest_database_wins_and_live_translations_come_last(lib) -> None:
    from db.lookup import Lookup

    g = lib.add_game("Some Game")
    d = lib.db_dir(g)
    old = make_db(d / "old.sqlite", {"はい": "Yes (old)", "いいえ": "No"})
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    make_db(d / "new.sqlite", {"はい": "Yes"})
    make_db(d / "live-en.sqlite", {"はい": "Yes (live)", "たぶん": "Maybe"}, source="live")
    make_db(d / "live-es-419.sqlite", {"たぶん": "Quizás"}, lang="es-419", source="live")
    paths = lib.db_paths(g)
    assert [p.name for p in paths] == ["new.sqlite", "old.sqlite", "live-en.sqlite"]
    lookup = Lookup.from_dbs(paths)
    assert lookup.match("はい").translation == "Yes"  # the newest
    assert lookup.match("いいえ").translation == "No"  # only the old one has it
    assert lookup.match("たぶん").translation == "Maybe"  # only live translation has it


def test_a_file_dropped_in_or_taken_out_counts_at_once(lib) -> None:
    g = lib.add_game("Some Game")
    d = lib.db_dir(g)
    assert lib.db_paths(g) == []
    make_db(d / "mine.db", {"はい": "Yes"})
    assert lib.db_paths(g) == [d / "mine.db"]
    (d / "mine.db").unlink()
    assert lib.db_paths(g) == []


def test_a_file_the_player_has_is_moved_in_only_when_it_is_a_database(lib, tmp_path) -> None:
    from sync.download import SyncError

    g = lib.add_game("Some Game")
    (tmp_path / "downloads").mkdir()
    good = make_db(tmp_path / "downloads" / "shared.sqlite", {"はい": "Yes"})
    dest = lib.use_database_file(g, good)
    assert dest == lib.db_dir(g) / "shared.sqlite" and dest.exists() and not good.exists()
    assert lib.info(g)["translated"] == 1
    fake = tmp_path / "downloads" / "virus.sqlite"
    fake.write_bytes(b"MZ\x90\x00" + b"\x00" * 200)  # a program renamed
    with pytest.raises(SyncError):
        lib.use_database_file(g, fake)
    assert fake.exists() and not (lib.db_dir(g) / "virus.sqlite").exists()  # left alone
    again = make_db(tmp_path / "downloads" / "shared.sqlite", {"はい": "Yes!", "いいえ": "No"})
    lib.use_database_file(g, again)  # the same name: the old one goes to the backups
    assert lib.info(g)["translated"] == 2
    assert list(lib.backups_dir(g).glob("shared.*.backup.sqlite"))


def test_a_file_the_player_has_comes_in_as_text_only(lib, tmp_path) -> None:
    import sqlite3

    g = lib.add_game("Some Game")
    src = make_db(tmp_path / "shared.sqlite", {"はい": "Yes"})
    with sqlite3.connect(src) as conn:  # schema of its own: never taken in
        conn.execute("CREATE TABLE extra (x)")
        conn.execute("CREATE TRIGGER t AFTER INSERT ON lines BEGIN DELETE FROM lines; END")
    conn.close()
    dest = lib.use_database_file(g, src)
    with sqlite3.connect(dest) as conn:
        kinds = {(r[0], r[1]) for r in conn.execute("SELECT type, name FROM sqlite_master")}
    conn.close()
    assert ("trigger", "t") not in kinds and ("table", "extra") not in kinds
    assert lib.info(g)["translated"] == 1 and not src.exists()


def test_a_database_that_comes_in_tells_the_game_its_language(lib, tmp_path) -> None:
    g = lib.add_game("Some Game")
    src = tmp_path / "zh.sqlite"
    with Store(src) as store:
        store.set_info(game="x", source_lang="zh-hant", target_lang="en")
    dest = lib.use_database_file(g, src)
    assert lib.take_source_language(g, dest) and g.source_lang == "zh-hant"
    g.source_lang = "ja"  # the player's choice is never overruled
    assert not lib.take_source_language(g, dest) and g.source_lang == "ja"


def test_files_from_app_1_1_are_moved_in_place(lib) -> None:
    from app.paths import games_data_dir

    folder = games_data_dir() / "old-game"
    (folder / "databases").mkdir(parents=True)
    (folder / "cache").mkdir()
    make_db(folder / "database.sqlite", {"はい": "Yes"}, lang=None)
    make_db(folder / "databases" / "live.sqlite", {"たぶん": "Maybe"}, lang=None, source="live")
    (folder / "database.20261101-120000.backup.sqlite").write_bytes(b"old")
    (folder / "memory.json").write_text("{}", encoding="utf-8")
    (folder / "cache" / "covers.json").write_text("{}", encoding="utf-8")
    g = GameEntry("old-game", "Old Game")
    lib.settings.games.append(g)
    assert names(lib.db_files(g)) == ["database.sqlite", "live-en.sqlite"]
    assert not (folder / "database.sqlite").exists()
    assert (folder / "backups" / "database.20261101-120000.backup.sqlite").exists()
    assert lib.memory_path(g) == folder / "memory-en.json" and lib.memory_path(g).exists()
    assert (lib.cache_dir(g) / "covers.json").exists()


def test_live_translations_go_to_their_own_file_per_language(lib) -> None:
    from app.session import HudSession

    g = lib.add_game("Some Game")
    make_db(lib.db_dir(g) / "community.sqlite", {"はい": "Yes"})
    s = HudSession(lib, lib.settings)
    s._keep("ありがとう", "Thank you", "dialogue", g)
    live = lib.live_path(g)
    assert live.name == "live-en.sqlite"
    with Store(live, readonly=True) as store:
        line = store.get_line("ありがとう")
        assert line.translation == "Thank you" and line.source == "live"
        assert store.info().target_lang == "en"  # the file says what it is
    with Store(lib.db_dir(g) / "community.sqlite", readonly=True) as store:
        assert store.get_line("ありがとう") is None  # the shared database is never written
    info = lib.info(g)
    assert info["translated"] == 1 and info["live"] == 1
    g.target_lang = "es-419"
    s._keep("ありがとう", "Gracias", "dialogue", g)
    assert lib.live_path(g).name == "live-es-419.sqlite"
    assert lib.info(g)["live"] == 1 and names(lib.db_files(g)) == ["live-es-419.sqlite"]


def test_the_overlay_sees_a_changed_folder_but_not_its_own_live_translations(lib) -> None:
    g = lib.add_game("Some Game")
    d = lib.db_dir(g)
    make_db(d / "community.sqlite", {"はい": "Yes"})
    before = lib.db_state(g)
    make_db(d / "live-en.sqlite", {"たぶん": "Maybe"}, source="live")  # written while it runs
    assert lib.db_state(g) == before
    make_db(d / "more.sqlite", {"いいえ": "No"})  # dropped in: the overlay takes it
    assert lib.db_state(g) != before


def test_the_miss_log_writes_each_text_once(tmp_path) -> None:
    from app.misslog import MissLog, files, read_records

    log = MissLog(tmp_path / "misses", "some-game", "en", player="ルミ")
    assert log.add("残り時間 5秒", "ui")
    assert not log.add("残り時間 4秒", "ui")  # numbers left out: the same text
    assert log.add("ルミ、こっちだよ！", "dialogue", "テララ")
    assert not log.add("ルミ、こっちだよ！", "dialogue", "テララ")
    again = MissLog(tmp_path / "misses", "some-game", "en", player="ルミ")
    assert not again.add("残り時間 9秒", "ui") and again.count() == 2  # across sessions
    recs = list(read_records(files(tmp_path / "misses")))
    assert [r["text"] for r in recs] == ["残り時間 5秒", "ルミ、こっちだよ！"]
    assert recs[1]["speaker"] == "テララ" and recs[1]["screen_type"] == "dialogue"
