"""A shared database never comes in by itself: at start the app only asks the link's server
whether the file changed (nothing is downloaded) and asks the player; and whatever comes
in is checked first and copied as data (text and numbers), never run."""

from __future__ import annotations

import sqlite3

import pytest

from app.config import Settings
from sync.download import SyncError
from tests.test_sync import keys, make_db, server  # noqa: F401 - (the fixture)


# ---- the server is asked, nothing is downloaded ------------------------------------------
def test_the_link_is_asked_whether_the_file_changed(server, tmp_path):  # noqa: F811
    import os
    import time

    from sync.download import check_for_update, sync_database

    root, base = server
    link = f"{base}/shared.sqlite"
    shared = make_db(root / "shared.sqlite", ["一"])
    dest = tmp_path / "db" / "game.sqlite"
    assert check_for_update(link, {}) == "available"  # (nothing downloaded from it yet)
    _, known = sync_database(link, dest)
    assert check_for_update(link, known) == "same"
    shared.unlink()
    make_db(shared, ["一", "二"])  # the sharer uploads a new version
    later = time.time() + 5
    os.utime(shared, (later, later))
    assert check_for_update(link, known) == "newer"
    assert keys(dest) == ["一"]  # asked only: the database is as it was


class _Answer:
    def __init__(self, status: int, headers: dict[str, str]) -> None:
        self.status_code, self.headers = status, headers

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


@pytest.mark.parametrize(
    ("headers", "want"),
    [
        # Google Drive: Last-Modified and the size, no ETag, never a 304
        ({"Last-Modified": "Thu, 20 May 2021 11:16:29 GMT", "Content-Length": "4096"}, "same"),
        ({"Last-Modified": "Fri, 21 May 2021 09:00:00 GMT", "Content-Length": "4096"}, "newer"),
        ({"Last-Modified": "Thu, 20 May 2021 11:16:29 GMT", "Content-Length": "8192"}, "newer"),
        ({"Content-Length": "8192"}, "newer"),  # (another size is another file)
        ({"Content-Length": "4096"}, "unknown"),  # (the same size alone says nothing)
        ({}, "unknown"),
    ],
)
def test_what_the_headers_tell(monkeypatch, headers, want):
    import requests

    from sync.download import check_for_update, direct_url

    link = "https://drive.google.com/file/d/abcDEF123/view?usp=sharing"
    known = {"url": direct_url(link), "sha256": "x", "size": "4096",
             "last_modified": "Thu, 20 May 2021 11:16:29 GMT"}  # fmt: skip
    asked = []

    def get(url, **kw):  # noqa: ANN001, ANN003
        asked.append(kw)
        assert kw["stream"]  # (the body is never read)
        return _Answer(200, headers)

    monkeypatch.setattr(requests, "get", get)
    assert check_for_update(link, known) == want
    assert len(asked) == 1


def test_a_link_never_downloaded_from_is_asked_too(monkeypatch):
    """A database link that no longer works is never announced as a database to get."""
    import requests

    from sync.download import check_for_update

    link = "https://example.com/db.sqlite"
    monkeypatch.setattr(requests, "get", lambda url, **kw: _Answer(404, {}))
    with pytest.raises(SyncError, match="not found"):
        check_for_update(link, {})


# ---- indexes: one database per language ---------------------------------------------------
INDEX = {
    "game": "azur-promilia",
    "schema": 2,
    "databases": [
        {"lang": "en", "url": "azur-promilia.en.1.sqlite", "sha256": "AB", "size": 10,
         "version": "1"},
        {"lang": "es-419", "url": "https://cdn.example/azur-promilia.es-419.1.sqlite"},
        {"lang": "fr"},  # no url: left out
        "junk",
    ],
}


def test_an_index_lists_one_database_per_language():
    from sync.download import parse_index

    idx = parse_index(INDEX, "https://example.com/releases/download/database/index.json")
    assert [e.lang for e in idx.entries] == ["en", "es-419"]
    en = idx.entry_for("en")
    assert en.url == "https://example.com/releases/download/database/azur-promilia.en.1.sqlite"
    assert (en.sha256, en.size, en.version) == ("ab", 10, "1")
    assert idx.entry_for("es").lang == "es-419"  # a variant serves its language
    assert idx.entry_for("es-419").lang == "es-419"
    assert idx.entry_for("de") is None


def test_an_index_that_is_none_is_refused():
    from sync.download import parse_index

    for bad in ({}, {"databases": []}, {"databases": [{"lang": "en"}]}, [],
                {"databases": [{"lang": "en", "url": "file:///etc/passwd"}]}):
        with pytest.raises(SyncError, match="list of databases"):
            parse_index(bad, "https://example.com/index.json")


def test_a_download_that_is_not_the_one_listed_is_refused(server, tmp_path):  # noqa: F811
    from sync.download import sync_database

    root, base = server
    make_db(root / "shared.sqlite", ["一"])
    dest = tmp_path / "db" / "game.sqlite"
    with pytest.raises(SyncError, match="not the one the index lists"):
        sync_database(f"{base}/shared.sqlite", dest, expect_sha256="0" * 64)
    assert not dest.exists()


def test_a_link_that_fails_says_why(monkeypatch):
    import requests

    from sync.download import check_for_update

    link = "https://example.com/db.sqlite"
    monkeypatch.setattr(requests, "get", lambda url, **kw: _Answer(404, {}))
    with pytest.raises(SyncError, match="not found"):
        check_for_update(link, {"url": link, "sha256": "y"})


# ---- what comes in is checked and copied as data -----------------------------------------
def test_only_text_and_numbers_come_in(tmp_path):
    from db.store import Store
    from sync.download import adopt_database

    src = tmp_path / "shared.sqlite"
    conn = sqlite3.connect(src)
    conn.executescript(
        "CREATE TABLE lines (key TEXT, text_jp TEXT, en_text TEXT, seq INTEGER);"
        "CREATE TABLE meta (k TEXT, v TEXT);"
        "INSERT INTO meta VALUES ('target_lang', 'en');"
    )
    conn.execute("INSERT INTO lines VALUES ('はい', 'はい', 'Yes', 1)")
    conn.execute("INSERT INTO lines VALUES ('いいえ', 'いいえ', ?, 2)", (b"MZ\x90\x00",))
    conn.execute("INSERT INTO lines VALUES (?, 'x', 'x', 3)", (b"\x00\x01",))  # a blob key
    conn.execute("INSERT INTO lines VALUES ('長い', '長い', ?, 'four')", ("A" * 9000,))
    conn.execute("INSERT INTO meta VALUES ('blob', ?)", (b"\x00",))
    conn.commit()
    conn.close()
    dest = tmp_path / "db" / "game.sqlite"
    dest.parent.mkdir()
    adopt_database(src, dest)
    with Store(dest, readonly=True) as s:
        got = {ln.key: ln for ln in s.iter_lines()}
        meta = s.get_meta("blob")
    assert set(got) == {"はい", "いいえ", "長い"}  # (the row keyed by a blob is gone)
    assert got["はい"].translation == "Yes"
    assert got["いいえ"].translation is None  # a blob is no translation
    assert len(got["長い"].translation) == 4000 and got["長い"].seq is None
    assert meta is None


def test_a_file_that_is_no_database_is_never_used(tmp_path):
    from sync.download import adopt_database

    dest = make_db(tmp_path / "db" / "game.sqlite", ["古い"])
    fake = tmp_path / "database.sqlite"
    fake.write_bytes(b"MZ\x90\x00" + b"\x00" * 4096)  # a program renamed .sqlite
    with pytest.raises(SyncError, match="not a Game Translation HUD database"):
        adopt_database(fake, dest)
    damaged = tmp_path / "damaged.sqlite"
    good = make_db(tmp_path / "good.sqlite", [f"行{i}" for i in range(400)]).read_bytes()
    damaged.write_bytes(good[:4096] + bytes(len(good) - 4096))  # every page after the first
    with pytest.raises(SyncError):
        adopt_database(damaged, dest)
    assert keys(dest) == ["古い"]  # untouched


def test_a_file_far_too_big_is_never_downloaded(server, tmp_path, monkeypatch):  # noqa: F811
    import importlib

    dl = importlib.import_module("sync.download")  # (the package's `download` is a function)
    root, base = server
    make_db(root / "shared.sqlite", ["一"])
    monkeypatch.setattr(dl, "MAX_BYTES", 1000)
    dest = tmp_path / "db" / "game.sqlite"
    with pytest.raises(SyncError, match="far too big"):
        dl.sync_database(f"{base}/shared.sqlite", dest)
    assert not dest.exists() and not list(dest.parent.glob("*.download"))


# ---- the panel asks -------------------------------------------------------------------------
@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "data"))
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from app.panel import MainWindow
    from app.theme import apply_theme

    apply_theme(app)
    win = MainWindow(Settings(), desktop=False)
    for w in (win, win.handle):
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    yield win
    win.close_down()
    win.deleteLater()


def _wait(cond, seconds: float = 5.0) -> None:  # noqa: ANN001
    import time

    from PySide6.QtWidgets import QApplication

    end = time.time() + seconds
    while time.time() < end and not cond():
        QApplication.processEvents()
        time.sleep(0.01)


def test_the_start_window_asks_before_anything_is_downloaded(panel, monkeypatch):
    from PySide6.QtCore import Qt

    g = panel.library.add_game("Some Visual Novel")
    g.db_link = "https://drive.google.com/file/d/abcDEF123/view"
    panel.settings.current = g.id
    downloads = []
    monkeypatch.setattr(panel.library, "check_update", lambda game: ("available", None))
    monkeypatch.setattr(panel.library, "sync", lambda game, **kw: downloads.append(game) or
                        (False, {}, panel.library.db_dir(game) / "x.sqlite"))  # fmt: skip
    start = panel.start
    start.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    start.open()  # it looks at the link by itself, once
    _wait(lambda: not start.notice.isHidden())
    assert not start.notice.isHidden() and "is available" in start.notice_text.text()
    assert downloads == []  # asked, nothing fetched
    start.download()  # "Download it"
    _wait(lambda: downloads and start.task is None)
    assert downloads == [g] and start.notice.isHidden()


def test_no_offer_when_the_link_has_nothing_new(panel, monkeypatch):
    g = panel.library.add_game("Some Visual Novel")
    g.db_link = "https://example.com/db.sqlite"
    panel.settings.current = g.id
    start = panel.start
    start.fill()
    said = {"same": "Up to date", "unknown": "doesn't say", "none": "no English database"}
    for answer, text in said.items():
        monkeypatch.setattr(panel.library, "check_update", lambda game, a=answer: (a, None))
        start.check_updates()  # the player's own look
        _wait(lambda: start.check_task is None)
        assert start.notice.isHidden()
        assert text in start.msg.text()


# ---- a link to an index: the database in the player's language ----------------------------
def test_a_game_downloads_the_database_in_the_language_it_is_read_in(server, tmp_path,  # noqa: F811
                                                                     monkeypatch):
    import hashlib
    import json

    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "data"))
    from app.library import Library

    root, base = server
    en = make_db(root / "game.en.1.sqlite", ["一"], target_lang="en", source_lang="ja")
    es = make_db(root / "game.es-419.1.sqlite", ["一", "二"], target_lang="es-419",
                 source_lang="ja")

    def entry(path, lang):  # noqa: ANN001, ANN202
        data = path.read_bytes()
        return {"lang": lang, "url": path.name, "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "version": "1"}

    index = {"game": "game", "databases": [entry(en, "en"), entry(es, "es-419")]}
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    lib = Library(Settings())
    g = lib.add_game("Game")
    g.db_link, g.target_lang = f"{base}/index.json", "es"
    assert lib.check_update(g) == ("available", ["en", "es-419"])
    replaced, state, dest = lib.sync(g)
    lib.synced(g, state, dest)
    assert replaced and dest.name == "game.es-419.sqlite" and keys(dest) == ["一", "二"]
    assert lib.db_paths(g) == [dest] and lib.offered_languages(g) == ["en", "es-419"]
    assert lib.check_update(g)[0] == "same"
    assert lib.sync(g)[0] is False  # the listed file is the one here: not fetched again
    index["databases"][1] = entry(make_db(root / "game.es-419.2.sqlite", ["一", "二", "三"],
                                          target_lang="es-419"), "es-419")
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    assert lib.check_update(g)[0] == "newer"
    g.target_lang = "de"
    assert lib.check_update(g)[0] == "none"
    with pytest.raises(SyncError, match="no Deutsch database"):
        lib.sync(g)


def test_a_shared_drive_folder_works_as_an_index(server, tmp_path, monkeypatch):  # noqa: F811
    """The official databases are files in a shared Google Drive folder: the newest in the
    player's language comes in, and a newer file put in the folder is noticed."""
    import importlib

    from tests.test_sync import drive_page

    dl = importlib.import_module("sync.download")  # (the package's `download` is a function)

    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "data"))
    from app.library import Library

    root, base = server
    monkeypatch.setattr(dl, "DRIVE_FOLDER_VIEW", f"{base}/folder.html?id={{id}}")
    monkeypatch.setattr(dl, "DRIVE_DOWNLOAD", f"{base}/files/{{id}}")

    def upload(*files: tuple[str, list[str]]) -> None:
        for name, texts in files:
            make_db(root / "files" / name.replace(".", "_"), texts, target_lang="en")
        (root / "folder.html").write_text(drive_page(*(n for n, _ in files)), encoding="utf-8")

    upload(("game.en.1.sqlite", ["一"]), ("game.en.2.sqlite", ["一", "二"]))
    lib = Library(Settings())
    g = lib.add_game("Game")
    g.db_link = "https://drive.google.com/drive/folders/F00?usp=sharing"
    assert lib.check_update(g) == ("available", ["en"])
    replaced, state, dest = lib.sync(g)
    lib.synced(g, state, dest)
    assert replaced and dest.name == "game.en.sqlite" and keys(dest) == ["一", "二"]
    assert lib.check_update(g)[0] == "same"
    upload(("game.en.2.sqlite", ["一", "二"]), ("game.en.3.sqlite", ["一", "二", "三"]))
    assert lib.check_update(g)[0] == "newer"
    replaced, state, dest = lib.sync(g)
    assert replaced and keys(dest) == ["一", "二", "三"]
