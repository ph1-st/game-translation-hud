"""sync.download: share links, downloads from a local web server, database info."""

from __future__ import annotations

import functools
import random
import sqlite3
import threading
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from db.store import Line, Store
from sync import SyncError, database_info, direct_url, download

DRIVE_ID = "1AbC_dEf-GhIjKlMnOpQrStUvWxYz"
DRIVE = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"
DROPBOX = "https://www.dropbox.com"


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        # Google Drive: every form of a file link -> the direct download
        (f"https://drive.google.com/file/d/{DRIVE_ID}/view?usp=sharing", DRIVE),
        (f"https://drive.google.com/file/d/{DRIVE_ID}/view?usp=drive_link", DRIVE),
        (f"https://drive.google.com/file/d/{DRIVE_ID}", DRIVE),
        (f"https://drive.google.com/file/d/{DRIVE_ID}/", DRIVE),
        (f"https://drive.google.com/open?id={DRIVE_ID}", DRIVE),
        (f"https://drive.google.com/uc?id={DRIVE_ID}&export=download", DRIVE),
        (f"https://drive.google.com/uc?export=download&id={DRIVE_ID}", DRIVE),
        (
            f"https://drive.google.com/file/d/{DRIVE_ID}/view?usp=sharing&resourcekey=0-k3y_X",
            DRIVE + "&resourcekey=0-k3y_X",
        ),
        # Dropbox: dl=1, everything else kept
        (f"{DROPBOX}/s/abc123/game.sqlite?dl=0", f"{DROPBOX}/s/abc123/game.sqlite?dl=1"),
        (
            f"{DROPBOX}/scl/fi/xyz789/game.sqlite?rlkey=k3y&st=s7t&dl=0",
            f"{DROPBOX}/scl/fi/xyz789/game.sqlite?rlkey=k3y&st=s7t&dl=1",
        ),
        (
            f"{DROPBOX}/scl/fi/xyz789/game.sqlite?rlkey=k3y",
            f"{DROPBOX}/scl/fi/xyz789/game.sqlite?rlkey=k3y&dl=1",
        ),
        # GitHub: file page -> raw; release assets are direct already
        (
            "https://github.com/owner/repo/blob/main/db/game.sqlite",
            "https://raw.githubusercontent.com/owner/repo/main/db/game.sqlite",
        ),
        (
            "https://github.com/owner/repo/blob/v1.2/game.sqlite?raw=true",
            "https://raw.githubusercontent.com/owner/repo/v1.2/game.sqlite",
        ),
        (
            "https://github.com/owner/repo/releases/download/v1/game.sqlite",
            "https://github.com/owner/repo/releases/download/v1/game.sqlite",
        ),
        # OneDrive: redir / embed -> download; short links redirect by themselves
        (
            "https://onedrive.live.com/redir?resid=ABC!123&authkey=!XyZ",
            "https://onedrive.live.com/download?resid=ABC!123&authkey=!XyZ",
        ),
        (
            "https://onedrive.live.com/embed?resid=ABC!123&authkey=!XyZ",
            "https://onedrive.live.com/download?resid=ABC!123&authkey=!XyZ",
        ),
        ("https://1drv.ms/u/s!AbCdEfGh", "https://1drv.ms/u/s!AbCdEfGh"),
        # anything else: as it is, trimmed
        ("  https://example.com/files/game.sqlite \n", "https://example.com/files/game.sqlite"),
        ("<https://example.com/game.sqlite>", "https://example.com/game.sqlite"),
        ("http://127.0.0.1:8000/game.sqlite", "http://127.0.0.1:8000/game.sqlite"),
        (
            "https://docs.google.com/spreadsheets/d/abc/edit",
            "https://docs.google.com/spreadsheets/d/abc/edit",
        ),
    ],
)
def test_direct_url(link, expected):
    assert direct_url(link) == expected


@pytest.mark.parametrize(
    "link",
    [
        "",
        "   ",
        "ftp://example.com/game.sqlite",
        "file:///C:/data/game.sqlite",
        r"C:\Users\me\game.sqlite",
        "www.dropbox.com/s/abc123/game.sqlite?dl=0",
        "not a link",
        "https://drive.google.com/drive/folders/1AbCdEfGh",  # a folder, not the file
    ],
)
def test_direct_url_rejects(link):
    with pytest.raises(SyncError):
        direct_url(link)


# ---- a shared Google Drive folder: an index of the databases in it -------------------------
def drive_page(*names: str) -> str:
    """A folder's file list as Google Drive shows it on other sites."""
    rows = "".join(
        f'<div class="flip-entry" id="entry-{n.replace(".", "_")}" tabindex="0"><div class='
        f'"flip-entry-info"><a href="#"><div class="flip-entry-title">{n}</div></a></div>'
        f'<div class="flip-entry-last-modified"><div>3:23 pm</div></div></div>' for n in names)
    return f'<html><body><div class="flip-entries">{rows}</div></body></html>'


@pytest.mark.parametrize(("link", "folder"), [
    ("https://drive.google.com/drive/folders/1kxJOPALzPs?usp=sharing", "1kxJOPALzPs"),
    ("https://drive.google.com/drive/u/0/folders/Ab_c-1", "Ab_c-1"),
    ("<https://drive.google.com/folderview?id=Xy9>", "Xy9"),
    ("https://drive.google.com/file/d/1AbC/view", None),
    ("https://example.com/drive/folders/1AbC", None),
])
def test_a_drive_folder_link_is_known_by_its_id(link, folder):
    from sync.download import drive_folder_id, is_index_link

    assert drive_folder_id(link) == folder
    assert is_index_link(link) == (folder is not None)


def test_a_drive_folder_serves_the_newest_database_of_each_language():
    from sync.download import parse_drive_folder

    page = drive_page("game.en.2026.11.25-prelaunch.1.sqlite", "game.en.2026.11.26.1.sqlite",
                      "game.en.2026.11.25.2.sqlite", "game.es-419.1.sqlite", "notes.txt",
                      "other.en.9.sqlite")
    idx = parse_drive_folder(page, "https://drive.google.com/drive/folders/F", game="game")
    assert idx.game == "game"
    assert [(e.lang, e.version) for e in idx.entries] == [("en", "2026.11.26.1"),
                                                         ("es-419", "1")]
    assert idx.entries[0].url.endswith("id=game_en_2026_11_26_1_sqlite&export=download&confirm=t")
    # a folder of one game serves it whatever the game is called here
    alone = parse_drive_folder(drive_page("game.en.1.sqlite"), "link", game="my-game")
    assert alone.game == "game"


@pytest.mark.parametrize(("page", "why"), [
    ("<html>Sign in</html>", "share it"),
    (drive_page("notes.txt"), "no translation database"),
    (drive_page("a.en.1.sqlite", "b.en.1.sqlite"), "several games"),
])
def test_a_drive_folder_without_this_games_database_says_why(page, why):
    from sync.download import parse_drive_folder

    with pytest.raises(SyncError, match=why):
        parse_drive_folder(page, "link", game="game")


def test_database_versions_compare_as_numbers_with_prereleases_first():
    from sync.download import version_key

    order = ["2026.11.25-prelaunch.1", "2026.11.25-prelaunch.2", "2026.11.25",
             "2026.11.25.1", "2026.11.26.1", "2026.12.1"]
    assert sorted(order, key=version_key) == order


# ---- downloads from a local web server --------------------------------------------------
class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@pytest.fixture
def server(tmp_path, monkeypatch):
    """(folder, base URL) of a web server on 127.0.0.1 that serves the folder."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")  # never through a system proxy
    root = tmp_path / "www"
    root.mkdir()
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def make_db(path: Path, texts: list[str], **meta: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Store(path) as store:
        for seq, jp in enumerate(texts, 1):
            store.upsert_line(
                Line(
                    key=jp, source_text=jp, speaker=None, screen_type="ui", is_template=False,
                    translation=None, tier="machine", source="batch", chapter=None, seq=seq,
                    game_version=None,
                )
            )  # fmt: skip
        for k, v in meta.items():
            store.set_meta(k, v)
    return path


def keys(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [row[0] for row in conn.execute("SELECT key FROM lines ORDER BY seq")]
    finally:
        conn.close()


def test_download_replaces_database_and_keeps_backup(server, tmp_path):
    root, base = server
    shared = make_db(root / "shared.sqlite", [f"行{i}" for i in range(3000)])  # > one chunk
    dest = make_db(tmp_path / "db" / "game.sqlite", ["古い"])
    calls: list[tuple[int, int | None]] = []

    out = download(f"{base}/shared.sqlite", dest, progress=lambda d, t: calls.append((d, t)))

    assert out == dest
    assert keys(dest) == [f"行{i}" for i in range(3000)]
    (backup,) = dest.parent.glob("game.*.backup.sqlite")  # the old one, with the time
    assert keys(backup) == ["古い"]
    assert sorted(p.name for p in dest.parent.iterdir()) == sorted([backup.name, "game.sqlite"])
    size = shared.stat().st_size
    assert len(calls) > 2 and calls[0] == (0, size) and calls[-1] == (size, size)
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)


def test_first_download_makes_the_folder(server, tmp_path):
    root, base = server
    make_db(root / "shared.sqlite", ["設定"])
    dest = tmp_path / "new" / "folder" / "game.sqlite"
    download(f"  {base}/shared.sqlite  ", dest)
    assert keys(dest) == ["設定"]
    assert [p.name for p in dest.parent.iterdir()] == ["game.sqlite"]  # nothing to back up


@pytest.mark.parametrize("name", ["page.html", "looks-like.sqlite"])
def test_download_rejects_a_web_page(server, tmp_path, name):
    root, base = server
    page = "<!DOCTYPE html>\n<html><head><title>Sign in</title></head><body>Sign in</body></html>"
    (root / name).write_text(page, encoding="utf-8")
    dest = make_db(tmp_path / "db" / "game.sqlite", ["古い"])
    before = dest.read_bytes()

    with pytest.raises(SyncError, match="web page instead of the database file"):
        download(f"{base}/{name}", dest)

    assert dest.read_bytes() == before
    assert [p.name for p in dest.parent.iterdir()] == ["game.sqlite"]  # no temp file, no backup


def test_download_not_found(server, tmp_path):
    _, base = server
    dest = tmp_path / "db" / "game.sqlite"
    with pytest.raises(SyncError, match="404"):
        download(f"{base}/missing.sqlite", dest)
    assert not dest.exists() and not list(dest.parent.glob("*"))


def _random_bytes(path: Path) -> None:
    path.write_bytes(random.Random(7).randbytes(50_000))


def _sqlite_without_lines(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE notes (body TEXT)")
    conn.execute("INSERT INTO notes VALUES ('hello')")
    conn.commit()
    conn.close()


def _damaged_sqlite(path: Path) -> None:
    path.write_bytes(b"SQLite format 3\x00" + random.Random(7).randbytes(8000))


@pytest.mark.parametrize("make", [_random_bytes, _sqlite_without_lines, _damaged_sqlite])
def test_download_rejects_other_files(server, tmp_path, make):
    root, base = server
    make(root / "shared.sqlite")
    dest = make_db(tmp_path / "db" / "game.sqlite", ["古い"])
    before = dest.read_bytes()

    with pytest.raises(SyncError, match="not a Game Translation HUD database"):
        download(f"{base}/shared.sqlite", dest)

    assert dest.read_bytes() == before
    assert [p.name for p in dest.parent.iterdir()] == ["game.sqlite"]


def test_progress_callback_can_cancel(server, tmp_path):
    root, base = server
    make_db(root / "shared.sqlite", [f"行{i}" for i in range(3000)])
    dest = tmp_path / "db" / "game.sqlite"

    class Cancelled(Exception):
        pass

    def cancel(done: int, total: int | None) -> None:
        if done:
            raise Cancelled

    with pytest.raises(Cancelled):
        download(f"{base}/shared.sqlite", dest, progress=cancel)
    assert not list(dest.parent.glob("*"))


# ---- database info ----------------------------------------------------------------------
def test_database_info(tmp_path):
    path = make_db(
        tmp_path / "game.sqlite",
        ["設定", "はい", "いいえ"],
        game="azur-promilia",
        display_name="Azur Promilia",
        version="3",
        updated="2026-10-16T10:00:00+00:00",
    )
    with Store(path) as store:
        store.set_translation("設定", "Settings", "machine")
        store.set_translation("はい", "Yes", "human")
    mtime = path.stat().st_mtime_ns

    info = database_info(path)

    assert {k: info[k] for k in ("lines", "translated", "game", "display_name", "version",
                                 "updated", "size_bytes")} == {
        "lines": 3,
        "translated": 2,
        "game": "azur-promilia",
        "display_name": "Azur Promilia",
        "version": "3",
        "updated": "2026-10-16T10:00:00+00:00",
        "size_bytes": path.stat().st_size,
    }
    assert info["schema_version"] == 2 and info["target_lang"] == ""
    assert path.stat().st_mtime_ns == mtime  # read-only: the file is not touched


def test_database_info_without_meta(tmp_path):
    path = make_db(tmp_path / "bare.sqlite", ["設定"])
    info = database_info(path)
    assert (info["game"], info["display_name"], info["version"]) == ("", "", "")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(microsecond=0)
    assert info["updated"] == mtime.isoformat()
    assert (info["lines"], info["translated"]) == (1, 0)


def test_database_info_errors(tmp_path):
    with pytest.raises(SyncError, match="no database"):
        database_info(tmp_path / "missing.sqlite")
    notes = tmp_path / "notes.sqlite"
    notes.write_text("just some text, not a database", encoding="utf-8")
    with pytest.raises(SyncError):
        database_info(notes)
    _sqlite_without_lines(tmp_path / "other.sqlite")
    with pytest.raises(SyncError, match="not a Game Translation HUD database"):
        database_info(tmp_path / "other.sqlite")


def test_the_same_database_again_is_no_change(server, tmp_path):
    from sync.download import sync_database

    root, base = server
    make_db(root / "shared.sqlite", ["新しい"])
    dest = make_db(tmp_path / "db" / "game.sqlite", ["古い"])
    replaced, known = sync_database(f"{base}/shared.sqlite", dest)
    assert replaced and keys(dest) == ["新しい"] and known["sha256"]
    # the next start: the server says nothing changed (Last-Modified), nothing is fetched
    replaced, again = sync_database(f"{base}/shared.sqlite", dest, known, only_if_newer=True)
    assert not replaced and again["sha256"] == known["sha256"]
    # "Sync now" asks for the file in full: the same file is still no reason for a backup
    for b in dest.parent.glob("game.*.backup.sqlite"):
        b.unlink()
    replaced, _ = sync_database(f"{base}/shared.sqlite", dest, again)
    assert not replaced
    assert sorted(p.name for p in dest.parent.iterdir()) == ["game.sqlite"]


def test_a_newer_database_on_the_server_comes_in(server, tmp_path):
    import os
    import time

    from sync.download import sync_database

    root, base = server
    shared = make_db(root / "shared.sqlite", ["一"])
    dest = tmp_path / "db" / "game.sqlite"
    _, known = sync_database(f"{base}/shared.sqlite", dest)
    shared.unlink()
    make_db(shared, ["一", "二"])  # the maintainer uploads a new version
    later = time.time() + 5
    os.utime(shared, (later, later))
    replaced, _ = sync_database(f"{base}/shared.sqlite", dest, known, only_if_newer=True)
    assert replaced and keys(dest) == ["一", "二"]


def test_a_synced_database_is_the_shared_one_only(server, tmp_path):
    """Live translations live in their own file per language (app.library): a download
    replaces the shared database as it is, never mixed with anything of the old one."""
    from db.store import Line, Store
    from sync.download import sync_database

    root, base = server
    make_db(root / "shared.sqlite", ["一", "二"])
    dest = make_db(tmp_path / "db" / "game.sqlite", ["一"])
    with Store(dest) as s:
        line = Line("三", "三", None, "ui", False, "Three", "machine", "live", None, 9, None)
        s.upsert_line(line)
    sync_database(f"{base}/shared.sqlite", dest)
    with Store(dest, readonly=True) as s:
        assert sorted(ln.key for ln in s.iter_lines()) == ["一", "二"]


def test_a_schema_1_database_comes_in_as_schema_2_with_its_names(tmp_path):
    import sqlite3

    from db.store import Store
    from sync.download import adopt_database

    old = tmp_path / "old.sqlite"
    conn = sqlite3.connect(old)
    conn.executescript(
        "CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);"
        "INSERT INTO meta VALUES ('game', 'g'), ('target_lang', 'es-419'), "
        "('source_lang', 'ja'), ('schema_version', '1');"
        "CREATE TABLE lines (key TEXT PRIMARY KEY, text_jp TEXT, speaker_jp TEXT, "
        "en_text TEXT, tier TEXT, source TEXT);"
        "INSERT INTO lines VALUES ('はい', 'はい', 'ルミィ', 'Sí', 'human', 'batch');"
        "CREATE TABLE names (source TEXT, translation TEXT, type TEXT);"
        "INSERT INTO names VALUES ('ルミィ', 'Lumie', 'character');"
    )
    conn.close()
    dest = tmp_path / "db" / "game.sqlite"
    adopt_database(old, dest)
    with Store(dest, readonly=True) as s:
        line = s.get_line("はい")
        info = s.info()
        names = s.names()
    assert (line.source_text, line.speaker, line.translation, line.tier) == (
        "はい", "ルミィ", "Sí", "human")
    assert (info.schema_version, info.target_lang, info.complete) == (2, "es-419", True)
    assert [(n.source, n.translation, n.type) for n in names] == [("ルミィ", "Lumie", "character")]


def test_a_schema_1_file_opened_for_writing_is_migrated_in_place(tmp_path):
    import sqlite3

    from db.store import Store, table_columns

    path = tmp_path / "v1.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE lines (key TEXT PRIMARY KEY, text_jp TEXT NOT NULL, speaker_jp TEXT, "
        "screen_type TEXT NOT NULL DEFAULT 'dialogue', is_template INTEGER NOT NULL DEFAULT 0,"
        " en_text TEXT, tier TEXT NOT NULL DEFAULT 'machine', source TEXT NOT NULL DEFAULT "
        "'batch', chapter TEXT, seq INTEGER, game_version TEXT, updated_at TEXT, samples "
        "INTEGER NOT NULL DEFAULT 0, ocr_confidence REAL);"
        "INSERT INTO lines (key, text_jp, en_text) VALUES ('一', '一', 'One');"
    )
    conn.close()
    with Store(path, readonly=True) as s:  # read through aliases, file untouched
        assert s.get_line("一").translation == "One"
    with Store(path) as s:
        assert {"source_text", "translation", "speaker"} <= table_columns(s.conn, "lines")
        assert s.get_line("一").source_text == "一"


def test_a_database_that_is_a_view_or_misses_columns_never_comes_in(server, tmp_path):
    import sqlite3

    from sync.download import adopt_database

    bad = tmp_path / "bad.sqlite"
    conn = sqlite3.connect(bad)
    conn.executescript(
        "CREATE TABLE t (key TEXT, text_jp TEXT, en_text TEXT);"
        "CREATE VIEW lines AS SELECT * FROM t;"
    )
    conn.close()
    dest = make_db(tmp_path / "db" / "game.sqlite", ["古い"])
    with pytest.raises(SyncError):
        adopt_database(bad, dest)
    assert keys(dest) == ["古い"]  # untouched
    thin = tmp_path / "thin.sqlite"  # only the three columns: filled in as this app needs
    conn = sqlite3.connect(thin)
    conn.executescript(
        "CREATE TABLE lines (key TEXT, text_jp TEXT, en_text TEXT);"
        "INSERT INTO lines VALUES ('はい', 'はい', 'Yes');"
    )
    conn.close()
    adopt_database(thin, dest)
    from db.store import Store

    with Store(dest, readonly=True) as s:
        (line,) = [ln for ln in s.iter_lines() if ln.key == "はい"]
    assert line.translation == "Yes" and line.tier == "machine"


def test_restoring_the_oldest_backup_never_loses_it(tmp_path):
    import os

    from sync.download import adopt_database, prune_backups

    dest = make_db(tmp_path / "db" / "game.sqlite", ["今"])
    olds = []
    for i, stamp in enumerate(("20260101-000000", "20260102-000000", "20260103-000000")):
        b = make_db(dest.with_name(f"game.{stamp}.backup.sqlite"), [f"古{i}"])
        olds.append(b)
    oldest = olds[0]
    adopt_database(oldest, dest)  # "Use a database file": the oldest backup
    assert keys(dest) == ["古0"] and oldest.exists()
    left = sorted(p.name for p in dest.parent.glob("game.*.backup.sqlite"))
    assert len(left) == 4  # three kept, and the one restored
    # made in the same second: the one with -1 is the newer
    a = dest.with_name("game.20990104-000000.backup.sqlite")
    b = dest.with_name("game.20990104-000000-1.backup.sqlite")
    for p in (a, b):
        make_db(p, ["x"])
    os.utime(a, (1, 1))
    prune_backups(dest, keep=1)
    assert [p.name for p in dest.parent.glob("game.*.backup.sqlite")] == [b.name]


def test_a_failed_swap_keeps_every_backup(tmp_path, monkeypatch):
    import importlib

    dl = importlib.import_module("sync.download")  # (the package's "download" is a function)

    dest = make_db(tmp_path / "db" / "game.sqlite", ["今"])
    for stamp in ("20260101-000000", "20260102-000000", "20260103-000000"):
        make_db(dest.with_name(f"game.{stamp}.backup.sqlite"), ["古"])
    new = make_db(tmp_path / "new.sqlite", ["新"])

    def refuse(*_a):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(dl.os, "replace", refuse)
    with pytest.raises(SyncError):
        dl.adopt_database(new, dest)
    assert keys(dest) == ["今"]
    assert len(list(dest.parent.glob("game.*.backup.sqlite"))) == 3  # none pruned, none added


def test_a_cut_off_write_never_stops_the_database_from_opening(tmp_path):
    import shutil
    import sqlite3

    from db.store import Store

    db = make_db(tmp_path / "game.sqlite", ["一", "二"])
    # a write cut off: the journal of an unfinished transaction stays beside the file
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA cache_size = 1")
    conn.execute("BEGIN")
    conn.executemany("INSERT INTO lines (key, source_text) VALUES (?, ?)",
                     [(f"k{i}" * 200, "x" * 400) for i in range(300)])  # fmt: skip
    journal = db.with_name(db.name + "-journal")
    assert journal.exists()
    shutil.copy2(journal, tmp_path / "hot")
    snapshot = db.read_bytes()
    conn.rollback()
    conn.close()
    db.write_bytes(snapshot)  # the file as the cut-off write left it, with its journal
    shutil.copy2(tmp_path / "hot", journal)
    with Store(db, readonly=True) as s:
        assert sorted(ln.key for ln in s.iter_lines()) == ["一", "二"]
