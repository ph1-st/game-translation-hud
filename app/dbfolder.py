"""A game's database folder: every translation database in it, checked, with its language.

The overlay uses the databases in the player's language: that language's own first, then
other variants of it (a Latin American Spanish database serves a player who chose Spanish),
the newest first, the player's live translations last. A database in another language is
listed as not used, and why; so is any file that is no database. Files dropped into the
folder by hand count as soon as they are there.

Every file is checked once, again when its size or time changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from translate.languages import match, normalize

LIVE_PREFIX = "live-"  # live-<language>.sqlite: the player's live translations
DB_SUFFIXES = (".sqlite", ".sqlite3", ".db")
LEGACY_LANGUAGE = "en"  # databases from before app 1.2 say no language: they are English
# a file being written (a download, a swap, SQLite's journals) is not looked at yet
_IN_PROGRESS = ("-journal", "-wal", "-shm", ".download", ".adopting", ".part", ".tmp",
                ".crdownload")
_SYSTEM_FILES = {"desktop.ini", "thumbs.db", ".ds_store"}


@dataclass
class DbFile:
    """A file in a game's database folder, as checked."""

    path: Path
    ok: bool  # a translation database
    why: str = ""  # why it is none
    info: dict = field(default_factory=dict)  # sync.download.database_info

    @property
    def live(self) -> bool:
        return self.path.name.lower().startswith(LIVE_PREFIX)

    @property
    def lang(self) -> str:
        """The language it translates into (databases that do not say: English)."""
        return normalize(self.info.get("target_lang") or LEGACY_LANGUAGE)

    @property
    def lines(self) -> int:
        return int(self.info.get("lines") or 0)

    @property
    def translated(self) -> int:
        return int(self.info.get("translated") or 0)

    @property
    def version(self) -> str:
        return str(self.info.get("version") or "")


def live_name(lang: str) -> str:
    return f"{LIVE_PREFIX}{normalize(lang)}.sqlite"


class DatabaseFolder:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._checked: dict[str, tuple[tuple[int, int], DbFile]] = {}  # path -> (stat, check)

    def scan(self) -> list[DbFile]:
        """Every file in the folder: the databases (the newest first, live translations
        last), then the files that are none."""
        found: list[DbFile] = []
        try:
            paths = sorted(self.path.iterdir())
        except OSError:
            return []
        for path in paths:
            name = path.name.lower()
            if not path.is_file() or name in _SYSTEM_FILES or name.endswith(_IN_PROGRESS):
                continue
            found.append(self.check(path))
        ok = sorted((f for f in found if f.ok), key=lambda f: (f.live, -_mtime(f.path)))
        return ok + [f for f in found if not f.ok]

    def for_language(self, lang: str) -> list[DbFile]:
        """The databases a player who reads *lang* uses, the one that wins first: the
        language's own databases, then its other variants', then the live translations
        into exactly that language."""
        lang = normalize(lang)
        files = [f for f in self.scan() if f.ok]
        shared = [f for f in files if not f.live and match(f.lang, lang)]
        shared.sort(key=lambda f: match(f.lang, lang), reverse=True)  # (stable: newest first)
        live = [f for f in files if f.live and f.lang == lang]
        return shared + live

    def why_unused(self, f: DbFile, lang: str) -> str:
        """Why a file is not used for a player who reads *lang* ("" when it is)."""
        from i18n import _
        from translate.languages import native_name

        if not f.ok:
            return f.why
        used = f.lang == normalize(lang) if f.live else match(f.lang, lang) > 0
        if not used:
            return _("a {language} database").format(language=native_name(f.lang))
        return ""

    def languages(self) -> list[str]:
        """The languages the folder has databases in (live translations apart)."""
        return sorted({f.lang for f in self.scan() if f.ok and not f.live})

    def state(self, lang: str) -> tuple:
        """What the overlay's databases are now (live translations apart: the overlay
        writes those itself): when it changes, the overlay takes the folder anew."""
        out = []
        for f in self.for_language(lang):
            if f.live:
                continue
            try:
                st = f.path.stat()
            except OSError:
                continue
            out.append((f.path.name, st.st_size, st.st_mtime_ns))
        return tuple(out)

    def check(self, path: Path) -> DbFile:
        from i18n import _
        from sync.download import SyncError, check_file, database_info

        try:
            st = path.stat()
        except OSError:
            return DbFile(path, False, _("could not be read"))
        stamp = (st.st_size, st.st_mtime_ns)
        key = os.path.normcase(str(path))
        hit = self._checked.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        if not path.name.lower().endswith(DB_SUFFIXES):
            result = DbFile(path, False, _("not a database file"))
        else:
            try:
                # the app's own live translations: read as they are written (the full
                # checks are for files from elsewhere)
                live = path.name.lower().startswith(LIVE_PREFIX)
                info = database_info(path) if live else check_file(path)
                result = DbFile(path, True, "", info)
            except SyncError as e:
                result = DbFile(path, False, str(e))
            except Exception as e:  # noqa: BLE001 - whatever it is, it is not used
                result = DbFile(path, False, _("could not be read ({error})").format(
                    error=e.__class__.__name__))
        self._checked[key] = (stamp, result)
        return result

    def forget(self, path: Path) -> None:
        """Check *path* again next time (the app just wrote it)."""
        self._checked.pop(os.path.normcase(str(path)), None)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
