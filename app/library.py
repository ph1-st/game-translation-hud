"""The game library: every game has a folder of its own in the data folder.

    games/<id>/databases/        the translation databases (any language; each says which):
                                 the overlay uses those in the player's language
                                 (:mod:`app.dbfolder`). <game>.<lang>.sqlite comes from the
                                 game's link, live-<lang>.sqlite holds live translations;
                                 files the player drops in count too
    games/<id>/backups/          the databases a download or a file replaced
    games/<id>/cache/<lang>/     what the overlay learned about the game's screens
    games/<id>/memory-<lang>.json  what live translation remembers (context)
    games/<id>/misses/           the lines no database had, as seen while playing (JSONL)
    games/<id>/glossary.yaml     the player's own names and terms (optional)

A game uses a built-in profile (games/<profile>/profile.yaml in the app: screen layout,
dialogue boxes) or the generic one, which covers every text alike.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import yaml

from app import ONE_GAME
from app.config import GameEntry, Settings
from app.dbfolder import DatabaseFolder, DbFile, live_name
from app.paths import builtin_games_dir, games_data_dir
from capture.profile import Profile
from translate.languages import OCR, normalize, ocr_language

log = logging.getLogger("app")

GENERIC = "_generic"
DEFAULT_TARGET = "en"
DIRECT_DOWNLOAD = "database.sqlite"  # where a link to one database file downloads to


def parse_region(text: str) -> tuple[float, float, float, float] | None:
    """An area kept as "x,y,w,h" (fractions of the game's picture), if it is one."""
    try:
        x, y, w, h = (float(v) for v in str(text).split(","))
    except ValueError:
        return None
    if w < 0.01 or h < 0.01 or not (0 <= x < 1 and 0 <= y < 1):
        return None
    return x, y, min(w, 1 - x), min(h, 1 - y)


def region_text(rect: tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.4f}" for v in rect)


def with_region(raw: dict, area: tuple[float, float, float, float],
                name: tuple[float, float, float, float] | None, game: GameEntry) -> dict:
    """A profile whose dialogue is in the area the player chose: one layout over it,
    every text in it dialogue, the game's other dialogue settings replaced. Its name
    holds the area, so a new area starts afresh (what the cache learned of the old one, its
    font and colours, belonged to the old area)."""
    x, y, w, h = area
    lay = {
        "rows_top": [y, y + h], "panel": [x, y, w, h], "glass": [x, y, w, h],
        "anchor": "top", "typing": 30 if game.type_in else 0,
        "all_dialogue": True, "align": "auto", "keep": game.keep_glass,
        "reveal": game.type_in,
    }  # fmt: skip
    if name is not None:
        lay["speaker"] = list(name)
    out = dict(raw)
    regions = {
        k: v for k, v in (raw.get("regions") or {}).items()
        if str((v or {}).get("screen_type", "")) not in ("dialogue", "speaker", "speaker_npc")
    }
    regions["dialogue"] = {"x": x, "y": y, "w": w, "h": h, "screen_type": "dialogue",
                           "multiline": True}
    out["regions"] = regions
    tag = "".join(f"{round(v * 1000):03d}" for v in area)
    out["dialogue_layouts"] = {f"region-{tag}": lay}
    return out


def slug(name: str) -> str:
    """The folder name for a game called *name* (see config.GAME_ID_RE)."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48].strip("-")
    return s or "game"


class Library:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._folders: dict[str, DatabaseFolder] = {}
        self._profiles: dict[Path, tuple[float, dict]] = {}  # profile.yaml -> (mtime, data)

    # ---- folders -----------------------------------------------------------------------
    def folder(self, game: GameEntry) -> Path:
        d = games_data_dir() / game.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def db_dir(self, game: GameEntry) -> Path:
        """The game's database folder."""
        d = self.folder(game) / "databases"
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
        return d

    def databases(self, game: GameEntry) -> DatabaseFolder:
        """The game's database folder, checked (the checks are kept)."""
        folder = self._folders.get(game.id)
        if folder is None:
            self._migrate(game)
            folder = self._folders[game.id] = DatabaseFolder(self.db_dir(game))
        return folder

    def backups_dir(self, game: GameEntry) -> Path:
        return self.folder(game) / "backups"

    def live_path(self, game: GameEntry) -> Path:
        """The player's live translations into the language they read."""
        return self.db_dir(game) / live_name(self.target_lang(game))

    def memory_path(self, game: GameEntry) -> Path:
        return self.folder(game) / f"memory-{self.target_lang(game)}.json"

    def cache_dir(self, game: GameEntry) -> Path:
        """What the overlay learned of the game's screens, in the language read (the
        covers' layout and the remembered menus depend on the translation's words)."""
        d = self.folder(game) / "cache" / self.target_lang(game)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def misses_dir(self, game: GameEntry) -> Path:
        return self.folder(game) / "misses"

    def glossary_path(self, game: GameEntry) -> Path:
        """Names and terms for live translation and the dialogue's name box: the player's
        own glossary.yaml, else the preset's (English only)."""
        own = self.folder(game) / "glossary.yaml"
        if own.exists():
            return own
        return builtin_games_dir() / game.profile / "glossary.yaml"

    def _migrate(self, game: GameEntry) -> None:
        """Files from before databases were kept per language (app 1.1): the folder's
        database.sqlite and backups, live.sqlite, memory.json and the cache are English."""
        base, dbs = self.folder(game), self.db_dir(game)
        moves = [(base / "database.sqlite", dbs / DIRECT_DOWNLOAD),
                 (dbs / "live.sqlite", dbs / live_name(DEFAULT_TARGET)),
                 (base / "memory.json", base / f"memory-{DEFAULT_TARGET}.json")]
        moves += [(base / "cache" / n, base / "cache" / DEFAULT_TARGET / n)
                  for n in ("covers.json", "recall.npz")]
        moves += [(b, self.backups_dir(game) / b.name)
                  for b in base.glob("database.*.backup.sqlite")]
        for old, new in moves:
            if not old.exists() or new.exists():
                continue
            try:
                if old.suffix == ".sqlite":
                    from db.store import recover

                    recover(old)  # (a journal a cut-off write left is played back first)
                new.parent.mkdir(parents=True, exist_ok=True)
                os.replace(old, new)
            except OSError:
                log.exception("could not move %s to %s", old, new)

    # ---- the databases in use -----------------------------------------------------------
    def db_files(self, game: GameEntry) -> list[DbFile]:
        """The databases the overlay uses for the player's language, the one that wins
        first."""
        return self.databases(game).for_language(self.target_lang(game))

    def db_paths(self, game: GameEntry) -> list[Path]:
        return [f.path for f in self.db_files(game)]

    def db_state(self, game: GameEntry) -> tuple:
        return self.databases(game).state(self.target_lang(game))

    def info(self, game: GameEntry) -> dict:
        """What the databases in the player's language hold together: ``lines``,
        ``translated``, ``live`` (live translations), ``files`` (the shared ones used),
        ``version`` (the newest one's)."""
        files = self.db_files(game)
        used = [f for f in files if not f.live]
        return {
            "lines": sum(f.lines for f in used),
            "translated": sum(f.translated for f in used),
            "live": sum(f.translated for f in files if f.live),
            "files": len(used),
            "version": used[0].version if used else "",
        }

    # ---- languages ---------------------------------------------------------------------
    def source_lang(self, game: GameEntry) -> str:
        """The language the game shows (what the OCR reads): the player's choice, else
        what its profile reads."""
        if game.source_lang:
            return game.source_lang
        lang = str((self._raw_profile(game).get("ocr") or {}).get("lang", "japan"))
        return next((code for code, ocr in OCR.items() if ocr == lang), "ja")

    @staticmethod
    def target_lang(game: GameEntry) -> str:
        """The language the player reads."""
        return normalize(game.target_lang) if game.target_lang else DEFAULT_TARGET

    def offered_languages(self, game: GameEntry) -> list[str]:
        """Languages with a database for the game: in its folder, or at its link's index
        (when that was read: :meth:`index_languages`)."""
        return sorted(set(self.databases(game).languages()) | set(self.index_languages(game)))

    def index_languages(self, game: GameEntry) -> list[str]:
        known = self._sync_state(game).get("index", {})
        langs = known.get("languages") if isinstance(known, dict) else None
        return [str(x) for x in langs] if isinstance(langs, list) else []

    def take_source_language(self, game: GameEntry, path: Path) -> bool:
        """A database that came in says which language the game shows: the game takes it
        when the player set none. Returns whether it changed (the caller saves)."""
        if game.source_lang:
            return False
        f = self.databases(game).check(path)
        src = str(f.info.get("source_lang") or "") if f.ok else ""
        if src == "zh":
            src = "zh-hans"
        if src in OCR:
            game.source_lang = src
            return True
        return False

    # ---- profiles ----------------------------------------------------------------------
    def profiles(self) -> list[tuple[str, str, bool]]:
        """Built-in profiles: (id, display name, has a dialogue mode), generic last (an app
        made for one game: that game's only)."""
        out: list[tuple[str, str, bool]] = []
        for p in sorted(builtin_games_dir().glob("*/profile.yaml")):
            pid = p.parent.name
            if pid == GENERIC or (ONE_GAME and pid != ONE_GAME):
                continue
            d = self._read_yaml(p)
            out.append((pid, str(d.get("display_name", pid)), bool(d.get("dialogue_layouts"))))
        if not ONE_GAME:
            from i18n import _

            out.append((GENERIC, _("Any other game (no preset)"), False))
        return out

    def profile(self, game: GameEntry) -> Profile:
        """The game's profile, its window set to what the player picked, its OCR to the
        language the game shows, its dialogue area to the player's choices."""
        raw = self._raw_profile(game)
        win = dict(raw.get("window") or {})
        if game.window_title:
            win["title_patterns"] = [re.escape(game.window_title)]
            win["exact_titles"] = [game.window_title]
        if game.process_name:
            win["process_names"] = [game.process_name]
        raw["window"] = win
        if game.source_lang:  # the OCR reads the language the game shows
            raw["ocr"] = {**(raw.get("ocr") or {}), "lang": ocr_language(game.source_lang)}
        own = parse_region(game.region)
        if own is not None:
            raw = with_region(raw, own, parse_region(game.name_region), game)
        elif raw.get("dialogue_layouts"):  # the built-in area, with the player's choices
            raw["dialogue_layouts"] = {
                k: {**v, "keep": game.keep_glass, "reveal": game.type_in}
                for k, v in raw["dialogue_layouts"].items()
            }
        return Profile.from_dict(raw)

    def has_window(self, game: GameEntry) -> bool:
        """Whether the app knows how to find the game's window."""
        win = self.profile(game).window
        return bool(win.title_patterns or win.process_names)

    def find_window(self, game: GameEntry):  # noqa: ANN201 - hud.platform.WindowRef | None
        """The game's window, if it is open."""
        from hud.platform import find_game_window

        return find_game_window(self.profile(game).window)

    def _raw_profile(self, game: GameEntry) -> dict:
        pid = game.profile if (builtin_games_dir() / game.profile / "profile.yaml").exists() \
            else GENERIC
        return dict(self._read_yaml(builtin_games_dir() / pid / "profile.yaml"))

    def _read_yaml(self, path: Path) -> dict:
        """A profile file, read again only when it changed."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return {}
        hit = self._profiles.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}
        data = data if isinstance(data, dict) else {}
        self._profiles[path] = (mtime, data)
        return data

    def default_link(self, profile: str) -> str:
        """Where the community publishes a built-in profile's databases: its index of
        databases per language (``database_index``), else one database file
        (``database_url``). A game added with that profile downloads from it."""
        d = self._read_yaml(builtin_games_dir() / profile / "profile.yaml")
        for key in ("database_index", "database_url"):
            url = d.get(key)
            if isinstance(url, str) and url.strip():
                return url.strip()
        return ""

    def has_dialogue_mode(self, game: GameEntry) -> bool:
        """The game has a dialogue area: one the player chose, or its profile's own."""
        return bool(parse_region(game.region)) or self.builtin_region(game)

    def builtin_region(self, game: GameEntry) -> bool:
        return any(pid == game.profile and dlg for pid, _n, dlg in self.profiles())

    def cover_mode(self, game: GameEntry) -> str:
        """How the overlay covers dialogue (hud.pipeline.HudCore): "natural" without a
        dialogue area or with dialogue handling off, else the game's glass style."""
        if not (game.region_on and self.has_dialogue_mode(game)):
            return "natural"
        return "fixed" if game.glass_style == "window" else "grow"

    # ---- games -------------------------------------------------------------------------
    def add_game(self, name: str, profile: str = GENERIC, window_title: str = "",
                 process_name: str = "", player_name: str = "") -> GameEntry:
        base, n = slug(name), 1
        gid = base
        while self.settings.game(gid) is not None or (games_data_dir() / gid).exists():
            n += 1
            gid = f"{base}-{n}"
        game = GameEntry(gid, name.strip() or gid, profile, window_title, process_name,
                         player_name)
        game.db_link = self.default_link(profile)
        self.settings.games.append(game)
        self.db_dir(game)  # (its database folder, empty until a database comes in)
        self.settings.current = game.id
        self.settings.save()
        return game

    def remove_game(self, game: GameEntry, delete_files: bool = False) -> None:
        self.settings.games = [g for g in self.settings.games if g.id != game.id]
        if self.settings.current == game.id:
            self.settings.current = self.settings.games[0].id if self.settings.games else ""
        self.settings.save()
        self._folders.pop(game.id, None)
        if delete_files:
            shutil.rmtree(games_data_dir() / game.id, ignore_errors=True)

    def ensure_defaults(self) -> None:
        """On the first start: the games that have a built-in profile are in the library
        (a folder of a game already there, from settings that were lost, is taken back
        with its databases). An app made for one game has that game, and it is the one
        translated."""
        if ONE_GAME:
            mine = [g for g in self.settings.games if g.profile == ONE_GAME]
            if mine:
                if self.settings.current not in {g.id for g in mine}:
                    self.settings.current = mine[0].id
                    self.settings.save()
                return
        elif self.settings.games:
            return
        for pid, name, _dlg in self.profiles():
            if pid == GENERIC:
                continue
            gid = slug(name)
            old = games_data_dir() / gid
            if (old / "database.sqlite").exists() or any((old / "databases").glob("*")):
                game = GameEntry(gid, name, pid, db_link=self.default_link(pid))
                self.settings.games.append(game)
                self.settings.current = self.settings.current or gid
                self.settings.save()
            else:
                self.add_game(name, profile=pid)

    # ---- getting a database ------------------------------------------------------------
    def file_target(self, game: GameEntry, src: Path) -> Path:
        """Where a database file the player has goes in the game's folder."""
        from app.dbfolder import DB_SUFFIXES

        name = Path(src).name
        if not name.lower().endswith(DB_SUFFIXES):
            name += ".sqlite"
        return self.db_dir(game) / name

    def use_database_file(self, game: GameEntry, src: Path) -> Path:
        """Move a database file the player has into the game's database folder (any
        thread). It comes in as a download does (sync.download.adopt_database: checked, a
        file that is no database never goes in, and only its text and numbers are copied
        into a fresh file); the player's file then goes. A file of the same name there goes
        to the backups. Raises SyncError saying why a file is not used."""
        from sync.download import adopt_database, check_file

        src = Path(src)
        dest = self.file_target(game, src)
        if os.path.normcase(str(src.resolve())) == os.path.normcase(str(dest.resolve())):
            check_file(src)
            return dest  # (it is there already)
        adopt_database(src, dest, backups=self.backups_dir(game))
        try:
            src.unlink()
        except OSError:
            log.info("%s is in the folder; the original could not be removed", src.name)
        self.databases(game).forget(dest)
        return dest

    @staticmethod
    def db_game(game: GameEntry) -> str:
        """The name the game's databases go by (``azur-promilia.en.<version>.sqlite``): its
        preset's, else its own."""
        return game.profile if game.profile and game.profile != GENERIC else game.id

    def _sync_state(self, game: GameEntry) -> dict:
        try:
            state = json.loads(game.db_sync) if game.db_sync else {}
        except ValueError:
            return {}
        if not isinstance(state, dict):
            return {}
        if "url" in state:  # (app 1.1 kept one download: the direct link's)
            return {"": state}
        return state

    def sync(self, game: GameEntry, progress: Callable[[int, int | None], None] | None = None
             ) -> tuple[bool, dict, Path]:
        """Download the game's database in the player's language from its link (any
        thread; nothing is kept: hand the result to :meth:`synced` on the GUI thread).
        A link to an index takes the entry for the language read; a link to one database
        file takes that file. Returns (replaced, the state to keep, the database)."""
        from i18n import _
        from sync.download import SyncError, fetch_index, is_index_link, sync_database
        from translate.languages import native_name

        link = game.db_link.strip()
        if not link:
            raise SyncError(_("This game has no download link yet. Paste the link first."))
        state = self._sync_state(game)
        lang = self.target_lang(game)
        if is_index_link(link):
            index = fetch_index(link, game=self.db_game(game))
            entry = index.entry_for(lang)
            state["index"] = {"url": link, "languages": [e.lang for e in index.entries]}
            if entry is None:
                raise SyncError(_("There is no {language} database at this link yet. "
                                  "It has: {languages}.").format(
                    language=native_name(lang),
                    languages=", ".join(native_name(e.lang) for e in index.entries)))
            base = index.game or game.profile or game.id
            dest = self.db_dir(game) / f"{base}.{entry.lang}.sqlite"
            had = state.get(entry.lang) or {}
            if entry.sha256 and had.get("sha256") == entry.sha256 and dest.exists():
                return False, state, dest  # (the one listed is the one here)
            replaced, got = sync_database(
                entry.url, dest, state.get(entry.lang), progress=progress,
                backups=self.backups_dir(game), expect_sha256=entry.sha256,
            )  # fmt: skip
            state[entry.lang] = {**got, "version": entry.version}
        else:
            dest = self.db_dir(game) / DIRECT_DOWNLOAD
            replaced, got = sync_database(link, dest, state.get(""), progress=progress,
                                          backups=self.backups_dir(game))
            state[""] = got
        return replaced, state, dest

    def synced(self, game: GameEntry, state: dict, dest: Path) -> None:
        """What a download did, kept with the game (GUI thread)."""
        game.db_sync = json.dumps(state)
        self.databases(game).forget(dest)
        self.take_source_language(game, dest)
        self.settings.save()

    def check_update(self, game: GameEntry) -> tuple[str, list[str] | None]:
        """Whether the game's link has a database the player has not got yet (any thread;
        only the server's answer or the index is read, no database is downloaded):
        "newer", "available" (none downloaded yet), "same", "unknown" (the host does not
        say), or "none" (no database in the player's language there); and, for an index,
        the languages it lists (hand them to :meth:`remember_index` on the GUI thread)."""
        from i18n import _
        from sync.download import SyncError, check_for_update, fetch_index, is_index_link

        link = game.db_link.strip()
        if not link:
            raise SyncError(_("This game has no download link yet. Paste the link first."))
        state = self._sync_state(game)
        lang = self.target_lang(game)
        if not is_index_link(link):
            known = state.get("") or {}
            if not (self.db_dir(game) / DIRECT_DOWNLOAD).exists():
                known = {}
            return check_for_update(link, known), None
        index = fetch_index(link, game=self.db_game(game))
        langs = [e.lang for e in index.entries]
        entry = index.entry_for(lang)
        if entry is None:
            return "none", langs
        had = state.get(entry.lang) or {}
        have = any(f.lang == entry.lang and not f.live for f in self.db_files(game))
        if not (had.get("sha256") and have):
            return "available", langs
        if entry.sha256:
            return ("same" if had["sha256"] == entry.sha256 else "newer"), langs
        if entry.version:
            return ("same" if had.get("version") == entry.version else "newer"), langs
        return "unknown", langs

    def remember_index(self, game: GameEntry, languages: list[str]) -> bool:
        """The languages the game's link offers, as its index said (GUI thread). Returns
        whether they changed."""
        state = self._sync_state(game)
        old = state.get("index", {}).get("languages") if isinstance(state.get("index"),
                                                                     dict) else None
        if old == languages:
            return False
        state["index"] = {"url": game.db_link.strip(), "languages": languages}
        game.db_sync = json.dumps(state)
        self.settings.save()
        return True

