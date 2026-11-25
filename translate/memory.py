"""Per-game context memory for the live translators: the glossary and the last lines of
dialogue.

Every request carries both, so a line translated on its own still knows the names, who is
talking and what was just said. Persisted as JSON next to the other per-game caches, one
file per target language::

    {"version": 2, "game": "azur-promilia", "target": "es-419",
     "recent": [{"speaker": "テララ", "source": "...", "translation": "..."}],
     "glossary": [["テララ", "Terara"]]}

A missing, damaged or foreign file (another game, another language) starts the memory
empty; version 1 files (``jp``/``en`` keys, no language) are read as English. All methods
are thread-safe.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections import deque
from pathlib import Path

from translate import languages
from translate.base import Context

log = logging.getLogger("translate")

VERSION = 2
V1_TARGET = "en"  # what version 1 files (English only) were written in


class GameMemory:
    """The translation context of one game into one language. The file at ``path`` (None:
    kept in memory only) is read at once; :meth:`save` writes it back."""

    def __init__(self, path: str | Path | None, game: str = "", max_recent: int = 40,
                 source: str = "ja", target: str = "en") -> None:
        """*source* / *target*: the codes of the game's language and of the one to
        translate into (:mod:`translate.languages`)."""
        self.path = Path(path) if path is not None else None
        self.game = game
        self.source = languages.normalize(source)
        self.target = languages.normalize(target)
        self.max_recent = max(0, int(max_recent))
        self._recent: deque[tuple[str | None, str, str]] = deque(maxlen=self.max_recent)
        self._glossary: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self.load()

    # ---- persistence -----------------------------------------------------------------
    def load(self) -> None:
        """Replace the memory with the file's content. A missing, unreadable or damaged
        file, or one written for another game or language, leaves it empty; never raises."""
        recent: list[tuple[str | None, str, str]] = []
        glossary: list[tuple[str, str]] = []
        data = self._read()
        if isinstance(data, dict) and data.get("version") in (1, VERSION):
            other_game = data.get("game")
            other_target = data.get("target", V1_TARGET if data.get("version") == 1 else None)
            if self.game and other_game and other_game != self.game:
                log.info("translation memory %s is for %s: starting empty", self.path, other_game)
            elif other_target and languages.normalize(str(other_target)) != self.target:
                log.info("translation memory %s is in %s: starting empty", self.path,
                         other_target)
            else:
                recent = _read_recent(data.get("recent"))
                glossary = _read_glossary(data.get("glossary"))
        elif data is not None:
            log.info("translation memory %s has another format: starting empty", self.path)
        with self._lock:
            self._recent = deque(recent, maxlen=self.max_recent)
            self._glossary = glossary

    def _read(self) -> object:
        if self.path is None:
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as exc:  # whatever is wrong with the file, never stop the HUD
            log.warning("translation memory %s is unreadable (%s): starting empty",
                        self.path, exc)
            return None

    def save(self) -> bool:
        """Write the memory to its file (through a temporary file, so a crash never leaves
        half a file). Returns False, with a warning in the log, if it could not."""
        if self.path is None:
            return False
        with self._lock:
            data = {
                "version": VERSION,
                "game": self.game,
                "target": self.target,
                "recent": [{"speaker": s, "source": src, "translation": t}
                           for s, src, t in self._recent],
                "glossary": [[src, t] for src, t in self._glossary],
            }
        with self._io_lock:
            tmp = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(
                    prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent
                )
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
                return True
            except OSError as exc:  # e.g. another HUD instance holds the file
                log.warning("could not save translation memory %s: %s", self.path, exc)
                if tmp is not None:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                return False

    # ---- content ---------------------------------------------------------------------
    def remember(self, source: str, translation: str, speaker: str | None = None,
                 kind: str = "dialogue") -> None:
        """Add a translated line to the recent conversation. Only dialogue is kept; a line
        that repeats the last one exactly is skipped, and the oldest lines fall off."""
        if kind != "dialogue":
            return
        source, translation = (source or "").strip(), (translation or "").strip()
        if not source or not translation:
            return
        entry = ((speaker or "").strip() or None, source, translation)
        with self._lock:
            # (the same line again: as translated, then as shown, its source as read)
            if self._recent and self._recent[-1][2] == translation:
                return
            self._recent.append(entry)

    def set_glossary(self, entries: list[tuple[str, str]]) -> None:
        """Replace the glossary with these (source, translation) pairs; empty ones are
        dropped."""
        glossary = _read_glossary(entries)
        with self._lock:
            self._glossary = glossary

    def context(self) -> Context:
        """A snapshot of the memory to send along with a request."""
        with self._lock:
            return Context(game=self.game, glossary=list(self._glossary),
                           recent=list(self._recent), source=self.source, target=self.target)

    @property
    def recent(self) -> list[tuple[str | None, str, str]]:
        """The recent dialogue as (speaker, source, translation), oldest first."""
        with self._lock:
            return list(self._recent)

    @property
    def glossary(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._glossary)

    def clear(self) -> None:
        """Forget the conversation (the glossary stays)."""
        with self._lock:
            self._recent.clear()


def _read_recent(value: object) -> list[tuple[str | None, str, str]]:
    out: list[tuple[str | None, str, str]] = []
    for e in value if isinstance(value, list) else []:
        if not isinstance(e, dict):
            continue
        source = e.get("source", e.get("jp"))
        translation = e.get("translation", e.get("en"))
        speaker = e.get("speaker")
        if not (isinstance(source, str) and isinstance(translation, str)
                and source.strip() and translation.strip()):
            continue
        speaker = (speaker.strip() or None) if isinstance(speaker, str) else None
        out.append((speaker, source.strip(), translation.strip()))
    return out


def _read_glossary(value: object) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for e in value if isinstance(value, (list, tuple)) else []:
        if isinstance(e, dict):
            pair = (e.get("source", e.get("jp")), e.get("translation", e.get("en")))
        else:
            pair = e
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        source, translation = pair
        if (isinstance(source, str) and isinstance(translation, str) and source.strip()
                and translation.strip()):
            out.append((source.strip(), translation.strip()))
    return out
