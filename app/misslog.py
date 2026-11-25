"""The lines no database had, as they were seen while playing.

Written down on this computer only (never sent anywhere): one JSON-lines file a day in the
game's ``misses`` folder, one record a line (``text``, ``speaker``, ``screen_type``,
``timestamp``...): what the next version of the game's database should add (a player may
send the files to whoever makes the database). A text is written once: numbers and the
player's name are left out of what makes two readings the same, so a timer counting down
is one line.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from db.templates import templatize

log = logging.getLogger("app")

MAX_KEYS = 50_000  # texts remembered to write each once (beyond that: a fresh start)


class MissLog:
    def __init__(self, folder: Path, game: str, lang: str, player: str | None = None) -> None:
        self.folder = Path(folder)
        self.game, self.lang, self.player = game, lang, player
        self._lock = threading.Lock()
        self._keys: set[str] | None = None  # read from the folder when first needed

    def add(self, text: str, kind: str, speaker: str | None = None) -> bool:
        """Write *text* down unless it was already (any thread). Returns whether it was."""
        key = templatize(text, self.player).key
        if not key:
            return False
        with self._lock:
            keys = self._known()
            if key in keys:
                return False
            if len(keys) >= MAX_KEYS:
                keys.clear()
            keys.add(key)
            record = {
                "timestamp": round(time.time(), 3), "game": self.game,
                "target_lang": self.lang, "screen_type": kind, "speaker": speaker,
                "text": text, "source": "hud",
            }  # fmt: skip
            try:
                self.folder.mkdir(parents=True, exist_ok=True)
                with open(self.folder / f"{time.strftime('%Y-%m-%d')}.jsonl", "a",
                          encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                log.exception("could not write down an untranslated line")
                return False
        return True

    def count(self) -> int:
        with self._lock:
            return len(self._known())

    def _known(self) -> set[str]:
        if self._keys is None:
            self._keys = {templatize(r.get("text", ""), self.player).key
                          for r in read_records(files(self.folder))}
            self._keys.discard("")
        return self._keys


def files(folder: Path) -> list[Path]:
    """The miss logs in *folder*, oldest first."""
    try:
        return sorted(Path(folder).glob("*.jsonl"))
    except OSError:
        return []


def read_records(paths: list[Path]) -> Iterator[dict]:
    """The records of miss logs (a line cut off by a crash is skipped)."""
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and isinstance(rec.get("text"), str):
                        yield rec
        except OSError:
            continue
