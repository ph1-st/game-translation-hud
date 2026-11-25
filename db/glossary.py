"""A game's glossary: its names and terms in the language players read, looked up with
tolerance for OCR misreadings.

Entries come from a database's names table (:meth:`Glossary.from_names`) or from a YAML
file: a list of ``{<source language>: term, <target language>: rendering, type, note}`` as
the lore pass writes them (older files say ``jp`` and ``en``). An entry in another variant of
the target language counts too (``es-419`` for ``es``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz, process

from db.normalize import normalize_key
from db.store import Name, Store

log = logging.getLogger(__name__)

# the keys a file may name the game's own text by (old files: jp)
SOURCE_KEYS = ("ja", "jp", "zh-hans", "zh-hant", "zh", "ko")
NOT_SOURCES = {"type", "note", "notes", "count", "reviewed", "fixed", "translation"}
FUZZY_CUTOFF = 80.0  # a misread name still finds its entry above this similarity
MIN_FUZZY_LEN = 2  # names shorter than this are only taken exactly


class Glossary:
    def __init__(self, entries: Iterable[Name] = ()) -> None:
        """*entries*: names and terms; the first of two with the same source wins."""
        self.entries: list[Name] = []
        self.by_key: dict[str, str] = {}
        for e in entries:
            key = normalize_key(e.source) if e.source else ""
            if key and e.translation and key not in self.by_key:
                self.entries.append(e)
                self.by_key[key] = e.translation
        self.keys = list(self.by_key)

    # ---- making one ------------------------------------------------------------------
    @classmethod
    def from_names(cls, names: Iterable[Name]) -> Glossary:
        """The glossary of a database's names table (:meth:`db.store.Store.names`)."""
        return cls(names)

    @classmethod
    def from_dbs(cls, paths: Iterable[str | Path]) -> Glossary:
        """The names tables of databases, the first holding a name winning."""
        names: list[Name] = []
        for path in paths:
            with Store(path, readonly=True) as store:
                names += store.names()
        return cls(names)

    @classmethod
    def from_entries(cls, entries: Iterable[Any], target: str) -> Glossary:
        """The glossary of a file's entries, in *target* (entries without a rendering in
        that language are left out)."""
        return cls(e for e in (entry_name(x, target) for x in entries) if e is not None)

    @classmethod
    def load(cls, path: str | Path, target: str) -> Glossary:
        """The glossary in the YAML file at *path*, in *target*. A missing file, or one
        that cannot be read (a typo in a hand-edited file), is an empty glossary: the
        overlay runs without it (the log says why)."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            with open(p, encoding="utf-8-sig") as f:
                data = yaml.safe_load(f) or []
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
            log.warning("glossary %s not read: %s", p, e)
            return cls()
        return cls.from_entries(data if isinstance(data, list) else [], target)

    @classmethod
    def combine(cls, *glossaries: Glossary) -> Glossary:
        """One glossary of several, the first holding an entry winning."""
        return cls(e for g in glossaries for e in g.entries)

    # ---- using one -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def pairs(self) -> list[tuple[str, str]]:
        """(source, translation) of every entry: what the live translators are given."""
        return [(e.source, e.translation) for e in self.entries]

    def characters(self) -> list[Name]:
        """The entries that are characters' names (all entries when none says so)."""
        people = [e for e in self.entries if e.type == "character"]
        return people or list(self.entries)

    def translate(self, text: str, cutoff: float = FUZZY_CUTOFF) -> str | None:
        """The rendering of a name or term, tolerating OCR noise; None if unknown."""
        key = normalize_key(text)
        if not key:
            return None
        hit = self.by_key.get(key)
        if hit is not None:
            return hit
        if len(key) < MIN_FUZZY_LEN or not self.keys:
            return None
        best = process.extractOne(key, self.keys, scorer=fuzz.ratio, score_cutoff=cutoff)
        return self.by_key[best[0]] if best else None


def entry_name(entry: Any, target: str) -> Name | None:
    """A glossary file's entry as a :class:`Name` in *target*: its source term (the first of
    :data:`SOURCE_KEYS` that is not the target), its rendering (the target's key, a variant
    of the same language, or ``translation``); None when it has no rendering there."""
    if not isinstance(entry, dict):
        return None
    target = target.lower()
    source = next((str(entry[k]) for k in SOURCE_KEYS
                   if entry.get(k) and not _same_language(k, target)), None)
    rendering = entry.get(target) or entry.get("translation")
    if not rendering:
        rendering = next((v for k, v in entry.items()
                          if isinstance(k, str) and v and k not in NOT_SOURCES
                          and k not in SOURCE_KEYS and _same_language(k, target)), None)
    if not source or not rendering:
        return None
    return Name(source.strip(), str(rendering).strip(), str(entry.get("type") or "term"),
                str(entry.get("note") or entry.get("notes") or ""))


def _same_language(a: str, b: str) -> bool:
    """es-419 and es, en and en-gb: the same language in another variant (jp is ja)."""
    def primary(code: str) -> str:
        p = code.lower().replace("_", "-").split("-")[0]
        return "ja" if p == "jp" else p

    return primary(a) == primary(b)
