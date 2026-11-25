"""SQLite storage for a translation database.

Tables:

* ``lines``: one row per unique normalized key (what the HUD looks up)
* ``names``: the names and terms of the game in the database's language (its glossary)
* ``occurrences``: every captured appearance in script order (what the translator reads)
* ``meta``: what the database is (:class:`DbInfo`)

Tiers: machine < reviewed < human. A higher tier always overrides a lower one.

Schema 2 names its columns after their role (``source_text``, ``speaker``, ``translation``);
schema 1 (until app 1.1) called them ``text_jp``, ``speaker_jp`` and ``en_text``. A schema 1
file is migrated when opened for writing and read through aliases when opened read-only.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
FORMAT = "gthud-db"
BUSY_S = 10.0  # seconds a connection waits for another one's write
TIERS = {"machine": 0, "reviewed": 1, "human": 2}
# schema 1 column -> schema 2 column (lines and occurrences)
V1_COLUMNS = {"text_jp": "source_text", "speaker_jp": "speaker", "en_text": "translation"}
# a shared database must say which game and which languages it is for
REQUIRED_META = ("game", "source_lang", "target_lang")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
CREATE TABLE IF NOT EXISTS lines (
    key TEXT PRIMARY KEY,
    source_text TEXT NOT NULL,
    speaker TEXT,
    screen_type TEXT NOT NULL DEFAULT 'dialogue',
    is_template INTEGER NOT NULL DEFAULT 0,
    translation TEXT,
    tier TEXT NOT NULL DEFAULT 'machine',
    source TEXT NOT NULL DEFAULT 'batch',
    chapter TEXT,
    seq INTEGER,
    game_version TEXT,
    updated_at TEXT,
    samples INTEGER NOT NULL DEFAULT 0,
    ocr_confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_lines_seq ON lines (seq);
CREATE TABLE IF NOT EXISTS names (
    source TEXT PRIMARY KEY,
    translation TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'term',
    notes TEXT
);
CREATE TABLE IF NOT EXISTS occurrences (
    seq INTEGER PRIMARY KEY,
    key TEXT NOT NULL,
    source_text TEXT NOT NULL,
    speaker TEXT,
    screen_type TEXT NOT NULL,
    source TEXT,
    session_id TEXT,
    timestamp REAL,
    chapter TEXT,
    game_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_occ_key ON occurrences (key);
"""
LINE_COLUMNS = (
    "key", "source_text", "speaker", "screen_type", "is_template", "translation", "tier",
    "source", "chapter", "seq", "game_version", "updated_at", "samples", "ocr_confidence",
)  # fmt: skip


@dataclass(slots=True)
class Line:
    key: str  # the normalized, templated source text (db.templates.templatize)
    source_text: str  # the text as the game shows it
    speaker: str | None
    screen_type: str
    is_template: bool
    translation: str | None
    tier: str
    source: str  # where the translation came from: batch, live, import, capture...
    chapter: str | None
    seq: int | None  # script order
    game_version: str | None
    samples: int = 0  # captures: how often it was read
    ocr_confidence: float | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Line":
        return cls(
            key=row["key"],
            source_text=row["source_text"],
            speaker=row["speaker"],
            screen_type=row["screen_type"],
            is_template=bool(row["is_template"]),
            translation=row["translation"],
            tier=row["tier"],
            source=row["source"],
            chapter=row["chapter"],
            seq=row["seq"],
            game_version=row["game_version"],
            samples=row["samples"] or 0,
            ocr_confidence=row["ocr_confidence"],
        )


@dataclass(slots=True)
class Name:
    source: str  # as the game writes it
    translation: str
    type: str = "term"  # character, place, term, item...
    notes: str = ""


@dataclass(slots=True)
class DbInfo:
    """What a database says about itself (its meta table) and how much it holds."""

    game: str = ""  # the profile id (azur-promilia)
    source_lang: str = ""  # the game's language: ja, zh-hans, zh-hant, ko
    target_lang: str = ""  # the translation's language: en, es-419, pt-br...
    display_name: str = ""  # shown in the app ("Azur Promilia, English")
    version: str = ""  # the database release (2026.11.25-1)
    game_version: str = ""  # the client it was made for
    text_source: str = ""  # where its source text came from
    created: str = ""
    updated: str = ""
    generator: str = ""  # app version and model that made it
    notice: str = ""  # what reusing it takes (the terms of the model that translated it)
    schema_version: int = 1
    lines: int = 0
    translated: int = 0

    @property
    def complete(self) -> bool:
        """Whether it says which game and languages it is for (shared databases must)."""
        return all(getattr(self, k) for k in REQUIRED_META)


META_FIELDS = tuple(
    f.name for f in fields(DbInfo) if f.name not in ("schema_version", "lines", "translated")
)


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def recover(path: str | Path) -> None:
    """A write cut off (the app ended while one ran) leaves a journal beside the database
    that a read-only connection cannot play back ("attempt to write a readonly database"):
    a writable connection plays it back first."""
    path = Path(path)
    if not path.with_name(path.name + "-journal").exists():
        return
    try:
        conn = sqlite3.connect(str(path), timeout=BUSY_S)
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        pass  # (the read-only open says what is wrong)


def table_columns(conn: sqlite3.Connection, table: str, schema: str = "main") -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA {schema}.table_info({table})")}


def select_lines(cols: set[str], table: str = "lines") -> str:
    """A SELECT of *table* that gives schema 2 column names whatever the file's schema
    (``text_jp AS source_text``; a column the file lacks is NULL)."""
    old = {new: old for old, new in V1_COLUMNS.items()}
    parts = []
    for c in LINE_COLUMNS:
        if c in cols:
            parts.append(c)
        elif old.get(c) in cols:
            parts.append(f"{old[c]} AS {c}")
        else:
            parts.append(f"NULL AS {c}")
    return f"SELECT {', '.join(parts)} FROM {table}"


def migrate_v1(conn: sqlite3.Connection) -> bool:
    """Rename schema 1 columns in place. Returns whether anything changed."""
    changed = False
    for table in ("lines", "occurrences"):
        cols = table_columns(conn, table)
        for old, new in V1_COLUMNS.items():
            if old in cols and new not in cols:
                conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
                changed = True
    return changed


class Store:
    def __init__(self, path: str | Path, readonly: bool = False) -> None:
        """*readonly*: for reading only (the HUD loading its lines): nothing is written, not
        even the schema, and a missing file is an error instead of a new empty database."""
        self.path = Path(path)
        self.readonly = readonly
        if readonly:
            if not self.path.is_file():
                raise FileNotFoundError(f"no database at {self.path}")
            recover(self.path)
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, timeout=BUSY_S)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), timeout=BUSY_S)
        self.conn.row_factory = sqlite3.Row
        if not readonly:
            migrate_v1(self.conn)
            self.conn.executescript(SCHEMA)
            if self.get_meta("schema_version") != str(SCHEMA_VERSION):
                self.set_meta("schema_version", str(SCHEMA_VERSION))
                self.set_meta("format", FORMAT)
            self.conn.commit()  # (an open write would lock everyone else out meanwhile)
        self._select = select_lines(table_columns(self.conn, "lines"))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, *exc: object) -> None:
        try:
            if exc_type is None and not self.readonly:
                self.conn.commit()
            else:
                self.conn.rollback()  # a failure halfway leaves the database as it was
        finally:
            self.close()

    def commit(self) -> None:
        self.conn.commit()

    # ---- meta --------------------------------------------------------------------------
    def set_meta(self, k: str, v: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (k, v))

    def get_meta(self, k: str) -> str | None:
        try:
            row = self.conn.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
        except sqlite3.OperationalError:  # a file without a meta table
            return None
        return row["v"] if row else None

    def set_info(self, **values: str) -> None:
        """Set meta fields of :class:`DbInfo` (unknown names are an error)."""
        for k, v in values.items():
            if k not in META_FIELDS:
                raise KeyError(f"not a database meta field: {k}")
            self.set_meta(k, str(v))

    def info(self) -> DbInfo:
        info = DbInfo()
        for k in META_FIELDS:
            v = self.get_meta(k)
            if v:
                setattr(info, k, v.strip())
        try:
            info.schema_version = int(self.get_meta("schema_version") or 1)
        except ValueError:
            info.schema_version = 1
        info.target_lang = info.target_lang.lower()
        info.source_lang = info.source_lang.lower()
        n = self.count()
        info.lines, info.translated = n["lines"], n["translated"]
        return info

    # ---- lines -------------------------------------------------------------------------
    def upsert_line(self, line: Line) -> None:
        """Insert or merge a line. Keeps the better OCR reading; never touches an existing
        translation (see :meth:`set_translation`)."""
        row = self.conn.execute("SELECT * FROM lines WHERE key = ?", (line.key,)).fetchone()
        if row is None:
            self.conn.execute(
                f"INSERT INTO lines ({', '.join(LINE_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(LINE_COLUMNS))})",
                (
                    line.key, line.source_text, line.speaker, line.screen_type,
                    int(line.is_template), line.translation, line.tier, line.source,
                    line.chapter, line.seq, line.game_version, now_iso(), line.samples,
                    line.ocr_confidence,
                ),
            )  # fmt: skip
            return
        old = Line.from_row(row)
        better_reading = (line.samples, line.ocr_confidence or 0.0) > (
            old.samples,
            old.ocr_confidence or 0.0,
        )
        self.conn.execute(
            """UPDATE lines SET source_text = ?, speaker = ?, samples = MAX(samples, ?),
               ocr_confidence = MAX(COALESCE(ocr_confidence, 0), ?), seq = COALESCE(seq, ?),
               chapter = COALESCE(chapter, ?), updated_at = ? WHERE key = ?""",
            (
                line.source_text if better_reading else old.source_text,
                old.speaker or line.speaker,
                line.samples,
                line.ocr_confidence or 0.0,
                line.seq,
                line.chapter,
                now_iso(),
                line.key,
            ),
        )

    def set_translation(self, key: str, translation: str, tier: str,
                        source: str = "batch") -> bool:
        """Store a translation unless a higher tier already exists. Returns True if written."""
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r} (one of {', '.join(TIERS)})")
        row = self.conn.execute(
            "SELECT tier, translation FROM lines WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return False
        # (a tier this app does not know, from a database made elsewhere, ranks as machine)
        if row["translation"] and TIERS.get(row["tier"], 0) > TIERS[tier]:
            return False
        self.conn.execute(
            "UPDATE lines SET translation = ?, tier = ?, source = ?, updated_at = ? "
            "WHERE key = ?",
            (translation, tier, source, now_iso(), key),
        )
        return True

    def get_line(self, key: str) -> Line | None:
        row = self.conn.execute(self._select + " WHERE key = ?", (key,)).fetchone()
        return Line.from_row(row) if row else None

    def iter_lines(self, screen_type: str | None = None) -> Iterable[Line]:
        q, args = self._select, ()
        if screen_type:
            q, args = q + " WHERE screen_type = ?", (screen_type,)
        for row in self.conn.execute(q + " ORDER BY seq, key", args):
            yield Line.from_row(row)

    def count(self) -> dict[str, int]:
        cols = table_columns(self.conn, "lines")
        if not cols:
            return {"lines": 0, "translated": 0}
        tr = "translation" if "translation" in cols else "en_text"
        total, translated = self.conn.execute(
            f"SELECT COUNT(*), COUNT({tr}) FROM lines"
        ).fetchone()
        return {"lines": total, "translated": translated}

    # ---- names -------------------------------------------------------------------------
    def set_names(self, names: Iterable[Name], replace: bool = True) -> int:
        """Store the game's names and terms (the glossary in the database's language)."""
        if replace:
            self.conn.execute("DELETE FROM names")
        n = 0
        for nm in names:
            if nm.source.strip() and nm.translation.strip():
                self.conn.execute(
                    "INSERT OR REPLACE INTO names (source, translation, type, notes) "
                    "VALUES (?, ?, ?, ?)",
                    (nm.source.strip(), nm.translation.strip(), nm.type or "term",
                     nm.notes or None),
                )
                n += 1
        return n

    def names(self) -> list[Name]:
        try:
            rows = self.conn.execute(
                "SELECT source, translation, type, notes FROM names ORDER BY source"
            ).fetchall()
        except sqlite3.OperationalError:  # schema 1: no names table
            return []
        return [Name(r["source"], r["translation"], r["type"] or "term", r["notes"] or "")
                for r in rows]

    # ---- occurrences -------------------------------------------------------------------
    def next_seq(self) -> int:
        row = self.conn.execute("SELECT MAX(seq) FROM occurrences").fetchone()
        return (row[0] or 0) + 1

    def add_occurrence(self, seq: int, key: str, rec: dict[str, Any]) -> None:
        """*rec*: a capture record (capture.run: text, speaker, screen_type...)."""
        self.conn.execute(
            """INSERT OR REPLACE INTO occurrences (seq, key, source_text, speaker, screen_type,
               source, session_id, timestamp, chapter, game_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                seq,
                key,
                record_text(rec),
                record_speaker(rec),
                rec.get("screen_type", "dialogue"),
                rec.get("source"),
                rec.get("session_id"),
                rec.get("timestamp"),
                rec.get("chapter"),
                rec.get("game_version"),
            ),
        )

    def iter_occurrences(self, screen_type: str | None = None) -> Iterable[sqlite3.Row]:
        q, args = "SELECT * FROM occurrences", ()
        if screen_type:
            q, args = q + " WHERE screen_type = ?", (screen_type,)
        yield from self.conn.execute(q + " ORDER BY seq", args)


def record_text(rec: dict[str, Any]) -> str:
    """The text of a capture record (``text``; captures before app 1.2 wrote ``text_jp``)."""
    v = rec.get("text", rec.get("text_jp"))
    return v if isinstance(v, str) else ""


def record_speaker(rec: dict[str, Any]) -> str | None:
    v = rec.get("speaker", rec.get("speaker_jp"))
    return v.strip() or None if isinstance(v, str) else None
