"""The shared vocabulary of the translators: one piece of on-screen text to translate
(:class:`Item`), what the model is told about the game (:class:`Context`), the backend
interface and its errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from translate import languages


class TranslationError(Exception):
    """A translation failed. The message is plain text and tells the player what to do."""


class StopError(TranslationError):
    """A failure no retry mends until the player changes something: the key refused or
    missing, no credit left, the model not available, a language the service does not
    offer. The live translator sends nothing more after one."""


class BadAnswerError(TranslationError):
    """The model answered, but the answer cannot be used: not JSON, the wrong number of
    lines, cut off, or refused. Unlike a refused key or a dead connection this depends on
    the text that was sent, so the same lines in smaller batches usually go through."""


@dataclass
class Item:
    """One piece of on-screen text to translate, in the game's language. ``kind`` is
    "dialogue" or "ui"; ``speaker`` is the speaker's name as the game shows it, if known."""

    text: str
    kind: str = "ui"
    speaker: str | None = None


@dataclass
class Context:
    """What the model is told besides the items: the game, the language codes of the game's
    text and of the translation (``ja``, ``es-419``: :mod:`translate.languages`), its
    glossary as (source, translation) pairs, and the recent dialogue as (speaker, source,
    translation), oldest first."""

    game: str = ""
    source: str = "ja"
    target: str = "en"
    glossary: list[tuple[str, str]] = field(default_factory=list)
    recent: list[tuple[str | None, str, str]] = field(default_factory=list)

    @property
    def source_name(self) -> str:
        """The game's language in English ("Japanese"), for prompts."""
        return languages.source_name(languages.normalize(self.source))

    @property
    def target_name(self) -> str:
        """The translation's language in English ("Spanish (Latin America)"), for prompts."""
        return languages.target_name(languages.normalize(self.target))


class Backend(Protocol):
    """A translation service. ``translate`` returns one translation per item, in the items'
    order; ``check`` verifies the service works and returns a short status such as
    "Connected (Claude Haiku 5.5)". Both raise :class:`TranslationError` with a message the
    player can act on."""

    name: str

    def translate(self, items: list[Item], context: Context) -> list[str]: ...

    def check(self) -> str: ...
