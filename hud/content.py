"""What the player wants translated: the story (dialogue, subtitles, narration, the
player's choices) and/or everything else on screen (menus, labels, key hints).

A reading is told apart by the database line it matched: its screen type says so, and a
line a capture stored as UI still counts as story when it reads as prose (a sentence, not
a label: the narration between lines, a story summary). Text no line matches is judged by
the reading itself, the same way."""

from __future__ import annotations

from dataclasses import dataclass

from db.store import Line
from db.text import cjk_count

UI = "ui"  # a reading's kind when it is not story (live translation's "kind")
DIALOGUE = "dialogue"
STORY_TYPES = frozenset({DIALOGUE, "narration", "choice", "subtitle"})
PROSE_MIN_CHARS = 12  # source characters: shorter text is a label, a button, a name
SENTENCE_ENDS = ("。", "！", "？", "!", "?", "…", "」", "』", ".")


def is_prose(text: str) -> bool:
    """A sentence rather than a label."""
    t = text.strip()
    return cjk_count(t) >= PROSE_MIN_CHARS and t.endswith(SENTENCE_ENDS)


def is_story(line: Line) -> bool:
    return line.screen_type in STORY_TYPES or bool(line.speaker) or is_prose(line.source_text)


@dataclass
class Content:
    """The player's choice; both on unless they say otherwise."""

    dialogue: bool = True  # the story
    ui: bool = True  # menus and everything else

    @property
    def everything(self) -> bool:
        return self.dialogue and self.ui

    def allows(self, story: bool) -> bool:
        return self.dialogue if story else self.ui
