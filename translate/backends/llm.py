"""What the language-model backends (Claude, ChatGPT) share: one request asked for again
once when its answer cannot be used, and how much room and time a request gets."""

from __future__ import annotations

import logging
from collections.abc import Callable

from translate.base import BadAnswerError, Item
from translate.prompts import fix_placeholders, parse_translations, placeholders_kept

log = logging.getLogger("translate")

MAX_TOKENS = 16000  # the most a request without streaming should ask for


def ask_twice(items: list[Item], ask: Callable[[str | None], str]) -> list[str]:
    """Run one request through ``ask`` and read its answer. ``ask`` gets None, or on the one
    retry what was wrong with the first answer. An answer that is not JSON or has the wrong
    number of lines is asked for once more and then raises :class:`BadAnswerError`; one that
    lost a placeholder is asked for once more and then kept as it is (logged)."""
    note: str | None = None
    kept: list[str] | None = None  # the first answer, when only its placeholders were off
    for attempt in (1, 2):
        text = ask(note)
        try:
            out = parse_translations(text, len(items))
        except BadAnswerError as exc:
            if attempt == 2:
                if kept is not None:
                    return kept
                raise
            note = str(exc)
            log.info("unusable answer for %d items (%s): asking again", len(items), exc)
            continue
        out = [fix_placeholders(it.text, t) for it, t in zip(items, out, strict=True)]
        lost = [i for i, (it, t) in enumerate(zip(items, out, strict=True), 1)
                if not placeholders_kept(it.text, t)]
        if not lost:
            return out
        if attempt == 2:
            log.warning("items %s lost a placeholder twice: keeping the answer", lost)
            return out
        kept = out
        which = ", ".join(map(str, lost))
        note = (f"item {which}" if len(lost) == 1 else f"items {which}") + (
            " did not keep the {player} / {n} / {s} placeholders exactly."
        )
        log.info("placeholders lost in items %s: asking again", which)
    raise AssertionError("unreachable")


def answer_budget(items: list[Item]) -> int:
    """max_tokens for a batch: room for a little thinking, the translation and the JSON."""
    chars = sum(len(it.text) for it in items)
    return min(MAX_TOKENS, 2048 + 4 * chars + 48 * len(items))


def time_budget(items: list[Item]) -> float:
    """Seconds an answer to this batch may take: the tokens it is likely to need (a little
    thinking, the translation, the JSON) at a slow 40 a second, plus 15 s to start. About
    30 s for one line, about 75 s for 30 lines of dialogue."""
    chars = sum(len(it.text) for it in items)
    return 15.0 + (600 + 2 * chars + 16 * len(items)) / 40.0
