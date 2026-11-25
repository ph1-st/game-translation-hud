"""What the translation models are told, and how their answers are read.

Claude and the local models get the same two messages. The system prompt holds what only
changes with the game, the rules and the glossary, so the API can cache it. The user
message holds the recent conversation and the numbered items, each tagged with its kind and
speaker::

    1. [ui] 設定
    2. [dialogue, Terara] いまなら…

The answer is JSON, ``{"translations": ["...", ...]}``, one string per item, in order.
:func:`parse_translations` finds it however a model wraps it (code fences, a sentence
before or after, numbered keys).

Database lines that differ only in the player's name, a number or a value the game fills
in share one template, so items carry the placeholders ``{player}``, ``{n}`` and ``{s}`` and
the translation must keep them (the HUD fills them in again: :func:`db.templates.fill`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from db.normalize import normalize_key
from db.templates import NUMBER, PLAYER, WORD
from translate import languages
from translate.base import BadAnswerError, Context, Item

# How a language variant is to be written, where its name alone does not say it.
STYLE_NOTES = {
    "es-419": "Write neutral Latin American Spanish that reads naturally in every Latin "
              "American country: ustedes (never vosotros), no regional slang or idioms "
              "particular to one country.",
    "es": "Write Spanish as it is written in Spain (vosotros where plural you is meant).",
    "pt-br": "Write Brazilian Portuguese.",
    "pt-pt": "Write European Portuguese.",
    "en-gb": "Write British English (British spelling).",
}


def style_note(target: str) -> str:
    """What the translator should know about writing *target* (a language code), or ""."""
    return STYLE_NOTES.get(languages.normalize(target), "")


def label_case(target: str) -> str:
    """How short labels are written: English titles take Title Case; other languages keep
    their own capitalization (Spanish menus are in sentence case)."""
    if languages.primary(target) == "en":
        return " in Title Case"
    if languages.cased(target):
        return " (capitalized as names and menu entries are in that language)"
    return ""


def _rules(context: Context) -> list[str]:
    src, tgt = context.source_name, context.target_name
    rules = [
        f"- Write natural, idiomatic {tgt}, the way an official {tgt} release would read. "
        "Do not translate word for word.",
        "- Each item is tagged with its kind: [ui] for interface text, [dialogue, Speaker] for "
        "a spoken line.",
        "- UI text (menus, buttons, tabs, labels, item and skill names, quest titles) has "
        "little room on screen: use the shortest natural wording"
        f"{label_case(context.target)} for names and short labels. UI text that is a full "
        "sentence (tips, descriptions) stays a sentence.",
        "- Dialogue is spoken by characters: keep each speaker's voice, tone and personality, "
        "and keep the expressive punctuation (…, !, ?).",
        f"- Keep the placeholders {PLAYER} (the player's name), {NUMBER} (a number) and {WORD} "
        "(a value the game fills in) exactly as written and as many times as they appear, "
        f"placed where {tgt} grammar puts them. Never translate, spell out or drop them.",
        f"- Use the glossary's {tgt} for every name and term it lists, spelled exactly as "
        "given. Render names it does not list consistently.",
        "- The text was read off the screen by OCR and may contain small misreadings: translate "
        f"what it most likely says. Copy text that is not {src} (numbers, codes, Latin-letter "
        "words) as it is.",
        "- Translate each numbered item on its own; items are often unrelated pieces of screen "
        "text. The recent conversation is context only (who is speaking, pronouns, tone): do "
        "not translate it again.",
    ]
    note = style_note(context.target)
    if note:
        rules.insert(1, f"- {note}")
    return rules


def _answer_rule(tgt: str) -> str:
    return ('Answer with JSON only, without code fences or comments: {"translations": '
            f'["...", "..."]}}, exactly one {tgt} string per numbered item, in the same order.')


# ---- the request ---------------------------------------------------------------------------
def build_system_prompt(context: Context, only_for: list[Item] | None = None) -> str:
    """The static part of the request: the rules and the glossary. It changes only with the
    game, so Claude caches it. ``only_for`` keeps just the glossary entries that occur in
    these items (text or speaker), for local models with a small context window."""
    src, tgt = context.source_name, context.target_name
    lines = [
        f"You translate the {src} text of {_game_title(context.game)} into {tgt} for a "
        f"fan-made overlay that draws your {tgt} over the {src} on screen.",
        "",
        "Rules:",
        *_rules(context),
        "",
        _answer_rule(tgt),
    ]
    glossary = _glossary(context.glossary, only_for)
    if glossary:
        lines += ["", f"Glossary ({src} = {tgt}):"]
        lines += [f"{source} = {translation}" for source, translation in glossary]
    return "\n".join(lines)


def build_user_message(
    items: list[Item],
    context: Context,
    max_recent: int | None = None,
    note: str | None = None,
) -> str:
    """The changing part of the request: the recent conversation (the last ``max_recent``
    lines, all if None) and the numbered items. ``note`` says what was wrong with a previous
    answer, for the one retry."""
    names = {normalize_key(s): t for s, t in context.glossary if s and t}
    recent = context.recent
    if max_recent is not None:
        recent = recent[-max_recent:] if max_recent > 0 else []
    parts: list[str] = []
    if recent:
        parts.append("Recent conversation, oldest first (already translated, context only):")
        for speaker, source, translation in recent:
            who = _speaker(speaker, names)
            line = f"{_one_line(source)} => {_one_line(translation)}"
            parts.append(f"{who}: {line}" if who else line)
        parts.append("")
    n = len(items)
    parts.append("Translate this item:" if n == 1 else f"Translate these {n} items:")
    for i, item in enumerate(items, 1):
        tag = item.kind or "ui"
        who = _speaker(item.speaker, names)
        if who:
            tag = f"{tag}, {who}"
        parts.append(f"{i}. [{tag}] {_one_line(item.text)}")
    if note:
        parts += ["", retry_note(note, n)]
    return "\n".join(parts)


def retry_note(problem: str, expected: int) -> str:
    """The reminder sent with the one retry after an unusable answer."""
    strings = "string" if expected == 1 else "strings"
    return (
        f"Your previous answer could not be used: {problem} Answer again with JSON only, "
        f'{{"translations": [...]}} with exactly {expected} {strings}, one per numbered item, '
        "in order."
    )


def _game_title(game: str) -> str:
    game = game.strip()
    if not game:
        return "a video game"
    if " " not in game and ("-" in game or "_" in game):  # a profile id: azur-promilia
        game = game.replace("-", " ").replace("_", " ").title()
    return f"the video game {game}"


def _glossary(entries: list[tuple[str, str]],
              only_for: list[Item] | None) -> list[tuple[str, str]]:
    texts: list[str] | None = None
    if only_for is not None:
        texts = [normalize_key(it.text) for it in only_for]
        texts += [normalize_key(it.speaker) for it in only_for if it.speaker]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, translation in entries:
        source, translation = str(source).strip(), str(translation).strip()
        key = normalize_key(source)
        if not key or not translation or key in seen:
            continue
        if texts is not None and not any(key in t for t in texts):
            continue
        seen.add(key)
        out.append((source, translation))
    return out


def _speaker(speaker: str | None, names: dict[str, str]) -> str | None:
    """The speaker as the model should call them: the glossary's rendering of the name if
    it has one, else the name as shown."""
    if not speaker or not speaker.strip():
        return None
    return names.get(normalize_key(speaker), speaker.strip())


def _one_line(text: str) -> str:
    return " ".join(text.split())


# ---- the answer ----------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


def parse_translations(text: str, expected: int) -> list[str]:
    """The translations in a model's answer, one string per item. Accepts the JSON object
    on its own, inside a code fence or with text around it (the first ``{`` that starts a
    complete object is the outermost), a bare list, and numbered keys. Raises
    :class:`~translate.base.BadAnswerError` (a TranslationError) when there is no such list
    or it has the wrong length."""
    parsed = False
    values: list[object] | None = None
    for value in json_values(text or ""):
        parsed = True
        values = _translations_of(value, expected)
        if values is not None:
            break
    if values is None:
        if not parsed:
            raise BadAnswerError("The translator's answer was not JSON.")
        raise BadAnswerError('The translator\'s answer had no "translations" list.')
    if len(values) != expected:
        raise BadAnswerError(
            f"The translator returned {len(values)} translations for {expected} lines."
        )
    return [_as_text(v) for v in values]


def json_values(text: str) -> Iterator[object]:
    """Every JSON value an answer could be, best guess first."""
    for candidate in [text.strip(), *(m.group(1).strip() for m in _FENCE_RE.finditer(text))]:
        try:
            yield json.loads(candidate)
        except ValueError:
            pass
    for opener in ("{", "["):
        pos = text.find(opener)
        while pos != -1:
            try:
                value, _ = _DECODER.raw_decode(text, pos)
            except ValueError:
                pos = text.find(opener, pos + 1)
                continue
            yield value
            pos = text.find(opener, pos + 1)


def _translations_of(value: object, expected: int) -> list[object] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if expected == 1 else None
    if not isinstance(value, dict) or not value:
        return None
    for k, v in value.items():
        if isinstance(k, str) and k.strip().lower() == "translations":
            if isinstance(v, dict):
                return _numbered(v)
            return _translations_of(v, expected) if isinstance(v, (list, str)) else None
    numbered = _numbered(value)
    if numbered is not None:
        return numbered
    if len(value) == 1:  # {"result": [...]}, {"translation": "..."}
        return _translations_of(next(iter(value.values())), expected)
    return None


def _numbered(d: dict) -> list[object] | None:
    """{"1": "...", "2": "..."} in key order."""
    if not d or not all(isinstance(k, str) and k.strip().isdigit() for k in d):
        return None
    return [d[k] for k in sorted(d, key=lambda k: int(k))]


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):  # {"id": 1, "translation": "..."}
        for k in ("translation", "text", "english", "en"):
            if isinstance(value.get(k), str):
                return value[k].strip()
        strings = [v for v in value.values() if isinstance(v, str)]
        if len(strings) == 1:
            return strings[0].strip()
    raise BadAnswerError("The translator's answer had a translation that is not text.")


# ---- placeholders --------------------------------------------------------------------------
_PLACEHOLDER_RE = re.compile(r"[{｛]\s*(n|s|player)\s*[}｝]", re.IGNORECASE)


def fix_placeholders(source: str, translation: str) -> str:
    """Restore placeholders a model wrote slightly differently ({N}, { n }, ｛player｝)."""
    if "{" not in source:
        return translation
    return _PLACEHOLDER_RE.sub(lambda m: "{" + m.group(1).lower() + "}", translation)


def placeholders_kept(source: str, translation: str) -> bool:
    """The translation has as many {n} and {s} as the source, and {player} if (and only if)
    the source has it."""
    return (translation.count(NUMBER) == source.count(NUMBER)
            and translation.count(WORD) == source.count(WORD)
            and (PLAYER in translation) == (PLAYER in source))
