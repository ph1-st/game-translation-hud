"""Templates: the parts of a line that change from one showing to the next become
placeholders, so one database row covers every value.

* ``{player}``: the player character's name
* ``{n}``: a number (the screen shows 3, the database says ``{n}``)
* ``{s}``: a value the game fills in that may be a word (``%s`` and ``{0}`` in the game's
  files: an item name, a place, a count)

The importers and the HUD run every text through :func:`templatize` to get its key (a
line with its translation: :func:`keyed`); the HUD fills a matched line's translation with
the values it read (:func:`fill`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from db.normalize import normalize_char, normalize_key

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")  # what becomes {n}
_THOUSANDS_RE = re.compile(r"(?<=\d)[,、](?=\d{3}(?!\d))")  # 1,000 is one number
_SPLIT_NUMBERS_RE = re.compile(r"\{n\}[,、]\{n\}")  # ...that older keys split in two
PLAYER = "{player}"
NUMBER = "{n}"
WORD = "{s}"
PLACEHOLDERS = (PLAYER, NUMBER, WORD)
WORD_MAX = 24  # characters a {s} value may have on screen
_SCRIPTS = (
    ("katakana", re.compile("[ァ-ヿ]")),
    ("hiragana", re.compile("[ぁ-ゟ]")),
    ("kanji", re.compile("[㐀-䶿一-鿿]")),
    ("latin", re.compile(r"[A-Za-z0-9]")),
)
_SCRIPT_CLASS = {
    "katakana": "ァ-ヿ",
    "hiragana": "ぁ-ゟ",
    "kanji": "㐀-䶿一-鿿",
    "latin": "A-Za-z0-9",
}


@dataclass
class Templated:
    key: str
    values: list[str] = field(default_factory=list)  # captured {n} values, in order
    player: str | None = None  # captured player name, if replaced
    words: list[str] = field(default_factory=list)  # captured {s} values, in order

    @property
    def is_template(self) -> bool:
        return any(p in self.key for p in PLACEHOLDERS)


def templatize(text: str, player_name: str | None = None) -> Templated:
    """Normalize *text* to a key and replace the player name and numbers with placeholders."""
    key, player = _named(text, player_name)
    values = NUMBER_RE.findall(key)
    if values:
        key = NUMBER_RE.sub(NUMBER, key)
    return Templated(key=key, values=values, player=player)


def literal_key(text: str, player_name: str | None = None) -> str:
    """The key of *text* with its numbers as they are: for lines whose translation does not
    carry their numbers along (第一章 -> "Chapter 1" cannot be a template)."""
    return _named(text, player_name)[0]


def keyed(text: str, translation: str | None = None,
          player_name: str | None = None) -> tuple[Templated, str | None]:
    """The key a line is stored under and its translation in the key's placeholders. The
    numbers are placeholders unless a translation comes along that does not carry them
    (:func:`template_translation`): the line then keeps them as they are and matches only
    the same numbers. The key is "" for a text without one."""
    t = templatize(text, player_name)
    if not t.key or translation is None:
        return t, translation
    templated = template_translation(translation, t)
    if templated is not None:
        return t, templated
    if t.player and PLAYER not in translation:
        translation = translation.replace(t.player, PLAYER)
    return Templated(key=literal_key(text, player_name), player=t.player), translation


def with_player(text: str, player_name: str | None) -> str:
    """*text* as written, with the player's name (a word of its own) as ``{player}``: text
    collected on screen made like the game's own, which names the player that way."""
    name = (player_name or "").strip()
    return _name_re(name).sub(PLAYER, text) if name else text


def _named(text: str, player_name: str | None) -> tuple[str, str | None]:
    """The key of *text* with the player's name as ``{player}``, and the name when it was
    there."""
    key = _THOUSANDS_RE.sub("", normalize_key(text))
    if player_name:
        pkey = normalize_key(player_name)
        if pkey:
            named = _name_re(pkey).sub(PLAYER, key)
            if named != key:
                return named, player_name
    return key, None


def _script(ch: str) -> str | None:
    for name, rx in _SCRIPTS:
        if rx.match(ch):
            return name
    return None


def _name_re(pkey: str) -> re.Pattern:
    """The player's name as a word of its own: never inside another word written in the
    same script (a player called アイ is not the アイ of アイテム)."""
    head, tail = _script(pkey[0]), _script(pkey[-1])
    before = f"(?<![{_SCRIPT_CLASS[head]}])" if head else ""
    after = f"(?![{_SCRIPT_CLASS[tail]}])" if tail else ""
    return re.compile(before + re.escape(pkey) + after)


def rekey(key: str) -> str:
    """A key a database was made with, brought up to the current normalization (keys are
    stored; the rules that made them may have changed since)."""
    return normalize_key(_SPLIT_NUMBERS_RE.sub(NUMBER, key))


def word_pattern(key: str) -> re.Pattern | None:
    """A regex that matches the keys of readings of a line whose key has ``{s}``: each
    ``{s}`` stands for any short run of text (captured in order). None for a key without."""
    if WORD not in key:
        return None
    parts = [re.escape(p) for p in key.split(WORD)]
    return re.compile(f"(.{{1,{WORD_MAX}}}?)".join(parts))


def template_translation(translation: str, t: Templated) -> str | None:
    """*translation* with the placeholders of its source line (*t*: the source templated):
    ``{player}`` where the player's name stands, and ``{n}`` for the numbers when the
    translation has the same numbers in the same order (they are filled back in that
    order; "1,000" and "1000" are the same number). A translation templated already (an
    import that writes ``{n}``) is kept when it has as many ``{n}`` as the source. None when
    the source's numbers cannot be found in the translation (written out in words,
    reordered): such a line cannot be a template (see :func:`literal_key`)."""
    if t.player and PLAYER not in translation:
        translation = translation.replace(t.player, PLAYER)
    if NUMBER in translation:
        return translation if translation.count(NUMBER) == len(t.values) else None
    if not t.values:
        return translation
    found = NUMBER_RE.findall(translation)
    if [_digits(v) for v in found] != [_digits(v) for v in t.values]:
        return None
    return NUMBER_RE.sub(NUMBER, translation)


def _digits(number: str) -> str:
    return re.sub(r"\D", "", number)


def templated_text(reading: str, key: str) -> str:
    """The text to translate for an OCR *reading* whose templated key is *key*: the reading
    made readable (:func:`db.normalize.normalize_text`) with the player's name and the
    numbers replaced by the placeholders its key has, so the translation comes back as the
    same template. When the reading cannot be lined up with the key, the key itself."""
    from db.normalize import normalize_text

    text = normalize_text(reading)
    if PLAYER not in key and NUMBER not in key:
        return text or key
    out = text
    if PLAYER in key:
        name = _player_name(text, key)
        if not name or name not in out:
            return key
        out = out.replace(name, PLAYER)
    out = NUMBER_RE.sub(NUMBER, out)
    if out.count(NUMBER) != key.count(NUMBER) or out.count(PLAYER) != key.count(PLAYER):
        return key
    return out


def _player_name(text: str, key: str) -> str | None:
    """The player's name as it stands in the reading (normalized): the part the key's
    ``{player}`` stands for."""
    pattern = ""
    for part in _SLOTS_RE.split(key):
        if part == PLAYER:
            pattern += "(?P=p)" if "(?P<p>" in pattern else "(?P<p>.+?)"
        elif part == NUMBER:
            pattern += NUMBER_RE.pattern
        elif part == WORD:
            pattern += f".{{1,{WORD_MAX}}}?"
        else:
            pattern += re.escape(part)
    m = re.fullmatch(pattern, normalize_key(text))
    return m.group("p") if m else None


_SLOTS_RE = re.compile(r"(\{player\}|\{n\}|\{s\})")


def capture(reading: str, line_key: str) -> tuple[str | None, list[str]] | None:
    """What *reading* shows where the line (a template) has its placeholders, as the
    screen shows it: (the player's name, the ``{s}`` words in order). None when the rest of
    the reading does not spell the rest of the line."""
    folded: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(reading):
        for c in normalize_char(ch):
            folded.append(c)
            idx.append(i)
    text = _THOUSANDS_RE.sub(",", "".join(folded))
    pattern, n_words = "", 0
    for part in _SLOTS_RE.split(line_key):
        if part == PLAYER:
            pattern += "(?P=p)" if "(?P<p>" in pattern else "(?P<p>.+?)"
        elif part == NUMBER:
            pattern += NUMBER_RE.pattern
        elif part == WORD:
            n_words += 1
            pattern += f"(?P<w{n_words}>.{{1,{WORD_MAX}}}?)"
        else:
            pattern += re.escape(part)
    m = re.fullmatch(pattern, text)
    if m is None:
        return None

    def raw(group: str) -> str | None:
        a, b = m.span(group)
        if b <= a or b > len(idx):
            return None
        return reading[idx[a] : idx[b - 1] + 1].strip() or None

    player = raw("p") if "(?P<p>" in pattern else None
    words = [raw(f"w{i}") or "" for i in range(1, n_words + 1)]
    return player, words


def player_in(reading: str, line_key: str) -> str | None:
    """The player's name as *reading* shows it where the line (a template with {player})
    has it, when the rest of the reading spells the rest of the line."""
    if PLAYER not in line_key:
        return None
    found = capture(reading, line_key)
    return found[0] if found else None


def fill(translation: str, values: list[str], player_name: str | None,
         words: list[str] | None = None) -> str:
    """Substitute captured values back into a translated template: *words* for ``{s}``
    (each of which may hold a ``{n}``: a number read where the game filled in a value),
    then *values* for ``{n}`` in order. A placeholder no value is left for goes, with the
    separator before it (a reading matched to a line with more numbers: "3/5" read as
    "35"). *player_name*: the name, or the target language's word for "you" when none is
    known (:class:`db.lookup.Lookup` passes it; "you" for callers that do not know the
    language)."""
    out = translation.replace(PLAYER, player_name or "you")
    for w in words or ():
        out = out.replace(WORD, w, 1)
    for v in values:
        out = out.replace(NUMBER, v, 1)
    if NUMBER in out or WORD in out:
        out = _LEFTOVER_RE.sub("", out).strip()
    return out


_LEFTOVER_RE = re.compile(r"\s*(?:[/:×+]|(?<![A-Za-z])x)?\s*\{[ns]\}")
