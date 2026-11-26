"""From an OCR reading to the database line it shows, and which of its characters that is.

Only the matched text is ever covered. Two things make a reading longer than its line:

* characters the line does not have at all: an icon read as 日 before 物理防御力, a NEW
  badge glued to a list row. The alignment with the stored text leaves them out.
* characters the database itself picked up: a capture may store a label together with an
  icon it misread (金詳細 next to 詳細). When a shorter part of the reading is, on its
  own, exactly a stored line with the same translation, that part is the text.
"""

from __future__ import annotations

from dataclasses import dataclass

from db.lookup import Lookup
from db.store import Line
from db.templates import NUMBER, PLAYER, WORD, fill, templatize
from db.text import cjk_count
from hud.extent import align_span

TRIM_MAX = 2  # characters tried off either end of a reading
WHOLE_SHARE = 0.8  # a label reading must hold this much of the line it matches...
WHOLE_MIN = 3  # ...and this many characters besides placeholders
PARTIAL_REST = 3  # a line this many characters longer than the reading may wrap to the next row
# Characters an icon, a bar, a dash or a key badge is read as: a line that is one of them
# alone is never matched (an icon is not "Mouth", a bar is not "One")
STROKES = set("一ー－‐—―二三口ロ囗十人入八丁卜トエ工日曰目田ハニ了〇○○|")
NEAR_MIN = 3  # a line matched near, not exactly, holds this many characters besides
NEAR_CJK = 2  # placeholders, and the reading this many of the game's own characters


@dataclass
class Hit:
    template: str  # the line's translation with placeholders: covers and the cache key it
    translation: str  # what to show (placeholders filled in)
    span: tuple[int, int]  # the characters of the reading that are the line
    line: Line
    kind: str  # exact | prefix | fuzzy
    score: float
    partial: bool = False  # the line goes on past the reading: its next row may hold the rest


def _bare(key: str) -> str:
    """A key without its placeholders."""
    for p in (NUMBER, WORD, PLAYER):
        key = key.replace(p, "")
    return key


def match_reading(
    lookup: Lookup, text: str, whole: bool = False, after_seq: int | None = None
) -> Hit | None:
    """The translated database line this reading shows, or None (unknown or untranslated).
    *whole*: the reading is a label, not a line still typing out: the start of a longer
    line is no match ("HP 1234/5678" is not the line "07/02 rhu."). *after_seq*: the
    script position of the line shown before (a line typing out that starts several
    lines is the one that comes next)."""
    m = lookup.match(text, remember=False, after_seq=after_seq)
    if m is None or not m.translation or not m.translation.strip():
        return None
    line_bare, read_bare = _bare(m.line.key), _bare(m.templated.key)
    if m.line.key in STROKES:  # (a line of that one character alone: "{n}日" is a line)
        return None  # an icon or a stroke read as a character
    if m.kind != "exact" and (len(line_bare) < NEAR_MIN or cjk_count(read_bare) < NEAR_CJK):
        return None  # too little text to tell: "1234/5678" is no "{n}分{n}秒"
    if whole and m.kind == "prefix":
        key = m.templated.key
        bare = key.replace(NUMBER, "").replace(PLAYER, "")
        if len(key) < WHOLE_SHARE * len(m.line.key) or len(bare) < WHOLE_MIN:
            return None
    translation = m.translation
    i0, i1 = align_span(text, m.line.source_text) or (0, len(text))
    best = (i0, i1)
    for cut_l in range(TRIM_MAX + 1):
        for cut_r in range(TRIM_MAX + 1):
            a, b = i0 + cut_l, i1 - cut_r
            if (cut_l == cut_r == 0) or b - a < 2 or b - a >= best[1] - best[0]:
                continue
            if 2 * (b - a) < i1 - i0:
                continue
            t = templatize(text[a:b], lookup.player_name)
            other = lookup.by_key.get(t.key)
            if other is None or not other.translation:
                continue
            if fill(other.translation, t.values, t.player or m.player_fallback) == translation:
                best = (a, b)
    return Hit(
        template=m.line.translation or m.line.key,
        translation=translation,
        span=best,
        line=m.line,
        kind=m.kind,
        score=m.score,
        partial=m.kind == "prefix" or len(m.line.key) - len(m.templated.key) >= PARTIAL_REST,
    )
