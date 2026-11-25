"""Text metrics of the HUD font, usable from worker threads.

The GUI thread measures every character the translation languages use once (its advance at
a 100 px font) and the font's vertical metrics (:func:`build_table`); the layout then gets
the width of any string at any size by summing, and the font size for Japanese glyphs of a
given height. Before :func:`build_table` runs (code without Qt), typical values are used.

The font is Segoe UI where the system has it (Windows), else the Noto Sans that ships in
``assets/fonts`` (registered with Qt at start). Widths always come from the font in use.

Line breaking: languages written with spaces break between words; Japanese, Chinese and
Thai (``translate.languages.spaced``) break where Unicode's line breaking rules allow,
which for CJK is between most characters (never before a closing bracket or 。).
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

log = logging.getLogger("hud")

SYSTEM_FAMILY = "Segoe UI"
BUNDLED_FONT = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSans-Regular.ttf"
BUNDLED_FAMILY = "Noto Sans"
STRETCH = 90  # slightly condensed: lean, like the game's own UI type
TEXT_SCALE = 0.8  # the translation's capitals are this share of the Japanese glyph height
FALLBACK_ADVANCE = 50.0  # advance (at 100 px) of a character that was not measured
WIDE_ADVANCE = 100.0  # ...unless it is a wide (CJK) character: a full em
# The characters measured: printable ASCII, typographic punctuation, then the letters of
# the translation languages: Latin-1 and Extended-A (Spanish, French, German, Polish,
# Turkish...), Greek, Cyrillic, Hebrew, Arabic, Thai, Vietnamese.
EXTRA_CHARS = "‘’“”…–—·×«»„‚"
MEASURED_RANGES = (
    (0x20, 0x7F), (0xA0, 0x180), (0x370, 0x400), (0x400, 0x530), (0x590, 0x600),
    (0x600, 0x700), (0xE00, 0xE80), (0x1EA0, 0x1F00),
)  # fmt: skip

_TABLE: dict[str, float] = {}
# vertical metrics as fractions of the pixel size (Segoe UI's, replaced when measured)
METRICS = {"cap": 0.70, "ascent": 1.064, "descent": 0.25}
_family = SYSTEM_FAMILY


def family() -> str:
    """The font family the HUD draws translations in."""
    return _family


def hud_font(px: float, stretch: int = STRETCH):
    """The HUD font: lean, unhinted and unkerned, so that its widths scale exactly with
    the size and the layout's arithmetic is what gets drawn."""
    from PySide6.QtGui import QFont

    font = QFont(_family)
    font.setWeight(QFont.Weight.Normal)
    font.setStretch(stretch)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    font.setKerning(False)
    font.setPixelSize(max(1, int(round(px))))
    return font


def choose_family() -> str:
    """Segoe UI when the system has it, else the bundled Noto Sans (registered with Qt
    here). Needs a QGuiApplication."""
    global _family
    from PySide6.QtGui import QFontDatabase

    if SYSTEM_FAMILY in QFontDatabase.families():
        _family = SYSTEM_FAMILY
        return _family
    font_id = QFontDatabase.addApplicationFont(str(BUNDLED_FONT))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    if families:
        _family = families[0]
    else:
        log.warning("bundled font %s could not be loaded: Qt's default is used", BUNDLED_FONT)
        _family = BUNDLED_FAMILY
    return _family


def build_table() -> None:
    """Choose the font and measure it (GUI thread; needs a QGuiApplication)."""
    from PySide6.QtGui import QFontMetricsF

    choose_family()
    fm = QFontMetricsF(hud_font(100))
    chars = list(EXTRA_CHARS)
    for lo, hi in MEASURED_RANGES:
        chars += [chr(c) for c in range(lo, hi) if chr(c).isprintable()]
    _TABLE.clear()
    for ch in chars:
        _TABLE[ch] = float(fm.horizontalAdvance(ch))
    METRICS["cap"] = float(fm.capHeight()) / 100.0
    METRICS["ascent"] = float(fm.ascent()) / 100.0
    METRICS["descent"] = float(fm.descent()) / 100.0


def text_width(text: str, font_px: float) -> float:
    """Width in pixels of *text* at *font_px* (HUD font, lean stretch)."""
    total = sum(_TABLE.get(ch) or _unmeasured(ch) for ch in text)
    return total * font_px / 100.0


def _unmeasured(ch: str) -> float:
    return WIDE_ADVANCE if unicodedata.east_asian_width(ch) in ("W", "F") else FALLBACK_ADVANCE


def font_px_for_cap(cap_px: float) -> int:
    """The pixel size whose capital letters are *cap_px* tall."""
    return max(6, int(round(cap_px / METRICS["cap"])))


def font_px_for_glyph(glyph_px: float) -> int:
    """The translation's size for Japanese glyphs *glyph_px* tall: capitals at TEXT_SCALE
    of the glyph height (as tall as the glyphs reads too big next to the game's text)."""
    return font_px_for_cap(TEXT_SCALE * glyph_px)


# ---- line breaking ------------------------------------------------------------------------
def pieces(text: str, spaced: bool = True) -> list[tuple[str, str]]:
    """*text* as unbreakable pieces, each with the separator that goes before it when it
    is not the first on its line: words with " " (spaced languages), or, without spaces,
    the runs between Unicode line-break opportunities with ""."""
    if spaced:
        return [(w, " ") for w in text.split()]
    out: list[tuple[str, str]] = []
    start = 0
    for end in _break_points(text):
        piece = text[start:end].strip()
        if piece:
            out.append((piece, ""))
        start = end
    return out


def _break_points(text: str) -> list[int]:
    """Where a line may break in *text* (indices after which it may), by Unicode's line
    breaking rules (Qt's QTextBoundaryFinder; usable on any thread)."""
    try:
        from PySide6.QtCore import QTextBoundaryFinder
    except ImportError:  # pragma: no cover - PySide6 is a dependency
        return list(range(1, len(text) + 1))
    finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Line, text)
    points = []
    pos = finder.toNextBoundary()
    while pos != -1:
        points.append(pos)
        pos = finder.toNextBoundary()
    if not points or points[-1] != len(text):
        points.append(len(text))
    return points


def wrap(text: str, font_px: float, width: float, spaced: bool = True) -> list[str]:
    """Greedy wrap at *width* pixels (a piece longer than the width gets a line of its
    own). *spaced*: the language separates words with spaces."""
    parts = pieces(text, spaced)
    lines: list[str] = []
    cur = ""
    cur_w = 0.0
    for piece, sep in parts:
        pw = text_width(piece, font_px)
        sw = text_width(sep, font_px)
        if cur and cur_w + sw + pw > width:
            lines.append(cur)
            cur, cur_w = piece, pw
        else:
            cur, cur_w = (piece, pw) if not cur else (cur + sep + piece, cur_w + sw + pw)
    if cur:
        lines.append(cur)
    return lines


def wrap_width(text: str, font_px: float, rows: int, spaced: bool = True) -> float:
    """The narrowest line width at which greedy wrapping fits *text* in *rows* lines."""
    parts = pieces(text, spaced)
    if not parts:
        return 0.0
    widths = [text_width(p, font_px) for p, _sep in parts]
    seps = [text_width(sep, font_px) for _p, sep in parts]
    lo, hi = max(widths), sum(widths) + sum(seps[1:])
    if rows <= 1:
        return hi

    def lines_at(limit: float) -> int:
        n, cur = 1, 0.0
        for w, s in zip(widths, seps, strict=True):
            if cur == 0.0:
                cur = w
            elif cur + s + w <= limit:
                cur += s + w
            else:
                n += 1
                cur = w
        return n

    for _ in range(24):
        mid = (lo + hi) / 2
        if lines_at(mid) <= rows:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.5:
            break
    return hi
