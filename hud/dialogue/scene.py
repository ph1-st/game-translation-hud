"""A dialogue scene: what is shown in one dialogue layout while a conversation runs, and
where the text of its lines goes.

A layout's look is fixed the first time it is seen and kept in the cover cache: its font
(from the character pitch of its first long line), its colours, where its first row is and
the distance between its rows. Every line of the layout is then laid out the same way in
the same place, the whole line from its first characters on (never fitted to the part typed
out so far): wrapped like the game's rows (centred, or flush left in a message box), its
lines on the game's rows, starting at the layout's first row and growing away from its
anchor (down from a top-anchored layout, up from a bottom-anchored one), never over the
speaker name. A line that needs more rows than the layout's area holds is wrapped over the
whole width, then condensed, then made smaller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from capture.ocr import OcrLine
from capture.profile import DialogueLayout
from hud.anchors import Anchor
from hud.covers import Placement, px
from hud.ink import Rect
from hud.layout import PAD_X, drawn_size, fit_block
from hud.looks import PLAIN_LIGHT, Color
from hud.matching import Hit
from hud.textwidth import STRETCH, text_width, wrap, wrap_width

WRAP_SHARE = 0.62  # a centred line is wrapped at this share of the layout's width (about
# as wide as the game's own rows)
DIALOGUE_PITCH = 1.5  # lines are this many font sizes apart until the layout's own row
# pitch is known (from its first line of two rows)
PANEL_PAD = 0.45  # space between the text and the layout's edges, in font sizes
REVEAL_MIN_S = 0.3  # text types out in this long at least (a line that came whole)...
REVEAL_TAIL_S = 0.15  # ...and ends this long after its Japanese is due to


@dataclass
class Shown:
    """The line on screen."""

    text: str  # its translation, as shown
    anchor: Anchor  # its Japanese strokes, watched
    reading: int = 0  # characters of the reading that matched it
    source: str = ""  # the stored Japanese of the line
    doubtful: int = 0
    typed: int = 0  # characters of it read so far (it types out)...
    typed_at: float = 0.0  # ...in the reading of this moment
    reveal: tuple[float, float, float] = (0.0, 0.0, 0.0)  # how its text types out


@dataclass
class Pane:
    """A pane of frosted glass."""

    rect: Rect
    key: int


@dataclass
class Speaker:
    """A speaker name read in (or, from the database, expected in) a layout's name band."""

    layout: DialogueLayout
    name: str  # in the translation language
    ink: Rect  # its glyphs
    fill: Color
    edge: Color | None
    verified: bool = True  # False: the database's speaker, not yet tied to the name's pixels


@dataclass
class Scene:
    layout: DialogueLayout
    panel: Rect
    font_px: int  # the layout's font
    glyph: float  # the layout's Japanese glyph height
    fill: Color = PLAIN_LIGHT  # the layout's dialogue colours
    edge: Color | None = None
    line: Shown | None = None
    lines: list[str] = field(default_factory=list)  # the shown line's text, wrapped
    line_gone_at: float = 0.0
    name: str | None = None
    name_box: Rect | None = None
    name_font: int = 0
    name_glyph: float = 0.0
    name_ink: Rect | None = None  # the Japanese name being replaced
    name_fill: Color = PLAIN_LIGHT
    name_edge: Color | None = None
    name_anchor: Anchor | None = None  # the Japanese name's strokes, watched
    name_doubt: int = 0
    text_box: Rect | None = None  # the text of the current line: its block...
    text_top: float = 0.0  # ...the top of its first row...
    text_band: float = 0.0  # ...the glyph band each line is centred on...
    text_pitch: float = 0.0  # ...the distance between lines...
    text_font: int = 0  # ...and its size and stretch (smaller than the layout's only when
    text_stretch: int = STRETCH  # the line would not fit otherwise)
    text_y: float = 0.0  # the layout's first row top (its last row bottom when anchored low)
    row_pitch: float = 0.0  # the layout's distance between rows, once known
    panes: list[Pane] = field(default_factory=list)  # the frosted glass over the line
    pane_seq: int = 0
    fixed: bool = False  # one pane over the layout's whole glass area
    name_glass: Rect | None = None  # the frosted box over the speaker name
    align: str = "center"  # how the game lays its rows out: center | left


@dataclass
class DialogueRead:
    """What one OCR pass of the dialogue territory found."""

    layout: DialogueLayout | None = None
    rows: list[OcrLine] = field(default_factory=list)
    hit: Hit | None = None
    rows_cover: Rect | None = None
    glyph: float = 0.0
    light: bool = True  # polarity of the line's glyphs
    reading: int = 0  # characters read of the line's rows
    text: str = ""  # what was read
    fill: Color = PLAIN_LIGHT  # the line's colours
    edge: Color | None = None
    row_inks: list[Rect] = field(default_factory=list)  # its Japanese rows, top to bottom
    speaker: Speaker | None = None
    speaker_hint: str | None = None  # the database's speaker, in the translation language
    labels: list[Placement] = field(default_factory=list)


def text_area(sc: Scene, fw: int, fh: int) -> Rect:
    """Where a layout's text may lie: its glass (the frosted area) or else its panel."""
    return px(sc.layout.glass, fw, fh) if sc.layout.glass is not None else sc.panel


def centre_x(sc: Scene, fw: int, fh: int) -> float:
    """Where centred rows are centred: the middle of the layout's text area (the screen's
    middle for the built-in layouts, whose glass is centred)."""
    x, _y, w, _h = text_area(sc, fw, fh)
    return x + w / 2.0


def lay_out(sc: Scene, text: str, fw: int, fh: int, spaced: bool = True) -> None:
    """Where the text of a line goes: the same place for every line of the layout,
    whatever part of the Japanese has typed out so far (the module docstring)."""
    font = sc.font_px
    x, y, w, h = text_area(sc, fw, fh)
    inner = w - 2 * PANEL_PAD * font
    pitch = sc.row_pitch or DIALOGUE_PITCH * font
    band = sc.glyph
    room = sc.text_y - y if sc.layout.anchor == "bottom" else (y + h) - sc.text_y
    rows = max(1, int(math.floor((room - band) / pitch)) + 1)
    if sc.align == "left":  # (a message box: the text fills its lines, as the game's)
        lines = wrap(text, font, inner, spaced)
    else:
        lines = wrap(text, font, WRAP_SHARE * inner, spaced)
        if 1 < len(lines) <= rows:  # balanced lines, as the game's centred rows are
            lines = wrap(text, font, wrap_width(text, font, len(lines), spaced) + 1, spaced)
    size, stretch = font, STRETCH
    if len(lines) > rows:
        size, stretch, lines = fit_block(text, font, rows, inner, spaced)
    height = (len(lines) - 1) * pitch + band
    top = sc.text_y - height if sc.layout.anchor == "bottom" else sc.text_y
    widest = max(text_width(t, drawn_size(size, stretch)) for t in lines) if lines else 0.0
    bw = int(math.ceil(widest + 2 * PAD_X))
    if sc.align == "left":
        left = x + PANEL_PAD * font - PAD_X
    else:
        left = centre_x(sc, fw, fh) - bw / 2.0
    sc.lines = lines
    sc.text_box = (int(round(left)), int(round(top)), bw, int(math.ceil(height)))
    sc.text_top, sc.text_band, sc.text_pitch = top, band, pitch
    sc.text_font, sc.text_stretch = size, stretch


def reveal(sc: Scene, share: float, now: float) -> tuple[float, float, float]:
    """How the text of the line types out from *share* at *now*: at an even pace, ending
    just after the Japanese is due to (the game's typing pace for what is left of it; a
    line that came whole, in a moment)."""
    ln = sc.line
    if ln is None:
        return 1.0, now, 0.0
    rate = sc.layout.typing
    left = max(0, len(ln.source) - ln.typed)
    seconds = left / rate + REVEAL_TAIL_S if rate > 0 and left else 0.0
    return share, now, max(REVEAL_MIN_S * (1.0 - share), seconds)


ALIGN_FLUSH = 0.03  # left edges this close (share of the area's width) are flush...
ALIGN_NEAR_LEFT = 0.15  # ...near the area's left edge...
ALIGN_OFF_CENTRE = 0.05  # ...and the first row clearly off its centre: left-aligned


def align_of(lay: DialogueLayout, rows: list[Rect], fw: int, fh: int) -> str:
    """How the game lays a layout's rows out ("auto": from its first line): left when they
    start at the area's left edge and are not centred in it."""
    if lay.align != "auto":
        return lay.align
    if not rows:
        return "center"
    x, _y, w, _h = px(lay.panel, fw, fh)
    lefts = [r[0] for r in rows]
    mid = (rows[0][0] + rows[0][2] / 2.0) - (x + w / 2.0)
    flush = max(lefts) - min(lefts) <= ALIGN_FLUSH * w
    near_left = rows[0][0] - x <= ALIGN_NEAR_LEFT * w
    left = flush and near_left and abs(mid) > ALIGN_OFF_CENTRE * w
    return "left" if left else "center"
