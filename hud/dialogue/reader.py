"""Reading the dialogue territory: which rows are a line of dialogue, which reading is the
speaker's name, and what else there is a label.

The dialogue block starts at the first row whose top lies in a layout's band of first rows
(``rows_top``) and takes the rows stacked below it that are part of the matched line. A
line is dialogue when its database line has a speaker or reads as a sentence; text on a
flat popup panel in a layout without a name band is a popup (a tutorial tip), covered like
any label. No secondary OCR happens on this path: the database's speaker is shown with the
line, and a visual check of the name comes after it (the component's ``refine_speaker``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz

from capture.ocr import OcrLine, merge_lines
from capture.profile import DialogueLayout, Profile, Region
from db.lookup import Lookup
from db.normalize import normalize_key
from hud.content import DIALOGUE
from hud.covers import MIN_CONF, CoverMaker, box_px, px
from hud.dialogue.names import JP_ADVANCE, Names
from hud.dialogue.scene import DialogueRead, Speaker
from hud.extent import GLYPH_MID_EM, measure, row_em
from hud.ink import Rect, on_panel
from hud.layout import union
from hud.looks import PLAIN_LIGHT, Color
from hud.matching import Hit, match_reading

# punctuation that makes a stored line read as a sentence: Japanese and Chinese marks, and a
# Latin period or comma that ends a sentence or clause (Korean), never one inside "Lv.5"
SENTENCE = re.compile(r"[。、！？…!?]|[.,](?:\s|$)")
ROW_GAP_EM = 1.5  # pieces of one dialogue row are never further apart than this
SAME_ROW = 0.5  # pieces whose centres are this many row heights apart are one row
STACK_GAP = 1.0  # rows of one block are at most this many row heights apart...
STACK_SIZE = 0.45  # ...and their heights differ by at most this share
FIRST_ROW_SLACK = 0.5  # a dialogue block's line starts within this share of its first row
SAME_LINE = 75  # a reading this similar to the shown line's Japanese is that line
SAME_START = 90  # ...or this similar to a part of it
NAME_GLYPH = 0.025  # a name's glyph height (of the frame) when no line tells it

Unmatched = Callable[[str, str, str | None, float, Rect], None]


@dataclass
class Running:
    """What the reader needs of the scene on screen (taken under the component's lock)."""

    layout: DialogueLayout | None = None
    panel: Rect | None = None
    name_box: Rect | None = None


def is_dialogue(hit: Hit) -> bool:
    """A database line that is dialogue: spoken by someone, or a sentence. Menu labels
    (tabs, buttons) are short and unpunctuated, even the junk a capture stored as dialogue
    when it read a tab bar in the dialogue region."""
    if hit.line.speaker:
        return True
    return bool(SENTENCE.search(hit.line.source_text or ""))


def reads_as(reading: str, source: str) -> bool:
    """Does a reading still read as this stored line (the OCR dropped or misread a few
    characters of it, or it is the start of it)?"""
    a, b = normalize_key(reading), normalize_key(source)
    if not a or not b:
        return False
    if b.startswith(a) or fuzz.ratio(a, b) >= SAME_LINE:
        return True
    return fuzz.partial_ratio(a, b) >= SAME_START


def centre(ln: OcrLine) -> tuple[float, float]:
    return ln.box.x + ln.box.w / 2, ln.box.y + ln.box.h / 2


def part_of(ln: OcrLine, row: OcrLine) -> bool:
    """Is reading *ln* (one piece) part of the joined row *row*?"""
    a, b = ln.box, row.box
    return a.x >= b.x - 1e-4 and a.x2 <= b.x2 + 1e-4 and a.y >= b.y - 1e-4 and a.y2 <= b.y2 + 1e-4


def grown(x0: float, y0: float, x1: float, y1: float, m: float) -> Region:
    """The rect (x0, y0)-(x1, y1) with a margin *m*, inside the frame (relative)."""
    x0, y0 = max(0.0, x0 - m), max(0.0, y0 - m)
    x1, y1 = min(1.0, x1 + m), min(1.0, y1 + m)
    return Region(x0, y0, x1 - x0, y1 - y0, name="dialogue_crop")


def rows_of(lines: list[OcrLine], fw: int, fh: int) -> list[OcrLine]:
    """Readings joined into rows, left to right, only where they continue each other (the
    detector sometimes splits a dialogue row): the separate labels of a tab bar stay
    separate. Top to bottom."""
    rows: list[OcrLine] = []
    for ln in sorted(lines, key=lambda ln: ln.box.x):
        for i, r in enumerate(rows):
            h_px = max(r.box.h, ln.box.h) * fh  # a detector row is a little over an em
            same_row = abs((r.box.y + r.box.h / 2) - (ln.box.y + ln.box.h / 2)) * fh <= (
                SAME_ROW * h_px
            )
            gap_px = (ln.box.x - r.box.x2) * fw
            if same_row and -SAME_ROW * h_px <= gap_px <= ROW_GAP_EM * h_px:
                rows[i] = merge_lines([r, ln])[0]
                break
        else:
            rows.append(ln)
    return sorted(rows, key=lambda r: r.box.y)


def layout_at(profile: Profile, row: OcrLine) -> DialogueLayout | None:
    """The layout whose band of first rows holds the top of *row* (and whose panel its
    centre lies over)."""
    for lay in profile.dialogue_layouts.values():
        lo, hi = lay.rows_top
        cx = row.box.x + row.box.w / 2
        if lo <= row.box.y < hi and lay.panel.x <= cx <= lay.panel.x2:
            return lay
    return None


def stack(rows: list[OcrLine], start: int, lay: DialogueLayout) -> list[OcrLine]:
    """Row *start* and the rows stacked right below it, over the layout's panel."""
    block = [rows[start]]
    for r in rows[start + 1 :]:
        last = block[-1]
        gap = r.box.y - last.box.y2
        cx = r.box.x + r.box.w / 2
        if gap > STACK_GAP * last.box.h or not (lay.panel.x <= cx <= lay.panel.x2):
            break
        if abs(r.box.h - last.box.h) > STACK_SIZE * last.box.h:
            break
        block.append(r)
    return block


def match_block(
    lookup: Lookup, block: list[OcrLine], lay: DialogueLayout | None, after_seq: int | None
) -> tuple[Hit, list[OcrLine]] | None:
    """The dialogue line these rows show: all of them read as one text, else the longest
    leading run of them, else a single row (barks stack below each other). The first row
    must be part of the line: a misread name above the text (テラフ for テララ, so not
    taken for a name) still matches fuzzily together with the line, and would move the
    line to the layout the name sits in."""
    for n in range(len(block), 0, -1):
        if min(r.conf for r in block[:n]) < MIN_CONF:
            continue
        hit = match_reading(lookup, "".join(r.text for r in block[:n]), after_seq=after_seq)
        if hit is None or not (is_dialogue(hit) or (lay is not None and lay.all_dialogue)):
            continue
        first = len(block[0].text)
        if hit.span[0] >= first or hit.span[0] > FIRST_ROW_SLACK * first:
            continue  # the first row is not (mostly) part of the line
        return hit, block[:n]
    return None


class TerritoryReader:
    def __init__(self, profile: Profile, lookup: Lookup, maker: CoverMaker, names: Names) -> None:
        self.profile = profile
        self.lookup = lookup
        self.maker = maker
        self.names = names

    def read(
        self,
        frame: np.ndarray,
        lines: list[OcrLine],
        ui_boxes: list[Rect],
        running: Running,
        after_seq: int | None,
        unmatched: Unmatched | None,
    ) -> DialogueRead:
        """What the territory shows (*lines*: its readings)."""
        fh, fw = frame.shape[:2]
        out = DialogueRead()
        # speaker names (a known name in a layout's name band) are a scene's name box and
        # nothing else: never a row of dialogue, never a label
        names = [ln for ln in lines if self.names.is_name(ln)]
        rest = [ln for ln in lines if ln not in names]
        unknown = self._find_line(frame, rows_of(rest, fw, fh), out, after_seq)
        if out.rows:
            rest = [ln for ln in rest if not any(part_of(ln, r) for r in out.rows)]
        rest = self._find_speaker(frame, out, names, rest, running.layout)
        if out.hit is None and unknown and unmatched is not None:
            # a line of dialogue the database does not know: live translation may (it is no
            # label either: its rows are not covered as UI text)
            box = box_px(unknown[0], fw, fh)
            for r in unknown[1:]:
                box = union(box, box_px(r, fw, fh))
            name = out.speaker.name if out.speaker is not None else None
            text = "".join(r.text for r in unknown)
            unmatched(text, DIALOGUE, name, min(r.conf for r in unknown), box)
            rest = [ln for ln in rest if not any(part_of(ln, r) for r in unknown)]
        # everything else in the territory is a label, covered like any UI text; never
        # where the running scene's panel or name box is
        stay_out = list(ui_boxes) + [r for r in (running.panel, running.name_box) if r]
        out.labels = self.maker.placements(
            frame,
            [ln for ln in rest if ln.conf >= MIN_CONF],
            stay_out,
            accept=lambda row: not is_dialogue(row.hit) or on_panel(frame, row.extent.cover),
        )
        return out

    def _find_line(
        self, frame: np.ndarray, rows: list[OcrLine], out: DialogueRead, after_seq: int | None
    ) -> list[OcrLine] | None:
        """Fill *out* with the dialogue line among *rows*; returns the first block that
        looked like dialogue but matched no database line (for live translation)."""
        fh, fw = frame.shape[:2]
        ignore = [px(r, fw, fh) for r in self.profile.ignore_regions]
        unknown: list[OcrLine] | None = None
        for start in range(len(rows)):
            lay = layout_at(self.profile, rows[start])
            if lay is None:
                continue
            block = stack(rows, start, lay)
            found = match_block(self.lookup, block, lay, after_seq)
            if found is None:
                unknown = unknown or block
                continue
            hit, used = found
            look = _measure_rows(frame, used, ignore, fw)
            if look is None or (
                lay.speaker is None and not lay.all_dialogue and on_panel(frame, look.cover)
            ):
                # the game draws dialogue straight onto the scene; text on a flat panel in a
                # band without a name is a popup (a tutorial tip): a label, covered like UI
                # text. Where a layout has a name band, its lines are dialogue (and in an
                # area the player chose, all of them: a message box is a flat panel).
                continue
            out.layout, out.hit, out.rows = lay, hit, used
            if hit.line.speaker:
                out.speaker_hint = self.names.name_of(hit.line.speaker)
            out.rows_cover, out.light, out.glyph = look.cover, look.light, look.glyph
            out.fill, out.edge, out.row_inks = look.fill, look.edge, look.inks
            out.reading = sum(len(r.text) for r in used)
            out.text = "".join(r.text for r in used)
            return unknown
        return unknown

    def _find_speaker(
        self,
        frame: np.ndarray,
        out: DialogueRead,
        names: list[OcrLine],
        rest: list[OcrLine],
        running: DialogueLayout | None,
    ) -> list[OcrLine]:
        """Fill *out* with the speaker name in the name band of the line's layout (another
        layout's name band can hold dialogue: NPC speech sits where in-world dialogue puts
        the name), or the database's speaker until the name is read. Returns *rest* without
        what is in that name band."""
        fh, fw = frame.shape[:2]
        lay = out.layout or running
        sp = lay.speaker if lay is not None else None
        if sp is None:
            return rest
        inside = [ln for ln in rest if sp.contains_point(*centre(ln))]
        out.speaker = self.names.find(frame, lay, names + inside)
        if out.speaker is None and out.speaker_hint and out.layout is not None:
            # the database's speaker is enough to show the name with the line; the OCR of
            # the name comes after, and ties it to the name's pixels
            x, y, w, h = px(sp, fw, fh)
            glyph = max(6, int(round(out.glyph or min(h, NAME_GLYPH * fh))))
            iw = min(w, max(glyph, int(round(self.names.longest_jp * JP_ADVANCE * glyph))))
            ink = (x + (w - iw) // 2, y + max(0, (h - glyph) // 2), iw, glyph)
            out.speaker = Speaker(lay, out.speaker_hint, ink, out.fill, out.edge, verified=False)
        return [ln for ln in rest if ln not in inside]


@dataclass
class _Look:
    """How a dialogue line's rows look on screen."""

    cover: Rect
    light: bool
    fill: Color
    edge: Color | None
    inks: list[Rect]  # its rows' glyphs, top to bottom
    glyph: float  # the glyph height of its font


def _measure_rows(
    frame: np.ndarray, rows: list[OcrLine], ignore: list[Rect], fw: int
) -> _Look | None:
    cover = None
    inked, pitched = [], []
    fills: list[Color] = []
    edges: list[Color] = []
    inks: list[Rect] = []
    lights = 0
    for row in rows:
        ext = measure(frame, row, None, ignore)
        if ext is None:
            continue
        inks.append(ext.ink)
        inked.append(ext.glyph_h)
        lights += 1 if ext.light else -1
        fills.append(ext.fill)
        if ext.edge is not None:
            edges.append(ext.edge)
        cover = ext.cover if cover is None else union(cover, ext.cover)
        em = row_em(row, fw)
        if em is not None:
            pitched.append(GLYPH_MID_EM * em)
    if cover is None:
        return None
    # the size of dialogue glyphs, fixed by the first line of a layout: from the character
    # pitch of its long rows, which is the same line after line (the ink of light text over
    # the scene varies by a few pixels with what is behind it)
    sizes = pitched or inked
    return _Look(
        cover=cover,
        light=lights >= 0,
        fill=_median(fills),
        edge=_median(edges) if 2 * len(edges) > len(fills) else None,
        inks=sorted(inks, key=lambda r: r[1]),
        glyph=float(np.median(sizes)) if sizes else 0.0,
    )


def _median(colours: list[Color]) -> Color:
    if not colours:
        return PLAIN_LIGHT
    return tuple(int(np.median([c[i] for c in colours])) for i in range(3))
