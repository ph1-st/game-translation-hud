"""From OCR readings to covers: the one way a tight cover is made, wherever it is.

For every reading of a scan:

1. match it against the database (:mod:`hud.matching`); unknown and untranslated text is
   never covered. Rows the OCR reads one by one that are one text in the database (an item
   description, a tutorial popup) are matched together, as the paragraph they are
2. find exactly the ink of the matched characters (:mod:`hud.extent`): the cover
3. rows that clearly form a paragraph become one block
4. snap the glyph height to the game's text sizes and the glyphs' colours to the game's
   text colours (:mod:`hud.looks`): the translation's size and colours
5. take the block this line had at this place from the permanent cache, or lay it out
   (:mod:`hud.layout`) and remember it

A placement goes on a board only if the text under it held still between the frame the
OCR read and the newest frame (a reading taken mid-animation or of text that has gone
since is dropped; the area is read again), and only if nothing is drawn there yet: the OCR
never moves, replaces or removes a cover, the pixel watch alone removes them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from capture.ocr import OcrLine
from capture.profile import Profile, Region
from db.lookup import Lookup
from db.normalize import looks_like_text
from db.text import cjk_count
from hud.anchors import make_anchor, settled
from hud.board import Board, Cover
from hud.content import UI, Content, is_prose, is_story
from hud.covercache import CoverCache, place
from hud.extent import Extent, measure
from hud.ink import Rect
from hud.layout import (
    BLOCK_ALIGN,
    BLOCK_GAP,
    BLOCK_MIN_CHARS,
    PAD_X,
    Geom,
    blobs,
    drawn_size,
    intersects,
    layout_block,
    layout_line,
    near_blobs,
    union,
)
from hud.looks import Colours, same_colour
from hud.matching import Hit, match_reading
from hud.textwidth import font_px_for_glyph, text_width, wrap

log = logging.getLogger("hud")

MIN_CONF = 0.6  # readings below this are never matched (they still count as obstacles)
REREAD_MAX = 12  # lines a scan reads a second time at most (CoverMaker.reread)
ALIGN_TOL = 0.4  # stacked labels whose edges agree within this many glyph heights
SAME_SIZE = 1.0  # glyph heights (pixels) this close may be one size of the game's text
KANA_SHORT = 0.12  # kana-only text inks up to this much shorter than kanji of one size
CACHE_AFTER_S = 1.0  # a cover that stayed this long is settled and goes into the cache
STACK_REACH = 7.0  # labels this many glyph heights apart in one column belong together
LEVEL = 0.35  # labels whose centres are this many glyph heights apart sit on one row
NEIGHBOUR_ROWS = 4.0  # stacked labels that vote on alignment: this many rows away at most
PARAGRAPH_GAP = 1.0  # rows read one by one: box gaps up to this share of a row's height...
PARAGRAPH_SIZE = 0.25  # ...box heights within this share of each other...
PARAGRAPH_MAX = 12  # ...at most this many rows matched as one text
OVERLAP_SHARE = 0.5  # stacked rows share at least this much of the narrower one's columns
ROW_OVERLAP = 0.25  # rows may overlap by this share of a row height (detector boxes)
_PREFER = {"left": 0, "center": 1, "right": 2}  # equally good alignments: the first
_KANA_RE = re.compile(r"[぀-ヿ]")
_KANJI_RE = re.compile(r"[㐀-䶿一-鿿]")


@dataclass
class Row:
    line: OcrLine
    hit: Hit  # rows of one paragraph share one hit
    extent: Extent


@dataclass
class Placement:
    template: str
    text: str
    geom: Geom
    cached: bool  # the block came from the permanent cache
    look: float  # the game's text size this is (pixels)
    light: bool  # light text (the polarity of the strokes the watch looks at)
    cells: list[tuple[int, int]] | None = None  # one row: each character's columns
    story: bool = False  # dialogue, narration, a choice (hud.content), not a menu label


def px(region: Region, fw: int, fh: int) -> Rect:
    x0, y0, x1, y1 = region.to_pixels(fw, fh)
    return x0, y0, x1 - x0, y1 - y0


def box_px(line: OcrLine, fw: int, fh: int) -> Rect:
    return px(line.box, fw, fh)


class CoverMaker:
    def __init__(
        self, lookup: Lookup, cache: CoverCache, profile: Profile, spaced: bool = True
    ) -> None:
        """*spaced*: the translation language separates words with spaces (where lines
        break)."""
        self.lookup = lookup
        self.cache = cache
        self.profile = profile
        self.spaced = spaced
        self.unsettled = 0  # readings dropped: their text moved while they were read
        # (text, kind, speaker, confidence, box) of readings with no database line
        self.unmatched: Callable[[str, str, str | None, float, Rect], None] | None = None
        self.content = Content()  # what the player wants translated
        self._unsettled: list[Rect] = []  # covers of readings dropped: read them again
        # a second reading of lines that match nothing, by another reader, in one batch
        # (MeikiOCR's readings, by the standard one: it misreads a kanji now and then)
        self.reread: Callable[[np.ndarray, list[OcrLine]], list[OcrLine | None]] | None = None

    # ---- readings -> rows ----------------------------------------------------------------
    def readable(self, line: OcrLine) -> bool:
        return (
            line.conf >= MIN_CONF
            and cjk_count(line.text) > 0
            and looks_like_text(line.text, min_jp=1)
            and not self.profile.is_ignored(line.box)
        )

    def rows(self, frame: np.ndarray, lines: list[OcrLine]) -> tuple[list[Row], list[OcrLine]]:
        """(matched rows with their exact extents, readings that are not covered)."""
        fh, fw = frame.shape[:2]
        ignore = [px(r, fw, fh) for r in self.profile.ignore_regions]
        rows: list[Row] = []
        unmatched: list[OcrLine] = []
        other: list[OcrLine] = []
        first = [
            (ln, match_reading(self.lookup, ln.text, whole=True) if self.readable(ln) else None)
            for ln in lines
        ]
        reread = self.reread  # (set from another thread: read once)
        if reread is not None:
            ask = [i for i, (ln, hit) in enumerate(first) if hit is None and (
                cjk_count(ln.text) or len(ln.text.strip()) <= 3
            ) and not self.profile.is_ignored(ln.box)][:REREAD_MAX]  # fmt: skip
            for i, better in zip(ask, reread(frame, [first[i][0] for i in ask]), strict=True):
                if better is not None and self.readable(better):
                    found = match_reading(self.lookup, better.text, whole=True)
                    if found is not None:
                        first[i] = (better, found)
        for ln, hit in first:
            readable = hit is not None or self.readable(ln)
            ext = measure(frame, ln, hit.span, ignore) if hit is not None else None
            if hit is not None and ext is not None:
                rows.append(Row(ln, hit, ext))
            elif readable and hit is None:
                unmatched.append(ln)
            else:
                other.append(ln)
        # rows the OCR read one by one that are one text in the database: rows no line
        # matched, and rows that hold only part of their line (a line that wraps, its next
        # row read on its own), joined only where that matches better than the row alone
        alone = {id(r.line): r.hit.score for r in rows if r.hit.partial}
        started = [r.line for r in rows if r.hit.partial]
        for stack in paragraph_stacks(unmatched + started, fw, fh):
            found = self._match_stack(frame, stack, ignore, alone)
            rows = [r for r in rows if not any(f.line is r.line for f in found)] + found
            unmatched = [ln for ln in unmatched if not any(r.line is ln for r in found)]
        callback = self.unmatched  # (set from another thread: read once)
        for ln in unmatched:
            if callback is not None and self.content.allows(is_prose(ln.text)):
                # text the database does not know: live translation may (hud/pipeline.py)
                callback(ln.text, UI, None, ln.conf, box_px(ln, fw, fh))
        return rows, other + unmatched

    def _match_stack(
        self,
        frame: np.ndarray,
        stack: list[OcrLine],
        ignore: list[Rect],
        alone: dict[int, float],
    ) -> list[Row]:
        """The longest run of *stack* (rows top to bottom) that reads as one database line
        (a title row above a description is left out), as rows sharing that line's hit. A
        row that matched on its own (*alone*: its score by id) joins only a better match."""
        n = len(stack)
        for size in range(n, 1, -1):
            for start in range(n - size + 1):
                part = stack[start : start + size]
                hit = match_reading(self.lookup, "".join(r.text for r in part), whole=True)
                if hit is None or any(hit.score <= alone.get(id(r), -1.0) for r in part):
                    continue
                extents = [measure(frame, r, None, ignore) for r in part]
                if any(e is None for e in extents):
                    continue
                return [Row(r, hit, e) for r, e in zip(part, extents, strict=True)]
        return []

    # ---- rows -> placements --------------------------------------------------------------
    def placements(
        self,
        frame: np.ndarray,
        lines: list[OcrLine],
        obstacles: list[Rect],
        accept: Callable[[Row], bool] | None = None,
    ) -> list[Placement]:
        """Covers for every matched reading of a scan, laid out clear of *obstacles* (live
        covers, areas this maker must stay out of) and of each other. Matches *accept*
        turns down are not covered (they still keep the translations of others clear)."""
        fh, fw = frame.shape[:2]
        rows, other = self.rows(frame, lines)
        if not self.content.everything:  # the player turned the story or the UI off
            other += [r.line for r in rows if not self.content.allows(is_story(r.hit.line))]
            rows = [r for r in rows if self.content.allows(is_story(r.hit.line))]
        if accept is not None:
            other += [r.line for r in rows if not accept(r)]
            rows = [r for r in rows if accept(r)]
        rows.sort(key=lambda r: (r.extent.ink[1], r.extent.ink[0]))
        heights = [r.extent.glyph_h for r in rows]
        blocks = group_blocks(rows, heights, other, fw, fh)
        # one size per look: labels of one row (a tab bar) or one column (a stat list, a
        # menu, stacked key hints) share the median of their measured heights, which is
        # then snapped to the game's text sizes
        looks = [0.0] * len(rows)
        for group in size_groups(blocks, rows, heights):
            value = float(np.median([heights[i] for b in group for i in blocks[b]]))
            look = self.cache.looks.snap(value, fh)
            for b in group:
                for i in blocks[b]:
                    looks[i] = look
        # what the translation must never be drawn over: every other text the scan saw, and
        # the parts of a detector box that are not the matched text (an icon, a badge)
        beside = [rest_of_box(frame, box_px(r.line, fw, fh), r.extent) for r in rows]
        texts: list[Rect] = [box_px(ln, fw, fh) for ln in other]
        for r, parts in zip(rows, beside, strict=True):
            texts.append(r.extent.cover)
            texts.extend(parts)
        out: list[Placement] = []
        taken: list[Rect] = list(obstacles)
        for blk in blocks:
            own = [part for i in blk for part in beside[i]]
            p = self._place(frame, blk, rows, looks, texts, taken, own)
            if p is None:
                continue
            taken.append(p.geom.box)
            out.append(p)
        return out

    def _place(
        self,
        frame: np.ndarray,
        blk: list[int],
        rows: list[Row],
        looks: list[float],
        texts: list[Rect],
        taken: list[Rect],
        beside: list[Rect],
    ) -> Placement | None:
        fh, fw = frame.shape[:2]
        rs = [rows[i] for i in blk]
        look = looks[blk[0]]
        font_px = font_px_for_glyph(look)
        hits = _distinct_hits(rs)
        template = " ".join(h.template for h in hits)
        text = " ".join(h.translation for h in hits)
        cover = rs[0].extent.cover
        for r in rs[1:]:
            cover = union(cover, r.extent.cover)
        light = 2 * sum(1 for r in rs if r.extent.light) >= len(rs)
        fill, edge = self.cache.palette.snap(*block_colours(rs))
        story = any(is_story(h.line) for h in hits)
        rec = self.cache.find(template, cover, fw, fh)
        if rec is not None:
            geom = self._cached(rec, cover, text, fw, fh)
            if geom is not None:
                if geom.fg is None:
                    geom.fg, geom.edge = fill, edge
                return Placement(template, text, geom, True, look, light, _cells(rs), story)
        obstacles = [t for t in texts if not any(intersects(t, r.extent.cover) for r in rs)]
        obstacles += taken + beside
        ink_top = min(r.extent.ink[1] for r in rs)
        if len(rs) == 1:
            r = rs[0]
            if text_width(text, font_px) + 2 * PAD_X > cover[2]:
                obstacles += near_blobs(frame, cover, look)
            align = self.alignment(cover, look, rows, blk, fw, fh)
            geom = layout_line(cover, text, font_px, align, obstacles, fw, ink_top, r.extent.ink[3])
        else:
            pitch = (rs[-1].extent.ink[1] - rs[0].extent.ink[1]) / (len(rs) - 1)
            geom = layout_block(
                cover,
                text,
                font_px,
                len(rs),
                pitch,
                "left",
                obstacles,
                fw,
                ink_top,
                look,
                self.spaced,
            )
        geom.fg, geom.edge = fill, edge
        return Placement(template, text, geom, False, look, light, _cells(rs), story)

    def _cached(self, rec: dict, cover: Rect, text: str, fw: int, fh: int) -> Geom | None:
        """The remembered block for this line, when the text (a template's numbers may be
        others now: "Lv.5", then "Lv.100") still fits it."""
        geom = place(rec, cover, [text], fw, fh)
        size = drawn_size(geom.font_px, geom.stretch)
        if rec.get("rows", 1) > 1:
            geom.lines = wrap(text, size, geom.box[2] - 2 * PAD_X, self.spaced)
            if len(geom.lines) > rec["rows"]:
                return None
        fits = all(text_width(ln, size) <= geom.box[2] - PAD_X for ln in geom.lines)
        return geom if fits else None

    def alignment(
        self, cover: Rect, look: float, rows: list[Row], blk: list[int], fw: int, fh: int
    ) -> str:
        """How the text is laid out, so a longer translation grows the right way: a hint in
        the game profile for the area, else how the labels stacked above or below this one
        line up with it (each votes for its left edge, right edge or centre, whichever is
        closest: a centred title card's rows of different widths share their centres, and
        their left edges only roughly), else centred."""
        x, y, w, h = cover
        hint = self.profile.align_at((x + w / 2) / fw, (y + h / 2) / fh)
        if hint:
            return hint
        tol = ALIGN_TOL * look
        votes = dict.fromkeys(_PREFER, 0)
        for i, r in enumerate(rows):
            if i in blk:
                continue
            ox, oy, ow, oh = r.extent.cover
            far = abs((oy + oh / 2) - (y + h / 2)) > NEIGHBOUR_ROWS * max(h, oh)
            if far or abs(oy - y) < LEVEL * h:
                continue
            d = {
                "left": abs(ox - x),
                "center": abs((ox + ow / 2) - (x + w / 2)),
                "right": abs((ox + ow) - (x + w)),
            }
            best = min(d, key=lambda k: (d[k], _PREFER[k]))
            if d[best] <= tol:
                votes[best] += 1
        if not any(votes.values()):
            return "center"
        return max(votes, key=lambda k: (votes[k], -_PREFER[k]))

    # ---- placements -> board -------------------------------------------------------------
    def commit(
        self,
        placements: list[Placement],
        read: np.ndarray,
        newest: np.ndarray,
        board: Board,
        now: float,
        others: list[Board] | None = None,
    ) -> int:
        """Put placements on *board*. Skipped: a line that is already shown there, a spot
        another cover still occupies (the watch removes it when its text goes; the OCR
        never does), and text that did not hold still while the OCR ran (still fading in,
        sliding, typing out under it): it is read again on the next tick. A cover memory
        brought back (:mod:`hud.recall`) is confirmed by reading the same line there, and
        gives way to a reading of another line."""
        placed = 0
        for p in placements:
            shown = board.find(p.template, p.geom.cover)
            if shown is not None:
                shown.recalled = False
                if shown.text != p.text:  # the same line with other numbers
                    self._retext(board, shown, p, read, newest)
                continue
            clash = board.clashing(p.geom.box)
            # a cover memory brought back gives way to a reading of another line, if that
            # reading held still: one of a list still scrolling is of what was there before
            overrule = bool(clash) and all(c.recalled and c.template != p.template for c in clash)
            if (clash and not overrule) or any(b.taken(p.geom.box) for b in others or []):
                continue
            anchor = self._still(p, read, newest)
            if anchor is None:
                continue
            if overrule:
                board.remove(clash, "the OCR read another line here")
            cover = Cover(
                p.template, p.text, p.geom, anchor, placed=now, cached=p.cached,
                light=p.light, cells=p.cells, story=p.story,
            )  # fmt: skip
            # (checked again under the board's lock: a recall on the GUI thread may have put
            # the same line there while this one was measured)
            if board.add_if_free(cover, others or []):
                placed += 1
        return placed

    def _still(self, p: Placement, read: np.ndarray, newest: np.ndarray):  # noqa: ANN202
        """The anchor of the text under *p* in the newest frame, if it held still since the
        frame that was read; else None (and the area is read again)."""
        ref = make_anchor(read, p.geom.cover, p.light, p.cells)
        if ref is None or (newest is not read and not settled(newest, ref)):
            self.unsettled += 1
            self._unsettled.append(p.geom.cover)
            return None
        if newest is read:
            return ref
        return make_anchor(newest, p.geom.cover, p.light, p.cells)

    def _retext(self, board: Board, shown: Cover, p: Placement, read, newest) -> None:
        """A cover of a templated line whose numbers changed (a timer counting down): the
        new text, once the new digits held still."""
        anchor = self._still(p, read, newest)
        if anchor is not None:
            board.retext(shown, p.text, p.geom, anchor)

    def take_unsettled(self) -> list[Rect]:
        out, self._unsettled = self._unsettled, []
        return out

    def remember(self, board: Board, now: float, fw: int, fh: int) -> None:
        """Menu covers that have stayed for CACHE_AFTER_S are settled: their blocks go into
        the permanent cache (a block measured mid-animation never lasts that long). Story
        covers are never kept."""
        for c in board.snapshot():
            if c.cached or now - c.placed < CACHE_AFTER_S:
                continue
            c.cached = True
            if not c.story and self.cache.put(c.template, c.geom, fw, fh):
                log.info("cover cached: %s at %s", c.text[:40], c.geom.box)


def _distinct_hits(rs: list[Row]) -> list[Hit]:
    """The hits of a block's rows, each once (the rows of one paragraph share theirs)."""
    out: list[Hit] = []
    for r in rs:
        if not any(h is r.hit for h in out):
            out.append(r.hit)
    return out


def block_colours(rs: list[Row]) -> Colours:
    """The colours of a block's Japanese: the median fill of its rows, and an outline when
    most rows have one."""
    fill = tuple(int(np.median([r.extent.fill[c] for r in rs])) for c in range(3))
    edges = [r.extent.edge for r in rs if r.extent.edge is not None]
    if 2 * len(edges) <= len(rs):
        return fill, None
    return fill, tuple(int(np.median([e[c] for e in edges])) for c in range(3))


def _cells(rs: list[Row]) -> list[tuple[int, int]] | None:
    """The characters' columns of a one-row cover (a block's rows share columns)."""
    return rs[0].extent.cells if len(rs) == 1 else None


def rest_of_box(frame: np.ndarray, box: Rect, extent: Extent) -> list[Rect]:
    """What the detector took in left and right of the matched text that must stay bare:
    the characters read but not matched (the recognizer saw something there: an icon, a
    bracket the line does not have, however faint) and any crisp shape in the rest (an
    arrow, a badge). Empty slack is not an obstacle."""
    bx, by, bw, bh = box
    cx, cy, cw, ch = extent.cover
    out: list[Rect] = []
    for ex, ey, ew, eh in extent.excluded:
        # the part outside the cover (a cell may reach into the cover's margin)
        x0, x1 = ex, ex + ew
        if x1 <= cx + cw / 2:
            x1 = min(x1, cx - 1)
        else:
            x0 = max(x0, cx + cw + 1)
        if x1 > x0:
            out.append((x0, ey, x1 - x0, eh))
    for part in ((bx, cy, cx - bx, ch), (cx + cw, cy, (bx + bw) - (cx + cw), ch)):
        if part[2] >= 3:
            found = blobs(frame, part, extent.glyph_h)
            if found is not None:
                out.append(found)
    return out


def paragraph_stacks(lines: list[OcrLine], fw: int, fh: int) -> list[list[OcrLine]]:
    """Readings stacked like the rows of one paragraph (top to bottom): the same height,
    the same left edge or the same middle, a line's spacing apart, overlapping columns,
    nothing else between them. Only stacks of two rows or more."""
    order = sorted(lines, key=lambda ln: (ln.box.y, ln.box.x))
    stacks: list[list[OcrLine]] = []
    used: set[int] = set()
    for i, first in enumerate(order):
        if i in used:
            continue
        stack = [first]
        used.add(i)
        for j in range(i + 1, len(order)):
            if j in used or len(stack) >= PARAGRAPH_MAX:
                continue
            if _continues(stack[-1], order[j], order, fw, fh):
                stack.append(order[j])
                used.add(j)
        if len(stack) >= 2:
            stacks.append(stack)
    return stacks


def _continues(a: OcrLine, b: OcrLine, everything: list[OcrLine], fw: int, fh: int) -> bool:
    ax, ay, aw, ah = box_px(a, fw, fh)
    bx, by, bw, bh = box_px(b, fw, fh)
    h = max(ah, bh)
    gap = by - (ay + ah)
    if not (-ROW_OVERLAP * h <= gap <= PARAGRAPH_GAP * h):
        return False
    if abs(ah - bh) > PARAGRAPH_SIZE * h:
        return False
    # the same left edge, or the same middle (a centred popup message)
    if abs(ax - bx) > BLOCK_ALIGN * h and abs((ax + aw / 2) - (bx + bw / 2)) > BLOCK_ALIGN * h:
        return False
    if min(ax + aw, bx + bw) - max(ax, bx) < OVERLAP_SHARE * min(aw, bw):
        return False
    between = (min(ax, bx), ay + ah, max(ax + aw, bx + bw) - min(ax, bx), max(1, gap))
    return not any(
        ln is not a and ln is not b and intersects(box_px(ln, fw, fh), between) for ln in everything
    )


def size_groups(blocks: list[list[int]], rows: list[Row], heights: list[float]) -> list[list[int]]:
    """Blocks that look alike and belong together: on one row (centres level, a tab bar)
    or stacked in one column (left edges, right edges or centres in line), with glyph
    heights within SAME_SIZE of each other. Returns groups of block indices."""
    parent = list(range(len(blocks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def box(b: list[int]) -> tuple[float, float, float, float, float]:
        x0 = min(rows[i].extent.ink[0] for i in b)
        y0 = min(rows[i].extent.ink[1] for i in b)
        x1 = max(rows[i].extent.ink[0] + rows[i].extent.ink[2] for i in b)
        y1 = max(rows[i].extent.ink[1] + rows[i].extent.ink[3] for i in b)
        return x0, y0, x1, y1, float(np.median([heights[i] for i in b]))

    def kana_only(b: list[int]) -> bool:
        text = "".join(rows[i].line.text[slice(*rows[i].hit.span)] for i in b)
        return bool(_KANA_RE.search(text)) and not _KANJI_RE.search(text)

    boxes = [box(b) for b in blocks]
    kana = [kana_only(b) for b in blocks]
    for a in range(len(blocks)):
        ax0, ay0, ax1, ay1, ah = boxes[a]
        for b in range(a + 1, len(blocks)):
            bx0, by0, bx1, by1, bh = boxes[b]
            # (katakana sit lower in the em than kanji: ボイス beside 情報, one size)
            same = SAME_SIZE if kana[a] == kana[b] else max(SAME_SIZE, KANA_SHORT * max(ah, bh))
            if abs(ah - bh) > same:
                continue
            h = max(ah, bh)
            level = abs((ay0 + ay1) / 2 - (by0 + by1) / 2) <= LEVEL * h
            stacked = min(abs(by0 - ay1), abs(ay0 - by1)) <= STACK_REACH * h
            flush = (
                abs(ax0 - bx0) <= ALIGN_TOL * h
                or abs(ax1 - bx1) <= ALIGN_TOL * h
                or abs((ax0 + ax1) / 2 - (bx0 + bx1) / 2) <= ALIGN_TOL * h
            )
            if level or (stacked and flush):
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(len(blocks)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def group_blocks(
    rows: list[Row], heights: list[float], other: list[OcrLine], fw: int, fh: int
) -> list[list[int]]:
    """Paragraphs: the rows of one database text (matched together: they share a hit), and
    matched rows (sorted top to bottom) of the same size and colour that follow each other
    at text-line spacing, start at the same left edge (a hanging indent allowed) and run
    long, with no unmatched text between them. Everything else is a block of one."""
    blocks: list[list[int]] = []
    unmatched = [box_px(ln, fw, fh) for ln in other]
    used: set[int] = set()
    for i in range(len(rows)):
        if i in used:
            continue
        shared = [j for j in range(len(rows)) if j not in used and rows[j].hit is rows[i].hit]
        if len(shared) > 1:  # one text the OCR read row by row
            used.update(shared)
            blocks.append(shared)
            continue
        blk = [i]
        used.add(i)
        while True:
            nxt = _next_row(rows, heights, blk[-1], used, unmatched)
            if nxt is None:
                break
            blk.append(nxt)
            used.add(nxt)
        blocks.append(blk)
    return blocks


def _next_row(
    rows: list[Row], heights: list[float], i: int, used: set[int], unmatched: list[Rect]
) -> int | None:
    a = rows[i]
    ax, ay, aw, ah = a.extent.ink
    look = heights[i]
    if len(a.hit.line.source_text) < BLOCK_MIN_CHARS and len(a.line.text) < BLOCK_MIN_CHARS:
        return None
    for j in range(len(rows)):
        if j in used or j == i:
            continue
        b = rows[j]
        if any(rows[k].hit is b.hit for k in range(len(rows)) if k != j):
            continue  # part of a text read row by row: a block of its own
        bx, by, bw, bh = b.extent.ink
        gap = by - (ay + ah)
        if not (-ROW_OVERLAP * look <= gap <= BLOCK_GAP * look):
            continue
        if abs(heights[j] - look) > SAME_SIZE:
            continue
        if not same_colour(a.extent.fill, b.extent.fill):
            continue  # a gold quest title over its grey objective is two texts
        if abs(bx - ax) > BLOCK_ALIGN * look:
            continue
        overlap = min(ax + aw, bx + bw) - max(ax, bx)
        if overlap < OVERLAP_SHARE * min(aw, bw):
            continue
        # nothing unread between the two rows
        between = (min(ax, bx), ay + ah, max(ax + aw, bx + bw) - min(ax, bx), max(1, gap))
        if any(intersects(u, between) for u in unmatched):
            continue
        return j
    return None
