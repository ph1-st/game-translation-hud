"""The dialogue band's one owner: what the OCR reads there goes nowhere else."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

from capture.ocr import OcrLine
from capture.profile import DialogueLayout, Profile, Region
from db.glossary import Glossary
from db.lookup import Lookup
from hud.anchors import HARD_FAIL, HOLDS, Anchor, anchor_state, make_anchor, settled
from hud.backdrop import text_pixels
from hud.board import DOUBT_FRAMES, Board
from hud.covercache import CoverCache
from hud.covers import CoverMaker, px
from hud.dialogue import panes
from hud.dialogue.divider import DividerWatch
from hud.dialogue.names import Names
from hud.dialogue.panes import FIXED, GROW, MODES
from hud.dialogue.reader import Running, TerritoryReader, centre, grown, reads_as
from hud.dialogue.scene import (
    PANEL_PAD,
    DialogueRead,
    Scene,
    Shown,
    Speaker,
    align_of,
    lay_out,
    reveal,
    text_area,
)
from hud.extent import PAD
from hud.ink import Rect
from hud.items import OverlayItem, reveal_share
from hud.layout import fit_line, intersects
from hud.looks import PLAIN_LIGHT
from hud.style import CoverStyle
from hud.textwidth import font_px_for_glyph

log = logging.getLogger("hud")

BALANCED, INSTANT = "balanced", "instant"
CONFIRMATIONS = (BALANCED, INSTANT)
PENDING_NAME_S = 1.0  # a name read before its line is the line's speaker only this soon
SCENE_GAP_S = 1.5  # a scene stays this long after a line's Japanese went (the next line of
# a conversation comes up to a second later)...
SCENE_HOLD_S = 6.0  # ...and as long as this while dialogue text is still in the band
TEXT_SHARE = 0.002  # stroke pixels (share of the panel) that mean "dialogue text in the band"
TEXT_MIN = 25  # ...and at least this many
GREW = 1.3  # a reading this much longer than the shown line's may replace it (typing on)
CROP_MARGIN = 0.02  # around the territory when it is read (relative to the frame)
SPEAKER_MARGIN = 0.01  # around a speaker band when it is read on its own
PANE_IN_S = 0.15  # a pane of frosted glass fades in this long...
PANE_OUT_S = 0.25  # ...and out once its scene is over
DEFAULT_GLYPH = 0.02  # a layout's glyph height (of the frame) before one is measured


class DialogueComponent:
    def __init__(
        self,
        profile: Profile,
        lookup: Lookup,
        glossary: Glossary,
        cache: CoverCache,
        maker: CoverMaker,
        style: CoverStyle,
        mode: str = GROW,
        confirmation: str = BALANCED,
        spaced: bool = True,
        rtl: bool = False,
    ) -> None:
        """*mode*: how the frosted glass lies (:mod:`hud.dialogue.panes`); *confirmation*:
        ``balanced`` (a line is shown once its Japanese held still while it was read) or
        ``instant`` (a strong match is shown from the frame that was read); *spaced* and
        *rtl*: how the translation language is written."""
        if mode not in MODES:
            raise ValueError(f"unknown dialogue mode {mode!r} (one of {', '.join(MODES)})")
        self.profile = profile
        self.lookup = lookup
        self.cache = cache
        self.maker = maker
        self.style = style
        self.mode = mode
        self.confirmation = confirmation if confirmation in CONFIRMATIONS else BALANCED
        self.spaced, self.rtl = spaced, rtl
        self.names = Names(profile, glossary, lookup, cache)
        self.dividers = DividerWatch(profile, cache)
        self.reader = TerritoryReader(profile, lookup, maker, self.names)
        self.board = Board("band", style)  # labels in the territory (tabs, buttons)
        self.board.rtl = rtl
        self.scene: Scene | None = None
        self.version = 0
        self.provisional: set[str] = set()  # lines translated live: shown a little faded
        # (text, kind, speaker, confidence, box) of a dialogue line no database line matches
        self.unmatched: Callable[[str, str, str | None, float, Rect], None] | None = None
        self.on_line: Callable[[str, str, str | None], None] | None = None  # (source, text, name)
        self._last_seq: int | None = None  # the script position of the line shown last
        # a name read before its line: (speaker, the frame it was read from, when)
        self._pending_name: tuple[Speaker, np.ndarray, float] | None = None
        self._lock = threading.Lock()

    # ---- where it lives ----------------------------------------------------------------
    def territory(self) -> list[Region]:
        out: list[Region] = []
        if self.profile.dialogue is not None:
            out.append(self.profile.dialogue)
        for lay in self.profile.dialogue_layouts.values():
            if lay.speaker is not None:
                out.append(lay.speaker)
        return out

    def territory_px(self, fw: int, fh: int) -> list[Rect]:
        return [px(r, fw, fh) for r in self.territory()]

    def crop(self) -> Region | None:
        """The region the OCR reads for this component: the territory's bounding box, with
        a margin (the detector misses text that touches the edge of its image)."""
        regs = self.territory()
        if not regs:
            return None
        return grown(
            min(r.x for r in regs), min(r.y for r in regs),
            max(r.x2 for r in regs), max(r.y2 for r in regs), CROP_MARGIN,
        )  # fmt: skip

    def owns(self, line: OcrLine) -> bool:
        return any(r.contains_point(*centre(line)) for r in self.territory())

    def glass_areas(self, fw: int, fh: int) -> list[Rect]:
        """Where the running scene's glass may lie (nothing else is recalled into it)."""
        with self._lock:
            sc = self.scene
            if sc is None:
                return []
            return [sc.panel, text_area(sc, fw, fh)]

    def divider_seen(self, now: float) -> bool:
        return self.dividers.any_seen(now)

    # ---- reading (worker thread) -------------------------------------------------------
    def read(self, frame: np.ndarray, lines: list[OcrLine], ui_boxes: list[Rect]) -> DialogueRead:
        """What the territory shows; no secondary OCR on this path (see :meth:`refine_speaker`)."""
        lines = [ln for ln in lines if self.owns(ln) and not self.profile.is_ignored(ln.box)]
        with self._lock:
            sc = self.scene
            running = Running(
                sc.layout if sc else None, sc.panel if sc else None, sc.name_box if sc else None
            )
        return self.reader.read(frame, lines, ui_boxes, running, self._last_seq, self.unmatched)

    def needs_name_check(self, read: DialogueRead) -> bool:
        """Does the line's speaker still want reading on its own (the database's speaker,
        not yet seen on screen; or no name read at all)?"""
        lay = read.layout
        if lay is None or lay.speaker is None:
            return False
        if read.speaker is not None and read.speaker.verified:
            return False
        with self._lock:
            sc = self.scene
            if sc is not None and sc.name_anchor is not None and sc.name == read.speaker_hint:
                return False  # the name on screen is that one, watched by its strokes
        return True

    def refine_speaker(
        self, frame: np.ndarray, read: DialogueRead, ocr: Callable[[Region], list[OcrLine]]
    ) -> Speaker | None:
        """Read the line's name band on its own (after the line is shown), to tie the name
        to its pixels."""
        lay = read.layout
        sp = lay.speaker if lay is not None else None
        if sp is None:
            return None
        # with the database's speaker one crop is a check; without it, a tighter second
        # crop too (light names over bright scenes are found in tight crops only)
        margins = (SPEAKER_MARGIN,) if read.speaker_hint else (SPEAKER_MARGIN, 0.0)
        for m in margins:
            found = self.names.find(frame, lay, ocr(grown(sp.x, sp.y, sp.x2, sp.y2, m)))
            if found is not None:
                return found
        return None

    # ---- applying a read (worker thread) -----------------------------------------------
    def commit(
        self, read: DialogueRead, frame: np.ndarray, newest: np.ndarray, now: float, ui_board: Board
    ) -> None:
        fh, fw = newest.shape[:2]
        shown = None
        with self._lock:
            changed = self._labels_end_scene(read)
            changed = self._commit_speaker(read, frame, newest, now, fw, fh) or changed
            if read.hit is not None and read.layout is not None and read.rows_cover is not None:
                line_changed, shown = self._commit_line(read, frame, newest, now, fw, fh)
                changed = line_changed or changed
            if changed:
                self.version += 1
            labels = read.labels
            if self.scene is not None:
                labels = [p for p in labels if not intersects(p.geom.box, self.scene.panel)]
        on_line = self.on_line  # (outside the lock: a slow or failing listener holds nothing)
        if shown is not None and on_line is not None:
            try:
                on_line(*shown)
            except Exception:  # noqa: BLE001 - the conversation's memory is a nicety
                log.exception("dialogue listener failed")
        self.maker.commit(labels, frame, newest, self.board, now, others=[ui_board])

    def _labels_end_scene(self, read: DialogueRead) -> bool:
        """A menu's labels on the panel: the scene is over (under the lock)."""
        sc = self.scene
        if sc is None or sc.layout.keep:
            return False
        if any(intersects(p.geom.cover, sc.panel) for p in read.labels):
            log.info("dialogue: labels on the panel, scene over")
            self.scene = None
            return True
        return False

    def _commit_speaker(
        self, read: DialogueRead, frame, newest, now: float, fw: int, fh: int
    ) -> bool:
        spk = read.speaker
        if spk is None:
            return False
        sc = self.scene
        if sc is not None and sc.layout.name == spk.layout.name:
            if sc.name != spk.name or (spk.verified and sc.name_anchor is None):
                return self.names.set(sc, spk, fw, fh, newest, frame)
            return False
        self._pending_name = (spk, frame, now)
        return False

    def _commit_line(
        self, read: DialogueRead, frame, newest, now: float, fw: int, fh: int
    ) -> tuple[bool, tuple[str, str, str | None] | None]:
        """Show the read line, or let the one shown type on (under the lock). Returns
        (changed, (source, text, speaker) of a new line)."""
        sc = self.scene
        hit = read.hit
        assert hit is not None and read.layout is not None and read.rows_cover is not None
        # one line is one translation, whichever OCR variant of it was matched and whatever
        # layout a partial reading of it suggested: it stays as it is
        same = sc is not None and sc.line is not None and sc.line.text == hit.translation
        # a shown line is replaced only when its strokes went (the watch) or when the text
        # typed on and now reads as another line; another OCR variant of the same Japanese
        # (the same length) never swaps the translation
        grew = (
            sc is not None
            and sc.line is not None
            and read.reading >= GREW * max(1, sc.line.reading)
            and not reads_as(read.text, sc.line.source)
        )
        if same and sc is not None and sc.line is not None:
            return self._typed_on(sc, read, now, fw, fh), None
        if sc is not None and sc.line is not None and not grew:
            return False, None  # another reading while a line is shown: the watch decides
        changed = False
        if sc is not None and sc.layout.name != read.layout.name:
            self.scene = sc = None
            changed = True
        anchor = self._stable(read.rows_cover, read.light, frame, newest)
        if anchor is None:
            return changed, None
        if sc is None:
            sc = self.scene = self._start(read, fw, fh)
        sc.line = Shown(hit.translation, anchor, read.reading, hit.line.source_text)
        sc.line.typed, sc.line.typed_at = len(read.text), now
        sc.line.reveal = reveal(sc, 0.0, now)
        self._learn_pitch(sc, read, fh)
        lay_out(sc, hit.translation, fw, fh, self.spaced)
        panes.from_read(sc, [r for r in read.row_inks], fw, fh)
        log.info("dialogue (%s): %s", sc.layout.name, hit.translation[:60])
        self._last_seq = hit.line.seq
        self._take_pending_name(sc, now, fw, fh, newest)
        return True, (hit.line.source_text, hit.translation, sc.name or read.speaker_hint)

    def _typed_on(self, sc: Scene, read: DialogueRead, now: float, fw: int, fh: int) -> bool:
        """The line shown typed on: its translation types on with it, at a pace that changes
        from where it is (never a pause)."""
        assert sc.line is not None
        changed = False
        if len(read.text) > sc.line.typed:
            sc.line.typed, sc.line.typed_at = len(read.text), now
            sc.line.reveal = reveal(sc, reveal_share(sc.line.reveal, now), now)
            changed = True
        return panes.from_read(sc, read.row_inks, fw, fh) or changed

    def _learn_pitch(self, sc: Scene, read: DialogueRead, fh: int) -> None:
        """The layout's row pitch, from its first line of two rows (learned once, kept)."""
        if sc.row_pitch or len(read.row_inks) < 2:
            return
        pitch = read.row_inks[1][1] - read.row_inks[0][1]
        if pitch > sc.glyph:
            rec = self.cache.set_dialogue_field(sc.layout.name, "pitch", pitch / fh)
            sc.row_pitch = rec["pitch"] * fh

    def _take_pending_name(self, sc: Scene, now: float, fw: int, fh: int, newest) -> None:
        """A name read just before its line is that line's speaker (its strokes checked like
        any name read: they held still since)."""
        pend, self._pending_name = self._pending_name, None
        if (
            pend is not None
            and pend[0].layout.name == sc.layout.name
            and now - pend[2] <= PENDING_NAME_S
            and sc.name is None
        ):
            self.names.set(sc, pend[0], fw, fh, newest, pend[1])

    def _stable(
        self, rect: Rect, light: bool, frame: np.ndarray, newest: np.ndarray
    ) -> Anchor | None:
        """The anchor of a line's Japanese if it held still while it was read (what is
        already typed out stays as it is while the rest types out). In the instant mode a
        strong match is trusted from the frame that was read: a short line the next frame
        no longer shows is still shown."""
        ref = make_anchor(frame, rect, light)
        if ref is None:
            return None
        if self.confirmation == INSTANT or newest is frame:
            return ref
        if not settled(newest, ref):
            return None
        return make_anchor(newest, rect, light)

    # ---- scenes ------------------------------------------------------------------------
    def _start(self, read: DialogueRead, fw: int, fh: int) -> Scene:
        """A scene in the line's layout: its font and colours are fixed the first time the
        layout is seen and never change (kept in the cover cache), as is where its text
        goes (the first row of the first line read in it; where rows grow upwards, the
        bottom of its panel)."""
        lay = read.layout
        assert lay is not None
        rec = self.cache.dialogue_font(lay.name)
        if rec is None or "font" not in rec:
            look = self.cache.looks.snap(read.glyph if read.glyph > 0 else DEFAULT_GLYPH * fh, fh)
            fill, edge = self.cache.palette.snap(read.fill, read.edge)
            rec = self.cache.set_dialogue_font(lay.name, {
                "glyph": look / fh,
                "font": font_px_for_glyph(look) / fh,
                "fill": list(fill),
                "edge": list(edge) if edge is not None else None,
            })  # fmt: skip
        if "text_y" not in rec:
            font_px = max(6, int(round(rec["font"] * fh)))
            x, y, w, h = px(lay.panel, fw, fh)
            if lay.anchor == "bottom":
                y_px = y + h - PANEL_PAD * font_px
            elif read.row_inks:
                y_px = float(read.row_inks[0][1])
            else:
                y_px = y + PANEL_PAD * font_px
            rec = self.cache.set_dialogue_field(lay.name, "text_y", y_px / fh)
        if "align" not in rec:
            rec = self.cache.set_dialogue_field(
                lay.name, "align", align_of(lay, read.row_inks, fw, fh)
            )
        sc = self._scene(lay, rec, fw, fh)
        log.info("dialogue scene (%s): font %d", lay.name, sc.font_px)
        return sc

    def _start_cached(self, lay: DialogueLayout, fw: int, fh: int) -> Scene | None:
        """A scene in *lay* from what the cache keeps of the layout (its divider is up, no
        line has been read yet); None when the layout was never seen."""
        rec = self.cache.dialogue_font(lay.name)
        if rec is None or not all(k in rec for k in ("font", "glyph", "text_y")):
            return None
        sc = self._scene(lay, rec, fw, fh)
        log.info("dialogue scene (%s): up from its cached look", lay.name)
        return sc

    def _scene(self, lay: DialogueLayout, rec: dict, fw: int, fh: int) -> Scene:
        sc = Scene(lay, px(lay.panel, fw, fh), max(6, int(round(rec["font"] * fh))),
                   rec["glyph"] * fh)  # fmt: skip
        sc.fill = tuple(rec.get("fill") or PLAIN_LIGHT)
        sc.edge = tuple(rec["edge"]) if rec.get("edge") is not None else None
        sc.text_y = rec["text_y"] * fh
        sc.row_pitch = rec.get("pitch", 0.0) * fh
        sc.align = lay.align if lay.align != "auto" else str(rec.get("align") or "center")
        if "name" in rec:
            self.names.fix_box(sc, rec["name"], fw, fh)
        if self.mode == FIXED and lay.glass is not None:
            panes.fixed_pane(sc, px(lay.glass, fw, fh))
        return sc

    def end_scene(self) -> None:
        with self._lock:
            self._pending_name = None
            if self.scene is not None:
                self.scene = None
                self.version += 1

    # ---- the watch (GUI thread) --------------------------------------------------------
    def watch(self, frame: np.ndarray, now: float) -> list[Rect]:
        """The line's translation goes the moment its Japanese strokes vanish or change. A
        layout with a divider is up the moment its divider is on screen and over once the
        divider has been gone a moment with no line shown; one without ends after a short
        gap without a line. The glass grows over the text on screen. Returns areas to read
        again."""
        freed: list[Rect] = []
        fh, fw = frame.shape[:2]
        up = self.dividers.up(frame, now)
        with self._lock:
            sc = self._scene_now(up, now, fw, fh, freed)
            if sc is None:
                return freed
            if self.names.watch(frame, sc):
                self.version += 1
            if sc.line is not None:
                self._watch_line(frame, sc, now, freed)
            elif self._scene_over(frame, sc, now):
                log.info("dialogue scene (%s) over", sc.layout.name)
                freed.append(sc.panel)
                self.scene = None
                self.version += 1
                return freed
            if panes.from_strokes(frame, sc):
                self.version += 1
        return freed

    def _scene_now(
        self, up: list[DialogueLayout], now: float, fw: int, fh: int, freed: list[Rect]
    ) -> Scene | None:
        """The scene after the dividers' news: over when its divider went with no line
        shown, up (from the cache) when a divider came or a layout's glass is kept."""
        sc = self.scene
        if (
            sc is not None
            and sc.layout.divider is not None
            and sc.line is None
            and not sc.layout.keep
            and self.dividers.gone(sc.layout, now)
        ):
            log.info("dialogue scene (%s) over: its divider went", sc.layout.name)
            freed.append(sc.panel)
            self.scene = sc = None
            self.version += 1
        if sc is not None:
            return sc
        # glass kept on screen: up from the start once the cache knows the layout (the
        # fixed window shows at once; the growing pane comes with the first text)
        kept = [lay for lay in self.profile.dialogue_layouts.values()
                if lay.keep and lay.divider is None]  # fmt: skip
        for lay in up + kept:
            sc = self._start_cached(lay, fw, fh)
            if sc is not None:
                self.scene = sc
                self.version += 1
                return sc
        return None

    def _watch_line(self, frame: np.ndarray, sc: Scene, now: float, freed: list[Rect]) -> None:
        assert sc.line is not None
        state = anchor_state(frame, sc.line.anchor)
        if state == HOLDS:
            sc.line.doubtful = 0
            return
        sc.line.doubtful += 1
        if state == HARD_FAIL or sc.line.doubtful >= DOUBT_FRAMES:
            log.info("dialogue gone: %s", sc.line.text[:40])
            freed.append(sc.line.anchor.rect)
            sc.line, sc.lines, sc.line_gone_at = None, [], now
            self.version += 1

    def _scene_over(self, frame: np.ndarray, sc: Scene, now: float) -> bool:
        """A scene without a divider ends a moment after its last line went, unless the
        next line is coming in (dialogue text in the band: the glass stays over it)."""
        if sc.layout.divider is not None or sc.layout.keep:
            return False
        if now - sc.line_gone_at <= SCENE_GAP_S:
            return False
        return not (now - sc.line_gone_at < SCENE_HOLD_S and self._text_in_band(frame, sc))

    @staticmethod
    def _text_in_band(frame: np.ndarray, sc: Scene) -> bool:
        """Is there dialogue text (strokes in the dialogue colours) in the panel?"""
        _x, _y, w, h = sc.panel
        found = text_pixels(frame, sc.panel, sc.glyph, sc.fill, sc.edge)
        return found >= max(TEXT_MIN, TEXT_SHARE * w * h)

    # ---- drawing ---------------------------------------------------------------------
    def items(self) -> list[OverlayItem]:
        with self._lock:
            sc = self.scene
            if sc is None:
                return []
            out = [self._pane_item(sc, p.rect, f"dlg:pane:{sc.layout.name}:{p.key}")
                   for p in sc.panes]  # fmt: skip
            if sc.name_glass is not None:
                out.append(self._pane_item(sc, sc.name_glass, f"dlg:namebox:{sc.layout.name}",
                                           sc.name_glyph or sc.glyph))  # fmt: skip
            if sc.line is not None and sc.lines and sc.text_box is not None:
                out.append(self._line_item(sc))
            if sc.name is not None and sc.name_box is not None:
                out.append(self._name_item(sc))
            return out

    def _pane_item(self, sc: Scene, rect: Rect, ident: str, glyph: float = 0.0) -> OverlayItem:
        return OverlayItem(
            rect, row_h=glyph or sc.glyph, style="panel", frost=rect, shine=self.style.shine,
            fade_in=PANE_IN_S, fade_out=PANE_OUT_S, id=ident,
        )  # fmt: skip

    def _line_item(self, sc: Scene) -> OverlayItem:
        assert sc.line is not None and sc.text_box is not None
        return OverlayItem(
            sc.text_box,
            lines=sc.lines,
            font_px=sc.text_font or sc.font_px,
            stretch=sc.text_stretch,
            row_top=sc.text_top,
            row_h=sc.text_band,
            pitch=sc.text_pitch,
            align="left" if sc.align == "left" else "center",
            style="text",
            fg=sc.fill,
            edge=sc.edge,
            reveal=sc.line.reveal if sc.layout.reveal else None,
            rtl=self.rtl,
            provisional=sc.line.text in self.provisional,
            id=f"dlg:line:{sc.line.text}",
        )

    def _name_item(self, sc: Scene) -> OverlayItem:
        assert sc.name is not None and sc.name_box is not None
        box = sc.name_glass if sc.name_glass is not None else sc.name_box
        font, stretch, name = fit_line(sc.name, sc.name_font, box[2] - 2 * PAD)
        return OverlayItem(
            box,
            lines=[name],
            font_px=font,
            stretch=stretch,
            row_top=sc.name_box[1] + PAD,
            row_h=sc.name_glyph,
            align="center",
            style="text",
            fg=sc.name_fill,
            edge=sc.name_edge,
            rtl=self.rtl,
            id=f"dlg:name:{sc.name}",
        )
