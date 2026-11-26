"""The HUD's core, without any window: one OCR tick, and the 50 ms pixel watch.

Once per tick (as fast as the OCR goes) the worker takes the newest frame and reads only
what needs reading:

* the areas whose pixels changed since they were last OCRed (a small grey copy of the
  screen as it looked when each part was last read is kept), and
* the areas the pixel watch just freed: a cover went because its Japanese went, and
  whatever replaced it is read at once.

Changed rows become full-width bands, so a line of text is never cut in half; when they
add up to much of the screen, the whole frame is read. The dialogue territory is never part
of a UI read (it is masked out): when it changed, the dialogue component reads it, and its
readings go nowhere else. What is read and how often follows what is on screen:

* a conversation (or the story only translated): the band on every tick, the rest of the
  screen once a second
* a menu (many labels covered, no conversation): the band at most every 0.6 s
* exploring (neither): a full read at most every 0.3 s, so the game keeps its GPU

and a picture at rest (the same as at the last tick: a menu that stopped scrolling) is read
at once, whatever it shows. With no conversation up, the whole frame is read in one go,
the dialogue territory's share handed to the dialogue.

Every reading then takes the one cover path (:mod:`hud.covers`); in parallel, every 50 ms,
the pixel watch removes covers whose Japanese strokes vanished or changed and hands their
areas back to the next tick, and memory brings back menu covers whose text is on screen
again (:mod:`hud.recall`): a menu seen before is covered within a frame or two, no OCR.

The Japanese under the covers shown is masked out of the OCR's image (the watch vouches for
it). Slots the game scrolls text through (the quest title) belong to :mod:`hud.ticker`: one
line read at rest, shown still while the text slides.

Live translation hears of a reading no database line matches once its text held still while
it was read, the test a cover passes (text still moving is read again). A label's reading is
kept with the frame it was read in: its translation is placed the moment it comes, without
another read.

``now`` is a monotonic clock in seconds (``time.monotonic``), the same one the overlay
types text out by.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from capture.change import ChangeTracker
from capture.ocr import OcrChar, OcrEngine, OcrLine, recognize_boxes
from capture.profile import Profile, Region
from db.glossary import Glossary
from db.lookup import Lookup
from db.store import Line
from db.templates import NUMBER, fill, templatize
from hud.anchors import make_anchor, settled
from hud.board import Board, Cover
from hud.config import COVER_MODES
from hud.content import UI, Content
from hud.covercache import CoverCache
from hud.covers import CoverMaker, box_px
from hud.dialogue import BALANCED, INSTANT, MODES, DialogueComponent, DialogueRead
from hud.ink import Rect
from hud.items import OverlayItem
from hud.layout import intersects
from hud.recall import Recall
from hud.style import CoverStyle
from hud.ticker import Ticker
from translate import languages

log = logging.getLogger("hud")

NATURAL = "natural"  # no dialogue handling: every text covered alike
BAND_MARGIN = 0.025  # a changed row is read with this much (of the frame height) around it
BAND_STEP = 64  # band heights are rounded up to this many pixels: the detector's input
# shapes repeat, and DirectML compiles each new shape anew
FULL_READ = 0.45  # bands covering more than this share of the frame: read the whole frame
MAX_BANDS = 3  # more bands than this cost more than one read of the whole frame
SCENE_UI_EVERY_S = 1.0  # in a conversation, the rest of the screen this long after its last
# read ended (that read takes ~0.45 s in a 3D scene); the band on every tick in between
MENU_COVERS = 4  # this many labels covered and no conversation: a menu is up...
MENU_BAND_EVERY_S = 0.6  # ...and the dialogue band is read at most this often
EXPLORE_FULL_EVERY_S = 0.3  # neither: the whole frame is read at most this often
TIMINGS_KEPT = 120  # read times kept for the diagnostics
LIVE_KEEP_S = 10.0  # a label sent to be translated: its reading and frame are kept this
LIVE_KEEP = 64  # long, this many at most (its translation is placed the moment it comes)
REREAD_KEEP = 512  # second readings remembered (Rereader)

Unmatched = Callable[[str, str, "str | None", float, Rect], bool]


def bands(rows: np.ndarray, extra: list[Rect], fw: int, fh: int) -> list[Rect] | None:
    """Full-width bands around the changed rows (*rows*: per row of the change map) and
    the *extra* rects, their heights rounded up to :data:`BAND_STEP`; None when reading the
    whole frame is cheaper."""
    margin = int(round(BAND_MARGIN * fh))
    spans: list[list[int]] = []
    sy = fh / max(1, len(rows))
    idx = np.nonzero(rows)[0]
    if len(idx):
        start = prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i != prev + 1:
                spans.append([int(start * sy), int((prev + 1) * sy)])
                start = i
            prev = i
        spans.append([int(start * sy), int((prev + 1) * sy)])
    for _x, y, _w, h in extra:
        spans.append([y, y + h])
    if not spans:
        return []
    spans = [_rounded(max(0, a - margin), min(fh, b + margin), fh) for a, b in spans]
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if sum(b - a for a, b in merged) > FULL_READ * fh or len(merged) > MAX_BANDS:
        return None
    return [(0, a, fw, b - a) for a, b in merged]


def _rounded(a: int, b: int, fh: int) -> list[int]:
    """The span [a, b) grown to a multiple of BAND_STEP rows (downwards, or upwards at the
    bottom of the frame)."""
    h = min(fh, int(math.ceil((b - a) / BAND_STEP)) * BAND_STEP)
    a = min(a, fh - h)
    return [a, a + h]


@dataclass
class TickReport:
    ui: str = ""  # "", "full" or "n bands"
    dialogue: bool = False
    ocr_ms: float = 0.0
    dialogue_ms: float = 0.0
    ui_ms: float = 0.0
    placed: int = 0
    unsettled: int = 0  # matched readings dropped because their text was still moving
    readings: list[OcrLine] = field(default_factory=list)


@dataclass
class ReadPlan:
    """What one tick reads."""

    territory: list[Rect]  # the dialogue territory (never part of a UI read)
    quiet: list[Rect]  # ticker slots left alone this tick
    dialogue: bool  # read the dialogue territory
    ui: list[Rect] | None  # UI bands to read; None: the whole frame; []: nothing


class HudCore:
    """Everything between captured frames and overlay items, driven by the app (live) or
    by the offline replay."""

    def __init__(
        self,
        profile: Profile,
        lookup: Lookup,
        glossary: Glossary,
        cache: CoverCache,
        engine: OcrEngine,
        style: CoverStyle,
        cover_mode: str = "grow",
        confirmation_mode: str = BALANCED,
        target_lang: str = "en",
        text_scale: float = 1.0,
        plain_text: bool = False,
    ) -> None:
        """*cover_mode*: ``grow`` (dialogue on frosted glass that grows with the text),
        ``fixed`` (one pane over a dialogue layout's whole glass area) or ``natural`` (no
        dialogue handling: every text covered alike). *confirmation_mode*: ``balanced`` or
        ``instant`` (:class:`hud.dialogue.DialogueComponent`). *target_lang*: the
        translation's language (how its lines break, which way it is written). *text_scale*:
        the player's text size (1: the game's own); *plain_text*: every translation white
        with a black outline instead of the Japanese's colours."""
        if cover_mode not in COVER_MODES:
            raise ValueError(f"unknown cover mode {cover_mode!r} ({', '.join(COVER_MODES)})")
        if cover_mode == NATURAL:
            profile = profile.without_dialogue()
        self.cover_mode = cover_mode
        self.confirmation_mode = INSTANT if confirmation_mode == INSTANT else BALANCED
        self.target_lang = target_lang
        spaced, rtl = languages.spaced(target_lang), languages.rtl(target_lang)
        self.profile = profile
        self.glossary = glossary  # names and terms (live translation is given them too)
        self.engine = engine
        self.cache = cache
        self.maker = CoverMaker(lookup, cache, profile, spaced)
        self.maker.text_scale = text_scale
        self.plain_text = plain_text
        second = getattr(engine, "second", None)  # (MeikiOCR's: capture/cuda_helper.py)
        if second is not None:
            self.maker.reread = Rereader(second)
        self.ui_board = Board("ui", style)
        self.ui_board.rtl = rtl
        self.dialogue = DialogueComponent(
            profile, lookup, glossary, cache, self.maker, style,
            mode=cover_mode if cover_mode in MODES else "grow",
            confirmation=self.confirmation_mode, spaced=spaced, rtl=rtl,
        )  # fmt: skip
        # live translation: lines that come in while the HUD runs, applied by the OCR tick
        self.provisional: set[str] = set()
        self.ui_board.provisional = self.dialogue.board.provisional = self.provisional
        self.dialogue.provisional = self.provisional
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.wake: Callable[[], None] | None = None  # the worker's: a translation came in
        self._live: Unmatched | None = None
        self._unknown: list[tuple[str, str, str | None, float, Rect]] = []  # this tick's
        # labels sent to be translated -> [(reading, frame it was read in, when)]
        self._sent: OrderedDict[str, list[tuple[OcrLine, np.ndarray, float]]] = OrderedDict()
        self.ticker = Ticker(profile, style)
        self.ticker.rtl = rtl
        self._size = (0, 0)  # of the frames: where the ticker's slots are
        self.tracker = ChangeTracker()
        self.recall = Recall()
        self.recall.passes_required = 1 if self.confirmation_mode == INSTANT else 2
        self._recall_path = cache.path.with_name("recall.npz") if cache.path else None
        if self._recall_path is not None:
            self.recall.load(self._recall_path)
        self.ocr_calls = 0
        self.ocr_ms = 0.0
        self._ui_read_at = -1e9
        self._band_read_at = -1e9
        self._moving_since: float | None = None  # since when the picture has been moving
        self._read_size: tuple[int, int] | None = None  # of the frames the ticks read
        self._deferred = False  # the last tick left a read for later (see pending())
        self._buffers: dict[str, np.ndarray] = {}  # reused images the OCR reads
        # bumped by clear() and set_content() (GUI thread): a tick that began before does
        # not place what it read (it would bring back what was just taken away)
        self._epoch = 0
        self._timings: dict[str, deque[float]] = {
            k: deque(maxlen=TIMINGS_KEPT) for k in ("dialogue", "ui", "tick")
        }
        self._timing_lock = threading.Lock()

    # ---- what to translate -----------------------------------------------------------
    @property
    def content(self) -> Content:
        return self.maker.content

    def set_content(self, dialogue: bool = True, ui: bool = True) -> None:
        """What the player wants translated (GUI thread, while it runs): covers of a kind
        switched off go at once, and the screen is read again for a kind switched on."""
        self.maker.content = Content(dialogue=dialogue, ui=ui)
        self._epoch += 1
        for b in self.boards:
            off = [c for c in b.snapshot() if not self.maker.content.allows(c.story)]
            if off:
                b.remove(off, "switched off by the player")
        if not dialogue:
            self.dialogue.end_scene()
        if not ui:
            self.ticker.clear()
        self.tracker.reset()  # everything counts as changed: read it all again

    @property
    def boards(self) -> list[Board]:
        return [self.ui_board, self.dialogue.board]

    # ---- live translation ------------------------------------------------------------
    def set_live(
        self,
        unmatched: Unmatched | None,
        on_line: Callable[[str, str, str | None], None] | None = None,
    ) -> None:
        """*unmatched(text, kind, speaker, confidence, box)* hears of every reading no
        database line matches whose text held still while it was read (worker thread); it
        returns True to have the area read again (a reading it waits to see settle).
        *on_line(source, translation, speaker)* hears of every line of dialogue shown (the
        conversation, as context for the next ones)."""
        self._live = unmatched
        hook = None if unmatched is None else self._note_unknown
        self.maker.unmatched = hook
        self.dialogue.unmatched = hook
        self.dialogue.on_line = on_line

    def _note_unknown(self, text: str, kind: str, speaker: str | None, conf: float,
                      box: Rect) -> None:  # fmt: skip
        self._unknown.append((text, kind, speaker, conf, box))  # (told at the tick's end)

    def add_translation(
        self,
        source: str,
        translation: str,
        kind: str = UI,
        speaker: str | None = None,
        box: Rect | None = None,
    ) -> None:
        """A translation made while the HUD runs (any thread), shown a little faded
        (provisional): a label at once, if the text it was read from is still there, else
        (and a line of dialogue) once its area was read again, on the next tick."""
        self._inbox.put((source, translation, kind, speaker, box))
        wake = self.wake
        if wake is not None:
            wake()

    def _take_inbox(self, frame: np.ndarray, now: float, epoch: int) -> None:
        freed: list[Rect] = []
        lookup = self.maker.lookup
        while True:
            try:
                source, translation, kind, speaker, box = self._inbox.get_nowait()
            except queue.Empty:
                break
            t = templatize(source, lookup.player_name)
            if not t.key:
                continue
            lookup.add(
                Line(
                    key=t.key, source_text=source, speaker=speaker, screen_type=kind,
                    is_template=t.is_template, translation=translation, tier="machine",
                    source="live", chapter=None, seq=None, game_version=None,
                )
            )  # fmt: skip
            self.provisional.add(translation)
            if t.is_template:  # (dialogue shows it filled in)
                self.provisional.add(fill(translation, t.values, lookup.player_name))
            readings = self._sent.pop(source, []) if kind == UI else []
            missed = self._cover_sent(readings, frame, now, epoch)
            if box is not None and (missed or not readings):
                freed.append(box)
        if freed:
            self.tracker.free(freed)

    def _cover_sent(self, readings: list[tuple[OcrLine, np.ndarray, float]],
                    frame: np.ndarray, now: float, epoch: int) -> bool:  # fmt: skip
        """Cover the labels whose translation just came, from the frames they were read in,
        where their text is still on screen. Returns whether one could not be (its area is
        read again)."""
        missed = False
        for line, read, _at in readings:
            if read.shape != frame.shape or epoch != self._epoch:
                return True
            fh, fw = frame.shape[:2]
            stay_out = self.dialogue.territory_px(fw, fh) if self.content.dialogue else []
            stay_out += self.ui_board.boxes() + self.dialogue.board.boxes() + self.ticker.boxes()
            places = self.maker.placements(read, [line], stay_out)
            placed = self.maker.commit(places, read, frame, self.ui_board, now,
                                       others=[self.dialogue.board])  # fmt: skip
            missed = missed or not placed
        return missed

    def _tell_unknown(self, frame: np.ndarray, last: np.ndarray, lines: list[OcrLine],
                      now: float) -> None:  # fmt: skip
        """Tell live translation of the readings no database line matched whose text held
        still while it was read (the others are read again, as is what it waits for); keep
        the labels with their frame, to cover them the moment their translation comes."""
        live, unknown, self._unknown = self._live, self._unknown, []
        if live is None or not unknown:
            return
        fh, fw = frame.shape[:2]
        again: list[Rect] = []
        for text, kind, speaker, conf, box in unknown:
            if last is not frame and not _held_still(frame, last, box):
                again.append(box)
                continue
            if kind == UI:
                found = [ln for ln in lines if ln.text == text and box_px(ln, fw, fh) == box]
                if found:
                    self._sent[text] = [(ln, frame, now) for ln in found]
                    self._sent.move_to_end(text)
            if live(text, kind, speaker, conf, box):
                again.append(box)
        while len(self._sent) > LIVE_KEEP:
            self._sent.popitem(last=False)
        for text in [t for t, got in self._sent.items() if now - got[0][2] > LIVE_KEEP_S]:
            del self._sent[text]
        if again:
            self.tracker.free(again)

    # ---- the OCR tick (worker thread) ------------------------------------------------
    def tick(self, frame: np.ndarray, newest: Callable[[], np.ndarray], now: float) -> TickReport:
        """Read what changed in *frame* and place what was matched. *newest* returns the
        most recent captured frame at the moment covers are placed."""
        rep = TickReport()
        fh, fw = frame.shape[:2]
        if self._read_size not in (None, (fw, fh)):
            # another picture size (the window resized, black bars came or went): what was
            # placed belongs to the old picture; this one is read from scratch
            log.info("frame size %s -> %s: covers cleared", self._read_size, (fw, fh))
            self.clear()
        epoch = self._epoch
        self._read_size = self._size = (fw, fh)
        self._unknown = []
        self._take_inbox(frame, now, epoch)
        plan = self._plan(frame, now, fw, fh)
        t0 = time.perf_counter()
        before = self.maker.unsettled
        dlg_lines: list[OcrLine] = []
        if plan.dialogue and plan.ui is None and self.dialogue.scene is None:
            # no conversation and everything due: the whole frame in one read, the dialogue
            # territory's share handed to the dialogue
            got = self._read_whole(frame, newest, plan, now, rep, epoch)
            if got is None:  # the picture changed while it was read
                return rep
            ui_lines, dlg_lines = got
        else:
            if plan.dialogue:
                got_dlg = self._read_dialogue(frame, newest, now, rep, epoch)
                if got_dlg is None:
                    return rep
                dlg_lines = got_dlg
            ui_lines = self._read_ui(frame, plan, now, rep) if plan.ui is None or plan.ui else []
        rep.ocr_ms = (time.perf_counter() - t0) * 1000.0
        self._record_timing("tick", rep.ocr_ms)
        rep.readings = ui_lines + dlg_lines
        if not rep.ui and not rep.dialogue:
            self._remember(now, fw, fh)
            return rep
        last = self._place(frame, newest, ui_lines, plan, now, rep, epoch)
        if last is None:
            return rep
        rep.unsettled = self.maker.unsettled - before
        self._mark(frame, plan, rep)
        self._tell_unknown(frame, last, ui_lines, now)
        return rep

    def _plan(self, frame: np.ndarray, now: float, fw: int, fh: int) -> ReadPlan:
        """What to read this tick: what changed, what the watch freed, what is due."""
        territory = self.dialogue.territory_px(fw, fh) if self.content.dialogue else []
        # a ticker's slot with its line shown is left alone (its text slides all the time)
        # but for a look every few seconds
        quiet = self.ticker.quiet(fw, fh, now)
        changed = self.tracker.changed(frame)
        # a picture at rest (a menu that stopped scrolling) is read at once: the waits below
        # spare the game's GPU while it moves, when a read is mostly of text in motion
        rest = self.tracker.at_rest(frame)
        if rest and self._moving_since is not None:
            log.debug("picture at rest after %.2f s of motion", now - self._moving_since)
        if rest:
            self._moving_since = None
        elif self._moving_since is None:
            self._moving_since = now
        freed = self.tracker.take_freed()
        tmask = _mask(changed.shape, territory, fw, fh)
        in_band = [r for r in freed if any(intersects(r, t) for t in territory)]
        out_band = [r for r in freed if not any(_inside(r, t) for t in territory)]
        dialogue = self.content.dialogue and (bool((changed & tmask).any()) or bool(in_band))
        ui_rows = (changed & ~tmask & ~_mask(changed.shape, quiet, fw, fh)).any(axis=1)
        ui = bands(ui_rows, out_band + self.ticker.due(fw, fh, now), fw, fh)
        self._deferred = False
        scene = self.dialogue.scene
        menu = (
            scene is None
            and not self.dialogue.divider_seen(now)
            and len(self.ui_board) >= MENU_COVERS
        )
        since_ui = now - self._ui_read_at
        if (ui is None or ui) and not rest:
            later = (
                (scene is not None or not self.content.ui) and since_ui < SCENE_UI_EVERY_S
            ) or (ui is None and scene is None and not menu and since_ui < EXPLORE_FULL_EVERY_S)
            if later:  # (its rows stay marked as changed; what the watch freed waits too)
                self.tracker.free(out_band)
                ui, self._deferred = [], True
        if dialogue and menu and not rest and now - self._band_read_at < MENU_BAND_EVERY_S:
            self.tracker.free(in_band)
            dialogue, self._deferred = False, True
        return ReadPlan(territory, quiet, dialogue, ui)

    def _read_whole(
        self,
        frame: np.ndarray,
        newest: Callable[[], np.ndarray],
        plan: ReadPlan,
        now: float,
        rep: TickReport,
        epoch: int,
    ) -> tuple[list[OcrLine], list[OcrLine]] | None:
        """One read of the whole frame, what is covered blanked out: (the rest's readings,
        the dialogue territory's, which the dialogue took). None as _read_dialogue."""
        t0 = time.perf_counter()
        known = _known(self.ui_board) + _known(self.dialogue.board)
        src = self._image("ui", frame, None, [], known, plan.quiet)
        lines = self._read(src, None)
        rep.ui = "full"
        self._ui_read_at = now + (time.perf_counter() - t0)  # when the read was done
        rep.ui_ms = (time.perf_counter() - t0) * 1000.0
        self._record_timing("ui", rep.ui_ms)
        dlg = self._read_dialogue(frame, newest, now, rep, epoch, lines)
        if dlg is None:
            return None
        return [ln for ln in lines if not self.dialogue.owns(ln)], dlg

    def _read_dialogue(
        self,
        frame: np.ndarray,
        newest: Callable[[], np.ndarray],
        now: float,
        rep: TickReport,
        epoch: int,
        whole: list[OcrLine] | None = None,
    ) -> list[OcrLine] | None:
        """Read and commit the dialogue territory, before any slower read of the rest, or
        take its share of *whole*, a read of the whole frame (a line of dialogue found there
        is read again at full size: its rows' places set the layout; labels are not). The
        line shows with the database's speaker; the name is read on its own afterwards.
        None when the picture changed size or was cleared meanwhile."""
        crop = self.dialogue.crop()
        if crop is None:
            return []
        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]
        read = lines = None
        told = len(self._unknown)
        if whole is not None:
            read = self.dialogue.read(frame, whole, self.ui_board.boxes())
            lines = [ln for ln in whole if self.dialogue.owns(ln)]
        if read is None or read.rows or read.hit is not None:
            del self._unknown[told:]  # (what the sharper read finds is told instead)
            known = _known(self.dialogue.board)
            src = self._image("dialogue", frame, [_px(crop, fw, fh)], [], known) if known else frame
            lines = self._read(src, crop, fine=True)
            read = self.dialogue.read(frame, lines, self.ui_board.boxes())
        rep.dialogue = True
        self._band_read_at = now
        shown = newest()
        if shown is None:
            shown = frame
        if shown.shape != frame.shape or epoch != self._epoch:
            self.tracker.reset()
            return None
        self.dialogue.commit(read, frame, shown, now, self.ui_board)
        rep.dialogue_ms = (time.perf_counter() - t0) * 1000.0
        self._record_timing("dialogue", rep.dialogue_ms)
        if self.dialogue.needs_name_check(read):
            verified = self.dialogue.refine_speaker(
                frame, read, lambda r: self._read(frame, r, fine=True)
            )
            if verified is not None:
                named = DialogueRead(speaker=verified)
                self.dialogue.commit(named, frame, shown, now, self.ui_board)
        return lines

    def _read_ui(
        self, frame: np.ndarray, plan: ReadPlan, now: float, rep: TickReport
    ) -> list[OcrLine]:
        """Read the rest of the screen: the bands of the plan, or all of it. What is covered
        is known (the watch vouches for it): only new text is read; a cover recalled from
        memory is read once, to be confirmed or overruled."""
        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]
        src = self._image("ui", frame, plan.ui, plan.territory, _known(self.ui_board), plan.quiet)
        if plan.ui is None:
            lines = self._read(src, None)
            rep.ui = "full"
        else:
            lines = []
            for x, y, w, h in plan.ui:
                lines.extend(self._read(src, Region(x / fw, y / fh, w / fw, h / fh)))
            rep.ui = f"{len(plan.ui)} bands"
        lines = [ln for ln in lines if not self.dialogue.owns(ln)]
        self._ui_read_at = now + (time.perf_counter() - t0)  # when the read was done
        rep.ui_ms = (time.perf_counter() - t0) * 1000.0
        self._record_timing("ui", rep.ui_ms)
        return lines

    def _place(
        self,
        frame: np.ndarray,
        newest: Callable[[], np.ndarray],
        ui_lines: list[OcrLine],
        plan: ReadPlan,
        now: float,
        rep: TickReport,
        epoch: int,
    ) -> np.ndarray | None:
        """Cover what the UI read matched, placed against the newest frame (returned): only
        text that held still while it was read. None when the picture changed size, or the
        player cleared or switched what is shown meanwhile (nothing placed: the next tick
        reads again)."""
        fh, fw = frame.shape[:2]
        stay_out = plan.territory + self.ui_board.boxes() + self.dialogue.board.boxes()
        stay_out += self.ticker.boxes()
        places = self.maker.placements(frame, ui_lines, stay_out) if ui_lines else []
        slotted = [p for p in places if self.ticker.slot_of(p.geom.cover, fw, fh)]
        places = [p for p in places if p not in slotted]
        last = newest()
        if last is None:
            last = frame
        if last.shape != frame.shape or epoch != self._epoch:
            self.tracker.reset()
            return None
        if rep.ui and self.content.ui:
            looked = [r for _n, r in self.ticker.slots(fw, fh) if r not in plan.quiet]
            if plan.ui is not None:
                looked = [r for r in looked if any(intersects(r, b) for b in plan.ui)]
            self.ticker.offer(slotted, frame, last, now, looked)
        rep.placed = self.maker.commit(
            places, frame, last, self.ui_board, now, others=[self.dialogue.board]
        )
        # text that was still moving (fading in, sliding) is read again on the next tick:
        # its area counts as changed even if the change map did not notice the last step
        self.tracker.free(self.maker.take_unsettled())
        self._remember(now, fw, fh)
        return last

    def _mark(self, frame: np.ndarray, plan: ReadPlan, rep: TickReport) -> None:
        """Remember what this tick read (its pixels are no longer changed)."""
        keep = ([] if rep.dialogue else plan.territory) + plan.quiet
        self.tracker.mark_read(frame, plan.ui, keep=keep)
        if plan.ui is not None and rep.dialogue:
            self.tracker.mark_read(frame, plan.territory)

    def _image(
        self,
        name: str,
        frame: np.ndarray,
        areas: list[Rect] | None,
        dark: list[Rect],
        covered: list[Cover],
        blank: list[Rect] = (),
    ) -> np.ndarray:
        """The picture the OCR reads: *frame*'s pixels in *areas* (None: everywhere) in a
        buffer kept for the next tick, with *dark* areas blacked out and the covered text
        and *blank* areas painted in their surroundings' colour. Only the areas read are
        copied."""
        buf = self._buffers.get(name)
        if buf is None or buf.shape != frame.shape:
            buf = self._buffers[name] = np.empty_like(frame)
        if areas is None:
            np.copyto(buf, frame)
        else:
            for x, y, w, h in areas:
                buf[y : y + h, x : x + w] = frame[y : y + h, x : x + w]
        for x, y, w, h in dark:
            buf[max(0, y) : y + h, max(0, x) : x + w] = 0
        for c in covered:
            _blank(buf, frame, c.geom.cover)
        for r in blank:
            _blank(buf, frame, r)
        return buf

    def _remember(self, now: float, fw: int, fh: int) -> None:
        for b in self.boards:
            self.maker.remember(b, now, fw, fh)
        self.cache.save_soon()

    def _read(self, frame: np.ndarray, region: Region | None, fine: bool = False) -> list[OcrLine]:
        """*fine*: the dialogue and the names, detected at full resolution (their rows'
        places set the layout); everything else shrunk (capture.ocr.SCREEN_DETECT_HEIGHT)."""
        res = self.engine.read(frame, region=region, fine=fine)
        self.ocr_calls += 1
        self.ocr_ms += res.elapsed_ms
        return res.lines

    def _record_timing(self, kind: str, elapsed_ms: float) -> None:
        with self._timing_lock:
            self._timings[kind].append(elapsed_ms)

    def diagnostics(self) -> dict[str, object]:
        """Counters for the live diagnostics panel (safe enough from any thread)."""
        with self._timing_lock:
            timings = {k: _summary(list(v)) for k, v in self._timings.items()}
        return {
            "confirmation": self.confirmation_mode,
            "ocr_calls": self.ocr_calls,
            "ocr_total_ms": round(self.ocr_ms, 1),
            "timings": timings,
            "lookup": self.maker.lookup.diagnostics(),
            "cover_cache": self.cache.diagnostics(),
            "recall": self.recall.diagnostics(),
            "unsettled": self.maker.unsettled,
            "frame": list(self._size),
        }

    def pending(self) -> bool:
        """Is there work for another tick even without a new frame (a still screen: a
        paused game, an idle menu)? A live translation came in, the watch freed an area,
        or the last tick left a read for later."""
        return not self._inbox.empty() or self.tracker.has_freed() or self._deferred

    # ---- the pixel watch (GUI thread, every 50 ms) ---------------------------------
    def watch(self, frame: np.ndarray, now: float) -> None:
        freed: list[Rect] = []
        for b in self.boards:
            freed.extend(b.watch(frame))
        if self.content.dialogue:
            freed.extend(self.dialogue.watch(frame, now))
        self.tracker.free(freed)
        self.ticker.watch(frame, now)
        self.recall.learn(self.boards, frame)
        fh, fw = frame.shape[:2]
        self._size = (fw, fh)
        if self.content.ui:  # (memory holds menus only)
            keep_out = [r for _n, r in self.ticker.slots(fw, fh)]
            keep_out += self.dialogue.glass_areas(fw, fh)
            self.recall.bring_back(frame, now, self.boards, keep_out)

    def items(self) -> list[OverlayItem]:
        out = (
            self.ui_board.items()
            + self.dialogue.board.items()
            + self.dialogue.items()
            + self.ticker.items(*self._size)
        )
        if self.plain_text:
            out = [replace(it, plain=True) if it.style in ("label", "text") else it
                   for it in out]
        return out

    def version(self) -> tuple[int, int, int, int]:
        return (
            self.ui_board.version,
            self.dialogue.board.version,
            self.dialogue.version,
            self.ticker.version,
        )

    def clear(self) -> None:
        self._epoch += 1
        for b in self.boards:
            b.clear()
        self.dialogue.end_scene()
        self.ticker.clear()
        self.tracker.reset()

    def save_memory(self) -> None:
        """Keep what was learned this session for the next one: the menu covers and their
        blocks (at stop)."""
        if self._recall_path is not None:
            self.recall.save(self._recall_path)
        self.cache.save()


def _known(board: Board) -> list[Cover]:
    """The covers whose text the OCR need not read again: all but the ones memory brought
    back (read once, to be confirmed) and those with numbers in them (they change)."""
    return [c for c in board.snapshot() if not (c.recalled or NUMBER in c.template)]


def _px(region: Region, fw: int, fh: int) -> Rect:
    x0, y0, x1, y1 = region.to_pixels(fw, fh)
    return x0, y0, x1 - x0, y1 - y0


def _mask(shape: tuple[int, ...], rects: list[Rect], fw: int, fh: int) -> np.ndarray:
    """The change map's pixels covered by frame *rects*."""
    out = np.zeros(shape, dtype=bool)
    sy, sx = shape[0] / fh, shape[1] / fw
    for x, y, w, h in rects:
        out[int(y * sy) : int(np.ceil((y + h) * sy)), int(x * sx) : int(np.ceil((x + w) * sx))] = (
            True
        )
    return out


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"latest": 0.0, "median": 0.0, "p95": 0.0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "latest": round(float(a[-1]), 1),
        "median": round(float(np.median(a)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
    }


def _blank(dst: np.ndarray, src: np.ndarray, rect: Rect) -> None:
    """Paint *rect* (grown by a pixel or two) of *dst* in the colour around it in *src*,
    so the OCR finds no text there."""
    fh, fw = src.shape[:2]
    x, y, w, h = rect
    x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(fw, x + w + 2), min(fh, y + h + 2)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return
    ring = np.concatenate(
        [src[y0, x0:x1], src[y1 - 1, x0:x1], src[y0:y1, x0], src[y0:y1, x1 - 1]], axis=0
    )
    dst[y0:y1, x0:x1] = np.median(ring, axis=0).astype(dst.dtype)


class Rereader:
    """Lines read again by another reader's recognizer, in their own boxes, in one batch.
    The same misreading in a box of the same size gets the same second reading (a label
    read again on every tick costs it once)."""

    def __init__(self, engine: OcrEngine) -> None:
        self.engine = engine
        self._known: OrderedDict[tuple, tuple[str, float] | None] = OrderedDict()

    def __call__(self, frame: np.ndarray, lines: list[OcrLine]) -> list[OcrLine | None]:
        fh, fw = frame.shape[:2]
        keys = [(ln.text, round(ln.box.w * fw / 4), round(ln.box.h * fh / 4)) for ln in lines]
        new = [i for i, k in enumerate(keys) if k not in self._known]
        if new:
            got = recognize_boxes(self.engine, frame, [lines[i].box for i in new])
            for i, g in zip(new, got, strict=True):
                self._known[keys[i]] = g
        while len(self._known) > REREAD_KEEP:
            self._known.popitem(last=False)
        out: list[OcrLine | None] = []
        for ln, k in zip(lines, keys, strict=True):
            g = self._known.get(k)
            if g is None:
                out.append(None)
                continue
            chars = None  # (the first reader's character cells, when they line up)
            if ln.chars is not None and len(ln.chars) == len(g[0]):
                chars = [OcrChar(c, a.x0, a.x1) for c, a in zip(g[0], ln.chars, strict=True)]
            out.append(OcrLine(text=g[0], box=ln.box, conf=g[1], chars=chars))
        return out


def _held_still(read: np.ndarray, newest: np.ndarray, rect: Rect) -> bool:
    """Did the text in *rect* hold still between the frame read and the newest one (the
    test a cover is placed by: :func:`hud.anchors.settled`)? An area too bare to tell is
    compared pixel for pixel."""
    ref = make_anchor(read, rect)
    if ref is not None:
        return settled(newest, ref)
    x, y, w, h = rect
    a, b = read[max(0, y) : y + h, max(0, x) : x + w], newest[max(0, y) : y + h, max(0, x) : x + w]
    return a.shape == b.shape and np.array_equal(a, b)


def _inside(r: Rect, t: Rect) -> bool:
    cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
    return t[0] <= cx <= t[0] + t[2] and t[1] <= cy <= t[1] + t[3]
