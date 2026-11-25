"""Speaker names: which readings are names, their translation, and the fixed box they are
shown in.

A name is a character's: a glossary entry of type ``character``, a speaker the database's
lines name, or the player's own name in roman letters (shown as it is). Anything else in a
name band (a menu's 決定 or 確認) is a label, never a name.

The name box of a layout is fixed by the first name read there: as tall as that name's
glyphs (plus the cover margin), as wide as the longest known name, centred on the speaker
band, a size up from the layout's dialogue font. A name's colours are the first ones read
for it. Both are kept in the cover cache, from names read on screen only: the database's
speaker (shown with its line before the name is read) never fixes a box or a colour.
"""

from __future__ import annotations

import logging
import re

import numpy as np
from rapidfuzz import fuzz, process

from capture.ocr import OcrLine
from capture.profile import DialogueLayout, Profile
from db.glossary import Glossary
from db.lookup import Lookup
from db.normalize import normalize_key
from db.text import cjk_count
from hud.anchors import HARD_FAIL, HOLDS, Anchor, anchor_state, make_anchor, recognizes, settled
from hud.board import DOUBT_FRAMES
from hud.covercache import CoverCache
from hud.covers import px
from hud.dialogue.scene import Scene, Speaker
from hud.extent import PAD, measure
from hud.layout import PAD_X
from hud.looks import is_light
from hud.textwidth import text_width

log = logging.getLogger("hud")

# a name in Latin letters, accented ones included (José, Zoë, Łukasz): Latin-1 and the
# Latin Extended-A and -B blocks, without the signs × and ÷ that sit among them
_LATIN = "A-Za-zÀ-ÖØ-öø-ɏ"
ROMAN = re.compile(rf"[{_LATIN}][{_LATIN}0-9 ._'\-]{{1,23}}")
NAME_CONF = 0.5  # readings of names below this confidence are not trusted
NAME_FUZZY = 80  # a reading this similar to a character's name is that name
SPEAKER_STEP = 1.2  # the name's font, relative to the layout's dialogue font
JP_ADVANCE = 1.05  # a Japanese character is about this many glyph heights wide
DEFAULT_NAME_CHARS = 4  # a name box's width in Japanese characters when no name is known
MARGIN = PAD + 1  # pixels around a name's glyphs that its box and anchor take in


def is_roman(text: str) -> bool:
    """Is *text* a name in roman letters (the player's), that needs no translation?"""
    t = text.strip()
    return bool(ROMAN.fullmatch(t)) and sum(c.isalpha() for c in t) >= 2


class Names:
    """The names a conversation can show, and the boxes they are shown in."""

    def __init__(
        self, profile: Profile, glossary: Glossary, lookup: Lookup, cache: CoverCache
    ) -> None:
        self.profile = profile
        self.lookup = lookup
        self.cache = cache
        # (only entries typed as characters: a term is never a speaker's name)
        characters = [e for e in glossary.entries if e.type == "character"]
        self._characters = {normalize_key(e.source): e.translation for e in characters}
        self._character_keys = list(self._characters)
        self._speakers = {normalize_key(ln.speaker) for ln in lookup.by_key.values()
                          if ln.speaker}  # fmt: skip
        self._widest_en: list[str] | None = None  # (worked out the first time it is needed)
        self.longest_jp = max((len(e.source) for e in characters), default=DEFAULT_NAME_CHARS)
        # layout -> name -> (speaker, its strokes): names read before come back by them
        self.known: dict[str, dict[str, tuple[Speaker, Anchor]]] = {}

    # ---- which text is a name ------------------------------------------------------------
    def name_of(self, text: str) -> str | None:
        """The translation of a speaker name, or None when *text* is no character's."""
        key = normalize_key(text)
        if not key:
            return None
        if key in self._characters:
            return self._characters[key]
        if is_roman(text):
            return text.strip().rstrip(".,:;!?")  # (a dot the OCR saw after it is no part)
        if key in self._speakers:
            line = self.lookup.by_key.get(key)
            if line is not None and line.translation:
                return line.translation
        if len(key) >= 2 and self._character_keys:
            best = process.extractOne(key, self._character_keys, scorer=fuzz.ratio,
                                      score_cutoff=NAME_FUZZY)  # fmt: skip
            if best is not None:
                return self._characters[best[0]]
        return None

    def is_name(self, ln: OcrLine) -> bool:
        """Is *ln* a character's name read in a layout's name band?"""
        if ln.conf < NAME_CONF or (cjk_count(ln.text) < 1 and not is_roman(ln.text)):
            return False
        cx, cy = ln.box.x + ln.box.w / 2, ln.box.y + ln.box.h / 2
        in_band = any(
            lay.speaker is not None and lay.speaker.contains_point(cx, cy)
            for lay in self.profile.dialogue_layouts.values()
        )
        return in_band and self.name_of(ln.text) is not None

    def find(self, frame: np.ndarray, lay: DialogueLayout, lines: list[OcrLine]) -> Speaker | None:
        """The speaker name among *lines* read in *lay*'s name band."""
        sp = lay.speaker
        if sp is None:
            return None
        for ln in sorted(lines, key=lambda ln: -ln.conf):
            cx, cy = ln.box.x + ln.box.w / 2, ln.box.y + ln.box.h / 2
            if not sp.contains_point(cx, cy) or ln.conf < NAME_CONF:
                continue
            name = self.name_of(ln.text)
            ext = measure(frame, ln) if name is not None else None
            if name is not None and ext is not None:
                return Speaker(lay, name, ext.ink, ext.fill, ext.edge)
        return None

    def widest(self, font_px: int) -> float:
        """The width of the longest known name at *font_px*."""
        if self._widest_en is None:
            names = set(self._characters.values())
            for key in self._speakers:
                line = self.lookup.by_key.get(key)
                if line is not None and line.translation:
                    names.add(line.translation)
            self._widest_en = sorted(names)
        return max((text_width(n, font_px) for n in self._widest_en), default=0.0)

    # ---- the name box ----------------------------------------------------------------------
    def set(
        self,
        sc: Scene,
        spk: Speaker,
        fw: int,
        fh: int,
        frame: np.ndarray | None = None,
        read: np.ndarray | None = None,
    ) -> bool:
        """Show *spk* in the scene's name box. Its strokes in *frame* (the newest) are
        watched (another name takes their place) and kept: the name comes back by them. A
        name whose strokes changed since the frame it was *read* from is not set (the
        speaker changed while the OCR ran: the next read tells the new one)."""
        ink = _grown(spk.ink)
        light = is_light(spk.fill)
        if spk.verified and frame is not None and read is not None and read is not frame:
            ref = make_anchor(read, ink, light)
            if ref is not None and not settled(frame, ref):
                return False
        self.apply(sc, spk, fw, fh)
        if spk.verified and frame is not None:
            anchor = make_anchor(frame, ink, light)
            sc.name_anchor, sc.name_doubt = anchor, 0
            if anchor is not None:
                self.known.setdefault(sc.layout.name, {})[spk.name] = (spk, anchor)
        log.info("dialogue name (%s): %s", sc.layout.name, spk.name)
        return True

    def apply(self, sc: Scene, spk: Speaker, fw: int, fh: int) -> None:
        """Put *spk*'s name in the scene's name box (a verified name fixes the box and its
        colours for good; the database's speaker borrows them)."""
        rec = (self.cache.dialogue_font(sc.layout.name) or {}).get("name")
        if rec is None:
            rec = {
                "font": round(SPEAKER_STEP * sc.font_px) / fh,
                "top": (spk.ink[1] - PAD) / fh,
                "height": (spk.ink[3] + 2 * PAD) / fh,
            }
            if spk.verified:
                rec = self.cache.set_dialogue_name(sc.layout.name, rec)
        self.fix_box(sc, rec, fw, fh)
        sc.name = spk.name
        if spk.verified:
            fill, edge = self.cache.palette.snap(spk.fill, spk.edge)
            colours = self.cache.name_colours(spk.name, [list(fill), _listed(edge)])
        else:
            colours = self.cache.name_colours(spk.name) or [list(spk.fill), _listed(spk.edge)]
        sc.name_fill = tuple(colours[0])
        sc.name_edge = tuple(colours[1]) if colours[1] is not None else None
        sc.name_ink = _grown(spk.ink)
        if sc.name_glass is not None:
            sc.name_glass = _widened(sc.name_glass, sc.name_ink)

    def fix_box(self, sc: Scene, rec: dict, fw: int, fh: int) -> None:
        """The name box of the scene's layout, from its cached record."""
        if sc.layout.speaker is None:
            return
        font = max(6, int(round(rec["font"] * fh)))
        sx, _sy, sw, _sh = px(sc.layout.speaker, fw, fh)
        top, h = int(round(rec["top"] * fh)), int(round(rec["height"] * fh))
        glyph = h - 2 * PAD
        # on glass the box hides the Japanese name the moment the divider is up, before the
        # name is read: as wide as the longest known one, in either language
        widest = max(self.widest(font), self.longest_jp * JP_ADVANCE * glyph)
        w = int(np.ceil(min(sw, widest + 2 * PAD_X + 2)))
        cx = sx + sw / 2.0
        sc.name_box = (int(round(cx - w / 2)), top, w, h)
        sc.name_font, sc.name_glyph = font, glyph
        if sc.name_glass is None:
            sc.name_glass = sc.name_box

    # ---- the watch -----------------------------------------------------------------------
    def watch(self, frame: np.ndarray, sc: Scene) -> bool:
        """The name goes the moment the Japanese name's strokes change (another speaker:
        under frosted glass it would not be seen); a name read before in this layout comes
        back by its strokes. Returns whether the name changed."""
        changed = False
        if sc.name is not None and sc.name_anchor is not None:
            state = anchor_state(frame, sc.name_anchor)
            sc.name_doubt = 0 if state == HOLDS else sc.name_doubt + 1
            if state == HARD_FAIL or sc.name_doubt >= DOUBT_FRAMES:
                log.info("dialogue name gone: %s", sc.name)
                sc.name, sc.name_anchor = None, None
                changed = True
        if sc.name is None:
            fh, fw = frame.shape[:2]
            for name, (spk, anchor) in self.known.get(sc.layout.name, {}).items():
                if recognizes(frame, anchor):
                    self.apply(sc, spk, fw, fh)
                    sc.name_anchor, sc.name_doubt = anchor, 0
                    log.info("dialogue name (%s): %s, seen before", sc.layout.name, name)
                    changed = True
                    break
        return changed


def _listed(colour: tuple[int, int, int] | None) -> list[int] | None:
    return list(colour) if colour is not None else None


def _grown(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    return (x - MARGIN, y - MARGIN, w + 2 * MARGIN, h + 2 * MARGIN)


def _widened(
    base: tuple[int, int, int, int], rect: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """*base* widened alike on both sides (it is centred) to reach across *rect*."""
    bx, by, bw, bh = base
    cx = bx + bw / 2.0
    half = max(bw / 2.0, cx - rect[0], rect[0] + rect[2] - cx)
    return (int(round(cx - half)), by, int(round(2 * half)), bh)
