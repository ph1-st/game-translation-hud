"""Translations that fit: a paragraph the OCR reads row by row matched as one text, longer
text (Spanish) kept inside its box, languages without spaces and written right to left,
and speaker names that are characters' names only."""

from __future__ import annotations

import pytest

from capture.profile import load_profile
from db.lookup import Lookup
from db.store import Line
from db.templates import templatize
from hud.covercache import CoverCache
from hud.dialogue import Scene, Speaker
from hud.dialogue.names import Names
from hud.dialogue.scene import lay_out, text_area
from hud.items import OverlayItem, painted_align
from hud.layout import PAD_X, drawn_size, layout_block
from hud.pipeline import HudCore
from hud.style import get_style
from hud.textwidth import text_width, wrap
from tools.replay import RecordedEngine, load_fixture

GAME = "azur-promilia"
PROFILE = (
    "生い立ちが謎に包まれた少女。すこし鈍感なところがあるが、非常に強い心を持ち、信念を曲げる"
    "ことは決してない。記憶を失っているにもかかわらず、世界を探索しようと旅に出て、冒険を"
    "通じて自分の生き方や存在意義を見いだそうとしている。そんな彼女の周りには、気づけば頼もし"
    "い仲間たちが集っていた。"
)  # one text in the game's files; the screen shows it in eight rows
PROFILE_ES = (
    "Una joven de origen misterioso. Algo despistada, pero con una voluntad de hierro: nunca "
    "renuncia a sus convicciones. Aunque perdió la memoria, emprendió un viaje para explorar "
    "el mundo y descubrir, a través de la aventura, su forma de vivir y el sentido de su "
    "existencia. Sin darse cuenta, se ha rodeado de compañeros en los que puede confiar."
)


def line(source: str, translation: str, speaker: str | None = None, kind: str = "ui") -> Line:
    t = templatize(source)
    return Line(t.key, source, speaker, kind, False, translation, "human", "test", None, 1, None)


@pytest.mark.footage
def test_a_paragraph_read_row_by_row_is_matched_as_one_text(glossary):
    frame, meta = load_fixture(GAME, "004711_paragraph")
    core = HudCore(load_profile(GAME), Lookup([line(PROFILE, PROFILE_ES)]), glossary,
                   CoverCache(None), RecordedEngine(meta["ocr"]), get_style("glow"))  # fmt: skip
    core.tick(frame, lambda: frame, 0.0)
    (block,) = [it for it in core.items() if it.style == "label"]
    fh = frame.shape[0]
    assert " ".join(block.lines) == PROFILE_ES  # the whole text, once
    assert 2 <= len(block.lines) <= 8  # in the rows the Japanese has, never more
    assert block.row_top <= 0.54 * fh and block.row_top + 7 * block.pitch >= 0.70 * fh


def test_a_block_never_gets_more_lines_than_rows():
    cover = (100, 100, 220, 44)
    walls = [(90, 100, 8, 44), (324, 100, 8, 44)]  # it cannot widen
    g = layout_block(cover, PROFILE_ES, 19, 2, 22.0, "left", walls, 1600, 102, 14)
    assert len(g.lines) <= 2
    size = drawn_size(g.font_px, g.stretch)
    assert all(text_width(t, size) <= g.box[2] - 2 * PAD_X + 0.5 for t in g.lines)


def test_languages_without_spaces_break_between_characters():
    text = "这是一个很长的中文句子，用来测试换行是否正确。"
    lines = wrap(text, 20, 160, spaced=False)
    assert len(lines) > 1 and "".join(lines) == text
    assert not any(t.startswith(("，", "。")) for t in lines)  # never before a closing mark
    assert wrap("Two words", 20, 10) == ["Two", "words"]  # (spaced languages: at spaces)


def test_right_to_left_text_starts_at_the_right():
    item = OverlayItem((0, 0, 100, 20), lines=["שלום"], align="left", rtl=True)
    assert painted_align(item) == "right"
    item.rtl = False
    assert painted_align(item) == "left"


def test_a_right_to_left_target_draws_every_item_right_to_left(lookup, glossary):
    frame, meta = load_fixture(GAME, "004720_list")
    core = HudCore(load_profile(GAME), lookup, glossary, CoverCache(None),
                   RecordedEngine(meta["ocr"]), get_style("glow"), target_lang="he")  # fmt: skip
    core.tick(frame, lambda: frame, 0.0)
    labels = [it for it in core.items() if it.style == "label"]
    assert labels and all(it.rtl for it in labels)


# ---- dialogue ------------------------------------------------------------------------------
def test_long_dialogue_stays_below_the_speaker_name(run):
    core, frame = run("004653_walk")  # NPC speech: the name above the line
    sc = core.dialogue.scene
    fh, fw = frame.shape[:2]
    lay_out(sc, PROFILE_ES, fw, fh)  # far longer than any line of the layout
    x, y, w, h = text_area(sc, fw, fh)
    top = sc.text_box[1]
    assert abs(top - sc.text_y) <= 1  # it starts on the first row, never higher
    assert top + (len(sc.lines) - 1) * sc.text_pitch + sc.text_band <= y + h + 1  # in the area
    size = drawn_size(sc.text_font, sc.text_stretch)
    assert all(text_width(t, size) <= w for t in sc.lines)


def test_a_menu_label_in_a_name_band_is_no_name(glossary):
    lk = Lookup([
        line("決定", "OK"),
        line("見てくれ", "Look at this.", speaker="アダイヤ", kind="dialogue"),
        line("アダイヤ", "Adaiya"),
    ])  # fmt: skip
    names = Names(load_profile(GAME), glossary, lk, CoverCache(None))
    assert names.name_of("決定") is None  # a short line, but no one's name
    assert names.name_of("アダイヤ") == "Adaiya"  # a speaker the database's lines name
    assert names.name_of("テララ") == "Terara"  # a character of the glossary
    assert names.name_of("シャルル村") is None  # a place of the glossary is no name


def test_only_names_read_on_screen_fix_the_name_box(lookup, glossary):
    profile = load_profile(GAME)
    cache = CoverCache(None)
    names = Names(profile, glossary, lookup, cache)
    lay = profile.dialogue_layouts["npc"]
    sc = Scene(lay, (0, 0, 100, 50), 20, 18.0)
    hint = Speaker(lay, "Terara", (700, 620, 60, 18), (220, 220, 255), None, verified=False)
    names.apply(sc, hint, 1600, 900)  # the database's speaker, shown with its line
    assert sc.name == "Terara" and sc.name_box is not None
    assert "name" not in (cache.dialogue_font("npc") or {})  # nothing fixed by it
    assert cache.name_colours("Terara") is None
    read = Speaker(lay, "Terara", (702, 622, 58, 18), (120, 140, 250), None)
    names.apply(sc, read, 1600, 900)  # the name as read on screen
    assert "name" in cache.dialogue_font("npc")
    assert cache.name_colours("Terara") is not None
