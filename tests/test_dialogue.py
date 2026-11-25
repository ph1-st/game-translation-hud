"""The dialogue band's decisions that need no frame."""

from __future__ import annotations

from capture.ocr import OcrLine
from capture.profile import Region
from db.lookup import Lookup
from db.store import Line
from db.templates import templatize
from hud.covercache import CoverCache
from hud.covers import CoverMaker
from hud.dialogue import DialogueComponent, is_dialogue, reads_as
from hud.dialogue.reader import layout_at, match_block
from hud.matching import match_reading
from hud.style import get_style


def line(jp: str, en: str, speaker: str | None = None) -> Line:
    t = templatize(jp)
    return Line(t.key, jp, speaker, "dialogue", False, en, "human", "fixture", None, 1, None)


def row(text: str, y: float) -> OcrLine:
    return OcrLine(text, Region(0.36, y, 0.28, 0.028), 0.95)


def test_dialogue_is_spoken_or_a_sentence_never_a_menu_label():
    lk = Lookup(
        [
            line("見てくれ", "Look at this.", speaker="アダイヤ"),
            line("伝説…", "Legend..."),
            line("装備プロフイ", "Equipment · Profile"),  # a tab bar the capture stored
            line("売ける", "Sell"),  # a centred button in the subtitle band
        ]
    )
    assert is_dialogue(match_reading(lk, "見てくれ"))
    assert is_dialogue(match_reading(lk, "伝説…"))
    assert not is_dialogue(match_reading(lk, "装備プロフイ"))
    assert not is_dialogue(match_reading(lk, "売ける"))


def test_a_misread_name_never_starts_the_line(profile, glossary):
    text = "なんだか薄くてもろい殻のようなものを突き破った。"
    lk = Lookup([line(text, "I broke through something like a thin, brittle shell.")])
    d = DialogueComponent(
        profile, lk, glossary, CoverCache(None), CoverMaker(lk, CoverCache(None), profile),
        get_style("glow"),
    )  # fmt: skip
    name = row("テラフ", 0.775)  # テララ misread: not a known name, so a candidate row
    first = row(text, 0.835)
    # the name and the line read together still match the line fuzzily ...
    assert match_reading(lk, "テラフ" + text) is not None
    # ... but a block that starts at the name is not the line; the one at the text is
    assert match_block(lk, [name, first], None, None) is None
    hit, used = match_block(lk, [first], None, None)
    assert used == [first] and hit.translation.startswith("I broke through")
    # and the text row decides the layout: in-world dialogue, not NPC speech
    assert layout_at(d.profile, first).name == "world"
    assert layout_at(d.profile, name).name == "npc"


def test_a_shown_line_stays_while_readings_still_read_as_it():
    jp = "おお、テララ。任務はもう終わつたかの？"
    assert reads_as("おお、任務はもう終わつたかの？", jp)  # the OCR dropped the name
    assert reads_as("おお、テラ", jp)  # still typing out
    assert not reads_as("と一ぜん!ラル師匠の任務、カンタンだもん。", jp)  # the next line
