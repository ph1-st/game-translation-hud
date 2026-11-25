"""What the player wants translated: the story (dialogue, subtitles, narration, choices)
and/or menus and other UI text, switched while the HUD runs."""

from __future__ import annotations

from db.store import Line
from hud.content import Content, is_prose, is_story
from hud.covercache import CoverCache
from hud.pipeline import HudCore
from hud.style import get_style
from tools.replay import RecordedEngine, load_fixture

GAME = "azur-promilia"


def line(text: str, kind: str = "ui", speaker: str | None = None) -> Line:
    return Line(
        key=text, source_text=text, speaker=speaker, screen_type=kind, is_template=False,
        translation="x", tier="human", source="fixture", chapter=None, seq=None, game_version=None,
    )  # fmt: skip


def test_story_is_dialogue_narration_and_choices_never_menu_labels():
    assert is_story(line("おはよう", "dialogue"))
    assert is_story(line("あげる。", "choice"))
    assert is_story(line("はい", "ui", speaker="テララ"))  # a line someone says
    # narration a capture stored as UI: a sentence
    assert is_story(line("魚を投げると、スカイホークは器用に丸呑みした。"))
    for label in ("キボを呼び戻す", "マウントホイールを開く", "作物を5回収穫する", "確認"):
        assert not is_story(line(label)), label
    assert is_prose("その隣ではパモーラが落ち込んだような表情をしている。")
    assert not is_prose("第一章 迫り来る影")
    c = Content(dialogue=True, ui=False)
    assert c.allows(True) and not c.allows(False) and not c.everything


def core_for(name, profile, lookup, glossary):
    frame, meta = load_fixture(GAME, name)
    core = HudCore(
        profile, lookup, glossary, CoverCache(None), RecordedEngine(meta["ocr"]),
        get_style("glow"),
    )  # fmt: skip
    return core, frame


def dialogue_items(core):
    return [it for it in core.items() if it.id.startswith("dlg:line:")]


def test_story_only_leaves_the_menus_in_japanese(profile, lookup, glossary):
    core, frame = core_for("004653_walk", profile, lookup, glossary)
    everything, _ = core_for("004653_walk", profile, lookup, glossary)
    everything.tick(frame, lambda: frame, 0.0)
    assert everything.ui_board.snapshot()  # the gameplay HUD's labels, when all is on
    core.set_content(dialogue=True, ui=False)
    core.tick(frame, lambda: frame, 0.0)
    assert dialogue_items(core)  # the NPC's line is translated...
    assert all(c.story for c in core.ui_board.snapshot())  # ...the labels are not


def test_menus_only_leaves_the_story_in_japanese(profile, lookup, glossary):
    core, frame = core_for("004653_walk", profile, lookup, glossary)
    core.set_content(dialogue=False, ui=True)
    core.tick(frame, lambda: frame, 0.0)
    assert not dialogue_items(core) and core.dialogue.scene is None
    assert core.ui_board.snapshot()


def test_switching_the_menus_off_takes_their_covers_away_at_once(profile, lookup, glossary):
    core, frame = core_for("004653_walk", profile, lookup, glossary)
    core.tick(frame, lambda: frame, 0.0)
    labels = [c for c in core.ui_board.snapshot() if not c.story]
    assert labels
    core.set_content(dialogue=True, ui=False)
    assert not [c for c in core.ui_board.snapshot() if not c.story]
    assert dialogue_items(core)  # the story stays as it is
    core.set_content(dialogue=True, ui=True)  # and back: the next read covers them again
    assert core.tracker._forget  # the next tick reads everything again
    core.tick(frame, lambda: frame, 2.0)  # (in a conversation: the rest every second)
    assert [c for c in core.ui_board.snapshot() if not c.story]
