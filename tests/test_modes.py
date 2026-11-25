"""Cover modes (frosted glass that grows, one fixed pane, natural) and the hooks live
translation uses: text the database does not know is reported, and a translation made while
the HUD runs is shown from the next tick on, a little faded."""

from __future__ import annotations

import pytest

from capture.profile import load_profile
from db.lookup import Lookup
from hud.covercache import CoverCache
from hud.covers import px
from hud.pipeline import HudCore
from hud.style import get_style
from tools.replay import RecordedEngine, load_fixture

GAME = "azur-promilia"
pytestmark = pytest.mark.footage


def make(name, lookup, glossary, mode="grow", confirmation="balanced"):
    frame, meta = load_fixture(GAME, name)
    core = HudCore(
        load_profile(GAME), lookup, glossary, CoverCache(None), RecordedEngine(meta["ocr"]),
        get_style("glow"), cover_mode=mode, confirmation_mode=confirmation,
    )  # fmt: skip
    return core, frame


def test_natural_mode_has_no_dialogue_handling(lookup, glossary):
    core, frame = make("004653_walk", lookup, glossary, "natural")
    assert core.profile.dialogue is None and not core.profile.dialogue_layouts
    assert core.dialogue.territory() == []
    core.tick(frame, lambda: frame, 0.0)
    assert not [it for it in core.items() if it.id.startswith("dlg:")]


def test_fixed_mode_lays_one_pane_over_the_whole_dialogue_area(lookup, glossary):
    core, frame = make("004636_subtitle", lookup, glossary, "fixed")
    core.tick(frame, lambda: frame, 0.0)
    panes = [it for it in core.items() if it.id.startswith("dlg:pane:")]
    fh, fw = frame.shape[:2]
    glass = px(core.profile.dialogue_layouts["cutscene"].glass, fw, fh)
    assert [p.frost for p in panes] == [glass]
    core.dialogue.watch(frame, 0.5)  # the strokes never add or change a pane
    assert [it.frost for it in core.items() if it.id.startswith("dlg:pane:")] == [glass]


def test_text_the_database_does_not_know_is_reported(glossary, player):
    core, frame = make("004653_walk", Lookup([], player_name=player), glossary)
    heard = []
    core.set_live(lambda *a: heard.append(a))
    core.tick(frame, lambda: frame, 0.0)
    kinds = {kind for _text, kind, *_ in heard}
    assert kinds == {"ui", "dialogue"}
    (line,) = [h for h in heard if h[1] == "dialogue"]
    text, _kind, _speaker, conf, box = line
    assert "朝" in text and conf > 0.5 and box[2] > 100  # Terara's line, its whole row
    assert core.glossary is glossary  # the names and terms the live translator is given


def test_a_live_translation_is_shown_from_the_next_tick_faded(glossary, player):
    core, frame = make("004653_walk", Lookup([], player_name=player), glossary)
    heard = []
    core.set_live(lambda *a: heard.append(a))
    core.tick(frame, lambda: frame, 0.0)
    assert not [it for it in core.items() if it.id.startswith("dlg:line:")]
    text, kind, speaker, _conf, box = [h for h in heard if h[1] == "dialogue"][0]
    core.add_translation(text, "We might make it for breakfast!", kind, speaker, box)
    core.tick(frame, lambda: frame, 0.2)
    (line,) = [it for it in core.items() if it.id.startswith("dlg:line:")]
    assert line.lines == ["We might make it for breakfast!"] and line.provisional


def test_text_still_moving_is_not_reported_but_read_again(glossary, player):
    import numpy as np

    core, frame = make("004720_list", Lookup([], player_name=player), glossary)
    heard = []
    core.set_live(lambda *a: heard.append(a))
    moved = np.roll(frame, 6, axis=0)  # the list scrolled on while it was read
    core.tick(frame, lambda: moved, 0.0)
    assert not [h for h in heard if h[1] == "ui"] and core.tracker.has_freed()
    core.tracker.reset()  # (read whole: the fixture recorded no bands)
    core.tick(moved, lambda: moved, 0.5)  # it moved: read once it rests...
    core.tick(moved, lambda: moved, 0.55)  # ...and it does: what it reads is told
    assert [h for h in heard if h[1] == "ui"]


def test_a_labels_live_translation_is_placed_at_once_without_reading_again(glossary, player):
    core, frame = make("004720_list", Lookup([], player_name=player), glossary)
    heard = []
    core.set_live(lambda *a: heard.append(a))
    core.tick(frame, lambda: frame, 0.0)
    fh = frame.shape[0]
    text, kind, speaker, _conf, box = next(
        h for h in heard if h[1] == "ui" and len(h[0]) >= 3 and h[4][1] < fh / 2
    )
    reads = []
    real = core.engine.read
    core.engine.read = lambda *a, **k: reads.append(a) or real(*a, **k)
    woken = []
    core.wake = lambda: woken.append(1)
    core.add_translation(text, "Translated label", kind, speaker, box)
    assert woken  # (the worker ticks at once)
    core.tick(frame, lambda: frame, 0.3)
    assert not reads  # placed from the reading it came from
    shown = [it for it in core.items() if " ".join(it.lines) == "Translated label"]
    assert shown and shown[0].provisional


def test_instant_mode_keeps_a_short_dialogue_match(lookup, glossary):
    import numpy as np

    balanced, frame = make("004653_walk", lookup, glossary, confirmation="balanced")
    gone = np.zeros_like(frame)  # the next captured frame arrived before OCR returned
    balanced.tick(frame, lambda: gone, 0.0)
    assert not [it for it in balanced.items() if it.id.startswith("dlg:line:")]
    assert balanced.recall.passes_required == 2  # a remembered label: two frames in a row

    instant, frame = make("004653_walk", lookup, glossary, confirmation="instant")
    instant.tick(frame, lambda: gone, 0.0)
    assert [it for it in instant.items() if it.id.startswith("dlg:line:")]
    assert instant.recall.passes_required == 1
