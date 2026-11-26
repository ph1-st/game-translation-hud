"""The look of the English: the colours of the Japanese, the Japanese painted over with what
surrounds it, a faint glow with no edge, never a box."""

from __future__ import annotations

import cv2
import numpy as np

from hud.backdrop import build
from hud.extent import PAD
from hud.looks import Palette
from hud.overlay import OverlayItem


def by_text(core, text):
    found = [it for it in core.items() if it.lines and " ".join(it.lines) == text]
    assert found, f"no cover with {text!r}: {[it.lines for it in core.items()]}"
    return found[0]


def strokes(frame: np.ndarray, rect) -> float:
    """How much stroke there is in *rect*: local contrast energy."""
    x, y, w, h = rect
    g = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 4)
    return float(np.mean(np.abs(hp)))


def composed(frame: np.ndarray, item: OverlayItem) -> np.ndarray:
    """*frame* with the backdrop of *item* drawn over it (no text)."""
    out = frame.astype(np.float32)
    got = build(frame, item, item.shine)
    if got is None:
        return frame
    x, y, rgba = got
    h, w = rgba.shape[:2]
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    region = out[y : y + h, x : x + w]
    out[y : y + h, x : x + w] = rgba[:, :, 2::-1].astype(np.float32) + region * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---- colours ------------------------------------------------------------------------------
def test_english_takes_the_japanese_colours(run):
    core, _ = run("004653_walk")
    quest = by_text(core, "Chapter 1: Charles Village")
    r, g, b = quest.fg
    assert r > b + 60 and g > b + 50  # the quest title is gold
    name = by_text(core, "Terara")
    r, g, b = name.fg
    assert b > r + 30  # the speaker name is blue
    # the dialogue is white, outlined in black
    (line,) = [it for it in core.items() if it.id.startswith("dlg:line:")]
    assert min(line.fg) >= 200 and line.edge is not None and max(line.edge) <= 120
    core, _ = run("004711_paragraph")
    (block,) = [it for it in core.items() if len(it.lines) > 1]
    assert max(block.fg) <= 140 and block.edge is None  # dark grey on the light panel


def test_text_that_looks_alike_gets_one_colour(run):
    core, _ = run("004720_list")
    rows = [it for it in core.items() if it.style == "label" and 1080 < it.rect[0] < 1120]
    rows = [it for it in rows if it.rect[1] < 500]
    assert len(rows) >= 5
    assert len({it.fg for it in rows}) == 1, [(it.lines, it.fg) for it in rows]


def test_colours_snap_to_the_games_colours():
    palette = Palette()
    white = palette.snap((240, 240, 238), (40, 40, 40))
    assert palette.snap((228, 232, 236), (60, 55, 50)) == white  # the same, sampled elsewhere
    assert palette.snap((228, 232, 236), None) != white  # no outline: another look
    assert palette.snap((200, 180, 60), (40, 40, 40)) != white  # gold


# ---- the Japanese is painted over, and nothing else --------------------------------------
def scene(seed: int, size=(120, 400)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(30, 220, (size[0] // 8, size[1] // 8, 3)).astype(np.uint8)
    return cv2.GaussianBlur(cv2.resize(small, (size[1], size[0])), (0, 0), 3)


def outlined(frame: np.ndarray, text: str, at=(40, 60)) -> np.ndarray:
    f = frame.copy()
    cv2.putText(f, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.putText(f, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2, cv2.LINE_AA)
    return f


def label(rect, erase, **kw) -> OverlayItem:
    x, y, w, h = erase
    return OverlayItem(
        rect, lines=["Text"], font_px=16, row_top=y + 2, row_h=h - 2 * PAD - 2,
        style="label", fg=(250, 250, 250), edge=(20, 20, 20), erase=erase, under=erase, **kw,
    )  # fmt: skip


def test_the_japanese_is_painted_over_with_the_scene_behind_it():
    bg = scene(3)
    frame = outlined(bg, "NIHONGO")
    erase = (34, 36, 160, 32)
    out = composed(frame, label(erase, erase))
    assert strokes(out, erase) < 0.3 * strokes(frame, erase)  # the text is gone
    # ... and replaced by what was around it, not by a box of one colour
    x, y, w, h = erase
    assert (
        abs(float(out[y : y + h, x : x + w].mean()) - float(bg[y : y + h, x : x + w].mean())) < 12
    )
    # far from it, the picture is untouched
    assert np.array_equal(out[:, 260:], frame[:, 260:])


def test_text_on_a_pill_is_painted_over_with_the_pill():
    frame = np.full((80, 300, 3), 245, np.uint8)  # a white panel ...
    cv2.rectangle(frame, (40, 24), (200, 50), (70, 66, 64), -1)  # ... a dark pill on it
    cv2.putText(frame, "ABCDEFG", (52, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    erase = (50, 27, 120, 21)  # the text fills the pill nearly top to bottom
    it = label(erase, erase, glow=0.0)
    it.fg, it.edge = (230, 230, 230), None
    out = composed(frame, it)
    x, y, w, h = erase
    inside = out[y + 2 : y + h - 2, x + 2 : x + w - 2].reshape(-1, 3).astype(np.float32)
    assert np.abs(inside - np.array([70, 66, 64], np.float32)).max() < 12  # no white bleeds in


def test_english_longer_than_its_pill_gets_a_longer_pill():
    frame = np.full((80, 300, 3), 245, np.uint8)
    cv2.rectangle(frame, (80, 24), (160, 50), (70, 66, 64), -1)
    cv2.putText(frame, "ABC", (92, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    erase = (90, 27, 60, 21)
    it = label((50, 27, 140, 21), erase, glow=0.0)  # the English needs 140 px
    it.fg, it.edge = (230, 230, 230), None
    out = composed(frame, it)
    # under the English, beyond the old pill: the pill's colour, not the white panel
    assert out[37, 60].max() < 100 and out[37, 180].max() < 100
    assert out[10, 60].min() > 230  # above it, the panel is as it was


def test_the_glow_has_no_edge():
    it = OverlayItem(
        (100, 100, 200, 20), lines=["A faint glow"], font_px=16, row_top=102, row_h=16,
        style="text", glow=0.2,
    )  # fmt: skip
    x, y, rgba = build(None, it, (255, 255, 255))
    a = rgba[:, :, 3].astype(np.float32) / 255.0
    assert 0.15 <= a.max() <= 0.2 + 1e-3  # faint
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    assert border.max() < 0.01  # it fades out before the edge of its image: no rectangle


def _union(rects):
    x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
    x1, y1 = max(r[0] + r[2] for r in rects), max(r[1] + r[3] for r in rects)
    return x0, y0, x1 - x0, y1 - y0


def panes_of(core, layout=""):
    return [it for it in core.items() if it.id.startswith(f"dlg:pane:{layout}")]


def composed_all(frame: np.ndarray, items) -> np.ndarray:
    """*frame* with the backdrops of *items* (built from *frame*) drawn over it in turn."""
    out = frame.astype(np.float32)
    for it in items:
        got = build(frame, it, it.shine)
        if got is None:
            continue
        x, y, rgba = got
        h, w = rgba.shape[:2]
        a = rgba[:, :, 3:4].astype(np.float32) / 255.0
        out[y : y + h, x : x + w] = rgba[:, :, 2::-1] + out[y : y + h, x : x + w] * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def test_frosted_glass_fits_the_rows_centred(run, profile):
    from hud.covers import px

    core, frame = run("004636_subtitle")
    (pane,) = panes_of(core)  # one pane over everything shown
    fh, fw = frame.shape[:2]
    x, y, w, h = pane.frost
    assert abs(x + w / 2 - fw / 2) <= 1  # centred on the screen, like the rows
    assert pane.fade_in > 0 and pane.fade_out > 0  # it comes and goes softly
    rows = (515, 798, 575, 75)  # the two Japanese rows
    gx, gy, gw, gh = pane.frost
    assert gx <= rows[0] and gx + gw >= rows[0] + rows[2]
    assert gy <= rows[1] and gy + gh >= rows[1] + rows[3]
    ax, ay, aw, ah = px(profile.dialogue_layouts["cutscene"].glass, fw, fh)
    assert gx >= ax and gx + gw <= ax + aw and gy >= ay and gy + gh <= ay + ah
    out = composed_all(frame, [pane])
    assert strokes(out, rows) < 0.2 * strokes(frame, rows)  # a blur: no glyph is left
    # a crisp outer edge (a pane, not a fade), a light line along it
    cx = fw // 2
    x, y, rgba = build(frame, pane, pane.shine)
    col = rgba[:, cx - x, 3].astype(np.int16)
    assert int(np.argmax(col > 250)) - int(np.argmax(col > 5)) <= 3
    top = gy
    edge_row = out[top : top + 2, cx - 60 : cx + 60].astype(np.float32).mean()
    inside = out[top + 6 : top + 8, cx - 60 : cx + 60].astype(np.float32).mean()
    assert edge_row > inside + 5
    # and the scene right outside them is as it was
    near = (slice(gy - 6, gy - 2), slice(gx, gx + gw))
    assert np.abs(out[near].astype(np.int16) - frame[near]).max() == 0


def test_the_glass_grows_with_a_new_row_and_never_shrinks(run):
    import cv2

    core, frame = run("004636_subtitle")
    d = core.dialogue
    sc = d.scene
    first_only = frame.copy()  # the second row (y 846-866) not typed out yet
    first_only[836:880, 470:1130] = cv2.medianBlur(frame[836:880, 470:1130], 31)
    sc.line, sc.lines, sc.line_gone_at = None, [], 9.9  # just the strokes on screen now
    sc.panes = []
    d.watch(first_only, 10.0)
    (pane,) = sc.panes
    first = pane.rect
    d.watch(frame, 10.05)
    (grown,) = sc.panes
    assert grown.key == pane.key  # the same pane...
    x, y, w, h = grown.rect  # ...grown over the second row, its top where it was
    assert y == first[1] and y + h > first[1] + first[3]
    d.watch(first_only, 10.1)
    assert sc.panes[0].rect == grown.rect  # it never shrinks


def test_the_divider_star_tells_a_conversation(run):
    core, frame = run("004653_walk")  # an NPC line, its name and divider over a busy scene
    d = core.dialogue
    lays = d.profile.dialogue_layouts

    def star(picture, lay) -> bool:
        return d.dividers.score(picture, lay) >= lay.divider.match

    assert star(frame, lays["npc"]) and not star(frame, lays["world"])
    for name in ("004636_subtitle", "004720_list", "004659_stats"):  # none in these
        _core, other = run(name)
        assert not star(other, lays["npc"]) and not star(other, lays["world"])


def test_a_conversation_is_covered_the_moment_its_divider_is_up(run, glossary):
    import cv2

    from hud.pipeline import HudCore
    from hud.style import get_style
    from tools.replay import RecordedEngine

    seen, frame = run("004653_walk")  # the layout seen once: the cache knows its look
    d0 = seen.dialogue
    core = HudCore(
        d0.profile, d0.lookup, glossary, seen.cache, RecordedEngine({}), get_style("glow")
    )
    core.watch(frame, 0.0)  # one watch, no OCR at all
    sc = core.dialogue.scene
    assert sc is not None and sc.layout.name == "npc" and sc.line is None
    panes = [p.frost for p in panes_of(core, "npc")]
    (name,) = [it.frost for it in core.items() if it.id == "dlg:namebox:npc"]
    fh, fw = frame.shape[:2]
    for b in panes + [name]:
        assert abs(b[0] + b[2] / 2 - fw / 2) <= 1  # centred
    # the panes cover the Japanese line (from its strokes on screen: the row as the OCR
    # read it), and the name's box stays above the divider line
    box = _union(panes)
    rx, ry, rw, rh = d0.scene.line.anchor.rect
    assert box[0] <= rx and box[0] + box[2] >= rx + rw and box[1] <= ry
    assert box[1] + box[3] >= ry + rh
    assert name[1] + name[3] <= round(0.7378 * fh) - 2
    # once the line is read, its English lies on the glass too
    bx, by, bw, bh = _union([p.frost for p in panes_of(seen, "npc")])
    (line,) = [it for it in seen.items() if it.id.startswith("dlg:line:")]
    lx, ly, lw, lh = line.rect
    assert bx <= lx and bx + bw >= lx + lw
    # the divider goes (its star blurred away): a moment later the conversation is over
    gone = frame.copy()
    y = round(0.7378 * fh)
    gone[y - 12 : y + 12, 740:860] = cv2.blur(frame[y - 12 : y + 12, 740:860], (25, 25))
    core.watch(gone, 0.05)
    assert core.dialogue.scene is not None  # not at once: a frame of noise is no ending
    core.watch(gone, 0.4)
    assert core.dialogue.scene is None


def test_english_types_out_at_an_even_pace():
    from hud.overlay import render_text, reveal_share, reveal_target, revealed, text_lines

    item = OverlayItem(
        (100, 100, 320, 60), lines=["Everyone here is kind.", "Truly kind."], font_px=18,
        row_top=106, row_h=14, pitch=26, style="text", fg=(255, 255, 250), edge=(40, 40, 40),
        reveal=(0.25, 100.0, 1.5),
    )  # fmt: skip
    assert reveal_target(item, 99.0) == 0.25  # where it starts...
    assert reveal_target(item, 100.0) == 0.25
    assert abs(reveal_target(item, 100.75) - (0.25 + 0.75 / 2)) < 1e-9  # ...half way...
    assert reveal_target(item, 101.5) == 1.0 and reveal_target(item, 105.0) == 1.0  # ...all
    assert reveal_share((0.0, 0.0, 0.0), 0.0) == 1.0  # no time to it: all of it at once
    full = render_text(item)
    x, y, rgba = full
    assert not revealed(full, item, 0.0)[2][:, :, 3].any()  # nothing yet
    assert np.array_equal(revealed(full, item, 1.0)[2], rgba)  # all of it
    # 30 % of it (the first line is two thirds of the text): the first line about half way
    # (left before right), the second not yet
    half = revealed(full, item, 0.3)[2][:, :, 3].astype(np.float32)
    (l0, r0, t0, b0), (l1, r1, t1, b1) = text_lines(item)
    first = half[int(t0) - y : int(b0) - y]
    second = half[int(t1) - y : int(b1) - y]
    third = (r0 - l0) / 3.0
    left = first[:, int(l0 - x) : int(l0 + third - x)]
    right = first[:, int(r0 - third - x) : int(r0 - x)]
    assert left.sum() > 0 and right.sum() < 0.2 * left.sum()
    assert second.sum() == 0


def test_a_newer_reading_changes_only_the_pace(run):
    from hud.dialogue.scene import reveal
    from hud.overlay import reveal_share

    core, _frame = run("004653_walk")
    sc = core.dialogue.scene
    s0, t0, seconds = sc.line.reveal
    assert s0 == 0.0 and seconds > 0  # from nothing, at the game's pace
    now = t0 + 0.1  # a line that came whole types out in 0.3 s
    before = reveal_share(sc.line.reveal, now)
    assert 0.0 < before < 1.0
    for typed in (sc.line.typed - 6, len(sc.line.source)):  # a reading behind, one at the end
        sc.line.typed = max(0, typed)
        new = reveal(sc, before, now)
        assert abs(reveal_share(new, now) - before) < 1e-9  # no jump...
        assert reveal_share(new, now + 0.05) > before  # ...and no pause


def test_a_dialogue_line_types_out_on_frosted_glass(run):
    core, _frame = run("004653_walk")
    (line,) = [it for it in core.items() if it.id.startswith("dlg:line:")]
    s0, t0, seconds = line.reveal
    assert s0 == 0.0 and seconds >= 0.3  # it types out, at an even pace
    fx, fy, fw, fh = _union([p.frost for p in panes_of(core)])
    bx, by, bw, bh = line.rect
    assert fx <= bx and fx + fw >= bx + bw and fy <= by  # the English lies on the glass


def test_a_replay_shows_english_as_far_as_it_typed(run):
    from tools.replay import Animator, compose

    core, frame = run("004653_walk")
    items = core.items()
    (line,) = [it for it in items if it.id.startswith("dlg:line:")]
    rest = [it for it in items if it is not line]
    x, y, w, h = line.rect
    with_line, without = Animator(), Animator()
    at = line.reveal[1]
    box = (slice(y, y + h), slice(x, x + w))
    for t, typed in ((at, False), (at + 2.0, True)):
        a = compose(frame, items, now=t, animator=with_line)
        b = compose(frame, rest, now=t, animator=without)
        diff = np.abs(a[box].astype(np.int16) - b[box]).mean()
        assert (diff > 5.0) if typed else (diff < 0.5)  # not typed yet, then typed out


def test_a_dash_under_the_longer_english_is_painted_over(run):
    core, frame = run("004720_list")
    it = by_text(core, "Gathered in the azure sky,")
    out = composed(frame, it)
    dash = (1170, 527, 50, 14)  # "——" after 蒼天に集いし, never read by the OCR
    assert strokes(out, dash) < 0.4 * strokes(frame, dash)


def test_backdrops_are_built_off_the_gui_thread():
    import time

    from hud.overlay import BackdropBuilder

    frame = outlined(scene(5), "NIHONGO")
    erase = (34, 36, 160, 32)
    builder = BackdropBuilder()
    builder.start()
    try:
        builder.request("a", label(erase, erase), frame)
        builder.request("a", label(erase, erase, glow=0.1), frame)  # the newest one wins
        done: dict = {}
        deadline = time.time() + 5
        while time.time() < deadline and not (done and builder.idle()):
            done.update(builder.take())
            time.sleep(0.01)
    finally:
        builder.stop()
    item, built = done["a"]
    assert item.glow == 0.1 and built is not None


def test_a_speaker_name_goes_with_its_pixels_and_comes_back_by_them(run):
    import cv2

    core, frame = run("004653_walk")
    d = core.dialogue
    assert d.scene.name == "Terara" and d.scene.name_anchor is not None
    # another speaker: the Japanese name changes (here: blurred away) under the glass, and
    # the English name goes with it at once
    x, y, w, h = d.scene.name_ink
    other = frame.copy()
    other[y : y + h, x : x + w] = cv2.blur(frame[y : y + h, x : x + w], (21, 21))
    for t in (1.0, 1.05, 1.1):
        d.watch(other, t)
    assert d.scene is not None and d.scene.name is None
    # the name is back: known by its strokes, no OCR
    d.watch(frame, 1.2)
    assert d.scene.name == "Terara"


def test_a_name_in_roman_letters_needs_no_translation(run):
    from hud.dialogue import is_roman

    core, _frame = run("004653_walk")
    names = core.dialogue.names
    assert is_roman("Wren") and is_roman("Mr. Smith") and not is_roman("テララ")
    assert is_roman("José") and is_roman("Zoë") and not is_roman("A")  # accents too
    assert names.name_of("Wren") == "Wren"  # the player, not in the database: as it is
    assert names.name_of("Wren.") == "Wren"  # (a dot read at its end is none of it)
    assert names.name_of("テララ") == "Terara"  # a known name: its English


def test_the_players_name_fills_the_english_while_the_line_types_out(lookup, player):
    from hud.matching import match_reading

    # read before the player's name typed out (and as the capture read the rest)
    hit = match_reading(lookup, "はしめまして、村長さん。私は")
    assert hit is not None and hit.translation == f"Nice to meet you, Chief. I'm {player}."


def test_white_text_is_white_with_a_black_outline():
    from hud.overlay import render_text

    item = OverlayItem((100, 100, 200, 30), lines=["Quest"], font_px=24, row_top=104,
                       row_h=20, style="label", fg=(80, 150, 255), edge=None)  # fmt: skip
    _x, _y, coloured = render_text(item)
    item.plain = True
    _x, _y, white = render_text(item)

    def colours(rgba):
        a = rgba[:, :, 3].astype(np.float32)
        solid = a >= 250  # premultiplied: opaque pixels are their colour
        return rgba[solid][:, :3].astype(int)

    c, w = colours(coloured), colours(white)
    assert (c[:, 2] - c[:, 0]).max() > 100  # blue
    assert (w.max(axis=1) - w.min(axis=1)).max() <= 2  # grey only: white and black
    assert (w.min(axis=1) >= 250).sum() > 20 and (w.max(axis=1) <= 5).sum() > 20
