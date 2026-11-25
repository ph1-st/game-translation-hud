"""Covers sooner: labels seen before come back from memory without the OCR, and the OCR
reads only what it has to (the text under the covers shown is masked out)."""

from __future__ import annotations

import cv2
import numpy as np

from capture.ocr import OcrResult
from hud.board import Cover
from hud.extent import PAD


def by_text(core, text):
    found = [it for it in core.items() if it.lines and " ".join(it.lines) == text]
    return found[0] if found else None


def cover_of(core, text) -> Cover:
    for b in core.boards:
        for c in b.snapshot():
            if c.text == text:
                return c
    raise AssertionError(f"no cover {text!r}")


def without(frame: np.ndarray, rect) -> np.ndarray:
    """*frame* with the text in *rect* gone (blurred into its background)."""
    x, y, w, h = rect
    out = frame.copy()
    m = 6
    out[y - m : y + h + m, x - m : x + w + m] = cv2.GaussianBlur(
        frame[y - m : y + h + m, x - m : x + w + m], (0, 0), 9
    )
    return out


def test_a_label_seen_before_comes_back_without_the_ocr(run):
    core, frame = run("004720_list")
    before = cover_of(core, "Details 2")
    core.watch(frame, 0.10)  # shown: remembered
    gone = without(frame, before.geom.cover)
    core.watch(gone, 0.15)
    core.watch(gone, 0.20)
    assert by_text(core, "Details 2") is None  # its text went: the watch took it off
    calls = core.ocr_calls
    core.watch(frame, 0.25)  # back on screen: recognized on two frames in a row...
    core.watch(frame, 0.30)
    back = by_text(core, "Details 2")
    assert back is not None and back.rect == before.geom.box  # ...exactly as it was
    assert core.ocr_calls == calls  # and no OCR was needed


def test_a_longer_line_that_starts_the_same_is_not_brought_back(run):
    core, frame = run("004720_list")
    c = cover_of(core, "Details 2")
    core.watch(frame, 0.10)
    gone = without(frame, c.geom.cover)
    core.watch(gone, 0.15)
    core.watch(gone, 0.20)
    # the line comes back with more text right after it ("Details 2" -> "Details 2 4")
    longer = frame.copy()
    x, y, w, h = c.geom.cover
    cv2.putText(longer, "4", (x + w + 2, y + h - PAD - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (90, 90, 95), 2, cv2.LINE_AA)  # fmt: skip
    for t in (0.25, 0.30, 0.35):
        core.watch(longer, t)
    assert by_text(core, "Details 2") is None


def test_a_reading_of_another_line_takes_the_place_of_a_recalled_cover(run):
    core, frame = run("004720_list")
    c = cover_of(core, "Details 2")
    # memory put a cover with another line at this place
    core.ui_board.remove([c], "test")
    wrong = Cover("Wrong line", "Wrong line", c.geom, c.anchor, light=c.light, cells=c.cells,
                  recalled=True)  # fmt: skip
    assert core.ui_board.add_if_free(wrong)
    core.tracker.ref = None  # the OCR reads the screen again
    core.tick(frame, lambda: frame, 1.0)
    assert by_text(core, "Wrong line") is None
    assert by_text(core, "Details 2") is not None


class Spy:
    """The recorded OCR, keeping the images it is given."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.device = engine.device
        self.images: list[tuple[np.ndarray, object]] = []

    def read(self, frame, region=None, min_conf=0.0, fine=False):
        self.images.append((frame, region))
        try:
            return self.engine.read(frame, region=region, min_conf=min_conf)
        except KeyError:  # a region the fixture never read: these tests count reads
            return OcrResult([], 0.0)


def test_text_already_covered_is_masked_out_of_the_next_read(run):
    core, frame = run("004720_list")
    c = cover_of(core, "Ultimate")
    n = cover_of(core, "World Lore 1")  # a template: its number may change, it is read
    core.engine = spy = Spy(core.engine)
    core.tracker.ref = None  # everything changed: the whole frame is read again
    core.tick(frame, lambda: frame, 1.0)
    # the UI read (of the whole frame): nothing left to read under a plain label...
    x, y, w, h = c.geom.cover
    under = [img[y : y + h, x : x + w] for img, region in spy.images if region is None]
    assert under and all(float(u.reshape(-1, 3).std(axis=0).max()) < 1.0 for u in under)
    # ...while a line with a number in it is read again (a timer, gold, a level)
    x, y, w, h = n.geom.cover
    under = [img[y : y + h, x : x + w] for img, region in spy.images if region is None]
    assert under and all(float(u.reshape(-1, 3).std(axis=0).max()) > 10.0 for u in under)


def test_in_a_conversation_the_band_is_read_first_the_rest_now_and_then(run):
    from hud.pipeline import SCENE_UI_EVERY_S

    core, frame = run("004653_walk")
    assert core.dialogue.scene is not None
    core.engine = spy = Spy(core.engine)

    def ui_reads():
        return [r for _img, r in spy.images if r is None or r.name != "dialogue_crop"]

    core.tracker.ref = None  # everything changed: the whole frame would be read
    moving = np.roll(frame, 8, axis=1)  # (the scene moves on behind: never at rest)
    core.tick(moving, lambda: moving, 0.2)
    assert spy.images and not ui_reads()  # the band only, a moment after the last full read
    moving = np.roll(frame, 16, axis=1)
    core.tick(moving, lambda: moving, 0.1 + SCENE_UI_EVERY_S)
    assert ui_reads()  # the rest of the screen, still marked as changed, once it is due


def test_what_was_learned_comes_back_in_the_next_session(run, tmp_path):
    from hud.recall import Recall

    core, frame = run("004720_list")
    core.watch(frame, 0.1)  # the covers shown are learned
    path = tmp_path / "recall.npz"
    core.recall.save(path)
    again = Recall()
    again.load(path)
    assert again.count == core.recall.count > 0
    # a new session: nothing shown yet, the list is on screen: it comes back at once
    fresh, _ = run("004659_stats")  # (another screen's core, its boards cleared)
    for b in fresh.boards:
        b.clear()
    fresh.recall = again
    fresh.dialogue.end_scene()
    calls = fresh.ocr_calls
    fresh.watch(frame, 1.0)
    fresh.watch(frame, 1.05)
    assert by_text(fresh, "Details 2") is not None and fresh.ocr_calls == calls


# ---- the menu memory holds menus only ----------------------------------------------------
def test_story_covers_are_never_remembered(run):
    core, frame = run("004720_list")
    covers = core.ui_board.snapshot()
    story, menu = covers[0], covers[1]
    story.story = True  # (as a line of narration would be)
    core.recall.learn(core.boards, frame)
    remembered = {m.template for ms in core.recall.memory.values() for m in ms}
    assert story.template not in remembered and menu.template in remembered


def test_story_blocks_never_go_into_the_cover_cache(run, tmp_path):
    from hud.covercache import CoverCache

    core, frame = run("004720_list", CoverCache(tmp_path / "covers.json"))
    c = core.ui_board.snapshot()[0]
    c.story, c.cached = True, False
    core.maker.remember(core.ui_board, 10.0, frame.shape[1], frame.shape[0])
    assert c.template not in core.cache.covers


def test_the_cover_cache_forgets_the_least_recently_used_line(monkeypatch):
    from hud import covercache
    from hud.covercache import CoverCache
    from hud.layout import Geom

    monkeypatch.setattr(covercache, "MAX_LINES", 2)
    cache = CoverCache(None)
    g = Geom((10, 10, 50, 12), (10, 10, 50, 12), 12, "left", ["x"], 10, 10, 0)
    cache.put("A", g, 100, 100)
    cache.put("B", g, 100, 100)
    assert cache.find("A", (10, 10, 50, 12), 100, 100) is not None  # A was used again
    cache.put("C", g, 100, 100)
    assert list(cache.covers) == ["A", "C"]  # B was used least recently


def test_the_cover_cache_is_written_now_and_then_not_every_tick(tmp_path):
    from hud.covercache import SAVE_EVERY_S, CoverCache
    from hud.layout import Geom

    path = tmp_path / "covers.json"
    cache = CoverCache(path)
    cache.put("A", Geom((10, 10, 50, 12), (10, 10, 50, 12), 12, "left", ["x"], 10, 10, 0),
              100, 100)  # fmt: skip
    cache.save_soon()
    assert not path.exists()  # written a moment ago (at start): not again yet
    cache._saved_at -= SAVE_EVERY_S
    cache.save_soon()
    assert path.exists()


def test_recall_memory_forgets_the_least_recently_used(run, monkeypatch):
    from hud import recall

    monkeypatch.setattr(recall, "MAX_MEMORY", 2)
    core, frame = run("004720_list")
    mem = recall.Recall()
    covers = core.ui_board.snapshot()[:3]
    for c in covers:
        board = recall.Board("ui", core.ui_board.style)
        assert board.add_if_free(c)
        mem.learn([board], frame)
    assert mem.count == 2 and covers[0].template not in mem.memory  # the oldest went


# ---- how often what is read --------------------------------------------------------------
def test_while_a_menu_is_up_the_dialogue_band_is_read_now_and_then(run):
    from hud.pipeline import MENU_BAND_EVERY_S

    core, frame = run("004720_list")  # a menu: many labels covered, no conversation
    assert len(core.ui_board) >= 4 and core.dialogue.scene is None
    core.engine = spy = Spy(core.engine)

    def band_reads():
        return [r for _img, r in spy.images if r is not None and r.name == "dialogue_crop"]

    y0 = int(core.dialogue.crop().y * frame.shape[0])
    t, seconds = 0.1, 2.0
    for k in range(1, int(seconds / 0.05)):  # something keeps moving in the band only
        f = frame.copy()
        f[y0:] = np.roll(frame[y0:], 8 * k, axis=1)
        core.tick(f, lambda f=f: f, t)
        t += 0.05
    assert 1 <= len(band_reads()) <= 1 + seconds / MENU_BAND_EVERY_S  # now and then, not every tick


def test_exploring_the_whole_screen_is_read_at_a_steady_pace(run):
    from hud.pipeline import EXPLORE_FULL_EVERY_S

    core, frame = run("004720_list")
    for b in core.boards:  # nothing covered, no conversation: exploring
        b.clear()
    core.engine = spy = Spy(core.engine)
    core.tracker.reset()
    first, t, seconds = None, 0.1, 2.0
    for k in range(1, int(seconds / 0.05)):  # the camera turns: the picture never rests
        for b in core.boards:  # (and nothing stays covered: no menu)
            b.clear()
        f = np.roll(frame, 8 * k, axis=1)
        core.tick(f, lambda f=f: f, t)
        if first is None and [r for _img, r in spy.images if r is None]:
            first = t
        t += 0.05
    full = [r for _img, r in spy.images if r is None]
    assert first is not None  # read at once, not after waiting for it to rest...
    assert len(full) <= 1 + seconds / EXPLORE_FULL_EVERY_S  # ...then at a steady pace


def test_a_picture_at_rest_is_read_at_once(run):
    # the waits spare the GPU while the picture moves; a menu that stopped scrolling is
    # read the moment it rests
    core, frame = run("004720_list")
    for b in core.boards:
        b.clear()
    core.engine = spy = Spy(core.engine)
    core.tracker.reset()
    moved = np.roll(frame, 8, axis=0)
    core.tick(moved, lambda: moved, 0.1)
    assert not [r for _img, r in spy.images if r is None]  # still moving: the read waits
    core.tick(moved, lambda: moved, 0.15)  # the same picture again: at rest
    assert [r for _img, r in spy.images if r is None]
