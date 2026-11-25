"""A slot the game scrolls its text through (the quest title): one line, read at rest,
shown still while the text slides underneath."""

from __future__ import annotations

import cv2
import numpy as np


def ticker_lines(core):
    return [it.lines[0] for it in core.items() if it.id.startswith("ticker:")]


def test_the_quest_title_stays_put_while_it_scrolls(run):
    core, frame = run("004653_walk")
    assert ticker_lines(core) == ["Chapter 1: Charles Village"]  # read at rest
    (slot,) = [r for _n, r in core.ticker.slots(1600, 900)]
    # nothing else covers the slot
    for c in core.ui_board.snapshot():
        cx = c.geom.cover[0] + c.geom.cover[2] / 2
        cy = c.geom.cover[1] + c.geom.cover[3] / 2
        x, y, w, h = slot
        assert not (x <= cx <= x + w and y <= cy <= y + h), c.text
    # the title slides through the slot: a fragment at every frame
    x, y, w, h = slot
    for t, shift in ((0.5, 30), (0.55, 60), (0.6, 90), (0.65, 120)):
        moved = frame.copy()
        moved[y : y + h, x : x + w] = np.roll(frame[y : y + h, x : x + w], -shift, axis=1)
        core.watch(moved, t)
        core.tick(moved, lambda m=moved: m, t)
        assert ticker_lines(core) == ["Chapter 1: Charles Village"], t
    # the tracker goes: the slot holds no title any more
    empty = frame.copy()
    empty[y : y + h, x : x + w] = cv2.medianBlur(frame[y : y + h, x : x + w], 31)
    core.watch(empty, 1.0)
    core.watch(empty, 1.5)
    assert ticker_lines(core) == []
