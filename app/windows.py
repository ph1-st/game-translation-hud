"""The windows the player can choose the game from."""

from __future__ import annotations

import os
from dataclasses import dataclass

from hud.platform import SHARED_PROGRAMS, current

SKIP_TITLES = {
    "Program Manager",
    "Windows Input Experience",
    "Settings",
    "Microsoft Text Input Application",
}
MIN_W, MIN_H = 320, 200  # smaller windows are no game's


@dataclass
class Choice:
    id: int  # the window (hud.platform.WindowRef.id)
    title: str
    process: str  # the program's file name, e.g. "game.exe"
    size: tuple[int, int]

    @property
    def own_program(self) -> str:
        """The program to know the game by: none for a program many windows share (a
        browser, an emulator, a video player): then the title alone tells the game."""
        return "" if self.process.lower() in SHARED_PROGRAMS else self.process

    def label(self) -> str:
        w, h = self.size
        proc = f"  —  {self.process}" if self.process else ""
        return f"{self.title}{proc}  ({w}×{h})"


def choices() -> list[Choice]:
    """Visible windows big enough to be a game, largest first (this app's own left out)."""
    me = os.getpid()
    out = [
        Choice(w.id, w.title, w.program, (w.client[2], w.client[3]))
        for w in current().windows()
        if w.pid != me and w.client[2] >= MIN_W and w.client[3] >= MIN_H
        and w.title not in SKIP_TITLES
    ]
    out.sort(key=lambda c: -(c.size[0] * c.size[1]))
    return out
