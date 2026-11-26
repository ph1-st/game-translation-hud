"""Game profile loading: window matching, OCR regions and ignore regions.

All coordinates are relative (0-1) to the game window / frame so that the same profile works
at any resolution. A profile lives in ``games/<game>/profile.yaml`` and must stay the only
place with game-specific knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

GAMES_DIR = Path(__file__).resolve().parent.parent / "games"
# screen types of regions that belong to dialogue handling (the rest is UI)
DIALOGUE_REGION = "dialogue"
SPEAKER_REGIONS = ("speaker", "speaker_npc")
DIALOGUE_REGIONS = (DIALOGUE_REGION, *SPEAKER_REGIONS)
DIVIDER_MATCH = 0.45  # a divider's star matches its picture at least this well...
DIVIDER_AGAIN = 0.28  # ...or this well, when its conversation was up a moment ago and its
# text is still there (a camera cut briefly hides the star)


@dataclass(frozen=True)
class Region:
    """A rectangle in relative coordinates (0-1) of the full frame."""

    x: float
    y: float
    w: float
    h: float
    name: str = ""
    screen_type: str = "ui"
    multiline: bool = False
    note: str = ""
    align: str = ""  # left | right | center: how the game lays out text in this area
    ticker: bool = False  # the game scrolls its text through this slot (a marquee)

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) pixel bounds for a frame of the given size."""
        x0 = max(0, int(round(self.x * width)))
        y0 = max(0, int(round(self.y * height)))
        x1 = min(width, int(round(self.x2 * width)))
        y1 = min(height, int(round(self.y2 * height)))
        return x0, y0, x1, y1

    def crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.to_pixels(w, h)
        return frame[y0:y1, x0:x1]

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def overlap_fraction(self, other: "Region") -> float:
        """Fraction of the other region's area that lies inside this region."""
        ix = max(0.0, min(self.x2, other.x2) - max(self.x, other.x))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y, other.y))
        area = other.w * other.h
        return 0.0 if area <= 0 else (ix * iy) / area

    def as_list(self) -> list[float]:
        return [round(self.x, 4), round(self.y, 4), round(self.w, 4), round(self.h, 4)]

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "Region":
        return cls(
            x=float(d["x"]),
            y=float(d["y"]),
            w=float(d["w"]),
            h=float(d["h"]),
            name=name,
            screen_type=str(d.get("screen_type", "ui")),
            multiline=bool(d.get("multiline", False)),
            note=str(d.get("note", "")),
            align=str(d.get("align", "")),
            ticker=bool(d.get("ticker", False)),
        )


@dataclass
class WindowMatch:
    process_names: list[str] = field(default_factory=list)
    title_patterns: list[str] = field(default_factory=list)
    # titles the player picked: a browser's, a folder's or a chat's window counts only with
    # one of these (whole, any case), never by a pattern
    exact_titles: list[str] = field(default_factory=list)
    # width / height of the game's picture: shown in a window of another shape, it sits
    # between black bars, and the regions are fractions of the picture (None: the window)
    aspect: float | None = None


@dataclass
class Divider:
    """The line with a star in its middle that the game draws under a speaker name: while
    it is there, so is the conversation."""

    y: float  # relative y of the line (the star sits on it, centred on the screen)
    template: str  # a picture of the star, relative to the game's profile folder...
    height: int = 900  # ...cut from a frame this many pixels tall
    match: float = DIVIDER_MATCH  # how well the star must match (normalized correlation)
    again: float = DIVIDER_AGAIN  # ...right after its conversation, with its text there


@dataclass
class DialogueLayout:
    """One way the game shows dialogue: where its rows start, the fixed panel that covers
    them (relative x, y, w, h; the text grows away from *anchor* when it needs more
    lines) and the band of the speaker name above it, if any."""

    name: str
    rows_top: tuple[float, float]  # the first row's top lies in this range (relative y)
    panel: Region
    anchor: str = "top"  # top | bottom: the edge the text starts from (rows grow away from it)
    speaker: Region | None = None
    glass: Region | None = None  # everywhere the layout's text can appear: the frosted
    # glass over a line stays inside it
    divider: Divider | None = None
    typing: float = 0.0  # characters a second the game types a line out (0: it appears
    # whole); the translation types out with it
    # set-region mode (a dialogue area the player chose, or a built-in one):
    keep: bool = False  # the glass stays once it is up (never goes between conversations)
    all_dialogue: bool = False  # every text read in it is dialogue (the player said so),
    # even on the game's own text box
    align: str = "center"  # center | left | auto: how the game lays its rows out (auto:
    # found from the first line), the translation and the glass follow
    reveal: bool = True  # the translation types in like the game's text (else at once)
    light_text: bool = False  # the game's text here is light (outlined): a dark colour read
    # for it is its outline, taken for the text over a bright scene

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "DialogueLayout":
        def reg(v: list[float] | None, label: str) -> Region | None:
            if not v:
                return None
            x, y, w, h = (float(t) for t in v)
            return Region(x, y, w, h, name=f"{name}_{label}", screen_type="dialogue")

        lo, hi = d["rows_top"]
        panel = reg(d.get("panel"), "panel")
        if panel is None:
            raise ValueError(f"dialogue layout {name!r} has no panel")
        div = d.get("divider")
        return cls(
            name=name,
            rows_top=(float(lo), float(hi)),
            panel=panel,
            anchor=str(d.get("anchor", "top")),
            speaker=reg(d.get("speaker"), "speaker"),
            glass=reg(d.get("glass"), "glass"),
            divider=(
                Divider(
                    float(div["y"]),
                    str(div["template"]),
                    int(div.get("height", 900)),
                    float(div.get("match", DIVIDER_MATCH)),
                    float(div.get("again", DIVIDER_AGAIN)),
                )
                if div
                else None
            ),
            typing=float(d.get("typing", 0.0)),
            keep=bool(d.get("keep", False)),
            all_dialogue=bool(d.get("all_dialogue", False)),
            align=str(d.get("align", "center")),
            reveal=bool(d.get("reveal", True)),
            light_text=bool(d.get("light_text", False)),
        )


@dataclass
class Profile:
    game: str
    display_name: str
    window: WindowMatch
    regions: dict[str, Region]
    ignore_regions: list[Region]
    ocr: dict[str, Any]
    raw: dict[str, Any]
    dialogue_layouts: dict[str, DialogueLayout] = field(default_factory=dict)

    @property
    def dialogue(self) -> Region | None:
        return self.regions.get(DIALOGUE_REGION)

    @property
    def speaker(self) -> Region | None:
        return self.regions.get(SPEAKER_REGIONS[0])

    def is_ignored(self, box: Region, threshold: float = 0.5) -> bool:
        """True if at least *threshold* of the box lies inside any ignore region."""
        return any(ig.overlap_fraction(box) >= threshold for ig in self.ignore_regions)

    @property
    def key_prompts(self) -> str | None:
        """How the game's text names a key it shows as an icon: a regular expression with
        an ``action`` group (db.lookup.Lookup's key_prompts); None when it has none."""
        value = self.raw.get("key_prompts")
        return str(value) if value else None

    @property
    def tickers(self) -> list[Region]:
        """Slots the game scrolls its text through (hud/ticker.py)."""
        return [r for r in self.regions.values() if r.ticker]

    def align_at(self, x: float, y: float) -> str:
        """The text alignment the profile declares for the area holding point (x, y)."""
        for r in self.regions.values():
            if r.align and r.contains_point(x, y):
                return r.align
        return ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Profile":
        regions = {k: Region.from_dict(k, v) for k, v in (d.get("regions") or {}).items()}
        ignores = [
            Region.from_dict(f"ignore_{i}", v) for i, v in enumerate(d.get("ignore_regions") or [])
        ]
        win = d.get("window") or {}
        layouts = {
            k: DialogueLayout.from_dict(k, v) for k, v in (d.get("dialogue_layouts") or {}).items()
        }
        return cls(
            game=str(d["game"]),
            display_name=str(d.get("display_name", d["game"])),
            window=WindowMatch(
                process_names=list(win.get("process_names") or []),
                title_patterns=list(win.get("title_patterns") or []),
                exact_titles=[str(t) for t in win.get("exact_titles") or []],
                aspect=float(win["aspect"]) if win.get("aspect") else None,
            ),
            regions=regions,
            ignore_regions=ignores,
            ocr=dict(d.get("ocr") or {}),
            raw=d,
            dialogue_layouts=layouts,
        )

    def without_dialogue(self) -> "Profile":
        """A copy with no dialogue handling: its dialogue and speaker regions and layouts
        gone, every text on screen covered alike (for games without a dialogue profile,
        or when the player wants plain covers everywhere)."""
        merged = dict(self.raw)
        merged["regions"] = {
            k: v
            for k, v in (self.raw.get("regions") or {}).items()
            if str((v or {}).get("screen_type", "")) not in DIALOGUE_REGIONS
        }
        merged["dialogue_layouts"] = {}
        return Profile.from_dict(merged)

    def with_overrides(self, d: dict[str, Any]) -> "Profile":
        """Return a copy with extra ignore regions / region overrides (e.g. for a recording)."""
        merged = dict(self.raw)
        merged["regions"] = {**(self.raw.get("regions") or {}), **(d.get("regions") or {})}
        merged["ignore_regions"] = list(self.raw.get("ignore_regions") or []) + list(
            d.get("ignore_regions") or []
        )
        return Profile.from_dict(merged)


def load_profile(game_or_path: str | Path) -> Profile:
    """Load ``games/<game>/profile.yaml`` or an explicit yaml path."""
    path = Path(game_or_path)
    if not path.suffix:
        path = GAMES_DIR / str(game_or_path) / "profile.yaml"
    with open(path, encoding="utf-8") as f:
        return Profile.from_dict(yaml.safe_load(f))


def load_overrides(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
