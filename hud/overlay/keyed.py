"""The overlay drawn with a colour key instead of per-pixel alpha, for a system that will
not keep a translucent window out of the screen capture: Windows 10 refuses a window drawn
with ``UpdateLayeredWindow`` (what Qt's translucent windows are) but keeps a colour-keyed
one out.

The window is then opaque. Every pixel the overlay leaves clear is the key colour, which
the desktop leaves out, so the game shows there, live. Where the overlay draws, its layers
are blended here onto the newest captured frame, the way the desktop blends a translucent
window onto the screen. The one difference: under the overlay's soft edges (a glow, a
pane of glass) the game is as it was in the newest frame, a capture tick ago.
"""

from __future__ import annotations

import cv2
import numpy as np

KEY = (255, 0, 255)  # RGB, never drawn as it is: an overlay pixel that would be is nudged
SHOWN = 3  # a layer's pixel at least this opaque (of 255, with its fade) is the overlay's

Layer = tuple[int, int, np.ndarray]  # x, y, premultiplied RGBA
Layers = list[tuple[Layer, float]]  # each with its fade
Tile = tuple[int, int, np.ndarray]  # x, y, BGRA: 255 where the overlay draws, 0 elsewhere


def prepare_layer(rgba: np.ndarray, fade: float) -> tuple[np.ndarray, np.ndarray]:
    """(premultiplied BGR, 255 - alpha per channel) of a layer at *fade*, as bytes: on a
    picture the layer is then ``picture * through / 255 + colour``."""
    faded = rgba if fade >= 1.0 else cv2.multiply(rgba, (fade, fade, fade, fade))
    colour = cv2.cvtColor(faded, cv2.COLOR_RGBA2BGR)
    through = 255 - faded[:, :, 3]
    return colour, cv2.merge([through, through, through])


def overlapping(boxes: list[tuple[int, int, int, int]]) -> list[list[int]]:
    """The indices of *boxes* (x0, y0, x1, y1) in groups that overlap, directly or through
    others; each group in the order given."""
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j in range(i):
            b = boxes[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


class KeyedComposer:
    """Tiles for a colour-keyed overlay window: each group of overlapping layers blended
    onto the frame under it, in the order the window paints them. What a layer needs for
    that is made once and kept while the overlay shows it with the same fade, and a tile is
    kept while its layers and the scene under it stay the same."""

    def __init__(self) -> None:
        self._ready: dict[int, tuple] = {}
        self._made: dict[tuple, tuple[np.ndarray, Tile]] = {}  # group -> (scene, tile)

    def tiles(self, frame: np.ndarray, layers: Layers) -> list[Tile]:
        fh, fw = frame.shape[:2]
        placed = []
        ready: dict[int, tuple] = {}
        for (x, y, rgba), fade in layers:
            h, w = rgba.shape[:2]
            box = (max(0, x), max(0, y), min(fw, x + w), min(fh, y + h))
            if fade <= 0.0 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            got = self._ready.get(id(rgba))
            if got is None or got[0] is not rgba or got[1] != fade:
                colour, through = prepare_layer(rgba, fade)
                drawn = cv2.compare(255 - through[:, :, 0], SHOWN - 1, cv2.CMP_GT)  # 255/0
                got = (rgba, fade, colour, through, drawn)
            ready[id(rgba)] = got
            placed.append((x, y, box, got))
        self._ready = ready  # layers no longer shown are let go
        made: dict[tuple, tuple[np.ndarray, Tile]] = {}
        out: list[Tile] = []
        for group in overlapping([p[2] for p in placed]):
            x0 = min(placed[i][2][0] for i in group)
            y0 = min(placed[i][2][1] for i in group)
            x1 = max(placed[i][2][2] for i in group)
            y1 = max(placed[i][2][3] for i in group)
            scene = frame[y0:y1, x0:x1]
            key = (x0, y0, x1, y1, *((id(placed[i][3][0]), placed[i][3][1]) for i in group))
            old = self._made.get(key)
            if old is not None and np.array_equal(old[0], scene):
                tile = old[1]  # the same layers on the same scene
            else:
                tile = (x0, y0, self._blend(scene, [placed[i] for i in group], x0, y0))
            made[key] = (scene, tile)
            out.append(tile)
        self._made = made
        return out

    @staticmethod
    def _blend(scene: np.ndarray, group: list, x0: int, y0: int) -> np.ndarray:
        bgr = scene.copy()
        mask = np.zeros(bgr.shape[:2], np.uint8)
        for x, y, (bx0, by0, bx1, by1), (_rgba, _fade, colour, through, drawn) in group:
            d = (slice(by0 - y0, by1 - y0), slice(bx0 - x0, bx1 - x0))
            s = (slice(by0 - y, by1 - y), slice(bx0 - x, bx1 - x))
            under = cv2.multiply(bgr[d], through[s], scale=1.0 / 255.0)
            bgr[d] = cv2.add(under, colour[s])
            mask[d] = cv2.bitwise_or(mask[d], drawn[s])
        # an overlay pixel that came out in the key colour would be a hole in it
        hole = cv2.bitwise_and(cv2.inRange(bgr, KEY[::-1], KEY[::-1]), mask)
        if cv2.countNonZero(hole):
            bgr[:, :, 1][hole > 0] = KEY[1] + 1
        tile = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        tile[:, :, 3] = mask
        return cv2.bitwise_and(tile, tile, mask=mask)  # (premultiplied: nothing where clear)
