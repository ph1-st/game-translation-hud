"""What lies under the translation: the Japanese painted over, and a faint glow.

No box is drawn anywhere. Two things are composed into one translucent image per item:

* **erase**: the Japanese strokes being replaced (and their outline) are painted over with
  what they sit on. Stroke pixels are found by local contrast and widened by a pixel or
  two. On a pill, a bar or a panel (the pixels between and around the glyphs that are not
  the text's colour fit one colour plane) they get that plane, gradient included, and
  nothing from beyond the pill's edge a pixel or two away. Over a scene they are inpainted
  from around them (at half resolution: the fill is softened anyway, and it costs a
  millisecond or two), so the stripes of a scene run on through them, and the painted area
  fades into the picture. In a wide area over a moving scene (a scrolling title's slot)
  only the text's own strokes are painted over: bright, about its colour, flanked by its
  outline, in dense rows; the scene in between stays as it is. Where a longer translation
  reaches past its Japanese, marks in the text's colour (a dash the OCR did not read) are
  painted over too, and on a pill whose translation would spill onto another colour, the
  pill is drawn longer under it.
  Built from the current frame; the overlay rebuilds it when the pixels under it change
  (a scene moving behind a key hint), so it never shows a stale picture.
* **glow**: a faint shine of one constant colour over the translated lines and the
  Japanese they replace: their area, blurred into a soft bloom, so it has no edge and is
  never a rectangle; rows of different widths melt into one cloud.

Dialogue lies on **frosted glass** instead (:func:`glass`): the scene behind blurred, a
crisp rounded pane with a light line along its edge.

The image is premultiplied RGBA; drawn over the screen it gives
``erased * (1 - shine_alpha) + shine * shine_alpha`` where it erases, and the screen with
the shine on it everywhere else.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np

from hud.ink import Rect, clip_rect, flanked_mask
from hud.items import OverlayItem, painted_align
from hud.layout import PAD_X, drawn_size
from hud.looks import Color
from hud.textwidth import text_width

ERASE_HP = 14  # local contrast (grey levels) that marks a pixel as part of a stroke
ERASE_GROW = 0.12  # strokes are widened by this share of a glyph height (min 1 px)
ERASE_LUM = 80  # where only the text is erased: grey levels a stroke may be off its fill
ERASE_DENSITY = 0.03  # ...and strokes come in rows: share of stroke pixels around one
ERASE_SOFT_HP = 6  # around a stroke, contrast this faint is still its edge (anti-aliasing)
OUTLINE_REACH = 0.2  # a stroke's outline lies within this many glyph heights of it
STROKE_BLUR = 0.6  # strokes stand out from their surroundings blurred this much (glyphs)
TYPING_AHEAD = 1.5  # ...and this many glyphs ahead (to the right): text that moves in
FILL_REACH = 0.15  # the inpainting looks this many glyph heights around (at half size)
FILL_SOFT = 0.12  # the fill is softened by this many glyph heights (sigma)
FILL_SPREAD = 0.25  # thin strokes are filled from the pixels this many glyph heights around
FADE = 0.3  # the painted area fades into the picture over this many glyph heights
TEXT_LIKE = 80  # pixels this close to the text's colour (RGB distance) are the text
PLANE_MIN = 12  # background pixels needed to tell a flat background
PLANE_COVER = 0.15  # ...and at least this share of the area
PLANE_SHARE = 0.6  # ...most of which fit one colour plane
PLANE_RESIDUAL = 6.0  # ...within this (RGB, root mean square): a pill, a bar, a panel
CONTAINER_STD = 10.0  # the Japanese sits on a pill or a button: its colour varies this little
CONTAINER_LIKE = 40  # pixels this close to that colour (sum over RGB) are the pill
CONTAINER_HALF = 0.5  # where the translation reaches out, this share of the pill's colour
# means the pill goes on there by itself
CONTAINER_REACH = 0.6  # the pill's rows are looked for this many glyph heights around
CONTAINER_ROW = 0.6  # ...rows with this share of its colour belong to it
CONTAINER_RADIUS = 0.25  # its drawn-on part has corners this round (of its height)
GLOW_PAD_X = 0.35  # the glow's core reaches this many glyph heights left and right of a line
GLOW_PAD_Y = 0.1  # ...and above and below its glyph band
GLOW_BLUR = 0.5  # ...and is blurred by this many glyph heights (sigma)
GLOW_GAIN = 1.25  # the blurred core is this much brighter, so its middle is full strength
GLASS_BLUR = 1.0  # behind a pane of frosted glass the scene is blurred by this many glyph
# heights (made from this much of the scene around it too)
GLASS_RADIUS = 0.5  # the pane's corners are this round (glyph heights)
GLASS_EDGE = 1.5  # its edge is this soft, in pixels: a crisp pane, not a fade (a fade mixes
# the blur, made from a frame behind, with the scene of this moment: that shimmers)
GLASS_BORDER = 1.5  # a light line along the edge, this many pixels wide...
GLASS_BORDER_ALPHA = 0.4  # ...this strong
GLASS_WORK_SCALE = 4  # the scene behind the glass is worked on at this fraction of its size
GLASS_MEDIAN = 5  # ...where a median of this many pixels takes thin strokes out


def glow_rects(item: OverlayItem) -> list[Rect]:
    """The area each line of *item* takes (x from its alignment, the Japanese row band it
    sits on), padded: the shapes the glow hugs."""
    out: list[Rect] = []
    x, _y, w, _h = item.rect
    g = max(1.0, item.row_h)
    px, py = GLOW_PAD_X * g, GLOW_PAD_Y * g
    size = drawn_size(item.font_px, item.stretch)
    align = painted_align(item)
    for i, line in enumerate(item.lines):
        tw = text_width(line, size)
        if align == "left":
            tx = x + PAD_X
        elif align == "right":
            tx = x + w - PAD_X - tw
        else:
            tx = x + (w - tw) / 2.0
        top = item.row_top + i * item.pitch
        out.append(
            (
                int(round(tx - px)),
                int(round(top - py)),
                int(round(tw + 2 * px)),
                int(round(g + 2 * py)),
            )
        )
    return out


def glow_alpha(
    shape: tuple[int, int], origin: tuple[int, int], rects: list[Rect], blur: float
) -> np.ndarray:
    """The glow's coverage (0-1) on an image of *shape* at *origin*: the *rects* filled
    and blurred by *blur* pixels, a bloom with no edge."""
    core = np.zeros(shape, np.float32)
    ox, oy = origin
    for x, y, w, h in rects:
        core[max(0, y - oy) : max(0, y + h - oy), max(0, x - ox) : max(0, x + w - ox)] = 1.0
    a = _blur(core, max(1.0, blur))
    peak = float(a.max())
    if peak <= 1e-6:
        return a
    return np.minimum(1.0, GLOW_GAIN * a / peak)  # full strength in its middle, any blur


_GREY = np.array([0.299, 0.587, 0.114], np.float32)


def text_strokes(
    grey: np.ndarray,
    hp: np.ndarray,
    region: np.ndarray,
    glyph: float,
    fill: Color,
    edge: Color | None,
) -> np.ndarray:
    """The strokes of text drawn in *fill* (outlined in *edge*) in a wide area over a scene,
    and nothing else of the scene: pixels on the text's side of their surroundings, about
    as bright as the fill, flanked on both sides by the outline when the text has one (a
    bright sleeve or a patch of sand has its dark side on one side only), in rows dense
    with such pixels (a lone highlight is not text)."""
    lum = float(np.array(fill, np.float32) @ _GREY)
    light = lum >= 128
    if light:
        cand = (hp > ERASE_HP) & (grey >= lum - ERASE_LUM)
        other = hp < -ERASE_HP
    else:
        cand = (hp < -ERASE_HP) & (grey <= lum + ERASE_LUM)
        other = hp > ERASE_HP
    cand &= region
    if edge is not None:
        cand = flanked_mask(cand, other, max(2, int(round(OUTLINE_REACH * glyph))))
    win = (max(3, int(3 * glyph)) | 1, max(3, int(glyph)) | 1)
    dense = cv2.boxFilter(cand.astype(np.float32), -1, win, normalize=True) >= ERASE_DENSITY
    return cand & dense


def text_mask(
    frame: np.ndarray, rect: Rect, glyph: float, fill: Color, edge: Color | None
) -> tuple[int, int, np.ndarray] | None:
    """(x, y, mask) of the strokes of text drawn in *fill* (and outlined in *edge*) in
    *rect* of the BGR *frame*; None when the area is off the frame."""
    fh, fw = frame.shape[:2]
    x, y, w, h = clip_rect(rect, fw, fh)
    if w < 4 or h < 4:
        return None
    grey = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = grey - _blur(grey, max(2.0, STROKE_BLUR * glyph))
    region = np.ones(grey.shape, bool)
    return x, y, text_strokes(grey, hp, region, glyph, fill, edge)


def text_pixels(
    frame: np.ndarray, rect: Rect, glyph: float, fill: Color, edge: Color | None
) -> int:
    """How many pixels of *rect* (BGR *frame*) are strokes of text drawn in *fill* (and
    outlined in *edge*): is there such text there at all?"""
    got = text_mask(frame, rect, glyph, fill, edge)
    return 0 if got is None else int(got[2].sum())


def erase(
    rgb: np.ndarray,
    region: np.ndarray,
    glyph: float,
    fill: Color,
    edge: Color | None = None,
    only_text: bool = False,
    marks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(the crop with the strokes in *region* painted over, how much of each pixel was
    painted over 0-1) for text drawn in *fill* (outlined in *edge*). *rgb* is float32.
    Every stroke in *region* is painted over, or, *only_text*, only that text's strokes and
    the pixel or two around them: for an area wider than the text, over a scene. In
    *marks* (where a longer translation reaches beyond its Japanese), strokes in the text's
    colour are painted over too: a dash or dots of the line the OCR did not read."""
    grey = rgb @ _GREY
    hp = grey - _blur(grey, max(2.0, STROKE_BLUR * glyph))
    grow = max(1, int(round(ERASE_GROW * glyph)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    if only_text:
        mask = _text_only(grey, hp, region, glyph, fill, edge, grow)
    else:
        mask = cv2.dilate(((np.abs(hp) > ERASE_HP) & region).astype(np.uint8), k) > 0
    mask &= region
    if marks is not None and marks.any():
        lum = float(np.array(fill, np.float32) @ _GREY)
        side = hp > ERASE_HP if lum >= 128 else hp < -ERASE_HP
        d = rgb - np.array(fill, np.float32)
        own = np.einsum("ijk,ijk->ij", d, d) < TEXT_LIKE * TEXT_LIKE
        mask |= (cv2.dilate((side & own & marks).astype(np.uint8), k) > 0) & marks
        region = region | marks
    if not mask.any():
        return rgb, np.zeros(region.shape, np.float32)
    if not only_text:
        # what the Japanese sits on, judged on the pixels between and around its glyphs
        # that are not its colour (nor its outline or anti-aliasing, a pixel or two from
        # it): inside a pill the gaps between the glyphs are the pill, behind a name over
        # a scene they are the scene
        cover = region & ~marks if marks is not None else region
        plane = _plane(rgb, cover & ~_near_text(rgb, fill), cover)
        if plane is not None:
            # a pill, a bar, a panel: its own colour (gradient included) and nothing from
            # beyond its edge, which may be a pixel or two away
            m = np.maximum(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.7), mask)
            m = m * region
            return rgb * (1.0 - m[:, :, None]) + plane * m[:, :, None], m
    return _filled(rgb, mask, glyph, only_text)


def _text_only(
    grey: np.ndarray,
    hp: np.ndarray,
    region: np.ndarray,
    glyph: float,
    fill: Color,
    edge: Color | None,
    grow: int,
) -> np.ndarray:
    """What to paint over of text in a wide area: around the strokes found, everything
    with some contrast (the outline, the anti-aliasing), then a pixel more, and a little
    ahead of them: text that moves in (a scrolling title) lands right of its last
    characters, under paint already there (the picture is a frame behind)."""
    core = text_strokes(grey, hp, region, glyph, fill, edge)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * grow + 1, 4 * grow + 1))
    mask = (cv2.dilate(core.astype(np.uint8), k2) > 0) & (np.abs(hp) > ERASE_SOFT_HP)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ahead = max(2, int(round(TYPING_AHEAD * glyph)))
    kernel = np.ones((1, ahead), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, anchor=(ahead - 1, 0)) > 0


def _filled(
    rgb: np.ndarray, mask: np.ndarray, glyph: float, only_text: bool
) -> tuple[np.ndarray, np.ndarray]:
    """*mask* filled in from around it (only around it: an area may be wide), faded into
    the picture over a few pixels beyond it (there the fill is the picture itself, softened:
    no edge, nothing new). Thin strokes over a scene are filled from the gaps between them;
    a label's dense block of glyphs needs the structure around it carried in."""
    fade = fade_px(glyph)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pad = int(np.ceil(fade + 2 * max(2.0, FILL_REACH * glyph) + 2))
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    sub = (slice(y0, y1), slice(x0, x1))
    spread = _spread if only_text else _inpaint
    filled = spread(rgb[sub], mask[sub], glyph)
    t = np.clip(cv2.distanceTransform((~mask[sub]).astype(np.uint8), cv2.DIST_L2, 3) / fade, 0, 1)
    ms = 1.0 - t * t * (3.0 - 2.0 * t)
    plate = rgb.copy()
    plate[sub] = rgb[sub] * (1.0 - ms[:, :, None]) + filled * ms[:, :, None]
    m = np.zeros(mask.shape, np.float32)
    m[sub] = ms
    return plate, m


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur; a wide one is done at half size (the same picture, a quarter of
    the work), a very wide one (large text at 1440p and 4K) smaller still."""
    if sigma < 4.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    f = max(2, int(sigma // 6))
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(
        cv2.GaussianBlur(small, (0, 0), sigma / f), (w, h), interpolation=cv2.INTER_LINEAR
    )


def _work_scale(glyph: float) -> int:
    """How much smaller the fill is worked out: half size for text up to ~31 px tall (all
    of the game's text at 1080p, as tuned), smaller for larger text, so the work per
    glyph stays about the same at 1440p and 4K (the fill is softened anyway)."""
    return max(2, int(round(glyph / 12.5)))


def _spread(rgb: np.ndarray, mask: np.ndarray, glyph: float) -> np.ndarray:
    """The crop with *mask* filled in from the pixels around it (float32 RGB): a
    normalized blur at a reduced size, where only pixels untouched by the mask count."""
    h, w = mask.shape
    q = _work_scale(glyph)
    sw, sh = max(4, (w + q - 1) // q), max(4, (h + q - 1) // q)
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    valid = (
        cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) < 0.01
    ).astype(np.float32)
    s = max(1.0, FILL_SPREAD * glyph / q)
    num = cv2.GaussianBlur(small * valid[:, :, None], (0, 0), s)
    den = cv2.GaussianBlur(valid, (0, 0), s)
    fill = num / np.maximum(den, 1e-4)[:, :, None]
    sparse = den < 0.05
    if sparse.any():  # nothing untouched nearby: look further out
        num = cv2.GaussianBlur(small * valid[:, :, None], (0, 0), 3 * s)
        den = cv2.GaussianBlur(valid, (0, 0), 3 * s)
        fill = np.where(sparse[:, :, None], num / np.maximum(den, 1e-4)[:, :, None], fill)
    up = cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(up, (0, 0), max(1.0, FILL_SOFT * glyph))


def _near_text(rgb: np.ndarray, fill: Color) -> np.ndarray:
    """Pixels in the text's colour and the two pixels around them (its outline, its
    anti-aliasing): what is left is what the text sits on."""
    d = rgb - np.array(fill, np.float32)
    textlike = np.einsum("ijk,ijk->ij", d, d) < TEXT_LIKE * TEXT_LIKE
    return cv2.dilate(textlike.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def fade_px(glyph: float) -> float:
    return max(1.0, FADE * glyph)


def _plane(rgb: np.ndarray, sel: np.ndarray, region: np.ndarray) -> np.ndarray | None:
    """The background of the Japanese as a colour plane (flat or a gradient), fitted to the
    background pixels *sel* of *region*; None when they are not that (a scene)."""
    if int(sel.sum()) < max(PLANE_MIN, PLANE_COVER * int(region.sum())):
        return None
    ys, xs = np.nonzero(sel)
    a = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float32)
    vals = rgb[sel]
    coef = np.linalg.lstsq(a, vals, rcond=None)[0]
    err = np.abs(vals - a @ coef).max(axis=1)
    keep = err <= max(3.0, 2.5 * float(np.median(err)))  # stray anti-aliased pixels
    if keep.sum() < max(PLANE_MIN, PLANE_SHARE * len(err)):
        return None
    coef = np.linalg.lstsq(a[keep], vals[keep], rcond=None)[0]
    res = vals[keep] - a[keep] @ coef
    if float(np.sqrt(np.mean(res * res))) > PLANE_RESIDUAL:
        return None
    h, w = region.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx[:, :, None] * coef[0] + yy[:, :, None] * coef[1] + coef[2]


def _inpaint(rgb: np.ndarray, mask: np.ndarray, glyph: float) -> np.ndarray:
    """The crop with *mask* filled in from around it (float32 RGB): inpainted at a reduced
    size (half for the game at 1080p, see :func:`_work_scale`), scaled back and softened."""
    h, w = mask.shape
    q = _work_scale(glyph)
    sw, sh = max(4, (w + q - 1) // q), max(4, (h + q - 1) // q)
    small = cv2.resize(
        np.clip(rgb, 0, 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA
    )
    sm = cv2.resize(mask.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_AREA) > 0
    sm = cv2.dilate(sm.astype(np.uint8), np.ones((3, 3), np.uint8))
    # (the same reach in screen pixels at any working size: tuned at half size)
    radius = max(2, int(round(FILL_REACH * glyph * 2.0 / q)))
    filled = cv2.inpaint(small, sm, radius, cv2.INPAINT_TELEA)
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return cv2.GaussianBlur(up, (0, 0), max(1.0, FILL_SOFT * glyph))


def build(
    frame: np.ndarray | None, item: OverlayItem, shine: Color
) -> tuple[int, int, np.ndarray] | None:
    """(x, y, premultiplied RGBA uint8) for *item* over *frame* (BGR); None when there is
    nothing to draw. Without a frame, only the glow (an item that erases needs one)."""
    if item.frost is not None:
        if frame is None:
            return None
        return glass(frame, item.frost, max(1.0, item.row_h), shine)
    rects = glow_rects(item) if item.glow > 0 and item.lines else []
    if item.glow > 0 and item.under is not None:
        rects.append(item.under)  # the shine also lies on the Japanese painted over
    blur = GLOW_BLUR * max(1.0, item.row_h)
    area = _area(frame, item, rects, blur)
    if area is None:
        return None
    x0, y0, cw, ch = area
    s = item.glow * glow_alpha((ch, cw), (x0, y0), rects, blur) if rects else None
    if item.erase is not None and frame is not None:
        plate, m = _erased(frame, item, area)
        return _composed(plate, m, s if s is not None else np.zeros((ch, cw), np.float32),
                         shine, x0, y0)  # fmt: skip
    if s is None:
        return None
    return _composed(None, None, s, shine, x0, y0)


def _area(
    frame: np.ndarray | None, item: OverlayItem, rects: list[Rect], blur: float
) -> Rect | None:
    """The part of the screen an item's backdrop covers: its glow with the blur's reach,
    the Japanese it paints over with the fade's, and the box of its text (a pill may have
    to grow under it)."""
    reach = 2.5 * blur
    parts: list[Rect] = [
        (int(np.floor(x - reach)), int(np.floor(y - reach)),
         int(np.ceil(w + 2 * reach)), int(np.ceil(h + 2 * reach)))
        for x, y, w, h in rects
    ]  # fmt: skip
    if item.erase is not None and frame is not None:
        ex, ey, ew, eh = item.erase
        f = int(np.ceil(fade_px(max(1.0, item.row_h))))
        parts.append((ex - f, ey - f, ew + 2 * f, eh + 2 * f))
        if item.lines:
            parts.append(item.rect)
    if not parts:
        return None
    x0, y0 = min(p[0] for p in parts), min(p[1] for p in parts)
    x1, y1 = max(p[0] + p[2] for p in parts), max(p[1] + p[3] for p in parts)
    if frame is not None:
        fh, fw = frame.shape[:2]
        x0, y0, cw, ch = clip_rect((x0, y0, x1 - x0, y1 - y0), fw, fh)
    else:
        cw, ch = x1 - x0, y1 - y0
    return (x0, y0, cw, ch) if cw >= 2 and ch >= 2 else None


def _erased(frame: np.ndarray, item: OverlayItem, area: Rect) -> tuple[np.ndarray, np.ndarray]:
    """(the area with the item's Japanese painted over, how much of each pixel was)."""
    x0, y0, cw, ch = area
    rgb = frame[y0 : y0 + ch, x0 : x0 + cw, ::-1].astype(np.float32)
    region = _rect_mask((ch, cw), item.erase, x0, y0)
    marks = None
    if item.lines and not item.gate:
        marks = _rect_mask((ch, cw), item.rect, x0, y0) & ~region
    glyph = max(1.0, item.row_h)
    plate, m = erase(rgb, region, glyph, item.fg, item.edge, item.gate, marks)
    if not item.gate and item.lines:
        plate, m = _extend_container(rgb, plate, m, region, item, (x0, y0))
    return plate, m


def _rect_mask(shape: tuple[int, int], rect: Rect, x0: int, y0: int) -> np.ndarray:
    """A mask of *shape* (an image at x0, y0) that is True inside the frame rect *rect*."""
    out = np.zeros(shape, bool)
    x, y, w, h = rect
    out[max(0, y - y0) : max(0, y + h - y0), max(0, x - x0) : max(0, x + w - x0)] = True
    return out


def _composed(
    plate: np.ndarray | None,
    m: np.ndarray | None,
    shine_alpha: np.ndarray,
    shine: Color,
    x0: int,
    y0: int,
) -> tuple[int, int, np.ndarray] | None:
    """The premultiplied RGBA image of a painted-over area (*plate* where *m*) under a
    shine, cut to where anything is drawn (a wide area paints a few rows)."""
    shown = shine_alpha > 1e-3 if m is None else (m > 1e-3) | (shine_alpha > 1e-3)
    if not shown.any():
        return None
    ys, xs = np.nonzero(shown)
    b0, b1, a0, a1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    sa = shine_alpha[b0:b1, a0:a1]
    white = np.array(shine, np.float32)
    out = np.empty((b1 - b0, a1 - a0, 4), np.float32)
    if m is None or plate is None:
        out[:, :, :3] = sa[:, :, None] * white
        out[:, :, 3] = sa
    else:
        mm = m[b0:b1, a0:a1]
        out[:, :, :3] = (mm * (1.0 - sa))[:, :, None] * plate[b0:b1, a0:a1] + sa[:, :, None] * white
        out[:, :, 3] = 1.0 - (1.0 - mm) * (1.0 - sa)
    out[:, :, 3] *= 255.0
    return x0 + a0, y0 + b0, np.clip(out + 0.5, 0, 255).astype(np.uint8)


def glass(
    frame: np.ndarray, rect: Rect, glyph: float, shine: Color
) -> tuple[int, int, np.ndarray] | None:
    """(x, y, premultiplied RGBA uint8) of a pane of frosted glass over *rect* of *frame*
    (BGR): the scene behind it with thin strokes taken out (a median at quarter size) and
    blurred, solid to a crisp rounded edge, a light line along the edge. Whatever comes in
    under it is never seen, and that the picture it is made from is a frame behind does not
    show in a blur; nothing outside it is touched."""
    fh, fw = frame.shape[:2]
    m = int(np.ceil(GLASS_BLUR * glyph))
    bx, by, bw, bh = clip_rect((rect[0] - m, rect[1] - m, rect[2] + 2 * m, rect[3] + 2 * m), fw, fh)
    x, y, w, h = clip_rect((rect[0] - 1, rect[1] - 1, rect[2] + 2, rect[3] + 2), fw, fh)
    if w < 8 or h < 8:
        return None
    q = GLASS_WORK_SCALE
    small = cv2.resize(
        frame[by : by + bh, bx : bx + bw], (max(4, bw // q), max(4, bh // q)),
        interpolation=cv2.INTER_AREA,
    )  # fmt: skip
    small = cv2.medianBlur(small, GLASS_MEDIAN).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, GLASS_BLUR * glyph / q))
    pic = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR)
    pic = pic[y - by : y - by + h, x - bx : x - bx + w, ::-1]  # RGB, the pane's area
    keep, add, alpha = _pane_shape(rect, (x, y, w, h), glyph, shine)
    out = np.empty((h, w, 4), np.uint8)
    np.clip(pic * keep + add, 0, 255, out=pic)
    out[:, :, :3] = pic
    out[:, :, 3] = alpha
    return x, y, out


_SHAPES: OrderedDict = OrderedDict()  # a pane's shape: the scene behind changes, it does not
_SHAPES_LOCK = threading.Lock()
SHAPES_KEPT = 64


def _pane_shape(
    rect: Rect, area: Rect, glyph: float, shine: Color
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What a pane over *area* does to the scene's pixels: (keep, add, alpha) with
    ``rgb = scene * keep + add`` (premultiplied) and alpha 0-255. The pane, its edge
    anti-aliased, a light line along its outline. Kept for the panes on screen: rebuilt 60
    times a second, they only change with the scene."""
    key = (tuple(rect), tuple(area), round(float(glyph), 3), tuple(shine))
    with _SHAPES_LOCK:
        got = _SHAPES.get(key)
        if got is not None:
            _SHAPES.move_to_end(key)
            return got
    x, y, w, h = area
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    radius = GLASS_RADIUS * glyph
    sd = _sd_box(xs, ys, (rect[0] - x, rect[1] - y, rect[2], rect[3]), radius)
    a = np.clip(0.5 - sd / GLASS_EDGE, 0.0, 1.0)
    d = -sd  # how far inside the pane
    line = np.clip(GLASS_BORDER + 0.5 - d, 0.0, 1.0) * np.clip(d + 0.5, 0.0, 1.0)
    b = (GLASS_BORDER_ALPHA * line)[:, :, None]
    keep = ((1.0 - b) * a[:, :, None]).astype(np.float32)
    add = (np.array(shine, np.float32) * b * a[:, :, None] + 0.5).astype(np.float32)
    alpha = np.clip(255.0 * a + 0.5, 0, 255).astype(np.uint8)
    got = (keep, add, alpha)
    with _SHAPES_LOCK:
        _SHAPES[key] = got
        while len(_SHAPES) > SHAPES_KEPT:
            _SHAPES.popitem(last=False)
    return got


def _sd_box(xs: np.ndarray, ys: np.ndarray, rect: Rect, radius: float) -> np.ndarray:
    """Signed distance to a rounded rectangle (negative inside)."""
    x, y, w, h = rect
    hx, hy = w / 2.0, h / 2.0
    r = min(radius, hx, hy)
    qx = np.abs(xs - (x + hx)) - (hx - r)
    qy = np.abs(ys - (y + hy)) - (hy - r)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    return outside + np.minimum(np.maximum(qx, qy), 0.0) - r


def _extend_container(
    rgb: np.ndarray,
    plate: np.ndarray,
    m: np.ndarray,
    region: np.ndarray,
    item: OverlayItem,
    origin: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """A translation longer than its Japanese, on a pill or a button: where it reaches past
    the flat colour the Japanese sits on (white text on a dark pill, out onto a white
    panel), the pill is drawn longer under it, in its own colour, so the translation stays
    as readable as the Japanese was. On a scene, or when it stays on the same colour,
    nothing changes."""
    ox, oy = origin
    ch, cw = m.shape
    g = max(1.0, item.row_h)
    bg = region & ~_near_text(rgb, item.fg)
    if bg.sum() < 12:
        return plate, m
    px = rgb[bg]
    c = np.median(px, axis=0)
    if float(np.std(px, axis=0).max()) > CONTAINER_STD:
        return plate, m
    bx, _by, bw, _bh = item.rect
    ex, ey, ew, eh = item.erase
    bx0, bx1 = max(0, bx - ox), min(cw, bx + bw - ox)
    ex0, ex1 = max(0, ex - ox), min(cw, ex + ew - ox)
    ey0, ey1 = max(0, ey - oy), min(ch, ey + eh - oy)
    if bx0 >= ex0 and bx1 <= ex1:
        return plate, m
    like = np.abs(rgb - c).sum(axis=2) < CONTAINER_LIKE
    side = np.zeros((ch, cw), bool)
    side[ey0:ey1, bx0 : max(bx0, ex0)] = True
    side[ey0:ey1, min(bx1, ex1) : bx1] = True
    if not side.any() or like[side].mean() >= CONTAINER_HALF:
        return plate, m
    # the container's rows: as far as its colour goes above and below the text
    rowlike = (like[:, ex0:ex1] | (m[:, ex0:ex1] > 0.05)).mean(axis=1)
    reach = CONTAINER_REACH * g
    lo, hi = max(0, int(ey0 - reach)), min(ch, int(ey1 + reach))
    top, bottom = ey0, ey1
    while top > lo and rowlike[top - 1] >= CONTAINER_ROW:
        top -= 1
    while bottom < hi and rowlike[bottom] >= CONTAINER_ROW:
        bottom += 1
    ys, xs = np.mgrid[0:ch, 0:cw].astype(np.float32) + 0.5
    left, right = min(bx0, ex0), max(bx1, ex1)
    radius = CONTAINER_RADIUS * (bottom - top)
    sd = _sd_box(xs, ys, (left, top, right - left, bottom - top), radius)
    a = np.clip(0.5 - sd, 0.0, 1.0)
    plate = plate * (1.0 - a[:, :, None]) + c[None, None, :] * a[:, :, None]
    return plate, np.maximum(m, a)


def signature(frame: np.ndarray, rect: Rect) -> np.ndarray | None:
    """A small picture of the pixels under *rect*, to tell when an erase must be rebuilt."""
    fh, fw = frame.shape[:2]
    x, y, w, h = clip_rect(rect, fw, fh)
    if w < 2 or h < 2:
        return None
    crop = frame[y : y + h, x : x + w]
    return cv2.resize(crop, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA).astype(
        np.int16
    )
