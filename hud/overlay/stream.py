"""The overlay as a source for streaming software (OBS): a page served on this computer only
(127.0.0.1) that shows what the overlay window shows, on a transparent background.

The overlay window is kept out of every screen capture, so the HUD never reads its own
translations; that keeps it out of OBS's captures too. Instead OBS adds this page as a
Browser Source, the size of the game's picture, above its own capture of the game: the
game at OBS's full frame rate, the translations over it exactly where the player sees them.

The page asks for the overlay again and again (a request waits until it changed, up to
:data:`WAIT_S`); each answer is the changed area as one PNG with straight alpha, at most
:data:`MAX_FPS` a second while text fades or types in. Nothing is sent anywhere: the
server listens on the loopback address and answers only requests addressed to it there
(a web page elsewhere cannot rebind a name to it).
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

log = logging.getLogger("hud")

DEFAULT_PORT = 47811
PORTS_TRIED = 10  # the next ports when one is taken
WAIT_S = 2.0  # a request for a newer overlay waits this long at most
POLL_S = 0.015  # how often a waiting request looks again
MAX_FPS = 30.0
HOST = "127.0.0.1"
NONE = "none"  # the version while there is no overlay

Layer = tuple[int, int, np.ndarray]  # x, y, premultiplied RGBA
Layers = list[tuple[Layer, float]]
Picture = tuple[tuple[int, int], tuple[int, int, int, int], bytes]  # size, rect, PNG

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Game Translation HUD</title>
<style>html,body{margin:0;background:transparent;overflow:hidden}
canvas{position:fixed;left:0;top:0;width:100vw;height:100vh}</style></head>
<body><canvas id="c" width="1920" height="1080"></canvas>
<script>
const c = document.getElementById("c"), g = c.getContext("2d");
const pause = ms => new Promise(r => setTimeout(r, ms));
let v = "";
async function loop() {
  for (;;) {
    try {
      const r = await fetch("frame?after=" + encodeURIComponent(v), {cache: "no-store"});
      if (r.status === 204) continue;
      if (r.status !== 200) { await pause(1000); continue; }
      v = r.headers.get("X-Version") || "";
      const [W, H] = r.headers.get("X-Size").split(",").map(Number);
      const [x, y, w, h] = r.headers.get("X-Rect").split(",").map(Number);
      const im = w > 0 ? await createImageBitmap(await r.blob()) : null;
      if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
      g.clearRect(0, 0, c.width, c.height);
      if (im) { g.drawImage(im, x, y); im.close(); }
    } catch (e) {
      v = ""; g.clearRect(0, 0, c.width, c.height); await pause(1000);
    }
  }
}
loop();
</script></body></html>
"""


Rect = tuple[int, int, int, int]


def flatten(layers: Layers, size: tuple[int, int]) -> tuple[Rect, np.ndarray] | None:
    """The layers (in the order the window paints them, each with its fade) on a clear
    picture of *size* (w, h): (the area they cover, its BGRA pixels with straight alpha,
    as a PNG holds them); None when nothing shows."""
    fw, fh = size
    parts = []
    for (x, y, rgba), fade in layers:
        if fade <= 0.0:
            continue
        h, w = rgba.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
        if x1 > x0 and y1 > y0:
            parts.append((x0, y0, x1, y1, rgba[y0 - y : y1 - y, x0 - x : x1 - x], fade))
    if not parts:
        return None
    bx0, by0 = min(p[0] for p in parts), min(p[1] for p in parts)
    bx1, by1 = max(p[2] for p in parts), max(p[3] for p in parts)
    out = np.zeros((by1 - by0, bx1 - bx0, 4), np.float32)  # premultiplied RGBA
    for x0, y0, x1, y1, px, fade in parts:
        src = px.astype(np.float32)
        if fade < 1.0:
            src *= fade
        dst = out[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0]
        dst *= 1.0 - src[:, :, 3:4] / 255.0
        dst += src
    alpha = out[:, :, 3:4]
    rgb = np.where(alpha > 0, out[:, :, :3] * 255.0 / np.maximum(alpha, 1e-3), 0.0)
    straight = np.concatenate([rgb, alpha], axis=2)
    rgba = np.clip(np.rint(straight), 0, 255).astype(np.uint8)
    return (bx0, by0, bx1 - bx0, by1 - by0), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


class _Server(ThreadingHTTPServer):
    """A port of its own: never shared with another program (Windows lets a second socket
    bind a port taken with SO_REUSEADDR, which http.server sets)."""

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class StreamServer:
    """Serves the overlay of ``source()`` (the overlay window, or None while the HUD is not
    running) at :attr:`url`. ``source`` is called on the server's threads: it must only
    read attributes (the window's :meth:`snapshot` is made for that)."""

    def __init__(self, source: Callable[[], Any]) -> None:
        self.source = source
        self.port = 0
        self._server: ThreadingHTTPServer | None = None
        self._lock = threading.Lock()
        self._made: tuple[str, Picture | None] | None = None  # (version, picture) last made
        self._made_at = 0.0

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/" if self._server is not None else ""

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self, port: int = DEFAULT_PORT) -> bool:
        """Listen on *port*, or one of the next when it is taken; False if none is free."""
        if self._server is not None:
            return True
        owner = self

        class Handler(_Handler):
            stream = owner

        for p in range(port, port + PORTS_TRIED):
            try:
                server = _Server((HOST, p), Handler)
            except OSError:
                continue
            self._server, self.port = server, int(server.server_address[1])  # (0: any)
            threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.2},
                name="stream-source",
                daemon=True,
            ).start()
            log.info("stream source at %s", self.url)
            return True
        log.warning("stream source: ports %d-%d are all taken", port, port + PORTS_TRIED - 1)
        return False

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()

    # ---- what is served ------------------------------------------------------------------
    def version(self) -> str:
        """What the overlay shows now, as a token (NONE while there is no overlay)."""
        overlay = self.source()
        if overlay is None:
            return NONE
        return _token(overlay, overlay.pixels_version)

    def picture(self) -> tuple[str, Picture | None]:
        """(version, the overlay as a PNG: (frame size, its area, the bytes); None when
        there is no overlay or no frame yet). Made once per version, for every page."""
        with self._lock:
            overlay = self.source()
            if overlay is None:
                self._made = (NONE, None)
                return self._made
            wait = 1.0 / MAX_FPS - (time.monotonic() - self._made_at)
            if wait > 0:
                time.sleep(wait)
            frame = overlay.newest_frame  # (its size is the game's picture)
            version, layers = overlay.snapshot()
            token = _token(overlay, version)
            if self._made is not None and self._made[0] == token:
                return self._made
            if frame is None:
                self._made = (token, None)
                return self._made
            size = (int(frame.shape[1]), int(frame.shape[0]))
            flat = flatten(layers, size)
            if flat is None:
                pic: Picture = (size, (0, 0, 0, 0), b"")
            else:
                rect, bgra = flat
                ok, png = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                pic = (size, rect, png.tobytes() if ok else b"")
            self._made, self._made_at = (token, pic), time.monotonic()
            return self._made


def _token(overlay: Any, version: int) -> str:
    return f"{id(overlay):x}.{version}.{int(bool(overlay.enabled))}"


class _Handler(BaseHTTPRequestHandler):
    stream: StreamServer

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        host = (self.headers.get("Host") or "").lower()
        if host not in (f"{HOST}:{self.stream.port}", f"localhost:{self.stream.port}"):
            self._send(403, b"", "text/plain")  # (a name rebound to this computer)
            return
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif url.path == "/frame":
            after = (parse_qs(url.query).get("after") or [""])[0]
            self._frame(after)
        else:
            self._send(404, b"", "text/plain")

    def _frame(self, after: str) -> None:
        end = time.monotonic() + WAIT_S
        while self.stream.version() == after:
            if time.monotonic() >= end:
                self._send(204, b"", "text/plain")
                return
            time.sleep(POLL_S)
        token, pic = self.stream.picture()
        if pic is None:  # no overlay (or no frame yet): nothing shows
            size, rect, png = (1, 1), (0, 0, 0, 0), b""
        else:
            size, rect, png = pic
        if token == after:
            self._send(204, b"", "text/plain")
            return
        self._send(200, png, "image/png", {
            "X-Version": token,
            "X-Size": f"{size[0]},{size[1]}",
            "X-Rect": ",".join(str(v) for v in rect),
        })  # fmt: skip

    def _send(
        self, status: int, body: bytes, kind: str, headers: dict[str, str] | None = None
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # (the page went away)
