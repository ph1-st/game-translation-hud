"""The overlay as a page for OBS: its layers flattened onto a clear picture, served on
127.0.0.1 only, a request waiting until the overlay changed."""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from hud.overlay.stream import NONE, StreamServer, flatten


def layer(x: int, y: int, w: int, h: int, rgb: tuple[int, int, int], a: int):
    px = np.zeros((h, w, 4), np.uint8)
    px[:, :, :3] = [round(c * a / 255) for c in rgb]  # premultiplied
    px[:, :, 3] = a
    return (x, y, px)


def test_layers_are_flattened_onto_a_clear_picture_with_straight_alpha():
    assert flatten([], (100, 50)) is None
    glow = layer(10, 10, 20, 10, (255, 255, 255), 64)
    text = layer(15, 12, 10, 4, (255, 0, 0), 255)
    off = layer(-5, 40, 10, 20, (0, 0, 255), 255)  # partly outside the picture
    rect, bgra = flatten([(glow, 1.0), (text, 1.0), (off, 0.5)], (100, 50))
    assert rect == (0, 10, 30, 40)  # the area they cover, clipped to the picture
    b, g, r, a = (int(v) for v in bgra[12 - 10, 15 - 0])  # the text over the glow
    assert (r, g, b, a) == (255, 0, 0, 255)
    b, g, r, a = (int(v) for v in bgra[11 - 10, 11 - 0])  # the glow alone
    assert (r, g, b) == (255, 255, 255) and a == 64
    b, g, r, a = (int(v) for v in bgra[45 - 10, 2])  # half faded
    assert (r, g, b) == (0, 0, 255) and abs(a - 128) <= 1
    assert bgra[25 - 10, 2, 3] == 0  # nothing there: clear


class FakeOverlay:
    """What the server reads of the overlay window."""

    def __init__(self) -> None:
        self.pixels_version = 1
        self.enabled = True
        self.newest_frame = np.zeros((90, 160, 3), np.uint8)
        self.layers = [(layer(20, 30, 40, 10, (255, 255, 0), 255), 1.0)]

    def snapshot(self):
        return self.pixels_version, list(self.layers) if self.enabled else []


@pytest.fixture
def served(monkeypatch):
    import hud.overlay.stream as stream

    monkeypatch.setattr(stream, "WAIT_S", 0.3)
    box = {"overlay": FakeOverlay()}
    server = StreamServer(lambda: box["overlay"])
    assert server.start(0) and server.url.startswith("http://127.0.0.1:")
    yield server, box
    server.stop()
    assert not server.running


def get(url: str, host: str | None = None):
    req = urllib.request.Request(url, headers={"Host": host} if host else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), b""


def test_the_page_gets_the_overlay_each_time_it_changes(served):
    server, box = served
    status, headers, body = get(server.url)
    assert status == 200 and b"<canvas" in body and b"frame?after=" in body
    status, headers, png = get(server.url + "frame?after=")
    assert status == 200 and headers["X-Size"] == "160,90" and headers["X-Rect"] == "20,30,40,10"
    pic = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert pic.shape == (10, 40, 4) and tuple(int(v) for v in pic[5, 5]) == (0, 255, 255, 255)
    version = headers["X-Version"]
    t0 = time.monotonic()
    status, _h, _b = get(server.url + "frame?after=" + version)  # nothing changed
    assert status == 204 and time.monotonic() - t0 >= 0.25

    def change() -> None:
        time.sleep(0.1)
        box["overlay"].enabled = False  # the translations hidden
        box["overlay"].pixels_version += 1

    threading.Thread(target=change).start()
    status, headers, png = get(server.url + "frame?after=" + version)  # waits for it
    assert status == 200 and headers["X-Rect"] == "0,0,0,0" and png == b""
    box["overlay"] = None  # the HUD stopped
    status, headers, _b = get(server.url + "frame?after=" + headers["X-Version"])
    assert status == 200 and headers["X-Version"] == NONE and headers["X-Rect"] == "0,0,0,0"


def test_only_requests_to_this_computer_by_its_address_are_answered(served):
    server, _box = served
    status, _h, _b = get(server.url, host=f"evil.example:{server.port}")
    assert status == 403
    status, _h, _b = get(server.url.replace("127.0.0.1", "localhost"),
                         host=f"localhost:{server.port}")  # fmt: skip
    assert status == 200


def test_a_taken_port_moves_to_the_next():
    first = StreamServer(lambda: None)
    assert first.start(0)
    second = StreamServer(lambda: None)
    try:
        assert second.start(first.port) and second.port != first.port
    finally:
        first.stop()
        second.stop()
