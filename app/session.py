"""The overlay for one game, run from the app: the HUD core (hud/) on the game's window, in
the language the player reads, with the dialogue style they chose and, when switched on,
live translation of text no database has.

A reading no database has is acted on once it has settled (:class:`LiveReadings`: the core
reports only text that held still while it was read, never a line still typing out or a
glimpse of noise), only if it holds enough of the game's language, and each text only once:
it is written to the miss log (:mod:`app.misslog`) and, with live translation on, sent to be
translated. What comes back is shown at once (a little faded: provisional) and kept in the
game's live translations, so the next time it is there from the start.
"""

from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from app.config import GameEntry, Settings
from app.library import Library
from app.misslog import MissLog
from app.paths import models_dir
from app.reader import MeikiReader, reader_ready
from db.store import Line, Store
from db.templates import templatize
from db.text import cjk_count
from hud.app import HudApp, build_core
from hud.config import HudConfig
from hud.hotkey import pretty
from hud.pipeline import HudCore
from i18n import _
from translate.languages import target_name
from translate.local import stop_servers

log = logging.getLogger("app")

SETTLE_OPEN_S = 1.2  # dialogue that does not end a sentence counts once read the same this
SETTLE_OPEN_LOCAL_S = 0.6  # much later (it may be pausing at a comma); sooner for the model
# on this computer (a line sent early costs nothing)
SENTENCE_ENDS = "。！？!?…」』）)♪～"
MIN_CHARS = 2  # a reading is translated if it holds this many characters of the game's
MIN_CONF = {"ui": 0.75, "dialogue": 0.6}  # language, read at least this sure
SEEN_MAX = 4000  # readings remembered while they settle (older ones are forgotten)
SEEN_KEEP_S = 120.0
WAIT, SKIP = "", None  # LiveReadings.judge: read it again later; never translated


class LiveReadings:
    """What becomes of a reading no database has (any thread). The core reports only text
    that held still while it was read, so a label is translated at once, and so is a line of
    dialogue that ends a sentence; one that does not may be pausing at a comma (おお、…) and
    counts once it read the same *open_wait* seconds later (its area is read again until)."""

    def __init__(self, open_wait: float = SETTLE_OPEN_S) -> None:
        self.open_wait = open_wait
        self._first: dict[str, float] = {}  # template key -> when an open line was first read
        self._lock = threading.Lock()

    def judge(self, text: str, kind: str, conf: float, player: str | None) -> str | None:
        """The text to translate (numbers and the player's name placeholders: a timer
        counting down is one text, translated once), WAIT, or SKIP (read unsure, or no
        Japanese to speak of)."""
        from db.templates import templated_text

        if conf < MIN_CONF.get(kind, MIN_CONF["ui"]) or cjk_count(text) < MIN_CHARS:
            return SKIP
        t = templatize(text, player)
        if not t.key:
            return SKIP
        if kind == "dialogue" and not text.rstrip().endswith(tuple(SENTENCE_ENDS)):
            now = time.monotonic()
            with self._lock:
                first = self._first.setdefault(t.key, now)
                if len(self._first) > SEEN_MAX:  # forget old readings (a long session)
                    self._first = {k: v for k, v in self._first.items()
                                   if now - v < SEEN_KEEP_S}  # fmt: skip
            if now - first < self.open_wait:
                return WAIT
        return templated_text(text, t.key)

_BUILD_LOCK = threading.Lock()  # one core built at a time (a restart while one builds waits)


class _CoreBuilder(QObject):
    """Builds the HUD core (loads the databases, starts the OCR engine) off the GUI, on a
    daemon thread: closing the app while it builds never waits for it."""

    built = Signal(object)
    failed = Signal(str)

    def __init__(self, config: HudConfig) -> None:
        super().__init__()
        self.config = config
        self._thread = threading.Thread(target=self.run, name="hud-build", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def run(self) -> None:
        try:
            with _BUILD_LOCK:
                core = build_core(self.config)
        except Exception as e:  # noqa: BLE001 - shown to the player
            log.exception("could not start the overlay")
            self._emit(self.failed, str(e) or e.__class__.__name__)
            return
        self._emit(self.built, core)

    @staticmethod
    def _emit(signal, value) -> None:  # noqa: ANN001
        try:
            signal.emit(value)
        except RuntimeError:  # the app closed while the core was built: nobody to tell
            pass


class HudSession(QObject):
    """Starts and stops the overlay for a game; the window shows its state and messages."""

    state = Signal(str)  # starting | running | stopped
    message = Signal(str)  # what to tell the player

    def __init__(self, library: Library, settings: Settings) -> None:
        super().__init__()
        self.library = library
        self.settings = settings
        self.game: GameEntry | None = None
        self.hud: HudApp | None = None
        self.core: HudCore | None = None
        self._builder: _CoreBuilder | None = None
        self.live = None  # translate.LiveTranslator, when switched on
        self.memory = None  # translate.GameMemory
        self.misses: MissLog | None = None
        self.readings = LiveReadings()
        self._asked: dict[str, tuple] = {}  # text sent -> (what was read, where it was)
        self._lock = threading.Lock()
        self._gen = 0  # bumped by every stop: a core built for an older start is dropped
        self._live_gen = 0  # bumped when live translation stops: late answers are dropped
        self.on_raise = None  # called when the overlay goes to the top (the panel follows)
        self.db_state: tuple = ()  # the game's databases when the overlay started
        self.stream = None  # hud.overlay.stream.StreamServer, while it is switched on

    @property
    def running(self) -> bool:
        return self.hud is not None

    # ---- the stream source (OBS) -------------------------------------------------------
    def apply_stream(self) -> None:
        """Start or stop the overlay's page for OBS as the settings say. It serves the
        running overlay, whichever game it is for (none: a clear page)."""
        from hud.overlay.stream import StreamServer

        if not self.settings.stream_source:
            if self.stream is not None:
                self.stream.stop()
                self.stream = None
            return
        if self.stream is None:
            self.stream = StreamServer(self._stream_overlay)
        if not self.stream.start(self.settings.stream_port):
            self.stream = None
            self.message.emit(_("The page for OBS could not start: every port it tries is "
                                "taken."))

    def _stream_overlay(self):  # noqa: ANN202 - the overlay window (the server's threads)
        hud = self.hud
        return hud.overlay if hud is not None else None

    @property
    def stream_url(self) -> str:
        return self.stream.url if self.stream is not None else ""

    def picture_size(self) -> tuple[int, int] | None:
        """The size of the game's picture (what OBS's source must be), when known."""
        hud = self.hud
        frame = hud.overlay.newest_frame if hud is not None else None
        return (int(frame.shape[1]), int(frame.shape[0])) if frame is not None else None

    @property
    def busy(self) -> bool:
        return self._builder is not None

    # ---- start / stop ------------------------------------------------------------------
    def config(self, game: GameEntry) -> HudConfig:
        """What the overlay runs with, for *game*."""
        lib, s = self.library, self.settings
        return HudConfig(
            game=game.profile,
            dbs=lib.db_paths(game),
            player=game.player_name or None,
            target_lang=lib.target_lang(game),
            profile=lib.profile(game),
            glossary=lib.glossary_path(game),
            cache_dir=lib.cache_dir(game),
            device=s.ocr_device,
            reader=s.ocr_reader if reader_ready() else "standard",
            reader_dir=MeikiReader(models_dir()).folder,
            confirmation_mode=s.confirmation_mode,
            cover_mode=lib.cover_mode(game),
            text_scale=s.text_size / 100.0,
            plain_text=s.white_text,
            toggle_hotkey=s.toggle_hotkey,
            quit_hotkey=s.stop_hotkey,
            # (the stop hotkey must not tear down its own handler while it runs)
            on_quit=lambda: QTimer.singleShot(0, self.stop),
            on_raise=self._raised,
        )

    def start(self, game: GameEntry) -> None:
        if self.running or self.busy:
            return
        self.game = game
        self.db_state = self.library.db_state(game)  # (the folder as the overlay took it)
        config = self.config(game)
        self.state.emit("starting")
        self.message.emit(_("Starting the overlay (loading the databases and the text "
                            "reader)…"))
        gen = self._gen
        self._builder = _CoreBuilder(config)
        self._builder.built.connect(lambda core: self._started(gen, config, core))
        self._builder.failed.connect(lambda why: self._failed(gen, why))
        self._builder.start()

    def _raised(self) -> None:
        if self.on_raise is not None:
            self.on_raise()

    def _failed(self, gen: int, why: str) -> None:
        if gen != self._gen:
            return
        self._builder = None
        self.state.emit("stopped")
        self.message.emit(_("The overlay could not start: {why}").format(why=why))

    def _started(self, gen: int, config: HudConfig, core: HudCore) -> None:
        if gen != self._gen:
            return  # stopped (or restarted) while it was being built: never shown
        self._builder = None
        self.core = core
        try:
            self.apply_content()
            self._start_misses()
            self._start_live(core)
            self._hook(core)
            self.hud = HudApp(config, core)
            self.hud.start(QApplication.instance())
        except Exception as e:  # noqa: BLE001 - shown to the player
            log.exception("could not start the overlay")
            self.stop(quiet=True)
            self.message.emit(_("The overlay could not start: {why}").format(
                why=e or e.__class__.__name__))
            return
        self.state.emit("running")
        n = len(core.maker.lookup.keys)
        text = _("Overlay running ({n} lines in the databases).").format(n=f"{n:,}")
        if self.live is not None:
            text += " " + _("Live translation is on.")
        taken = ", ".join(pretty(k) for k in self.hud.hotkey_failures)
        if taken:
            text += " " + _("Another program has the hotkey {keys}: choose others under "
                            "More.").format(keys=taken)
        self.message.emit(text)

    def stop(self, quiet: bool = False) -> None:
        was = self.running or self.busy
        self._gen += 1
        self._builder = None
        hud, self.hud = self.hud, None
        if hud is not None:
            hud.stop()
        self._stop_live()
        stop_servers()  # (the local model leaves the graphics card with the overlay)
        self.misses = None
        self.core = None
        self.state.emit("stopped")
        if was and not quiet:
            self.message.emit(_("Overlay stopped."))

    def toggle_overlay(self) -> None:
        if self.hud is not None:
            self.hud.toggle()

    def apply_content(self) -> None:
        """What the player wants translated, applied to the running overlay at once."""
        if self.core is not None:
            s = self.settings
            self.core.set_content(dialogue=s.translate_dialogue, ui=s.translate_ui)

    # ---- readings no database has --------------------------------------------------------
    def _start_misses(self) -> None:
        g = self.game
        if g is None or not self.settings.log_misses:
            self.misses = None
            return
        self.misses = MissLog(self.library.misses_dir(g), g.profile or g.id,
                              self.library.target_lang(g), g.player_name or None)

    def misses_count(self) -> int | None:
        return self.misses.count() if self.misses is not None else None

    def _hook(self, core: HudCore) -> None:
        """Readings no database has reach this session while it logs them or live
        translation is on."""
        if self.misses is not None or self.live is not None:
            core.set_live(self._unmatched, on_line=self._shown)
        else:
            core.set_live(None, None)

    def _unmatched(self, text: str, kind: str, speaker, conf: float, box) -> bool:  # noqa: ANN001
        """A reading no database has, its text held still (OCR thread). Returns True to
        have its area read again: it waits to settle."""
        player = (self.game.player_name or None) if self.game is not None else None
        verdict = self.readings.judge(text, kind, conf, player)
        if not verdict:
            return verdict == WAIT
        misses, live = self.misses, self.live  # (the GUI thread may stop them meanwhile)
        if misses is not None:
            misses.add(text, kind, speaker)
        if live is None:
            return False
        from translate import Item

        item = Item(verdict, kind, speaker)
        with self._lock:
            if len(self._asked) > SEEN_MAX:  # (a long session)
                self._asked = {}
            self._asked[item.text] = (text, box)
        live.request(item)
        return False

    def _shown(self, source: str, translation: str, speaker) -> None:  # noqa: ANN001
        """A line of dialogue shown: the conversation, context for the next translations."""
        if self.memory is not None:
            self.memory.remember(source, translation, speaker, "dialogue")

    # ---- live translation ----------------------------------------------------------------
    def reload_live(self) -> None:
        """Live translation was switched on or off, or to another engine: the running
        overlay takes it as it is (no restart)."""
        core = self.core
        if core is None:
            return
        self._stop_live()
        self._start_live(core)  # (says why, when it cannot start)
        if self.live is None or self.settings.live.engine != "plamo":
            stop_servers()
        self._hook(core)
        if self.live is not None:
            self.message.emit(_("Live translation is on."))
        elif not self.settings.live.enabled:
            self.message.emit(_("Live translation is off."))

    def _stop_live(self) -> None:
        self._live_gen += 1  # an answer still on its way is dropped
        if self.live is not None:
            self.live.stop(timeout=0)  # a request under way is let go, never waited for
            self.live = None
        if self.memory is not None:
            self.memory.save()
            self.memory = None
        with self._lock:
            self._asked = {}

    def _start_live(self, core: HudCore) -> None:
        live, g = self.settings.live, self.game
        self.readings = LiveReadings()
        if not live.enabled or g is None:
            return
        try:
            backend = make_backend(self.settings, live.engine)
        except Exception as e:  # noqa: BLE001 - shown, the overlay runs without it
            self.message.emit(_("Live translation is off: {why}").format(why=e))
            return
        from translate import GameMemory, LiveTranslator, PlamoBackend

        lib = self.library
        self.memory = GameMemory(
            lib.memory_path(g), g.name, source=lib.source_lang(g), target=lib.target_lang(g)
        )  # (read at once)
        self.memory.set_glossary(core.glossary.pairs())
        gen = self._live_gen
        local = isinstance(backend, PlamoBackend)
        self.live = LiveTranslator(
            backend, self.memory,
            on_result=lambda item, tr: self._translated(item, tr, gen, g),
            batch_wait=0.02 if local else 0.15,
        )  # fmt: skip
        self.live.start()
        if local:  # the model loads now, not when the first line needs it
            self.readings = LiveReadings(SETTLE_OPEN_LOCAL_S)
            threading.Thread(target=backend.warm_up, name="local-model", daemon=True).start()

    def _translated(self, item, translation: str, gen: int, game: GameEntry) -> None:  # noqa: ANN001
        """A live translation came back (translator thread): shown, and kept in the
        game's live translations into the language read."""
        core = self.core
        if gen != self._live_gen or core is None or not translation.strip():
            return  # live translation stopped (or the game changed) meanwhile
        with self._lock:
            text, box = self._asked.get(item.text, (item.text, None))
        core.add_translation(text, translation, item.kind, item.speaker, box)
        try:
            self._keep(text, translation, item.kind, game)
        except Exception:  # noqa: BLE001 - the overlay shows it anyway
            log.exception("could not keep a live translation")

    def _keep(self, source: str, translation: str, kind: str, game: GameEntry) -> None:
        from sync.download import db_lock

        t = templatize(source, game.player_name or None)
        if not t.key:
            return
        lib = self.library
        path = lib.live_path(game)
        with db_lock(path), Store(path) as store:
            if not store.get_meta("target_lang"):  # (a new file says what it is)
                store.set_info(
                    game=game.profile or game.id, source_lang=lib.source_lang(game),
                    target_lang=lib.target_lang(game),
                    display_name=f"{game.name} ({target_name(lib.target_lang(game))}, live)",
                )  # fmt: skip
            if store.get_line(t.key) is None:
                store.upsert_line(Line(
                    key=t.key, source_text=source, speaker=None, screen_type=kind,
                    is_template=t.is_template, translation=None, tier="machine",
                    source="live", chapter=None, seq=None, game_version=None,
                ))  # fmt: skip
            store.set_translation(t.key, translation, "machine", "live")

    def live_stats(self) -> dict | None:
        return self.live.stats() if self.live is not None else None


def make_backend(settings: Settings, engine: str):  # noqa: ANN201 - a translate.Backend
    """The live translation engine *engine* (one of app.config.LIVE_ENGINES) as the
    settings have it (raises with a message to show)."""
    from app.keys import get_key
    from translate import (
        ClaudeBackend,
        CompatibleBackend,
        DeepLBackend,
        OpenAIBackend,
        PlamoBackend,
        TranslationError,
    )
    from translate.local import MODELS, LocalModels, shared_server

    live = settings.live
    if engine == "plamo":
        server = shared_server(LocalModels(models_dir()), live.plamo_model)
        return PlamoBackend(server, MODELS[live.plamo_model].name)
    if engine == "openai":
        key = get_key("openai")
        if not key:
            raise TranslationError(_("no OpenAI API key yet: paste one under Live translation "
                                     "(you pay OpenAI for what you use)."))
        return OpenAIBackend(key, model=live.openai_model)
    if engine == "deepl":
        key = get_key("deepl")
        if not key:
            raise TranslationError(_("no DeepL API key yet: paste one under Live translation "
                                     "(DeepL's free plan translates 500,000 characters a "
                                     "month)."))
        return DeepLBackend(key)
    if engine == "custom":
        if not live.custom_url.strip():
            raise TranslationError(_("no server address yet: type it under Live translation."))
        return CompatibleBackend(live.custom_url, live.custom_model, get_key("custom"))
    key = get_key()
    if not key:
        raise TranslationError(_("no Claude API key yet: paste one under Live translation "
                                 "(you pay Anthropic for what you use)."))
    return ClaudeBackend(key, model=live.claude_model)
