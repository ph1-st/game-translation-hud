"""Live translation of the text the database does not know, in the background.

The HUD calls :meth:`LiveTranslator.request` for each line it cannot look up; the call
returns at once. A worker thread collects what comes in within ``batch_wait`` seconds (up
to ``max_batch`` lines), sends it to the backend as one request with the game's context,
and hands each translation to ``on_result`` (on the worker thread). Translated dialogue
joins the memory's recent conversation, so the next lines know what was just said.

Each text, by its normalized key, is translated once per translator. When there is more
to send than one request takes, the newest lines go first (the screen as it is now), and a
line that waited ``stale_after`` seconds is dropped (read again, it is asked for again).

Failures:

* an unusable answer to a batch (:class:`BadAnswerError`): its lines are sent again at
  once, one by one, so a line the model cannot handle does not spoil the others;
* a failing service (no connection, busy): the lines can be requested again after
  ``retry_after`` seconds (together, as before), and nothing is sent for a while
  (``cooldown``: 2 s, doubling with each failure in a row, up to 60 s);
* a line the model could not answer ``max_tries`` times is not asked for again (an outage
  never counts: the line was not the problem);
* a failure no retry mends (:class:`StopError`: the key refused, no credit, the model
  missing): nothing more is sent; ``stats()["halted"]`` says why.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from db.normalize import normalize_key
from translate.base import Backend, BadAnswerError, Item, StopError, TranslationError
from translate.memory import GameMemory

log = logging.getLogger("translate")


class LiveTranslator:
    def __init__(
        self,
        backend: Backend,
        memory: GameMemory,
        on_result: Callable[[Item, str], None],
        batch_wait: float = 0.15,
        max_batch: int = 12,
        retry_after: float = 30.0,
        cooldown: tuple[float, float] = (2.0, 60.0),
        stale_after: float = 20.0,
        max_tries: int = 3,
    ) -> None:
        self.backend = backend  # may be swapped while running: each batch reads it anew
        self.memory = memory
        self.on_result = on_result
        self.batch_wait = batch_wait
        self.max_batch = max(1, int(max_batch))
        self.retry_after = retry_after
        self.cooldown = cooldown  # (first, longest) wait after the service failed
        self.stale_after = stale_after
        self.max_tries = max(1, int(max_tries))
        self._cond = threading.Condition()
        self._pending: dict[str, tuple[Item, float]] = {}  # key -> (item, asked at), in order
        self._tries: dict[str, int] = {}  # key -> unusable answers
        self._bad: set[str] = set()  # keys whose last failure was an unusable answer
        self._given_up: set[str] = set()  # failed max_tries times
        self._wait = 0.0  # the current cool-down (0: none)
        self._cool_until = 0.0  # time.monotonic() before which nothing is sent
        self._halted: str | None = None  # why nothing is sent any more (a StopError)
        self._inflight: set[str] = set()
        self._done: set[str] = set()
        self._failed: dict[str, float] = {}  # key -> time.monotonic() of the failure
        self._alone: set[str] = set()  # retries of failed lines: sent one at a time
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None  # the running worker's stop flag
        self._requests = 0
        self._translated = 0
        self._failures = 0
        self._last_error: str | None = None
        self._last_latency_ms: int | None = None

    # ---- control -----------------------------------------------------------------------
    def start(self) -> None:
        """Start the worker thread (again after :meth:`stop`)."""
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = threading.Event()
            self._halted, self._wait, self._cool_until = None, 0.0, 0.0
            self._thread = threading.Thread(
                target=self._run, args=(self._stop,), name="live-translator", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Stop the worker and wait for it (up to ``timeout`` seconds; None: as long as a
        request in flight takes). Queued lines are dropped, and so is the answer to a
        request still in flight: once this returns (without timing out), no ``on_result``
        comes any more."""
        with self._cond:
            if self._stop is not None:
                self._stop.set()
            self._pending.clear()
            self._cond.notify_all()
            thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def request(self, item: Item) -> bool:
        """Queue a line for translation. False (nothing queued) if its text is empty, is
        queued, in flight or translated already, failed less than ``retry_after`` seconds
        ago or ``max_tries`` times, or the translator was stopped or halted."""
        key = normalize_key(item.text)
        if not key:
            return False
        with self._cond:
            if self._stop is not None and self._stop.is_set() or self._halted is not None:
                return False
            if key in self._pending or key in self._inflight or key in self._done:
                return False
            if key in self._given_up:
                return False
            failed_at = self._failed.get(key)
            if failed_at is not None:
                if time.monotonic() - failed_at < self.retry_after:
                    return False
                del self._failed[key]
                if key in self._bad:  # (the model could not answer it: on its own)
                    self._alone.add(key)
            self._pending[key] = (item, time.monotonic())
            self._requests += 1
            self._cond.notify_all()
            return True

    def stats(self) -> dict[str, Any]:
        """Counters for a status display. ``pending`` counts queued and in-flight lines;
        ``last_error`` is cleared by the next successful request; ``halted`` says why
        nothing is sent any more (None while it works); ``waiting`` is the seconds left of
        a cool-down after the service failed."""
        with self._cond:
            return {
                "backend": getattr(self.backend, "name", type(self.backend).__name__),
                "requests": self._requests,
                "translated": self._translated,
                "failed": self._failures,
                "pending": len(self._pending) + len(self._inflight),
                "last_error": self._last_error,
                "last_latency_ms": self._last_latency_ms,
                "halted": self._halted,
                "waiting": max(0.0, self._cool_until - time.monotonic()),
                # tokens used so far (backends that count them: Claude, ChatGPT), else None
                "input_tokens": getattr(self.backend, "input_tokens", None),
                "output_tokens": getattr(self.backend, "output_tokens", None),
            }

    # ---- worker ------------------------------------------------------------------------
    def _run(self, stop: threading.Event) -> None:
        while True:
            with self._cond:
                batch = self._next_batch(stop)
            if batch is None:
                return
            if batch:
                self._translate(batch, stop)

    def _next_batch(self, stop: threading.Event) -> list[tuple[str, Item]] | None:
        """Wait for work and take the next batch off the queue (holding the lock): the
        newest lines first. None once stopped."""
        while True:
            while not self._pending and not stop.is_set():
                self._cond.wait()
            if stop.is_set():
                return None
            now = time.monotonic()
            if now < self._cool_until:  # the service failed a moment ago
                self._cond.wait(self._cool_until - now)
                continue
            for k in [k for k, (_it, t) in self._pending.items() if now - t > self.stale_after]:
                del self._pending[k]  # the screen moved on (read again, it comes back)
                self._alone.discard(k)
            if self._pending:
                break
        newest = next(reversed(self._pending))
        if newest in self._alone:
            keys = [newest]
        else:  # give the lines of one screen a moment to arrive together
            deadline = time.monotonic() + self.batch_wait
            while True:
                together = [k for k in self._pending if k not in self._alone]
                keys = together[-self.max_batch :]
                left = deadline - time.monotonic()
                if len(keys) >= self.max_batch or left <= 0 or stop.is_set() or not keys:
                    break
                self._cond.wait(left)
            if stop.is_set():
                return None
            if not keys:  # (taken by a stop and a start meanwhile)
                return []
        batch = [(k, self._pending.pop(k)[0]) for k in keys]
        self._inflight.update(keys)
        return batch

    def _translate(self, batch: list[tuple[str, Item]], stop: threading.Event) -> None:
        keys = [k for k, _ in batch]
        items = [it for _, it in batch]
        t0 = time.monotonic()
        try:
            out = self.backend.translate(items, self.memory.context())
            if len(out) != len(items):
                raise TranslationError(
                    f"The translator returned {len(out)} translations for {len(items)} lines."
                )
        except Exception as exc:  # a bug in a backend must not end the worker either
            if isinstance(exc, TranslationError):
                error = str(exc)
                log.warning("live translation of %d lines failed: %s", len(items), exc)
            else:
                error = f"Unexpected error in the translator: {exc}"
                log.exception("live translation of %d lines failed", len(items))
            self._failed_batch(batch, exc, error, stop)
            return
        ms = round((time.monotonic() - t0) * 1000)
        with self._cond:
            self._inflight.difference_update(keys)
            if stop.is_set():  # stopped meanwhile: the lines can be requested again
                return
            self._done.update(keys)
            self._alone.difference_update(keys)
            self._translated += len(keys)
            self._last_latency_ms = ms
            self._last_error = None
            self._wait = 0.0
        name = getattr(self.backend, "name", type(self.backend).__name__)
        log.info("live: %d lines translated in %d ms by %s", len(items), ms, name)
        for item, translation in zip(items, out, strict=True):
            if stop.is_set():
                return
            try:
                if item.kind == "dialogue":
                    self.memory.remember(item.text, translation, item.speaker, kind="dialogue")
                self.on_result(item, translation)
            except Exception:  # the HUD's handler must not end the worker
                log.exception("handling the live translation of %r failed", item.text)

    def _failed_batch(
        self, batch: list[tuple[str, Item]], exc: Exception, error: str, stop: threading.Event
    ) -> None:
        keys = [k for k, _ in batch]
        now = time.monotonic()
        with self._cond:
            self._inflight.difference_update(keys)
            if stop.is_set():  # stopped meanwhile: a translator started since is not ours
                return
            self._last_error = error
            if isinstance(exc, StopError):  # the key, the credit, the model: nothing more
                self._halted = error
                self._failures += len(keys)
                self._pending.clear()
                return
            if isinstance(exc, BadAnswerError) and len(keys) > 1:
                # the answer depends on the text: each line on its own, at once
                for k, item in batch:
                    self._pending[k] = (item, now)
                self._alone.update(keys)
                self._cond.notify_all()
                return
            self._failures += len(keys)
            bad = isinstance(exc, BadAnswerError)
            for k in keys:
                self._alone.discard(k)
                if bad:  # (one line on its own: the model cannot answer it)
                    self._bad.add(k)
                    self._tries[k] = self._tries.get(k, 0) + 1
                    if self._tries[k] >= self.max_tries:
                        self._given_up.add(k)
                        continue
                else:
                    self._bad.discard(k)
                self._failed[k] = now
            if not bad:  # the service failed, not the text: nothing is sent for a while
                first, longest = self.cooldown
                self._wait = min(longest, max(first, self._wait * 2))
                self._cool_until = now + self._wait
