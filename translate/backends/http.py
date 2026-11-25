"""What the backends that speak plain HTTP (the OpenAI API, the local model's server) share:
one session kept alive, and a busy service asked again."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from i18n import _
from translate.base import StopError, TranslationError

log = logging.getLogger("translate")

RETRY_WAIT_MAX_S = 30.0  # the longest wait before asking a busy service again


class WebService:
    """One HTTP session kept alive between requests (a new TLS connection costs about as
    much as translating a line), a connection the server closed while idle tried again at
    once, and a busy service (429, 5xx) tried up to three times, ``retry_delays`` apart or as
    long as the server asks. A ``session`` with ``get`` and ``post`` like
    :class:`requests.Session` may be passed in (tests)."""

    service = ""
    no_key = ""
    key_refused = ""
    busy = ""
    offline = ""
    connect_timeout = 5.0

    def __init__(self, api_key: str, session: Any = None, timeout: float = 30.0) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self.retry_delays: tuple[float, ...] = (2.0, 6.0)
        self._session = session
        self._lock = threading.Lock()
        self._sleep: Callable[[float], None] = time.sleep

    def _need_key(self) -> None:
        if not self.api_key:
            raise StopError(_(self.no_key))
        if not self.api_key.isascii() or any(c.isspace() for c in self.api_key):
            # e.g. the masked key ("abc…xyz") pasted back: it cannot even be sent
            raise StopError(_(self.key_refused))

    def _http(self) -> Any:
        with self._lock:
            if self._session is None:
                import requests

                self._session = requests.Session()
            return self._session

    def _call(self, method: str, url: str, timeout: float | None = None, **kwargs: Any) -> Any:
        """``session.get`` / ``session.post``; the response, unless the service stayed busy
        or could not be reached (TranslationError)."""
        import requests

        session = self._http()
        wait_s = timeout or self.timeout
        tries = len(self.retry_delays) + 1
        attempt, reconnected = 0, False
        while True:
            try:
                resp = getattr(session, method)(
                    url, timeout=(self.connect_timeout, wait_s), **kwargs
                )
            except requests.ConnectTimeout as exc:
                raise TranslationError(_(self.offline)) from exc
            except requests.Timeout as exc:
                raise TranslationError(_("{offline} No answer came within {seconds} s.").format(
                    offline=_(self.offline), seconds=f"{wait_s:.0f}")) from exc
            except requests.ConnectionError as exc:
                if reconnected:
                    raise TranslationError(_(self.offline)) from exc
                reconnected = True  # most likely a kept-alive connection the server closed
                continue
            except requests.RequestException as exc:
                raise TranslationError(_("Could not talk to {service} ({error}).").format(
                    service=self.service, error=type(exc).__name__)) from exc
            if not self._is_busy(resp):
                return resp
            if attempt == tries - 1:
                raise TranslationError(_(self.busy))
            wait = retry_wait(resp, self.retry_delays[attempt])
            log.info("%s busy (HTTP %s): trying again in %.1f s", self.service,
                     resp.status_code, wait)
            self._sleep(wait)
            attempt += 1

    def _is_busy(self, resp: Any) -> bool:
        return resp.status_code in (429, 529) or resp.status_code >= 500


def json_body(resp: Any) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def service_message(data: Any, resp: Any) -> str:
    """The error message of an answer (``{"error": {"message": ...}}`` or
    ``{"error": "..."}``), or the HTTP status."""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
    return f"HTTP {resp.status_code}"


def retry_wait(resp: Any, delay: float) -> float:
    """The wait before trying a busy service again: ``delay``, or longer if its Retry-After
    asks for it, never over RETRY_WAIT_MAX_S, with a little jitter."""
    try:
        delay = max(delay, float(resp.headers.get("retry-after") or 0))
    except (AttributeError, TypeError, ValueError):
        pass
    return min(delay, RETRY_WAIT_MAX_S) * random.uniform(1.0, 1.25)
