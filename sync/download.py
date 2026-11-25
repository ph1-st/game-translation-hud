"""Share a translation database through a link anyone can open.

A maintainer uploads the database file to any file host and shares its link; players paste
the link into the app, which downloads the file and swaps it in for their copy. A share
link usually leads to a preview page, so :func:`direct_url` turns the links of the common
hosts into links that return the file itself:

* Google Drive: ``drive.google.com/file/d/<id>/view``, ``.../file/d/<id>``,
  ``open?id=<id>`` and ``uc?id=<id>`` become
  ``drive.usercontent.google.com/download?id=<id>&export=download&confirm=t``
  (``confirm=t`` skips the page that warns that a large file cannot be scanned for
  viruses). A shared folder is an index of its own (below).
* Dropbox: ``www.dropbox.com/s/...`` and ``/scl/fi/...?rlkey=...`` become the same link
  with ``dl=1``.
* GitHub: a file page ``github.com/<owner>/<repo>/blob/<ref>/<path>`` becomes the raw
  file. Release asset links (``.../releases/download/...``) are direct already.
* OneDrive: ``onedrive.live.com/redir?...`` (and ``embed?...``) become ``download?...``.
  Short ``1drv.ms`` links are kept as they are (they redirect), but they may end on
  OneDrive's viewer page: the link from OneDrive's "Embed" dialog, or a ``download``
  link, works best.
* Anything else is used as it is.

:func:`check_for_update` asks the server whether the file changed since the last
download, from its answer's headers alone (nothing is downloaded): the app does that at
start and asks the player before it downloads anything. :func:`sync_database` streams the
file next to its destination, checks that it is a translation database (a web page or any
other file is rejected) and swaps it in, unless it is the file already there.

A database from someone else is data, never code: before it touches anything it must be an
SQLite file (at most :data:`MAX_BYTES`) whose ``lines`` table SQLite reads in full
(``PRAGMA integrity_check``), and then only the text and number values of its rows are
copied into a fresh database of this app's own schema (no view, trigger or other part of
the file's schema ever runs; a value of any other kind is dropped). The database it
replaces is backed up first. Every failure is a :class:`SyncError` whose message says what
to do and can be shown to the user as it is.

A link may also point at an index of databases (``index.json``): one database per
language, each with its download URL, SHA-256 and size. :func:`fetch_index` reads it and
the app downloads the entry for the player's language; the file must match the index. A
shared Google Drive folder works the same way: the files in it named as the app names
databases (``<game>.<lang>.<version>.sqlite``) are its entries, the newest version of each
language (:func:`parse_drive_folder`).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, parse_qs, quote, urlsplit, urlunsplit

log = logging.getLogger("sync")

SQLITE_HEADER = b"SQLite format 3\x00"
USER_AGENT = "GameTranslationHUD (translation database download)"
CHUNK = 1 << 16
KEEP_BACKUPS = 3  # backups of a game's database kept (the oldest go)
MAX_BYTES = 1 << 30  # a bigger file is no translation database (a whole game's is ~50 MB)
MAX_TEXT = 4000  # characters a text of a database may have (a game's line is far shorter)
INDEX_MAX_BYTES = 1 << 20  # an index of databases is a few hundred bytes per language
_NUMBER_FIELDS = {"is_template", "seq", "samples", "ocr_confidence"}
_FIELD_SQL = {  # a value the file leaves out (or has wrong) as this app needs it
    "source_text": "COALESCE({source_text}, {key})",
    "screen_type": "COALESCE({screen_type}, 'dialogue')",
    "is_template": "COALESCE({is_template}, 0)",
    "tier": "CASE WHEN {tier} IN ('machine', 'reviewed', 'human') THEN {tier} ELSE 'machine' END",
    "source": "COALESCE({source}, 'batch')",
    "samples": "COALESCE({samples}, 0)",
}

WEB_PAGE = (
    "The link returned a web page instead of the database file. Make sure the file is shared "
    "publicly ('Anyone with the link can view') and that the link points to the file itself."
)
NOT_A_DATABASE = (
    "The downloaded file is not a Game Translation HUD database. Make sure the link points to "
    "the database file itself (the .sqlite file, not a zip file or a folder)."
)
DAMAGED = (
    "The downloaded file is damaged or is not a Game Translation HUD database. Try again; if "
    "it keeps happening, ask for a new link."
)
INTERRUPTED = (
    "The download stopped before the end (the connection was lost or timed out). Try again."
)
NOT_AS_LISTED = (
    "The downloaded file is not the one the index lists (it may have been replaced while "
    "it downloaded). Try again in a minute."
)
BAD_INDEX = (
    "The link points at a list of databases that could not be read. Ask for the right link."
)
TOO_LARGE = (
    "The file behind the link is far too big for a translation database (over 1 GB): it is "
    "not one. Ask for the right link."
)

_DRIVE_HOSTS = {"drive.google.com", "docs.google.com", "drive.usercontent.google.com"}
_DRIVE_FILE_RE = re.compile(r"/file/(?:u/\d+/)?d/([A-Za-z0-9_-]+)")
_DRIVE_FOLDER_RE = re.compile(r"/folders/([A-Za-z0-9_-]+)")
_DRIVE_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
# a shared folder's file list as Google shows it on other sites (no sign-in, no API key)
DRIVE_FOLDER_VIEW = "https://drive.google.com/embeddedfolderview?id={id}"
DRIVE_DOWNLOAD = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
_DRIVE_ENTRY_RE = re.compile(
    r'id="entry-([A-Za-z0-9_-]+)".*?class="flip-entry-title">([^<]*)<', re.S)
# a database file as the app names it: <game>.<lang>.<version>.sqlite
DB_FILE_RE = re.compile(r"([a-z0-9][a-z0-9_-]*)\.([A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})*)\.(.+)"
                        r"\.sqlite", re.I)
_DROPBOX_HOSTS = {"dropbox.com", "www.dropbox.com"}
_GITHUB_HOSTS = {"github.com", "www.github.com"}
_LOCAL_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|\\\\|/")


class SyncError(Exception):
    """A download or database problem, worded for the user: what went wrong and what to do."""


# ---- links ------------------------------------------------------------------------------
def direct_url(link: str) -> str:
    """The direct-download URL of a share link (the hosts it knows are listed in the module
    docstring; any other http(s) link is returned as it is). Surrounding whitespace and the
    ``<...>`` that chat apps put around links are removed. Raises :class:`SyncError` when
    *link* is not an http(s) link."""
    url = link.strip()
    if url.startswith("<") and url.endswith(">"):
        url = url[1:-1].strip()
    if not url:
        raise SyncError("Paste the link to the database file first.")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.netloc:
        if scheme == "file" or _LOCAL_PATH_RE.match(url):
            raise SyncError(
                "That is a file on this computer, not a web link. Paste the share link of the "
                "database file instead (it starts with https://)."
            )
        raise SyncError(
            "That is not a web link. Paste the share link of the database file (it starts "
            "with https://)."
        )
    host = (parts.hostname or "").lower()
    if host in _DRIVE_HOSTS:
        return _google_drive(parts) or url
    if host in _DROPBOX_HOSTS and parts.path.startswith(("/s/", "/scl/fi/")):
        return urlunsplit(parts._replace(query=_with_param(parts.query, "dl", "1")))
    if host in _GITHUB_HOSTS:
        bits = parts.path.split("/")  # "", owner, repo, "blob", ref, path...
        if len(bits) >= 6 and bits[3] == "blob" and all(bits[1:3]) and bits[4] and bits[-1]:
            return "https://raw.githubusercontent.com/" + "/".join(bits[1:3] + bits[4:])
    if host == "onedrive.live.com" and parts.path.lower() in ("/redir", "/embed"):
        return urlunsplit(parts._replace(path="/download"))
    return url


def _google_drive(parts: SplitResult) -> str | None:
    """The download URL of a Google Drive file link; None for a link without a file id
    (a Google Docs document, for instance), which is used as it is."""
    if "/folders/" in parts.path or "folderview" in parts.path:
        raise SyncError(
            "That is the link of a Google Drive folder. Open the folder, right-click the "
            "database file, choose Share > Copy link and paste that link."
        )
    query = parse_qs(parts.query)
    m = _DRIVE_FILE_RE.search(parts.path)
    file_id = m.group(1) if m else (query.get("id") or [""])[0]
    if not _DRIVE_ID_RE.fullmatch(file_id):
        return None
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    resource_key = (query.get("resourcekey") or [""])[0]  # older link-shared files need it
    if resource_key:
        url += "&resourcekey=" + quote(resource_key, safe="")
    return url


def _with_param(query: str, name: str, value: str) -> str:
    """*query* with the parameter *name* set to *value*, everything else kept as it was."""
    out: list[str] = []
    found = False
    for part in query.split("&"):
        if not part:
            continue
        if part.split("=", 1)[0] == name:
            if not found:
                out.append(f"{name}={value}")
                found = True
            continue
        out.append(part)
    if not found:
        out.append(f"{name}={value}")
    return "&".join(out)


# ---- indexes of databases ---------------------------------------------------------------
@dataclass
class IndexEntry:
    """One database an index lists."""

    lang: str  # the language it translates into (en, es-419...)
    url: str  # where to download it (absolute)
    name: str = ""  # the language's name, as the index gives it
    sha256: str = ""
    size: int | None = None
    version: str = ""
    game_version: str = ""
    min_app: str = ""  # the oldest app version that reads it


@dataclass
class DbIndex:
    url: str
    game: str
    entries: list[IndexEntry]

    def entry_for(self, lang: str) -> IndexEntry | None:
        """The database for a player who reads *lang*: that language, else another variant
        of it (es-419 for es)."""
        from translate.languages import match

        scored = [(match(e.lang, lang), e) for e in self.entries]
        scored = [(m, e) for m, e in scored if m > 0]
        return max(scored, key=lambda p: p[0])[1] if scored else None


def drive_folder_id(link: str) -> str | None:
    """The folder id of a Google Drive folder link (``drive.google.com/drive/folders/<id>``,
    ``.../drive/u/0/folders/<id>``, ``folderview?id=<id>``); None for any other link."""
    url = link.strip()
    if url.startswith("<") and url.endswith(">"):
        url = url[1:-1].strip()
    parts = urlsplit(url)
    if (parts.hostname or "").lower() not in _DRIVE_HOSTS:
        return None
    m = _DRIVE_FOLDER_RE.search(parts.path)
    folder = m.group(1) if m else ""
    if not folder and parts.path.rstrip("/").endswith("folderview"):
        folder = (parse_qs(parts.query).get("id") or [""])[0]
    return folder if _DRIVE_ID_RE.fullmatch(folder) else None


def is_index_link(link: str) -> bool:
    """Whether *link* names an index of databases (a .json file, or a shared Google Drive
    folder) rather than a database."""
    if drive_folder_id(link):
        return True
    try:
        path = urlsplit(direct_url(link)).path.lower()
    except SyncError:
        return False
    return path.endswith(".json")


def fetch_index(link: str, timeout: float = 20.0, game: str = "") -> DbIndex:
    """The index of databases behind *link*: an index file, or a shared Google Drive folder
    (*game*: the game whose databases to take from a folder that holds several games').
    Raises :class:`SyncError`."""
    import json

    folder = drive_folder_id(link)
    if folder:
        page = _get_small(DRIVE_FOLDER_VIEW.format(id=folder), timeout)
        return parse_drive_folder(page.decode("utf-8", "replace"), link.strip(), game)
    url = direct_url(link)
    body = _get_small(url, timeout)
    try:
        data = json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SyncError(BAD_INDEX) from e
    return parse_index(data, url)


def _get_small(url: str, timeout: float) -> bytes:
    """The body of a small answer (an index, a folder's file list). Raises SyncError."""
    import requests

    host = urlsplit(url).hostname or url
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True,
                          headers={"User-Agent": USER_AGENT}) as resp:  # fmt: skip
            if resp.status_code >= 400:
                raise SyncError(_http_problem(resp.status_code))
            body = b""
            for chunk in resp.iter_content(CHUNK):
                body += chunk
                if len(body) > INDEX_MAX_BYTES:
                    raise SyncError(BAD_INDEX)
    except requests.Timeout as e:
        raise SyncError(f"{host} did not answer within {timeout:g} seconds.") from e
    except requests.RequestException as e:
        raise SyncError(f"Could not reach {host} ({e.__class__.__name__}).") from e
    return body


def parse_drive_folder(page: str, link: str, game: str = "") -> DbIndex:
    """The databases a shared Google Drive folder holds, from its file list (*page*: the
    HTML of :data:`DRIVE_FOLDER_VIEW`), as an index: the files named as the app names
    databases (``azur-promilia.en.2026.11.25.sqlite``), the newest version of each
    language. A folder with several games' files serves *game*'s. Raises
    :class:`SyncError` when the folder is not shared or holds no database."""
    import html

    from translate.languages import normalize

    if "flip-entries" not in page:
        raise SyncError(
            "Google Drive did not show the folder: in Drive, share it with \"Anyone with the "
            "link\" and paste its link again."
        )
    found = []
    for file_id, name in _DRIVE_ENTRY_RE.findall(page):
        m = DB_FILE_RE.fullmatch(html.unescape(name).strip())
        if m:
            found.append((m.group(1).lower(), normalize(m.group(2)), m.group(3), file_id))
    games = {g for g, *_ in found}
    pick = game.lower() if game.lower() in games else next(iter(games)) if len(games) == 1 else ""
    if not pick:
        raise SyncError(
            "The folder has no translation database for this game (a file named like "
            "azur-promilia.en.2026.11.25.sqlite)." if not games else
            "The folder holds databases of several games, none of them this one: "
            + ", ".join(sorted(games)) + "."
        )
    newest: dict[str, tuple[str, str]] = {}  # language -> (version, file id)
    for g, lang, version, file_id in found:
        if g == pick and (lang not in newest
                          or version_key(version) > version_key(newest[lang][0])):
            newest[lang] = (version, file_id)
    entries = [IndexEntry(lang=lang, url=DRIVE_DOWNLOAD.format(id=file_id), version=version)
               for lang, (version, file_id) in sorted(newest.items())]
    return DbIndex(link, pick, entries)


def version_key(version: str) -> tuple:
    """A database version, to compare: its parts in order, numbers as numbers; a part with
    letters ("prelaunch") comes before the version without it, as pre-releases go
    (2026.11.25-prelaunch.1 < 2026.11.25 < 2026.11.25.1)."""
    parts = [(2, int(p), "") if p.isdigit() else (0, 0, p.lower())
             for p in re.split(r"[.\-_+]", version) if p]
    return (*parts, (1, 0, ""))  # (the end: after letters, before more numbers)


def parse_index(data: object, base_url: str) -> DbIndex:
    """An index from its JSON (``{"game", "databases": [{"lang", "url", "sha256", "size",
    "version", ...}]}``); entry URLs may be relative to the index. Raises
    :class:`SyncError` when it is not one."""
    from urllib.parse import urljoin

    from translate.languages import normalize

    if not isinstance(data, dict) or not isinstance(data.get("databases"), list):
        raise SyncError(BAD_INDEX)
    entries = []
    for raw in data["databases"]:
        if not isinstance(raw, dict):
            continue
        lang, url = raw.get("lang"), raw.get("url") or raw.get("file")
        if not isinstance(lang, str) or not isinstance(url, str) or not lang or not url:
            continue
        size = raw.get("size")
        full = urljoin(base_url, url)
        if urlsplit(full).scheme not in ("http", "https"):
            continue
        entries.append(IndexEntry(
            lang=normalize(lang), url=full, name=_text_field(raw, "name"),
            sha256=_text_field(raw, "sha256").lower(),
            size=size if isinstance(size, int) and not isinstance(size, bool) else None,
            version=_text_field(raw, "version"), game_version=_text_field(raw, "game_version"),
            min_app=_text_field(raw, "min_app"),
        ))  # fmt: skip
    if not entries:
        raise SyncError(BAD_INDEX)
    game = data.get("game")
    return DbIndex(base_url, game if isinstance(game, str) else "", entries)


def _text_field(raw: Mapping, key: str) -> str:
    v = raw.get(key)
    return v.strip()[:200] if isinstance(v, str) else ""


# ---- download ---------------------------------------------------------------------------
def download(
    link: str,
    dest: Path,
    progress: Callable[[int, int | None], None] | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download the database behind *link* (a share link, see :func:`direct_url`) to *dest*
    and return *dest* (see :func:`sync_database`)."""
    sync_database(link, dest, progress=progress, timeout=timeout)
    return Path(dest)


def check_for_update(
    link: str, known: Mapping[str, str] | None, timeout: float = 20.0
) -> str:
    """Whether the file behind *link* changed since the download *known* describes (what
    :func:`sync_database` returned), from the server's answer alone: the headers are read
    and the connection closed, nothing is downloaded. "newer", "same", "unknown" (the
    server says nothing that tells: pressing Download checks the file itself) or
    "available" (nothing was downloaded from this link yet, and it answers). Raises
    :class:`SyncError` (the link fails: 404, no connection...).

    Google Drive sends Last-Modified and the size (no ETag) and answers a conditional
    request with the whole file, so the headers are compared here rather than left to the
    server."""
    import requests

    url = direct_url(link)
    was = dict(known or {}) if (known or {}).get("url") == url else {}
    ask = {}
    if was.get("etag"):
        ask["If-None-Match"] = was["etag"]
    if was.get("last_modified"):
        ask["If-Modified-Since"] = was["last_modified"]
    host = urlsplit(url).hostname or url
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True,
                          headers={"User-Agent": USER_AGENT, **ask}) as resp:  # fmt: skip
            if resp.status_code == 304:
                return "same"
            if resp.status_code >= 400:
                raise SyncError(_http_problem(resp.status_code))
            now = _validators(resp.headers)
    except requests.Timeout as e:
        raise SyncError(f"{host} did not answer within {timeout:g} seconds.") from e
    except requests.RequestException as e:
        raise SyncError(f"Could not reach {host} ({e.__class__.__name__}).") from e
    if not was.get("sha256"):
        return "available"
    told = False
    for key in ("etag", "last_modified", "size"):
        if was.get(key) and now.get(key):
            if was[key] != now[key]:
                return "newer"
            told = told or key != "size"  # (the same size alone does not mean the same file)
    return "same" if told else "unknown"


def _validators(headers: Mapping[str, str]) -> dict[str, str]:
    """What tells one version of the file from another: ETag, Last-Modified, the size."""
    found = {
        k: headers[h]
        for k, h in (("etag", "ETag"), ("last_modified", "Last-Modified"))
        if headers.get(h)
    }
    size = _expected_size(headers)
    if size is not None:
        found["size"] = str(size)
    return found


def sync_database(
    link: str,
    dest: Path,
    known: Mapping[str, str] | None = None,
    only_if_newer: bool = False,
    progress: Callable[[int, int | None], None] | None = None,
    timeout: float = 60.0,
    backups: Path | None = None,
    expect_sha256: str = "",
) -> tuple[bool, dict[str, str]]:
    """Bring *dest* up to date with the database behind *link* (a share link, see
    :func:`direct_url`). *backups*: the folder the replaced database's copy goes to (default:
    beside it). Returns (whether *dest* was replaced, what to pass as *known* the
    next time: the direct URL, the server's ETag and Last-Modified, the file's SHA-256).

    With *only_if_newer* and *known* from the last time, the server is asked first whether
    the file changed since (most answer at once when it did not). A file identical to the
    one *dest* holds is never swapped in. Otherwise the file is streamed to
    ``<dest>.download`` in the same folder and checked before it replaces anything: an
    SQLite file with a ``lines`` table that SQLite reads in full. The database it replaces
    is copied to ``<stem>.backup<suffix>`` first. *progress* is called on the calling
    thread with (bytes so far, total bytes or None when the server does not say); an
    exception raised by it cancels the download. *timeout* is how long to wait, in seconds,
    for the server to answer or to send more data (not a limit on the whole download).
    *expect_sha256*: the digest the file must have (from an index): any other file is
    refused. Raises :class:`SyncError`; the temporary file is removed on any failure, and
    *dest* is left untouched unless the new file is good (see :func:`adopt_database`).
    """
    url = direct_url(link)
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".download")
    was = dict(known or {}) if (known or {}).get("url") == url else {}
    ask: dict[str, str] = {}
    if only_if_newer and dest.exists():
        if was.get("etag"):
            ask["If-None-Match"] = was["etag"]
        if was.get("last_modified"):
            ask["If-Modified-Since"] = was["last_modified"]
    try:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise SyncError(f"Could not create the folder {dest.parent} ({_reason(e)}).") from e
        validators = _fetch(url, tmp, progress, timeout, ask)
        if validators is None:  # 304: not changed since the last time
            log.info("%s has not changed", url)
            return False, {**was, "url": url}
        digest = _sha256(tmp)
        if expect_sha256 and digest != expect_sha256.lower():
            raise SyncError(NOT_AS_LISTED)
        state = {"url": url, "sha256": digest, **validators}
        if dest.exists() and digest == was.get("sha256"):
            tmp.unlink(missing_ok=True)
            log.info("%s is the database we have", url)
            return False, state
        adopt_database(tmp, dest, backups=backups)
        tmp.unlink(missing_ok=True)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # never hide the real problem behind a cleanup failure
            log.warning("could not remove %s", tmp)
        raise
    log.info("downloaded %s -> %s (%d bytes)", url, dest, dest.stat().st_size)
    return True, state


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def _fetch(
    url: str,
    tmp: Path,
    progress: Callable[[int, int | None], None] | None,
    timeout: float,
    ask: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Stream *url* into *tmp*. Stops at the first bytes when they are not an SQLite file.
    *ask*: conditional headers; None comes back when the server says nothing changed,
    else its validators (``etag``, ``last_modified``) for the next time."""
    import requests  # imported here: it takes ~0.2 s and only a download needs it

    host = urlsplit(url).hostname or url
    answered = False
    done = 0
    total: int | None = None
    content_type = ""
    head = b""
    try:
        with requests.get(
            url,
            stream=True,
            headers={"User-Agent": USER_AGENT, **(ask or {})},
            timeout=timeout,
            allow_redirects=True,
        ) as resp:
            if resp.status_code == 304 and ask:
                return None
            if resp.status_code >= 400:
                raise SyncError(_http_problem(resp.status_code))
            answered = True
            validators = _validators(resp.headers)
            content_type = resp.headers.get("Content-Type", "").lower()
            total = _expected_size(resp.headers)
            if total is not None and total > MAX_BYTES:
                raise SyncError(TOO_LARGE)
            if progress is not None:
                progress(0, total)
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(CHUNK):
                    if not chunk:
                        continue
                    if len(head) < len(SQLITE_HEADER):
                        head += chunk
                        if len(head) >= len(SQLITE_HEADER) and not head.startswith(SQLITE_HEADER):
                            raise SyncError(_wrong_file(head, content_type))
                    f.write(chunk)
                    done += len(chunk)
                    if done > MAX_BYTES:
                        raise SyncError(TOO_LARGE)
                    if progress is not None:
                        progress(done, total)
    except requests.Timeout as e:
        raise SyncError(
            f"{host} did not answer within {timeout:g} seconds. Check your internet connection "
            "and try again."
        ) from e
    except requests.ConnectionError as e:
        if answered:
            raise SyncError(INTERRUPTED) from e
        raise SyncError(
            f"Could not connect to {host}. Check your internet connection and the link, then "
            "try again."
        ) from e
    except (
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ContentDecodingError,
    ) as e:
        raise SyncError(INTERRUPTED) from e
    except requests.TooManyRedirects as e:
        raise SyncError("The link keeps redirecting and never reaches a file. Check it.") from e
    except requests.RequestException as e:
        raise SyncError(f"The download failed ({e}). Check the link and try again.") from e
    except OSError as e:  # writing the temporary file
        raise SyncError(
            f"Could not save the download in {tmp.parent} ({_reason(e)}). Check that the disk "
            "has free space and that the folder can be written to."
        ) from e
    if not head.startswith(SQLITE_HEADER):  # empty, or fewer bytes than the header
        raise SyncError(_wrong_file(head, content_type))
    if total is not None and done < total:
        raise SyncError(INTERRUPTED)
    return validators


def _http_problem(status: int) -> str:
    if status in (401, 403):
        return (
            f"The file host refused the download (HTTP {status}: access denied). Make sure the "
            "file is shared publicly ('Anyone with the link can view'). If it is, the host may "
            "be limiting downloads of this file: try again later."
        )
    if status in (404, 410):
        return (
            f"The file was not found (HTTP {status}). Check that the whole link was copied and "
            "that the file has not been moved or deleted."
        )
    if status == 429:
        return (
            "The file host is limiting downloads right now (HTTP 429). Wait a few minutes and "
            "try again."
        )
    if status >= 500:
        return f"The file host had a problem (HTTP {status}). Try again later."
    return f"The download failed (HTTP {status}). Check the link and try again."


def _expected_size(headers: Mapping[str, str]) -> int | None:
    """The byte count the server announces, or None. A compressed body is unpacked on the
    way, so its announced length says nothing about the bytes that arrive."""
    if headers.get("Content-Encoding", "identity").strip().lower() not in ("", "identity"):
        return None
    try:
        size = int(headers.get("Content-Length", ""))
    except ValueError:
        return None
    return size if size >= 0 else None


def _wrong_file(head: bytes, content_type: str) -> str:
    low = head[:4096].lower()
    if "html" in content_type or b"<html" in low or b"<!doctype html" in low:
        return WEB_PAGE
    return NOT_A_DATABASE


def _check_database(path: Path) -> None:
    """Raise :class:`SyncError` unless *path* is a translation database SQLite reads in full
    (an SQLite file of at most :data:`MAX_BYTES`, a ``lines`` table with the columns that
    make one, every page and index intact)."""
    try:
        with open(path, "rb") as f:
            head = f.read(len(SQLITE_HEADER))
        size = path.stat().st_size
    except OSError as e:
        raise SyncError(f"Could not read {path.name} ({_reason(e)}).") from e
    if head != SQLITE_HEADER:
        raise SyncError(NOT_A_DATABASE)
    if size > MAX_BYTES:
        raise SyncError(TOO_LARGE)
    try:
        conn = sqlite3.connect(_uri(path, "immutable=1"), uri=True)
        try:
            conn.execute("PRAGMA trusted_schema = OFF")
            kind = conn.execute("SELECT type FROM sqlite_master WHERE name = 'lines'").fetchone()
            columns = {row[1] for row in conn.execute("PRAGMA table_info(lines)")}
            if not kind or kind[0] != "table":
                columns = set()
            intact = (
                bool(columns) and conn.execute("PRAGMA integrity_check(1)").fetchone()[0] == "ok"
            )
        finally:
            conn.close()
    except sqlite3.Error as e:
        raise SyncError(DAMAGED) from e
    if not is_lines_table(columns):
        raise SyncError(NOT_A_DATABASE)
    if not intact:
        raise SyncError(DAMAGED)


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def db_lock(path: Path) -> threading.Lock:
    """One lock per database file (in this process): a write made while the database is
    replaced (a live translation kept) waits, and is never lost with the old file."""
    key = os.path.normcase(str(Path(path).resolve()))
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def adopt_database(src: Path, dest: Path, backups: Path | None = None) -> None:
    """Make the database file *src* (downloaded, chosen by the player, or made by the
    Create window) the one at *dest*, once it passed :func:`_check_database`: the text and
    number values of its rows copied into a fresh database of this app's own schema
    (nothing of the file's own schema runs here: no views, no triggers; any other value is
    dropped, an unknown tier is "machine"), the old one backed up first. Raises
    :class:`SyncError`; *dest* is untouched unless all of it worked, and the old backups
    are thinned out only then (never *src*: it may be a backup being restored). *backups*:
    where the copy of the old one goes (default: beside it)."""
    dest, src = Path(dest), Path(src)
    fresh = dest.with_name(dest.name + ".adopting")
    made: Path | None = None
    done = False
    try:
        _check_database(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fresh.unlink(missing_ok=True)
        _copy_rows(src, fresh)
        with db_lock(dest):
            made = _backup(dest, backups)
            for leftover in ("-journal", "-wal", "-shm"):  # (a crash's: never played into it)
                dest.with_name(dest.name + leftover).unlink(missing_ok=True)
            os.replace(fresh, dest)
            done = True
    except OSError as e:
        raise SyncError(
            f"Could not replace {dest} ({_reason(e)}). Close any program that has the database "
            "open and try again."
        ) from e
    finally:
        fresh.unlink(missing_ok=True)
        if not done and made is not None:
            made.unlink(missing_ok=True)  # (the database is as it was: no copy of it needed)
        if done:
            prune_backups(dest, spare=src, where=backups)


def is_lines_table(columns: set[str]) -> bool:
    """Whether a ``lines`` table with these columns makes a translation database (schema 2,
    or schema 1 of app 1.0-1.1)."""
    return "key" in columns and (
        {"source_text", "translation"} <= columns or {"text_jp", "en_text"} <= columns
    )


def _copy_rows(src: Path, out: Path) -> None:
    from db.store import FORMAT, LINE_COLUMNS, SCHEMA, SCHEMA_VERSION, V1_COLUMNS

    old_name = {new: old for old, new in V1_COLUMNS.items()}
    conn = sqlite3.connect(str(out))
    try:
        conn.executescript(SCHEMA)
        conn.execute("PRAGMA trusted_schema = OFF")
        conn.execute("ATTACH DATABASE ? AS src", (str(src),))
        kinds = dict(conn.execute(
            "SELECT name, type FROM src.sqlite_master WHERE name IN ('lines', 'meta', 'names')"
        ))  # fmt: skip
        if kinds.get("lines") != "table":
            raise SyncError(NOT_A_DATABASE)
        cols = {row[1] for row in conn.execute("PRAGMA src.table_info(lines)")}
        if not is_lines_table(cols):
            raise SyncError(NOT_A_DATABASE)
        # each column of this app's schema from the file's column of that name (or of its
        # schema 1 name)
        source = {c: c if c in cols else old_name.get(c) for c in LINE_COLUMNS}
        have = [c for c in LINE_COLUMNS if source[c] in cols]
        clean = {c: _clean(source[c]) for c in have}
        sel = ", ".join(_FIELD_SQL.get(c, f"{{{c}}}").format(**clean) for c in have)
        # a key twice (only a file made elsewhere can): the best row comes last and stays
        order = f"({source['translation']} IS NOT NULL)"
        if "tier" in cols:
            order += ", CASE tier WHEN 'human' THEN 2 WHEN 'reviewed' THEN 1 ELSE 0 END"
        conn.execute(
            f"INSERT OR REPLACE INTO main.lines ({', '.join(have)}) "
            f"SELECT {sel} FROM src.lines WHERE typeof(key) = 'text' "
            f"AND length(key) BETWEEN 1 AND {MAX_TEXT} ORDER BY {order}"
        )
        names = {row[1] for row in conn.execute("PRAGMA src.table_info(names)")}
        if kinds.get("names") == "table" and {"source", "translation"} <= names:
            kind = _clean("type") if "type" in names else "NULL"
            notes = _clean("notes") if "notes" in names else "NULL"
            conn.execute(
                "INSERT OR REPLACE INTO main.names (source, translation, type, notes) "
                f"SELECT {_clean('source')}, {_clean('translation')}, "
                f"COALESCE({kind}, 'term'), {notes} FROM src.names "
                "WHERE typeof(source) = 'text' AND typeof(translation) = 'text'"
            )
        meta_cols = {row[1] for row in conn.execute("PRAGMA src.table_info(meta)")}
        if kinds.get("meta") == "table" and {"k", "v"} <= meta_cols:
            conn.execute(
                "INSERT OR REPLACE INTO main.meta (k, v) SELECT k, "
                f"substr(v, 1, {MAX_TEXT}) FROM src.meta WHERE typeof(k) = 'text' "
                "AND length(k) <= 100 AND typeof(v) IN ('text', 'null') "
                "AND k NOT IN ('schema_version', 'format')"
            )
        conn.executemany("INSERT OR REPLACE INTO main.meta (k, v) VALUES (?, ?)",
                         [("schema_version", str(SCHEMA_VERSION)), ("format", FORMAT)])
        conn.commit()
        conn.execute("DETACH DATABASE src")
    except sqlite3.Error as e:
        raise SyncError(DAMAGED) from e
    finally:
        conn.close()


def _clean(column: str) -> str:
    """SQL for a column of the incoming file's ``lines`` as this app stores it: a number
    field keeps numbers only, a text field text (a number read as text, cut at
    :data:`MAX_TEXT`); anything else (a blob) is NULL."""
    if column in _NUMBER_FIELDS:
        return f"(CASE WHEN typeof({column}) IN ('integer', 'real') THEN {column} END)"
    return (
        f"(CASE typeof({column}) WHEN 'text' THEN substr({column}, 1, {MAX_TEXT}) "
        f"WHEN 'integer' THEN CAST({column} AS TEXT) WHEN 'real' THEN CAST({column} AS TEXT) "
        "END)"
    )


def backup_database(path: Path, keep: int = KEEP_BACKUPS,
                    where: Path | None = None) -> Path | None:
    """A copy of the database at *path* (``<stem>.<time>.backup<suffix>``) in the folder
    *where* (default: beside it), made through SQLite (a journal a crash left is played
    back first, so the copy is whole); the oldest copies beyond *keep* go."""
    out = _backup(path, where)
    prune_backups(Path(path), keep, where=where)
    return out


def prune_backups(path: Path, keep: int = KEEP_BACKUPS, spare: Path | None = None,
                  where: Path | None = None) -> None:
    """The oldest backups of the database at *path* (in *where*, default beside it) beyond
    the newest *keep* go (never *spare*)."""
    path = Path(path)
    folder = Path(where) if where is not None else path.parent
    olds = sorted(folder.glob(f"{path.stem}.*.backup{path.suffix}"), key=_backup_age)
    spare_key = os.path.normcase(str(spare.resolve())) if spare is not None else None
    for old in olds[:-keep] if keep > 0 else olds:
        if spare_key is not None and os.path.normcase(str(old.resolve())) == spare_key:
            continue
        old.unlink(missing_ok=True)


def _backup_age(p: Path) -> tuple:
    """When a backup was made, from its name: (date, time, n) of <stem>.<date>-<time>[-n].
    backup<suffix> (made in the same second, -1 is newer than none)."""
    middle = p.name.split(".")[-3] if p.name.count(".") >= 3 else p.name
    parts = middle.split("-")
    n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return (parts[0], parts[1] if len(parts) > 1 else "", n)


def _backup(path: Path, where: Path | None = None) -> Path | None:
    path = Path(path)
    if not path.exists():
        return None
    folder = Path(where) if where is not None else path.parent
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = folder / f"{path.stem}.{stamp}.backup{path.suffix}"
    n = 1
    while out.exists():
        out = folder / f"{path.stem}.{stamp}-{n}.backup{path.suffix}"
        n += 1
    try:
        src = sqlite3.connect(str(path))
        try:
            dst = sqlite3.connect(str(out))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except sqlite3.Error:  # not readable as a database: the file as it is
        shutil.copy2(path, out)
    return out


# ---- info -------------------------------------------------------------------------------
def check_file(path: Path) -> dict[str, Any]:
    """What the file at *path* holds (:func:`database_info`), once it passed the checks a
    download passes (:func:`_check_database`); raises :class:`SyncError` saying why it is
    no database the app uses."""
    path = Path(path)
    _check_database(path)
    return database_info(path)


def database_info(path: Path) -> dict[str, Any]:
    """What the database at *path* is and holds, for showing it to the user: the fields of
    :class:`db.store.DbInfo` (game, source_lang, target_lang, display_name, version,
    game_version, lines, translated...; "" when its meta table lacks them), ``updated``
    (meta, else the file's modification time; ISO 8601) and ``size_bytes``. The file is
    opened read-only. Raises :class:`SyncError` when there is no file or it is not a
    translation database."""
    from dataclasses import asdict

    from db.store import Store, table_columns

    path = Path(path)
    if not path.is_file():
        raise SyncError(f"There is no database at {path}. Download one or create one first.")
    try:
        with Store(path, readonly=True) as store:
            if not is_lines_table(table_columns(store.conn, "lines")):
                raise SyncError(f"{path.name} is not a Game Translation HUD database.")
            info = store.info()
    except sqlite3.Error as e:
        raise SyncError(
            f"{path.name} could not be read as a Game Translation HUD database ({e})."
        ) from e
    stat = path.stat()
    out = asdict(info)
    out["updated"] = info.updated or (
        datetime.fromtimestamp(stat.st_mtime, UTC).replace(microsecond=0).isoformat()
    )
    out["size_bytes"] = stat.st_size
    return out


def _uri(path: Path, query: str) -> str:
    """An SQLite URI for *path* (spaces, ``#`` and non-ASCII names are escaped)."""
    return f"{path.resolve().as_uri()}?{query}"


def _reason(e: OSError) -> str:
    return e.strerror or str(e)
