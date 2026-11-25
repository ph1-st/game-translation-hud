# Contributing

Thanks for helping. This file covers setting up, testing, translating the app, adding a
game, and the style the code keeps. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains
how the parts fit together.

## Setting up

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11 (pinned in
`.python-version`).

```
uv sync --extra dev
uv run python -m app            # the app
uv run pytest                   # the tests
uv run ruff check .             # the linter
```

On Windows, Qt tests that run without a screen need the system's fonts:
`QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR=C:/Windows/Fonts uv run pytest`. On Linux, run
the X11 tests under Xvfb (`xvfb-run -a uv run pytest`).

Some tests replay screenshots of a game and its translated lines (`tests/frames/`,
`tests/fixtures/`); the public repository does not include them, and those tests skip.

## Tests

- Every change comes with tests; `uv run pytest` and `uv run ruff check .` pass before a
  pull request.
- Tests never open a window on the screen (`WA_DontShowOnScreen`, the offscreen platform)
  and never call a paid API (fake clients and local HTTP servers stand in).
- A test that writes the app's data folder sets `GTHUD_DATA` to a temporary folder (the
  shared fixture does this).

## Translating the app

The app's own words live in `i18n/locale/<language>.po` (gettext). To add a language,
copy `i18n/locale/messages.pot` to `i18n/locale/<code>.po`, set its header
(`Language`, `Plural-Forms`) and translate the entries; keep every `{placeholder}` as it
is. After changing strings in the code, run:

```
uv run python tools/i18n_extract.py           # update messages.pot and every .po
uv run python tools/i18n_extract.py --check   # is anything untranslated?
```

In code, wrap a string the player reads in `_()` (from `i18n`), use `ngettext()` for
plurals and named placeholders (`_("{n} lines").format(n=n)`); mark constants with `N_()`
and translate them where they are shown.

## Adding a game

A game's specifics live in `games/<id>/`: `profile.yaml` (window, screen regions, dialogue
layouts) and `glossary.yaml`. The core code stays game-agnostic. Any game also works without a profile: the player adds
it in the app and sets its dialogue area.

## Code style

- Clean, small modules and functions with type hints; plain docstrings that say what a
  thing is for and why, not a story of how it came to be.
- Numbers with a meaning are named constants with a one-line reason.
- No dead code, no commented-out code, no leftovers of earlier designs.
- User-facing text goes through `i18n`; errors the player sees say what went wrong and
  what to do.
- Line length 100 (ruff enforces it).

## The rules the app keeps

The app never reads or writes game memory, injects code, hooks the game's process, sends
it input or changes its files; it only reads the screen and draws its own window. Only
text is ever sent anywhere, and only with the player's choice. A contribution that breaks
one of these rules cannot be accepted.
