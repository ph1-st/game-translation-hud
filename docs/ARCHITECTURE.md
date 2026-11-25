# Architecture

Game Translation HUD shows translations on top of a Japanese game's text (Chinese and
Korean work too): it reads the game's picture, recognises the text (OCR), looks each line
up in a translation database and draws the translation in its place, in a click-through
window of its own. It never touches the
game: no memory reads or writes, no injected code, no hooks, no input sent to it, no files
changed.

The databases are made ahead of time, outside the app: one SQLite file per language,
shared as a download (a file, a shared Google Drive folder, or an index). While playing,
the HUD finds each line on screen in that database, and translates only what is missing
live, if the player wants it: PLaMo on this computer, or the player's own Claude or
ChatGPT key.

## Packages

| Package | What it does |
|---|---|
| `app/` | The desktop app: start window, HUD panel, settings, the game library, the overlay session, the text reader choice, the miss log |
| `hud/` | The overlay: the per-tick pipeline (what to read, matching, covers, dialogue), the pixel watch, the recall memory of menus, painting, the overlay window, the HUD's own app loop |
| `hud/platform/` | The operating system behind one interface: windows, capture without the overlay, click-through overlay, hotkeys (Windows and X11) |
| `capture/` | OCR: the standard reader (RapidOCR on ONNX Runtime's DirectML build), MeikiOCR on CUDA in a helper process; game profiles |
| `db/` | The database: schema, text normalization, templates (`{player}`, `{n}`, `{s}`), lookup (exact, look-alike, prefix, fuzzy), glossary |
| `translate/` | Live translation: the engines (PLaMo on this computer, Claude, ChatGPT), the local model's install and server (`translate/local.py`), the language table |
| `sync/` | Downloading databases (links, shared Drive folders, indexes), checking every file before it is used |
| `i18n/` | The app's own interface in other languages (gettext `.po` files) |
| `games/` | Built-in game profiles (presets): screen layout, dialogue boxes, names |
| `tools/` | Developer tools: the dev HUD, replays of recordings, fixture builders, the string extractor, the public export |
| `packaging/` | Building the Windows and Linux apps |

## The database format (schema 2)

One SQLite file per language. Tables:

- `lines`: `key` (the normalized, templated source text, what the HUD looks up),
  `source_text`, `speaker`, `screen_type` (dialogue, choice, ui...), `is_template`,
  `translation`, `tier` (machine < reviewed < human), `source`, `chapter`, `seq` (script
  order), `game_version`.
- `names`: the game's names and terms in the database's language (the glossary travels
  with it; the HUD shows speaker names from it, then from the game profile's glossary).
- `meta`: what the file is: `format`, `schema_version`, `game`, `source_lang`,
  `target_lang`, `display_name`, `version`, `game_version`, `text_source`, `created`,
  `generator`.

Schema 1 files (app 1.0–1.1: `text_jp`, `speaker_jp`, `en_text`) are migrated when opened
for writing and read through aliases otherwise. A file that comes in (download, a file
the player picks) is checked (SQLite header, size limit, integrity) and only its text and
numbers are copied into a fresh file of the app's own schema: nothing in it ever runs.

An **index** (`index.json`) lists one database per language with its URL, SHA-256, size
and version. A link to an index downloads the entry for the player's language (a variant
serves its language: `es-419` for `es`); the file must match the listed digest. A shared
**Google Drive folder** is an index too: its files named `<game>.<lang>.<version>.sqlite`
are its entries, the newest version of each language (read from the folder's public file
list, with no Google account or API key).

## The model on this computer

PLaMo 3 Translate (Preferred Networks, GGUF) runs in llama.cpp's server, both downloaded
when the player presses Install, pinned to one release and checked by SHA-256
(`translate/local.py`). The server listens on 127.0.0.1, runs hidden, and is tied to the
app (a Windows job object, the parent-death signal on Linux). PLaMo takes its own prompt
format, one line at a time: a short context, the last two lines of the conversation as
source and translation pairs, the glossary entries the line holds
(`translate/backends/plamo.py`). The model is warmed up when the overlay starts, and lines
for it are sent sooner than for a paid service.

## Languages

- The **game's language** (`ja`, `zh-hans`, `zh-hant`, `ko`) chooses the OCR recognizer.
- The **player's language** is chosen per game in the start window. The overlay uses the
  databases in that language; live translations, the context memory and the menu cache are
  kept per language. `translate/languages.py` holds every language the app offers: its
  own name, writing direction, word spacing, capitals, and the word that
  stands for an unnamed player.
- The **interface's language** is the system's or the player's choice (`i18n/`).

## The HUD at runtime

Every tick (`hud/pipeline.py`):

1. **Capture** the game window's picture (`hud/platform`: Desktop Duplication on Windows,
   the window's own contents through XComposite on X11; the overlay and the panel are never
   in it).
2. **Plan**: which rows changed since the last read; the dialogue band first, menus at most
   once a second while a conversation runs. With no conversation up, the whole frame is
   read in one go. While the picture moves (a 3D scene, a list scrolling) reads are spaced
   out to spare the game's GPU; a picture at rest is read at once.
3. **Read** those bands with the OCR, the covered labels blanked out. Detection runs on the
   picture shrunk to 800 rows (`capture/ocr.py`), the dialogue and the names at full size
   (their rows' places set the layout); recognition always reads the original pixels, and
   a crop it has read before, pixel for pixel, comes from memory. On an NVIDIA card the
   player can choose MeikiOCR instead (`capture/meiki.py`, made for Japanese game text,
   about twice as fast): it runs on CUDA in a helper process (`capture/cuda_helper.py`),
   because ONNX Runtime's CUDA build cannot be loaded beside the DirectML one the app
   ships; that build, NVIDIA's libraries and the models (~1.9 GB) are downloaded once,
   after the player agrees (`app/reader.py`), never shipped. What MeikiOCR reads that
   matches nothing is read once more by the standard recognizer, in the same box.
4. **Match** each reading in the databases (`db/lookup.py`): exact, then OCR look-alikes,
   then a line still typing out (prefix), then fuzzy; stacked rows of a paragraph as one
   block.
5. **Cover**: the matched text's exact extent from its strokes, the translation laid out in
   the same place (wider when it must, never over other text), painted in the text's own
   colours; dialogue on frosted glass in the layout's area, typing out with the game.

Between ticks, the **pixel watch** (every 50 ms) compares each cover's stroke cores with
how they looked when it was placed: a cover goes the moment its text goes. The **recall
memory** brings menus back by their pixels before the OCR reads them again (menus only,
never story lines). Readings no database has go to the miss log and, when it is on, to
live translation, once their text held still while it was read (as a cover must); what
comes back is shown faded (provisional) and kept. A label's translation is placed the
moment it comes, from the reading it was made for, without another read.

## Threads

- The **GUI thread** (Qt) owns every widget, the overlay window and the pixel watch.
- The **capture thread** keeps the newest frame; the **OCR worker** runs the pipeline's
  ticks on it; the **backdrop builder** paints covers off the GUI thread.
- App **tasks** (`app.widgets.Task`) run downloads and installs on daemon threads and
  report through Qt signals; widgets are only touched on the GUI thread.
  Settings may be saved from any thread (`Settings.save` is atomic and locked).

## Hard rules

- Never read or write game memory, inject code, hook the process, send input to the game
  or modify game files.
- The overlay excludes itself from screen capture; it is click-through and always on top.
- Only text is ever sent anywhere (live translation with Claude or ChatGPT when the player
  switches it on), never pictures; PLaMo sends nothing. The miss log stays on the player's computer.
- Everything game-specific lives in a game profile; the core is game-agnostic.
