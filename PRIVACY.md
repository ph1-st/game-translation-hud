# Privacy and security

What the app does with your computer, your game and your data, in full.

## What it never does

- It never reads or writes the game's memory, never injects code, never hooks the game's
  process and never sends keystrokes or other input to the game. It reads the game's
  picture from the screen, like a screen recorder, and draws its own window on top.
- It never reads or changes the game's files.
- It never sends pictures of your screen anywhere.
- It has no telemetry, no analytics and no account.

## When it connects to the internet

Only for these, each started by you:

| What | Where to | What is sent |
|---|---|---|
| Downloading a translation database, or asking whether there is a newer one | The link you (or the game's preset) set, e.g. a shared Google Drive folder | An ordinary download request (for a Drive folder, first the list of its files) |
| Installing PLaMo for live translation (one button) | Hugging Face (the model, Preferred Networks' page) and GitHub (llama.cpp's server) | Ordinary download requests, once |
| Installing MeikiOCR, the faster reader on NVIDIA cards (one button) | PyPI (ONNX Runtime's CUDA build, NVIDIA's libraries) and Hugging Face (the models) | Ordinary download requests, once |
| Live translation with Claude or ChatGPT, if you choose one | Anthropic or OpenAI | The text of the lines no database has, a few recent lines of the conversation as context, and the game's names and terms (OpenAI is asked not to store it) |
| Live translation with DeepL, if you choose it | DeepL | The text of the lines no database has (with the game's names already in English) and a few recent lines of the conversation as context |
| Live translation with another model, if you choose one | The server whose address you typed (your own computer, or the service you chose) | The same as for Claude or ChatGPT |
| Testing an API key | That service | A short test request |

Live translation with **PLaMo** runs on your own computer: nothing is sent anywhere. Its server listens on 127.0.0.1 only and ends with the app.

**Streaming (OBS)**, when you switch it on, serves the overlay as a page on 127.0.0.1 only
(port 47811, or the next free one), for OBS on the same computer. It holds what the
overlay draws (the translations and the bits of the game's picture they paint over the
Japanese with) and answers no request from another computer. It stops when you switch it
off or close the app.

Your API keys are kept in your system's credential store (Windows Credential Manager, or
the Secret Service keyring on Linux). Without one, a key is kept in a file in the app's
data folder that only your account can read. They are never written to the settings file
and never sent anywhere but to their own service.

## What stays on your computer

In the app's data folder (`%LOCALAPPDATA%\GameTranslationHUD` on Windows,
`~/.local/share/GameTranslationHUD` on Linux):

- your settings, the games you added, their databases and backups;
- PLaMo and llama.cpp's server, if you installed them (`models/`);
- live translations you received, kept so they show at once next time;
- what the HUD learned about the game's screens (to show menus faster);
- the **miss log**: lines no database had, written down while you played (you can switch
  it off under More). It never leaves your computer unless you send the files yourself,
  for example to whoever makes the game's database;
- the app's log files.

Close the app and delete that folder to remove everything.

## Databases from others

Use database links only from people you trust. The app is careful with what comes in: it
asks before downloading, checks that a file is a translation database of reasonable size
that reads back without damage, and copies only its text and numbers into a database of
its own (nothing in a downloaded file ever runs). A database from an index must match the
SHA-256 the index lists.

## Reporting a security problem

Please report security problems privately through the repository's *Security* tab
(*Report a vulnerability*) rather than in a public issue.
