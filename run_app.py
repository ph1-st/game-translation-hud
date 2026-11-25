"""Entry point of the packaged app (PyInstaller) and a shortcut for: python -m app."""

import sys

from app.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
