"""The app's look: a calm dark theme that reads well next to a game. The colours every window
uses live here, in one table."""

from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from app.paths import assets_dir

COLOURS = {
    "ok": "#5fd38d",  # running, a database in use
    "wait": "#f0c060",  # starting, waiting for the game
    "off": "#8a96a8",  # stopped; quiet text
    "problem": "#ff9f43",  # something keeps it from working; a file not used
    "live": "#ef6b6b",  # scrape mode reading the screen
    "accent": "#5b86e6",
    "preset": "#7f9bd8",
}
UI_FONT = "Segoe UI" if sys.platform == "win32" else ""  # elsewhere: the system's own


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    base, window, text = QColor(24, 28, 38), QColor(32, 37, 50), QColor(226, 230, 238)
    roles = {
        QPalette.ColorRole.Window: window,
        QPalette.ColorRole.WindowText: text,
        QPalette.ColorRole.Base: base,
        QPalette.ColorRole.AlternateBase: window,
        QPalette.ColorRole.Text: text,
        QPalette.ColorRole.Button: QColor(44, 51, 68),
        QPalette.ColorRole.ButtonText: text,
        QPalette.ColorRole.Highlight: QColor(92, 140, 230),
        QPalette.ColorRole.HighlightedText: QColor(255, 255, 255),
        QPalette.ColorRole.ToolTipBase: base,
        QPalette.ColorRole.ToolTipText: text,
        QPalette.ColorRole.Link: QColor(130, 175, 255),
        QPalette.ColorRole.PlaceholderText: QColor(120, 130, 148),
    }
    for role, colour in roles.items():
        pal.setColor(role, colour)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(110, 118, 134))
    app.setPalette(pal)
    font = QFont(UI_FONT, 10) if UI_FONT else app.font()
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(STYLE.replace("CHECK", (assets_dir() / "check.png").as_posix()))


STYLE = """
QWidget#start { background: #141924; }
QFrame#sidebar { background: #10141d; border-right: 1px solid #262d3d; }
QLabel#appname { font-size: 15px; font-weight: 600; }
QLabel#section { color: #7d89a0; font-size: 11px; font-weight: 600; letter-spacing: 1px; }
QLabel#gametitle { font-size: 24px; font-weight: 600; }
QLabel#gamename { font-size: 13px; font-weight: 600; }
QLabel#summary { color: #c9d1e0; font-size: 13px; }
QLabel#chip { color: #9fb8f0; background: rgba(91, 134, 230, 38);
              border: 1px solid rgba(91, 134, 230, 110); border-radius: 9px;
              padding: 1px 8px; font-size: 10px; font-weight: 600; }
QLabel#path { color: #9aa6ba; font-family: Consolas, "DejaVu Sans Mono", monospace;
              font-size: 12px; }
QLabel#files { font-size: 12px; }
QLabel#note { color: #9aa6ba; }
QLabel#status { color: #ffe9a8; }
QLabel#subtitle { font-size: 14px; font-weight: 600; }
QLabel#paneltitle { font-size: 15px; font-weight: 600; }
QFrame#card { background: rgba(255, 255, 255, 9); border: 1px solid #2c3446;
              border-radius: 10px; }
QFrame#panel { background: rgba(17, 21, 31, 244); border: 1px solid #343c52;
               border-radius: 12px; }
QFrame#notice { background: rgba(63, 111, 216, 45); border: 1px solid #5b86e6;
                border-radius: 8px; }
QListWidget#games { background: transparent; border: none; outline: none; }
QListWidget#games::item { border-radius: 8px; margin: 1px 0; }
QListWidget#games::item:hover { background: rgba(255, 255, 255, 12); }
QListWidget#games::item:selected { background: rgba(91, 134, 230, 60); }
QPushButton { padding: 6px 12px; border-radius: 6px; border: 1px solid #4a5470; }
QPushButton:hover { border-color: #6f88c8; }
QPushButton:disabled { color: #6e768a; border-color: #3a4258; background: #2a3040; }
QPushButton#primary { background: #3f6fd8; border-color: #5b86e6; color: white;
                      font-weight: 600; }
QPushButton#primary:hover { background: #4b7ce4; }
QPushButton#play { background: #3f6fd8; border: 1px solid #5b86e6; color: white;
                   font-size: 15px; font-weight: 600; padding: 10px 34px; border-radius: 8px; }
QPushButton#play:hover { background: #4b7ce4; }
QPushButton#play:disabled { background: #2a3040; color: #6e768a; border-color: #3a4258; }
QPushButton#flat { border: 1px dashed #3d4660; color: #aab4c6; }
QPushButton#flat:hover { border-color: #6f88c8; color: white; }
QPushButton#danger { border-color: #8a3c3c; color: #ff9d9d; }
QPushButton#danger:hover { background: #8a3c3c; color: white; }
QToolButton#iconbtn { border: 1px solid #343c52; border-radius: 8px; font-size: 15px;
                      padding: 3px 10px; color: #c9d1e0; }
QToolButton#iconbtn:hover { border-color: #6f88c8; color: white; }
QToolButton#linkbtn { border: none; color: #82afff; padding: 2px 2px; }
QToolButton#linkbtn:hover { color: #b4ceff; }
QToolButton#headbtn, QToolButton#closebtn { border: none; border-radius: 6px;
                      padding: 2px 9px; font-size: 16px; color: #c9d1e0; }
QToolButton#headbtn:hover { background: rgba(255, 255, 255, 30); }
QToolButton#closebtn:hover { background: #c94a4a; color: white; }
QGroupBox { border: 1px solid #3a4258; border-radius: 8px; margin-top: 14px;
            padding: 10px 8px 8px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLineEdit, QComboBox { padding: 4px 6px; }
QScrollArea { background: transparent; }
QCheckBox::indicator, QRadioButton::indicator, QListView::indicator {
    width: 14px; height: 14px; border: 1px solid #6a7590; background: #1c2230; }
QCheckBox::indicator, QListView::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #93a9dd; }
QCheckBox::indicator:checked, QListView::indicator:checked {
    background: #4b7ce4; border-color: #6f93ea; image: url(CHECK); }
QRadioButton::indicator:checked { border-color: #6f93ea;
    background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5,
    stop: 0 white, stop: 0.38 white, stop: 0.5 #4b7ce4, stop: 1 #4b7ce4); }
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border-color: #3a4258; background: #202636; }
"""
