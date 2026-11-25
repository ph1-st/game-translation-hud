"""The dialogue band, owned by exactly one component.

Everything the OCR reads in the dialogue territory (the profile's dialogue region and the
speaker bands) belongs here and nowhere else: the UI pipeline never reads or covers it.

* A line of dialogue (:mod:`hud.dialogue.reader`) is shown on frosted glass
  (:mod:`hud.dialogue.panes`): the scene behind blurred, the translation in the colours of
  the Japanese. The whole line is known from its first characters, so it is laid out once,
  the same way in the same place for every line of a layout (:mod:`hud.dialogue.scene`),
  and types out with the Japanese at an even pace that ends just after it (never a pause:
  a newer reading only changes the pace, from where the text is).
* The speaker's name (:mod:`hud.dialogue.names`) is a fixed box above the line, a size up.
  The database's speaker puts it up with the line; reading the name on screen afterwards
  ties it to its pixels: it goes the moment they change (another speaker) and comes back
  by them within a frame. A name in roman letters (the player's) is shown as it is.
* A layout with a divider (:mod:`hud.dialogue.divider`: the line with a star under the
  name) is up the moment the star is on screen, before any OCR, and over once it has been
  gone a moment with no line shown.
* A line goes the moment its Japanese strokes vanish or change. The scene stays through
  the gaps between lines, so the next line comes in under the glass; it ends when the band
  has held no dialogue text for a moment, or when a menu's labels appear on its panel.
* Anything else read in the territory (a menu's tab bar, a button) is a label: a tight
  cover made like every UI cover, on this component's own board.

Set-region mode: the same for a dialogue area the player chose for any game (a visual
novel's text box, an RPG Maker message window: one layout, ``all_dialogue``: every text in
it is dialogue, even on the game's own box). The glass is one window over the whole area
(mode ``fixed``) or one pane that grows with the text and never shrinks (mode ``grow``);
with ``keep`` it stays once it is up. The game's rows may be centred or left-aligned
(``align``, found from the first line when ``auto``): the text and the glass follow.
"""

from hud.dialogue.component import BALANCED, CONFIRMATIONS, INSTANT, DialogueComponent
from hud.dialogue.names import is_roman
from hud.dialogue.panes import FIXED, GROW, MODES
from hud.dialogue.reader import is_dialogue, reads_as
from hud.dialogue.scene import DialogueRead, Pane, Scene, Shown, Speaker

__all__ = [
    "BALANCED",
    "CONFIRMATIONS",
    "FIXED",
    "GROW",
    "INSTANT",
    "MODES",
    "DialogueComponent",
    "DialogueRead",
    "Pane",
    "Scene",
    "Shown",
    "Speaker",
    "is_dialogue",
    "is_roman",
    "reads_as",
]
