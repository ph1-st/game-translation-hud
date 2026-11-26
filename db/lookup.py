"""Runtime lookup: OCR text -> database line, as early as possible.

Order of attempts (all on templated, normalized keys; :func:`db.templates.rekey` brings the
keys a database was made with up to the current normalization):

1. exact key. A key the database holds without English gives way to a look-alike line
   that has English (see 2): an OCR misreading stored as a line of its own must not hide
   the line it misreads.
2. look-alikes folded (voiced marks, ソ/ン, シ/ツ: :func:`db.normalize.fold_key`), then
   punctuation left out as well (:func:`bare_key`: a dropped 。 or an extra one): the line
   a misread key stands for, when it stands for one English only.
3. prefix: the reading is the start of stored lines (a line still typing out). Taken with
   a sequence hint (the line shown before: what follows it in the script wins), or when
   every candidate means one English; otherwise the reading is ambiguous and waits for
   more characters: no guessing, and no fuzzy match standing in for it.
4. fuzzy: rapidfuzz ratio against the folded keys of about the reading's length (OCR
   errors), searched outside Python's lock when there are many (the overlay stays
   smooth); then a start with OCR errors: the keys that begin with the reading's first
   characters, compared over the reading's length.

The HUD reads the same text over and over (a label stays on screen): a reading is looked up
once, its result kept until the database changes.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from db.normalize import fold_key
from db.store import Line, Store
from db.templates import (
    PLAYER,
    WORD,
    Templated,
    capture,
    fill,
    literal_key,
    player_in,
    rekey,
    templatize,
    word_pattern,
)

_BADGE_RE = re.compile(r"^[!-~]{1,6}(?=[^!-~])|(?<=[^!-~])[!-~]{1,6}$")
# a count after a name: an item received or dropped ("繊維×2", "Fiber x2"), an upgrade ("+5")
_COUNT_RE = re.compile(r"\s*([×xX✕✖＊*+＋])\s*([0-9０-９]{1,6})\s*$")
_PUNCT_RE = re.compile(r"[。、！？!?…「」『』()（）・,.:;~〜\"'“”]")
_WIDE_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
MEMO_SIZE = 8192  # readings whose result is kept
PREFIX_LIMIT = 200  # candidates looked at for a reading that starts lines
SEQ_AHEAD = 6  # with a sequence hint: candidates this far on in the script
CDIST_FROM = 2000  # this many keys of about the right length: searched off Python's lock...
FUZZY_WORKERS = 2  # ...on this many threads (the game needs the rest)
START_LIMIT = 500  # keys that begin like a misread start, compared at most
_TIERS = {"human": 3, "reviewed": 2, "machine": 1}
SEQ_SPAN = 100_000_000  # line numbers of the next database start this much higher


def _fold(key: str) -> str:
    """A key with look-alikes folded, the player's placeholder one character (a reading
    holds a name of a few characters there, not the eight of "{player}")."""
    return fold_key(key).replace(PLAYER, "☆")


def bare_key(key: str) -> str:
    """A key with look-alikes folded and punctuation left out."""
    return _PUNCT_RE.sub("", _fold(key))


@dataclass
class Match:
    line: Line
    kind: str  # exact | prefix | fuzzy
    score: float  # 0-1
    templated: Templated
    player_fallback: str = "you"  # {player} when no name is known (in the target language)
    count: str = ""  # a count the reading had after the line (" ×2"), kept in the English
    prompts: re.Pattern[str] | None = None  # the game's key prompts (Lookup's key_prompts)

    @property
    def translation(self) -> str | None:
        """The line's translation with this reading's values filled in."""
        if self.line.translation is None:
            return None
        text = self.line.translation
        if self.prompts is not None:
            text = named_prompts(text, self.prompts)
        return fill(text, self.templated.values, self.templated.player or self.player_fallback,
                    self.templated.words) + self.count  # fmt: skip


def prompt_label(action: str) -> str:
    """A key prompt's action as words: "MainUltimateSkill" -> "[Ultimate Skill]"."""
    name = re.sub(r"^(?:Main|Common_|UI_)", "", action) or action
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_", " ", name).split()
    return "[" + " ".join(words) + "]"


def named_prompts(text: str, prompts: re.Pattern[str]) -> str:
    """*text* with each key prompt the game shows as an icon ("MainInput/MainEvade") named
    by its action ("[Evade]"); the same one several times in a row once."""
    out = prompts.sub(lambda m: prompt_label(m.group("action")), text)
    return re.sub(r"(\[[^\]]+\])(?:[\s/+]*\1)+", r"\1", out)


def _without_badges(text: str, numbers: bool = False) -> str:
    """A reading without the badges the OCR glued to it (NEW, MAX, UP!). A number is no
    badge (a level or a count is part of the label) unless *numbers*: an icon read as a
    digit ("6星祝")."""

    def keep(m: re.Match[str]) -> str:
        return m.group() if m.group().isdigit() and not numbers else ""

    return _BADGE_RE.sub(keep, text).strip()


class Lookup:
    def __init__(
        self,
        lines: list[Line],
        player_name: str | None = None,
        min_prefix: int = 4,
        fuzzy_cutoff: float = 85.0,
        player_fallback: str = "you",
        key_prompts: str | re.Pattern[str] | None = None,
    ) -> None:
        """*player_fallback*: what stands for the player's name in a translation when no
        name is set and the reading does not show it (``translate.languages.you``).
        *key_prompts*: how the game's text names a key it shows as an icon (a pattern with
        an ``action`` group, from the game's profile): its lines are found without it, and
        their translations name the action."""
        self.player_name = player_name
        self.player_fallback = player_fallback
        self.prompts = re.compile(key_prompts) if isinstance(key_prompts, str) else key_prompts
        self.min_prefix = min_prefix
        self.fuzzy_cutoff = fuzzy_cutoff
        self.by_key: dict[str, Line] = {}
        for ln in lines:
            key = rekey(ln.key)
            if self.prompts is not None:
                key = self.prompts.sub("", key)
            old = self.by_key.get(key)
            if old is None or _rank(ln) > _rank(old):
                self.by_key[key] = ln
        self.keys: list[str] = sorted(self.by_key)
        self.by_fold: dict[str, list[Line]] = {}
        self.by_bare: dict[str, list[Line]] = {}
        for key, ln in self.by_key.items():
            self.by_fold.setdefault(_fold(key), []).append(ln)
            self.by_bare.setdefault(bare_key(key), []).append(ln)
        # the fuzzy search: folded keys by length (a reading is compared with the keys of
        # about its length only), and in order (keys that begin like a misread start)
        by_len = sorted((len(f), f, k) for k, f in ((k, _fold(k)) for k in self.by_key))
        self._lens = np.array([n for n, _f, _k in by_len], np.int32)
        self._len_folds = [f for _n, f, _k in by_len]
        self._len_keys = [k for _n, _f, k in by_len]
        starts = sorted((_fold(k), k) for k in self.by_key)
        self._start_folds = [f for f, _k in starts]
        self._start_keys = [k for _f, k in starts]
        # lines with a value the game fills in ({s}): matched by pattern
        self._words: dict[str, re.Pattern] = {}
        for key in self.by_key:
            self._add_word_pattern(key)
        self.last_seq: int | None = None  # sequence hint from the previous match
        self._memo: OrderedDict[str, Match | None] = OrderedDict()
        self.stats: Counter[str] = Counter()

    @classmethod
    def from_db(cls, path: str | Path, **kw: object) -> "Lookup":
        with Store(path, readonly=True) as store:
            lines = list(store.iter_lines())
        return cls(lines, **kw)  # type: ignore[arg-type]

    @classmethod
    def from_dbs(cls, paths: list[str | Path], **kw: object) -> "Lookup":
        """Several databases as one, the first winning where two have a line (a better
        tier still wins). Each keeps its own script order (a database's line numbers are
        kept apart from the next one's)."""
        lines: list[Line] = []
        for i, path in enumerate(paths):
            with Store(path, readonly=True) as store:
                for ln in store.iter_lines():
                    if ln.seq is not None:
                        ln.seq += i * SEQ_SPAN
                    lines.append(ln)
        return cls(lines, **kw)  # type: ignore[arg-type]

    def add(self, line: Line) -> None:
        """A line that became known while the HUD runs (a live translation)."""
        key = rekey(line.key)
        old = self.by_key.get(key)
        if old is None:
            bisect.insort(self.keys, key)
            folded = _fold(key)
            i = int(np.searchsorted(self._lens, len(folded), side="right"))
            self._lens = np.insert(self._lens, i, len(folded))
            self._len_folds.insert(i, folded)
            self._len_keys.insert(i, key)
            j = bisect.bisect_left(self._start_folds, folded)
            self._start_folds.insert(j, folded)
            self._start_keys.insert(j, key)
            self._add_word_pattern(key)
        for index, k in ((self.by_fold, _fold(key)), (self.by_bare, bare_key(key))):
            same = index.setdefault(k, [])
            if old is not None and old in same:
                same.remove(old)
            same.append(line)
        self.by_key[key] = line
        self._memo.clear()

    def _add_word_pattern(self, key: str) -> None:
        pattern = word_pattern(key)
        if pattern is not None:
            self._words[key] = pattern

    # ---- candidates ----------------------------------------------------------------------
    def prefix_candidates(self, key: str, limit: int = PREFIX_LIMIT) -> list[Line]:
        i = bisect.bisect_left(self.keys, key)
        out: list[Line] = []
        while i < len(self.keys) and self.keys[i].startswith(key) and len(out) < limit:
            out.append(self.by_key[self.keys[i]])
            i += 1
        return out

    def _pick_by_sequence(self, cands: list[Line], after: int | None) -> Line | None:
        """The candidate that comes next in the script after the line shown before."""
        if after is None:
            return None
        ahead = [c for c in cands if c.seq is not None and 0 < c.seq - after <= SEQ_AHEAD]
        if ahead:
            return min(ahead, key=lambda c: c.seq - after)  # type: ignore[operator]
        return None

    def _lookalike(self, key: str) -> Line | None:
        """The line a misread key stands for: look-alikes folded, then punctuation left
        out too; a line with English when the candidates mean one English only."""
        for index, k in ((self.by_fold, _fold(key)), (self.by_bare, bare_key(key))):
            cands = index.get(k)
            if not cands:
                continue
            english = {c.translation for c in cands if c.translation}
            if len(english) == 1:
                return next(c for c in cands if c.translation)
            if not english:
                return cands[0]  # known, without English
        return None

    # ---- public --------------------------------------------------------------------------
    def match(
        self, ocr_text: str, remember: bool = True, after_seq: int | None = None
    ) -> Match | None:
        """The database line *ocr_text* shows. *after_seq*: the script position of the
        line shown before (a dialogue hint: a reading that starts several lines is the one
        that comes next)."""
        hint = after_seq if after_seq is not None else self.last_seq
        memo = hint is None
        self.stats["lookups"] += 1
        if memo and ocr_text in self._memo:
            self.stats["memo_hits"] += 1
            self._memo.move_to_end(ocr_text)
            result = self._memo[ocr_text]
        else:
            result = self._match(ocr_text, hint)
            if result is None or result.kind != "exact":
                # an item and its count: the item exactly beats a near line with the count
                result = self._counted(ocr_text, hint) or result
            if result is None:
                # badges glued onto a label by the OCR ("...NEW", "MAX...") hide the text
                stripped = _without_badges(ocr_text)
                if stripped and stripped != ocr_text:
                    result = self._match(stripped, hint)
            if result is None:
                # an icon read as a digit ("6星祝"): taken only when the rest is a line as
                # it is (never the start of a line with another number: レベル10 for レベル5)
                bare = _without_badges(ocr_text, numbers=True)
                if bare and bare != ocr_text:
                    found = self._match(bare, hint)
                    result = found if found is not None and found.kind == "exact" else None
            if memo:
                self._memo[ocr_text] = result
                if len(self._memo) > MEMO_SIZE:
                    self._memo.popitem(last=False)
        if result is not None and remember:
            self.last_seq = result.line.seq
        self.stats[result.kind if result is not None else "misses"] += 1
        return result

    def _counted(self, ocr_text: str, hint: int | None) -> Match | None:
        """A line with a count after it ("繊維×2": two Fiber received): the line as it is,
        the count kept in the English ("Fiber ×2"). Only an exact line: the start of a
        longer one is no item."""
        m = _COUNT_RE.search(ocr_text)
        if m is None or m.start() == 0:
            return None
        found = self._match(ocr_text[: m.start()], hint)
        if found is None or found.kind != "exact":
            return None
        sign = "+" if m.group(1) in "+＋" else "×"
        digits = m.group(2).translate(_WIDE_DIGITS)
        found.count = f" {sign}{digits}" if sign == "×" else f" +{digits}"
        return found

    def diagnostics(self) -> dict[str, int]:
        return dict(self.stats)

    def _match(self, ocr_text: str, hint: int | None) -> Match | None:
        t = templatize(ocr_text, self.player_name)
        # the player's name fills the template's {player} whether or not this reading
        # holds it yet (a line read while it types out)
        t.player = t.player or self.player_name
        key = t.key
        if not key:
            return None
        result: Match | None = None
        line = self.by_key.get(key)
        if line is None or line.translation is None:
            line = self._lookalike(key) or line
        if line is None and t.values:  # a line kept with its numbers as they are
            line = self.by_key.get(rekey(literal_key(ocr_text, self.player_name)))
        if line is None:
            line = self._with_words(ocr_text, t)
        # with a sequence hint the candidate set is tiny, so a shorter prefix is safe
        min_prefix = self.min_prefix - 1 if hint is not None else self.min_prefix
        if line is not None:
            result = self._hit(line, "exact", 1.0, t)
        elif len(key) >= min_prefix:
            cands = self.prefix_candidates(key)
            if cands:
                pick = self._pick_by_sequence(cands, hint)
                if pick is None and len(cands) < PREFIX_LIMIT:
                    english = {c.translation for c in cands if c.translation}
                    if len(english) == 1:
                        pick = next(c for c in cands if c.translation)
                    elif len(cands) == 1:
                        pick = cands[0]
                if pick is None:
                    return None  # the start of lines that differ: wait for more of it
                result = self._hit(pick, "prefix", len(key) / max(len(pick.key), 1), t)
        if result is None and len(key) >= self.min_prefix:
            result = self._fuzzy(key, t)
        if result is not None and PLAYER in result.line.key and not t.player:
            # no name set: the one the reading shows where the line has it
            t.player = player_in(ocr_text, rekey(result.line.key))
        return result

    def _with_words(self, ocr_text: str, t: Templated) -> Line | None:
        """The line whose key has ``{s}`` that the reading is, with the words it shows
        there (in *t*), each in the database's translation when it knows the word."""
        for key, pattern in self._words.items():
            if not pattern.fullmatch(t.key):
                continue
            line = self.by_key[key]
            found = capture(ocr_text, line.key)
            if found is None:
                continue
            player, words = found
            t.words = [self._word(w) for w in words]
            if player and not t.player:
                t.player = player
            return line
        return None

    def _word(self, word: str) -> str:
        """A word the game filled into a line, in the database's translation when the
        database has the word as a line of its own (an item, a place)."""
        line = self.by_key.get(templatize(word).key)
        if line is not None and line.translation and WORD not in line.key:
            return line.translation
        return word

    def _fuzzy(self, key: str, t: Templated) -> Match | None:
        # longer strings tolerate more OCR noise: 85 below 6 chars, 80 below 12, 75 beyond
        cutoff = self.fuzzy_cutoff - (0 if len(key) < 6 else 5 if len(key) < 12 else 10)
        q = _fold(key)
        n, c = len(q), cutoff / 100.0
        # a ratio of c needs a length within these bounds
        lo, hi = math.floor(n * c / (2 - c)), math.ceil(n * (2 - c) / c)
        i0 = int(np.searchsorted(self._lens, lo, side="left"))
        i1 = int(np.searchsorted(self._lens, hi, side="right"))
        hit: tuple[int, float] | None = None
        if i1 - i0 > CDIST_FROM:
            scores = process.cdist(
                [q], self._len_folds[i0:i1], scorer=fuzz.ratio, score_cutoff=cutoff,
                workers=FUZZY_WORKERS,
            )[0]  # fmt: skip
            j = int(np.argmax(scores))
            if scores[j] >= cutoff:
                hit = (i0 + j, float(scores[j]))
        elif i1 > i0:
            r = process.extractOne(
                q, self._len_folds[i0:i1], scorer=fuzz.ratio, score_cutoff=cutoff
            )
            if r is not None:
                hit = (i0 + int(r[2]), float(r[1]))
        if hit is not None:
            i, score = hit
            line = self.by_key[self._len_keys[i]]
            twins = [ln for ln in self.by_fold.get(self._len_folds[i], []) if ln.translation]
            if len({ln.translation for ln in twins}) > 1:  # look-alike lines: the one read wins
                line = max(twins, key=lambda ln: fuzz.ratio(key, ln.key))
            return self._hit(line, "fuzzy", score / 100.0, t)
        return self._misread_start(q, t)

    def _misread_start(self, q: str, t: Templated) -> Match | None:
        """A line still typing out, read with an error: the keys that begin with its first
        characters, compared over the reading's length (one English among them only)."""
        n = len(q)
        if n < 2 * self.min_prefix:
            return None
        head = q[:2]
        j0 = bisect.bisect_left(self._start_folds, head)
        j1 = bisect.bisect_left(self._start_folds, head + "￿")
        if not 0 < j1 - j0 <= START_LIMIT:
            return None
        cut = self.fuzzy_cutoff + 5
        found = []
        for j in range(j0, j1):
            fk = self._start_folds[j]
            if len(fk) > n:
                s = fuzz.ratio(q, fk[:n])
                if s >= cut:
                    found.append((self.by_key[self._start_keys[j]], s))
        english = {ln.translation for ln, _s in found if ln.translation}
        if len(english) != 1:
            return None
        line, s = max(((ln, s) for ln, s in found if ln.translation), key=lambda p: p[1])
        return self._hit(line, "prefix", s / 100.0 * n / max(len(line.key), 1), t)

    def _hit(self, line: Line, kind: str, score: float, t: Templated) -> Match:
        return Match(line, kind, score, t, self.player_fallback, prompts=self.prompts)

    def reset_sequence(self) -> None:
        self.last_seq = None


def _rank(line: Line) -> tuple[int, int, int]:
    """Which of two lines with one key wins: the one with English, the higher tier, one
    from a database over a live translation."""
    live = line.source == "live"
    return (1 if line.translation else 0, _TIERS.get(line.tier, 0), 0 if live else 1)
