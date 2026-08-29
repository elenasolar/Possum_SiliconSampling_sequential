"""
Parse PoSSUM-style structured model output into item answers.

Expected format (stars optional, case-insensitive keys, order of fields flexible)::

    **title: TRUST_COMPETENT**
    **explanation: ...**
    **answer: 72**
    **speculation: 40**

For slider items the answer must be an integer 0-100 (we accept "72", "72/100", "72%", "72.0",
"about 72"). For choice items the answer must be one of the item's symbols (we also accept
the option label, e.g. "$3", "Yes", "No", "3 dollars").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.codebook import Item

_TITLE_SPLIT = re.compile(r"(?:^|\n)\s*\**\s*title\s*[:：]\s*", re.I)
_FIELD = {
    "answer": re.compile(r"\**\s*(?:answer|selected answer|response|value)\s*[:：]\s*\**\s*(.+?)\s*\**\s*(?:\n|$)", re.I),
    "speculation": re.compile(r"\**\s*speculation(?:\s*level)?\s*[:：]\s*\**\s*(.+?)\s*\**\s*(?:\n|$)", re.I),
    "explanation": re.compile(r"\**\s*explanation\s*[:：]\s*\**\s*(.+?)\s*\**\s*(?=\n\s*\**\s*(?:answer|speculation|symbol|category|title)\s*[:：]|\Z)", re.I | re.S),
    "symbol": re.compile(r"\**\s*symbol\s*[:：]\s*\**\s*(.+?)\s*\**\s*(?:\n|$)", re.I),
    "category": re.compile(r"\**\s*category\s*[:：]\s*\**\s*(.+?)\s*\**\s*(?:\n|$)", re.I),
}
_INT = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class ParsedItem:
    title: str
    answer: object = None          # int for sliders, symbol str for choices
    speculation: int | None = None
    explanation: str = ""
    raw_answer: str = ""
    ok: bool = False
    error: str = ""


@dataclass
class ParseResult:
    items: dict[str, ParsedItem] = field(default_factory=dict)  # by title
    missing: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing and not self.invalid


def _first_int(s: str, lo: int, hi: int) -> int | None:
    m = _INT.search(s.replace(",", ""))
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    if v != v:  # nan
        return None
    v = int(round(v))
    if v < lo or v > hi:
        return None
    return v


def _match_choice(raw: str, item: Item) -> str | None:
    s = raw.strip().strip("*").strip()
    s_low = s.lower().rstrip(").:")
    # exact symbol
    for sym, _label, _val in item.choices:
        if s_low == sym.lower() or s_low.startswith(sym.lower() + ")") or s_low.startswith(sym.lower() + " "):
            return sym
    # symbol appears anywhere as a token
    for sym, _label, _val in item.choices:
        if re.search(rf"\b{re.escape(sym)}\b", s, re.I):
            return sym
    # label match (longest first to avoid "$1" matching "$10")
    for sym, label, _val in sorted(item.choices, key=lambda c: -len(c[1])):
        core = label.split(" (")[0].strip().lower()
        if core and (s_low == core or re.search(rf"(?<![\w$]){re.escape(core)}(?!\d)", s_low)):
            return sym
    # yes/no
    if {c[1].lower() for c in item.choices} >= {"yes", "no"}:
        if re.search(r"\byes\b|\bsubscribed\b|\btrue\b", s_low) and not re.search(r"\bnot\b|\bno\b", s_low):
            return next(c[0] for c in item.choices if c[1].lower() == "yes")
        if re.search(r"\bno\b|\bnot\b|\bfalse\b|\bdid not\b", s_low):
            return next(c[0] for c in item.choices if c[1].lower() == "no")
    # bare dollar amount / integer for donation-type choices
    m = re.search(r"\$?\s*(\d{1,2})(?:\s*(?:usd|dollars?|\$))?", s_low)
    if m and item.choices and all(c[2] == i for i, c in enumerate(item.choices)):
        v = int(m.group(1))
        if 0 <= v < len(item.choices):
            return item.choices[v][0]
    return None


def parse_output(text: str, items: list[Item]) -> ParseResult:
    """Parse ``text`` for the given ``items``. Never raises."""
    res = ParseResult()
    by_title = {it.title.upper(): it for it in items}
    text = (text or "").replace("\r", "")
    chunks = _TITLE_SPLIT.split(text)
    seen: dict[str, int] = {}
    for chunk in chunks[1:]:
        head, _, body = chunk.partition("\n")
        title = head.strip().strip("*").strip().rstrip(":").strip().upper()
        # tolerate "TITLE (some note)" or "TITLE:" variants
        title = re.split(r"[\s(]", title)[0] if title not in by_title else title
        item = by_title.get(title)
        if item is None:
            # try prefix match against known titles (e.g. "TRUST_COMPETENT_1")
            cands = [t for t in by_title if title.startswith(t)]
            if len(cands) == 1:
                item = by_title[cands[0]]
                title = cands[0]
            else:
                continue
        seen[title] = seen.get(title, 0) + 1
        if seen[title] == 2:
            res.duplicated.append(title)
        pi = ParsedItem(title=title)
        m_ans = _FIELD["answer"].search(body)
        raw = m_ans.group(1) if m_ans else ""
        if not raw:  # PoSSUM-style symbol/category fallback
            m_sym = _FIELD["symbol"].search(body) or _FIELD["category"].search(body)
            raw = m_sym.group(1) if m_sym else ""
        pi.raw_answer = raw.strip()
        m_spec = _FIELD["speculation"].search(body)
        if m_spec:
            pi.speculation = _first_int(m_spec.group(1), 0, 100)
        m_exp = _FIELD["explanation"].search(body)
        if m_exp:
            pi.explanation = " ".join(m_exp.group(1).split())[:2000]
        if not raw:
            pi.error = "no answer field"
        elif item.kind == "slider":
            v = _first_int(raw, 0, 100)
            if v is None:
                pi.error = f"slider answer not an integer 0-100: {raw!r}"
            else:
                pi.answer, pi.ok = v, True
        else:
            sym = _match_choice(raw, item)
            if sym is None:
                pi.error = f"choice answer not recognised: {raw!r}"
            else:
                pi.answer, pi.ok = sym, True
        # On repeats keep the LAST valid occurrence: reasoning models (Qwen3 without a
        # reasoning parser) may drift a draft "title:/answer:" block into their visible
        # chain of thought before the real formatted answer; the final block must win.
        prev = res.items.get(title)
        if prev is None or pi.ok or not prev.ok:
            res.items[title] = pi
    for t in by_title:
        if t not in res.items:
            res.missing.append(t)
        elif not res.items[t].ok:
            res.invalid.append(t)
    return res


def answers_by_key(res: ParseResult, items: list[Item]) -> dict[str, object]:
    """{item.key: answer} for successfully parsed items."""
    out = {}
    for it in items:
        pi = res.items.get(it.title)
        if pi and pi.ok:
            out[it.key] = pi.answer
    return out
