"""
Per-item answer *distributions* from token logprobs, replacing repeated sampling.

Instead of calling the model K times per item and averaging the sampled answers,
one call with ``{"logprobs": true, "top_logprobs": 20}`` returns, at every output
token position, the model's top-20 alternatives with probabilities. Qwen (like
most modern tokenizers with digit splitting) writes an integer 0-100 as 1-3
single-digit tokens, so top-20 covers the *entire* local branching of the answer
(digits 0-9 plus the ``**`` terminator) and the answer's probability
distribution can be reconstructed from a single response.

Works on the shared PoSSUM output format::

    **title: CONCERN_1**
    **explanation: ...**
    **answer: 72**
    **speculation: 40**

What is exact and what is approximated
--------------------------------------
The API reveals next-token distributions only along the *sampled* path:

* P(first digit) is **exact** (position 1 shows all 10 digit alternatives);
* P(next token | first digit) is observed only for the first digit that was
  actually sampled. For the other nine first-digit branches we **borrow** the
  sampled branch's conditional continuation ("borrowed-conditional
  approximation"). ``AnswerDist.borrowed`` reports the pmf mass that rests on
  borrowed factors (0.0 = fully exact); it is small whenever the sampled first
  digit carries most probability.
* ``100`` is resolved exactly when the sampled path went ``1,0,...``; on
  borrowed branches three-digit numbers are folded into their two-digit prefix.
* The distribution is conditional on everything sampled before it in the same
  response -- in particular on the sampled explanation text.

Estimators derived from the pmf (all returned; the caller picks):

    sampled   the answer as actually sampled (unchanged current behavior)
    expected  probability-weighted mean of 0..100
    mode      argmax of the pmf ("take the highest")

Note for survey use: ``expected``/``mode`` remove the response noise of one
draw, which sharpens treatment-effect estimates but also removes human-like
answer variability -- with identical prompts every agent lands on the same
number, so distribution-level realism metrics (within-condition SD) must then
come from the stored pmf, not from the point estimates.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TITLE_RE = re.compile(r"\*\*title:\s*([A-Za-z0-9_\- ]+?)\s*\*\*")
_ANS_RE = re.compile(r"\*\*answer:\s*(\d{1,3})\s*\*\*")


@dataclass
class AnswerDist:
    sampled: int            # the answer as sampled in this response
    pmf: dict[int, float]   # approximate P(answer = n), n in 0..100, sums to 1
    expected: float
    mode: int
    borrowed: float         # pmf mass built from borrowed (unobserved) conditionals

    def compact(self) -> dict:
        """Small JSON-friendly form for responses.jsonl (top-12 of the pmf)."""
        top = sorted(self.pmf.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        return {"sampled": self.sampled, "expected": round(self.expected, 2),
                "mode": self.mode, "borrowed": round(self.borrowed, 3),
                "top": {str(n): round(p, 4) for n, p in top}}


def _position_dist(entry: dict) -> dict[str, float]:
    """token -> probability at one output position (own token + top_logprobs)."""
    out: dict[str, float] = {}
    for alt in entry.get("top_logprobs") or []:
        out[alt["token"]] = max(out.get(alt["token"], 0.0), math.exp(alt["logprob"]))
    tok = entry.get("token")
    if tok is not None and tok not in out and entry.get("logprob") is not None:
        out[tok] = math.exp(entry["logprob"])
    return out


def _is_ascii_digit(t: str) -> bool:
    # NOT str.isdigit(): the tokenizer's top-20 contains Unicode digit variants
    # (fullwidth 9, subscript 1, ...) for which isdigit() is also true -- they
    # must not collide with (or crash on int()) the real ASCII digits
    return len(t) == 1 and "0" <= t <= "9"


def _digits(dist: dict[str, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for t, p in dist.items():
        if _is_ascii_digit(t):
            out[ord(t) - 48] = out.get(ord(t) - 48, 0.0) + p
    return out


def _end_mass(dist: dict[str, float]) -> float:
    return sum(p for t, p in dist.items() if not _is_ascii_digit(t))


def _build_pmf(q1_raw: dict[int, float], q2_raw: dict[str, float],
               q3_raw: dict[str, float] | None, sampled: int) -> AnswerDist | None:
    """Digit-tree pmf over 0..100 from the observed position distributions.

    q1_raw: P(token) at the first answer digit (digits only, unnormalized)
    q2_raw: P(token) at the position after the first digit (digits = the answer
            continues, anything else = the answer ended as one digit)
    q3_raw: P(token) after the second digit, if the sampled answer had >= 2
            digits (used only to resolve 10 vs 100 on the sampled path)
    """
    z1 = sum(q1_raw.values())
    if z1 <= 0:
        return None
    q1 = {d: p / z1 for d, p in q1_raw.items()}
    s_digits = str(sampled)
    s1 = int(s_digits[0])

    q2d, q2e = _digits(q2_raw), _end_mass(q2_raw)
    z2 = sum(q2d.values()) + q2e
    if z2 <= 0:
        return None
    q2d = {t: p / z2 for t, p in q2d.items()}
    q2e /= z2

    pmf: dict[int, float] = {}
    borrowed = 0.0
    for d, p1 in q1.items():
        exact = d == s1
        if not exact:
            borrowed += p1
        if d == 0:
            # a canonical 0 never continues with more digits
            pmf[0] = pmf.get(0, 0.0) + p1
            continue
        pmf[d] = pmf.get(d, 0.0) + p1 * q2e
        for t, p2 in q2d.items():
            n = 10 * d + t
            pmf[n] = pmf.get(n, 0.0) + p1 * p2
    # resolve 10 vs 100 exactly along the sampled "1,0,..." path
    if s1 == 1 and len(s_digits) >= 2 and s_digits[1] == "0" and q3_raw:
        q3d, q3e = _digits(q3_raw), _end_mass(q3_raw)
        z3 = q3d.get(0, 0.0) + q3e
        if z3 > 0 and 10 in pmf:
            base = q1.get(1, 0.0) * q2d.get(0, 0.0)  # exact mass on the "1,0" prefix
            pmf[10] -= base * (q3d.get(0, 0.0) / z3)
            pmf[100] = pmf.get(100, 0.0) + base * (q3d.get(0, 0.0) / z3)

    pmf = {n: p for n, p in pmf.items() if 0 <= n <= 100 and p > 0}
    z = sum(pmf.values())
    if z <= 0:
        return None
    pmf = {n: p / z for n, p in pmf.items()}
    expected = sum(n * p for n, p in pmf.items())
    mode = max(pmf.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return AnswerDist(sampled=sampled, pmf=pmf, expected=expected, mode=mode,
                      borrowed=min(1.0, borrowed / z))


def extract_answer_dists(text: str, logprobs_content: list, items: list) -> dict[str, "AnswerDist"]:
    """{item.key: AnswerDist} for every slider item whose answer digits could be
    located in the token stream. Mirrors the parser's last-valid-occurrence-wins
    rule per title. Items that cannot be extracted are simply absent."""
    tokens = [e.get("token") or "" for e in logprobs_content]
    joined = "".join(tokens)
    # the stream may carry tokens beyond the content (e.g. the <|im_end|> EOS);
    # offsets are computed from the front, so a trailing surplus is harmless
    if not joined.startswith(text):
        return {}
    # char offset -> token index
    starts = []
    pos = 0
    for t in tokens:
        starts.append(pos)
        pos += len(t)

    def token_at(char_i: int) -> int | None:
        lo, hi = 0, len(starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if starts[mid] <= char_i and (mid + 1 == len(starts) or starts[mid + 1] > char_i):
                return mid
            if starts[mid] > char_i:
                hi = mid - 1
            else:
                lo = mid + 1
        return None

    titles = [(m.start(), m.group(1).strip()) for m in _TITLE_RE.finditer(text)]
    chosen: dict[str, re.Match] = {}   # title -> last valid answer match after it
    for m in _ANS_RE.finditer(text):
        val = int(m.group(1))
        if not 0 <= val <= 100:
            continue
        preceding = [t for p, t in titles if p < m.start()]
        if preceding:
            chosen[preceding[-1]] = m

    key_by_title = {it.title: it.key for it in items}
    out: dict[str, AnswerDist] = {}
    for title, m in chosen.items():
        key = key_by_title.get(title)
        if key is None:
            continue
        d_start, d_end = m.span(1)
        i0 = token_at(d_start)
        if i0 is None:
            continue
        n_digits = d_end - d_start
        # require pure single-digit tokens for the answer (Qwen digit splitting)
        span = tokens[i0: i0 + n_digits]
        if "".join(span) != m.group(1) or any(not _is_ascii_digit(t) for t in span):
            continue
        if i0 + n_digits >= len(tokens):
            continue
        q1 = _digits(_position_dist(logprobs_content[i0]))
        q2 = _position_dist(logprobs_content[i0 + 1])
        q3 = _position_dist(logprobs_content[i0 + 2]) if (n_digits >= 2 and i0 + 2 < len(tokens)) else None
        dist = _build_pmf(q1, q2, q3, int(m.group(1)))
        if dist is not None:
            out[key] = dist
    return out
