"""
Pipeline position: Stage 1, step 05 -- runs after 01_build_inference_pool.py
(revised 2026-08-25 -- was 07_classify_comment_structure.py's --inference-pool-
output, which no longer exists; that script now runs AFTER sampling, scoped
to the sample, so it can't gate demographic inference, which must run BEFORE
sampling over the whole cohort) and optionally 04_infer_demographics_embed.py;
feeds 06_stratified_sample.py. See claude_project_plan/07_REPO_STRUCTURE.md §1.

LLM-based demographic inference from Reddit post/comment history.

Methodology: Staab et al. 2024's validated free-text attribute-inference
prompt structure (reasoning-before-guess, explicit in-prompt category
options, joint single-call extraction of all attributes) — see
`claude_project_plan/references/staab2024_summary.md` and the cloned
reference implementation at `claude_project_plan/references/llmprivacy/`
(esp. `src/reddit/reddit.py::create_prompts`, `src/reddit/reddit_utils.py`).
Wrapped in PoSSUM's prompt-design principles — see
`claude_project_plan/references/PoSSUM_Summary_for_Reddit_Replication.md`:
randomized feature order, self-reported speculation/confidence score, and
Waller & Anderson embedding scores fed in as background/independent-variable
context before asking the LLM to infer education/income/race. System-prompt
framing deliberately uses Staab's literal "expert investigator" persona, NOT
PoSSUM's neutral-annotator framing — this is demographic profiling from
text, the task Staab's framing was actually validated for; neutral-annotator
framing is reserved for Stage 4's survey-response elicitation, a different
task (answering as the person) with a different bias risk. See
`build_system_prompt` and `02_DECISIONS_LOG.md` 2026-08-18.

Per `02_DECISIONS_LOG.md` (2026-08-18): this is the ONLY place demographics
get inferred. Stage 2/4 must NOT re-run this — they consume its locked
output as given context.

Deliberate improvements over the 2023-era reference implementation:
- Structured JSON output (schema-constrained where the model client
  supports it) instead of free-text "Type:/Inference:/Guess:" parsing —
  the reference repo's own parser (`chat_parser.py`) needed heavy
  Levenshtein/embedding fuzzy-matching to recover that format reliably;
  modern JSON-mode/tool-calling avoids that failure class entirely.
- Both papers' guess-richness ideas combined rather than picking one: top-3
  ranked guesses (Staab) AND a single 0-100 confidence score for the top
  guess (PoSSUM) — the two are complementary, not alternatives (see
  `01_PROJECT_PLAN.md` §2, this was an open point, resolved here).
- `age_band` asks for the author's CURRENT age band, not "age when the
  comment was written" (Staab's literal instruction) — our submission
  needs a stable present-day trait per respondent, not a historical
  snapshot tied to old comment timestamps.

=============================================================================
EXPECTED INPUT 1 — post/comment history CSV (required)
=============================================================================
Exactly the schema `src/stage1_sampling/sql/alaska_pilot.sql` already
produces as `author_full_history.csv` — this script is designed to consume
that file directly, for this or any other subreddit pull, once real data
exists. Columns:

    post_type       "submission" | "comment"
    id              str
    author          str (case doesn't matter -- lowercased on load, Reddit
                     usernames are unique case-insensitively)
    author_created_utc  int (unix epoch) | empty
    subreddit       str
    created_utc     int (unix epoch seconds)
    title           str | empty (submissions only)
    body            str (comment text, or submission selftext)
    score           int | empty
    link_id         str | empty (comments only)
    parent_id       str | empty (comments only)

One row per submission/comment; multiple rows per author.

=============================================================================
EXPECTED INPUT 2 — embedding scores CSV (optional)
=============================================================================
Exactly the schema `04_infer_demographics_embed.py --output` produces:

    author              str
    ideology_score      float | empty
    gender_score        float | empty
    age_score           float | empty
    n_subreddits_total  int
    n_subreddits_scored int

If provided, an author's scores are rendered as background/independent-
variable context in the prompt (PoSSUM's independent-before-dependent
ordering trick) before asking the LLM to infer the other moderators.

=============================================================================
OUTPUT
=============================================================================
One row per author: `author`, the 6 moderators (`gender`, `age_band`,
`race`, `education`, `income`, `party`), `<moderator>_confidence` (0-100,
top guess only), `<moderator>_alternates` (2nd/3rd-ranked guesses, `;`-
joined, diagnostic only -- not part of the submission schema), `raw_response`
(full model output, archived per the K.2 raw-output requirement), `model_used`.

Usage:
    python 05_infer_demographics_llm.py --input author_full_history.csv \\
        --embedding-scores user_scores.csv --output llm_demographics.csv \\
        --model-client openai:gpt-4o-2024-05-13

    # Any other OpenAI-compatible endpoint (open-weight model served via
    # vLLM/Ollama/TGI/etc.) -- e.g. Qwen/Qwen3.8-27B behind vLLM on a GPU box.
    # --workers runs authors concurrently (ThreadPoolExecutor, same as
    # run_survey.py/the testbed scripts) -- default 1 (sequential) is far too
    # slow for a real run once the model is doing any real amount of work per
    # call; --extra-body disabling Qwen3 "thinking" mode is usually the other
    # big lever on latency (see OpenAICompatibleClient's docstring):
    python 05_infer_demographics_llm.py --input author_full_history.csv \\
        --output llm_demographics.csv --workers 32 \\
        --model-client compatible:Qwen/Qwen3.8-27B --base-url http://<gpu-host>:8000/v1 \\
        --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'

    python 05_infer_demographics_llm.py --selftest   # no data/model required
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> src/
from common.codebook import MODERATOR_CATEGORIES  # noqa: E402

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")  # model-agnostic approximation

    def count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))

except ImportError:  # pragma: no cover - fallback if tiktoken isn't installed
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)  # rough approximation


MAX_CONTEXT_TOKENS = 3000  # matches Staab et al.'s validated ~3000 GPT-4 token budget

# Order here is just the default; build_user_prompt randomizes it per call.
MODERATOR_LABELS = {
    "party": "political party affiliation",
    "gender": "gender",
    "age_band": "current age band",
    "race": "race / ethnicity",
    "education": "highest level of education completed",
    "income": "approximate yearly personal income",
}


# =============================================================================
# Input loading
# =============================================================================

def load_author_history(csv_path: Path, limit: Optional[int] = None) -> dict[str, list[dict]]:
    """author (lowercased) -> list of their post/comment records, as dicts.

    The input file does NOT need to be grouped/sorted by author -- every row
    is read regardless of order, so a given author's rows scattered anywhere
    across the file are still collected correctly.

    If `limit` is given, only the first `limit` distinct authors encountered
    (in file order) are loaded, via two passes: a cheap first pass that just
    finds which authors to target (stops as soon as it's seen that many --
    often long before the end of a large file), then a second pass that
    retains rows ONLY for those authors, discarding everything else
    immediately. Without this, a huge file (e.g. millions of rows) would get
    fully grouped into memory before the limit was ever applied -- fine for
    a real full run (which needs everyone anyway), but wasteful or outright
    OOM-risky for a quick smoke test on a handful of authors."""
    selected: Optional[set[str]] = None
    if limit is not None:
        selected = set()
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if len(selected) >= limit:
                    break
                selected.add(row["author"].strip().lower())

    by_author: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            author = row["author"].strip().lower()
            if selected is not None and author not in selected:
                continue
            by_author[author].append(row)
    return dict(by_author)


def load_embedding_scores(csv_path: Path) -> dict[str, dict]:
    """author (lowercased) -> {ideology_score, gender_score, age_score, ...}."""
    scores: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["author"].strip().lower()] = row
    return scores


# =============================================================================
# Mould construction (recency-biased selection within a token budget, then
# chronological formatting -- matches PoSSUM's "last m posts" recency bias
# and Staab's date-prefixed comment format / token-budget cap)
# =============================================================================

def record_text(record: dict) -> str:
    title = (record.get("title") or "").strip()
    body = (record.get("body") or "").strip()
    return f"{title}: {body}" if title else body


def parse_epoch(value) -> Optional[int]:
    """Parses a created_utc value that may be a plain int string ("123")
    or a float-formatted one ("123.0") -- the latter is a common artifact
    of exporting a pandas column that had any NaN values elsewhere in it
    (forces the whole column to float64, so even valid timestamps get
    written with a trailing ".0"). Returns None for anything unparseable,
    so those rows get filtered out rather than crashing downstream."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def select_within_budget(records: list[dict], max_tokens: int) -> list[dict]:
    """Most-recent-first selection up to max_tokens, then returned in
    chronological order for a coherent narrative read."""
    dated = [r for r in records if parse_epoch(r.get("created_utc")) is not None and record_text(r)]
    dated.sort(key=lambda r: parse_epoch(r["created_utc"]), reverse=True)

    selected: list[dict] = []
    used_tokens = 0
    for r in dated:
        t = count_tokens(record_text(r))
        if used_tokens + t > max_tokens and selected:
            break
        selected.append(r)
        used_tokens += t

    selected.sort(key=lambda r: parse_epoch(r["created_utc"]))
    return selected


def format_mould(records: list[dict]) -> str:
    lines = []
    for r in records:
        from datetime import datetime, timezone

        date_str = datetime.fromtimestamp(parse_epoch(r["created_utc"]), tz=timezone.utc).strftime("%Y-%m-%d")
        lines.append(f"{date_str} (r/{r['subreddit']}): {record_text(r)}")
    return "\n".join(lines)


def format_embedding_background(scores: Optional[dict]) -> str:
    """Waller & Anderson scores as background/independent-variable context,
    per PoSSUM's independent-before-dependent prompt-ordering principle.
    Returns "" if no scores are available for this author."""
    if not scores:
        return ""

    def fmt(key: str, label: str) -> Optional[str]:
        val = scores.get(key)
        if val in (None, ""):
            return None
        val = float(val)
        direction = "leans right/conservative" if val > 0 else "leans left/liberal"
        if key == "gender_score":
            direction = "leans women" if val > 0 else "leans men"
        if key == "age_score":
            direction = "skews older" if val > 0 else "skews younger"
        return f"{label}: {val:+.3f} ({direction}, based on subreddit activity)"

    parts = [
        fmt("ideology_score", "Community-embedding ideology signal"),
        fmt("gender_score", "Community-embedding gender signal"),
        fmt("age_score", "Community-embedding age signal"),
    ]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return (
        "Background (independent signal from which subreddits this author is active "
        "in, unrelated to the text below -- use as supporting context, not as ground "
        "truth):\n" + "\n".join(parts) + "\n"
    )


# =============================================================================
# Prompt construction
# =============================================================================

def build_system_prompt() -> str:
    """Staab et al.'s literal system prompt (their non-anonymized config;
    see reference repo `configs/reddit/running/reddit_gpt4_anonymized.yaml`
    vs. the plain default in `src/reddit/reddit.py::create_prompts`) --
    kept deliberately, not swapped for PoSSUM's neutral-annotator framing.
    Decision 2026-08-18 (see 02_DECISIONS_LOG.md): this is demographic
    PROFILING from text, the exact task this framing was validated for
    (85.5%/95.2% top-1/3 accuracy); PoSSUM's neutral-annotator framing is
    reserved for Stage 4's survey-response elicitation, a different task
    (answering AS the person) where persona-driven attitude bias is the
    actual risk being guarded against. Not a conflict between the two
    papers -- different framing for different tasks, each empirically
    motivated for its own task."""
    return (
        "You are an expert investigator with years of experience in online "
        "profiling and text analysis. You work with an analytical mindset and "
        "try to answer questions as precisely as possible."
    )


def build_response_schema(moderator_order: list[str]) -> dict:
    """JSON schema for a structured-output-capable model client. Each
    moderator: a top guess constrained to its category enum, up to 2
    alternates from the same enum, a 0-100 confidence score for the top
    guess, and free-text reasoning (written before the guess, not after --
    preserve this order in any prompt/parsing changes)."""
    properties = {}
    for mod in moderator_order:
        categories = MODERATOR_CATEGORIES[mod]
        properties[mod] = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "top_guess": {"type": "string", "enum": categories},
                "alternates": {
                    "type": "array",
                    "items": {"type": "string", "enum": categories},
                    "maxItems": 2,
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 100},
            },
            "required": ["reasoning", "top_guess", "alternates", "confidence"],
        }
    return {"type": "object", "properties": properties, "required": moderator_order}


def build_user_prompt(mould: str, background: str, moderator_order: list[str], rng: random.Random) -> str:
    order = list(moderator_order)
    rng.shuffle(order)  # PoSSUM: randomize feature order to cancel order effects

    options_block = []
    for mod in order:
        categories = MODERATOR_CATEGORIES[mod]
        options_block.append(
            f'- "{mod}" ({MODERATOR_LABELS[mod]}): choose exactly one of '
            + ", ".join(f'"{c}"' for c in categories)
        )

    header = (
        "The following are public comments and posts made on Reddit by one author. "
        "Your task is to infer the author's demographic attributes from this text.\n\n"
        f"{background}\n"
        "Comments and posts (oldest to newest):\n"
    )

    footer = (
        "\n\nFor each attribute below, first reason step-by-step over the evidence in "
        "the text (word choice, topics, subreddits, self-disclosures, implied context) "
        "before deciding -- do not state a guess before reasoning about it. Then give "
        "your top guess, up to 2 ranked alternate guesses, and a confidence score "
        "(0 = pure speculation with no textual support, 100 = the text states this "
        "directly) for your top guess. Respond with a single JSON object, one key per "
        "attribute below, matching this structure exactly:\n"
        '{"<attribute>": {"reasoning": "...", "top_guess": "...", '
        '"alternates": ["...", "..."], "confidence": 0-100}, ...}\n\n'
        "Attributes:\n" + "\n".join(options_block)
    )

    return header + mould + footer


# =============================================================================
# Model client abstraction — model choice is still an open decision
# (04_TASKS.md Stage 2), so this stays pluggable rather than hard-wired.
# =============================================================================

class ModelClient:
    """Interface every model client must implement."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class NotConfiguredClient(ModelClient):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError(
            "No model client configured. Pass --model-client (e.g. "
            "'openai:gpt-4o-2024-05-13') once a model choice is finalized, or use "
            "--selftest to validate the pipeline without calling a real model."
        )


class OpenAIClient(ModelClient):
    """Example concrete client. Lazy-imports `openai` so the rest of this
    script works without it installed -- add `openai` to requirements.txt
    once this (or another provider) is actually the chosen closed-source
    model (still open, see 04_TASKS.md Stage 2)."""

    def __init__(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Install the `openai` package to use OpenAIClient.") from e
        self.model_name = model_name
        self.client = OpenAI()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content


class OpenAICompatibleClient(ModelClient):
    """Any OpenAI-compatible chat-completions endpoint that ISN'T OpenAI
    itself -- this is how open-weight models (e.g. Qwen) get called here.
    Point `base_url` at whatever is actually serving the model: vLLM or
    text-generation-inference on a GPU box (`vllm serve <repo> --port 8000`
    -> base_url `http://<host>:8000/v1`), Ollama (`http://localhost:11434/v1`),
    Hugging Face's Inference Providers router, or any hosted provider. This
    class doesn't know or care which -- it's the same `openai` SDK, just
    pointed elsewhere, which is what "OpenAI-compatible" means in practice.

    Deliberately does NOT request JSON mode via `response_format` the way
    OpenAIClient does -- support for that varies across servers/versions and
    an unrecognized parameter can hard-error on some. Relies on the prompt's
    own JSON instruction (build_user_prompt) plus the reformat-retry already
    in parse_and_validate/infer_for_author, and on extract_json_object's
    tolerance for reasoning text around the JSON, instead.

    `extra_body`, if given, is passed straight through to the API call --
    e.g. for Qwen3-family "thinking" models, --extra-body
    '{"chat_template_kwargs": {"enable_thinking": false}}' asks the server
    to skip the chain-of-thought preamble (shorter/cheaper responses, and
    the reasoning we actually want is already requested per-field in our
    own prompt schema -- see build_user_prompt). Not hardcoded here since
    this class is meant to work with any OpenAI-compatible server, not just
    Qwen, and some vLLM versions reportedly ignore this flag anyway (a
    known gotcha, not something we control) -- extract_json_object handles
    the "thinking" text staying in the response regardless, either way."""

    def __init__(self, model_name: str, base_url: str, api_key: Optional[str] = None, extra_body: Optional[dict] = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Install the `openai` package -- it's also the client for non-OpenAI OpenAI-compatible endpoints.") from e
        self.model_name = model_name
        self.extra_body = extra_body
        # Most self-hosted servers don't check the key at all; the `openai`
        # SDK still requires some non-empty string to be passed.
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body=self.extra_body or {},
        )
        return resp.choices[0].message.content


def get_model_client(
    spec: Optional[str],
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    extra_body: Optional[dict] = None,
) -> ModelClient:
    if not spec:
        return NotConfiguredClient()
    provider, _, name = spec.partition(":")  # only splits on the FIRST ":" -- safe for names containing "/"
    if provider == "openai":
        return OpenAIClient(name)
    if provider == "compatible":
        if not base_url:
            raise ValueError("--base-url is required for --model-client compatible:<model_name>")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        return OpenAICompatibleClient(name, base_url, api_key, extra_body)
    raise ValueError(f"Unknown model client spec: {spec!r}")


# =============================================================================
# Response parsing + validation
# =============================================================================

def find_json_object_candidates(text: str) -> list[str]:
    """Finds every top-level balanced-brace substring in a raw model
    response that may have free-text reasoning before/around it -- e.g.
    Qwen3-family "thinking" models can emit a long chain-of-thought in
    plain prose (observed: ~13,000 chars before the real answer even
    starts) within the same `message.content` string, not wrapped in any
    <think> tag we could strip mechanically. Critically, that reasoning
    text can itself contain an earlier, smaller brace-delimited span
    (observed: the model discussing/illustrating the target JSON structure
    mid-reasoning, e.g. "But JSON structure expects array maybe") that
    parses as valid-but-incomplete JSON -- so returning just the FIRST
    balanced span isn't reliable. Returns every top-level span found, in
    the order they appear; parse_and_validate tries them last-first."""
    candidates: list[str] = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None
    return candidates


def parse_and_validate(raw_text: str, moderator_order: list[str]) -> Optional[dict]:
    """Returns the parsed dict if every moderator's top_guess is a valid
    category, else None (caller should retry once, then give up). Tries
    every top-level JSON-object-shaped span in the response, LAST first --
    the model's final/complete answer is virtually always the last such
    span when reasoning precedes it; an illustrative/partial JSON fragment
    mentioned mid-reasoning, if any, would appear earlier and won't
    validate against the full schema anyway (see find_json_object_candidates)."""
    for candidate in reversed(find_json_object_candidates(raw_text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if all(
            isinstance(data.get(mod), dict) and data[mod].get("top_guess") in MODERATOR_CATEGORIES[mod]
            for mod in moderator_order
        ):
            return data
    return None


REFUSAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bI\s+(?:cannot|can't|won't|will not)\b",
        r"\bI\s*(?:'m| am)\s+(?:not able|unable)\b",
        r"\bI must decline\b",
        r"\bI\s*(?:'m| am)\s+sorry,?\s+but\b",
        r"\bI\s+(?:don't|do not)\s+feel comfortable\b",
        r"\b(?:unable|not able) to\s+(?:speculate|infer|provide|determine|guess|"
        r"make (?:this|any|such) (?:assumption|inference))",
        r"\bnot appropriate (?:for me )?to\b",
        r"\bethical(?:ly)?\s+(?:concerns?|issues?|reasons?|guidelines)\b",
        r"\bviolat\w*\s+(?:my |the )?(?:guidelines|policy|policies)\b",
        r"\bas an AI\b",
        r"\brefuse to\b",
        r"\bagainst my\s+(?:guidelines|programming|principles)\b",
        r"\bnot something I (?:can|will|would)\b",
        r"\bcannot (?:make|draw)\s+(?:assumptions|inferences|conclusions)\s+about\s+"
        r"(?:real|actual)\s+(?:people|individuals|persons)\b",
        r"\bprivacy concerns?\b",
    ]
]


def detect_refusal(text: str) -> bool:
    """Heuristic keyword/pattern scan for the model declining the task on
    ethics/policy grounds, as distinct from just producing malformed JSON --
    used to split "malformed_json" from "refusal" in the `status` column for
    a refusal-rate pilot (see 02_DECISIONS_LOG.md, 2026-08-20). Not a
    certified classifier: false negatives are possible for creatively-phrased
    refusals not covered by these patterns."""
    return any(p.search(text) for p in REFUSAL_PATTERNS)


REFORMAT_INSTRUCTION = (
    "Your previous response did not match the required JSON structure. Respond with "
    "ONLY a single valid JSON object, one key per attribute, exactly matching the "
    'structure {"<attribute>": {"reasoning": "...", "top_guess": "...", '
    '"alternates": ["...", "..."], "confidence": 0-100}, ...} -- no other text.'
)


def infer_for_author(
    author: str,
    records: list[dict],
    embedding_scores: Optional[dict],
    client: ModelClient,
    model_name: str,
    max_tokens: int,
    rng: random.Random,
) -> dict:
    moderator_order = list(MODERATOR_CATEGORIES.keys())
    mould = format_mould(select_within_budget(records, max_tokens))
    background = format_embedding_background(embedding_scores)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(mould, background, moderator_order, rng)

    raw = client.complete(system_prompt, user_prompt)
    parsed = parse_and_validate(raw, moderator_order)

    if parsed is None:
        raw_retry = client.complete(system_prompt, user_prompt + "\n\n" + REFORMAT_INSTRUCTION)
        parsed = parse_and_validate(raw_retry, moderator_order)
        raw = raw + "\n---RETRY---\n" + raw_retry

    if parsed is not None:
        status = "success"
    elif detect_refusal(raw):
        status = "refusal"
    else:
        status = "malformed_json"

    row = {"author": author, "model_used": model_name, "status": status, "raw_response": raw}
    for mod in moderator_order:
        entry = (parsed or {}).get(mod, {})
        row[mod] = entry.get("top_guess", "")
        row[f"{mod}_confidence"] = entry.get("confidence", "")
        row[f"{mod}_alternates"] = ";".join(entry.get("alternates", []))
    return row


# =============================================================================
# Orchestration
# =============================================================================

def format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. '3h54m', '2m18s', '42s'."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m}m"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


def already_done_authors(output_csv: Path) -> set[str]:
    """Authors with a row already written in an existing output file, for
    --resume. Reads only the `author` column, not the whole file into
    memory as a dataframe -- fine even for a 10,000+ row file."""
    done = set()
    with open(output_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["author"])
    return done


def run_pipeline(
    input_csv: Path,
    output_csv: Path,
    embedding_scores_csv: Optional[Path],
    model_client: ModelClient,
    model_name: str,
    max_tokens: int,
    seed: Optional[int],
    limit: Optional[int],
    resume: bool,
    workers: int = 1,
) -> None:
    # Per-author rng is derived from (run_seed, author) rather than sharing one
    # random.Random across calls -- required once workers > 1 (a single shared
    # Random instance isn't thread-safe under concurrent .shuffle() calls), and
    # as a side benefit stays fully reproducible for a given --seed regardless
    # of how many workers ran or what order calls happened to complete in.
    run_seed = seed if seed is not None else random.SystemRandom().randrange(2**32)

    print(f"Loading {input_csv} ...", flush=True)
    load_start = time.time()
    history = load_author_history(input_csv, limit=limit)
    print(f"Loaded {len(history):,} authors in {format_duration(time.time() - load_start)}.", flush=True)

    if embedding_scores_csv:
        print(f"Loading {embedding_scores_csv} ...", flush=True)
        scores = load_embedding_scores(embedding_scores_csv)
        print(f"Loaded embedding scores for {len(scores):,} authors.", flush=True)
    else:
        scores = {}

    all_authors = list(history.keys())  # already limited by load_author_history if `limit` was given

    fieldnames = ["author", "model_used", "status"] + [
        f
        for mod in MODERATOR_CATEGORIES
        for f in (mod, f"{mod}_confidence", f"{mod}_alternates")
    ] + ["raw_response"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists() and not resume:
        raise FileExistsError(
            f"{output_csv} already exists. Pass --resume to continue from where a previous "
            "run left off (skips authors already in the file), or delete/rename it to start "
            "over -- refusing to silently overwrite completed work."
        )

    skip = already_done_authors(output_csv) if (resume and output_csv.exists()) else set()
    todo = [a for a in all_authors if a not in skip]
    if skip:
        print(f"Resuming: {len(skip)} authors already done in {output_csv}, {len(todo)} remaining of {len(all_authors)} total.")

    def process(author: str) -> tuple[str, dict, float]:
        """Runs one author's model call(s); never raises -- a client/network
        error becomes an 'error' status row instead of aborting the whole run
        (matters more once workers > 1 multiplies exposure to a transient
        failure on any one of many in-flight requests)."""
        author_rng = random.Random(f"{run_seed}:{author}")
        call_start = time.time()
        try:
            row = infer_for_author(
                author, history[author], scores.get(author), model_client, model_name, max_tokens, author_rng
            )
        except Exception as e:
            row = {"author": author, "model_used": model_name, "status": "error",
                   "raw_response": f"{type(e).__name__}: {e}"}
            for mod in MODERATOR_CATEGORIES:
                row[mod] = ""
                row[f"{mod}_confidence"] = ""
                row[f"{mod}_alternates"] = ""
        return author, row, time.time() - call_start

    file_mode = "a" if (resume and output_csv.exists()) else "w"
    run_start = time.time()
    write_lock = threading.Lock()
    done_this_run = 0

    with open(output_csv, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()

        def record(author: str, row: dict, call_elapsed: float) -> None:
            nonlocal done_this_run
            with write_lock:
                writer.writerow(row)
                f.flush()
                done_this_run += 1
                total_done = len(skip) + done_this_run
                elapsed = time.time() - run_start
                avg = elapsed / done_this_run
                remaining = len(todo) - done_this_run
                eta = avg * remaining
                pct = total_done / len(all_authors) * 100
                print(
                    f"[{total_done}/{len(all_authors)} = {pct:.1f}%] {author} -- {call_elapsed:.1f}s "
                    f"(avg {avg:.1f}s/user) | elapsed {format_duration(elapsed)} | "
                    f"ETA {format_duration(eta)} remaining (~{format_duration(elapsed + eta)} total this run)"
                )

        if workers <= 1:
            for author in todo:
                record(*process(author))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(process, author): author for author in todo}
                for fut in as_completed(futs):
                    record(*fut.result())

    print(f"Saved: {output_csv}")

    status_counts = defaultdict(int)
    with open(output_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status_counts[row.get("status", "")] += 1
    total = sum(status_counts.values())
    if total:
        print(f"\nStatus breakdown ({total} rows total in {output_csv}):")
        for status in ("success", "refusal", "malformed_json"):
            n = status_counts.get(status, 0)
            print(f"  {status:15s} {n:6d}  ({n / total:.1%})")
        other = total - sum(status_counts.get(s, 0) for s in ("success", "refusal", "malformed_json"))
        if other:
            print(f"  {'(other/blank)':15s} {other:6d}  ({other / total:.1%})")


# =============================================================================
# Self-test: validates prompt construction + parsing end to end with a tiny
# built-in synthetic profile -- no real data or model call required. This is
# the thing to run right now, before real Reddit history is available.
# =============================================================================

def selftest() -> None:
    rng = random.Random(0)
    fake_records = [
        {"post_type": "comment", "id": "c1", "author": "TestUser", "author_created_utc": "1600000000",
         "subreddit": "personalfinance", "created_utc": "1700000000", "title": "",
         "body": "Just maxed out my 401k contribution for the year, feels good.", "score": "5",
         "link_id": "", "parent_id": ""},
        {"post_type": "submission", "id": "s1", "author": "TestUser", "author_created_utc": "1600000000",
         "subreddit": "AskWomen", "created_utc": "1710000000",
         "title": "Anyone else juggling a PhD and a toddler?", "body": "It's exhausting some days.",
         "score": "20", "link_id": "", "parent_id": ""},
    ]
    fake_scores = {"ideology_score": "-0.04", "gender_score": "0.06", "age_score": "0.02"}

    moderator_order = list(MODERATOR_CATEGORIES.keys())
    mould = format_mould(select_within_budget(fake_records, MAX_CONTEXT_TOKENS))
    background = format_embedding_background(fake_scores)
    prompt = build_user_prompt(mould, background, moderator_order, rng)

    print("=" * 60 + "\nSYSTEM PROMPT\n" + "=" * 60)
    print(build_system_prompt())
    print("=" * 60 + "\nUSER PROMPT\n" + "=" * 60)
    print(prompt)

    fake_response = json.dumps(
        {
            mod: {
                "reasoning": "test reasoning",
                "top_guess": MODERATOR_CATEGORIES[mod][0],
                "alternates": MODERATOR_CATEGORIES[mod][1:3],
                "confidence": 70,
            }
            for mod in moderator_order
        }
    )
    parsed = parse_and_validate(fake_response, moderator_order)
    assert parsed is not None, "Self-test parser round-trip failed"
    print("=" * 60 + "\nPARSED (validated against a hand-built fake response)\n" + "=" * 60)
    print(json.dumps(parsed, indent=2))

    # -- refusal vs. malformed-JSON classification (2026-08-20) --------------
    refusal_examples = [
        "I'm sorry, but I cannot make assumptions about real individuals' demographics.",
        "I can't speculate about this person's race or income due to ethical concerns.",
        "As an AI, I am unable to provide demographic guesses about real Reddit users.",
        "I must decline this request -- it violates my guidelines around privacy.",
    ]
    for text in refusal_examples:
        assert detect_refusal(text), f"expected refusal detected: {text!r}"

    non_refusal_examples = [
        json.dumps({mod: {"reasoning": "test", "top_guess": MODERATOR_CATEGORIES[mod][0],
                           "alternates": [], "confidence": 50} for mod in moderator_order}),
        "Let's think through this step by step. The author mentions tax policy...",
        "{ this is not valid json at all, just garbled text from the model }",
    ]
    for text in non_refusal_examples:
        assert not detect_refusal(text), f"expected no refusal detected: {text!r}"

    class _FixedResponseClient(ModelClient):
        def __init__(self, response: str):
            self.response = response

        def complete(self, system_prompt, user_prompt):
            return self.response

    success_row = infer_for_author(
        "TestUser", fake_records, fake_scores, _FixedResponseClient(fake_response),
        "selftest-model", MAX_CONTEXT_TOKENS, rng,
    )
    assert success_row["status"] == "success", success_row["status"]

    refusal_row = infer_for_author(
        "TestUser", fake_records, fake_scores,
        _FixedResponseClient("I'm sorry, but I cannot speculate about a real person's demographics."),
        "selftest-model", MAX_CONTEXT_TOKENS, rng,
    )
    assert refusal_row["status"] == "refusal", refusal_row["status"]

    malformed_row = infer_for_author(
        "TestUser", fake_records, fake_scores,
        _FixedResponseClient("this is just garbled text, no JSON object anywhere in sight"),
        "selftest-model", MAX_CONTEXT_TOKENS, rng,
    )
    assert malformed_row["status"] == "malformed_json", malformed_row["status"]

    print("\nSelf-test passed: prompt construction, mould/background rendering, response "
          "parsing, and refusal-vs-malformed-JSON status classification all work end to "
          "end without real data or a model call.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="Post/comment history CSV (see module docstring for schema)")
    parser.add_argument("--embedding-scores", type=Path, default=None, help="Optional 04_infer_demographics_embed.py output CSV")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    parser.add_argument("--model-client", type=str, default=None, help="'openai:<model>' or 'compatible:<model>' (any other OpenAI-compatible endpoint, e.g. vLLM/Ollama serving an open-weight model) -- omit to use --selftest only")
    parser.add_argument("--base-url", type=str, default=None, help="Required for --model-client compatible:... -- e.g. http://<gpu-host>:8000/v1 for a local vLLM, http://localhost:11434/v1 for Ollama, or http://<node-ip>:<nodePort>/v1 for a Kubernetes NodePort Service (no kubectl/port-forward needed once that's set up)")
    parser.add_argument("--api-key-env", type=str, default=None, help="Name of an env var holding the API key for --model-client compatible:...; omit if the endpoint doesn't check auth")
    parser.add_argument("--extra-body", type=str, default=None, help="Optional JSON string passed through to the API call, for server/model-specific options -- e.g. for Qwen3-family 'thinking' models: '{\"chat_template_kwargs\": {\"enable_thinking\": false}}'. Not required for correctness (extract_json_object handles reasoning-prefixed responses either way) -- this is purely to shorten/speed up responses if the server honors it.")
    parser.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS)
    parser.add_argument("--seed", type=int, default=None, help="Seed for feature-order randomization (omit for true per-run randomness)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N authors (for a quick real-data smoke test)")
    parser.add_argument("--resume", action="store_true", help="If --output already exists, skip authors already in it and append the rest, instead of refusing to overwrite. Use after a crash/interruption to continue a partial run.")
    parser.add_argument("--workers", type=int, default=1, help="Number of authors to run concurrently against the model endpoint (ThreadPoolExecutor, same pattern as run_survey.py/run_heldout_task.py). Default 1 = sequential, matching prior behavior. vLLM handles concurrent requests fine -- e.g. the testbed used --workers 32 against the same Qwen k8s deployment.")
    parser.add_argument("--selftest", action="store_true", help="Validate the pipeline with a built-in synthetic profile; no data or model needed")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --selftest is given")

    extra_body = json.loads(args.extra_body) if args.extra_body else None
    client = get_model_client(args.model_client, args.base_url, args.api_key_env, extra_body)
    model_name = args.model_client or "unconfigured"
    run_pipeline(
        args.input, args.output, args.embedding_scores, client, model_name,
        args.max_context_tokens, args.seed, args.limit, args.resume, args.workers,
    )


if __name__ == "__main__":
    main()
