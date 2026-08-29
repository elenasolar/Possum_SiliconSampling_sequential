"""
Pipeline position: Stage 2, step 00 -- runs after stage1_sampling/
07_classify_comment_structure.py, which (revised 2026-08-25) now itself runs
AFTER stage1_sampling/06_stratified_sample.py, scoped to just the sampled
agents via its own --authors flag -- so this script's input is already
restricted to the ~1,000 sampled agents by the time it gets here, not the
whole ~10,000-candidate cohort. Feeds 01_build_persona_posts.py, which joins
this output against the sampled agents.csv again (a cheap, defensive
re-filter, not load-bearing now that scoping already happened upstream)
before it reaches 02_build_personas.py. See claude_project_plan/
07_REPO_STRUCTURE.md §1 for the full execution order.

Filter 07_classify_comment_structure.py's `reply_depth2` rows (a comment-reply-
to-a-first-level-comment, with parent text already resolved) down to
"meaningful" interactions for persona construction, per the 2026-08-24
decision with David (see 02_DECISIONS_LOG.md): comment-reply interactions
carry more behavioral/reactive signal than a submission or a context-free
comment, but only the ones that show actual engagement are worth including.

An interaction qualifies if its own text has at least MIN_WORDS words AND at
least one of:
    - a question (a question word, or a literal "?")
    - second-person address ("you", "you're", "your", ...)
    - a disagreement marker (but, however, no, disagree, wrong, ...)
    - a quotation of the parent comment (Reddit's ">" markdown quote syntax,
      or a run of the parent's own words reappearing in the reply)
    - a hedge (obviously, clearly, honestly, ...)

PROVISIONAL, not yet team-signed-off (per 02_DECISIONS_LOG.md, 2026-08-24's
open follow-up): the exact word lists below and the "quotation" definition
are a first concrete operationalization of David's spec, not a finalized
one -- adjust CUE_PATTERNS / MIN_QUOTE_WORDS if the team specifies otherwise.

Input: 07_classify_comment_structure.py's --output (needs `structure_kind`,
`body`, `parent_body`, `parent_author`, `parent_id` columns).

Output: one row per kept interaction -- author, id, subreddit, created_utc,
body, word_count, parent_author, parent_body, matched_cues (";"-joined,
diagnostic) -- 01_build_persona_posts.py filters this down to the sampled
agents and formats it as 02_build_personas.py's posts.jsonl input.

=============================================================================
--pending-fetch-ids-output -- narrowing what Peer needs to fetch
=============================================================================
07_classify_comment_structure.py's `unresolved` rows include every reply whose
parent lives outside our sample -- some of these would turn out to be
depth-2 (in scope) and some depth-3+ (never in scope), and we can't tell
which without the parent. Handing Peer *every* such parent id to fetch is
wasteful: an unresolved reply whose own text has zero qualifying cue (too
short, no question/second-person/disagreement/hedge) will never pass the
meaningful-interaction filter regardless of what its parent turns out to be
or how deep it is -- no reason to fetch that parent at all.

`--pending-fetch-ids-output`/`--pending-fetch-output` apply the SAME cue
check to `unresolved` rows' own text (quotation is skipped -- it needs the
still-unknown parent text) and only list the parent ids of replies that
already show a qualifying cue from their own text alone. This is a real,
accepted narrowing, not a bug: a reply that would ONLY have qualified via
quoting its parent is excluded from this round's fetch list -- fine for an
initial, not-yet-comprehensive pass (per Elena, 2026-08-24); revisit if a
fuller sweep is ever wanted. `--pending-fetch-ids-output` is deduplicated,
single "id" column -- feed it straight into sql/fetch_parent_comments.sql's
`\\copy ... FROM` step unchanged, same as
07_classify_comment_structure.py's own --missing-parent-ids-output.

Usage:
    python 00_extract_meaningful_interactions.py \\
        --input author_full_history_classified.csv \\
        --output meaningful_interactions.csv

    python 00_extract_meaningful_interactions.py --selftest   # no data required
"""

import argparse
import re
from pathlib import Path

import pandas as pd

MIN_WORDS = 10
MIN_QUOTE_WORDS = 5  # consecutive words from the parent that, if echoed in the reply, count as a quotation

# PROVISIONAL word lists -- see module docstring.
CUE_PATTERNS = {
# --- existing ---
    "question": re.compile(r"\b(who|what|when|where|why|how|which)\b|\?", re.IGNORECASE),
    "second_person": re.compile(r"\b(you|you're|youre|your|ur)\b", re.IGNORECASE),
    "disagreement": re.compile(r"\b(but|however|no|nah|disagree|wrong|incorrect|actually)\b", re.IGNORECASE),
    "booster": re.compile(r"\b(obviously|clearly|honestly|frankly|surely|of course)\b", re.IGNORECASE),  # renamed from "hedge"

    # --- agreement / alignment ---
    "agreement": re.compile(r"\b(agree|agreed|exactly|this[\s.,!]|totally|absolutely|100%|yes|yeah|yep|precisely|well said|couldn't agree more|nailed it|spot on)\b", re.IGNORECASE),
    "concession": re.compile(r"\b(fair point|fair enough|good point|you('re| are) right|that's true|i stand corrected|touch[ée]|didn't (know|realize|consider) that|changed my mind|TIL)\b", re.IGNORECASE),

    # --- true epistemic hedging / uncertainty ---
    "hedge": re.compile(r"\b(i think|i believe|maybe|perhaps|possibly|not sure|i guess|kind of|sort of|might be|could be|as far as i know|afaik|i'm not (certain|100%|entirely sure))\b", re.IGNORECASE),

    # --- elaboration / adding information ---
    "elaboration": re.compile(r"\b(also|additionally|furthermore|moreover|to add|worth (noting|mentioning)|another (thing|point|reason)|on top of that|building on)\b", re.IGNORECASE),
    "evidence_marker": re.compile(r"\b(source|according to|study|studies|research shows|data shows|citation|link|read that|article|paper)\b|https?://", re.IGNORECASE),

    # --- clarification-seeking (subtype of question, but different function) ---
    "clarification_request": re.compile(r"\b(what do you mean|can you (clarify|explain|elaborate)|not sure (i|what) (understand|you mean)|could you expand)\b", re.IGNORECASE),

    # --- personal grounding (often marks substantive, experience-based contribution) ---
    "personal_experience": re.compile(r"\b(i've|i have|in my experience|i work(ed)? (in|as)|as a\b|when i|i once|i used to)\b", re.IGNORECASE),

    # --- quote-block rebuttal/reference (structural, but doable as regex on markdown) ---
    "quotes_parent": re.compile(r"^>", re.MULTILINE),

    
}

OUTPUT_COLUMNS = [
    "author", "id", "subreddit", "created_utc", "body",
    "word_count", "parent_author", "parent_body", "matched_cues",
]


def word_count(text: str) -> int:
    return len(text.split())


def quotes_parent(reply_body: str, parent_body: str, min_words: int = MIN_QUOTE_WORDS) -> bool:
    """True if the reply uses Reddit's literal ">" markdown quote syntax, or
    echoes a run of >= min_words consecutive words from the parent verbatim
    (case-insensitive, whitespace-normalized)."""
    if any(line.strip().startswith(">") for line in reply_body.splitlines()):
        return True
    if not parent_body:
        return False
    parent_words = re.findall(r"\w+", parent_body.lower())
    if len(parent_words) < min_words:
        return False
    reply_norm = " ".join(re.findall(r"\w+", reply_body.lower()))
    for i in range(len(parent_words) - min_words + 1):
        chunk = " ".join(parent_words[i:i + min_words])
        if chunk in reply_norm:
            return True
    return False


def evaluate_interaction(body: str, parent_body: str) -> tuple[bool, list[str]]:
    """Returns (qualifies, matched_cue_names)."""
    if word_count(body) < MIN_WORDS:
        return False, []
    matched = [name for name, pattern in CUE_PATTERNS.items() if pattern.search(body)]
    if quotes_parent(body, parent_body):
        matched.append("quotation")
    return (len(matched) > 0), matched


def extract(classified: pd.DataFrame, relaxed_authors: dict | None = None) -> pd.DataFrame:
    """relaxed_authors, if given, maps a lowercased author name -> the
    word-count floor to use for just that author's reply_depth2 rows (the
    cue requirement is dropped entirely for these authors, same as before).
    Per Elena, 2026-08-27 -- for agents who ended up with too little
    persona-building material under the full "meaningful interaction"
    definition, fill them up with more of their own words even where none of
    the linguistic cues apply, rather than leaving them thin. Per-author
    floors (2026-08-28 follow-up): most relaxed authors keep the normal
    relaxed_min_words floor (typically 10), but a couple whose reply_depth2
    rows are ALL still under even that floor get 0 instead -- keep every
    reply_depth2 row they have, no word-count floor at all. Every other
    (non-relaxed) author is completely unaffected -- same rule as always.
    Relaxed-qualifying rows are tagged "relaxed_min_words:<N>" in
    matched_cues so it's visible afterward which rows passed which way and
    at what floor, rather than looking identical to a normally-qualifying
    interaction."""
    candidates = classified[classified["structure_kind"] == "reply_depth2"]
    rows = []
    for _, row in candidates.iterrows():
        body = row.get("body") or ""
        parent_body = row.get("parent_body") or ""
        author_key = str(row.get("author") or "").strip().lower()
        if relaxed_authors and author_key in relaxed_authors:
            min_words = relaxed_authors[author_key]
            ok = word_count(body) >= min_words
            cues = [f"relaxed_min_words:{min_words}"] if ok else []
        else:
            ok, cues = evaluate_interaction(body, parent_body)
        if not ok:
            continue
        rows.append({
            "author": row["author"],
            "id": row["id"],
            "subreddit": row.get("subreddit", ""),
            "created_utc": row.get("created_utc", ""),
            "body": body,
            "word_count": word_count(body),
            "parent_author": row.get("parent_author", ""),
            "parent_body": parent_body,
            "matched_cues": ";".join(cues),
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


PENDING_FETCH_COLUMNS = ["reply_id", "reply_author", "parent_bare_id", "matched_cues_own_text_only"]


def strip_prefix(fullname) -> str:
    """"t1_abc123" -> "abc123". Local copy of 07_classify_comment_structure.py's
    helper -- small enough not to warrant a cross-stage import."""
    fullname = (fullname or "").strip()
    return fullname[3:] if fullname.startswith("t1_") else fullname


def find_pending_fetch_candidates(classified: pd.DataFrame) -> pd.DataFrame:
    """`unresolved` rows that are genuinely a pending reply (parent_id starts
    with "t1_") whose OWN text already shows a qualifying cue -- see module
    docstring for why this deliberately excludes quotation-only qualifiers."""
    pending = classified[
        (classified["structure_kind"] == "unresolved")
        & (classified["parent_id"].fillna("").str.startswith("t1_"))
    ]
    rows = []
    for _, row in pending.iterrows():
        body = row.get("body") or ""
        ok, cues = evaluate_interaction(body, "")  # no parent text yet -> quotation unreachable
        if not ok:
            continue
        rows.append({
            "reply_id": row["id"],
            "reply_author": row["author"],
            "parent_bare_id": strip_prefix(row["parent_id"]),
            "matched_cues_own_text_only": ";".join(cues),
        })
    return pd.DataFrame(rows, columns=PENDING_FETCH_COLUMNS)


def selftest() -> None:
    classified = pd.DataFrame([
        # too short (4 words) -- dropped even though it has a disagreement marker
        {"structure_kind": "reply_depth2", "author": "a", "id": "r1", "subreddit": "test",
         "created_utc": "1", "body": "No, that is wrong.", "parent_author": "p", "parent_body": "Some claim."},
        # >= MIN_WORDS, but no cue at all -- dropped
        {"structure_kind": "reply_depth2", "author": "a", "id": "r2", "subreddit": "test",
         "created_utc": "2", "body": "I went to the store yesterday afternoon and bought some milk.",
         "parent_author": "p", "parent_body": "Some claim."},
        # second-person + disagreement -- kept
        {"structure_kind": "reply_depth2", "author": "b", "id": "r3", "subreddit": "test",
         "created_utc": "3", "body": "But you're wrong about that, and everyone else can see it too.",
         "parent_author": "p", "parent_body": "Some claim."},
        # booster only (renamed from "hedge" -- see CUE_PATTERNS) -- kept
        {"structure_kind": "reply_depth2", "author": "b", "id": "r4", "subreddit": "test",
         "created_utc": "4", "body": "Obviously that is not correct given the particular situation right now.",
         "parent_author": "p", "parent_body": "Some claim."},
        # question mark, no second-person -- kept
        {"structure_kind": "reply_depth2", "author": "c", "id": "r5", "subreddit": "test",
         "created_utc": "5", "body": "When did that event happen, and why did nobody report it immediately?",
         "parent_author": "p", "parent_body": "Some claim."},
        # markdown quote of parent -- kept
        {"structure_kind": "reply_depth2", "author": "c", "id": "r6", "subreddit": "test",
         "created_utc": "6", "body": "> the earth is flat and always has been\nthat claim is simply not true at all",
         "parent_author": "p", "parent_body": "the earth is flat and always has been"},
        # verbatim word-run echo of parent (no ">", no other cue) -- kept via quotation
        {"structure_kind": "reply_depth2", "author": "c", "id": "r7", "subreddit": "test",
         "created_utc": "7", "body": "the earth is flat and always has been said again here",
         "parent_author": "p", "parent_body": "the earth is flat and always has been"},
        # not a reply_depth2 at all -- excluded regardless of content
        {"structure_kind": "reply_deeper", "author": "a", "id": "r8", "subreddit": "test",
         "created_utc": "8", "body": "But you're wrong about that too.", "parent_author": "p", "parent_body": ""},
        # unresolved (parent outside our sample), but own text already has a
        # disagreement marker -- worth fetching the parent for
        {"structure_kind": "unresolved", "author": "d", "id": "r9", "subreddit": "test",
         "created_utc": "9", "body": "No, however you look at it, the outcome remains the same for everyone.",
         "parent_author": "", "parent_body": "", "parent_id": "t1_ext1"},
        # unresolved, own text has zero cue -- NOT worth fetching, even though
        # it might theoretically have qualified via quoting its (unknown) parent
        {"structure_kind": "unresolved", "author": "d", "id": "r10", "subreddit": "test",
         "created_utc": "10", "body": "I went there again yesterday afternoon and stayed for a couple hours.",
         "parent_author": "", "parent_body": "", "parent_id": "t1_ext2"},
        # unresolved but NOT a pending reply at all (blank parent_id) -- excluded
        # from the fetch list regardless of its own text
        {"structure_kind": "unresolved", "author": "d", "id": "r11", "subreddit": "test",
         "created_utc": "11", "body": "But this one has no parent at all.", "parent_author": "", "parent_body": "",
         "parent_id": ""},
        # reply_depth2, >=10 words, but NO cue at all -- dropped under the normal
        # rule; only kept for "e" once "e" is passed as a relaxed author at min_words=10
        {"structure_kind": "reply_depth2", "author": "e", "id": "r12", "subreddit": "test",
         "created_utc": "12", "body": "The weather today was mild and calm across most of the region.",
         "parent_author": "p", "parent_body": "Unrelated parent text."},
        # reply_depth2, only 2 words, no cue -- dropped even at relaxed min_words=10;
        # only kept for "f" once "f" is passed as a relaxed author at min_words=0
        {"structure_kind": "reply_depth2", "author": "f", "id": "r13", "subreddit": "test",
         "created_utc": "13", "body": "Sure thing.", "parent_author": "p", "parent_body": "Unrelated parent text."},
    ])

    kept = extract(classified)
    kept_ids = set(kept["id"])
    assert kept_ids == {"r3", "r4", "r5", "r6", "r7"}, kept_ids  # r12/r13 NOT kept -- no relaxed_authors given

    by_id = kept.set_index("id")
    assert set(by_id.loc["r3", "matched_cues"].split(";")) == {"second_person", "disagreement"}
    assert by_id.loc["r4", "matched_cues"] == "booster"
    assert by_id.loc["r5", "matched_cues"] == "question"
    assert "quotation" in by_id.loc["r6", "matched_cues"].split(";")
    assert by_id.loc["r7", "matched_cues"] == "quotation"

    pending = find_pending_fetch_candidates(classified)
    assert set(pending["reply_id"]) == {"r9"}, set(pending["reply_id"])
    assert pending.set_index("reply_id").loc["r9", "parent_bare_id"] == "ext1"

    # --relaxed-authors: per-author floor -- "e" at 10 words, "f" at 0 (take
    # everything); everyone else (including "f" at any floor above 2 words) unaffected
    relaxed_kept = extract(classified, relaxed_authors={"e": 10, "f": 0})
    relaxed_ids = set(relaxed_kept["id"])
    assert relaxed_ids == {"r3", "r4", "r5", "r6", "r7", "r12", "r13"}, relaxed_ids  # r12 and r13 now included
    by_id_relaxed = relaxed_kept.set_index("id")
    assert by_id_relaxed.loc["r12", "matched_cues"] == "relaxed_min_words:10"
    assert by_id_relaxed.loc["r13", "matched_cues"] == "relaxed_min_words:0"
    # b's r3 wasn't relaxed-eligible (only "e"/"f" were) -- must still show its real cues, unchanged
    assert set(by_id_relaxed.loc["r3", "matched_cues"].split(";")) == {"second_person", "disagreement"}
    # a floor of 10 for "e" alone (no "f" entry) leaves r13 out, confirming per-author floors are independent
    relaxed_kept_e_only = extract(classified, relaxed_authors={"e": 10})
    assert "r13" not in set(relaxed_kept_e_only["id"])

    print("Self-test passed: word-count floor, each individual cue (question, "
          "second-person, disagreement, hedge), both quotation mechanisms "
          "(markdown '>' and verbatim word-run echo), the reply_depth2-only "
          "scope restriction, the pending-fetch narrowing (own-text cue "
          "required, blank parent_id excluded), and --relaxed-authors with "
          "independent per-author word-count floors all work end "
          "to end without "
          "real data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="07_classify_comment_structure.py's --output")
    ap.add_argument("--output", type=Path, default=None, help="Output CSV of kept meaningful interactions (from resolved reply_depth2 rows)")
    ap.add_argument("--pending-fetch-output", type=Path, default=None,
                     help="Diagnostic CSV: unresolved replies whose own text already qualifies -- "
                          "reply_id, reply_author, parent_bare_id, matched_cues_own_text_only")
    ap.add_argument("--pending-fetch-ids-output", type=Path, default=None,
                     help="Deduplicated single-column ('id') CSV of just the parent ids from the above -- "
                          "feed straight into sql/fetch_parent_comments.sql's \\copy ... FROM step")
    ap.add_argument("--relaxed-authors", type=Path, default=None,
                     help="Optional CSV with an 'author' column (e.g. diagnose_interaction_coverage.py's "
                          "--below-threshold-output) -- for just these authors, drop the cue requirement "
                          "entirely and keep any reply_depth2 row with >= a word-count floor. The floor is "
                          "--relaxed-min-words by default, or per-author if the CSV has its own 'min_words' "
                          "column (e.g. 0 for an author whose reply_depth2 rows are all still under the "
                          "default floor -- keeps every reply_depth2 row that author has). Every other "
                          "author is unaffected.")
    ap.add_argument("--relaxed-min-words", type=int, default=10,
                     help="Default word-count floor for --relaxed-authors, used for any row lacking its own "
                          "'min_words' column value (default 10)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.input:
        ap.error("--input is required unless --selftest is given")
    if not args.output and not args.pending_fetch_output and not args.pending_fetch_ids_output:
        ap.error("give at least one of --output / --pending-fetch-output / --pending-fetch-ids-output")

    classified = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    relaxed_authors = None
    if args.relaxed_authors:
        relaxed_df = pd.read_csv(args.relaxed_authors, dtype=str, keep_default_na=False)
        relaxed_df["author_key"] = relaxed_df["author"].astype(str).str.strip().str.lower()
        if "min_words" in relaxed_df.columns:
            has_value = relaxed_df["min_words"].astype(str).str.strip() != ""
            relaxed_df.loc[has_value, "min_words"] = relaxed_df.loc[has_value, "min_words"].astype(int)
            relaxed_df.loc[~has_value, "min_words"] = args.relaxed_min_words
        else:
            relaxed_df["min_words"] = args.relaxed_min_words
        relaxed_authors = dict(zip(relaxed_df["author_key"], relaxed_df["min_words"].astype(int)))
        by_floor = relaxed_df["min_words"].astype(int).value_counts().sort_index()
        summary = ", ".join(f"{n} at >= {mw} word(s)" for mw, n in by_floor.items())
        print(f"--relaxed-authors: {len(relaxed_authors):,} author(s) with the cue requirement dropped ({summary}).")

    if args.output:
        n_candidates = int((classified["structure_kind"] == "reply_depth2").sum())
        kept = extract(classified, relaxed_authors=relaxed_authors)
        print(f"{n_candidates:,} reply_depth2 candidates -> {len(kept):,} kept as \"meaningful\" "
              f"({len(kept) / n_candidates:.1%})." if n_candidates else "No reply_depth2 rows found in --input.")
        if len(kept):
            print("Cue frequency among kept interactions:")
            for cue in sorted({c for cues in kept["matched_cues"] for c in cues.split(";")}):
                n = kept["matched_cues"].str.contains(cue, regex=False).sum()
                print(f"  {cue:15s} {n:,}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        kept.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")

    if args.pending_fetch_output or args.pending_fetch_ids_output:
        n_unresolved = int((classified["structure_kind"] == "unresolved").sum())
        pending = find_pending_fetch_candidates(classified)
        n_ids = pending["parent_bare_id"].nunique()
        print(f"\n{n_unresolved:,} unresolved rows -> {len(pending):,} replies worth fetching a parent for "
              f"({n_ids:,} distinct parent id(s), after excluding replies whose own text has no qualifying cue).")

        if args.pending_fetch_output:
            args.pending_fetch_output.parent.mkdir(parents=True, exist_ok=True)
            pending.to_csv(args.pending_fetch_output, index=False)
            print(f"Saved: {args.pending_fetch_output}")

        if args.pending_fetch_ids_output:
            ids = pd.DataFrame({"id": sorted(pending["parent_bare_id"].unique())})
            args.pending_fetch_ids_output.parent.mkdir(parents=True, exist_ok=True)
            ids.to_csv(args.pending_fetch_ids_output, index=False)
            print(f"Saved {len(ids):,} deduplicated parent id(s) -> {args.pending_fetch_ids_output} "
                  "(feed to sql/fetch_parent_comments.sql)")


if __name__ == "__main__":
    main()
