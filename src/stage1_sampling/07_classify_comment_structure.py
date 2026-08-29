"""
Pipeline position: Stage 1, step 07 (revised 2026-08-25 -- moved from step 03;
see 02_DECISIONS_LOG.md) -- runs AFTER 06_stratified_sample.py, scoped to just
the sampled agents via --authors (NOT the whole ~10,000-candidate cohort --
this step's external-parent fetch is expensive and manual, so scoping it to
the ~1,000 selected agents instead of everyone cuts that workload by ~10x).
Feeds stage2_agents/00_extract_meaningful_interactions.py. Does NOT feed
Stage 1 demographic inference any more -- that's 01_build_inference_pool.py's
job now, a lightweight fetch-free filter that runs right after step 00, so
inference (needed BEFORE sampling) never has to wait on anything sampling-
adjacent. Run TWICE (pass 1, then pass 2 with --external-parents after the
manual SQL fetch) -- see claude_project_plan/07_REPO_STRUCTURE.md §1.

Classify each row of an author_full_history.csv-shaped file (submissions +
all-depth comments -- e.g. 00_build_author_history.py's output once run with
both --submissions and --comments) into its structural role:

    submission            -- the original post
    first_level_comment   -- a comment whose parent is the submission itself
                             (parent_id starts with "t3_") -- feeds Stage 1
                             demographic inference alongside submissions
    reply_depth2          -- a comment replying to a FIRST-LEVEL comment
                             (parent_id starts with "t1_", and THAT parent's
                             own parent_id starts with "t3_") -- feeds Stage 2
                             persona construction ("comment-reply
                             interactions", see 02_DECISIONS_LOG.md 2026-08-24)
    reply_deeper          -- a comment replying to another reply (depth 3+) --
                             out of scope per the 2026-08-24 decision; kept
                             here only for visibility/diagnostics, not dropped
    unresolved            -- parent_id is a "t1_" reply whose parent comment
                             could not be found, even after --external-parents
                             (deleted/removed parent, or genuinely still
                             missing) -- reported, never silently dropped

Why this needs more than a row's own parent_id: the t3_/t1_ prefix alone
already distinguishes "comment" from "reply" -- no lookup needed for that.
But telling a depth-2 reply apart from a depth-3+ one requires knowing
whether the PARENT comment was itself first-level -- i.e. the parent's own
parent_id. If the parent was authored by one of our own sampled users, its
row is already sitting in the same input file (free local lookup). If it was
authored by someone OUTSIDE our sample -- the common case, since most
commenters on any thread aren't one of our ~10,000 sampled users -- we don't
have that row at all, because our DB pull only ever selected rows authored
by our own cohort. Resolving those needs one targeted, id-indexed lookup
against the full reddit.comments table (see sql/fetch_parent_comments.sql)
-- NOT a full-subreddit scan, since Reddit comment ids are looked up by
primary key.

Two-pass workflow:
    1. Run once without --external-parents. Parent ids that can't be
       resolved from --input alone are written to --missing-parent-ids-output.
    2. Hand that file to sql/fetch_parent_comments.sql (via \\copy ... FROM)
       on the data server, run it, pull fetched_parent_comments.csv back.
    3. Re-run this script with --external-parents fetched_parent_comments.csv
       to finish classifying every row pass 1 left unresolved.

Output: the same rows as --input, plus `structure_kind`, `parent_author`,
`parent_body`, `parent_created_utc` (populated wherever resolvable).

--authors, if given, restricts --input to just the authors listed in that
file's 'author' column (e.g. 06_stratified_sample.py's agents.csv) BEFORE
classifying -- this is how the ~10,000-candidate cohort narrows down to the
~1,000 sampled agents this step is meant to run against. Omit it to process
everyone (rarely what you want now that this step runs after sampling).

Usage:
    python 07_classify_comment_structure.py \\
        --input author_full_history.csv \\
        --authors agents.csv \\
        --output author_full_history_classified.csv \\
        --missing-parent-ids-output missing_parent_ids.csv

    # once sql/fetch_parent_comments.sql has been run on the data server:
    python 07_classify_comment_structure.py \\
        --input author_full_history.csv \\
        --authors agents.csv \\
        --external-parents fetched_parent_comments.csv \\
        --output author_full_history_classified.csv \\
        --missing-parent-ids-output missing_parent_ids.csv

    python 07_classify_comment_structure.py --selftest   # no data required
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def strip_prefix(fullname) -> Optional[str]:
    """"t1_abc123" -> "abc123"; blank/NaN -> None."""
    if not fullname or not isinstance(fullname, str):
        return None
    fullname = fullname.strip()
    if not fullname:
        return None
    if fullname.startswith("t1_") or fullname.startswith("t3_"):
        return fullname[3:]
    return fullname  # already bare -- tolerate it


def filter_to_authors(history: pd.DataFrame, authors_df: pd.DataFrame) -> pd.DataFrame:
    """Restricts history to rows whose author appears in authors_df's
    'author' column (case-insensitive) -- e.g. narrowing the whole candidate
    cohort down to just the sampled agents before the expensive external-
    parent fetch."""
    keep = set(authors_df["author"].astype(str).str.strip().str.lower())
    mask = history["author"].astype(str).str.strip().str.lower().isin(keep)
    return history[mask]


def _index_lookup(ids, parent_ids, authors, bodies, created_utcs,
                   parent_id_lookup: dict, author_lookup: dict, body_lookup: dict,
                   created_utc_lookup: dict) -> None:
    """Fills the four bare-id -> value dicts in place, first-write-wins (so
    calling this with own_comments first, then external_parents, makes our
    own pull take priority over the external fetch on any id collision)."""
    for rid, pid, author, body, created_utc in zip(ids, parent_ids, authors, bodies, created_utcs):
        if rid in parent_id_lookup:
            continue
        parent_id_lookup[rid] = pid or ""
        author_lookup[rid] = author or ""
        body_lookup[rid] = body or ""
        created_utc_lookup[rid] = created_utc or ""


def classify(history: pd.DataFrame, external_parents: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Returns (history with structure_kind/parent_* columns added, sorted
    list of parent ids that couldn't be resolved from either source).

    Vectorized (2026-08-26, see 02_DECISIONS_LOG.md): the original row-by-row
    df.iterrows() + df.at[] implementation doesn't scale to real pull sizes
    (~9M rows) -- this rewrite uses boolean masks + dict .map() lookups
    (C-level) instead, same behavior, verified against the same selftest."""
    df = history.copy()
    for col in ("parent_body", "parent_author", "parent_created_utc"):
        df[col] = ""
    df["structure_kind"] = ""

    parent_id_col = df["parent_id"].fillna("").astype(str).str.strip()

    is_submission = df["post_type"] == "submission"
    is_comment = df["post_type"] == "comment"
    is_first_level = is_comment & parent_id_col.str.startswith("t3_")
    is_reply = is_comment & parent_id_col.str.startswith("t1_")
    is_unresolved_early = ~(is_submission | is_first_level | is_reply)

    df.loc[is_submission, "structure_kind"] = "submission"
    df.loc[is_first_level, "structure_kind"] = "first_level_comment"
    df.loc[is_unresolved_early, "structure_kind"] = "unresolved"

    # Local lookup: any comment in OUR OWN pull, keyed by bare id -- covers
    # the case where the parent was also authored by one of our sampled users.
    own_comments = df[is_comment]

    parent_id_lookup: dict = {}
    author_lookup: dict = {}
    body_lookup: dict = {}
    created_utc_lookup: dict = {}
    _index_lookup(own_comments["id"], own_comments["parent_id"], own_comments["author"],
                  own_comments["body"], own_comments["created_utc"],
                  parent_id_lookup, author_lookup, body_lookup, created_utc_lookup)
    if external_parents is not None and len(external_parents):
        _index_lookup(external_parents["id"], external_parents["parent_id"], external_parents["author"],
                      external_parents["body"], external_parents["created_utc"],
                      parent_id_lookup, author_lookup, body_lookup, created_utc_lookup)

    parent_bare = parent_id_col.where(is_reply).str.slice(3)
    resolved_grandparent = parent_bare.map(parent_id_lookup)
    found = is_reply & resolved_grandparent.notna()
    missing = is_reply & ~found

    df.loc[found, "parent_author"] = parent_bare[found].map(author_lookup)
    df.loc[found, "parent_body"] = parent_bare[found].map(body_lookup)
    df.loc[found, "parent_created_utc"] = parent_bare[found].map(created_utc_lookup)

    grandparent_id = resolved_grandparent.fillna("").astype(str).str.strip()
    is_depth2 = found & grandparent_id.str.startswith("t3_")
    is_deeper = found & grandparent_id.str.startswith("t1_")
    is_found_unresolved = found & ~(is_depth2 | is_deeper)

    df.loc[is_depth2, "structure_kind"] = "reply_depth2"
    df.loc[is_deeper, "structure_kind"] = "reply_deeper"
    df.loc[is_found_unresolved | missing, "structure_kind"] = "unresolved"

    missing_parent_ids = set(parent_bare[missing].dropna())
    return df, sorted(missing_parent_ids)


def selftest() -> None:
    # Two of OUR sampled users: author_a (submission + first-level comment)
    # and author_b (replies to author_a's comment, and to an external one).
    history = pd.DataFrame([
        {"post_type": "submission", "id": "sub1", "author": "author_a", "author_created_utc": "",
         "subreddit": "test", "created_utc": "1700000000", "title": "Topic", "body": "", "score": "10",
         "link_id": "", "parent_id": ""},
        {"post_type": "comment", "id": "c1", "author": "author_a", "author_created_utc": "",
         "subreddit": "test", "created_utc": "1700000100", "title": "", "body": "First take on this.",
         "score": "5", "link_id": "t3_sub1", "parent_id": "t3_sub1"},
        # depth-2 reply to c1 (parent IS in our own pull -- local resolution)
        {"post_type": "comment", "id": "r1", "author": "author_b", "author_created_utc": "",
         "subreddit": "test", "created_utc": "1700000200", "title": "", "body": "But you're wrong about that.",
         "score": "2", "link_id": "t3_sub1", "parent_id": "t1_c1"},
        # depth-3 reply to r1 (reply to a reply) -> reply_deeper
        {"post_type": "comment", "id": "r2", "author": "author_a", "author_created_utc": "",
         "subreddit": "test", "created_utc": "1700000300", "title": "", "body": "No, actually I disagree.",
         "score": "1", "link_id": "t3_sub1", "parent_id": "t1_r1"},
        # depth-2 reply to an EXTERNAL comment (not authored by author_a/b) -> needs external fetch
        {"post_type": "comment", "id": "r3", "author": "author_b", "author_created_utc": "",
         "subreddit": "test", "created_utc": "1700000400", "title": "", "body": "Clearly that is not correct.",
         "score": "0", "link_id": "t3_sub1", "parent_id": "t1_ext1"},
        # reply to a truly-unresolvable parent (not in our pull, not in external-parents either)
        {"post_type": "comment", "id": "r4", "author": "author_a", "author_created_utc": "",
         "subreddit": "test", "created_utc": "1700000500", "title": "", "body": "Agreed with this one.",
         "score": "0", "link_id": "t3_sub1", "parent_id": "t1_ghost"},
    ])

    # -- Pass 1: no external parents yet --
    classified1, missing1 = classify(history, None)
    by_id = classified1.set_index("id")
    assert by_id.loc["sub1", "structure_kind"] == "submission"
    assert by_id.loc["c1", "structure_kind"] == "first_level_comment"
    assert by_id.loc["r1", "structure_kind"] == "reply_depth2"
    assert by_id.loc["r1", "parent_body"] == "First take on this."
    assert by_id.loc["r2", "structure_kind"] == "reply_deeper"
    assert by_id.loc["r3", "structure_kind"] == "unresolved"  # not yet resolvable
    assert by_id.loc["r4", "structure_kind"] == "unresolved"
    assert set(missing1) == {"ext1", "ghost"}, missing1

    # -- Pass 2: external-parents resolves ext1 (author outside our sample);
    # "ghost" simulates a deleted/never-returned parent, still missing --
    external = pd.DataFrame([
        {"id": "ext1", "author": "outsider", "author_created_utc": "", "subreddit": "test",
         "created_utc": "1699999900", "link_id": "t3_sub1", "parent_id": "t3_sub1", "body": "Outside first take."},
    ])
    classified2, missing2 = classify(history, external)
    by_id2 = classified2.set_index("id")
    assert by_id2.loc["r3", "structure_kind"] == "reply_depth2"
    assert by_id2.loc["r3", "parent_author"] == "outsider"
    assert by_id2.loc["r3", "parent_body"] == "Outside first take."
    assert by_id2.loc["r4", "structure_kind"] == "unresolved"  # still missing -- reported, not hidden
    assert missing2 == ["ghost"], missing2

    # --authors: scope down to just author_b before classifying -- author_a's
    # rows (sub1, c1, r2, r4) must be gone entirely, author_b's (r1, r3) kept.
    # Real trade-off, not a bug: r1's parent "c1" was locally resolvable
    # BEFORE scoping (author_a was in the same input file); after scoping to
    # just author_b, c1 is gone too, so r1 now needs an external fetch same
    # as r3 does. Scoping to the sample means more "external" fetches than
    # scoping to the whole cohort would, in exchange for needing far fewer
    # fetches overall (~1,000 sampled agents' unresolved parents, not
    # ~10,000 candidates') -- see 02_DECISIONS_LOG.md, 2026-08-25.
    agents = pd.DataFrame([{"author": "Author_B"}])  # case-insensitive on purpose
    scoped = filter_to_authors(history, agents)
    assert set(scoped["id"]) == {"r1", "r3"}, set(scoped["id"])
    scoped_classified, scoped_missing = classify(scoped, None)
    assert set(scoped_classified["id"]) == {"r1", "r3"}
    assert scoped_missing == ["c1", "ext1"], scoped_missing

    print("Self-test passed: submission/first-level/depth-2/deeper/unresolved "
          "classification, local vs. external parent resolution, and the "
          "--authors scoping filter all work end to end without real data or "
          "a database connection.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="author_full_history.csv-shaped file (submissions + all-depth comments)")
    ap.add_argument("--authors", type=Path, default=None,
                     help="Optional filter: only classify rows whose author appears in this file's 'author' column "
                          "(e.g. 06_stratified_sample.py's agents.csv) -- scopes the expensive external-parent "
                          "fetch down to just the sampled agents instead of the whole candidate cohort.")
    ap.add_argument("--external-parents", type=Path, default=None,
                     help="fetch_parent_comments.sql's export -- comments authored by users outside our sample. "
                          "Columns: id, author, author_created_utc, subreddit, created_utc, link_id, parent_id, body")
    ap.add_argument("--output", type=Path, help="Full classified output (all rows, structure_kind + parent_* columns added)")
    ap.add_argument("--missing-parent-ids-output", type=Path, default=None,
                     help="Parent ids still unresolved after --external-parents (or all of them, on a first pass) -- "
                          "hand this to sql/fetch_parent_comments.sql's \\copy ... FROM step")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.input or not args.output:
        ap.error("--input and --output are required unless --selftest is given")

    history = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if args.authors:
        authors_df = pd.read_csv(args.authors, dtype=str, keep_default_na=False)
        before = len(history)
        history = filter_to_authors(history, authors_df)
        print(f"--authors filter: {before:,} rows -> {len(history):,} rows "
              f"({history['author'].nunique():,} authors)")
    external = pd.read_csv(args.external_parents, dtype=str, keep_default_na=False) if args.external_parents else None

    classified, missing = classify(history, external)

    print("Structure kind breakdown:")
    for kind, n in classified["structure_kind"].value_counts().items():
        print(f"  {kind:20s} {n:,}")
    if missing:
        note = " or --external-parents" if external is not None else ""
        print(f"\n{len(missing):,} parent comment id(s) could not be resolved from --input{note} -- "
              "likely authored by users outside our sample.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    if args.missing_parent_ids_output:
        args.missing_parent_ids_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": missing}).to_csv(args.missing_parent_ids_output, index=False)
        print(f"Saved {len(missing):,} missing parent id(s) -> {args.missing_parent_ids_output} "
              "(feed to sql/fetch_parent_comments.sql)")

    if args.inference_pool_output:
        pool = classified[classified["structure_kind"].isin(["submission", "first_level_comment"])][HISTORY_COLUMNS]
        args.inference_pool_output.parent.mkdir(parents=True, exist_ok=True)
        pool.to_csv(args.inference_pool_output, index=False)
        print(f"Saved {len(pool):,} rows (submissions + first-level comments) -> {args.inference_pool_output}")


if __name__ == "__main__":
    main()
