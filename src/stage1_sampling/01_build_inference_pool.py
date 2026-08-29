"""
Pipeline position: Stage 1, step 01 -- runs after 00_build_author_history.py;
feeds 05_infer_demographics_llm.py directly. See claude_project_plan/
07_REPO_STRUCTURE.md §1.

Builds the demographic-inference input pool: submissions + first-level
comments only (a comment directly on a submission, not a reply to another
comment), mixed together in the canonical author_full_history.csv schema.

Deliberately NOT built from 07_classify_comment_structure.py's classification
(revised 2026-08-25 -- see 02_DECISIONS_LOG.md): that script's full
reply-depth classification needs an expensive external lookup (fetching
parent comments authored by users outside the sample) to tell a depth-2
reply from a deeper one -- but distinguishing "first-level comment" from
"any-depth reply" needs NONE of that. It's fully determined by a comment's
OWN parent_id prefix alone (t3_ = points at the submission = first-level;
t1_ = points at another comment = a reply, any depth -- no lookup needed
either way). Keeping this as its own lightweight, fetch-free step means
Stage 1's demographic inference (needed BEFORE sampling, over the whole
~10,000-candidate cohort) never has to wait on the manual SQL fetch that
07_classify_comment_structure.py needs for ITS purpose (Stage 2 persona
content, which now runs AFTER sampling, scoped to just the ~1,000 selected
agents). Sampling depends on demographic inference; demographic inference
must not depend on anything sampling-adjacent -- that dependency cycle is
exactly what this split resolves.

No recency/token-budget selection happens here -- 05_infer_demographics_llm.py
already does that itself (select_within_budget: most-recent-first, mixing
submissions and comments, up to its own token budget) on whatever file it's
given. This script's only job is the type filter.

Input: author_full_history.csv (00_build_author_history.py's output).

Output: same schema, filtered to submission + first-level-comment rows only.

Usage:
    python 01_build_inference_pool.py \\
        --input author_full_history.csv \\
        --output author_full_history_for_inference.csv

    python 01_build_inference_pool.py --selftest   # no data required
"""

import argparse
from pathlib import Path

import pandas as pd

HISTORY_COLUMNS = [
    "post_type", "id", "author", "author_created_utc", "subreddit",
    "created_utc", "title", "body", "score", "link_id", "parent_id",
]


def is_inference_eligible(row) -> bool:
    if row.get("post_type") == "submission":
        return True
    if row.get("post_type") == "comment":
        return (row.get("parent_id") or "").strip().startswith("t3_")
    return False


def filter_inference_pool(history: pd.DataFrame) -> pd.DataFrame:
    mask = history.apply(is_inference_eligible, axis=1)
    kept = history[mask]
    cols = [c for c in HISTORY_COLUMNS if c in kept.columns]
    return kept[cols]


def selftest() -> None:
    history = pd.DataFrame([
        {"post_type": "submission", "id": "s1", "author": "alice", "author_created_utc": "",
         "subreddit": "test", "created_utc": "1", "title": "T", "body": "", "score": "1",
         "link_id": "", "parent_id": ""},
        {"post_type": "comment", "id": "c1", "author": "alice", "author_created_utc": "",
         "subreddit": "test", "created_utc": "2", "title": "", "body": "first-level comment", "score": "1",
         "link_id": "t3_s1", "parent_id": "t3_s1"},
        # reply -- excluded regardless of depth, no lookup needed to know that
        {"post_type": "comment", "id": "r1", "author": "alice", "author_created_utc": "",
         "subreddit": "test", "created_utc": "3", "title": "", "body": "a reply to c1", "score": "1",
         "link_id": "t3_s1", "parent_id": "t1_c1"},
    ])
    pool = filter_inference_pool(history)
    assert set(pool["id"]) == {"s1", "c1"}, set(pool["id"])
    assert list(pool.columns) == HISTORY_COLUMNS

    print("Self-test passed: submissions and first-level comments kept, replies "
          "(any depth) excluded, with no external lookup required, end to end "
          "without real data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="author_full_history.csv (00_build_author_history.py's output)")
    ap.add_argument("--output", type=Path, help="Output path: author_full_history_for_inference.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.input or not args.output:
        ap.error("--input and --output are required unless --selftest is given")

    history = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    pool = filter_inference_pool(history)
    print(f"{len(history):,} rows -> {len(pool):,} kept (submissions + first-level comments), "
          f"{pool['author'].nunique():,} authors.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
