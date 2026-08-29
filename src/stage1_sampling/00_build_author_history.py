"""
Pipeline position: Stage 1, step 00 (first step) -- see claude_project_plan/
07_REPO_STRUCTURE.md §1 for the full execution-order table and data-location
map; that file is the source of truth for pipeline order, not this docstring.

Convert cohort_history_* exports (the real Stage-1 candidate pool) into the
canonical author_full_history.csv schema that 05_infer_demographics_llm.py and
02_compute_subreddit_activity.py both consume.

Context: the real 1,000-user sample landed as DB tables/exports --
cohort_users (author_key, n_seed_posts, n_recency_posts_applied, bucket),
cohort_variants (author, author_key, bucket -- the author_key<->real-username
mapping), and cohort_history_submissions (author_key, kind, id, created_utc,
subreddit, text, title, ...). Only the last one is consumed here; the other
two aren't joined in (see the keying note below). Comments aren't in the DB
pull yet -- run with --submissions only for now, add --comments once they
land (no other changes needed).

Per 2026-08-23 decision: pipeline files are keyed on the pseudonymous
`author_key`, never resolved back to the real Reddit username via
cohort_variants -- keeps real usernames out of every downstream file
(demographics output, personas, the K.2 raw-output archive), which matters
for the eventual disclosure-class call. Pass --author-column to key on a
different column instead.

`kind` is NOT trusted as a submission/comment discriminator -- its actual
values weren't confirmed against a real export at build time (it may encode
something else entirely, e.g. which selection criterion a post satisfied,
given cohort_users' own n_seed_posts/n_recency_posts_applied columns).
post_type is instead set from which CLI flag a file was passed under
(--submissions vs --comments), which is unambiguous by construction.

Column presence is detected per input file (case-insensitive, with common
aliases like text/body/selftext for the body field) rather than hardcoded,
since cohort_history_submissions' full column list wasn't fully confirmed
either -- a per-file report prints what was found vs. defaulted blank, so a
schema mismatch is visible immediately instead of silently producing blanks.

--submissions/--comments each already accept MULTIPLE files (nargs="+") --
this is the intended way to combine an original pull with a later
supplementary batch (e.g. Peer's targeted query for hard-to-reach demographic
cells, 02_DECISIONS_LOG.md 2026-08-24/2026-08-25), not a separate merge
script: just pass both files under the same flag. Rows are deduplicated by
Reddit's own id (globally unique across submissions and comments) after
concatenating, so an author who happens to appear in both batches doesn't
get double-counted downstream.

Usage:
    python 00_build_author_history.py \\
        --submissions cohort_history_submissions.csv \\
        --output author_full_history.csv

    # once comments land:
    python 00_build_author_history.py \\
        --submissions cohort_history_submissions.csv \\
        --comments cohort_history_comments.csv \\
        --output author_full_history.csv

    # combining an original pull with a supplementary batch (both types):
    python 00_build_author_history.py \\
        --submissions cohort_history_submissions_v1.csv.gz cohort_history_submissions_v2.csv.gz \\
        --comments cohort_history_comments_v1.csv.gz cohort_history_comments_v2.csv.gz \\
        --output author_full_history.csv

    python 00_build_author_history.py --selftest   # no data required
"""

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

OUTPUT_COLUMNS = [
    "post_type", "id", "author", "author_created_utc", "subreddit",
    "created_utc", "title", "body", "score", "link_id", "parent_id",
]

# Candidate source-column names per output field (besides author/post_type,
# which are handled specially), case-insensitive, first match wins.
FIELD_ALIASES = {
    "id": ["id"],
    "author_created_utc": ["author_created_utc"],
    "subreddit": ["subreddit"],
    "created_utc": ["created_utc"],
    "title": ["title"],
    "body": ["text", "body", "selftext"],
    "score": ["score"],
    "link_id": ["link_id"],
    "parent_id": ["parent_id"],
}


def find_column(columns_lower: dict[str, str], candidates: list[str]) -> Optional[str]:
    for cand in candidates:
        if cand in columns_lower:
            return columns_lower[cand]
    return None


def convert_file(path: Path, post_type: str, author_column: str) -> tuple[pd.DataFrame, dict]:
    """Returns (converted rows in OUTPUT_COLUMNS order, {field: source_column_or_None})."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns_lower = {c.lower(): c for c in df.columns}

    author_src = columns_lower.get(author_column.lower())
    if author_src is None:
        raise ValueError(
            f"{path}: no '{author_column}' column found (columns present: {list(df.columns)}). "
            "Pass --author-column to point at the right one."
        )

    report = {"author": author_src}
    out = pd.DataFrame(index=df.index)
    out["post_type"] = post_type
    out["author"] = df[author_src].astype(str).str.strip().str.lower()

    for field, candidates in FIELD_ALIASES.items():
        src = find_column(columns_lower, [c.lower() for c in candidates])
        report[field] = src
        out[field] = df[src] if src is not None else ""

    return out[OUTPUT_COLUMNS], report


def print_report(path: Path, post_type: str, n_rows: int, n_authors: int, report: dict) -> None:
    print(f"\n{path} (post_type={post_type}): {n_rows:,} rows, {n_authors:,} distinct authors")
    for field in ["author"] + list(FIELD_ALIASES.keys()):
        src = report.get(field)
        status = f"<- '{src}'" if src else "-- NOT FOUND, defaulted blank"
        print(f"  {field:20s} {status}")


def build(submissions: list[Path], comments: list[Path], author_column: str) -> pd.DataFrame:
    frames = []
    for path in submissions:
        df, report = convert_file(path, "submission", author_column)
        print_report(path, "submission", len(df), df["author"].nunique(), report)
        frames.append(df)
    for path in comments:
        df, report = convert_file(path, "comment", author_column)
        print_report(path, "comment", len(df), df["author"].nunique(), report)
        frames.append(df)
    if not frames:
        raise ValueError("No --submissions or --comments files given.")
    combined = pd.concat(frames, ignore_index=True)

    # Reddit's own id is globally unique across all content (submissions and
    # comments share one id space) -- so a duplicate id across the given
    # files always means the same real post, not a coincidence. This matters
    # once --submissions/--comments are given MULTIPLE files each (e.g.
    # combining an original pull with a later supplementary batch, per
    # 02_DECISIONS_LOG.md 2026-08-25) -- if the batches overlap at all (a
    # supplementary query that happens to re-include an already-collected
    # author), the same post would otherwise be double-counted downstream
    # (subreddit-activity counts, inference token budget, etc.).
    n_before = len(combined)
    combined = combined.drop_duplicates(subset="id", keep="first")
    n_dupes = n_before - len(combined)
    if n_dupes:
        print(f"\n{n_dupes:,} duplicate row(s) (same Reddit id, appeared in more than one "
              f"input file) -- kept the first occurrence, dropped the rest.")

    return combined


def selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sub_path = tmp / "cohort_history_submissions.csv"
        pd.DataFrame([
            {"author_key": "User_A", "kind": "seed", "id": "s1", "created_utc": "1700000000",
             "subreddit": "Alaska", "text": "Loving the aurora tonight.", "title": "Aurora watch"},
            {"author_key": "user_a", "kind": "recency", "id": "s2", "created_utc": "1710000000",
             "subreddit": "personalfinance", "text": "Maxed my 401k.", "title": ""},
            {"author_key": "User_B", "kind": "seed", "id": "s3", "created_utc": "1705000000",
             "subreddit": "Alaska", "text": "", "title": "Best fishing spots?"},
        ]).to_csv(sub_path, index=False)

        com_path = tmp / "cohort_history_comments.csv"
        pd.DataFrame([
            {"author_key": "User_A", "id": "c1", "created_utc": "1701000000",
             "subreddit": "Alaska", "body": "Agreed, gorgeous.", "link_id": "t3_s1", "parent_id": "t3_s1"},
        ]).to_csv(com_path, index=False)

        combined = build([sub_path], [com_path], author_column="author_key")

        assert list(combined.columns) == OUTPUT_COLUMNS
        assert len(combined) == 4
        assert set(combined["author"]) == {"user_a", "user_b"}, combined["author"].tolist()
        assert (combined.loc[combined["id"] == "s1", "post_type"] == "submission").all()
        assert (combined.loc[combined["id"] == "c1", "post_type"] == "comment").all()

        row_s1 = combined.loc[combined["id"] == "s1"].iloc[0]
        assert row_s1["title"] == "Aurora watch"
        assert row_s1["body"] == "Loving the aurora tonight."
        assert row_s1["author_created_utc"] == ""
        assert row_s1["score"] == ""

        row_c1 = combined.loc[combined["id"] == "c1"].iloc[0]
        assert row_c1["title"] == ""
        assert row_c1["body"] == "Agreed, gorgeous."
        assert row_c1["link_id"] == "t3_s1"

        # "User_A" and "user_a" must collapse to one author (case-insensitive key)
        assert (combined["author"] == "user_a").sum() == 3

        # --- multi-batch merge: a second submissions file overlapping the first ---
        sub_path_v2 = tmp / "cohort_history_submissions_v2.csv"
        pd.DataFrame([
            # s1 again -- same id, e.g. author_a got re-pulled by a supplementary
            # query -- must be deduplicated, not double-counted
            {"author_key": "User_A", "kind": "seed", "id": "s1", "created_utc": "1700000000",
             "subreddit": "Alaska", "text": "Loving the aurora tonight.", "title": "Aurora watch"},
            # a genuinely new author from the supplementary batch
            {"author_key": "User_C", "kind": "seed", "id": "s4", "created_utc": "1706000000",
             "subreddit": "Alaska", "text": "", "title": "New here, hi!"},
        ]).to_csv(sub_path_v2, index=False)

        merged = build([sub_path, sub_path_v2], [com_path], author_column="author_key")
        assert len(merged) == 5, len(merged)  # 4 original + 1 new (s4) -- s1 duplicate dropped
        assert (merged["id"] == "s1").sum() == 1, "duplicate id across batches was not deduplicated"
        assert set(merged["author"]) == {"user_a", "user_b", "user_c"}

    print("Self-test passed: submissions+comments conversion, column-aliasing "
          "(text->body), post_type assignment by source file, case-insensitive "
          "author-key normalization, and multi-batch merging with id-based "
          "deduplication all work end to end without real data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submissions", nargs="+", type=Path, default=[],
                    help="One or more cohort_history_submissions-shaped CSV exports")
    ap.add_argument("--comments", nargs="+", type=Path, default=[],
                    help="One or more comment-history-shaped CSV exports (once available)")
    ap.add_argument("--output", type=Path, help="Output path: author_full_history.csv")
    ap.add_argument("--author-column", type=str, default="author_key",
                    help="Column to use as the pipeline identity/join key "
                         "(default: author_key, the pseudonymous id -- never resolved "
                         "back to the real username)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.output:
        ap.error("--output is required unless --selftest is given")

    combined = build(args.submissions, args.comments, args.author_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nSaved {len(combined):,} rows ({combined['author'].nunique():,} distinct authors) -> {args.output}")


if __name__ == "__main__":
    main()
