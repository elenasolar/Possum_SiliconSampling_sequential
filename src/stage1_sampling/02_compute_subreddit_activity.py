"""
Pipeline position: Stage 1, step 01 -- runs after 00_build_author_history.py;
feeds 03_assign_state.py and 04_infer_demographics_embed.py. See
claude_project_plan/07_REPO_STRUCTURE.md §1 for the full execution-order table.

Per-user x subreddit post/comment counts from one or more author_full_history-
shaped CSVs (e.g. 00_build_author_history.py's output) -- the activity signal
Waller & Anderson community-embedding scoring needs.

Mirrors sql/alaska_pilot.sql's Step 4a/4b (group by author + lower(subreddit),
count(*)), done here in Python since the real cohort data already landed as
CSV exports rather than a fresh SQL pull -- same output shape either way.

Outputs (give at least one):
    --counts-output   flat CSV: author, subreddit, total
    --activity-output nested JSONL, one line per author:
                      {"author": ..., "subreddit_activity": {sub: {"total": n}, ...}}
                      -- exact schema 04_infer_demographics_embed.py --user-activity expects.

Usage:
    python 02_compute_subreddit_activity.py \\
        --input author_full_history.csv \\
        --activity-output user_subreddit_activity.jsonl \\
        --counts-output user_subreddit_counts.csv

    python 02_compute_subreddit_activity.py --selftest   # no data required
"""

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd


def compute_counts(paths: list[Path]) -> pd.DataFrame:
    frames = []
    n_input_rows = 0
    for path in paths:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        n_input_rows += len(df)
        for col in ("author", "subreddit"):
            if col not in df.columns:
                raise ValueError(f"{path}: missing required column '{col}'")
        frames.append(df[["author", "subreddit"]])
    combined = pd.concat(frames, ignore_index=True)
    combined["author"] = combined["author"].str.strip().str.lower()
    combined["subreddit"] = combined["subreddit"].str.strip().str.lower()
    combined = combined[(combined["author"] != "") & (combined["subreddit"] != "")]

    counts = (
        combined.groupby(["author", "subreddit"])
        .size()
        .reset_index(name="total")
        .sort_values(["author", "total"], ascending=[True, False])
        .reset_index(drop=True)
    )
    n_authors = counts["author"].nunique()
    print(f"{n_input_rows:,} input rows -> {len(counts):,} (author, subreddit) pairs across {n_authors:,} authors.")
    return counts


def write_activity_jsonl(counts: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for author, group in counts.groupby("author"):
            activity = {row.subreddit: {"total": int(row.total)} for row in group.itertuples()}
            f.write(json.dumps({"author": author, "subreddit_activity": activity}, ensure_ascii=False) + "\n")


def selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / "author_full_history.csv"
        pd.DataFrame([
            {"author": "User_A", "subreddit": "Alaska"},
            {"author": "user_a", "subreddit": "alaska"},
            {"author": "user_a", "subreddit": "personalfinance"},
            {"author": "User_B", "subreddit": "Alaska"},
            {"author": "", "subreddit": "spam"},
        ]).to_csv(path, index=False)

        counts = compute_counts([path])
        assert set(counts["author"]) == {"user_a", "user_b"}
        alaska_a = counts[(counts.author == "user_a") & (counts.subreddit == "alaska")]["total"].iloc[0]
        assert alaska_a == 2, alaska_a

        activity_path = tmp / "activity.jsonl"
        write_activity_jsonl(counts, activity_path)
        lines = [json.loads(l) for l in activity_path.read_text(encoding="utf-8").splitlines()]
        by_author = {l["author"]: l["subreddit_activity"] for l in lines}
        assert by_author["user_a"] == {"alaska": {"total": 2}, "personalfinance": {"total": 1}}
        assert by_author["user_b"] == {"alaska": {"total": 1}}

    print("Self-test passed: case-insensitive author/subreddit aggregation and "
          "JSONL activity export both match 04_infer_demographics_embed.py's expected schema.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", type=Path,
                     help="One or more author_full_history-shaped CSVs (needs 'author' and 'subreddit' columns)")
    ap.add_argument("--counts-output", type=Path, default=None, help="Flat CSV: author, subreddit, total")
    ap.add_argument("--activity-output", type=Path, default=None,
                     help="Nested JSONL for 04_infer_demographics_embed.py --user-activity")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.input:
        ap.error("--input is required unless --selftest is given")
    if not args.counts_output and not args.activity_output:
        ap.error("give at least one of --counts-output / --activity-output")

    counts = compute_counts(args.input)

    if args.counts_output:
        args.counts_output.parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(args.counts_output, index=False)
        print(f"Saved: {args.counts_output}")
    if args.activity_output:
        write_activity_jsonl(counts, args.activity_output)
        print(f"Saved: {args.activity_output}")


if __name__ == "__main__":
    main()
