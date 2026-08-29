"""
Pipeline position: Stage 1, step 04 -- runs after 02_compute_subreddit_activity.py;
its output optionally feeds 05_infer_demographics_llm.py as background context.
See claude_project_plan/07_REPO_STRUCTURE.md §1 for the full execution order.

Waller & Anderson community-embedding demographic scoring.

Computes per-user ideology, gender, and age scores as an activity-weighted
average of subreddit-level community-embedding scores.

Reference: claude_project_plan/references/02_compute_user_scores.ipynb
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Waller & Anderson subreddit-score column -> output user-score column.
# (scores.csv also has "* B" / "* neutral" / affluence / edginess / sociality / time
# columns; per 02_DECISIONS_LOG.md only ideology/gender/age are used from embeddings.)
# Sign convention: partisan negative=left/positive=right; gender negative=men-leaning/
# positive=women-leaning; age negative=younger-skewing/positive=older-skewing.
SCORE_DIMENSIONS = {
    "partisan": "ideology_score",
    "gender": "gender_score",
    "age": "age_score",
}


def load_subreddit_scores(scores_path: Path) -> dict[str, dict[str, float]]:
    """Return {output_col: {community: score}}, one lookup dict per dimension."""
    scores_df = pd.read_csv(scores_path)
    scores_df["community"] = scores_df["community"].str.lower()
    return {
        output_col: dict(zip(scores_df["community"], scores_df[source_col]))
        for source_col, output_col in SCORE_DIMENSIONS.items()
    }


def load_user_activity(activity_jsonl_path: Path) -> dict[str, dict[str, int]]:
    """Return {author: {subreddit: post_count}}, one line per user record."""
    with open(activity_jsonl_path) as f:
        records = [json.loads(line) for line in f]
    return {
        record["author"].lower(): {
            subreddit.lower(): counts["total"]
            for subreddit, counts in record["subreddit_activity"].items()
        }
        for record in records
    }


def weighted_score(user_activity: dict[str, int], score_lookup: dict[str, float]) -> float | None:
    """Activity-weighted average of subreddit scores; None if no scored subreddit overlaps."""
    total_weight = 0
    weighted_sum = 0.0
    for subreddit, freq in user_activity.items():
        score = score_lookup.get(subreddit)
        if score is None:
            continue
        weighted_sum += freq * score
        total_weight += freq
    return weighted_sum / total_weight if total_weight > 0 else None


def score_users(
    user_activity: dict[str, dict[str, int]],
    scores_by_dim: dict[str, dict[str, float]],
) -> pd.DataFrame:
    any_dim_lookup = next(iter(scores_by_dim.values()))
    rows = []
    for author, activity in user_activity.items():
        row = {"author": author}
        for output_col, lookup in scores_by_dim.items():
            row[output_col] = weighted_score(activity, lookup)
        row["n_subreddits_total"] = len(activity)
        row["n_subreddits_scored"] = sum(1 for s in activity if s in any_dim_lookup)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subreddit-scores", type=Path, required=True,
        help="Waller & Anderson subreddit scores CSV (columns: community, partisan, gender, age, ...)",
    )
    parser.add_argument(
        "--user-activity", type=Path, required=True,
        help="Per-user subreddit activity JSONL: one record per line, "
             '{"author": ..., "subreddit_activity": {sub: {"total": n}, ...}}',
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV: author, ideology_score, gender_score, age_score, coverage columns",
    )
    args = parser.parse_args()

    scores_by_dim = load_subreddit_scores(args.subreddit_scores)
    user_activity = load_user_activity(args.user_activity)

    df = score_users(user_activity, scores_by_dim)

    n_total = len(df)
    n_valid = df["ideology_score"].notna().sum()
    print(
        f"Scored {n_total:,} users; {n_valid:,} ({n_valid / n_total:.1%}) have at least "
        "one subreddit covered by the Waller & Anderson embedding."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
