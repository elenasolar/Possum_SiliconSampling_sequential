"""
Diagnostic: how many meaningful comment-reply interactions does each sampled
agent actually have, once 01_build_persona_posts.py has run? Not a pipeline
step (no numeric prefix, doesn't feed anything downstream) -- read-only
analysis over already-produced output, run whenever a distribution check is
useful.

Context (02_DECISIONS_LOG.md, 2026-08-27): 06_stratified_sample.py draws the
1,000-agent sample purely on demographics, with no visibility into comment-
reply-interaction richness at draw time -- that's only knowable after the
expensive classify+fetch step, which by design only runs on the sampled
1,000 (02_DECISIONS_LOG.md, 2026-08-25), not the whole ~13,000-candidate
cohort. This means some sampled agents can end up with too little persona-
building material after the fact (observed: 2/1,000 with zero interactions
at all). This script surfaces exactly who, and by how much, so a deliberate
decision can be made about backup-sampling those agents rather than
silently proceeding with thin personas.

Input:
    --agents  06_stratified_sample.py's agents.csv (needs author; quota_cell_id
              and profile_id are carried through if present, for a later
              backup-sampling step to use)
    --posts   01_build_persona_posts.py's posts.jsonl -- agents entirely
              absent from this file are counted as 0 interactions, not
              skipped.

Output:
    --output (optional)  one row per SAMPLED agent: profile_id, author,
    quota_cell_id, n_interactions.
    --below-threshold-output (optional)  just the agents with
    n_interactions < --threshold -- single 'author' column, ready to hand
    straight to 00_extract_meaningful_interactions.py's --relaxed-authors
    (per Elena, 2026-08-27: below-threshold agents get the word-count-only
    relaxed rule rather than being backup-sampled, see 02_DECISIONS_LOG.md).
    Console (always): summary stats, a histogram of n_interactions, and
    coverage counts below a few candidate thresholds (<1, <3, <5, <10), to
    help judge where to draw the "needs a backup" line.

Usage:
    python diagnose_interaction_coverage.py \\
        --agents data/interim/final_sample/agents.csv \\
        --posts data/interim/final_sample/posts.jsonl \\
        --output data/interim/final_sample/interaction_coverage.csv \\
        --threshold 10 \\
        --below-threshold-output data/interim/final_sample/relaxed_authors.csv

    python diagnose_interaction_coverage.py --selftest   # no data required
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_interaction_counts(posts_path: Path) -> dict[str, int]:
    """author (lowercased) -> number of posts (interactions) in posts.jsonl."""
    counts: dict[str, int] = {}
    with open(posts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counts[str(rec["author"]).strip().lower()] = len(rec.get("posts") or [])
    return counts


def build_coverage(agents: pd.DataFrame, interaction_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for _, row in agents.iterrows():
        author_key = str(row["author"]).strip().lower()
        rows.append({
            "profile_id": row.get("profile_id", ""),
            "author": row["author"],
            "quota_cell_id": row.get("quota_cell_id", ""),
            "n_interactions": interaction_counts.get(author_key, 0),
        })
    return pd.DataFrame(rows, columns=["profile_id", "author", "quota_cell_id", "n_interactions"])


def print_report(coverage: pd.DataFrame, histogram_cap: int = 20) -> None:
    n = len(coverage)
    counts = coverage["n_interactions"]
    print(f"{n:,} sampled agents.")
    print(f"  mean={counts.mean():.2f}  median={counts.median():.0f}  "
          f"min={counts.min()}  max={counts.max()}")

    print(f"\nDistribution (n_interactions -> agent count, capped display at {histogram_cap}):")
    dist = counts.value_counts().sort_index()
    shown = dist[dist.index <= histogram_cap]
    for k, v in shown.items():
        bar = "#" * min(v, 60)
        print(f"  {k:>4} {v:>5}  {bar}")
    tail_n = int(dist[dist.index > histogram_cap].sum())
    if tail_n:
        print(f"  >{histogram_cap:<3} {tail_n:>5}  (not itemized)")

    print("\nCoverage at candidate thresholds:")
    for threshold in (1, 3, 5, 10):
        below = int((counts < threshold).sum())
        print(f"  <{threshold:>2} interactions: {below:>4} agents ({below / n:.1%})")


def selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        agents_path = tmp / "agents.csv"
        pd.DataFrame([
            {"profile_id": "agent_0001", "author": "Alice", "quota_cell_id": "Female|30-44|White / Caucasian"},
            {"profile_id": "agent_0002", "author": "bob", "quota_cell_id": "Male|18-29|White / Caucasian"},
            {"profile_id": "agent_0003", "author": "carol", "quota_cell_id": "Female|60+|Black / African American"},
        ]).to_csv(agents_path, index=False)

        posts_path = tmp / "posts.jsonl"
        with open(posts_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"author": "alice", "posts": [{"body": "x"}, {"body": "y"}]}) + "\n")
            f.write(json.dumps({"author": "bob", "posts": [{"body": "x"}]}) + "\n")
            # carol absent entirely -- must show up as 0, not be skipped

        agents = pd.read_csv(agents_path, dtype=str, keep_default_na=False)
        interaction_counts = load_interaction_counts(posts_path)
        coverage = build_coverage(agents, interaction_counts)

        by_author = coverage.set_index("author")
        assert by_author.loc["Alice", "n_interactions"] == 2  # case-insensitive match to "alice"
        assert by_author.loc["bob", "n_interactions"] == 1
        assert by_author.loc["carol", "n_interactions"] == 0
        assert by_author.loc["carol", "quota_cell_id"] == "Female|60+|Black / African American"
        assert len(coverage) == 3  # every sampled agent present, none dropped

        # --below-threshold-output: threshold=10 -> alice(2) and carol(0) are
        # below, bob(1) is too -- all three qualify at this threshold, but the
        # column subset (author only) and the < (not <=) comparison are what matter
        below = coverage[coverage["n_interactions"] < 10][["author"]]
        assert set(below["author"]) == {"Alice", "bob", "carol"}
        below_strict = coverage[coverage["n_interactions"] < 1][["author"]]
        assert set(below_strict["author"]) == {"carol"}  # only the zero-interaction agent

    print("Self-test passed: interaction counts correctly loaded (case-insensitive "
          "author matching), zero-interaction agents (absent from posts.jsonl) "
          "correctly counted as 0 rather than skipped, quota_cell_id carried "
          "through for a later backup-sampling step, and the below-threshold "
          "filter (strict < comparison) -- all without real data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=Path, help="06_stratified_sample.py's agents.csv")
    ap.add_argument("--posts", type=Path, help="01_build_persona_posts.py's posts.jsonl")
    ap.add_argument("--output", type=Path, default=None,
                     help="Optional per-agent CSV: profile_id, author, quota_cell_id, n_interactions")
    ap.add_argument("--threshold", type=int, default=10,
                     help="Used only with --below-threshold-output: agents with n_interactions below this "
                          "are written out (default 10, matching the sampling target)")
    ap.add_argument("--below-threshold-output", type=Path, default=None,
                     help="Optional CSV, single 'author' column, of just the agents with n_interactions < "
                          "--threshold -- feed straight into 00_extract_meaningful_interactions.py's "
                          "--relaxed-authors")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.agents or not args.posts:
        ap.error("--agents and --posts are required unless --selftest is given")

    agents = pd.read_csv(args.agents, dtype=str, keep_default_na=False)
    interaction_counts = load_interaction_counts(args.posts)
    coverage = build_coverage(agents, interaction_counts)
    print_report(coverage)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(args.output, index=False)
        print(f"\nSaved per-agent coverage -> {args.output}")

    if args.below_threshold_output:
        below = coverage[coverage["n_interactions"] < args.threshold][["author"]]
        args.below_threshold_output.parent.mkdir(parents=True, exist_ok=True)
        below.to_csv(args.below_threshold_output, index=False)
        print(f"\n{len(below):,} agent(s) with n_interactions < {args.threshold} -> "
              f"{args.below_threshold_output}")


if __name__ == "__main__":
    main()
