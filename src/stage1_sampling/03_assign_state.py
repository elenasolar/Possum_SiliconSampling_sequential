"""
Pipeline position: Stage 1, step 02 -- runs after 02_compute_subreddit_activity.py.
Called TWICE in the real pipeline: once early (no --merge-into) to produce a
standalone per-author lookup, and again after 06_stratified_sample.py (with
--merge-into) to backfill assigned_state into the drawn 1,000-agent sample.
See claude_project_plan/07_REPO_STRUCTURE.md §1 for the full execution order.

Assign each author's US state from their Reddit activity, for intervention
#16's state-contingent branching (06_CODEBOOK.md Section 6).

Definition (per Elena, 2026-08-23): the state where the author posted the
most, aggregated across every subreddit tied to that state -- not just their
single highest-count subreddit. A user active in both r/Texas and r/houston
should have those two subreddits' counts summed under "texas" before picking
a winner, since both are the same state signal.

Inputs:
    --subreddit-counts   02_compute_subreddit_activity.py's flat CSV output
                          (author, subreddit, total)
    --state-subreddits   data/external/reddit/state_subreddits_v01.csv --
                          semicolon-delimited (state, subreddit, members,
                          weekly_user_count, since, url); subreddit values
                          are "r/Name"-prefixed, mixed case.

Output: one row per author with any matched state-subreddit activity:
    author, assigned_state, state_post_count (posts in the winning state's
    subreddits), n_states_matched (how many distinct states this author had
    ANY activity in -- 1 means an unambiguous signal, >1 means the winner
    beat out others), tied (true if the top count was a tie, broken here by
    alphabetically-first state name for determinism).

Authors with zero activity in any listed state subreddit are simply absent
from the output -- normalize_state()/state_to_case() (src/common/codebook.py)
already treat a missing/blank assigned_state as Case 4 (fallback), matching
the study's own "Prefer not to say" handling.

KNOWN GAP: data/external/reddit/state_subreddits_v01.csv currently lists
subreddits for all 50 states but NONE for Washington, D.C. -- so no author
can ever be assigned to D.C. here, even though it's a real Case 1 (flood)
state in the codebook. Not fixed here (curating which subreddit(s) represent
D.C., e.g. r/washingtondc, is a call for whoever maintains that list, same
as the race-affinity-subreddit caveat in 01_PROJECT_PLAN.md) -- this script
just prints a warning so the gap isn't silently invisible.

Usage:
    python 03_assign_state.py \\
        --subreddit-counts user_subreddit_counts.csv \\
        --state-subreddits data/external/reddit/state_subreddits_v01.csv \\
        --output assigned_states.csv

    # backfill a blank assigned_state column into an already-built agents.csv
    # (e.g. one 06_stratified_sample.py already produced, run before this existed):
    python 03_assign_state.py \\
        --subreddit-counts user_subreddit_counts.csv \\
        --state-subreddits data/external/reddit/state_subreddits_v01.csv \\
        --output assigned_states.csv \\
        --merge-into data/interim/final_sample/agents.csv

    python 03_assign_state.py --selftest   # no data required
"""

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

US_STATE_COUNT = 50


def load_state_subreddit_map(path: Path) -> dict[str, str]:
    """Returns {normalized_subreddit_name: state}, e.g. {"alabama": "alabama", "houston": "texas"}."""
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    df["subreddit_norm"] = (
        df["subreddit"].str.strip().str.lower().str.replace("r/", "", regex=False)
    )
    dup = df.groupby("subreddit_norm")["state"].nunique()
    ambiguous = sorted(dup[dup > 1].index)
    if ambiguous:
        print(f"WARNING: {len(ambiguous)} subreddit(s) map to more than one state in "
              f"{path} -- ambiguous, dropped from the lookup: {ambiguous}")

    n_states = df["state"].str.strip().str.lower().nunique()
    if n_states < US_STATE_COUNT + 1:
        missing_note = "" if n_states >= US_STATE_COUNT else " (fewer than all 50 states, too)"
        print(f"WARNING: {path} covers {n_states} state(s), not the 50 states + D.C. -- "
              f"no author can be assigned to a missing state/territory through this script"
              f"{missing_note}. Known gap as of 2026-08-23: Washington, D.C. has no subreddit "
              f"listed here at all.")

    return {
        row.subreddit_norm: row.state.strip().lower()
        for row in df.itertuples()
        if row.subreddit_norm not in ambiguous
    }


def compute_assigned_state(counts: pd.DataFrame, subreddit_to_state: dict[str, str]) -> pd.DataFrame:
    """counts: DataFrame with 'author', 'subreddit', 'total' (as produced by
    02_compute_subreddit_activity.py). Returns one row per author with any
    matched state-subreddit activity -- authors with none are simply absent."""
    df = counts.copy()
    df["author"] = df["author"].str.strip().str.lower()
    df["subreddit"] = df["subreddit"].str.strip().str.lower().str.replace("r/", "", regex=False)
    df["state"] = df["subreddit"].map(subreddit_to_state)
    matched = df.dropna(subset=["state"])

    by_state = matched.groupby(["author", "state"], as_index=False)["total"].sum()

    rows = []
    for author, group in by_state.groupby("author"):
        group = group.sort_values(["total", "state"], ascending=[False, True])  # tie-break: alphabetical
        top = group.iloc[0]
        tied = (group["total"] == top["total"]).sum() > 1
        rows.append({
            "author": author,
            "assigned_state": top["state"],
            "state_post_count": int(top["total"]),
            "n_states_matched": group["state"].nunique(),
            "tied": tied,
        })
    return pd.DataFrame(rows, columns=["author", "assigned_state", "state_post_count", "n_states_matched", "tied"])


def merge_into_agents(agents_path: Path, assigned: pd.DataFrame, output_path: Path) -> None:
    agents = pd.read_csv(agents_path, dtype=str, keep_default_na=False)
    if "author" not in agents.columns:
        raise ValueError(f"{agents_path}: no 'author' column to join --merge-into on")
    agents["author"] = agents["author"].str.strip().str.lower()

    n_before = int((agents["assigned_state"] != "").sum()) if "assigned_state" in agents.columns else 0
    lookup = assigned.set_index("author")["assigned_state"].to_dict()
    agents["assigned_state"] = agents["author"].map(lookup).fillna(agents.get("assigned_state", ""))
    agents["assigned_state"] = agents["assigned_state"].fillna("")
    n_after = int((agents["assigned_state"] != "").sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agents.to_csv(output_path, index=False)
    print(f"\nMerged into {agents_path}: {n_after}/{len(agents)} agents now have an assigned_state "
          f"(was {n_before}/{len(agents)} before) -> {output_path}")


def selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        state_path = tmp / "state_subreddits.csv"
        with open(state_path, "w", encoding="utf-8") as f:
            f.write("state;subreddit;members;weekly_user_count;since;url;\n")
            f.write("texas;r/Texas;;1;01.01.2010;https://x;\n")
            f.write("texas;r/houston;;1;01.01.2010;https://x;\n")
            f.write("california;r/California;;1;01.01.2010;https://x;\n")

        subreddit_to_state = load_state_subreddit_map(state_path)
        assert subreddit_to_state == {"texas": "texas", "houston": "texas", "california": "california"}

        counts = pd.DataFrame([
            # user_a: texas wins on AGGREGATE (3+2=5) even though california alone (4) beats either single texas sub
            {"author": "user_a", "subreddit": "texas", "total": 3},
            {"author": "user_a", "subreddit": "houston", "total": 2},
            {"author": "user_a", "subreddit": "california", "total": 4},
            # user_b: only unrecognized subreddits -> absent from output entirely
            {"author": "user_b", "subreddit": "askreddit", "total": 100},
            # user_c: exact tie between texas and california -> alphabetical tie-break (california)
            {"author": "user_c", "subreddit": "texas", "total": 5},
            {"author": "user_c", "subreddit": "california", "total": 5},
        ])

        assigned = compute_assigned_state(counts, subreddit_to_state)
        by_author = assigned.set_index("author").to_dict("index")

        assert by_author["user_a"]["assigned_state"] == "texas", by_author["user_a"]
        assert by_author["user_a"]["state_post_count"] == 5
        assert by_author["user_a"]["n_states_matched"] == 2
        assert by_author["user_a"]["tied"] == False

        assert "user_b" not in by_author, "user_b has no state-subreddit activity, should be absent"

        assert by_author["user_c"]["assigned_state"] == "california"  # alphabetically first on a tie
        assert by_author["user_c"]["tied"] == True

        # --merge-into: backfills a blank agents.csv column, keeps other agents unchanged
        agents_path = tmp / "agents.csv"
        pd.DataFrame([
            {"profile_id": "agent_0001", "author": "User_A", "gender": "Male", "assigned_state": ""},
            {"profile_id": "agent_0002", "author": "user_b", "gender": "Female", "assigned_state": ""},
            {"profile_id": "agent_0003", "author": "user_d", "gender": "Other", "assigned_state": "Nevada"},
        ]).to_csv(agents_path, index=False)

        merged_path = tmp / "agents_merged.csv"
        merge_into_agents(agents_path, assigned, merged_path)
        merged = pd.read_csv(merged_path, dtype=str, keep_default_na=False)
        merged_by_author = merged.set_index("author").to_dict("index")
        assert merged_by_author["user_a"]["assigned_state"] == "texas"
        assert merged_by_author["user_b"]["assigned_state"] == "", "no match -- stays blank, not dropped"
        assert merged_by_author["user_d"]["assigned_state"] == "Nevada", "untouched when already present and no match overrides it"

    print("Self-test passed: state-subreddit lookup loading (incl. ambiguous-mapping "
          "detection), aggregate-by-state author assignment (not single-subreddit), "
          "alphabetical tie-breaking, and --merge-into backfill all work end to end "
          "without real data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subreddit-counts", type=Path, help="02_compute_subreddit_activity.py's flat CSV (author, subreddit, total)")
    ap.add_argument("--state-subreddits", type=Path, help="data/external/reddit/state_subreddits_v01.csv")
    ap.add_argument("--output", type=Path, help="Output CSV: author, assigned_state, state_post_count, n_states_matched, tied")
    ap.add_argument("--merge-into", type=Path, default=None,
                     help="Existing agents.csv to backfill 'assigned_state' into (joined on 'author'); "
                          "an author already present in --merge-into but absent from this script's result "
                          "keeps whatever assigned_state it already had, rather than being blanked out.")
    ap.add_argument("--merged-output", type=Path, default=None,
                     help="Where to write the merged file (default: overwrite --merge-into in place)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.subreddit_counts or not args.state_subreddits or not args.output:
        ap.error("--subreddit-counts, --state-subreddits, and --output are required unless --selftest is given")

    subreddit_to_state = load_state_subreddit_map(args.state_subreddits)
    counts = pd.read_csv(args.subreddit_counts, dtype=str, keep_default_na=False)
    counts["total"] = counts["total"].astype(int)

    assigned = compute_assigned_state(counts, subreddit_to_state)
    n_input_authors = counts["author"].str.strip().str.lower().nunique()
    print(f"{n_input_authors:,} authors in {args.subreddit_counts}; "
          f"{len(assigned):,} ({len(assigned) / n_input_authors:.1%}) matched at least one state subreddit.")
    if assigned["tied"].any():
        print(f"  {int(assigned['tied'].sum()):,} had a tie at the top, broken alphabetically.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

    if args.merge_into:
        merge_into_agents(args.merge_into, assigned, args.merged_output or args.merge_into)


if __name__ == "__main__":
    main()
