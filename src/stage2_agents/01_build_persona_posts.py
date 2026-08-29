"""
Pipeline position: Stage 2, step 01 -- runs after BOTH stage1_sampling/
06_stratified_sample.py (the 1,000-agent sample) and stage2_agents/
00_extract_meaningful_interactions.py; feeds 02_build_personas.py. See
claude_project_plan/07_REPO_STRUCTURE.md §1.

Builds each SAMPLED agent's post history for persona construction from their
meaningful comment-reply interactions only, per the 2026-08-24 decision with
David (02_DECISIONS_LOG.md). As of 2026-08-25, 00_extract_meaningful_
interactions.py's input is already scoped to the sampled agents upstream (via
stage1_sampling/07_classify_comment_structure.py's --authors flag), so the
filter this script applies below is now a cheap, defensive re-check rather
than the primary scoping mechanism -- kept because it costs nothing and
documents the intent explicitly. Stratified sampling (who gets selected) and
post-history construction (what those selected people's persona is built
from) are deliberately separate concerns -- this script is the seam between
them.

Each kept interaction becomes one "comment" post. The reply's own text is
kept verbatim; the parent comment it responded to is prepended inline as
bracketed context -- there's no separate "reply-to" field in the posts.jsonl
schema, and that parent context IS the reactive signal this whole
comment-reply-interaction design was chosen for in the first place (per
02_DECISIONS_LOG.md, 2026-08-24: "more behavioral cues on reactions ...
compared to a submission"); dropping it here would silently lose that signal
before it ever reaches 02_build_personas.py.

Input:
    --agents        06_stratified_sample.py's agents.csv (needs an 'author' column)
    --interactions  00_extract_meaningful_interactions.py's output CSV

Output:
    posts.jsonl -- one line per sampled author WITH at least one kept
    interaction: {"author": ..., "posts": [{"type": "comment", "subreddit":
    ..., "created_utc": ..., "title": "", "body": "(Replying to another
    user's comment: \\"...\\") ...", "score": ""}, ...]}. Sampled authors
    with zero kept interactions are reported (never silently dropped or
    silently given an empty entry) -- 02_build_personas.py's own "no post
    history" warning will also flag them independently.

Usage:
    python 01_build_persona_posts.py \\
        --agents data/interim/final_sample/agents_submissions.csv \\
        --interactions data/interim/meaningful_interactions/meaningful_interactions.csv \\
        --output data/interim/final_sample/posts.jsonl

    python 01_build_persona_posts.py --selftest   # no data required
"""

import argparse
import json
from pathlib import Path

import pandas as pd

MAX_PARENT_CHARS = 300


def format_body(row) -> str:
    parent_body = str(row.get("parent_body") or "").strip()
    own_body = str(row.get("body") or "").strip()
    if not parent_body:
        return own_body
    truncated = parent_body if len(parent_body) <= MAX_PARENT_CHARS else parent_body[:MAX_PARENT_CHARS - 3] + "..."
    return f"(Replying to another user's comment: \"{truncated}\") {own_body}"


def build_posts(agents: pd.DataFrame, interactions: pd.DataFrame) -> tuple[dict[str, list[dict]], int]:
    """Returns ({author_lower: [post, ...]}, n_sampled_agents_with_zero_interactions)."""
    sampled_authors = set(agents["author"].astype(str).str.strip().str.lower())

    interactions = interactions.copy()
    interactions["author_lower"] = interactions["author"].astype(str).str.strip().str.lower()
    matched = interactions[interactions["author_lower"].isin(sampled_authors)]

    posts_by_author: dict[str, list[dict]] = {}
    for _, row in matched.iterrows():
        post = {
            "type": "comment",
            "subreddit": row.get("subreddit", ""),
            "created_utc": row.get("created_utc", ""),
            "title": "",
            "body": format_body(row),
            "score": "",
        }
        posts_by_author.setdefault(row["author_lower"], []).append(post)

    n_zero = sum(1 for a in sampled_authors if a not in posts_by_author)
    return posts_by_author, n_zero


def selftest() -> None:
    agents = pd.DataFrame([
        {"profile_id": "agent_0001", "author": "Alice"},
        {"profile_id": "agent_0002", "author": "bob"},
        {"profile_id": "agent_0003", "author": "carol"},  # no meaningful interactions at all
    ])
    interactions = pd.DataFrame([
        {"author": "alice", "id": "r1", "subreddit": "test", "created_utc": "1",
         "body": "But you're wrong about that.", "parent_body": "The earth is flat.",
         "parent_author": "someoutsider", "word_count": 6, "matched_cues": "second_person;disagreement"},
        {"author": "bob", "id": "r2", "subreddit": "test", "created_utc": "2",
         "body": "When did this happen exactly?", "parent_body": "",
         "parent_author": "", "word_count": 5, "matched_cues": "question"},
        # "dave" is NOT in agents.csv -- must be excluded even though it's a valid interaction
        {"author": "dave", "id": "r3", "subreddit": "test", "created_utc": "3",
         "body": "Obviously that is not correct.", "parent_body": "x", "parent_author": "y",
         "word_count": 5, "matched_cues": "hedge"},
    ])

    posts_by_author, n_zero = build_posts(agents, interactions)

    assert set(posts_by_author.keys()) == {"alice", "bob"}, posts_by_author.keys()
    assert n_zero == 1, n_zero  # carol

    alice_post = posts_by_author["alice"][0]
    assert alice_post["body"] == "(Replying to another user's comment: \"The earth is flat.\") But you're wrong about that.", alice_post["body"]
    assert alice_post["type"] == "comment"

    bob_post = posts_by_author["bob"][0]
    assert bob_post["body"] == "When did this happen exactly?"  # no parent_body -> no bracketed prefix

    print("Self-test passed: filtering interactions down to sampled agents only "
          "(excluding a valid interaction from a non-sampled author), parent-context "
          "prefixing (and its correct absence when there's no parent text), and "
          "zero-interaction-agent reporting all work end to end without real data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=Path, help="06_stratified_sample.py's agents.csv (needs an 'author' column)")
    ap.add_argument("--interactions", type=Path, help="00_extract_meaningful_interactions.py's output CSV")
    ap.add_argument("--output", type=Path, help="Output posts.jsonl path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.agents or not args.interactions or not args.output:
        ap.error("--agents, --interactions, and --output are required unless --selftest is given")

    agents = pd.read_csv(args.agents, dtype=str, keep_default_na=False)
    if "author" not in agents.columns:
        raise ValueError(f"{args.agents}: no 'author' column")
    interactions = pd.read_csv(args.interactions, dtype=str, keep_default_na=False)

    posts_by_author, n_zero = build_posts(agents, interactions)
    n_sampled = agents["author"].astype(str).str.strip().str.lower().nunique()
    print(f"{n_sampled:,} sampled agents; {len(posts_by_author):,} have >=1 meaningful interaction "
          f"({n_zero:,} have none -- will show up as 'no post history' in 02_build_personas.py).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for author, posts in sorted(posts_by_author.items()):
            f.write(json.dumps({"author": author, "posts": posts}, ensure_ascii=False) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
