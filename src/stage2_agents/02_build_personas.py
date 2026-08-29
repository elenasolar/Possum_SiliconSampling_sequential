"""
Pipeline position: Stage 2, step 02 (last Stage-1/2 step) -- runs after
stage1_sampling/03_assign_state.py's --merge-into call (for agents.csv) and
01_build_persona_posts.py (for posts.jsonl, built from meaningful comment-reply
interactions rather than a verbatim post/comment dump -- see that script's
docstring). See claude_project_plan/07_REPO_STRUCTURE.md §1.

Stage 2 — build agent personas ("moulds") from the final stratified sample.

Inputs
------
1. Agents table (CSV or JSONL), one row per synthetic respondent. Required columns:
       profile_id, gender, age_band, race, education, income, party
   Optional columns:
       author            Reddit username (used to join the post history; defaults to profile_id)
       assigned_state    US state (any spelling/abbr) for intervention #16; blank -> Case 4 fallback
       year_birth | age  if present, used for the raw export's year_birth column
       account_created_utc, n_comments_12mo, ...  (any other columns are kept under "extra")

   The six moderators MUST already carry the exact submission strings (see 06_CODEBOOK.md §1);
   this script validates them and fails loudly otherwise.

2. Post history, either
   a) JSONL, one record per user:
        {"author": "...", "posts": [{"type": "comment"|"submission", "subreddit": "...",
                                     "created_utc": 1712345678, "title": "...", "body": "...",
                                     "score": 3}, ...],
         "subreddit_activity": {"texas": {"total": 12}, ...}}     # optional, embed-script format
   b) flat CSV with columns: author (or profile_id), type, subreddit, created_utc, body [, title, score]

Output
------
JSONL (default data/processed/personas/personas.jsonl), one record per agent:
    {"profile_id", "author", "demographics": {...}, "assigned_state", "state_case",
     "year_birth", "profile_text": "<verbalized Reddit activity>", "n_posts_included", "n_posts_total",
     "extra": {...}}

The verbalization is a FIXED TEMPLATE (registration item D.2): no LLM-generated narrative. The
LLM only ever sees (i) explicit demographics and (ii) the user's own words.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import codebook as cb  # noqa: E402
from common.io_utils import read_jsonl, write_jsonl  # noqa: E402

MODERATOR_COLS = ["gender", "age_band", "race", "education", "income", "party"]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_agents(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".json"}:
        df = pd.DataFrame(list(read_jsonl(path)))
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in ["profile_id"] + MODERATOR_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"agents file is missing required columns: {missing}")
    if "author" not in df.columns:
        df["author"] = df["profile_id"]
    df["author"] = df["author"].fillna("").astype(str)
    df.loc[df["author"].str.strip() == "", "author"] = df["profile_id"]
    df["author_key"] = df["author"].str.lower().str.strip()
    # validate moderator levels
    bad = []
    for col in MODERATOR_COLS:
        allowed = set(cb.MODERATORS[col])
        vals = df[col].astype(str).str.strip()
        df[col] = vals
        wrong = sorted(set(vals) - allowed)
        if wrong:
            bad.append(f"{col}: {wrong[:8]}")
    if bad:
        raise SystemExit("agents file has non-canonical moderator values (must be exact submission strings):\n  "
                         + "\n  ".join(bad))
    if df["profile_id"].duplicated().any():
        raise SystemExit("agents file has duplicated profile_id values")
    return df


def load_posts(path: Path | None) -> dict[str, dict]:
    """Return {author_lower: {"posts": [...], "subreddit_activity": {...}|None}}."""
    if path is None:
        return {}
    out: dict[str, dict] = {}
    if path.suffix.lower() in {".jsonl", ".json"}:
        for rec in read_jsonl(path):
            key = str(rec.get("author", rec.get("profile_id", ""))).lower().strip()
            posts = rec.get("posts") or rec.get("comments") or []
            entry = out.setdefault(key, {"posts": [], "subreddit_activity": None})
            entry["posts"].extend(posts)
            if rec.get("subreddit_activity"):
                entry["subreddit_activity"] = rec["subreddit_activity"]
    else:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = str(row.get("author") or row.get("profile_id") or "").lower().strip()
                entry = out.setdefault(key, {"posts": [], "subreddit_activity": None})
                entry["posts"].append(row)
    return out


# --------------------------------------------------------------------------- #
# verbalization (fixed template)
# --------------------------------------------------------------------------- #

def _fmt_date(ts) -> str:
    try:
        ts = float(ts)
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        s = str(ts or "").strip()
        return s[:10] if s else "unknown date"


def _clean_text(s, max_chars: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r", " ").strip()
    if len(s) > max_chars:
        s = s[: max_chars - 3].rstrip() + "..."
    return s


def _is_removed(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"", "[deleted]", "[removed]"}


def verbalize_activity(posts: list[dict], *, max_posts: int, max_chars_per_post: int,
                       max_total_chars: int, subreddit_activity: dict | None = None,
                       most_recent_first: bool = True) -> tuple[str, int]:
    """Render the user's own words as a numbered, dated list. Returns (text, n_included)."""

    def ts(p):
        try:
            return float(p.get("created_utc") or 0)
        except (TypeError, ValueError):
            return 0.0

    usable = []
    for p in posts:
        body = p.get("body") if p.get("body") not in (None, "") else p.get("selftext")
        title = p.get("title")
        if _is_removed(body) and _is_removed(title):
            continue
        usable.append(p)
    usable.sort(key=ts, reverse=most_recent_first)

    # subreddit summary
    if subreddit_activity:
        counts = Counter({k: (v.get("total") if isinstance(v, dict) else v) or 0 for k, v in subreddit_activity.items()})
    else:
        counts = Counter(str(p.get("subreddit") or "unknown") for p in usable)
    top = counts.most_common(15)
    lines = []
    if top:
        lines.append("Most active subreddits (by number of posts and comments): "
                     + ", ".join(f"r/{s} ({n})" for s, n in top))
    lines.append(f"Total posts and comments available: {len(usable)}")
    lines.append("")
    lines.append(f"Recent posts and comments written by this user (most recent first, up to {max_posts} shown):")

    total = 0
    n = 0
    for p in usable:
        if n >= max_posts:
            break
        kind = str(p.get("type") or ("submission" if p.get("title") else "comment")).lower()
        kind = "post" if kind in {"submission", "post", "s"} else "comment"
        sub = p.get("subreddit") or "unknown"
        date = _fmt_date(p.get("created_utc"))
        body = p.get("body") if p.get("body") not in (None, "") else p.get("selftext")
        title = _clean_text(p.get("title"), 300)
        body = _clean_text(body, max_chars_per_post)
        if kind == "post":
            entry = f"[{n + 1}] {date} | post in r/{sub} | title: \"{title}\"" + (f" | text: \"{body}\"" if body and not _is_removed(body) else "")
        else:
            entry = f"[{n + 1}] {date} | comment in r/{sub} | text: \"{body}\""
        if total + len(entry) > max_total_chars and n > 0:
            break
        lines.append(entry)
        total += len(entry)
        n += 1
    if n == 0:
        lines.append("(no post or comment text available for this user)")
    return "\n".join(lines), n


def build_persona(row: dict, hist: dict | None, *, max_posts: int, max_chars_per_post: int,
                  max_total_chars: int) -> dict:
    posts = (hist or {}).get("posts") or []
    text, n_inc = verbalize_activity(
        posts, max_posts=max_posts, max_chars_per_post=max_chars_per_post,
        max_total_chars=max_total_chars, subreddit_activity=(hist or {}).get("subreddit_activity"),
    )
    demo = {c: row[c] for c in MODERATOR_COLS}
    state = cb.normalize_state(row.get("assigned_state"))
    # year of birth for the raw export
    yb = None
    for k in ("year_birth", "birth_year"):
        if row.get(k) not in (None, ""):
            try:
                yb = int(float(row[k]))
            except ValueError:
                yb = None
    if yb is None and row.get("age") not in (None, ""):
        try:
            yb = cb.SURVEY_YEAR - int(float(row["age"]))
        except ValueError:
            yb = None
    if yb is None or cb.age_band_from_year(yb) != demo["age_band"]:
        yb = cb.AGE_BAND_REPRESENTATIVE_BIRTH_YEAR[demo["age_band"]]
    extra = {k: v for k, v in row.items() if k not in set(MODERATOR_COLS) | {"profile_id", "author", "author_key", "assigned_state"} and v not in (None, "")}
    return {
        "profile_id": row["profile_id"],
        "author": row["author"],
        "demographics": demo,
        "assigned_state": state,
        "state_case": cb.state_to_case(state),
        "year_birth": yb,
        "profile_text": text,
        "n_posts_included": n_inc,
        "n_posts_total": len(posts),
        "extra": extra,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=Path, required=True, help="agents CSV/JSONL (profile_id + 6 moderators [+ author, assigned_state, ...])")
    ap.add_argument("--posts", type=Path, default=None, help="post history JSONL (per user) or flat CSV (per post)")
    ap.add_argument("--output", type=Path, default=Path("data/processed/personas/personas.jsonl"))
    ap.add_argument("--max-posts", type=int, default=60)
    ap.add_argument("--max-chars-per-post", type=int, default=600)
    ap.add_argument("--max-total-chars", type=int, default=16000)
    args = ap.parse_args(argv)

    agents = load_agents(args.agents)
    hist = load_posts(args.posts)
    n_no_hist = 0
    personas = []
    for row in agents.to_dict(orient="records"):
        h = hist.get(row["author_key"]) or hist.get(str(row["profile_id"]).lower())
        if h is None:
            n_no_hist += 1
        personas.append(build_persona(row, h, max_posts=args.max_posts,
                                      max_chars_per_post=args.max_chars_per_post,
                                      max_total_chars=args.max_total_chars))
    n = write_jsonl(args.output, personas)
    n_state = sum(1 for p in personas if p["assigned_state"])
    print(f"Wrote {n} personas -> {args.output}")
    print(f"  with post history: {n - n_no_hist}/{n}; with assigned_state: {n_state}/{n} "
          f"(others -> intervention #16 Case 4 fallback)")
    if n_no_hist:
        print(f"  WARNING: {n_no_hist} agents had no post history (author not found in --posts)")
    inc = [p["n_posts_included"] for p in personas]
    if inc:
        print(f"  posts included per persona: min {min(inc)}, median {sorted(inc)[len(inc)//2]}, max {max(inc)}")


if __name__ == "__main__":
    main()
