"""
Generate a FAKE agents table + post history in the exact input format expected by
src/stage2_agents/02_build_personas.py, so the whole pipeline can be exercised end-to-end
before the real Stage-1 sample (1,000 Reddit users + demographics) is delivered.

    python src/tools/make_synthetic_agents.py --n 1000 --out-dir data/interim/final_sample --seed 1

Writes  <out-dir>/agents_synthetic.csv   and  <out-dir>/posts_synthetic.jsonl
(never use these for a real submission — the demographics/posts are random placeholders).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import codebook as cb  # noqa: E402

SUBS = ["AskReddit", "personalfinance", "news", "politics", "conservative", "gardening", "cars", "Parenting",
        "wallstreetbets", "climate", "collapse", "gaming", "fitness", "Cooking", "homeowners", "Christianity",
        "college", "teachers", "nursing", "trucking", "hunting", "solar", "environment", "science"]
SNIPPETS = [
    "Honestly the traffic on I-35 this morning was unreal, took me 50 minutes to get to work.",
    "Anyone else's electricity bill double this summer? AC running non-stop.",
    "We finally got the garden going, tomatoes are doing great this year despite the heat.",
    "I don't trust anything the government says at this point, both parties are the same.",
    "The kids' school is doing a fundraiser again, third one this semester lol.",
    "Just bought my first house, the inspection found the roof needs work. Any recommendations?",
    "Been reading a lot about the wildfire smoke lately, my asthma is killing me.",
    "Started biking to work, saving a ton on gas and it's actually kind of fun.",
    "My grandmother always said the winters used to be worse. Not sure I believe it.",
    "If you haven't tried the new brisket place on Main St you're missing out.",
    "Property taxes went up AGAIN. How is anyone supposed to afford this.",
    "Church potluck this Sunday, bring a dish if you can!",
    "The flooding on our street last week was the worst I've seen in 20 years living here.",
    "Solar panels paid for themselves in about 7 years for us, would recommend.",
    "Not sure who to vote for this time, both candidates seem weak on the economy.",
    "The university tuition hikes are out of control, my daughter is drowning in loans.",
    "Anyone know a good mechanic in the area? Dealership quoted me $2k for brakes.",
    "Scientists keep changing their minds, first eggs are bad then they're good. Hard to keep up.",
    "Watched the NASA launch with my son last night, he was so excited.",
    "Beef prices are insane. We've been eating a lot more chicken and beans.",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim/final_sample"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--posts-per-user", type=int, default=25)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import csv
    states = list(cb.STATE_CASE.keys())
    with open(args.out_dir / "agents_synthetic.csv", "w", newline="", encoding="utf-8") as f, \
            open(args.out_dir / "posts_synthetic.jsonl", "w", encoding="utf-8") as g:
        w = csv.writer(f)
        w.writerow(["profile_id", "author", "gender", "age_band", "race", "education", "income", "party",
                    "assigned_state", "year_birth"])
        for i in range(1, args.n + 1):
            pid = f"p{i:05d}"
            author = f"synthetic_user_{i:05d}"
            band = rng.choices(cb.MODERATORS["age_band"], weights=[22, 26, 24, 28])[0]
            yb = {"18-29": rng.randint(1997, 2008), "30-44": rng.randint(1982, 1996),
                  "45-59": rng.randint(1967, 1981), "60+": rng.randint(1940, 1966)}[band]
            state = rng.choice(states) if rng.random() < 0.9 else ""
            w.writerow([
                pid, author,
                rng.choices(cb.MODERATORS["gender"], weights=[48, 50, 2])[0],
                band,
                rng.choices(cb.MODERATORS["race"], weights=[60, 13, 18, 6, 3])[0],
                rng.choices(cb.MODERATORS["education"], weights=[8, 27, 28, 22, 12, 3])[0],
                rng.choices(cb.MODERATORS["income"], weights=[22, 22, 28, 18, 10])[0],
                rng.choices(cb.MODERATORS["party"], weights=[28, 31, 34, 7])[0],
                state, yb,
            ])
            home_subs = rng.sample(SUBS, k=rng.randint(3, 8))
            if state:
                home_subs.append(state.replace(" ", "").replace(",", "").replace(".", ""))
            posts = []
            t = 1_780_000_000  # ~mid 2026
            for j in range(args.posts_per_user):
                t -= rng.randint(3600, 5 * 86400)
                kind = "submission" if rng.random() < 0.15 else "comment"
                sub = rng.choice(home_subs)
                body = rng.choice(SNIPPETS)
                p = {"type": kind, "subreddit": sub, "created_utc": t, "score": rng.randint(-2, 40)}
                if kind == "submission":
                    p["title"] = body[:60]
                    p["body"] = body if rng.random() < 0.5 else ""
                else:
                    p["body"] = body
                posts.append(p)
            activity = {}
            for p in posts:
                activity.setdefault(p["subreddit"].lower(), {"total": 0})["total"] += 1
            g.write(json.dumps({"author": author, "posts": posts, "subreddit_activity": activity}) + "\n")
    print(f"wrote {args.n} synthetic agents -> {args.out_dir}/agents_synthetic.csv and posts_synthetic.jsonl")


if __name__ == "__main__":
    main()
