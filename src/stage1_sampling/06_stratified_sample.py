"""
Pipeline position: Stage 1, step 06 (last step) -- runs after 05_infer_demographics_llm.py;
its agents.csv output then goes back through 03_assign_state.py (--merge-into) to fill
assigned_state before Stage 2. See claude_project_plan/07_REPO_STRUCTURE.md §1.

Stratified draw of the final ~1,000-agent sample from the ~10,000-candidate pool
(05_DATA_REQUIREMENTS.md Section 3 -> Section 4).

Hard quotas on two joint distributions only, per 02_DECISIONS_LOG.md's
census-matching design decision: age_band x gender, and gender x race. These
are matched against the parent megastudy's own recruitment quota shares
(counts below, rescaled to N~18,000 in the source table -- the megastudy's
real target is N~22,000, but only the shares matter here), not an independent
census pull -- though both ultimately derive from the same source (2024 U.S.
Census Bureau Population Estimates Program), which is why they're numerically
close to the separate ACS/PUMS pull in data/external/census/. Education/
income/party stay marginal-only (validated as a diagnostic, not enforced as
a quota here).

Since age_band and race are each only jointly targeted with gender (not with
each other), the two 2-way tables are combined into one 3-way
(gender, age_band, race) target via the independence assumption within each
gender -- i.e. P(age, race | gender) = P(age | gender) * P(race | gender).
This is a one-step raking/IPF construction: aggregating the resulting 3-way
table back down reproduces both input margins exactly (up to rounding).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.codebook import AGE_BAND, GENDER, RACE  # noqa: E402

# --------------------------------------------------------------------------- #
# Study target distributions (age x gender, race x gender)
# --------------------------------------------------------------------------- #
# Source: "Table 3: Sampling quota targets (N ~ 18,000)" from the parent
# megastudy's own materials -- table caption: derived from the 2024 U.S. Census
# Bureau Population Estimates Program; proportions match the parent megastudy's
# actual recruitment quotas, with counts rescaled here to N~18,000 (the
# megastudy's own target is N~22,000 -- only the shares matter for our use,
# not the raw counts). This is why these shares are numerically close to the
# independent ACS/PUMS pull in data/external/census/acs_pums_crosstab_*.csv
# (e.g. age 60+: 30.9% here vs. 31.0% ACS) -- same ultimate source (Census
# Bureau population estimates), not a coincidence. Confirmed by Elena, 2026-08-20.

STUDY_AGE_GENDER_COUNTS: dict[tuple[str, str], int] = {
    ("18-29", "Male"): 1848, ("18-29", "Female"): 1781,
    ("30-44", "Male"): 2365, ("30-44", "Female"): 2323,
    ("45-59", "Male"): 2048, ("45-59", "Female"): 2074,
    ("60+", "Male"): 2566, ("60+", "Female"): 2995,
}

STUDY_RACE_GENDER_COUNTS: dict[tuple[str, str], int] = {
    ("Asian / Asian American", "Male"): 568, ("Asian / Asian American", "Female"): 633,
    ("Black / African American", "Male"): 1042, ("Black / African American", "Female"): 1170,
    ("Hispanic / Latino", "Male"): 1646, ("Hispanic / Latino", "Female"): 1617,
    ("Other", "Male"): 240, ("Other", "Female"): 252,
    ("White / Caucasian", "Male"): 5332, ("White / Caucasian", "Female"): 5500,
}

# The study table has no "Other" gender column/row. Candidates whose inferred
# gender is "Other" have no quota cell to fill under this design and are
# excluded from the hard-quota draw (counted and reported, not silently
# dropped) -- confirm with the team whether the real study truly had zero
# non-binary respondents or just didn't report them separately.
GENDER_WITH_TARGET: list[str] = ["Male", "Female"]

assert set(GENDER_WITH_TARGET) <= set(GENDER)
assert {a for a, _ in STUDY_AGE_GENDER_COUNTS} <= set(AGE_BAND)
assert {r for r, _ in STUDY_RACE_GENDER_COUNTS} <= set(RACE)


# --------------------------------------------------------------------------- #
# Apportionment
# --------------------------------------------------------------------------- #

def largest_remainder_apportion(weights: dict, total: int) -> dict:
    """Round `weights` (proportional to `total`) to integers summing exactly to `total`."""
    keys = list(weights.keys())
    wsum = sum(weights.values())
    if total <= 0 or wsum <= 0:
        return {k: 0 for k in keys}
    exact = {k: weights[k] / wsum * total for k in keys}
    floors = {k: int(exact[k]) for k in keys}
    remainder = total - sum(floors.values())
    order = sorted(keys, key=lambda k: exact[k] - floors[k], reverse=True)
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def build_cell_targets(n_total: int) -> dict[tuple[str, str, str], int]:
    """Target count per (gender, age_band, race) cell, summing exactly to n_total."""
    gender_total_from_age = defaultdict(float)
    for (_a, g), c in STUDY_AGE_GENDER_COUNTS.items():
        gender_total_from_age[g] += c
    gender_total_from_race = defaultdict(float)
    for (_r, g), c in STUDY_RACE_GENDER_COUNTS.items():
        gender_total_from_race[g] += c

    # Reconcile the two tables' implied gender marginals (they differ by a
    # handful of people out of 18,000 -- rounding noise in the source table).
    canonical_gender_weight = {
        g: (gender_total_from_age[g] + gender_total_from_race[g]) / 2
        for g in GENDER_WITH_TARGET
    }
    gender_targets = largest_remainder_apportion(canonical_gender_weight, n_total)

    cell_targets: dict[tuple[str, str, str], int] = {}
    for g in GENDER_WITH_TARGET:
        age_share = {
            a: STUDY_AGE_GENDER_COUNTS.get((a, g), 0) / gender_total_from_age[g]
            for a in AGE_BAND
        }
        race_share = {
            r: STUDY_RACE_GENDER_COUNTS.get((r, g), 0) / gender_total_from_race[g]
            for r in RACE
        }
        weight = {(a, r): age_share[a] * race_share[r] for a in AGE_BAND for r in RACE}
        sub_targets = largest_remainder_apportion(weight, gender_targets[g])
        for (a, r), cnt in sub_targets.items():
            cell_targets[(g, a, r)] = cnt
    return cell_targets


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def load_pool(csv_path: str) -> tuple[list[dict], int]:
    """Read the candidate pool; drop rows with missing/invalid gender, age_band, or race."""
    rows = []
    dropped = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("gender") in GENDER and row.get("age_band") in AGE_BAND and row.get("race") in RACE:
                rows.append(row)
            else:
                dropped += 1
    return rows, dropped


def allocate(pool: list[dict], cell_targets: dict, rng: random.Random):
    """First pass: fill each cell up to its target from available candidates."""
    by_cell = defaultdict(list)
    excluded_other_gender = 0
    for row in pool:
        g = row["gender"]
        if g not in GENDER_WITH_TARGET:
            excluded_other_gender += 1
            continue
        by_cell[(g, row["age_band"], row["race"])].append(row)

    selected: list[dict] = []
    leftover: dict[tuple, list[dict]] = {}
    shortfall: dict[tuple, int] = {}
    for cell, target in cell_targets.items():
        avail = by_cell.get(cell, [])
        rng.shuffle(avail)
        selected.extend(avail[:target])
        leftover[cell] = avail[target:]
        if len(avail) < target:
            shortfall[cell] = target - len(avail)
    return selected, leftover, shortfall, excluded_other_gender


def redistribute(selected: list[dict], leftover: dict, shortfall: dict, cell_targets: dict,
                  rng: random.Random, max_rounds: int = 10) -> int:
    """Second pass: fill the total shortfall from cells with surplus candidates,
    weighted by each surplus cell's own target size. Mutates `selected`/`leftover`
    in place. Returns the shortfall that remains unmet even after this."""
    remaining_need = sum(shortfall.values())
    for _ in range(max_rounds):
        if remaining_need <= 0:
            break
        capacity = {cell: len(rows) for cell, rows in leftover.items() if rows}
        if not capacity:
            break
        weight = {cell: cell_targets[cell] for cell in capacity}
        take_n = min(remaining_need, sum(capacity.values()))
        extra_alloc = largest_remainder_apportion(weight, take_n)
        drawn_this_round = 0
        for cell, extra in extra_alloc.items():
            extra = min(extra, len(leftover[cell]))
            if extra <= 0:
                continue
            selected.extend(leftover[cell][:extra])
            leftover[cell] = leftover[cell][extra:]
            drawn_this_round += extra
        remaining_need -= drawn_this_round
        if drawn_this_round == 0:
            break
    return remaining_need


def load_marginal_targets(*csv_paths: str) -> dict[str, dict[str, float]]:
    """Load ACS/CES long-format (variable, category, weighted_share) files into
    {variable: {category: share}}, skipping files that don't exist yet."""
    targets: dict[str, dict[str, float]] = defaultdict(dict)
    for path in csv_paths:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                targets[row["variable"]][row["category"]] = float(row["weighted_share"])
    return targets


def print_marginal_diagnostic(selected: list[dict], targets: dict[str, dict[str, float]]) -> None:
    n = len(selected)
    if n == 0:
        return
    for variable in ("education", "income", "party"):
        cat_targets = targets.get(variable)
        if not cat_targets:
            continue
        counts = defaultdict(int)
        missing = 0
        for row in selected:
            val = row.get(variable, "")
            if val:
                counts[val] += 1
            else:
                missing += 1
        print(f"\n  {variable} (marginal-only, not a hard quota; {missing}/{n} rows blank):")
        for cat, target_share in sorted(cat_targets.items(), key=lambda kv: -kv[1]):
            achieved = counts.get(cat, 0) / n
            print(f"    {cat:45s} target {target_share:6.1%}  achieved {achieved:6.1%}")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

OUTPUT_FIELDNAMES = [
    "profile_id", "author",
    "gender", "age_band", "race", "education", "income", "party",
    "gender_source", "age_source", "race_source", "education_source",
    "income_source", "party_source", "inference_confidence", "assigned_state",
    "sampling_weight", "quota_cell_id",
]


def run_pipeline(input_csv: str, output_csv: str, shortfall_csv: str, n: int, seed: int,
                  supplementary_csv: Optional[str] = None, strict: bool = False,
                  marginal_targets_csvs: tuple[str, ...] = ()) -> None:
    rng = random.Random(seed)

    pool, dropped = load_pool(input_csv)
    print(f"Loaded {len(pool)} candidates ({dropped} dropped: missing/invalid gender, age_band, or race).")

    supplementary_used = 0
    if supplementary_csv:
        supp_pool, supp_dropped = load_pool(supplementary_csv)
        pool.extend(supp_pool)
        supplementary_used = len(supp_pool)
        print(f"Added {supplementary_used} supplementary candidates ({supp_dropped} dropped from that file).")

    cell_targets = build_cell_targets(n)
    selected, leftover, shortfall, excluded_other = allocate(pool, cell_targets, rng)
    if excluded_other:
        print(f"Excluded {excluded_other} candidates with gender 'Other' (no quota cell in the study table).")

    unmet = 0
    if shortfall:
        total_short = sum(shortfall.values())
        print(f"\n{len(shortfall)} cell(s) short by {total_short} total before redistribution:")
        for (g, a, r), n_short in sorted(shortfall.items(), key=lambda kv: -kv[1]):
            print(f"    {g:6s} {a:6s} {r:28s} short by {n_short} (target {cell_targets[(g, a, r)]})")
        unmet = redistribute(selected, leftover, shortfall, cell_targets, rng)
        if unmet:
            print(f"\nWARNING: {unmet} agents still unmet after redistributing surplus cells "
                  f"-- final sample is {n - unmet} short of the requested {n}.")
            print("This is the 60+/sparse-cell scenario -- source additional candidates from "
                  "demographically-targeted subreddits and rerun with --supplementary-pool.")
            if strict:
                raise RuntimeError(f"--strict: {unmet} agents unmet, refusing to write a short sample")
        else:
            print("Fully redistributed -- final sample reaches the requested total.")

    rng.shuffle(selected)  # profile_id order shouldn't leak which cell/pass filled it
    for i, row in enumerate(selected, start=1):
        row["profile_id"] = f"agent_{i:04d}"
        row["quota_cell_id"] = f"{row['gender']}|{row['age_band']}|{row['race']}"
        row["sampling_weight"] = "1.0"

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for row in selected:
            w.writerow(row)
    print(f"\nWrote {len(selected)} agents to {output_csv}")

    Path(shortfall_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(shortfall_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gender", "age_band", "race", "target", "shortfall_before_redistribution"])
        for (g, a, r), target in sorted(cell_targets.items()):
            w.writerow([g, a, r, target, shortfall.get((g, a, r), 0)])
    print(f"Wrote per-cell shortfall report to {shortfall_csv}")

    print("\nAchieved vs. target (age_band x gender):")
    achieved_ag = defaultdict(int)
    for row in selected:
        achieved_ag[(row["age_band"], row["gender"])] += 1
    total_study_ag = sum(STUDY_AGE_GENDER_COUNTS.values())
    for (a, g), c in sorted(STUDY_AGE_GENDER_COUNTS.items()):
        target_share = c / total_study_ag
        achieved_share = achieved_ag.get((a, g), 0) / len(selected) if selected else 0
        print(f"    {a:6s} {g:8s} target {target_share:6.1%}  achieved {achieved_share:6.1%}")

    print("\nAchieved vs. target (race x gender):")
    achieved_rg = defaultdict(int)
    for row in selected:
        achieved_rg[(row["race"], row["gender"])] += 1
    total_study_rg = sum(STUDY_RACE_GENDER_COUNTS.values())
    for (r, g), c in sorted(STUDY_RACE_GENDER_COUNTS.items()):
        target_share = c / total_study_rg
        achieved_share = achieved_rg.get((r, g), 0) / len(selected) if selected else 0
        print(f"    {r:28s} {g:8s} target {target_share:6.1%}  achieved {achieved_share:6.1%}")

    if marginal_targets_csvs:
        targets = load_marginal_targets(*marginal_targets_csvs)
        print_marginal_diagnostic(selected, targets)


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #

def _make_synthetic_pool(rng: random.Random, n: int, skew_60plus_down: bool = True) -> list[dict]:
    """Synthetic candidate pool for testing, with 60+ deliberately underrepresented
    relative to the study target (the exact scenario Elena expects to hit for real)."""
    ages = list(AGE_BAND)
    genders = GENDER_WITH_TARGET
    races = list(RACE)
    weights = [0.4, 0.35, 0.2, 0.05] if skew_60plus_down else [0.25] * 4
    rows = []
    for i in range(n):
        age = rng.choices(ages, weights=weights)[0]
        rows.append({
            "author": f"synth_user_{i}",
            "gender": rng.choice(genders),
            "age_band": age,
            "race": rng.choice(races),
            "education": "", "income": "", "party": "",
        })
    return rows


def selftest() -> None:
    rng = random.Random(0)

    # 1. Apportionment sums exactly to total, even with an awkward weight set.
    weights = {"a": 1, "b": 1, "c": 1}
    result = largest_remainder_apportion(weights, 10)
    assert sum(result.values()) == 10, result

    # 2. Cell targets sum exactly to n and reproduce both input margins.
    n = 1000
    cell_targets = build_cell_targets(n)
    assert sum(cell_targets.values()) == n
    by_age_gender = defaultdict(int)
    by_race_gender = defaultdict(int)
    for (g, a, r), cnt in cell_targets.items():
        by_age_gender[(a, g)] += cnt
        by_race_gender[(r, g)] += cnt
    total_study_ag = sum(STUDY_AGE_GENDER_COUNTS.values())
    for (a, g), c in STUDY_AGE_GENDER_COUNTS.items():
        expected = round(c / total_study_ag * n)
        assert abs(by_age_gender[(a, g)] - expected) <= 2, (a, g, by_age_gender[(a, g)], expected)
    total_study_rg = sum(STUDY_RACE_GENDER_COUNTS.values())
    for (r, g), c in STUDY_RACE_GENDER_COUNTS.items():
        expected = round(c / total_study_rg * n)
        assert abs(by_race_gender[(r, g)] - expected) <= 2, (r, g, by_race_gender[(r, g)], expected)

    # 3. Plenty of supply: allocate should hit every target exactly, no shortfall.
    plenty_pool = _make_synthetic_pool(rng, 20000, skew_60plus_down=False)
    small_targets = build_cell_targets(200)
    selected, leftover, shortfall, excluded = allocate(plenty_pool, small_targets, rng)
    assert not shortfall, shortfall
    assert len(selected) == 200, len(selected)

    # 4. Scarce supply (60+ deliberately underrepresented): shortfall should appear
    #    for at least one 60+ cell, and redistribution should still hit the total.
    scarce_pool = _make_synthetic_pool(rng, 500, skew_60plus_down=True)
    selected2, leftover2, shortfall2, excluded2 = allocate(scarce_pool, small_targets, rng)
    assert any(a == "60+" for (_g, a, _r) in shortfall2), "expected a 60+ shortfall in the scarce-pool test"
    unmet = redistribute(selected2, leftover2, shortfall2, small_targets, rng)
    total_available = sum(1 for row in scarce_pool if row["gender"] in GENDER_WITH_TARGET)
    if total_available >= 200:
        assert unmet == 0, unmet
        assert len(selected2) == 200, len(selected2)
    else:
        assert len(selected2) == total_available, (len(selected2), total_available)

    # 5. "Other" gender candidates are excluded from hard-quota cells, not silently kept.
    other_pool = [{"author": "x", "gender": "Other", "age_band": "18-29", "race": "Other",
                   "education": "", "income": "", "party": ""}]
    _sel, _lo, _sf, excluded_other = allocate(other_pool, small_targets, rng)
    assert excluded_other == 1

    print("SELFTEST PASSED")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Candidate pool CSV (05_DATA_REQUIREMENTS.md Section 3 schema)")
    parser.add_argument("--output", default="data/interim/final_sample/agents.csv")
    parser.add_argument("--shortfall-report", default="data/interim/final_sample/shortfall_report.csv")
    parser.add_argument("--supplementary-pool", default=None,
                         help="Extra candidates (e.g. pulled from demographically-targeted subreddits) "
                              "to help fill sparse cells; same schema as --input")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true",
                         help="Fail instead of writing a short sample if quotas can't be fully met")
    parser.add_argument(
        "--marginal-targets", nargs="*",
        default=["data/external/census/acs_pums_marginals.csv", "data/external/ces/ces_party_marginal.csv"],
        help="Long-format (variable,category,weighted_share) files for the education/income/party diagnostic",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.input:
        parser.error("--input is required (or pass --selftest)")

    run_pipeline(
        input_csv=args.input,
        output_csv=args.output,
        shortfall_csv=args.shortfall_report,
        n=args.n,
        seed=args.seed,
        supplementary_csv=args.supplementary_pool,
        strict=args.strict,
        marginal_targets_csvs=tuple(args.marginal_targets),
    )


if __name__ == "__main__":
    main()
