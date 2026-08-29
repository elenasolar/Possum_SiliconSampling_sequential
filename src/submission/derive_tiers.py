"""
Derive Tier-2 (cell means) and Tier-3 (effects vs control) files from a Tier-1 file.

    python src/submission/derive_tiers.py [--tier1 predictions/team_30_T1_primary_v1.csv]
                                          [--team-id team_30 --entry primary --version 1]
                                          [--out-dir data/processed/derived_tiers]

Reads   the analysis-ready Tier-1 CSV (default: the T1 file listed in metadata.json)
Writes  <out>/<team_id>_T2_<entry>_v<n>_cells_main.csv       condition, outcome, mean
        <out>/<team_id>_T2_<entry>_v<n>_cells_moderator.csv  condition, moderator, moderator_level, outcome, mean
        <out>/<team_id>_T3_<entry>_v<n>.csv                  condition, outcome, ate
        <out>/derive_tiers_report.json                       cell counts, fallbacks, sanity checks

Aggregation (the benchmark documents no other rule; README "Tier-2 cell scales" +
"Tier-3 reports the 16 interventions' effects relative to control"):
  Tier-2 mean = unweighted, NA-omitting group average on the outcome's native scale
                (0-100 sliders, donation 0-10 dollars, newsletter_signup = share who
                subscribed, i.e. the mean of the 0/1 column).
  Tier-3 ate  = Tier-2 mean(intervention) - Tier-2 mean(control), i.e. the raw
                difference in means (no covariate adjustment, no weighting).
  Moderator cells with no Tier-1 respondent (e.g. gender "Other" if the sample
  has none) fall back to the condition's main-file mean = "no moderation", which
  is exactly what the FAQ prescribes; the report lists every such fallback.

NOTE: one repo = one tier. A Tier-1 entry is already scored on every Tier-2/3
metric, so these files are for our own reporting (RESULTS, paper tables), not
for the deposit -- hence the default out-dir is NOT predictions/. Point
--out-dir at predictions/ only for a separate Tier-2/3 entry repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import codebook as cb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# the 13 preregistered outcomes (scripts/lib/submission_spec.R `outcomes`)
OUTCOMES = [
    "trust_multidimensional",
    "trust_post", "distrust_post", "funding_perceptions",
    "policy_role_mean", "inst_trust_mean",
    "belief_post", "concern_mean", "policy_general",
    "policy_specific_mean", "behavior_mean",
    "donation_ams", "newsletter_signup",
]
RANGES = {**{o: (0.0, 100.0) for o in OUTCOMES}, "donation_ams": (0.0, 10.0), "newsletter_signup": (0.0, 1.0)}

T2_MAIN_COLS = ["condition", "outcome", "mean"]
T2_MOD_COLS = ["condition", "moderator", "moderator_level", "outcome", "mean"]
T3_COLS = ["condition", "outcome", "ate"]


def load_tier1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ["profile_id", "condition", *cb.MODERATORS, *OUTCOMES] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: Tier-1 file lacks required columns: {missing}")
    unknown = sorted(set(df["condition"]) - set(cb.CONDITIONS))
    if unknown:
        raise SystemExit(f"{path}: unknown condition labels: {unknown}")
    for o in OUTCOMES:
        df[o] = pd.to_numeric(df[o], errors="coerce")
    return df


def tier2_main(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("condition")[OUTCOMES].mean()          # NA-omitting per column
    g = g.reindex(cb.CONDITIONS)
    out = g.stack().rename("mean").reset_index()
    out.columns = T2_MAIN_COLS
    return out


def tier2_moderator(df: pd.DataFrame, main: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    main_lookup = main.set_index(["condition", "outcome"])["mean"]
    rows, fallbacks = [], []
    for mod, levels in cb.MODERATORS.items():
        bad = sorted(set(df[mod].dropna().astype(str)) - set(levels))
        if bad:
            raise SystemExit(f"moderator {mod}: Tier-1 has levels not in the spec: {bad}")
        g = df.groupby(["condition", mod])[OUTCOMES].mean()
        counts = df.groupby(["condition", mod]).size()
        for cond in cb.CONDITIONS:
            for lvl in levels:
                n = int(counts.get((cond, lvl), 0))
                for o in OUTCOMES:
                    val = g[o].get((cond, lvl)) if n else None
                    if val is None or pd.isna(val):
                        val = main_lookup[(cond, o)]
                        fallbacks.append({"condition": cond, "moderator": mod, "moderator_level": lvl,
                                          "outcome": o, "n_respondents": n})
                    rows.append((cond, mod, lvl, o, float(val)))
    return pd.DataFrame(rows, columns=T2_MOD_COLS), fallbacks


def tier3(main: pd.DataFrame) -> pd.DataFrame:
    wide = main.pivot(index="condition", columns="outcome", values="mean")
    ctrl = wide.loc[cb.CONTROL]
    eff = (wide.loc[cb.INTERVENTIONS] - ctrl)[OUTCOMES]
    out = eff.stack().rename("ate").reset_index()
    out.columns = T3_COLS
    return out


def sanity(main: pd.DataFrame, mod: pd.DataFrame, eff: pd.DataFrame) -> dict:
    n_levels = sum(len(v) for v in cb.MODERATORS.values())
    checks = {
        "t2_main_rows": (len(main), len(cb.CONDITIONS) * len(OUTCOMES)),
        "t2_moderator_rows": (len(mod), len(cb.CONDITIONS) * n_levels * len(OUTCOMES)),
        "t3_rows": (len(eff), len(cb.INTERVENTIONS) * len(OUTCOMES)),
        "no_na": (int(main["mean"].isna().sum() + mod["mean"].isna().sum() + eff["ate"].isna().sum()), 0),
        "no_duplicates": (int(main.duplicated(T2_MAIN_COLS[:2]).sum() + mod.duplicated(T2_MOD_COLS[:4]).sum()
                              + eff.duplicated(T3_COLS[:2]).sum()), 0),
    }
    out_of_range = []
    for frame in (main, mod):
        for o, (lo, hi) in RANGES.items():
            v = frame.loc[frame["outcome"] == o, "mean"]
            if ((v < lo) | (v > hi)).any():
                out_of_range.append(o)
    checks["in_range_outcomes"] = (len(RANGES) - len(set(out_of_range)), len(RANGES))
    ok = all(a == b for a, b in checks.values())
    return {"ok": ok, "checks": {k: {"got": a, "expected": b} for k, (a, b) in checks.items()},
            "out_of_range": sorted(set(out_of_range))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier1", type=Path, default=None, help="Tier-1 CSV (default: T1 file from metadata.json)")
    ap.add_argument("--metadata", type=Path, default=REPO_ROOT / "metadata.json")
    ap.add_argument("--team-id", default=None)
    ap.add_argument("--entry", default=None)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/processed/derived_tiers")
    ap.add_argument("--decimals", type=int, default=3, help="rounding as in the shipped example files")
    args = ap.parse_args(argv)

    meta = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata.exists() else {}
    team = args.team_id or meta.get("team_id", "team")
    entry = args.entry or meta.get("entry", "primary")
    t1 = args.tier1
    if t1 is None:
        cands = [REPO_ROOT / p["file"] for p in meta.get("prediction_files", []) if "_T1_" in p["file"]]
        if not cands:
            raise SystemExit("no Tier-1 file in metadata.json; pass --tier1")
        t1 = cands[0]

    df = load_tier1(t1)
    print(f"tier-1: {t1}  rows={len(df)}  agents={df['profile_id'].nunique()}  conditions={df['condition'].nunique()}")

    main_df = tier2_main(df)
    mod_df, fallbacks = tier2_moderator(df, main_df)
    eff_df = tier3(main_df)
    report = sanity(main_df, mod_df, eff_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{team}_T{{}}_{entry}_v{args.version}"
    paths = {
        "t2_main": args.out_dir / (stem.format(2) + "_cells_main.csv"),
        "t2_moderator": args.out_dir / (stem.format(2) + "_cells_moderator.csv"),
        "t3": args.out_dir / (stem.format(3) + ".csv"),
    }
    main_df.round(args.decimals).to_csv(paths["t2_main"], index=False)
    mod_df.round(args.decimals).to_csv(paths["t2_moderator"], index=False)
    eff_df.round(args.decimals).to_csv(paths["t3"], index=False)

    n_cells = df.groupby("condition").size()
    report.update({
        "tier1": str(t1), "n_rows": len(df), "n_agents": int(df["profile_id"].nunique()),
        "respondents_per_condition": {"min": int(n_cells.min()), "max": int(n_cells.max())},
        "moderator_fallback_cells": len(fallbacks),
        "moderator_fallback_levels": sorted({f"{f['moderator']}={f['moderator_level']}" for f in fallbacks}),
        "files": {k: str(v) for k, v in paths.items()},
    })
    (args.out_dir / "derive_tiers_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for k, p in paths.items():
        print(f"wrote {p}")
    if fallbacks:
        print(f"{len(fallbacks)} moderator cells had no respondents -> filled with the condition mean "
              f"(levels: {', '.join(report['moderator_fallback_levels'])})")
    print("sanity:", "PASS" if report["ok"] else "FAIL", json.dumps(report["checks"]))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
