"""
Python mirror of the benchmark's Tier-1 checks (scripts/lib/check_lib.R .check_t1 + filename/sha checks),
for machines without R. `make check` (R) remains the authoritative validator — run it before depositing.

    python src/submission/validate_submission.py                       # every prediction file in metadata.json
    python src/submission/validate_submission.py predictions/x_T1_primary_v1.csv

Exit code 1 on FAIL, 0 on PASS / PASS-WITH-WARNINGS.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import codebook as cb  # noqa: E402
from common.io_utils import load_json, sha256_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

SCALE_0_100 = [
    "trust_multidimensional", "trust_post", "distrust_post", "funding_perceptions", "policy_role_mean",
    "inst_trust_mean", "belief_post", "concern_mean", "policy_general", "policy_specific_mean", "behavior_mean",
]
TRUST_ITEMS = [c for c in cb.TIER1_COLUMNS if re.match(r"trust_(competence|integrity|benevolence|openness)_\d", c)]


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str = ""):
        self.rows.append((status, check, detail))

    def ok(self, cond: bool, check: str, bad: str = "", good: str = ""):
        self.add("PASS" if cond else "FAIL", check, good if cond else bad)

    def warn(self, cond: bool, check: str, bad: str = "", good: str = ""):
        self.add("PASS" if cond else "WARN", check, good if cond else bad)

    def verdict(self) -> str:
        st = {r[0] for r in self.rows}
        return "FAIL" if "FAIL" in st else ("PASS WITH WARNINGS" if "WARN" in st else "PASS")

    def print(self):
        for status, check, detail in self.rows:
            print(f"  [{status:4}] {check}" + (f" — {detail}" if detail else ""))
        print(f"VERDICT: {self.verdict()}")


def check_tier1_file(path: Path, rep: Report, expected_sha: str | None = None, team_id: str | None = None,
                     entry: str | None = None):
    f = path.name
    if team_id:
        pat = rf"^{re.escape(team_id)}_T1_{re.escape(entry or 'primary')}_v\d+\.csv$"
        rep.ok(bool(re.match(pat, f)), f"filename ok: {f}", "does not match <team_id>_T1_<entry>_v<n>.csv")
    if expected_sha:
        rep.ok(sha256_file(path) == expected_sha, f"sha256 matches metadata.json: {f}", "fingerprint mismatch — re-run manifest")
    d = pd.read_csv(path)
    miss = [c for c in cb.TIER1_COLUMNS if c not in d.columns]
    rep.ok(not miss, f"Tier-1 required columns: {f}", f"missing: {miss}")
    if miss:
        return
    bad = sorted(set(d["condition"].dropna().astype(str)) - set(cb.CONDITIONS))
    rep.ok(not bad, f"condition labels valid: {f}", f"unknown: {bad}")
    pres = [c for c in cb.CONDITIONS if c in set(d["condition"].astype(str))]
    rep.warn(len(pres) == len(cb.CONDITIONS), f"all 17 conditions present: {f}", f"{len(pres)} of 17 present")
    for mod, levels in cb.MODERATORS.items():
        vals = d[mod].dropna().astype(str)
        badm = sorted(set(vals) - set(levels))
        rep.ok(not badm, f"{mod} levels valid: {f}", f"unknown: {badm} — must exactly match spec strings")
        n_na = int(d[mod].isna().sum())
        if n_na == len(d):
            rep.add("FAIL", f"{mod} has data: {f}", "entirely NA")
        elif n_na > 0.1 * len(d):
            rep.add("WARN", f"{mod} mostly present: {f}", f"{n_na} of {len(d)} rows NA")
    n_cond = d["condition"].value_counts()
    rep.add("PASS", f"per-condition N: {f}",
            f"{n_cond.min()} in every condition" if n_cond.min() == n_cond.max() else
            f"min {n_cond.min()} ({n_cond.idxmin()}), max {n_cond.max()} ({n_cond.idxmax()})")
    below = [c for c in n_cond.index if c != "control" and n_cond[c] < 500]
    if "control" in n_cond and n_cond["control"] < 1000:
        below.append("control")
    rep.warn(not below, f"precision floor (500/intervention, 1,000 control): {f}",
             f"{len(below)} condition(s) below minimum: {below[:5]}")
    n_dup = int(d["profile_id"].duplicated().sum())
    rep.warn(n_dup == 0, f"profile_id unique: {f}", f"{n_dup} duplicate(s) (expected when personas are reused across conditions)")
    sub = pd.concat([d[[f"trust_{s}_{i}" for i in (1, 2, 3)]].mean(axis=1, skipna=True)
                     for s in ("competence", "integrity", "benevolence", "openness")], axis=1).mean(axis=1, skipna=True)
    n_bad = int(((d["trust_multidimensional"] - sub).abs() > 0.51).sum())
    rep.warn(n_bad == 0, f"trust_multidimensional consistent with items: {f}", f"{n_bad} row(s) deviate > 0.5")
    for o in SCALE_0_100 + TRUST_ITEMS:
        v = pd.to_numeric(d[o], errors="coerce")
        n_out = int(((v < 0) | (v > 100)).sum())
        if n_out:
            rep.add("WARN", f"{o} in [0,100]: {f}", f"{n_out} value(s) out of range")
        n_na = int(v.isna().sum())
        if n_na:
            rep.add("WARN", f"{o} complete: {f}", f"{n_na} NA value(s)")
    v = pd.to_numeric(d["donation_ams"], errors="coerce")
    rep.warn(int(((v < 0) | (v > 10)).sum()) == 0, f"donation_ams in [0,10]: {f}", "value(s) out of range")
    rep.warn(bool(((v.dropna() % 1) == 0).all()), f"donation_ams integer: {f}", "non-integer value(s)")
    vals = set(d["newsletter_signup"].dropna().astype(str))
    rep.warn(vals <= {"0", "1", "TRUE", "FALSE", "0.0", "1.0"}, f"newsletter_signup binary: {f}", f"unexpected: {sorted(vals)[:5]}")
    for col in ["condition", "profile_id"]:
        rep.ok(int(d[col].isna().sum()) == 0, f"{col} non-missing: {f}", "NA values")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="prediction CSVs (default: from metadata.json)")
    ap.add_argument("--metadata", type=Path, default=REPO_ROOT / "metadata.json")
    args = ap.parse_args(argv)
    rep = Report()
    meta = load_json(args.metadata) if args.metadata.exists() else {}
    team = meta.get("team_id")
    entry = meta.get("entry", "primary")
    if args.files:
        for f in args.files:
            check_tier1_file(f, rep, team_id=team, entry=entry)
    else:
        pf = meta.get("prediction_files") or []
        rep.ok(bool(pf), "prediction_files declared in metadata.json", "none declared")
        for e in pf:
            p = REPO_ROOT / e["file"]
            if not p.exists():
                rep.add("FAIL", f"file exists: {e['file']}", "missing")
                continue
            check_tier1_file(p, rep, expected_sha=e.get("sha256"), team_id=team, entry=entry)
        cov = meta.get("coverage") or {}
        rep.ok(cov.get("interventions") == 16 and cov.get("outcomes") == 13, "coverage full (16 interventions, 13 outcomes)", f"declared {cov}")
        rep.ok(meta.get("tier") == 1, "tier == 1", f"tier is {meta.get('tier')}")
        rep.warn(bool(meta.get("blinding_attestation")), "blinding_attestation true", "not attested")
        rep.warn(not any(Path(e["file"]).name.startswith("example_") for e in pf), "no example_* files in manifest", "example files still listed")
    rep.print()
    sys.exit(1 if rep.verdict() == "FAIL" else 0)


if __name__ == "__main__":
    main()
