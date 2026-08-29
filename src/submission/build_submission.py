"""
Build the Tier-1 submission from a survey run.

    python src/submission/build_submission.py --run-id dummy_pilot [--team-id team_30 --entry primary --version 1]

Reads   data/processed/survey_responses/<run_id>/responses.jsonl
Writes  raw_data_deposit/<team_id>_raw_export_<run_id>.csv     Qualtrics-format raw export (what `make clean` expects)
        predictions/<team_id>_T1_<entry>_v<version>.csv         analysis-ready Tier-1 file (built directly, mirrors clean.R)
        data/processed/survey_responses/<run_id>/item_level.csv  wide per (agent x condition) item table incl. speculation
and (unless --no-manifest) records the prediction file + SHA-256 in metadata.json (like `make manifest`).

The Tier-1 file is constructed exactly like scripts/lib/clean_lib.R does it:
  composites = row means (NA-omitting), funding_perceptions = 100 - funding_5,
  newsletter_signup = 1 if Yes else 0, age_band from year_birth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import codebook as cb  # noqa: E402
from common.io_utils import dump_json, load_json, read_jsonl, sha256_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_TO_RAW = {v: k for k, v in cb.RAW_TO_TARGET.items()}


def load_responses(run_id: str, root: Path, include_partial: bool) -> list[dict]:
    path = root / "survey_responses" / run_id / "responses.jsonl"
    if not path.exists():
        raise SystemExit(f"no responses at {path}")
    recs = list(read_jsonl(path))
    # keep the latest record per (profile, condition) — resumed runs may have re-run partial pairs
    latest: dict[tuple[str, str], dict] = {}
    for r in recs:
        key = (r["profile_id"], r["condition"])
        prev = latest.get(key)
        rank = {"complete": 2, "partial": 1, "failed": 0}
        if prev is None or rank.get(r["status"], 0) >= rank.get(prev["status"], 0):
            latest[key] = r
    out = []
    n_drop = 0
    for r in latest.values():
        if r["status"] == "complete" or (include_partial and r["status"] == "partial"):
            out.append(r)
        else:
            n_drop += 1
    print(f"loaded {len(recs)} records -> {len(latest)} unique pairs -> {len(out)} used ({n_drop} dropped as "
          f"{'failed' if include_partial else 'partial/failed'})")
    return out


def raw_export_rows(recs: list[dict]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(recs):
        d = r["demographics"]
        ans = r["answers"]
        ts = r.get("timestamp") or dt.datetime.now(dt.timezone.utc).isoformat()
        row = {c: "" for c in cb.RAW_EXPORT_COLUMNS}
        row.update({
            "StartDate": ts, "EndDate": ts, "Status": 0, "Progress": 100, "Duration (in seconds)": int(r.get("seconds", 0)),
            "Finished": 1, "RecordedDate": ts, "ResponseId": f"R_{r['run_id']}_{i:06d}", "DistributionChannel": "anonymous",
            "UserLanguage": "EN",
            "condition": r.get("codename") or cb.CONDITION_TO_CODENAME.get(r["condition"], "control neckties"),
            "profile_id": r["profile_id"],
            "gender": cb.MODERATOR_CODES["gender"][d["gender"]],
            "year_birth": r.get("year_birth") or cb.AGE_BAND_REPRESENTATIVE_BIRTH_YEAR[d["age_band"]],
            "race": cb.MODERATOR_CODES["race"][d["race"]],
            "education": cb.MODERATOR_CODES["education"][d["education"]],
            "income": cb.MODERATOR_CODES["income"][d["income"]],
            "party": cb.MODERATOR_CODES["party"][d["party"]],
        })
        for key in cb.OUTCOME_ITEM_KEYS:
            item = cb.ITEM_BY_KEY[key]
            if key in ans and ans[key] is not None:
                row[key] = item.export_value(ans[key])
        rows.append(row)
    return pd.DataFrame(rows, columns=cb.RAW_EXPORT_COLUMNS)


def tier1_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.rename(columns=cb.RAW_TO_TARGET).copy()
    num = [c for c in cb.RAW_TO_TARGET.values() if c not in ("donation_ams", "newsletter_signup", "funding_perceptions")]
    for c in num + ["donation_ams", "funding_perceptions"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["funding_perceptions"] = 100 - d["funding_perceptions"]
    d["newsletter_signup"] = d["newsletter_signup"].map(lambda v: 1 if str(v).strip() in ("1", "Yes", "yes", "True") else (0 if str(v).strip() in ("2", "0", "No", "no", "False") else pd.NA))
    d["newsletter_signup"] = pd.to_numeric(d["newsletter_signup"], errors="coerce").astype("Int64")
    d["donation_ams"] = d["donation_ams"].round().astype("Int64")

    inv = {v: k for k, v in cb.MODERATOR_CODES["gender"].items()}
    d["gender"] = d["gender"].map(lambda v: inv[int(v)])
    for mod in ("race", "education", "income", "party"):
        inv = {v: k for k, v in cb.MODERATOR_CODES[mod].items()}
        d[mod] = d[mod].map(lambda v, inv=inv: inv[int(v)])
    d["age_band"] = d["year_birth"].map(lambda y: cb.age_band_from_year(int(y)))
    d["condition"] = d["condition"].map(lambda c: cb.CODENAME_TO_CONDITION.get(c, c))

    def rm(cols):
        return d[cols].mean(axis=1, skipna=True)

    d["trust_competence"] = rm(["trust_competence_1", "trust_competence_2", "trust_competence_3"])
    d["trust_integrity"] = rm(["trust_integrity_1", "trust_integrity_2", "trust_integrity_3"])
    d["trust_benevolence"] = rm(["trust_benevolence_1", "trust_benevolence_2", "trust_benevolence_3"])
    d["trust_openness"] = rm(["trust_openness_1", "trust_openness_2", "trust_openness_3"])
    d["trust_multidimensional"] = rm(["trust_competence", "trust_integrity", "trust_benevolence", "trust_openness"])
    d["policy_role_mean"] = rm([f"policy_role_{i}" for i in range(1, 5)])
    d["inst_trust_mean"] = rm(["inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa", "inst_trust_universities", "inst_trust_federal_gov"])
    d["concern_mean"] = rm(["concern_1", "concern_2", "concern_3"])
    d["policy_specific_mean"] = rm([f"policy_specific_{i}" for i in range(1, 8)])
    d["behavior_mean"] = rm(["behavior_meat", "behavior_transport", "behavior_solar", "behavior_fly", "behavior_talk", "behavior_donate"])
    out = d[cb.TIER1_COLUMNS].copy()
    order = {c: i for i, c in enumerate(cb.CONDITIONS)}
    out = out.sort_values(["condition", "profile_id"], key=lambda s: s.map(order) if s.name == "condition" else s).reset_index(drop=True)
    return out


def item_level_table(recs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in recs:
        row = {"profile_id": r["profile_id"], "condition": r["condition"], "codename": r.get("codename"),
               "status": r["status"], "n_calls": r.get("n_calls"), "model": r.get("model")}
        row.update({f"demo_{k}": v for k, v in r["demographics"].items()})
        row["assigned_state"] = r.get("assigned_state")
        for k, v in r["answers"].items():
            row[k] = v
        for k, v in (r.get("speculation") or {}).items():
            row[f"spec_{k}"] = v
        for k, v in (r.get("session_meta") or {}).items():
            row[f"meta_{k}"] = v if not isinstance(v, (list, dict)) else str(v)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data/processed")
    ap.add_argument("--team-id", default=None, help="defaults to metadata.json team_id")
    ap.add_argument("--entry", default=None, help="defaults to metadata.json entry (primary | secondary-k)")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--include-partial", action="store_true", help="also keep partially answered sessions (NA items)")
    ap.add_argument("--no-manifest", action="store_true", help="do not touch metadata.json")
    ap.add_argument("--predictions-dir", type=Path, default=REPO_ROOT / "predictions")
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "raw_data_deposit")
    ap.add_argument("--profile-id-mode", choices=["agent", "row"], default="agent",
                    help="agent: profile_id = agent id (reused across the 17 conditions; the benchmark check "
                         "WARNs on duplicates); row: profile_id = <agent id>__<condition slug> (unique per row)")
    args = ap.parse_args(argv)

    meta_path = REPO_ROOT / "metadata.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    team = args.team_id or meta.get("team_id") or "team"
    entry = args.entry or meta.get("entry") or "primary"

    recs = load_responses(args.run_id, args.processed_root, args.include_partial)
    if not recs:
        raise SystemExit("no usable responses")

    raw = raw_export_rows(recs)
    if args.profile_id_mode == "row":
        import re
        slug = raw["condition"].map(lambda c: re.sub(r"[^a-z0-9]+", "_", cb.CODENAME_TO_CONDITION.get(c, c).lower()).strip("_"))
        raw["profile_id"] = raw["profile_id"].astype(str) + "__" + slug
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_dir / f"{team}_raw_export_{args.run_id}.csv"
    raw.to_csv(raw_path, index=False)
    print(f"raw export : {raw_path}  ({len(raw)} rows)")

    t1 = tier1_from_raw(raw)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.predictions_dir / f"{team}_T1_{entry}_v{args.version}.csv"
    t1.to_csv(pred_path, index=False)
    print(f"predictions: {pred_path}  ({len(t1)} rows, {t1['profile_id'].nunique()} profiles, "
          f"{t1['condition'].nunique()} conditions)")
    counts = t1["condition"].value_counts()
    print("  per-condition N: min", counts.min(), "max", counts.max())

    il = item_level_table(recs)
    il_path = args.processed_root / "survey_responses" / args.run_id / "item_level.csv"
    il.to_csv(il_path, index=False)
    print(f"item table : {il_path}")

    if not args.no_manifest and meta_path.exists():
        rel = f"predictions/{pred_path.name}"
        files = [f for f in meta.get("prediction_files", []) if f.get("file") != rel and not Path(f.get("file", "")).name.startswith("example_")]
        files.append({"file": rel, "sha256": sha256_file(pred_path)})
        meta["prediction_files"] = files
        dump_json(meta_path, meta)
        print(f"metadata.json: recorded {rel} sha256={files[-1]['sha256'][:12]}...")


if __name__ == "__main__":
    main()
