"""
Pipeline position: independent branch, not part of the main 00-06 Reddit-candidate
sequence -- only needed as 06_stratified_sample.py's --marginal-targets diagnostic.
Run any time; only requires CENSUS_API_KEY. See claude_project_plan/07_REPO_STRUCTURE.md §1.

Pull 2024 ACS 1-Year PUMS (Public Use Microdata Sample) person records from
the Census API, recode into the project's exact demographic bands, and
compute weighted population distributions for use as the census-matching
target in Stage 1 stratified sampling.

Sampling design (matches the megastudy's own approach): hard cross-quotas
are used only on age_band x gender and gender x race. Education, income, and
party are matched only at the marginal level (final sample should roughly
track census shares, but is not quota-sampled to them directly).

Requires a free Census API key: https://api.census.gov/data/key_signup.html
Set it as an environment variable before running:
    export CENSUS_API_KEY=your_key_here

Usage:
    python 00_pull_acs_pums.py

Output:
    data/external/census/acs_pums_2024_recoded.csv          (person-level, recoded)
    data/external/census/acs_pums_marginals.csv             (weighted share per category, per variable — all 5)
    data/external/census/acs_pums_crosstab_age_gender.csv    (quota target: age_band x gender)
    data/external/census/acs_pums_crosstab_gender_race.csv   (quota target: gender x race)

Notes:
- PUMS is queried per state and concatenated, since the Census API's PUMS
  endpoint requires a state (or "for=state:*") filter rather than a single
  national pull.
- PWGTP is the person-level sampling weight; all shares below are weighted,
  not raw record counts, to correctly reflect the US population.
- This pulls the 1-year file (most current, single-year snapshot). Consider
  the 5-year file (more stable estimates, esp. for small states/cells) if
  the two crosstabs below look too thin in any cell.
- No political party variable exists in ACS/PUMS — see 01_pull_ces_party.py.
  Only its marginal is needed under this design, which 01_pull_ces_party.py
  already produces.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

from recode_schema import apply_all_recodes  # colocated in this same ACS/ folder

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY")
if not CENSUS_API_KEY:
    raise SystemExit(
        "Set the CENSUS_API_KEY environment variable first. "
        "Get a free key at https://api.census.gov/data/key_signup.html"
    )

ACS_YEAR = 2024
BASE_URL = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs1/pums"

# Variables to pull. Keep this list minimal — the Census API rate-limits and
# large variable lists slow requests down. Add more here only if a later
# stage needs them (e.g., PUMA for finer geography).
# Note: no "ST" here -- each request already filters by state via `for=state:X`,
# and the API rejects ST as an unrecognized `get` variable when queried this way.
PUMS_VARS = ["AGEP", "SEX", "HISP", "RAC1P", "SCHL", "PINCP", "PWGTP"]

# All 50 states + DC (FIPS codes). PUMS doesn't include territories by default.
STATE_FIPS = [f"{i:02d}" for i in range(1, 57) if i not in (3, 7, 14, 43, 52)]

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "external" / "census"


def fetch_state(state_fips: str) -> pd.DataFrame:
    """Fetch PUMS person records for a single state."""
    params = {
        "get": ",".join(PUMS_VARS),
        "for": f"state:{state_fips}",
        "key": CENSUS_API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df


def fetch_all_states() -> pd.DataFrame:
    frames = []
    failed = []
    for fips in STATE_FIPS:
        print(f"Fetching state FIPS {fips}...")
        try:
            frames.append(fetch_state(fips))
        except requests.HTTPError as e:
            body = e.response.text if e.response is not None else "(no response body)"
            print(f"  WARNING: failed for state {fips}: {e}\n    Response: {body}")
            failed.append(fips)
        time.sleep(0.5)  # polite rate limiting
    if failed:
        raise RuntimeError(
            f"Failed to fetch {len(failed)}/{len(STATE_FIPS)} states: {failed}. "
            "Refusing to compute national marginals/crosstabs from partial state "
            "coverage -- every resulting share would be silently biased. Rerun "
            "(check API key / rate limits) rather than proceeding on partial data."
        )
    return pd.concat(frames, ignore_index=True)


def compute_marginals(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted marginal share (%) for each category of each demographic
    variable, restricted to records with a valid (non-null) recode."""
    rows = []
    for var in ["gender", "age_band", "race", "education", "income"]:
        sub = df.dropna(subset=[var])
        total_weight = sub["PWGTP"].sum()
        shares = sub.groupby(var)["PWGTP"].sum() / total_weight
        for category, share in shares.items():
            rows.append({"variable": var, "category": category, "weighted_share": share})
    return pd.DataFrame(rows)


def compute_two_way_crosstab(df: pd.DataFrame, var1: str, var2: str) -> pd.DataFrame:
    """Weighted share for every combination of two demographic variables.
    Used as a hard quota-sampling target (age_band x gender, gender x race),
    per the megastudy's own two-tier sampling design — see module docstring."""
    sub = df.dropna(subset=[var1, var2])
    total_weight = sub["PWGTP"].sum()
    crosstab = (
        sub.groupby([var1, var2])["PWGTP"].sum() / total_weight
    ).reset_index(name="weighted_share")
    return crosstab.sort_values("weighted_share", ascending=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Pulling ACS PUMS person records for all states...")
    raw = fetch_all_states()

    # Cast numeric columns (Census API returns everything as strings)
    for col in ["AGEP", "SEX", "HISP", "RAC1P", "SCHL", "PINCP", "PWGTP"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    print(f"Pulled {len(raw):,} person records. Applying recodes...")
    recoded = apply_all_recodes(raw)

    # Restrict to adults (age_band recode returns None for under-18s)
    recoded = recoded.dropna(subset=["age_band"])

    recoded_path = OUTPUT_DIR / "acs_pums_2024_recoded.csv"
    recoded.to_csv(recoded_path, index=False)
    print(f"Saved recoded person-level file: {recoded_path} ({len(recoded):,} rows)")

    marginals = compute_marginals(recoded)
    marginals_path = OUTPUT_DIR / "acs_pums_marginals.csv"
    marginals.to_csv(marginals_path, index=False)
    print(f"Saved marginal distributions: {marginals_path}")
    print(marginals)

    # Hard quota-sampling targets: age_band x gender, gender x race.
    # Education, income, and party are marginal-only under this design (see
    # module docstring) — their shares are already in acs_pums_marginals.csv
    # above (party's marginal comes from 01_pull_ces_party.py instead).
    age_gender = compute_two_way_crosstab(recoded, "age_band", "gender")
    age_gender_path = OUTPUT_DIR / "acs_pums_crosstab_age_gender.csv"
    age_gender.to_csv(age_gender_path, index=False)
    print(f"Saved age_band x gender crosstab ({len(age_gender)} cells): {age_gender_path}")

    gender_race = compute_two_way_crosstab(recoded, "gender", "race")
    gender_race_path = OUTPUT_DIR / "acs_pums_crosstab_gender_race.csv"
    gender_race.to_csv(gender_race_path, index=False)
    print(f"Saved gender x race crosstab ({len(gender_race)} cells): {gender_race_path}")

    for name, tab in [("age_band x gender", age_gender), ("gender x race", gender_race)]:
        n_sparse = (tab["weighted_share"] < 0.001).sum()
        if n_sparse:
            print(
                f"\nNOTE: {n_sparse} of {len(tab)} cells in the {name} crosstab "
                "have <0.1% population share — check against your oversampled "
                "pool size to confirm enough headroom for the sparsest cells."
            )


if __name__ == "__main__":
    main()
