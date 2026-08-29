"""
Pipeline position: independent branch (see ACS/00_pull_acs_pums.py's note) --
only needed as 06_stratified_sample.py's --marginal-targets diagnostic.

Compute weighted party-affiliation distributions (marginal, and jointly with
demographics where available) from the Cooperative Election Study (CES).

Unlike ACS/PUMS, CES is NOT available via a simple public API pull — it must
be downloaded manually first:

    1. Go to the Harvard Dataverse CES page:
       https://dataverse.harvard.edu/dataverse/cces
    2. Download the most recent year's "Common Content" file (CSV or Stata
       .dta format). Requires a free Dataverse account.
    3. Place the file at: data/external/ces/ces_common_content_<year>.dta
       (or .csv — set CES_FILE_PATH and CES_FILE_FORMAT below to match)

This script then recodes CES's party-ID variable into the project's four
categories (Republican / Democrat / Independent / Other) and computes
weighted shares using the CES sample weight.

IMPORTANT: CES variable names change slightly across years — check the
year-specific codebook (also on the Dataverse page) and update the
CES_PARTY_VAR / CES_WEIGHT_VAR constants below if they don't match.

Alternative / cross-check sources (no microdata, but useful as a sanity
check on the CES-derived marginal shares):
    - Pew Research Center party affiliation trend: https://www.pewresearch.org/politics/
    - Gallup party affiliation tracking: https://news.gallup.com/poll/15370/party-affiliation.aspx
"""

from pathlib import Path

import pandas as pd

from recode_schema import PARTY_CATEGORIES  # colocated in this same ACS/ folder -- single source for these 4 strings

# --- Configure these to match the downloaded CES year/file -----------------
CES_FILE_PATH = Path(__file__).resolve().parents[3] / "data" / "external" / "ces" / "ces_common_content_2025_year.dta"
CES_FILE_FORMAT = "stata"  # "stata" or "csv"

# CES's 3-category party ID variable (check codebook — commonly named
# something like "pid3" or "CC[year]_311" depending on year/wave).
CES_PARTY_VAR = "pid3"
# CES sample weight (commonly "commonweight" or "commonweight_vv" — check
# the codebook for the exact name in your downloaded year).
CES_WEIGHT_VAR = "commonweight"

# Demographic variables to cross-tab against party, IF you want the joint
# distribution rather than just the party marginal. Names vary by year —
# check the codebook and adjust. Set to [] to skip joint tabulation.
CES_DEMO_VARS = []  # e.g. ["gender", "race", "educ", "faminc_new"] if present

# CES pid3 coding (verify against your year's codebook — this is the
# standard scheme but has had minor variants):
#   1 = Democrat, 2 = Republican, 3 = Independent, 4 = Other, 5 = Not sure
REPUBLICAN, DEMOCRAT, INDEPENDENT, OTHER = PARTY_CATEGORIES

PID3_MAP = {
    1: DEMOCRAT,
    2: REPUBLICAN,
    3: INDEPENDENT,
    4: OTHER,
    5: OTHER,
}


def load_ces() -> pd.DataFrame:
    if not CES_FILE_PATH.exists():
        raise FileNotFoundError(
            f"CES file not found at {CES_FILE_PATH}. Download it manually from "
            "https://dataverse.harvard.edu/dataverse/cces first — see module "
            "docstring for instructions."
        )
    if CES_FILE_FORMAT == "stata":
        return pd.read_stata(CES_FILE_PATH, convert_categoricals=False)
    return pd.read_csv(CES_FILE_PATH)


def recode_party(df: pd.DataFrame) -> pd.DataFrame:
    if CES_PARTY_VAR not in df.columns:
        raise KeyError(
            f"'{CES_PARTY_VAR}' not found in CES file columns. Check the "
            f"year-specific codebook and update CES_PARTY_VAR. "
            f"Available columns include: {list(df.columns)[:20]}..."
        )
    out = df.copy()
    out["party"] = out[CES_PARTY_VAR].map(PID3_MAP)
    return out


def compute_party_marginal(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["party", CES_WEIGHT_VAR])
    total_weight = sub[CES_WEIGHT_VAR].sum()
    shares = sub.groupby("party")[CES_WEIGHT_VAR].sum() / total_weight
    return shares.reset_index(name="weighted_share").rename(columns={"party": "category"})


def compute_party_joint(df: pd.DataFrame) -> pd.DataFrame | None:
    if not CES_DEMO_VARS:
        return None
    group_vars = ["party"] + CES_DEMO_VARS
    sub = df.dropna(subset=group_vars + [CES_WEIGHT_VAR])
    total_weight = sub[CES_WEIGHT_VAR].sum()
    joint = (
        sub.groupby(group_vars)[CES_WEIGHT_VAR].sum() / total_weight
    ).reset_index(name="weighted_share")
    return joint.sort_values("weighted_share", ascending=False)


def main():
    print(f"Loading CES file from {CES_FILE_PATH}...")
    raw = load_ces()
    print(f"Loaded {len(raw):,} respondents.")

    recoded = recode_party(raw)

    output_dir = CES_FILE_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    marginal = compute_party_marginal(recoded)
    marginal["variable"] = "party"
    marginal = marginal[["variable", "category", "weighted_share"]]
    marginal_path = output_dir / "ces_party_marginal.csv"
    marginal.to_csv(marginal_path, index=False)
    print(f"Saved party marginal distribution: {marginal_path}")
    print(marginal)

    joint = compute_party_joint(recoded)
    if joint is not None:
        joint_path = output_dir / "ces_party_joint_distribution.csv"
        joint.to_csv(joint_path, index=False)
        print(f"Saved party joint distribution: {joint_path}")
    else:
        print(
            "CES_DEMO_VARS is empty — skipped joint tabulation. Set it if you "
            "want party jointly with other demographics (e.g. to cross-check "
            "against the ACS PUMS joint distribution rather than merging two "
            "independently-marginal sources)."
        )


if __name__ == "__main__":
    main()
