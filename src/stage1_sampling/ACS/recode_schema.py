"""
Recoding functions mapping raw ACS PUMS variable codes onto the project's
exact demographic categories, as defined in docs/06_CODEBOOK.md.

Category strings themselves come from `src/common/codebook.py` (the
machine-readable export 06_CODEBOOK.md calls for) -- this file only holds
the ACS/PUMS-specific numeric-code -> category mapping logic, not the
category strings. If docs/06_CODEBOOK.md changes, update codebook.py first;
this file follows automatically for anything it imports by name.

ACS PUMS 2024 data dictionary (variable code meanings):
https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> src/
from common.codebook import GENDER, RACE, EDUCATION, INCOME, PARTY as PARTY_CATEGORIES  # noqa: E402

MALE, FEMALE, OTHER_GENDER = GENDER
WHITE, BLACK, HISPANIC, ASIAN, OTHER_RACE = RACE
LESS_THAN_HS, HS_DIPLOMA, SOME_COLLEGE, BACHELORS, MASTERS, DOCTORATE = EDUCATION
INCOME_UNDER_30K, INCOME_30_56K, INCOME_56_100K, INCOME_100_168K, INCOME_OVER_168K = INCOME


# ---------------------------------------------------------------------------
# Gender
# ---------------------------------------------------------------------------
# ACS PUMS SEX: 1 = Male, 2 = Female. No "Other" category exists in ACS PUMS —
# this is a known limitation, not a bug in this recoding. Document this gap
# explicitly in registration.md (D.1) if it becomes relevant.
SEX_MAP = {1: MALE, 2: FEMALE}


def recode_gender(sex_code: int) -> str:
    return SEX_MAP.get(sex_code, OTHER_GENDER)


# ---------------------------------------------------------------------------
# Age band
# ---------------------------------------------------------------------------
def recode_age_band(agep: int) -> str | None:
    """AGEP is age in years. Restrict to 18+ (returns None for minors,
    which should be filtered out upstream — PUMS includes all ages)."""
    if agep < 18:
        return None
    if agep <= 29:
        return "18-29"
    if agep <= 44:
        return "30-44"
    if agep <= 59:
        return "45-59"
    return "60+"


# ---------------------------------------------------------------------------
# Race / ethnicity
# ---------------------------------------------------------------------------
# ACS treats Hispanic/Latino origin (HISP) as orthogonal to race (RAC1P).
# Per the project's schema, Hispanic/Latino is its own category and takes
# priority over race — i.e. someone who is both "Hispanic" and "White alone"
# by Census coding is bucketed as Hispanic/Latino here, not White, to avoid
# double-counting. This is a modeling choice — flag it in registration.md (D.1)
# since it's a deviation from Census's own race/ethnicity crosstab convention.
#
# HISP: 01 = Not Spanish/Hispanic/Latino; 02-24 = various Hispanic origins.
# RAC1P: 1 = White alone, 2 = Black alone, 3-5 = American Indian/Alaska
#        Native, 6 = Asian alone, 7 = Native Hawaiian/Pacific Islander,
#        8 = Some other race alone, 9 = Two or more races.
RAC1P_MAP = {
    1: WHITE,
    2: BLACK,
    3: OTHER_RACE,
    4: OTHER_RACE,
    5: OTHER_RACE,
    6: ASIAN,
    7: OTHER_RACE,
    8: OTHER_RACE,
    9: OTHER_RACE,
}


def recode_race(hisp_code: int, rac1p_code: int) -> str:
    if hisp_code != 1:  # anything other than "01" = some Hispanic/Latino origin
        return HISPANIC
    return RAC1P_MAP.get(rac1p_code, OTHER_RACE)


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------
# SCHL: educational attainment, 2024 ACS PUMS codes.
#   01-15 = less than regular high school diploma (various grade levels)
#   16    = Regular high school diploma
#   17    = GED or alternative credential
#   18    = Some college, less than 1 year
#   19    = Some college, 1+ years, no degree
#   20    = Associate's degree
#   21    = Bachelor's degree
#   22    = Master's degree
#   23    = Professional degree beyond a bachelor's degree
#   24    = Doctorate degree
def recode_education(schl_code: int) -> str | None:
    if schl_code is None:
        return None
    if schl_code <= 15:
        return LESS_THAN_HS
    if schl_code in (16, 17):
        return HS_DIPLOMA
    if schl_code in (18, 19, 20):
        return SOME_COLLEGE
    if schl_code == 21:
        return BACHELORS
    if schl_code in (22, 23):
        return MASTERS
    if schl_code == 24:
        return DOCTORATE
    return None


# ---------------------------------------------------------------------------
# Income
# ---------------------------------------------------------------------------
# PINCP = total person's income (past 12 months), can be negative (losses).
# Bands filter to non-negative income; negative/zero-income records are a
# judgment call — see note in 00_pull_acs_pums.py on how they're handled.
def recode_income(pincp: float) -> str | None:
    if pincp is None or pincp < 0:
        return None
    if pincp < 30_000:
        return INCOME_UNDER_30K
    if pincp < 56_000:
        return INCOME_30_56K
    if pincp < 100_000:
        return INCOME_56_100K
    if pincp < 168_000:
        return INCOME_100_168K
    return INCOME_OVER_168K


# ---------------------------------------------------------------------------
# Party (not derivable from ACS/PUMS — see 01_pull_ces_party.py)
# ---------------------------------------------------------------------------
# PARTY_CATEGORIES is imported above from common.codebook and re-exported
# here unchanged -- 01_pull_ces_party.py does `from recode_schema import
# PARTY_CATEGORIES` rather than importing common.codebook directly.


def apply_all_recodes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all recodes to a raw ACS PUMS person-record dataframe.
    Expects columns: AGEP, SEX, HISP, RAC1P, SCHL, PINCP, PWGTP, ST (all as
    returned by the Census API — see 00_pull_acs_pums.py for the fetch step)."""
    out = df.copy()
    out["gender"] = out["SEX"].astype(int).map(recode_gender)
    out["age_band"] = out["AGEP"].astype(int).map(recode_age_band)
    out["race"] = out.apply(
        lambda r: recode_race(int(r["HISP"]), int(r["RAC1P"])), axis=1
    )
    out["education"] = out["SCHL"].apply(
        lambda x: recode_education(int(x)) if pd.notna(x) else None
    )
    out["income"] = out["PINCP"].apply(
        lambda x: recode_income(float(x)) if pd.notna(x) else None
    )
    return out
