"""
Machine-readable codebook: the single import point for every category string,
condition name, item definition and mapping used across the pipeline.

Mirrors (and must stay in sync with):
  - claude_project_plan/06_CODEBOOK.md
  - codebook.csv, survey/condition_codenames.csv
  - scripts/lib/submission_spec.R  (canonical benchmark spec)

If the docs change, change this file first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 1. Demographic moderators (exact submission strings)
# --------------------------------------------------------------------------- #

MODERATORS: dict[str, list[str]] = {
    "gender": ["Male", "Female", "Other"],
    "age_band": ["18-29", "30-44", "45-59", "60+"],
    "race": [
        "White / Caucasian",
        "Black / African American",
        "Hispanic / Latino",
        "Asian / Asian American",
        "Other",
    ],
    "education": [
        "Less than high school",
        "High school diploma / GED",
        "Some college or Associate's degree",
        "Bachelor's degree",
        "Master's degree / Professional degree",
        "Doctorate degree / Ph.D.",
    ],
    "income": [
        "Less than $30,000",
        "$30,000 to $55,999",
        "$56,000 to $99,999",
        "$100,000 to $167,999",
        "$168,000 or more",
    ],
    "party": ["Republican", "Democrat", "Independent", "Other"],
}

# Flat aliases (used by src/stage1_sampling/05_infer_demographics_llm.py and the ACS/CES recoding scripts)
GENDER = MODERATORS["gender"]
AGE_BAND = MODERATORS["age_band"]
RACE = MODERATORS["race"]
EDUCATION = MODERATORS["education"]
INCOME = MODERATORS["income"]
PARTY = MODERATORS["party"]
MODERATOR_CATEGORIES = MODERATORS

# Raw Qualtrics export codes (used when writing raw_data_deposit/ exports that
# scripts/clean.R can consume). Party uses EXPORTED codes (1=Rep, 2=Dem, 3=Ind, 4=Other).
MODERATOR_CODES: dict[str, dict[str, int]] = {
    mod: {level: i + 1 for i, level in enumerate(levels)} for mod, levels in MODERATORS.items()
}

# Representative birth year per age band, used ONLY when an agent record carries
# no `year_birth`/`age` (age = 2026 - year_birth; band edges 18-29/30-44/45-59/60+).
AGE_BAND_REPRESENTATIVE_BIRTH_YEAR: dict[str, int] = {
    "18-29": 2026 - 24,
    "30-44": 2026 - 37,
    "45-59": 2026 - 52,
    "60+": 2026 - 68,
}

SURVEY_YEAR = 2026


def age_band_from_year(year_birth: int) -> str:
    age = SURVEY_YEAR - int(year_birth)
    if age <= 29:
        return "18-29"
    if age <= 44:
        return "30-44"
    if age <= 59:
        return "45-59"
    return "60+"


# --------------------------------------------------------------------------- #
# 2. Conditions
# --------------------------------------------------------------------------- #

CONTROL = "control"

INTERVENTIONS: list[str] = [
    "Corporate reliance",
    "Social justice",
    "Interview Prof. Maraun",
    "Funding",
    "Oil industry misinformation",
    "Measurement & modeling (1)",
    "Former skeptics",
    "High public trust",
    "Measurement & modeling (2)",
    "Peer-review",
    "Scientist community helpers",
    "Consensus",
    "Portrait Prof. Cherry",
    "Model accuracy",
    "Interview Prof. Sebille",
    "Extreme weather predictions",
]

CONDITIONS: list[str] = [CONTROL] + INTERVENTIONS

# raw survey code name -> canonical title (semicolons are part of the name)
CODENAME_TO_CONDITION: dict[str, str] = {
    "control neckties": CONTROL,
    "control baseball": CONTROL,
    "control dances": CONTROL,
    "practical planarian": "Extreme weather predictions",
    "complicated cockroach": "Portrait Prof. Cherry",
    "flimsy fish": "Interview Prof. Maraun",
    "honored haddock": "Peer-review",
    "jealous jaguar": "Consensus",
    "phony parrotfish": "Funding",
    "crushing chicken; gross grasshopper; homely halibut": "High public trust",
    "worse wildfowl": "Oil industry misinformation",
    "periwinkle partridge": "Scientist community helpers",
    "difficult dog": "Social justice",
    "giant gibbon; brick bobcat": "Corporate reliance",
    "limping llama; friendly frog": "Former skeptics",
    "perfect prawn": "Measurement & modeling (1)",
    "orchid orangutan; defiant dragonfly": "Measurement & modeling (2)",
    "apple aardvark": "Model accuracy",
    "heartfelt hummingbird": "Interview Prof. Sebille",
}

CONDITION_TO_CODENAME: dict[str, str] = {
    v: k for k, v in CODENAME_TO_CONDITION.items() if v != CONTROL
}
CONTROL_CODENAMES: list[str] = ["control neckties", "control baseball", "control dances"]

# --------------------------------------------------------------------------- #
# 3. Intervention #16 state -> case mapping
# --------------------------------------------------------------------------- #

STATE_CASE: dict[str, int] = {}
for _s in [
    "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maryland", "Mississippi", "Missouri", "Nebraska",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Pennsylvania", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Virginia", "West Virginia", "Washington, D.C.",
]:
    STATE_CASE[_s] = 1
for _s in [
    "Alaska", "Arizona", "California", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico",
    "Oregon", "Utah", "Washington", "Wyoming", "Hawaii",
]:
    STATE_CASE[_s] = 2
for _s in [
    "Connecticut", "Maine", "Massachusetts", "Michigan", "Minnesota", "New Hampshire",
    "New Jersey", "New York", "Rhode Island", "Vermont", "Wisconsin",
]:
    STATE_CASE[_s] = 3

CASE_LABEL: dict[int, str] = {
    1: "states with high or recurrent flood risk",
    2: "states with high or increasing wildfire risk",
    3: "states with severe cold, snow, ice, or blizzards",
}

US_STATES: list[str] = sorted(STATE_CASE.keys())

# Common aliases -> canonical state name used in STATE_CASE
STATE_ALIASES: dict[str, str] = {
    "washington dc": "Washington, D.C.",
    "washington d.c.": "Washington, D.C.",
    "washington, dc": "Washington, D.C.",
    "district of columbia": "Washington, D.C.",
    "dc": "Washington, D.C.",
    "d.c.": "Washington, D.C.",
}
_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "Washington, D.C.",
}


def normalize_state(value) -> str | None:
    """Map free-form state strings ("tx", "Texas", "washington dc") to canonical names; None if unknown."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "prefer not to say"}:
        return None
    if s.upper() in _ABBR:
        return _ABBR[s.upper()]
    low = s.lower().replace("r/", "")
    if low in STATE_ALIASES:
        return STATE_ALIASES[low]
    for st in STATE_CASE:
        if st.lower() == low:
            return st
    return None


def state_to_case(state: str | None) -> int:
    st = normalize_state(state)
    return STATE_CASE.get(st, 4) if st else 4


# --------------------------------------------------------------------------- #
# 4. Survey items (post-treatment outcome battery + optional pre-treatment)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Item:
    """One survey item as it is presented to the annotator.

    key:            internal key == raw Qualtrics export column (codebook.csv qualtrics_label)
    title:          the TITLE token used in the prompt / parsed output (unique, uppercase)
    text:           verbatim question wording
    kind:           'slider' (0-100 integer) | 'choice' (pick one symbol)
    lo/hi labels:   slider anchors; mid_label optional
    choices:        for 'choice': list of (symbol, label, export_value)
    """

    key: str
    title: str
    text: str
    kind: str = "slider"
    lo_label: str = ""
    hi_label: str = ""
    mid_label: str = ""
    choices: tuple = field(default_factory=tuple)
    prefix: str = ""  # optional prefix shown before the item text (e.g. "Statement:")

    def export_value(self, answer):
        """Convert a parsed answer (int for sliders, symbol for choices) into raw Qualtrics export value."""
        if self.kind == "slider":
            return int(answer)
        for sym, _label, val in self.choices:
            if sym == answer:
                return val
        raise ValueError(f"unknown choice symbol {answer!r} for item {self.key}")


@dataclass(frozen=True)
class Block:
    """A survey page/block: an optional intro shown above a set of items."""

    key: str
    intro: str
    items: tuple[Item, ...]
    pre_pages: tuple[str, ...] = field(default_factory=tuple)  # display-only pages shown before the items


SLIDER_INSTRUCTION = (
    "Answer options below range from 0 to 100. Click on any space within this range and a bar "
    "will appear. Feel free to move that bar around to the number that best represents your answer."
)

# ---- Primary outcome: multidimensional trust (always first) -------------------
TRUST_ITEMS = (
    Item("trust_competent_1", "TRUST_COMPETENT", "How incompetent or competent are most climate scientists?", lo_label="Very incompetent", hi_label="Very competent"),
    Item("trust_intelligent_1", "TRUST_INTELLIGENT", "How unintelligent or intelligent are most climate scientists?", lo_label="Very unintelligent", hi_label="Very intelligent"),
    Item("trust_qualified_1", "TRUST_QUALIFIED", "How unqualified or qualified are most climate scientists?", lo_label="Very unqualified", hi_label="Very qualified"),
    Item("trust_honest_1", "TRUST_HONEST", "How dishonest or honest are most climate scientists?", lo_label="Very dishonest", hi_label="Very honest"),
    Item("trust_ethical_1", "TRUST_ETHICAL", "How unethical or ethical are most climate scientists?", lo_label="Very unethical", hi_label="Very ethical"),
    Item("trust_sincere_1", "TRUST_SINCERE", "How insincere or sincere are most climate scientists?", lo_label="Very insincere", hi_label="Very sincere"),
    Item("trust_concerned_1", "TRUST_CONCERNED", "How unconcerned or concerned are most climate scientists about people’s wellbeing?", lo_label="Very unconcerned", hi_label="Very concerned"),
    Item("trust_improve_1", "TRUST_IMPROVE", "How uneager or eager are most climate scientists to improve others’ lives?", lo_label="Very uneager", hi_label="Very eager"),
    Item("trust_considerate_1", "TRUST_CONSIDERATE", "How inconsiderate or considerate are most climate scientists of others’ interests?", lo_label="Very inconsiderate", hi_label="Very considerate"),
    Item("trust_feedback_1", "TRUST_FEEDBACK", "How open, if at all, are most climate scientists to feedback?", lo_label="Not open at all", hi_label="Very open"),
    Item("trust_transparent_1", "TRUST_TRANSPARENT", "How unwilling or willing are most climate scientists to be transparent?", lo_label="Very unwilling", hi_label="Very willing"),
    Item("trust_attention_1", "TRUST_ATTENTION", "How much or how little attention do climate scientists pay to other people's views?", lo_label="Very little attention", hi_label="A great deal of attention"),
)

BLOCK_TRUST = Block(
    "trust_multidimensional",
    "Please answer the following questions on how you perceive climate scientists.\n\n" + SLIDER_INSTRUCTION,
    TRUST_ITEMS,
)

# ---- Secondary outcome blocks (presented in random order) ---------------------
BLOCK_TRUST_POST = Block(
    "trust_single_post",
    "",
    (Item("trust_post_1", "TRUST_POST", "How much do you trust climate scientists?", lo_label="Not at all", hi_label="Very strongly"),),
)
BLOCK_DISTRUST = Block(
    "distrust_single_post",
    "",
    (Item("distrust_1", "DISTRUST", "How much do you distrust climate scientists?", lo_label="Not at all", hi_label="Very strongly"),),
)
BLOCK_FUNDING = Block(
    "funding",
    "",
    (Item("funding_5", "FUNDING_PERCEPTION", "Do you think the federal government is spending too much, too little or about the right amount of money on climate change research?", lo_label="Far too little", hi_label="Far too much", mid_label="About the right amount"),),
)
BLOCK_INST_TRUST = Block(
    "institutional_trust",
    "How much do you trust the following institutions?\n\n" + SLIDER_INSTRUCTION,
    (
        Item("inst_trust_epa_1", "INST_TRUST_EPA", "Environmental Protection Agency (EPA)", lo_label="Not at all", hi_label="Very strongly"),
        Item("inst_trust_nasa_1", "INST_TRUST_NASA", "National Aeronautics and Space Administration (NASA)", lo_label="Not at all", hi_label="Very strongly"),
        Item("inst_trust_noaa_1", "INST_TRUST_NOAA", "National Oceanic and Atmospheric Administration (NOAA)", lo_label="Not at all", hi_label="Very strongly"),
        Item("inst_trust_uni_1", "INST_TRUST_UNIVERSITIES", "Universities and colleges", lo_label="Not at all", hi_label="Very strongly"),
        Item("inst_trust_gov_1", "INST_TRUST_FEDERAL_GOV", "Federal government", lo_label="Not at all", hi_label="Very strongly"),
    ),
)
BLOCK_POLICY_ROLE = Block(
    "scientists_role_in_policy",
    "To what extent do you agree or disagree with the following statements?\n\n" + SLIDER_INSTRUCTION,
    (
        Item("policy_1_1", "POLICY_ROLE_1", "Climate scientists should work closely with policy makers to integrate scientific results into policy-making.", lo_label="Strongly disagree", hi_label="Strongly agree"),
        Item("policy_2_1", "POLICY_ROLE_2", "Climate scientists should actively advocate for specific policies.", lo_label="Strongly disagree", hi_label="Strongly agree"),
        Item("policy_3_1", "POLICY_ROLE_3", "Climate scientists should communicate their findings to policy makers.", lo_label="Strongly disagree", hi_label="Strongly agree"),
        Item("policy_4_1", "POLICY_ROLE_4", "Climate scientists should be more involved in the policy-making process.", lo_label="Strongly disagree", hi_label="Strongly agree"),
    ),
)
DONATION_CHOICES = tuple(
    (f"Don{d}", f"${d}" + (" (keep all $10 for yourself)" if d == 0 else " (donate all $10 to AMS)" if d == 10 else ""), d)
    for d in range(0, 11)
)
BLOCK_DONATION = Block(
    "donation",
    "",
    (
        Item(
            "donation",
            "DONATION",
            "The organization you can choose to allocate real money to is the American Meteorological Society (AMS), "
            "a non-profit, non-partisan society of 12,000 scientists and other professionals that supports climate change "
            "research. With your donation, you help AMS to advance science for the benefit of society.\n\n"
            "You may allocate the $10 in any way you like:\n- keep all $10 for yourself\n- donate all $10 to AMS\n"
            "- or choose any split in between.\n\nOf the $10, how much would you like to donate to the AMS?",
            kind="choice",
            choices=DONATION_CHOICES,
        ),
    ),
    pre_pages=(
        "On the following page, you will have the opportunity to allocate real money between yourself and a "
        "non-profit organization.\n\nAfter data collection is complete, we will randomly select 100 participants "
        "from this study to receive a $10 bonus payment.\n\nIf you are selected, the amount you allocate to yourself "
        "will be paid to you as a bonus, and the amount you allocate to the organization will be donated on your behalf.",
    ),
)
NEWSLETTER_OFFER_PAGE = (
    "Learn more about climate science\n\n"
    "If you’d like to learn more about climate science and solutions, you can subscribe to the newsletter by "
    "climate scientist Katharine Hayhoe. Her newsletter \"Talking Climate\" provides short, accessible updates on "
    "climate science and climate solutions for a general audience.\n\n"
    "Signing up takes less than a minute. Please select the free subscription option — there is no need to "
    "choose a paid version.\n\n"
    "The link below will open the newsletter in a new tab. You can switch back to the current tab and continue the "
    "survey right away.\n\n[ Open Talking Climate newsletter (opens in a new tab) ]\n\n"
    "Note: Subscribing to this newsletter is optional."
)
BLOCK_NEWSLETTER = Block(
    "subscription_newsletter",
    "",
    (
        Item(
            "newsletter",
            "NEWSLETTER",
            "Did you subscribe to the “Talking Climate” newsletter on the previous page?",
            kind="choice",
            choices=(("News1", "Yes", 1), ("News2", "No", 2)),
        ),
    ),
    pre_pages=(NEWSLETTER_OFFER_PAGE,),
)

SECONDARY_BLOCKS: tuple[Block, ...] = (
    BLOCK_TRUST_POST,
    BLOCK_DONATION,
    BLOCK_DISTRUST,
    BLOCK_POLICY_ROLE,
    BLOCK_FUNDING,
    BLOCK_INST_TRUST,
    BLOCK_NEWSLETTER,
)

# ---- Tertiary outcome blocks (presented in random order) ----------------------
BLOCK_BELIEF_POST = Block(
    "belief_post",
    "",
    (Item("belief_post_1", "BELIEF_POST", "How accurate do you think this statement is?\n\n\"Human activities are causing climate change.\"", lo_label="Not at all accurate", hi_label="Extremely accurate"),),
)
BLOCK_CONCERN = Block(
    "climate_change_concern",
    "Please indicate your views on the following questions.\n\n" + SLIDER_INSTRUCTION,
    (
        Item("concern_1_1", "CONCERN_1", "How concerned are you about climate change?", lo_label="Not at all", hi_label="Extremely"),
        Item("concern_2_1", "CONCERN_2", "How serious a problem is climate change?", lo_label="Not at all", hi_label="Extremely"),
        Item("concern_3_1", "CONCERN_3", "Relative to other issues facing the U.S., how important is climate change?", lo_label="The least important issue", hi_label="The most important issue"),
    ),
)
BLOCK_BEHAVIOR = Block(
    "individual_level_behavior",
    "How likely are you to engage in the following activities in the next twelve months?\n\n" + SLIDER_INSTRUCTION,
    (
        Item("individual_meat_1", "BEHAVIOR_MEAT", "Eat less meat", lo_label="Not likely at all", hi_label="Extremely likely"),
        Item("individual_transport_1", "BEHAVIOR_TRANSPORT", "Walk, bicycle, carpool, or take public transportation more often instead of driving a vehicle by yourself", lo_label="Not likely at all", hi_label="Extremely likely"),
        Item("individual_solar_1", "BEHAVIOR_SOLAR", "Install a solar panel", lo_label="Not likely at all", hi_label="Extremely likely"),
        Item("individual_fly_1", "BEHAVIOR_FLY", "Go on less personal (non-business) air travel", lo_label="Not likely at all", hi_label="Extremely likely"),
        Item("individual_talk_1", "BEHAVIOR_TALK", "Talk to friends and family about the importance of climate change", lo_label="Not likely at all", hi_label="Extremely likely"),
        Item("individual_donate_1", "BEHAVIOR_DONATE", "Donate to an environmental NGO", lo_label="Not likely at all", hi_label="Extremely likely"),
    ),
)
BLOCK_POLICY_GENERAL = Block(
    "support_general_climate_policies",
    "",
    (Item("policy_general_1", "POLICY_GENERAL", "How much do you oppose or support the following statement?\n\n\"The U.S. government should do more to reduce global warming.\"", lo_label="Strongly oppose", hi_label="Strongly support"),),
)
BLOCK_POLICY_SPECIFIC = Block(
    "support_specific_climate_policies",
    "How much do you support or oppose the following policies?\n\n" + SLIDER_INSTRUCTION,
    (
        Item("policy_specific_1_1", "POLICY_SPECIFIC_1", "Raising taxes on fossil fuels (e.g., gas, oil, coal)", lo_label="Strongly oppose", hi_label="Strongly support"),
        Item("policy_specific_2_1", "POLICY_SPECIFIC_2", "Expanding infrastructure for public transportation", lo_label="Strongly oppose", hi_label="Strongly support"),
        Item("policy_specific_3_1", "POLICY_SPECIFIC_3", "Increasing the use of sustainable energy such as wind and solar energy", lo_label="Strongly oppose", hi_label="Strongly support"),
        Item("policy_specific_4_1", "POLICY_SPECIFIC_4", "Protecting forested and land areas", lo_label="Strongly oppose", hi_label="Strongly support"),
        Item("policy_specific_5_1", "POLICY_SPECIFIC_5", "Increasing taxes on carbon-intensive foods (e.g., beef and dairy products)", lo_label="Strongly oppose", hi_label="Strongly support"),
        Item("policy_specific_6_1", "POLICY_SPECIFIC_6", "Investing more in green jobs and businesses", lo_label="Strongly oppose", hi_label="Strongly support"),
        Item("policy_specific_7_1", "POLICY_SPECIFIC_7", "Introducing laws to keep waterways and oceans clean", lo_label="Strongly oppose", hi_label="Strongly support"),
    ),
)

TERTIARY_BLOCKS: tuple[Block, ...] = (
    BLOCK_BELIEF_POST,
    BLOCK_CONCERN,
    BLOCK_BEHAVIOR,
    BLOCK_POLICY_GENERAL,
    BLOCK_POLICY_SPECIFIC,
)

# ---- Optional pre-treatment items (not scored; realism only) ------------------
BLOCK_BELIEF_PRE = Block(
    "belief_pre",
    "",
    (Item("belief_pre", "BELIEF_PRE", "How accurate do you think this statement is?\n\n\"Human activities are causing climate change.\"", lo_label="Not at all accurate", hi_label="Extremely accurate"),),
)
BLOCK_TRUST_PRE = Block(
    "trust_single_pre",
    "",
    (Item("trust_pre", "TRUST_PRE", "How much do you trust climate scientists?", lo_label="Not at all", hi_label="Very strongly"),),
)
PRETREATMENT_BLOCKS: tuple[Block, ...] = (BLOCK_BELIEF_PRE, BLOCK_TRUST_PRE)

ALL_OUTCOME_BLOCKS: tuple[Block, ...] = (BLOCK_TRUST,) + SECONDARY_BLOCKS + TERTIARY_BLOCKS
OUTCOME_ITEMS: tuple[Item, ...] = tuple(it for b in ALL_OUTCOME_BLOCKS for it in b.items)
OUTCOME_ITEM_KEYS: tuple[str, ...] = tuple(it.key for it in OUTCOME_ITEMS)
ITEM_BY_KEY: dict[str, Item] = {it.key: it for it in OUTCOME_ITEMS}
ITEM_BY_TITLE: dict[str, Item] = {it.title: it for it in OUTCOME_ITEMS}

# --------------------------------------------------------------------------- #
# 5. Tier-1 submission schema
# --------------------------------------------------------------------------- #

TIER1_COLUMNS: list[str] = [
    "profile_id", "condition",
    "gender", "age_band", "race", "education", "income", "party",
    "trust_multidimensional",
    "trust_competence_1", "trust_competence_2", "trust_competence_3",
    "trust_integrity_1", "trust_integrity_2", "trust_integrity_3",
    "trust_benevolence_1", "trust_benevolence_2", "trust_benevolence_3",
    "trust_openness_1", "trust_openness_2", "trust_openness_3",
    "trust_post", "distrust_post", "funding_perceptions",
    "policy_role_mean", "inst_trust_mean",
    "belief_post", "concern_mean", "policy_general",
    "policy_specific_mean", "behavior_mean",
    "donation_ams", "newsletter_signup",
]

# raw qualtrics label -> target label (mirror of scripts/lib/clean_lib.R .rename_map)
RAW_TO_TARGET: dict[str, str] = {
    "trust_competent_1": "trust_competence_1",
    "trust_intelligent_1": "trust_competence_2",
    "trust_qualified_1": "trust_competence_3",
    "trust_honest_1": "trust_integrity_1",
    "trust_ethical_1": "trust_integrity_2",
    "trust_sincere_1": "trust_integrity_3",
    "trust_concerned_1": "trust_benevolence_1",
    "trust_improve_1": "trust_benevolence_2",
    "trust_considerate_1": "trust_benevolence_3",
    "trust_feedback_1": "trust_openness_1",
    "trust_transparent_1": "trust_openness_2",
    "trust_attention_1": "trust_openness_3",
    "trust_post_1": "trust_post",
    "distrust_1": "distrust_post",
    "donation": "donation_ams",
    "newsletter": "newsletter_signup",
    "funding_5": "funding_perceptions",
    "policy_1_1": "policy_role_1", "policy_2_1": "policy_role_2",
    "policy_3_1": "policy_role_3", "policy_4_1": "policy_role_4",
    "inst_trust_epa_1": "inst_trust_epa", "inst_trust_nasa_1": "inst_trust_nasa",
    "inst_trust_noaa_1": "inst_trust_noaa", "inst_trust_uni_1": "inst_trust_universities",
    "inst_trust_gov_1": "inst_trust_federal_gov",
    "belief_post_1": "belief_post",
    "concern_1_1": "concern_1", "concern_2_1": "concern_2", "concern_3_1": "concern_3",
    "policy_general_1": "policy_general",
    "policy_specific_1_1": "policy_specific_1", "policy_specific_2_1": "policy_specific_2",
    "policy_specific_3_1": "policy_specific_3", "policy_specific_4_1": "policy_specific_4",
    "policy_specific_5_1": "policy_specific_5", "policy_specific_6_1": "policy_specific_6",
    "policy_specific_7_1": "policy_specific_7",
    "individual_meat_1": "behavior_meat", "individual_transport_1": "behavior_transport",
    "individual_solar_1": "behavior_solar", "individual_fly_1": "behavior_fly",
    "individual_talk_1": "behavior_talk", "individual_donate_1": "behavior_donate",
}

# Raw export column order (mirrors raw_data_deposit/example_raw_export.csv)
RAW_EXPORT_COLUMNS: list[str] = [
    "StartDate", "EndDate", "Status", "IPAddress", "Progress", "Duration (in seconds)", "Finished",
    "RecordedDate", "ResponseId", "RecipientLastName", "RecipientFirstName", "RecipientEmail",
    "ExternalReference", "LocationLatitude", "LocationLongitude", "DistributionChannel", "UserLanguage",
    "condition", "profile_id", "gender", "year_birth", "race", "education", "income", "party",
    "donation", "newsletter",
    "trust_competent_1", "trust_intelligent_1", "trust_qualified_1", "trust_honest_1", "trust_ethical_1",
    "trust_sincere_1", "trust_concerned_1", "trust_improve_1", "trust_considerate_1", "trust_feedback_1",
    "trust_transparent_1", "trust_attention_1", "trust_post_1", "distrust_1", "funding_5",
    "policy_1_1", "policy_2_1", "policy_3_1", "policy_4_1",
    "inst_trust_epa_1", "inst_trust_nasa_1", "inst_trust_noaa_1", "inst_trust_uni_1", "inst_trust_gov_1",
    "belief_post_1", "concern_1_1", "concern_2_1", "concern_3_1", "policy_general_1",
    "policy_specific_1_1", "policy_specific_2_1", "policy_specific_3_1", "policy_specific_4_1",
    "policy_specific_5_1", "policy_specific_6_1", "policy_specific_7_1",
    "individual_meat_1", "individual_transport_1", "individual_solar_1", "individual_fly_1",
    "individual_talk_1", "individual_donate_1",
]

# --------------------------------------------------------------------------- #
# 6. Survey transition texts (verbatim from survey.json)
# --------------------------------------------------------------------------- #

TRANSITION_TO_STUDY = (
    "Thank you, you have qualified for the study.\n\n"
    "In the following sections, we’re interested in your opinion about climate change and climate scientists.\n\n"
    "Climate scientists study changes in the Earth's climate over time and how they might affect the planet in the "
    "future. Please keep this definition in mind when filling out this study."
)
TRANSITION_PRE_TO_TREATMENT = (
    "You are now moving on to a different section of the study.\n"
    "Please pay close attention to the information you will be provided.\n"
    "Thank you."
)
TRANSITION_TREATMENT_TO_POST = (
    "You are now moving on to the final section of the study.\n"
    "Please answer the following questions to the best of your ability.\n"
    "Thank you."
)
