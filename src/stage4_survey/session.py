"""
Survey session model: what a synthetic respondent sees, in what order, and how it is turned
into PoSSUM-style annotator prompts.

Concepts
--------
Element   : one thing on the respondent's screen — a display ``Page`` (text only) or an
            ``ItemPage`` (a page holding survey items to be answered).
Step      : a consecutive run of elements answered in ONE model call. Steps exist because
            some conditions elicit an answer *before* showing feedback (Funding, High public
            trust, Consensus); the answer from step k is shown to the model in step k+1.
SessionPlan: the whole ordered list of elements for one (agent x condition), cut into steps,
            plus bookkeeping (control filler used, block order, extreme-weather case, ...).

The flow mirrors survey/survey.json:
    [pre-treatment: belief_pre / trust_pre (optional, elicited once per agent)]
    transition -> condition pages -> transition ->
    trust_multidimensional (always first) -> 7 secondary blocks (random order) ->
    5 tertiary blocks (random order)

Prompt framing follows PoSSUM (Cerina & Duch): the model is a neutral annotator asked to
select, for every TITLE, the answer *this user* most likely gave, with a 0-100 speculation
score and a short explanation, in a strictly structured output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from common import codebook as cb
from common.codebook import Block, Item

STIMULI_DIR = Path(__file__).resolve().parent / "stimuli"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def stim(name: str) -> str:
    return (STIMULI_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def prompt_file(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# Elements
# --------------------------------------------------------------------------- #

@dataclass
class Page:
    text: str
    label: str = ""          # bookkeeping label (e.g. "stimulus_p1")


@dataclass
class ItemPage:
    block: Block
    label: str = ""

    @property
    def items(self) -> tuple[Item, ...]:
        return self.block.items


Element = Page | ItemPage


@dataclass
class Step:
    elements: list = field(default_factory=list)
    name: str = ""
    # True for the battery calls of an "independent" battery_mode: the step is prompted as a fresh
    # continuation of the treatment transcript and its answers are never shown to any other step
    # (no anchoring on the agent's own earlier outcome answers). Sequential steps still see all
    # earlier non-independent steps.
    independent: bool = False

    @property
    def items(self) -> list[Item]:
        return [it for e in self.elements if isinstance(e, ItemPage) for it in e.items]


@dataclass
class SessionPlan:
    profile_id: str
    condition: str
    codename: str
    steps: list[Step]
    meta: dict = field(default_factory=dict)

    @property
    def all_items(self) -> list[Item]:
        return [it for s in self.steps for it in s.items]


# --------------------------------------------------------------------------- #
# Elicitation items used inside conditions (not scored, but recorded)
# --------------------------------------------------------------------------- #

FUNDING_VALUE_ITEMS = (
    Item("funding_value_1", "FUNDING_VALUES_1", "Everyone should be held to the same standards of honesty and fairness.", lo_label="Strongly disagree", hi_label="Strongly agree"),
    Item("funding_value_2", "FUNDING_VALUES_2", "It is important that taxpayer-funded programs show exactly how the taxmoney is spent.", lo_label="Strongly disagree", hi_label="Strongly agree"),
    Item("funding_value_3", "FUNDING_VALUES_3", "Corporations have too much influence on what gets researched.", lo_label="Strongly disagree", hi_label="Strongly agree"),
    Item("funding_value_4", "FUNDING_VALUES_4", "Some people in powerful positions push certain ideas not because they’re true, but because they fit their political or financial interests.", lo_label="Strongly disagree", hi_label="Strongly agree"),
)
FUNDING_PAID = Item("funding_paid", "FUNDING_STATEMENT_PAID", "“Climate scientists are paid to support certain climate policies.”", lo_label="Strongly disagree", hi_label="Strongly agree")
FUNDING_FEDERAL = Item("funding_federal", "FUNDING_STATEMENT_FEDERAL", "“The federal government allocates significant resources to climate change research.”", lo_label="Strongly disagree", hi_label="Strongly agree")
FUNDING_PRIVATE = Item("funding_private", "FUNDING_STATEMENT_PRIVATE", "“Climate scientists receive large amounts of private research funding.”", lo_label="Strongly disagree", hi_label="Strongly agree")

TRUST_PCT_ESTIMATE = Item("public_trust_estimate", "PUBLIC_TRUST_ESTIMATE",
                          "Please provide your best estimate: What percentage of Americans trust climate scientists to provide full and accurate information on climate change?",
                          lo_label="0%", hi_label="100%")

CONSENSUS_ITEMS = {
    "human": Item("consensus_human", "CONSENSUS_HUMAN", "What percentage of scientists do you think agree with the statement:\n“Human activities are the primary cause of global warming since the mid-20th century.”", lo_label="0% of scientific agreement", hi_label="100% of scientific agreement"),
    "co2": Item("consensus_co2", "CONSENSUS_CO2", "What percentage of scientists do you think agree with the statement:\n“Increasing carbon dioxide in the atmosphere warms the planet.”", lo_label="0% of scientific agreement", hi_label="100% of scientific agreement"),
    "year": Item("consensus_year", "CONSENSUS_NETZERO_2085", "What percentage of scientists do you think agree with the statement:\n“The world will reach net-zero CO₂ emissions before 2085.”", lo_label="0% of scientific agreement", hi_label="100% of scientific agreement"),
}
CONSENSUS_FEEDBACK = {"human": "consensus_fb_human", "co2": "consensus_fb_co2", "year": "consensus_fb_year"}

ELICITATION_ITEMS: tuple[Item, ...] = FUNDING_VALUE_ITEMS + (FUNDING_PAID, FUNDING_FEDERAL, FUNDING_PRIVATE, TRUST_PCT_ESTIMATE) + tuple(CONSENSUS_ITEMS.values())
ELICITATION_KEYS = tuple(it.key for it in ELICITATION_ITEMS)


def _single(item: Item, intro: str = "") -> ItemPage:
    return ItemPage(Block(item.key, intro, (item,)), label=item.key)


# --------------------------------------------------------------------------- #
# Condition scripts
# --------------------------------------------------------------------------- #

def _pages(*names: str, prefix: str = "stimulus") -> list[Page]:
    return [Page(stim(n), label=f"{prefix}:{n}") for n in names]


def condition_elements(condition: str, rng: random.Random, agent: dict, meta: dict) -> list[list]:
    """Return the treatment part of the session as a list of *steps* (each a list of elements).

    All but the last step end with an item that must be answered before the next page is shown.
    The last step's elements are followed (by the caller) by the outcome battery.
    """
    if condition == cb.CONTROL:
        filler = rng.choice(cb.CONTROL_CODENAMES)
        meta["control_filler"] = filler
        return [_pages(filler.replace(" ", "_"))]

    if condition == "Corporate reliance":
        return [_pages("corporate_reliance_p1", "corporate_reliance_p2")]
    if condition == "Social justice":
        return [_pages("social_justice")]
    if condition == "Interview Prof. Maraun":
        return [_pages("interview_maraun")]
    if condition == "Oil industry misinformation":
        return [_pages("oil_industry_misinformation")]
    if condition == "Measurement & modeling (1)":
        return [_pages("measurement_modeling_1")]
    if condition == "Former skeptics":
        return [_pages("former_skeptics")]
    if condition == "Measurement & modeling (2)":
        return [_pages("measurement_modeling_2")]
    if condition == "Peer-review":
        return [_pages("peer_review")]
    if condition == "Scientist community helpers":
        return [_pages(*[f"scientist_community_helpers_p{i}" for i in range(1, 5)])]
    if condition == "Portrait Prof. Cherry":
        return [_pages("portrait_cherry")]
    if condition == "Model accuracy":
        return [_pages("model_accuracy")]
    if condition == "Interview Prof. Sebille":
        return [_pages("interview_sebille")]

    if condition == "Funding":
        step1 = [
            ItemPage(Block("funding_values",
                           "Please indicate how much you agree or disagree with the following statements.\n\n" + cb.SLIDER_INSTRUCTION,
                           FUNDING_VALUE_ITEMS), label="funding_values"),
            Page(stim("funding_p3_thanks"), label="stimulus:funding_p3_thanks"),
            _single(FUNDING_PAID, "How much do you agree or disagree with the following statement?"),
        ]
        step2 = [Page(stim("funding_p5_feedback_paid"), label="stimulus:funding_p5_feedback_paid"),
                 _single(FUNDING_FEDERAL, "How much do you agree or disagree with the following statement?")]
        step3 = [Page(stim("funding_p7_feedback_federal"), label="stimulus:funding_p7_feedback_federal"),
                 _single(FUNDING_PRIVATE, "How much do you agree or disagree with the following statement?")]
        step4 = [Page(stim("funding_p9_feedback_private"), label="stimulus:funding_p9_feedback_private"),
                 Page(stim("funding_p10_closing"), label="stimulus:funding_p10_closing")]
        return [step1, step2, step3, step4]

    if condition == "High public trust":
        step1 = [_single(TRUST_PCT_ESTIMATE)]
        step2 = [Page(stim("high_public_trust_p2_feedback"), label="stimulus:high_public_trust_p2_feedback")]
        return [step1, step2]

    if condition == "Consensus":
        order = ["human", "co2", "year"]
        rng.shuffle(order)  # survey.json: BlockRandomizer over the three item blocks
        meta["consensus_order"] = order
        intro = Page(stim("consensus_p1_intro") + "\n\n" + stim("consensus_p1_instructions"), label="stimulus:consensus_intro")
        steps = []
        for i, key in enumerate(order):
            elems = [] if i == 0 else [Page(stim(CONSENSUS_FEEDBACK[order[i - 1]]), label=f"stimulus:{CONSENSUS_FEEDBACK[order[i-1]]}")]
            if i == 0:
                elems.append(intro)
            elems.append(_single(CONSENSUS_ITEMS[key]))
            steps.append(elems)
        steps.append([Page(stim(CONSENSUS_FEEDBACK[order[-1]]), label=f"stimulus:{CONSENSUS_FEEDBACK[order[-1]]}"),
                      Page(stim("consensus_p_end"), label="stimulus:consensus_p_end")])
        return steps

    if condition == "Extreme weather predictions":
        state = agent.get("assigned_state")
        case = cb.state_to_case(state)
        meta["extreme_weather_case"] = case
        meta["extreme_weather_state"] = state or "Prefer not to say"
        state_q = ("Which U.S. state do you currently live in?\n"
                   "(You may choose not to answer. If so, please select “Prefer not to say.”)\n"
                   f"→ The user selected: {state if case != 4 else 'Prefer not to say'}")
        if case == 4:
            intro = stim("extreme_weather_p2_intro_generic")
        else:
            intro = stim("extreme_weather_p2_intro_state").replace("[STATE]", state).replace("[CASE]", cb.CASE_LABEL[case])
        case_file = {1: "extreme_weather_case1_flood", 2: "extreme_weather_case2_wildfire",
                     3: "extreme_weather_case3_winter", 4: "extreme_weather_case4_generic"}[case]
        return [[Page(state_q, label="extreme_weather_state_question"),
                 Page(intro, label=f"stimulus:extreme_weather_intro_case{case}"),
                 Page(stim(case_file), label=f"stimulus:{case_file}")]]

    raise ValueError(f"unknown condition {condition!r}")


# --------------------------------------------------------------------------- #
# Battery + full plan
# --------------------------------------------------------------------------- #

def battery_elements(rng: random.Random, meta: dict) -> tuple[list, list, list]:
    """(primary, secondary, tertiary) element lists, secondary/tertiary block order randomized."""
    sec = list(cb.SECONDARY_BLOCKS)
    ter = list(cb.TERTIARY_BLOCKS)
    rng.shuffle(sec)
    rng.shuffle(ter)
    meta["secondary_block_order"] = [b.key for b in sec]
    meta["tertiary_block_order"] = [b.key for b in ter]

    def to_elems(blocks):
        out = []
        for b in blocks:
            for pp in b.pre_pages:
                out.append(Page(pp, label=f"{b.key}:pre_page"))
            out.append(ItemPage(b, label=b.key))
        return out

    return to_elems([cb.BLOCK_TRUST]), to_elems(sec), to_elems(ter)


PRETREATMENT_BLOCK_BY_KEY = {b.key: b for b in cb.PRETREATMENT_BLOCKS}


def pretreatment_step(rng: random.Random, block_order: list[str] | None = None) -> Step:
    """The once-per-agent pre-treatment step (transition page + belief_pre / trust_pre in random order)."""
    if block_order is None:
        blocks = list(cb.PRETREATMENT_BLOCKS)
        rng.shuffle(blocks)
    else:
        blocks = [PRETREATMENT_BLOCK_BY_KEY[k] for k in block_order]
    elems = [Page(cb.TRANSITION_TO_STUDY, label="transition_to_study")]
    for b in blocks:
        elems.append(ItemPage(b, label=b.key))
    return Step(elems, name="pretreatment")


BATTERY_MODES = ("sequential", "independent_blocks", "independent_items")


def build_session_plan(agent: dict, condition: str, *, seed: int, battery_steps: int = 1,
                       pretreatment: dict | None = None, battery_mode: str = "sequential") -> SessionPlan:
    """Build the ordered, stepped session for one agent x condition.

    ``battery_mode`` -- how the post-treatment outcome battery is answered:
      "sequential"          one transcript; the battery follows the treatment in the same call
                            (``battery_steps`` 1) or in 3 consecutive calls that see each other's
                            answers (``battery_steps`` 3). The registered default.
      "independent_blocks"  the treatment part runs exactly as in sequential mode; then EVERY
                            battery block (trust, donation, ..., one survey page each) is asked in
                            its own call that sees the treatment transcript but NONE of the other
                            blocks' answers. Block order is still the randomized survey order.
      "independent_items"   as above, but one call per single outcome item.
    Independent modes are the anchoring mitigation piloted in src/testbed (RESULTS.md 2026-08-24).

    ``pretreatment`` — {"answers": {item_key: answer}, "block_order": [block keys]} as produced by
    the once-per-agent pre-treatment elicitation (run_survey.py). If given, those items are rendered
    as already answered at the start of the transcript.
    """
    rng = random.Random(seed)
    meta: dict = {"seed": seed}
    codename = cb.CONDITION_TO_CODENAME.get(condition, "control")

    prefix: list = []
    if pretreatment:
        pre = pretreatment_step(rng, pretreatment.get("block_order"))
        prefix.extend(pre.elements)
    else:
        prefix.append(Page(cb.TRANSITION_TO_STUDY, label="transition_to_study"))
    prefix.append(Page(cb.TRANSITION_PRE_TO_TREATMENT, label="transition_pre_to_treatment"))

    cond_steps = condition_elements(condition, rng, agent, meta)
    if condition == cb.CONTROL:
        codename = meta["control_filler"]

    prim, sec, ter = battery_elements(rng, meta)
    post_transition = Page(cb.TRANSITION_TREATMENT_TO_POST, label="transition_treatment_to_post")

    if battery_mode not in BATTERY_MODES:
        raise ValueError(f"battery_mode must be one of {BATTERY_MODES}, got {battery_mode!r}")
    steps: list[Step] = []
    # treatment steps: the first one is preceded by the (already answered) prefix
    for i, elems in enumerate(cond_steps):
        s = Step(list(elems), name=f"treatment_{i + 1}")
        steps.append(s)
    # battery
    if battery_mode != "sequential":
        return _independent_battery_plan(agent, condition, codename, meta, prefix, pretreatment, steps,
                                         prim + sec + ter, post_transition, battery_mode)
    if battery_steps == 1:
        steps[-1].elements.extend([post_transition] + prim + sec + ter)
        steps[-1].name = "battery" if len(cond_steps) == 1 else steps[-1].name + "+battery"
    else:
        steps[-1].elements.extend([post_transition] + prim)
        steps[-1].name = steps[-1].name + "+battery_primary" if len(cond_steps) > 1 else "battery_primary"
        steps.append(Step(sec, name="battery_secondary"))
        steps.append(Step(ter, name="battery_tertiary"))
    # steps that contain no items (pure display) are merged into the following step
    merged: list[Step] = []
    carry: list = []
    for s in steps:
        if not s.items:
            carry.extend(s.elements)
            continue
        s.elements = carry + s.elements
        carry = []
        merged.append(s)
    assert not carry, "session ended with display-only pages"

    plan = SessionPlan(profile_id=agent["profile_id"], condition=condition, codename=codename, steps=merged, meta=meta)
    plan.meta["prefix_elements"] = prefix  # rendered as already-seen material before step 1
    plan.meta["pretreatment_answers"] = dict((pretreatment or {}).get("answers") or {})
    return plan


def _independent_battery_plan(agent: dict, condition: str, codename: str, meta: dict, prefix: list,
                              pretreatment: dict | None, treat_steps: list[Step], battery: list,
                              post_transition: Page, battery_mode: str) -> SessionPlan:
    """Independent battery modes: treatment steps as in sequential mode, then one independent step
    per battery block (or item). Display-only treatment pages that would be merged into the battery
    call in sequential mode (control filler, single-page stimuli) are instead shared by every
    independent step, so each call still shows the full treatment the agent read."""
    merged_treat: list[Step] = []
    carry: list = []
    for st in treat_steps:
        if not st.items:
            carry.extend(st.elements)
            continue
        st.elements = carry + st.elements
        carry = []
        merged_treat.append(st)
    shared = carry  # display-only pages after the last item-bearing treatment step (all of control)

    units: list[tuple[str, list]] = []
    for elem in battery:
        if isinstance(elem, Page):
            units.append(("_pre", [elem]))  # a block's pre-page: shown in that block's call only
        elif battery_mode == "independent_blocks":
            units.append((elem.block.key, [elem]))
        else:
            for it in elem.items:
                units.append((it.key, [ItemPage(Block(it.key, elem.block.intro, (it,)), label=it.key)]))
    steps: list[Step] = []
    pending_pre: list = []
    for key, elems in units:
        if key == "_pre":
            pending_pre.extend(elems)
            continue
        steps.append(Step(shared + [post_transition] + pending_pre + elems, name=f"battery:{key}", independent=True))
        pending_pre = []
    assert not pending_pre, "battery ended with a display-only page"

    plan = SessionPlan(profile_id=agent["profile_id"], condition=condition, codename=codename,
                       steps=merged_treat + steps, meta=meta)
    plan.meta["battery_mode"] = battery_mode
    plan.meta["prefix_elements"] = prefix
    plan.meta["pretreatment_answers"] = dict((pretreatment or {}).get("answers") or {})
    return plan


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_item_scale(item: Item) -> str:
    if item.kind == "slider":
        mid = f', 50 = "{item.mid_label}"' if item.mid_label else ""
        return (f'Slider from 0 = "{item.lo_label}"{mid} to 100 = "{item.hi_label}". '
                "Answer with an integer between 0 and 100.")
    lines = [f"{sym}) {label}" for sym, label, _ in item.choices]
    return "\n".join(lines) + "\nAnswer with exactly one of the symbols above."


def render_answered_item(item: Item, answer) -> str:
    if item.kind == "slider":
        scale = f'(slider, 0 = "{item.lo_label}" … 100 = "{item.hi_label}")'
        return f"Q: {item.text}\n   {scale}\n   → The user answered: {answer}"
    label = next((lab for sym, lab, _ in item.choices if sym == answer), str(answer))
    return f"Q: {item.text}\n   → The user answered: {label}"


def render_seen_elements(elements: list, answers: dict) -> str:
    """Render already-shown pages, with any items shown together with the user's answers."""
    out = []
    for e in elements:
        if isinstance(e, Page):
            out.append("--- Page ---\n" + e.text)
        else:
            parts = [e.block.intro] if e.block.intro else []
            for it in e.items:
                ans = answers.get(it.key, "(no answer recorded)")
                parts.append(render_answered_item(it, ans))
            out.append("--- Page ---\n" + "\n\n".join(parts))
    return "\n\n".join(out)


def render_task_elements(elements: list) -> str:
    """Render the current step: pages verbatim, items with TITLE tokens the model must answer."""
    out = []
    for e in elements:
        if isinstance(e, Page):
            out.append("--- Page ---\n" + e.text)
        else:
            parts = [e.block.intro] if e.block.intro else []
            for it in e.items:
                parts.append(f"{it.title}:\n{it.text}\n{render_item_scale(it)}")
            out.append("--- Page ---\n" + "\n\n".join(parts))
    return "\n\n".join(out)


AGE_BAND_PHRASE = {"18-29": "18 to 29 years old", "30-44": "30 to 44 years old",
                   "45-59": "45 to 59 years old", "60+": "60 years old or older"}


def render_demographics(agent: dict, include_state: bool = True) -> str:
    d = agent["demographics"]
    lines = [
        f"gender: {d['gender']}",
        f"age: {AGE_BAND_PHRASE.get(d['age_band'], d['age_band'])}",
        f"race / ethnicity: {d['race']}",
        f"highest level of education completed: {d['education']}",
        f"total yearly family/household income before taxes: {d['income']}",
        f"party identification: {d['party']}",
    ]
    if include_state:
        st = agent.get("assigned_state")
        lines.append(f"US state of residence: {st if st else 'unknown'}")
    return "\n".join(lines)


def build_step_prompt(agent: dict, plan: SessionPlan, step_index: int, answers_so_far: dict,
                      *, include_demographics: bool = True, explanation_style: str = "brief") -> str:
    """Compose the full user-message for one step. Returns the prompt string."""
    step = plan.steps[step_index]
    seen: list = list(plan.meta.get("prefix_elements", []))
    for s in plan.steps[:step_index]:
        if s.independent:
            continue  # independent battery calls never see each other's pages or answers
        seen.extend(s.elements)
    # display-only pages at the start of the current step (e.g. the stimulus) are material the
    # user has read BEFORE answering -> they belong to the transcript, not to the item block
    first_item = next((i for i, e in enumerate(step.elements) if isinstance(e, ItemPage)), 0)
    seen.extend(step.elements[:first_item])
    task_elements = step.elements[first_item:]
    all_answers = dict(plan.meta.get("pretreatment_answers", {}))
    all_answers.update(answers_so_far)

    persona = prompt_file("persona_header")
    if include_demographics:
        persona += "\n\n" + prompt_file("demographics_intro").format(demographics=render_demographics(agent))
    persona += "\n\n" + prompt_file("activity_intro") + "\n" + agent["profile_text"]

    session_intro = prompt_file("session_intro")
    seen_txt = render_seen_elements(seen, all_answers)
    task_txt = render_task_elements(task_elements)

    instr = prompt_file("task_instructions")
    expl = {"long": prompt_file("explanation_long"),
            "brief": prompt_file("explanation_brief"),
            "none": prompt_file("explanation_none")}[explanation_style]
    fmt = prompt_file("output_format")

    n_titles = len(step.items)
    titles = ", ".join(it.title for it in step.items)
    return "\n\n".join([
        prompt_file("preamble"),
        "=== USER DATA ===",
        persona,
        "=== SURVEY SESSION ===",
        session_intro,
        seen_txt,
        "=== YOUR TASK ===",
        instr,
        "BEGIN SURVEY ITEMS",
        task_txt,
        "END SURVEY ITEMS",
        expl,
        fmt,
        f"There are {n_titles} titles to answer: {titles}.\nYOU MUST GIVE AN ANSWER FOR EVERY TITLE!",
    ])


def build_cleanup_prompt(raw_output: str, items: list[Item]) -> str:
    return prompt_file("cleanup_strict").format(
        titles=", ".join(it.title for it in items),
        raw_output=raw_output,
    )
