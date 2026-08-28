

# Silicon Sample Benchmark — submission template

> ## ⏰ Prediction lock: **August 31, 2026** (hard deadline)
>
> Deposit your predictions on Zenodo and email the DOI + fingerprints by this date. No revisions after.
> Deposit **within the last four days (August 28–31)** — see the *Deposit* step for why.

This repository **is** a submission to the [Silicon Sample Benchmark](https://janpfander.github.io/llm_predictions_megastudy/):
a multi-team benchmark of AI approaches for predicting the results of a behavioral megastudy on
trust in climate scientists, *before* the human data are revealed.

**This repository is meant to be worked in, not just filled out.** The ideal scenario: your *whole*
simulation lives here — your code, prompts, profile construction, and intermediate data, alongside the
predictions they produce. Clone it (or click **“Use this template”** on GitHub) and build your pipeline
inside it. If that isn’t possible — proprietary code, or work that already lives elsewhere — you can
instead treat it as a **deposit repository**: just drop in your prediction file(s), complete
`registration.md`, and release to Zenodo. Both are valid, but **having everything in one place is
strongly preferred** — and you can still keep proprietary parts private by **gitignoring** them (see
the disclosure policy).

The repo ships with a **random example submission** so a fresh clone is already valid — replace it with
your own.

> The numbers in the example are random placeholders with **no real effects** — for format only.

> **What we need from you:** (1) your **prediction file(s)** and (2) a completed
> **`registration.md`**, released together (see *Deposit*) — those two are all we read; everything else
> you keep in the repo is there for transparency and reproducibility.
> The benchmark ships the survey, codebook, validator, and intervention texts — but **no
> participant / profile pool**. You construct your own synthetic respondents; there is no pool
> to wait for. The `profile_id` column is simply a unique id you assign to each respondent.
> **Any language is fine.** The helper scripts happen to be in R, but nothing about a submission
> requires R — build your predictions and the submission file(s) in Python, Julia, or whatever you
> like. The `make` commands (`clean`, `manifest`, `check`) are **optional conveniences** — they help
> you build and self-validate those files, but you may produce them however you like (e.g. your own
> cleaning script driven by `codebook.csv`). We can’t guarantee that malformed submissions will be
> scored, so running `make check` first is strongly recommended — but it is not required.
> The one thing you **do** owe us regardless of tooling is a **SHA-256 fingerprint** of each
> prediction file, recorded in `metadata.json` and emailed at deposit (see *Deposit*): `make manifest`
> computes it for you, but if you skip the helpers you must generate it yourself (e.g.
> `shasum -a 256 <file>`).

## What counts as a submission

**One submission = one entry = one repository = one Zenodo deposit.** An *entry* is one method’s
complete set of predictions at a single tier; this repo holds exactly one, and its `metadata.json`
describes that one entry.

- **`primary` vs `secondary-k`** (the `entry` field in `metadata.json`) is *your own* ranking of your
  entries. Label the entry you consider your team’s single best effort `primary`; label alternatives
  `secondary-1`, `secondary-2`, … The organizers don’t assign this — you do. Every entry is scored
  in the main analyses; the primary designation feeds the primaries-only robustness analysis.
- **One tier per entry.** A repo is Tier 1 *or* 2 *or* 3; `metadata.json`’s `tier` and the prediction
  file name(s) must agree. **Tier 1 is preferred:** individual-level data is scored on *every*
  analysis, so a Tier-1 entry already yields the Tier-2 and Tier-3 metrics — you do **not** submit the
  same method again at a lower tier to get them.
- **Submitting several entries** — **at most three entries per tier per team** (so up to nine in
  total). An extra entry can be a genuinely different approach (e.g. a simulation vs. a direct
  forecast) or a structured variation of one approach (e.g. the same model with different amounts of
  individual-level information). Clone this template **once per entry**, fill each in independently,
  deposit each to Zenodo, and email **all** the deposit DOIs and file fingerprints together. Each
  entry being its own repo, entries may differ freely in tier and disclosure class. **Every entry
  enters all main analyses (the cross-team field statistics and the leaderboard)**; the per-tier cap
  limits how much any single team can shape the field distribution. If there are good reasons to
  systematically vary some aspect of an approach, more entries may be granted on request, but only
  three per tier enter the main analyses. Exactly one of your entries — across all tiers — is
  `primary`: a robustness analysis reruns the main results on primary entries only, one per team.
  In figures and the leaderboard, entries appear under neutral submission labels, with a
  table mapping each label to its team. The most useful extra entries vary one factor and hold the
  rest fixed.
- **Minimum sample size (Tier 1): 500 per intervention, 1,000 in control.** That is the size of
  the human half every submission is scored against (the benchmark preregistration’s *precision
  requirement*) — below it, your effect estimates are noisier than the reference for reasons
  unrelated to your method. Going well beyond the floor is encouraged: synthetic respondents are
  cheap, and a larger pool makes your estimates reflect the method rather than a lucky draw. Beyond
  precision there is no advantage — only point estimates are scored, so a huge pool stabilizes
  your estimates but cannot buy a better score. `make check` warns
  when a file is below the floor.

## Quickstart

Your job, end to end: **predict** the study’s results with your AI approach, **package** those
predictions into the file(s) this repo expects, **describe** how you made them, and **deposit** the
repo to Zenodo before the lock. The steps below walk one entry through that; the engine
(`make clean` / `make manifest` / `make check`) handles the mechanical parts.

1.  **Get your own copy** — “Use this template” on GitHub, or `git clone` and re-init.

2.  **See what you’re predicting.** Read the survey in `survey/` and the variable dictionary in
    `codebook.csv`. You’re predicting **13 outcomes** across **17 conditions** (control + 16
    interventions). Pick your **tier** — 1 = individual responses, 2 = per-condition cells, 3 = effects
    vs. control — one tier per entry.

3.  **Run your simulation.** Generate your predictions with any AI-based approach. You’re forecasting
    **blind** — the human results aren’t revealed until after the lock — so your method must not seek
    out or rely on any outcome data from this study (including pilots). You’ll attest to this in
    `registration.md`. Everything else about how you get there is yours to design.

    > **Budgeting compute (Tier 1).** A whole-session respondent sees one stimulus text (roughly
    > 300–900 words) plus ~90 items — on the order of 5–15k input tokens and 1–3k output tokens per
    > respondent, depending on how many calls you split the session into (per-item calls multiply the
    > input cost, since context is re-sent each call). A minimum-size run (500 per intervention +
    > 1,000 in control = 9,000 respondents) is then very roughly 50–130M input tokens; providers’
    > batch APIs typically halve the price. These are planning numbers — the registration form asks
    > you to report call counts, total tokens, and cost.

4.  **Turn your output into the submission file(s):**

    - **Tier 1 (individual-level).** Your simulation yields raw per-respondent answers in the survey’s
      Qualtrics format. Drop that export into `raw_data_deposit/` and run **`make clean`** — it converts
      the Qualtrics columns into the analysis-ready `predictions/<team_id>_T1_<entry>_v1.csv` and
      fingerprints it. (`make clean` names the file after you, so set `team_id` — assigned at
      registration and sent to your team by email, see *Prediction file naming* — and `entry` in
      `metadata.json` first — the rest of the metadata can wait for step 5.) Full walkthrough:
      [Tier 1: clean your raw data](#tier-1-clean-your-raw-data).
    - **Tier 2 / 3.** Write the cell- or effect-level CSV(s) straight into `predictions/`, copying the
      shape of the matching `example_*` file, then run **`make manifest`** to fingerprint them.

5.  **Describe your method.** Complete `metadata.json` (models, disclosure class, `code_repository`, …)
    and fill in `registration.md` (the reporting checklist; ★ items must be public).

6.  **Swap out the examples.** Delete every `example_*` file in `predictions/` and the shipped
    `raw_data_deposit/example_raw_export.csv`, leaving only your own — then re-run `make manifest`.

7.  **Check it.** Run **`make check`** and fix anything it flags until it passes.

8.  **Generate your Zenodo metadata.** Fill the deposit fields in `metadata.json` — `creators`
    (name, affiliation, ORCID), and optionally `abstract` (a custom description; left blank, one is
    generated for you) and `license` — then run **`make zenodo_citation`**. It writes `.zenodo.json`,
    which controls the title, description, authors, license, and keywords of your **permanent** Zenodo
    record; without it, Zenodo auto-generates a poor record (empty description, no affiliation or
    license) for a DOI you cannot undo. `.zenodo.json` is fully derived from `metadata.json` — don’t
    hand-edit it; edit `metadata.json` and re-run (it always overwrites to match), then commit the
    `.zenodo.json` so it ships in your release. ⚠️ Leave `orcid` empty unless it is a **real** ORCID —
    a malformed one makes Zenodo reject the deposit with an opaque HTTP 500.

9.  **Deposit (GitHub release → Zenodo).** Connect the repo to Zenodo once (Zenodo → log in with
    GitHub → flip your repository **on**), then publish a **GitHub release**. Zenodo automatically
    archives that release and gives it a **DOI** — see Zenodo’s guide,
    [Archiving a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/).
    That released snapshot — your predictions, `metadata.json`, and `registration.md` together — **is**
    your registration. Publish the release **within the deposit window, August 28–31, 2026** (the
    lock stays August 31): Zenodo deposits are public from the moment they are published, and the
    shared late window keeps any team’s predictions from being public while others are still
    working. Then email the core team at
    **<janlukas.pfaender@gmail.com>**: the DOI + your
    file fingerprints (already recorded in `metadata.json`), a note saying **which of your entries
    is your team’s `primary`**, and your team’s signed
    [exposure declaration](https://docs.google.com/document/d/1iY5j_MYT3oAG3XW7eHwy7qxUx25HYu2HdYnsCUsT0qE/edit?usp=sharing)
    (one per team — see the team status update email). Submitting several entries? Each is its own
    repo / release / DOI — send all the DOIs together.

    > The DOI is created *by* the release, so it can’t already be inside the released files — that’s
    > fine, you email it (step 10 records it back, optionally).

10. **Record your DOI in the repo (optional).** After the release, copy your Zenodo **DOI** (the
    permanent **concept DOI**, “Cite all versions”, is the stable one) into the `zenodo_doi` field of
    `metadata.json` and commit/push it — no new release needed; this just records the DOI in your
    repository for reference. (The snapshot Zenodo already archived won’t include this later edit,
    which is fine — that DOI identifies the snapshot regardless.) Not required for scoring — emailing
    the DOI in step 9 is enough.

## What you edit vs. what ships

| Path | Role |
|----|----|
| `metadata.json` | **edit** — machine-readable submission metadata; include `code_repository` (and optional `code_doi`) linking the code that generated your predictions |
| `registration.md` | **edit** — GUIDE-LLM-extended reporting checklist |
| `predictions/` | **edit** — your prediction file(s); ships with one `example_*` per tier (delete them before depositing) |
| `raw_data_deposit/` | **edit (Tier 1 only)** — drop your raw Qualtrics export here, then `make clean`; ships with `example_raw_export.csv` (delete it before depositing). Your own raw export **stays in the released deposit** — it is your simulation’s raw output and part of the transparency record. Tiers 2–3 leave this folder empty. |
| `survey/` | reference — `survey.qsf` (Qualtrics import) and `survey.json` (same instrument, readable without Qualtrics) are the full instrument; `questionnaire.txt` is a plain-text rendering; `condition_codenames.csv` maps the raw animal-pair condition code names (used in `survey.qsf`/`survey.json`) to the condition titles you predict |
| `codebook.csv` | reference — every variable: Qualtrics label → target label, wording, and response options |
| `scripts/` | the engine you run — `check.R`, `clean.R`, `manifest.R`, and `lib/` internals; do not edit |

## Commands

These are **optional helpers**, not a required pipeline. They exist so you can produce and
self-validate your files quickly; if you’d rather generate your prediction file(s) your own way from
`codebook.csv`, that’s completely fine — just make sure the result matches the format the benchmark
expects (`make check` is the easiest way to confirm, but not mandatory). The one deliverable that is
**not** optional is the SHA-256 **fingerprint** of each prediction file (recorded in `metadata.json`
and emailed at deposit): `make manifest` is simply the convenient way to compute it — skip the helpers
and you must produce the fingerprint yourself.

| Command | What it does |
|----|----|
| `make check` | Verifies the required files exist; validates `metadata.json`, the file name, the SHA-256 fingerprint, the per-tier data structure, value ranges, and `.zenodo.json`. **Coverage is enforced** — incomplete or duplicated cells fail; a set `team_id` whose predictions aren’t generated yet is reported as *staged*, not broken. Prints **PASS / PASS-WITH-WARNINGS / FAIL**. |
| `make clean` | Tier-1 only: cleans the raw export in `raw_data_deposit/` into `predictions/`, then runs `make manifest`. |
| `make manifest` | Fingerprints every `predictions/<team_id>_*.csv` and records the names + SHA-256 in `metadata.json`. Run it after writing or changing a prediction file (Tier 2 / 3, or after deleting examples). |
| `make zenodo_citation` | (Re)generates `.zenodo.json` from `metadata.json` so your Zenodo deposit gets a well-formed permanent record (title, description, creators, license, keywords, benchmark link). `.zenodo.json` is fully derived — edit `metadata.json` and re-run; it always overwrites to match. |

> A **SHA-256 fingerprint** is a 64-character code derived from a file’s exact contents. It’s the
> tamper-proof seal on your locked predictions: change one number and the code changes, so the
> organizers can later confirm your deposited file is the one you committed to. `make manifest`
> computes it for you — you never type it by hand.

## Requirements

You only need these if you choose to run the optional helper commands above:

- **R ≥ 4.2** (the current minimum for **tidyverse**; developed and tested on R 4.4), with the
  packages **tidyverse**, **jsonlite**, and **digest**. Install them with:

  ``` r
  install.packages(c("tidyverse", "jsonlite", "digest"))
  ```

- **GNU Make**, for the `make …` shortcuts. Without it, call the scripts directly:
  `Rscript scripts/clean.R`, `Rscript scripts/manifest.R`, `Rscript scripts/check.R`.

If you build and validate your submission another way, none of the above is required — only your
prediction file(s) and `registration.md` are.

## Tier 1: clean your raw data

Tier-1 predictions are individual-level: you simulate respondents through the **same Qualtrics
instrument** the human study uses, so your raw output carries Qualtrics variable names
(`trust_competent_1`, `policy_1_1`, `funding_5`, `donation`, …) and lacks the constructed scale
variables the locked analysis scores (`trust_multidimensional`, the `*_mean` composites, the
reverse-coded `funding_perceptions`, `age_band`). `make clean` bridges that gap for you — it applies
the study’s exact recodes and composites so you don’t have to.

Using it is optional. If you prefer to build the analysis-ready file yourself, that’s fine — but it
must reproduce those constructed variables exactly as `codebook.csv` documents them (e.g.
`funding_perceptions = 100 − funding_5`; `age_band` cut at 18–29 / 30–44 / 45–59 / 60+). `make check`
will tell you whether the result is well-formed. Note that **scoring reads the composite columns as
submitted** — it does not recompute them from the items — so a hand-built composite that deviates
from its definition is scored on the deviant values (`make check` warns when the primary outcome and
its trust items disagree).

1.  **Export your simulated responses** as a CSV using the Qualtrics variable names and value codes
    documented in `codebook.csv` — its `qualtrics_label` → `target_label` columns are the complete
    raw→clean field map (e.g. `trust_competent_1` → `trust_competence_1`, `funding_5` →
    `funding_perceptions`, `individual_*` → `behavior_*`, `donation` → `donation_ams`), so you can
    drive your export off it instead of transcribing names by hand. A genuine Qualtrics export — with its
    two extra header rows and system columns — works as-is; so does a plain one-header CSV. This repo
    holds **one** Tier-1 entry, so there is one raw export.
2.  **Drop that file into `raw_data_deposit/`.** Leave exactly one CSV in the folder (delete the shipped
    `example_raw_export.csv`).
3.  **Run `make clean`.** It reads the file in `raw_data_deposit/`, maps the Qualtrics labels to the
    analysis schema, builds the constructed variables, writes
    `predictions/<team_id>_T1_<entry>_v1.csv` (`team_id` and `entry` come from `metadata.json`), and
    records the file’s fingerprint in `metadata.json`. Edit `metadata.json` *before* this step.

A fresh clone ships `raw_data_deposit/example_raw_export.csv` and `metadata.json` with
`entry: "primary"`, so running `make clean` immediately reproduces
`predictions/example_T1_primary_v1.csv` — try it once to see the workflow, then swap in your own data.
To clean a file kept elsewhere, pass it explicitly: `make clean INPUT=path/to/raw.csv`.

## Prediction file naming

    <team_id>_T<tier>_<primary|secondary-k>_v<n>.csv          # Tier 1 and Tier 3
    <team_id>_T2_<primary|secondary-k>_v<n>_cells_main.csv    # Tier 2
    <team_id>_T2_<primary|secondary-k>_v<n>_cells_moderator.csv

Your `team_id` is not something you invent: it was **assigned at registration and sent to your
team by email** (`team_N`, in the team status update of August 15, 2026). Use it exactly as
received, in `metadata.json` and in the file names. If you are unsure which ID is yours, email us
before depositing.

`team_id`, `tier`, and `entry` must match `metadata.json`. `v<n>` is your version counter (start at
`v1`). It exists only for *your* bookkeeping **before** the deposit: if you regenerate predictions
after fingerprinting, bump to `v2` so stale files can’t be confused with current ones, and keep only
the latest version in `predictions/` (`make manifest` fingerprints whatever is there). The version
you deposit is final — after the lock there are no `v(n+1)`s.

**Coverage.** The study has **17 conditions** = control + **16 interventions**, scored on **13
outcomes**. In `metadata.json`, `coverage.interventions` counts the 16 interventions (not control)
and `coverage.outcomes` the 13 outcomes — that’s the `{ "interventions": 16, "outcomes": 13 }` you
see. Your data must still include the **control** condition: Tier-1 rows and Tier-2 cells cover all
17 conditions, while Tier-3 reports the 16 interventions’ effects *relative to* control (no control
row). The exact per-tier column schema and labels are enforced by `make check` (see the `example_*`
files in `predictions/` and `scripts/lib/submission_spec.R`). **Full coverage is required** —
partial coverage is not accepted; every intervention and outcome must be predicted, and `make check`
fails a `metadata.json` that declares anything less than `{ "interventions": 16, "outcomes": 13 }`.

**Completeness is enforced.** `make check` requires every cell of the full grid to be present
**exactly once**, with **no `NA` values in any file**: a Tier-2 main file covers all condition ×
outcome cells, the Tier-2 moderator file all condition × (moderator level) × outcome cells across
the six moderators, and a Tier-3 file all 16 × 13 intervention × outcome ATEs. Missing cells,
`NA` values, or duplicates **fail** the check. To predict *no moderation* for a demographic group,
repeat the condition mean from your main file in that group’s moderator cells — an always-available,
honest prediction that the intervention works the same for that group (see the FAQ).

**Tier-2 cell scales.** Each cell `mean` is the group average on the outcome’s **native** scale: most
outcomes are 0–100, `donation_ams` is 0–10 dollars, and **`newsletter_signup` is a 0–1 proportion**
(the share who subscribed) — not the individual 0/1 of Tier 1. `make check` range-checks these per
outcome. (Tier-3 `ate` is an unbounded difference and is not range-checked.)

**Tiers 2–3 are point predictions only.** A Tier-2 main file has the columns `condition, outcome, mean`, the Tier-2 moderator file `condition, moderator, moderator_level, outcome, mean`, and a
Tier-3 file `condition, outcome, ate`. No uncertainty intervals are submitted: the cross-team
scoring dropped the inferential-agreement and equivalence (TOST) metrics, which were the only
consumers of the intervals, and interval scoring (PI coverage / Winkler) was dropped with them.

## The survey

The full instrument is provided as two files. **Both encode the same survey**; they differ only in
format and intended use:

|  | `survey/survey.qsf` | `survey/survey.json` |
|----|----|----|
| **What it is** | Qualtrics’ proprietary survey-export file | Qualtrics’ documented Survey-Definitions API output |
| **Format** | JSON, but an undocumented proprietary structure | JSON with a documented schema (`result.Questions`, `result.Blocks`, `result.SurveyFlow`, …) |
| **Best for** | re-importing into Qualtrics to **run** the survey yourself | **reading / parsing** the instrument programmatically — e.g. individual participant simulations that need the items, response scales, block/flow order, branching and randomization a respondent saw |
| **Qualtrics license** | required (to import and run) | not required (it is plain JSON anyone can read) |

In short: use `survey.qsf` if you want to *run* the survey in Qualtrics; use `survey.json`
if you want to *read* it without a Qualtrics account.

> **Scope note.** These files are the reduced *LLM-simulation* instrument: respondents are routed
> through the non-interactive conditions only (assigned by a block randomizer); the interactive chatbot
> arms have been removed. The condition labels you are scored on are defined in
> `survey/condition_codenames.csv`, the outcomes in `codebook.csv`, and both in
> `scripts/lib/submission_spec.R`; treat those as authoritative for scope, and the two survey files as
> the faithful record of the instrument.

A human-readable rendering is also provided as `survey/questionnaire.txt`, laid out in
chronological survey order (the order a respondent moves through the instrument). Every item is
annotated as `[qualtrics_label · answer values] question`, alongside the condition labels and the
intervention stimulus texts.

Tier-1 runs export raw Qualtrics column names; `make clean` maps them to the analysis schema
documented in `codebook.csv`.

## Licensing of the shipped survey materials

Your Zenodo license (default `CC-BY-4.0` in `metadata.json`) applies to **your** contribution —
your code, predictions, and documentation. The shipped `survey/` folder is different: several
intervention stimulus texts adapt previously published journalism and other copyrighted material,
included here for scholarly research use. Keep `survey/` in your deposit unchanged (it documents
what your respondents saw), but your license grant does not — and cannot — re-license those
underlying texts.

## More

Common questions — the multi-pair condition code names, attention checks,
what feedback you get and when — are answered in [`FAQ.md`](FAQ.md). Tiers, scoring, disclosure
classes, and the full timeline are described in the
[call for participation](https://janpfander.github.io/llm_predictions_megastudy/). Questions:
see the call’s Contact page.
