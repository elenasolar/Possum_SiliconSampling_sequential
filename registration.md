# Silicon Sample Benchmark — method registration form

Fill in every item before the prediction lock; this file ships inside your repo's Zenodo release
(see the README's *Deposit* step). This form covers **one entry** (one repo / one Zenodo release,
`primary` or `secondary-k` — see the README's *What counts as a submission*); if you submit several
entries, fill one form per entry. Items marked **★**
must be disclosed **fully publicly** (never escrowed or withheld). Items marked **†** must be at
minimum escrowed — they may be sealed from the public, but never withheld from the core team. Items
not applicable to your approach: write `N/A`. When several models serve different pipeline stages, complete the model
sections (B) once per model. See the call's *Disclosure policy* for escrow rules.

---

## 0 · Approach identity and output
- **0.1 Team ★** — name, the one or two members (teams are at most two, unless a larger team was approved on request), affiliations, corresponding contact:
  Team `team_30`. David Garcia, Peer Saleth, Elena Solar — Universität Konstanz. Corresponding contact: elena.solar@uni-konstanz.de.

- **0.2 Plain-language summary ★** — one paragraph, what the approach does (not how):
  We simulate synthetic survey respondents as LLM personas built from real Reddit users' own post and comment histories. 1,000 personas, sampled to match the parent megastudy's demographic recruitment quotas, are each shown the study's climate-trust survey materials and asked to predict how that specific person would most likely have answered, given only what can be inferred about them from their own post history and infered demograpics.
  Our approach is based on "PoSSUM: A Protocol for Surveying Social-media Users with Multimodal LLMs" by Roberto Cerina (2025).

- **0.3 Submission tier & approach family ★** — tier (1/2/3); family (e.g. per-respondent simulation / agent / direct forecast; single model / ensemble / multi-agent; zero-shot / literature-conditioned):
  Tier 1 (individual-level). Family: per-respondent simulation, single model, zero-shot (no fine-tuning). Follows the PoSSUM protocol (Cerina (2025)): an LLM as neutral annotator predicting a specific real person's likely answer from their own social media post history, rather than a free-standing persona-generation approach.

- **0.4 Pipeline diagram** — ordered steps from raw inputs to submitted file:
  1. Candidate collection: SQL query against a Reddit database, filtered to users active in state-identifying subreddits (at least 5 posts there) and meeting a recent-activity threshold (at least 30 posts in the last 12 months).
  2. Build author history: merge/deduplicate each candidate's full submission + comment history.
  3. Build the demographic-inference pool: submissions + first-level comments only.
  4. Compute per-user subreddit activity counts.
  5. Assign each candidate's US state from aggregate subreddit activity (state-linked subreddits).
  6. Embedding-based demographic scoring: Waller & Anderson (2021) community-embedding ideology/gender/age scores, activity-weighted per user.
  7. LLM-based demographic inference (gender, age band, race, education, income, party), informed by the embedding scores, with a confidence score per attribute.
  8. Stratified sampling of 1,000 agents: hard quotas on age_band×gender and gender×race, matched to the parent megastudy's own recruitment quota shares.
  9. Classify each candidate's comments by structural role (submission / first-level comment / depth-2 reply / deeper reply / unresolved).
  10. Extract "meaningful" comment-reply interactions: a linguistic-cue filter (question / second-person / disagreement / hedge / booster / quotation / etc.) on depth-2 replies, with a relaxed word-count-only fallback for the small number (1%) of sampled agents who fell short of the interaction-count target under the full filter.
  11. Join the sampled agents to their meaningful interactions.
  12. Build personas: fixed-template verbalization of each agent's demographics + most-recent-first interaction history within a token budget.
  13. Run the survey: for each persona × 17 conditions, present the pretreatment items (once per agent), the condition's stimulus, and the 44-item outcome battery, eliciting a structured answer + speculation score + brief explanation per item.
  14. Parse structured output into item answers; build the raw Qualtrics-format export and the Tier-1 prediction file.

- **0.5 Coverage ★** — number of respondents/cells/estimates; mapping to conditions. Full coverage is required: every submission predicts **all 16 interventions and all 13 outcomes** (partial coverage is not accepted). Confirm here:
  1,000 personas × 17 conditions (control + 16 interventions) = 17,000 rows; all 13 scored outcomes answered for every row.

## A · Scope of LLM use
- **A.1 Purpose** — every workflow stage where LLMs are used:
  (1) Demographic inference (gender/age/race/education/income/party) from each candidate's own Reddit post/comment history; (2) the survey-response simulation itself (predicting each persona's answer + speculation score + brief explanation per item).

- **A.2 Degree of automation ★** — confirm fully automated, no human in the loop at prediction time; note any exception:
  Fully automated at prediction time: no human edited, selected, or reviewed individual model answers. Two exceptions worth disclosing transparently, both at the *design/data-construction* stage, not at prediction time: 
  (1) 2 sampled agents were reassigned to a Washington D.C. state label via a documented, seeded, weakest-evidence-first random procedure, since the candidate-collection query predated D.C. being added to the state-subreddit reference list; 
  (2) for a small number (~10) of sampled agents who fell under the interaction-count target, the "meaningful interaction" definition's linguistic-cue requirement was deliberately relaxed to a word-count-only floor to fill out their persona material. Every other agent used the same rule throughout. The pipeline does include this: strict boundary first, relax only if necessary.

## B · Model / system details (once per model)
- **B.1 Model name(s)** — exact identifiers incl. provider, size, version/timestamp, source link:
  `Qwen/Qwen3.8-27B-FP8` (Hugging Face: https://huggingface.co/Qwen/Qwen3.8-27B-FP8), FP8 quantized, self-hosted.

- **B.2 Access & context mode** — API/web/local; API name + version; chat vs stateless; exact call dates:
  Local/self-hosted, served via vLLM behind an OpenAI-compatible HTTP API on a university Kubernetes GPU cluster (not a third-party hosted API). 
  Stateless per call — no server-side conversation memory; each call's full context is explicitly included in that call's prompt.
  Sequential run was started 27.8.2026, roughly 16:00 and ran ~7h.

- **B.3 Configuration** — temperature, top-p/top-k, max tokens, penalties, stop sequences, seeds, reasoning effort, completions per item:
  Temperature 0.7, top_p 0.8 (Qwen3's recommended non-thinking settings), max_tokens 5000, no explicit penalties/stop sequences beyond the model's own. "Thinking" (chain-of-thought) mode explicitly disabled via `chat_template_kwargs.enable_thinking=false`. Base seed 20260818; per-(agent, condition[, step, attempt]) seeds derived deterministically from it.

- **B.4 Customization** — fine-tuning, RAG, prompt optimization, tool use, web search, agentic scaffolds (cross-ref H):
  No fine-tuning, no retrieval-augmented generation, no external tool use, no web search. Custom prompt templates (PoSSUM-adapted, see C.1). 
  Sequential single-chain design that described the user persona, showed the intervention, followed by the 13 outcome questions.
  Model and prompt design was validated against Voelkel et al. (2025) in an external testbed, based on Ashokkumar et al. (2026).

- **B.5 Persistent memory** — across interactions? what persisted:
   No memory persists across different respondents or conditions (each call is stateless server-side; the model has no memory beyond what's explicitly included in that call's prompt). 
   Within one respondent × condition session, the model answers all questions in one prompt, meaning it contains knowledge of prior questions and answers (reusing KV cache).

- **B.6 Inference stack** — for local models: serving framework + version, quantization, hardware:
  vLLM v0.19.1, FP8 quantization, single GPU per pod (university Kubernetes cluster), `--max-model-len 16384`, prefix caching enabled.

- **B.7 Ensembles** — members + exact aggregation rule:
  N/A — single model, no ensembling.

## C · Prompts
- **C.1 Exact prompts** — verbatim text or link to deposited file; were they iteratively refined? pre-specified vs in response to outputs:
  Verbatim prompt template pieces are included in this repository (`src/stage4_survey/prompts/`: `task_instructions.txt`, `explanation_brief.txt`, `output_format.txt`, `persona_header.txt`, `preamble.txt`, `session_intro.txt`, `demographics_intro.txt`, `activity_intro.txt`, `cleanup_strict.txt`); `session.py` composes the full per-step prompt from these plus each persona's own demographic/activity data. The prompt structure was validated against a held-out real-respondent task (external testbed based on Ashokkumar et al. (2026), validated against Voelkel et al. (2025)).

- **C.2 System-wide instructions**:
  See `preamble.txt`/`session_intro.txt`/`task_instructions.txt` in the deposited prompt files for the exact wording. In summary: the model is instructed to act as a neutral annotator predicting how *this specific user* (not itself, not an average American) would answer, using only what the provided user data supports, and to report a calibrated 0–100 speculation level per answer per the guidelines in `task_instructions.txt`.

- **C.3 Prompt-design rationale** — brief rationale for the prompt design: why prompts were structured as they were, and the reasoning behind major design choices (recommended, not required):
  The "neutral annotator" framing (PoSSUM, Cerina (2025)) was chosen to reduce the model substituting its own views for the persona's. A per-item speculation score was requested so downstream analysis can account for how much the available Reddit data actually supports each prediction, rather than treating every answer as equally grounded. A brief explanation was requested per item for auditability, to make visible which features of a persona's actual data drove each answer.


## D · Persona / profile construction (Tiers 1–2)
- **D.1 Profile source** — source of demographic profiles you constructed: a public survey (e.g. GSS / ANES / Census), other survey, fully synthetic, or none. The benchmark ships no participant pool; report how you built yours, incl. condition assignments:
  Reddit-derived, not from a public survey. Demographic attributes (gender, age band, race, education, income, party) are LLM-inferred from each candidate's own Reddit submissions and first-level comments, informed by a community-embedding prior (Waller & Anderson (2021)), with a per-attribute confidence score recorded. State is separately assigned from aggregate subreddit activity mapped to state-identifying subreddits (not LLM-inferred). All 1,000 sampled personas answer all 17 conditions (see D.3).

- **D.2 Profile verbalization** — which variables, rendered how (template vs generated narrative; if generated: model + prompt):
  Fixed template, not an LLM-generated narrative. Demographics and assigned state are presented as structured fields; a persona's post-history section is a template-rendered, most-recent-first list of that agent's real "meaningful" comment-reply interactions (with parent-comment context preserved), included up to a fixed token budget.

- **D.3 Assignment & weighting** — number of personas, assignment to conditions (your responsibility, all 17 conditions), reuse, weighting/matching:
  1,000 personas. Each persona answers all 17 conditions (control + 16 interventions). A single fixed pool reused across every condition, not separately sampled per condition. Sampling used hard quotas on two joint distributions (age_band × gender, gender × race) matched to the parent megastudy's own recruitment quota shares (which in turn derive from 2024 U.S. Census Bureau population estimates, the same ultimate source as a separate ACS/PUMS pull used only as a marginal-only diagnostic, not an enforced quota); education, income, and party were matched at the marginal level only, not quota-enforced.

## E · Stimulus and survey administration
- **E.1 Stimulus presentation** — verbatim vs paraphrase; how state-contingent content is handled:
  Verbatim. Stimulus texts are included in this repository unchanged from the benchmark's own materials (`src/stage4_survey/stimuli/`, sourced from `survey/questionnaire.txt`). State-contingent content (intervention #16, extreme weather): each persona's assigned state is mapped to one of three risk cases (flood / wildfire / severe cold) or a generic Case 4 fallback for personas with no inferable state; the "which state do you live in" pre-item is rendered as if the persona had answered with their assigned state (or "Prefer not to say" under Case 4).

- **E.2 Survey walk-through** — one item/call vs blocks vs whole survey; context carry-over; item/option ordering & randomization; scale display; attention/comprehension handling:
  **Secondary-1 entry (sequential, `qwen27b_v1`):** pretreatment once per agent, then per condition a single continuous call chain covering the stimulus pages and the full 44-item outcome battery together. 
  Item/option ordering and randomization are deterministic per (agent, condition) via a seeded hash (reproducible, not re-randomized on repeat runs); sliders are 0–100 integers with endpoints (and occasionally the midpoint) labeled; choice items list symbol-coded options. Attention/comprehension-check items are not simulated (out of scope per the benchmark's own FAQ).

- **E.3 Response elicitation** — free text / constrained choice / structured output / token log-probabilities (if logprobs: normalization & mapping):
  Structured output: each item is answered as a fixed-format text block (`**title: ...** **explanation: ...** **answer: ...** **speculation: ...**`), parsed by a tolerant parser (case-insensitive keys, flexible field order/formatting). 

## F · Stochasticity and aggregation
- **F.1 Runs & seeds** — runs per respondent/item/estimate; seeds; reproducibility under identical settings:
  One run per respondent × condition (no repeated sampling). Base seed 20260818; per-(agent, condition[, step, attempt]) seeds are derived deterministically (SHA-256-based hash of the base seed + identifiers), so a re-run against the identical model/settings uses identical per-call seeds — reproducible modulo any provider/serving-side nondeterminism (e.g. batching effects on a GPU inference server, which are outside the seed's control).

- **F.2 Aggregation rule** — how multiple generations become submitted values (mean/median/mode/first/sampled/…):
  N/A - no aggregations were performed. 

## G · Validation & post-processing
- **G.1 Human validation** — any human review of outputs (often N/A):
  N/A — no human reviewed or edited individual model outputs. 

- **G.2 Post-processing** — parsing rules; handling of refusals/malformed/missing/out-of-range; exclusions; for approaches that generate individual responses, the resulting effective N per condition (descriptive disclosure, not a scoring input):
  Up to 3 attempts per step to obtain a fully-parsed, complete answer set from the primary model; on persistent parse failure an optional "strict cleanup" pass by a smaller model is attempted (but was never needed). A session is marked complete only if every step parsed completely; otherwise partial (some items missing, left as NA) or failed. Any partial/failed sessions identified after a run were individually re-run (`--only-agents`) rather than backfilled or dropped. Effective N: 1,000 respondents × 17 conditions for the sequential entry, confirmed 17,000/17,000 complete with 0 partial/0 failed after resolving isolated context-length failures. 

- **G.3 Calibration corrections** — any post-hoc scaling/shifting/debiasing and exactly what data it was fit on (cross-ref H/I):
  N/A. No post-hoc scaling, shifting, or debiasing applied to any submitted value.

## H · Learning and conditioning components
- **H.1 Fine-tuning data** — exact corpus (hashes/DOIs), hyperparameters, checkpoints:
  N/A — no fine-tuning.

- **H.2 Context & retrieval corpora** — exact document set in context / indexed, archived in the deposit:
  No external retrieval corpus or index. Each persona's own Reddit post/comment history is included directly in-context as part of the prompt (constructed during profile-building, cross-ref D.1/D.2) — not retrieved from an external index at inference time.


## I · Data inputs, blinding, and competing interests
- **I.1 Competing interests ★** — funding, in-kind compute/model access, relationships with LLM-interested entities:
  N/A - no competing interests.

- **I.2 External human data †** — all external human datasets that informed the approach anywhere (training/fine-tuning/retrieval/ICL/calibration):
  Used only as marginal-level diagnostics/validation targets, not for training/fine-tuning/retrieval/ICL: (1) 2024 ACS/PUMS person records (Census Bureau) and CES (Cooperative Election Study, Harvard Dataverse): census-matching target distributions for the stratified sample; 
  (2) Waller & Anderson's (2021) published subreddit-level community-embedding scores: prior signal for demographic inference; 
  (3) the Voelkel et al. (2025) climate-message megastudy (13,821 real respondents): used *only* in our external validation testbed to test respondent-realism before running against this study's sealed data (cross-ref J.1), never as input to the actual submitted predictions.

- **I.3 Blinding attestation ★** — **mandatory.** Signed attestation that no team member accessed, solicited, or was shown any human outcome data from this study, including pilots, before the prediction lock:
  We attest that no team member accessed, solicited, or was shown any human outcome data from this study, including pilots, before the prediction lock. The signed form is attached.

- **I.4 Contamination note †** — training cutoff of every model vs public release dates of this project's materials; note any known exposure:
  Per the models own answer: around January 2025. No official knowlede cutoff is reported in the model card.

## J · Internal selection procedure
- **J.1 Design-space search †** — how the final pipeline was chosen: how many configurations tried, internal validation criterion, what data it ran against:
  This sequential single-chain design was deemed secondary-1 entry (against our primary submission) based on validation against a held-out real-respondent task: the *Voelkel et al (2025)* climate-message megastudy (13,821 real respondents, real ground truth, via the *Ashokkumar et al (2026)* benchmark). 
  Motivation to compare both approaches was the concern that within one long call chain, later answers may anchor on earlier ones in an unrealistic way.

## K · Reproducibility & frozen artifacts
- **K.1 Code & materials** — link/DOI, secrets removed, determinism/seeds documented (also record the link in `metadata.json` → `code_repository` / `code_doi`):
  https://github.com/elenasolar/Possum_SiliconSampling_sequential

- **K.2 Raw output logs †** — complete unprocessed model responses archived, hashed, time-stamped (required for Tiers 1–2, public or escrowed; Tier 3 where intermediate generations exist; oversized logs may be a separate linked Zenodo upload):
  Complete, unprocessed raw model call logs (`calls.jsonl`: full prompt, full response, tokens, timing, per call) are included in this deposit, hashed and time-stamped. Published in full as the underlying Reddit-derived text embedded in prompts/explanations is already public information.

- **K.3 Computational resources** — API-call counts, total tokens, cost, compute time:
  Sequential approach: 25,119 API calls 
  Runtime ~10h.
  Input Tokens: 204,179,786 
  Output Tokens: 57,401,268 

## L · Disclosure class
Each item above is deposited as **public**, **escrowed** (sealed from the public but available to the
core team and auditors under confidentiality, with a public SHA-256 hash + timestamp so the lock is
still verifiable — an embargo with a sunset date is encouraged), or **withheld** (permitted only for
items marked neither ★ nor †). Your entry's class is set by its **most restricted item** and recorded
in `metadata.json` → `disclosure_class` (and `escrow_doi` if anything is escrowed):
- **A · Open** — all items public. Full results-table standing; all features enter the design-choice analysis.
- **B · Escrowed** — some items sealed but every item is available to the core team/auditors under confidentiality. Full standing with an *escrowed* badge; only publicly disclosed features enter the design-choice analysis.
- **C · Sealed** — one or more permitted items withheld even from escrow. Scored and reported with a *not independently verifiable* flag; excluded from the approach catalogue and design-choice analysis.

★ items must always be public (never escrowed or withheld); † items must be at minimum escrowed. Full
policy: <https://janpfander.github.io/llm_predictions_megastudy/#disclosure>
