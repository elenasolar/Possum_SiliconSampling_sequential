# FAQ — Silicon Sample Benchmark submissions

Short answers to common questions. The README stays the canonical walkthrough;
where the two overlap, the README wins.

**Do Tier-2/3 submissions include uncertainty intervals?**
No — point predictions only. Earlier drafts asked for `pi_lower` / `pi_upper` columns; they are
gone. The cross-team scoring dropped the inferential-agreement and equivalence (TOST) metrics,
which were the only consumers of the intervals, and interval scoring (PI coverage / Winkler) was
dropped with them.

**The raw survey has code names like `"crushing chicken; gross grasshopper; homely halibut"` — is
that three conditions?**
No — one. A code name is either a single animal pair or a semicolon-joined list of pairs; the
semicolons are part of the name. Join `survey/condition_codenames.csv` on the *full* string without
splitting it. The four multi-pair names exist for allocation bookkeeping only; every pair inside one
name showed identical content.

**Do my synthetic respondents need to answer the consent, attention/AI filters, and other unscored
items?**
No. Only the columns in the Tier-1 schema (`scripts/lib/submission_spec.R` → `tier1_required`) are
read. The filter items exist to screen human participants; how — and whether — you simulate them is
your design choice. The same goes for unscored survey content (e.g. `education_climate`,
`social_class`): shipped for fidelity, ignored by scoring.

**Are composites like `trust_multidimensional` recomputed from my item columns at scoring?**
No — scoring reads the composite columns as submitted. `make clean` builds them exactly per the
codebook; if you build the file yourself, keep them consistent (`make check` warns when the primary
outcome disagrees with its items).

**My approach can't simulate some moderator levels (e.g. "Other" gender). Can I leave those Tier-2
moderator cells out, or set them to `NA`?**
No — every file must be complete, the moderator file included, and `make check` fails on `NA` cells.
But no cell is out of reach: your main file already predicts a mean for every condition × outcome,
and repeating that condition mean in a group's moderator cells is a real, honest prediction — it says
"this intervention works the same for this group as for everyone" (no moderation). It is scored
accordingly. Predicting only the subgroups you feel confident about would let teams pick their own
test set, which is why missing cells are not accepted anywhere.

**The control condition has three filler texts. Which one do my control respondents see?**
Mirror the survey: each control respondent is randomly assigned *one* of the three (neckties /
baseball / dances). All three map to the single condition label `control`.

**Where does my `team_id` come from?**
It is assigned by the organizers: every registered team received its ID (`team_N`) by email in the
team status update of August 15, 2026. Use it exactly as received, in `metadata.json` and in the
prediction file names. Don't invent one — the self-check cannot know your assigned ID, so a wrong
one passes `make check` but breaks the link to your registration. If you are unsure which ID is
yours, email us before depositing.

**When exactly should we deposit, and what goes in the deposit email?**
Publish your release within the deposit window, **August 28–31, 2026**. Zenodo deposits are public
from the moment they are published, and the shared late window keeps any team's predictions from
being public while others are still working. The deposit email then carries: all your deposit DOIs,
the SHA-256 fingerprints of your prediction files, a note saying which entry is your team's
`primary`, and your team's signed exposure declaration (the form linked in the team status update
email — one per team).

**How many entries can we submit?**
At most three per tier per team (up to nine total). Every entry enters all main analyses (the
cross-team field statistics and the leaderboard); the per-tier cap limits how much any single team
can shape the field distribution. If there are good reasons to systematically vary some aspect of
an approach, we may grant more entries on request, but only three per tier enter the main analyses.
Mark exactly one entry — across all tiers — `primary`: a robustness analysis reruns the main
results on primary entries only, one per team. In figures and the leaderboard, entries appear under
neutral submission labels, with a table mapping each label to its team. The most useful
extra entries vary one factor and hold the rest fixed.

**Is there a minimum number of synthetic respondents (Tier 1)?**
Yes — at least **500 per intervention and 1,000 in control**, the size of the human half every
submission is scored against (the benchmark preregistration's *precision requirement*). Below that
floor your effect estimates are noisier than the reference for reasons unrelated to your method.
More is encouraged; beyond precision a bigger pool buys nothing — only point estimates are scored,
so a larger pool stabilizes them but cannot buy a better score. `make check` warns when a file is
below the floor.

**What does `v2` in a file name mean, and when do I bump it?**
Only your own pre-deposit bookkeeping: bump when you regenerate predictions after fingerprinting so
stale files can't be mixed up with current ones. Keep only the latest version in `predictions/`.
The deposited version is final.

**What feedback do we get, and when?**
Your deposit is acknowledged when you email the DOI + fingerprints. After the prediction lock
(August 31, 2026) the sealed human data are opened and every submission is scored; each team receives its own scores when the first
manuscript draft is shared with all teams for comment (target: September 30, 2026). No scores or
human results are available to anyone before the lock.

**Do we have to deposit raw model output logs?**
Required for Tiers 1–2 (public or escrowed — withholding them makes the entry Class C); for Tier 3,
required where intermediate generations exist. If they fit, put them (or their archive + SHA-256
hash) in the repo; if too large for a GitHub release, deposit them as a separate Zenodo upload and
link it in `registration.md` item K.2. Escrow is fine for logs — see the disclosure policy.

**Can we publish the deposit under our own license?**
Yes for your own contribution (code, predictions, docs; default `CC-BY-4.0`). The shipped `survey/`
stimulus texts adapt previously published, partly copyrighted material included for scholarly
research use — keep them in the deposit unchanged, but your license grant does not re-license them.
