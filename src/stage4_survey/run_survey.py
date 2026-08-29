"""
Stage 4 — run every synthetic respondent through every condition (PoSSUM-style annotator calls).

Usage
-----
    python src/stage4_survey/run_survey.py --config configs/run_dummy.json \
        --personas data/processed/personas/personas.jsonl

    # inspect the exact prompt for one agent/condition without calling any model
    python src/stage4_survey/run_survey.py --config configs/run_dummy.json --print-prompt p00001 "Funding"

Config (JSON) — see configs/*.json:
    run_id            name of this run; outputs go to data/processed/{raw_model_outputs,survey_responses}/<run_id>/
    model             model client config (provider/model/base_url/api_key_env/temperature/...)
    cleanup_model     optional second (small, temperature 0) model used only to re-format
                      unparseable outputs (PoSSUM's "strict cleanup" pass); null disables it
    options:
        conditions           list of condition titles to run (default: all 17)
        battery_steps        1 (whole outcome battery in one call) or 3 (primary/secondary/tertiary)
        battery_mode         "sequential" (default) | "independent_blocks" | "independent_items":
                             independent modes ask every outcome block (or item) in its own call that
                             sees the treatment transcript but none of the other outcome answers
                             (anchoring mitigation, see session.build_session_plan)
        pretreatment         true -> elicit belief_pre/trust_pre ONCE per agent and show them in every session
        include_demographics true -> the six known moderators (+ state) are stated explicitly in the prompt
        explanation_style    "brief" | "long" | "none"
        max_attempts         max primary calls per step until the output parses completely
        seed                 base seed; per-(agent,condition) seeds are derived deterministically
        workers              parallel agents
        independent_concurrency  for independent battery modes: how many of one session's
                             independent block/item calls run at the same time (default 1 =
                             one after the other). Total in-flight requests <= workers x this.
        store_prompts        "full" | "hash"  (raw archive keeps full prompt text or only a hash)

Outputs
-------
    .../raw_model_outputs/<run_id>/calls.jsonl        every model call: prompt, response, tokens, timing (K.2)
    .../survey_responses/<run_id>/pretreatment.jsonl  once-per-agent pre-treatment answers (if enabled)
    .../survey_responses/<run_id>/responses.jsonl     one record per agent x condition: answers, speculation, status
    .../survey_responses/<run_id>/run_summary.json     counts, token usage, config snapshot

The run is resumable: re-running with the same config skips (agent, condition) pairs already
recorded as complete in responses.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import codebook as cb  # noqa: E402
from common.io_utils import JsonlAppender, dump_json, load_json, read_jsonl, stable_int_hash  # noqa: E402
from common.model_clients import ModelError, UsageCounter, add_endpoint_args, apply_endpoint_args, make_client  # noqa: E402
from stage4_survey import session as ss  # noqa: E402
from stage4_survey.parsing import answers_by_key, parse_output  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OPTIONS = {
    "conditions": None,
    "battery_steps": 1,
    "battery_mode": "sequential",
    "pretreatment": False,
    "include_demographics": True,
    "explanation_style": "brief",
    "max_attempts": 3,
    "seed": 20260818,
    "workers": 4,
    "independent_concurrency": 1,
    "store_prompts": "full",
    "log_every": 25,
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class SurveyRunner:
    def __init__(self, cfg: dict, personas: list[dict], out_root: Path):
        self.cfg = cfg
        self.run_id = cfg["run_id"]
        self.opts = {**DEFAULT_OPTIONS, **(cfg.get("options") or {})}
        self.conditions = self.opts["conditions"] or list(cb.CONDITIONS)
        unknown = [c for c in self.conditions if c not in cb.CONDITIONS]
        if unknown:
            raise SystemExit(f"unknown conditions in config: {unknown}")
        self.personas = personas
        self.usage = UsageCounter()
        self.client = make_client(cfg["model"], usage=self.usage)
        self.cleanup_client = make_client(cfg["cleanup_model"], usage=self.usage) if cfg.get("cleanup_model") else None

        self.raw_dir = out_root / "raw_model_outputs" / self.run_id
        self.resp_dir = out_root / "survey_responses" / self.run_id
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.resp_dir.mkdir(parents=True, exist_ok=True)
        self.n_target = 0
        self._tty = False
        self._bar_len = 0
        self.calls_log = JsonlAppender(self.raw_dir / "calls.jsonl")
        self.resp_log = JsonlAppender(self.resp_dir / "responses.jsonl")
        self.pre_log = JsonlAppender(self.resp_dir / "pretreatment.jsonl")

        # resume state
        self.done: set[tuple[str, str]] = set()
        if (self.resp_dir / "responses.jsonl").exists():
            for r in read_jsonl(self.resp_dir / "responses.jsonl"):
                if r.get("status") == "complete":
                    self.done.add((r["profile_id"], r["condition"]))
        self.pretreatment: dict[str, dict] = {}
        if (self.resp_dir / "pretreatment.jsonl").exists():
            for r in read_jsonl(self.resp_dir / "pretreatment.jsonl"):
                self.pretreatment[r["profile_id"]] = r
        self._lock = threading.Lock()
        self.n_complete = 0
        self.n_partial = 0
        self.n_failed = 0
        self.t0 = time.time()

    # ------------------------------------------------------------------ #
    def seed_for(self, profile_id: str, condition: str, extra: str = "") -> int:
        return stable_int_hash(self.opts["seed"], profile_id, condition, extra)

    def _log_call(self, **rec):
        if self.opts["store_prompts"] == "hash":
            rec["prompt_sha256"] = hashlib.sha256(rec["prompt"].encode("utf-8")).hexdigest()
            rec["prompt"] = None
        rec["run_id"] = self.run_id
        rec["timestamp"] = now_iso()
        self.calls_log.write(rec)

    # ------------------------------------------------------------------ #
    def answer_step(self, agent: dict, plan: ss.SessionPlan, step_index: int, answers_so_far: dict,
                    log_ctx: dict) -> tuple[dict, dict, dict, str, int]:
        """Run one step (with retries/cleanup). Returns (answers, speculation, explanations, status, n_calls)."""
        step = plan.steps[step_index]
        items = step.items
        prompt = ss.build_step_prompt(agent, plan, step_index, answers_so_far,
                                      include_demographics=self.opts["include_demographics"],
                                      explanation_style=self.opts["explanation_style"])
        best = None
        n_calls = 0
        for attempt in range(1, int(self.opts["max_attempts"]) + 1):
            seed = self.seed_for(agent["profile_id"], plan.condition, f"{step_index}:{attempt}")
            try:
                r = self.client.chat([{"role": "user", "content": prompt}], seed=seed)
            except ModelError as e:
                n_calls += 1
                self._log_call(**log_ctx, step=step_index, step_name=step.name, attempt=attempt, kind="primary",
                               model=self.client.model, provider=self.client.provider, prompt=prompt,
                               response=None, error=str(e)[:2000])
                continue
            n_calls += 1
            res = parse_output(r.text, items)
            self._log_call(**log_ctx, step=step_index, step_name=step.name, attempt=attempt, kind="primary",
                           model=r.model, provider=r.provider, prompt=prompt, response=r.text,
                           input_tokens=r.input_tokens, output_tokens=r.output_tokens, cached_tokens=r.cached_tokens, latency_s=r.latency_s,
                           finish_reason=r.finish_reason, parse_missing=res.missing, parse_invalid=res.invalid,
                           parse_duplicated=res.duplicated)
            if not res.complete and self.cleanup_client is not None:
                cprompt = ss.build_cleanup_prompt(r.text, items)
                try:
                    cr = self.cleanup_client.chat([{"role": "user", "content": cprompt}], temperature=0, seed=seed)
                    n_calls += 1
                    cres = parse_output(cr.text, items)
                    # keep explanations from the primary output where available
                    for t, pi in cres.items.items():
                        if not pi.explanation and t in res.items:
                            pi.explanation = res.items[t].explanation
                    self._log_call(**log_ctx, step=step_index, step_name=step.name, attempt=attempt, kind="cleanup",
                                   model=cr.model, provider=cr.provider, prompt=cprompt, response=cr.text,
                                   input_tokens=cr.input_tokens, output_tokens=cr.output_tokens, cached_tokens=cr.cached_tokens, latency_s=cr.latency_s,
                                   parse_missing=cres.missing, parse_invalid=cres.invalid)
                    if len(cres.missing) + len(cres.invalid) < len(res.missing) + len(res.invalid):
                        res = cres
                except ModelError as e:
                    n_calls += 1
                    self._log_call(**log_ctx, step=step_index, step_name=step.name, attempt=attempt, kind="cleanup",
                                   model=self.cleanup_client.model, provider=self.cleanup_client.provider,
                                   prompt=cprompt, response=None, error=str(e)[:2000])
            if best is None or (len(res.missing) + len(res.invalid)) < (len(best.missing) + len(best.invalid)):
                best = res
            if res.complete:
                break
        if best is None:
            return {}, {}, {}, "failed", n_calls
        answers = answers_by_key(best, items)
        spec = {it.key: best.items[it.title].speculation for it in items if it.title in best.items and best.items[it.title].ok}
        expl = {it.key: best.items[it.title].explanation for it in items if it.title in best.items and best.items[it.title].ok}
        status = "complete" if best.complete else "partial"
        return answers, spec, expl, status, n_calls

    # ------------------------------------------------------------------ #
    def run_pretreatment(self, agent: dict) -> dict | None:
        pid = agent["profile_id"]
        if pid in self.pretreatment:
            return self.pretreatment[pid]
        rng = random.Random(self.seed_for(pid, "__pretreatment__"))
        step = ss.pretreatment_step(rng)
        block_order = [e.block.key for e in step.elements if isinstance(e, ss.ItemPage)]
        plan = ss.SessionPlan(profile_id=pid, condition="__pretreatment__", codename="", steps=[step],
                              meta={"prefix_elements": [], "pretreatment_answers": {}})
        answers, spec, expl, status, n_calls = self.answer_step(
            agent, plan, 0, {}, {"profile_id": pid, "condition": "__pretreatment__"})
        rec = {"profile_id": pid, "answers": answers, "speculation": spec, "block_order": block_order,
               "status": status, "n_calls": n_calls, "timestamp": now_iso()}
        self.pre_log.write(rec)
        with self._lock:
            self.pretreatment[pid] = rec
        return rec

    def run_condition(self, agent: dict, condition: str) -> dict:
        pid = agent["profile_id"]
        pre = self.pretreatment.get(pid) if self.opts["pretreatment"] else None
        plan = ss.build_session_plan(agent, condition, seed=self.seed_for(pid, condition),
                                     battery_steps=int(self.opts["battery_steps"]), pretreatment=pre,
                                     battery_mode=self.opts.get("battery_mode", "sequential"))
        answers: dict = {}
        spec: dict = {}
        expl: dict = {}
        statuses = []
        n_calls = 0
        t0 = time.time()
        log_ctx = {"profile_id": pid, "condition": condition}
        inner = max(1, int(self.opts.get("independent_concurrency", 1)))
        i = 0
        while i < len(plan.steps):
            step = plan.steps[i]
            if step.independent and inner > 1:
                # a run of independent battery steps: none of them sees another's answers, so they
                # can be in flight simultaneously (same persona+treatment prefix -> prefix cache)
                j = i
                while j < len(plan.steps) and plan.steps[j].independent:
                    j += 1
                base = dict(answers)  # treatment answers only; frozen for all of them
                with ThreadPoolExecutor(max_workers=inner) as ex:
                    results = list(ex.map(lambda k: self.answer_step(agent, plan, k, base, log_ctx), range(i, j)))
                for a, s, e, st, n in results:
                    answers.update(a)
                    spec.update(s)
                    expl.update(e)
                    statuses.append(st)
                    n_calls += n
                i = j
                if "failed" in statuses:
                    break
                continue
            a, s, e, st, n = self.answer_step(agent, plan, i, answers, log_ctx)
            answers.update(a)
            spec.update(s)
            expl.update(e)
            statuses.append(st)
            n_calls += n
            i += 1
            if st == "failed":
                break
        outcome_keys = list(cb.OUTCOME_ITEM_KEYS)
        missing_outcomes = [k for k in outcome_keys if k not in answers]
        status = "complete" if not missing_outcomes and all(s == "complete" for s in statuses) else (
            "failed" if "failed" in statuses else "partial")
        meta = {k: v for k, v in plan.meta.items() if k not in ("prefix_elements", "pretreatment_answers")}
        rec = {
            "run_id": self.run_id,
            "profile_id": pid,
            "condition": condition,
            "codename": plan.codename,
            "model": self.client.model,
            "provider": self.client.provider,
            "answers": answers,
            "speculation": spec,
            "explanations": expl,
            "status": status,
            "missing_outcome_items": missing_outcomes,
            "n_steps": len(plan.steps),
            "n_calls": n_calls,
            "seconds": round(time.time() - t0, 2),
            "session_meta": meta,
            "demographics": agent["demographics"],
            "assigned_state": agent.get("assigned_state"),
            "year_birth": agent.get("year_birth"),
            "timestamp": now_iso(),
        }
        self.resp_log.write(rec)
        with self._lock:
            if status == "complete":
                self.n_complete += 1
                self.done.add((pid, condition))
            elif status == "partial":
                self.n_partial += 1
            else:
                self.n_failed += 1
            total = self.n_complete + self.n_partial + self.n_failed
            self._bar()
            if total % int(self.opts["log_every"]) == 0:
                self._progress()
        return rec

    def _status(self) -> str:
        el = time.time() - self.t0
        u = self.usage.as_dict()
        n = self.n_complete + self.n_partial + self.n_failed
        eta = (el / n) * (self.n_target - n) if n and self.n_target else None
        cached = f"{100 * u['cached_tokens'] / u['input_tokens']:.0f}%" if u["input_tokens"] else "n/a"
        return (f"{n}/{self.n_target} pairs | ok={self.n_complete} partial={self.n_partial} failed={self.n_failed} "
                f"| calls={u['calls']} in={u['input_tokens']:,} (cached {cached}) out={u['output_tokens']:,} "
                f"| {el / 60:.1f} min" + (f" | ETA {eta / 60:.0f} min" if eta is not None else ""))

    def _bar(self, width: int = 30):
        """Single-line live progress bar (only when stdout is a terminal; log files get _progress())."""
        if not self._tty:
            return
        n = self.n_complete + self.n_partial + self.n_failed
        frac = min(1.0, n / self.n_target) if self.n_target else 1.0
        filled = int(width * frac)
        bar = f"[{self.run_id}] |{'#' * filled}{'-' * (width - filled)}| {100 * frac:5.1f}% {self._status()}"
        sys.stdout.write(chr(13) + bar.ljust(self._bar_len))  # chr(13) = carriage return: redraw in place
        self._bar_len = len(bar)
        sys.stdout.flush()

    def _progress(self):
        if self._tty:
            sys.stdout.write(chr(10))  # end the live bar line before the permanent log line
        print(f"[{self.run_id}] {self._status()}", flush=True)

    def run_agent(self, agent: dict) -> None:
        pid = agent["profile_id"]
        try:
            if self.opts["pretreatment"]:
                self.run_pretreatment(agent)
            for cond in self.conditions:
                if (pid, cond) in self.done:
                    continue
                self.run_condition(agent, cond)
        except Exception:  # keep the pool alive; record failure
            with self._lock:
                self.n_failed += 1
            print(f"ERROR agent {pid}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ #
    def run(self, limit_agents: int | None = None):
        agents = self.personas[:limit_agents] if limit_agents else self.personas
        todo = [a for a in agents if any((a["profile_id"], c) not in self.done for c in self.conditions)]
        self.n_target = sum(1 for a in todo for c in self.conditions if (a["profile_id"], c) not in self.done)
        self._tty = sys.stdout.isatty()
        print(f"[{self.run_id}] {len(agents)} agents x {len(self.conditions)} conditions; "
              f"{len(self.done)} pairs already complete; {len(todo)} agents with work "
              f"| model={self.client.describe()} | cleanup={'yes' if self.cleanup_client else 'no'} "
              f"| workers={self.opts['workers']}", flush=True)
        workers = max(1, int(self.opts["workers"]))
        if workers == 1:
            for a in todo:
                self.run_agent(a)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(self.run_agent, a) for a in todo]
                for _ in as_completed(futs):
                    pass
        self._progress()
        summary = {
            "run_id": self.run_id,
            "finished": now_iso(),
            "elapsed_min": round((time.time() - self.t0) / 60, 2),
            "n_agents": len(agents),
            "conditions": self.conditions,
            "n_complete": self.n_complete,
            "n_partial": self.n_partial,
            "n_failed": self.n_failed,
            "n_pairs_complete_total": len(self.done),
            "usage": self.usage.as_dict(),
            "model": self.client.describe(),
            "cleanup_model": self.cleanup_client.describe() if self.cleanup_client else None,
            "options": self.opts,
        }
        dump_json(self.resp_dir / "run_summary.json", summary)
        self.calls_log.close()
        self.resp_log.close()
        self.pre_log.close()
        print(json.dumps(summary, indent=2))
        return summary


# --------------------------------------------------------------------------- #

def load_personas(path: Path) -> list[dict]:
    personas = list(read_jsonl(path))
    if not personas:
        raise SystemExit(f"no personas in {path}")
    return personas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--personas", type=Path, default=REPO_ROOT / "data/processed/personas/personas.jsonl")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/processed")
    ap.add_argument("--limit-agents", type=int, default=None, help="only the first N personas (pilot runs)")
    ap.add_argument("--only-agents", nargs="*", default=None, metavar="PROFILE_ID",
                    help="run only these personas (e.g. to redo failed pairs of a finished run; resume logic still skips complete pairs)")
    ap.add_argument("--conditions", nargs="*", default=None, help="override conditions from config")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--run-id", default=None, help="override run_id from config (e.g. pilot vs full run)")
    ap.add_argument("--print-prompt", nargs=2, metavar=("PROFILE_ID", "CONDITION"),
                    help="print the step-1 prompt for one agent/condition and exit (no model call)")
    ap.add_argument("--print-all-steps", action="store_true", help="with --print-prompt: print every step (dummy answers)")
    add_endpoint_args(ap)
    args = ap.parse_args(argv)

    cfg = load_json(args.config)
    apply_endpoint_args(cfg, args)
    if args.conditions:
        cfg.setdefault("options", {})["conditions"] = args.conditions
    if args.workers:
        cfg.setdefault("options", {})["workers"] = args.workers
    if args.run_id:
        cfg["run_id"] = args.run_id
    personas = load_personas(args.personas)
    if args.only_agents:
        wanted = set(args.only_agents)
        personas = [p for p in personas if p["profile_id"] in wanted]
        missing = wanted - {p["profile_id"] for p in personas}
        if missing:
            raise SystemExit(f"--only-agents: not in personas: {sorted(missing)}")

    if args.print_prompt:
        pid, cond = args.print_prompt
        agent = next((p for p in personas if p["profile_id"] == pid), None)
        if agent is None:
            raise SystemExit(f"profile_id {pid} not in personas")
        opts = {**DEFAULT_OPTIONS, **(cfg.get("options") or {})}
        plan = ss.build_session_plan(agent, cond, seed=stable_int_hash(opts["seed"], pid, cond),
                                     battery_steps=int(opts["battery_steps"]),
                                     battery_mode=opts.get("battery_mode", "sequential"))
        answers = {}
        for i, step in enumerate(plan.steps):
            print(f"\n{'#' * 30} STEP {i + 1}/{len(plan.steps)} [{step.name}] — {len(step.items)} items {'#' * 30}\n")
            print(ss.build_step_prompt(agent, plan, i, answers, include_demographics=opts["include_demographics"],
                                       explanation_style=opts["explanation_style"]))
            if not args.print_all_steps:
                break
            for it in step.items:  # placeholder answers so later steps render
                answers[it.key] = 50 if it.kind == "slider" else it.choices[0][0]
        print("\nSESSION META:", json.dumps({k: v for k, v in plan.meta.items() if k not in ("prefix_elements", "pretreatment_answers")}))
        return

    runner = SurveyRunner(cfg, personas, args.out_root)
    runner.run(limit_agents=args.limit_agents)


if __name__ == "__main__":
    main()
