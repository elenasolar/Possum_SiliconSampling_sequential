"""
One-off submission-prep utility, not a pipeline stage: derives a publication-
safe copy of stage4_survey/run_survey.py's raw call archive
(data/processed/raw_model_outputs/<run_id>/calls.jsonl) for deposit as
registration.md's K.2 item.

Why this exists (per Elena, 2026-08-28 -- see 02_DECISIONS_LOG.md): both
actual submission runs used "store_prompts": "full" (configs/run_qwen_k8s.json,
configs/run_qwen_k8s_indblocks.json), so every logged call keeps its complete
prompt text -- and every prompt embeds that persona's Reddit-derived
profile_text. That makes calls.jsonl, as generated, unsafe to publish
outright, even though K.2 (raw response logs) is required for Tier 1 (public
or escrowed). run_survey.py already has a mode built for exactly this --
_log_call()'s "store_prompts": "hash" branch SHA-256-hashes the prompt and
nulls it out -- but re-running all 17,000 sessions under that setting just to
get this is wasteful. This script applies that SAME transform after the
fact, to the calls.jsonl that already exists, so the result is byte-for-byte
what "store_prompts": "hash" would have produced from the start:

    rec["prompt_sha256"] = sha256(rec["prompt"]).hexdigest()
    rec["prompt"] = None

Every other field (response text, tokens, timing, profile_id, condition,
model, seed, ...) passes through completely unchanged -- this only ever
touches the "prompt" key. Streams line by line (calls.jsonl is ~1 GB) --
never loads the whole file into memory.

Input:
    --input   run_survey.py's raw calls.jsonl (needs a "prompt" key per record;
              records already missing it, or already null, pass through as-is)

Output:
    --output  same JSONL shape, "prompt" replaced by null + "prompt_sha256"
              on every record that had one

Usage:
    python redact_call_prompts.py \\
        --input data/processed/raw_model_outputs/qwen27b_v1_indblocks/calls.jsonl \\
        --output data/processed/raw_model_outputs/qwen27b_v1_indblocks/calls_redacted.jsonl

    python redact_call_prompts.py --selftest   # no data required
"""

import argparse
import hashlib
import json
from pathlib import Path


def redact_record(rec: dict) -> dict:
    """Returns a new dict; never mutates the input. A record with no
    "prompt" key, or "prompt": null already, passes through unchanged
    (idempotent -- safe to run twice, and tolerates a mixed file where some
    records were already logged under store_prompts: "hash")."""
    prompt = rec.get("prompt")
    if prompt is None:
        return dict(rec)
    out = dict(rec)
    out["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    out["prompt"] = None
    return out


def redact_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Returns (n_total, n_redacted)."""
    n_total = 0
    n_redacted = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)
            had_prompt = rec.get("prompt") is not None
            rec = redact_record(rec)
            if had_prompt:
                n_redacted += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return n_total, n_redacted


def selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_path = tmp / "calls.jsonl"
        prompt_text = "You are a 34-year-old from Ohio who once posted: 'I love hiking' on r/hiking."
        expected_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"profile_id": "agent_0001", "condition": "control",
                                 "prompt": prompt_text, "response": "42", "tokens": 123}) + "\n")
            # already-hashed record (e.g. from a differently-configured partial run) -- must pass through as-is
            f.write(json.dumps({"profile_id": "agent_0002", "condition": "control",
                                 "prompt": None, "prompt_sha256": "abc123", "response": "17", "tokens": 45}) + "\n")
            # record with no "prompt" key at all -- must not crash, must pass through
            f.write(json.dumps({"profile_id": "agent_0003", "condition": "control", "response": "8"}) + "\n")

        output_path = tmp / "calls_redacted.jsonl"
        n_total, n_redacted = redact_file(input_path, output_path)
        assert n_total == 3, n_total
        assert n_redacted == 1, n_redacted  # only the first record actually had a real prompt to redact

        lines = [json.loads(l) for l in output_path.read_text(encoding="utf-8").splitlines()]
        assert lines[0]["prompt"] is None
        assert lines[0]["prompt_sha256"] == expected_hash
        assert lines[0]["response"] == "42" and lines[0]["tokens"] == 123  # every other field untouched
        assert lines[1]["prompt"] is None and lines[1]["prompt_sha256"] == "abc123"  # unchanged, not re-hashed
        assert "prompt" not in lines[2]  # never had the key -- passed through untouched, none added
        assert lines[2]["response"] == "8"

        # idempotent: redacting the already-redacted output changes nothing further
        output_path2 = tmp / "calls_redacted_twice.jsonl"
        n_total2, n_redacted2 = redact_file(output_path, output_path2)
        assert n_redacted2 == 0, n_redacted2
        assert output_path2.read_text(encoding="utf-8") == output_path.read_text(encoding="utf-8")

    print("Self-test passed: prompt -> prompt_sha256 + null (matching run_survey.py's own "
          "store_prompts=\"hash\" behavior exactly), every other field untouched, already-hashed "
          "and prompt-less records pass through inertly, and re-running on already-redacted "
          "output is a no-op -- all without real data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="run_survey.py's raw calls.jsonl")
    ap.add_argument("--output", type=Path, help="Redacted copy: prompt -> null + prompt_sha256, everything else unchanged")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.input or not args.output:
        ap.error("--input and --output are required unless --selftest is given")

    n_total, n_redacted = redact_file(args.input, args.output)
    print(f"{n_total:,} call record(s); {n_redacted:,} had a prompt redacted (-> prompt_sha256, prompt=null).")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
