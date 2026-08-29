"""
Unified chat-completion client for open- and closed-weight models.

Providers
---------
- ``openai``     : any OpenAI-compatible ``/chat/completions`` endpoint. This covers OpenAI
                   itself, and — via ``base_url`` — vLLM, Ollama (``http://localhost:11434/v1``),
                   llama.cpp server, LM Studio, TGI, Together, OpenRouter, Groq, DeepSeek,
                   Mistral, Fireworks, Google's OpenAI-compatible endpoint, ... i.e. essentially
                   every open-weight serving stack.
- ``anthropic``  : Anthropic Messages API.
- ``dummy``      : offline stand-in that returns well-formed PoSSUM-style answers with random
                   values. For dry runs / tests of the pipeline plumbing (no network).

Only the standard library is used (``urllib``), so no SDK has to be installed.

Configuration is a plain dict (see ``configs/*.json``)::

    {
      "provider": "openai",
      "model": "gpt-4o-2024-05-13",
      "base_url": "https://api.openai.com/v1",   # optional
      "base_url_env": "QWEN_BASE_URL",           # env var holding the base URL (keeps cluster
                                                 # node IPs/ports out of tracked configs)
      "api_key_env": "OPENAI_API_KEY",           # env var holding the key (optional for local servers)
      "temperature": 0.7,
      "max_tokens": 4096,
      "timeout": 120,
      "max_retries": 8,
      "request_overrides": {}                     # merged into the JSON body (e.g. {"seed": 1})
    }
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ChatResult:
    text: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    # prompt tokens served from the KV/prefix cache (vLLM --enable-prefix-caching, OpenAI
    # automatic caching): usage.prompt_tokens_details.cached_tokens; None if not reported
    cached_tokens: int | None = None
    latency_s: float | None = None
    attempts: int = 1
    raw: dict | None = None
    finish_reason: str | None = None
    # OpenAI-format token logprobs (choices[0].logprobs.content) -- populated when the
    # request asked for them via request_overrides {"logprobs": true, "top_logprobs": N}
    logprobs: list | None = None


@dataclass
class UsageCounter:
    """Thread-safe token/call accounting across a run."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, r: ChatResult | None, failed: bool = False):
        with self._lock:
            self.calls += 1
            if failed:
                self.failures += 1
            if r is not None:
                self.input_tokens += r.input_tokens or 0
                self.output_tokens += r.output_tokens or 0
                self.cached_tokens += r.cached_tokens or 0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "failures": self.failures,
        }


class ModelError(RuntimeError):
    pass


_CTX_RE = re.compile(r"maximum context length is (\d+) tokens.*?prompt contains at least (\d+) input tokens", re.S)


def _max_tokens_that_fit(err_msg: str, margin: int = 32) -> int | None:
    """Parse a context-length error and return the largest max_tokens the prompt leaves room for."""
    m = _CTX_RE.search(err_msg)
    if not m:
        return None
    ctx, prompt = int(m.group(1)), int(m.group(2))
    return ctx - prompt - margin


class RetryableModelError(ModelError):
    pass


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
            return json.loads(payload)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover
            detail = ""
        msg = f"HTTP {e.code} from {url}: {detail[:800]}"
        if e.code in (408, 409, 425, 429, 500, 502, 503, 504, 529):
            raise RetryableModelError(msg) from e
        raise ModelError(msg) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise RetryableModelError(f"network error calling {url}: {e}") from e


class ChatClient:
    """Provider-agnostic chat client with retries + backoff.

    ``chat(messages, system=None, temperature=None, seed=None)`` -> ChatResult
    ``messages`` is a list of {"role": "user"|"assistant", "content": str}.
    """

    def __init__(self, cfg: dict, usage: UsageCounter | None = None):
        self.cfg = dict(cfg)
        self.provider = self.cfg.get("provider", "openai").lower()
        self.model = self.cfg.get("model", "dummy")
        self.temperature = self.cfg.get("temperature", 0.7)
        self.max_tokens = int(self.cfg.get("max_tokens", 4096))
        self.timeout = float(self.cfg.get("timeout", 120))
        self.max_retries = int(self.cfg.get("max_retries", 8))
        self.backoff_base = float(self.cfg.get("backoff_base", 2.0))
        self.overrides = dict(self.cfg.get("request_overrides") or {})
        self.usage = usage or UsageCounter()
        self.api_key = None
        if self.cfg.get("api_key"):
            self.api_key = self.cfg["api_key"]
        else:
            env = self.cfg.get("api_key_env") or {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(self.provider)
            if env:
                self.api_key = os.environ.get(env)
        default_base = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
        }.get(self.provider, "")
        base = self.cfg.get("base_url")
        if not base and self.cfg.get("base_url_env"):
            base = os.environ.get(self.cfg["base_url_env"])
            if not base:
                raise ValueError(f"base_url_env {self.cfg['base_url_env']!r} is set in the config "
                                 "but the environment variable is empty/unset")
        self.base_url = (base or default_base).rstrip("/")
        self.extra_headers = dict(self.cfg.get("extra_headers") or {})
        if self.provider not in {"openai", "anthropic", "dummy"}:
            raise ValueError(f"unknown provider {self.provider!r} (use openai | anthropic | dummy)")
        if self.provider == "anthropic" and not self.api_key:
            raise ValueError("anthropic provider needs an API key (api_key_env / api_key)")
        self._dummy_rng = random.Random(self.cfg.get("dummy_seed", 0))
        self._dummy_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "max_tokens": self.max_tokens}

    # ------------------------------------------------------------------ #
    def chat(self, messages: list[dict], system: str | None = None, temperature: float | None = None,
             seed: int | None = None, max_tokens: int | None = None) -> ChatResult:
        temp = self.temperature if temperature is None else temperature
        mt = max_tokens or self.max_tokens
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                if self.provider == "openai":
                    r = self._chat_openai(messages, system, temp, seed, mt)
                elif self.provider == "anthropic":
                    r = self._chat_anthropic(messages, system, temp, mt)
                else:
                    r = self._chat_dummy(messages, system, seed)
                r.latency_s = time.time() - t0
                r.attempts = attempt
                self.usage.add(r)
                return r
            except RetryableModelError as e:
                last_err = e
                sleep = min(60.0, self.backoff_base ** attempt) * (0.5 + random.random())
                time.sleep(sleep)
            except ModelError as e:
                # vLLM/OpenAI reject prompt + max_tokens > context window with HTTP 400 and a message
                # like "maximum context length is 16384 tokens. However, you requested 8192 output tokens
                # and your prompt contains at least 8193 input tokens". Shrink max_tokens to what fits
                # and retry once instead of failing the whole session.
                fit = _max_tokens_that_fit(str(e))
                if fit is not None and mt:
                    # vLLM reports "prompt contains at least N input tokens" with N = context - max_tokens + 1,
                    # i.e. a lower bound, not the real prompt length -- so `fit` alone would shrink by ~1 token
                    # per retry. Shrink geometrically as well so a few retries always reach a fitting value.
                    new_mt = max(256, min(fit, int(mt * 0.75)))
                    if new_mt < mt:
                        mt = new_mt
                        last_err = e
                        continue
                self.usage.add(None, failed=True)
                raise
        self.usage.add(None, failed=True)
        raise ModelError(f"giving up after {self.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------ #
    def _chat_openai(self, messages, system, temp, seed, max_tokens) -> ChatResult:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        body: dict = {"model": self.model, "messages": msgs}
        if temp is not None:
            body["temperature"] = temp
        if max_tokens:
            # newer OpenAI models want max_completion_tokens; most compatible servers accept max_tokens.
            body[self.cfg.get("max_tokens_field", "max_tokens")] = max_tokens
        if seed is not None and self.cfg.get("send_seed", True):
            body["seed"] = seed
        body.update(self.overrides)
        headers = dict(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = _post_json(f"{self.base_url}/chat/completions", headers, body, self.timeout)
        try:
            choice = resp["choices"][0]
            msg = choice["message"]
            text = msg.get("content") or ""
            if isinstance(text, list):  # some servers return content parts
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            # reasoning models (gpt-oss, DeepSeek-R1, Qwen3 ...) return the answer in `content` and the
            # chain of thought in `reasoning_content` / `reasoning`; only fall back to it if content is empty
            # (e.g. max_tokens hit during reasoning) so the parser at least sees something.
            if not text:
                text = msg.get("reasoning_content") or msg.get("reasoning") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"unexpected OpenAI-compatible response: {json.dumps(resp)[:800]}") from e
        usage = resp.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return ChatResult(text=text, model=resp.get("model", self.model), provider="openai",
                          input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                          cached_tokens=cached,
                          raw=resp if self.cfg.get("keep_raw") else None,
                          finish_reason=choice.get("finish_reason"),
                          logprobs=(choice.get("logprobs") or {}).get("content"))

    def _chat_anthropic(self, messages, system, temp, max_tokens) -> ChatResult:
        body: dict = {"model": self.model, "max_tokens": max_tokens, "messages": list(messages)}
        if system:
            if self.cfg.get("anthropic_cache_system", False):
                body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            else:
                body["system"] = system
        if temp is not None:
            body["temperature"] = temp
        body.update(self.overrides)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.cfg.get("anthropic_version", "2023-06-01"),
        }
        headers.update(self.extra_headers)
        resp = _post_json(f"{self.base_url}/messages", headers, body, self.timeout)
        try:
            text = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as e:
            raise ModelError(f"unexpected Anthropic response: {json.dumps(resp)[:800]}") from e
        usage = resp.get("usage") or {}
        return ChatResult(text=text, model=resp.get("model", self.model), provider="anthropic",
                          input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                          raw=resp if self.cfg.get("keep_raw") else None,
                          finish_reason=resp.get("stop_reason"))

    # ------------------------------------------------------------------ #
    _TITLE_RE = re.compile(r"^([A-Z][A-Z0-9_]+):\s*$", re.M)
    _SYMBOL_RE = re.compile(r"^([A-Za-z]+\d+)\)\s+(.*)$", re.M)

    def _chat_dummy(self, messages, system, seed) -> ChatResult:
        """Produce a well-formed answer for every TITLE found in the last user message.

        Slider titles get an integer 0-100 (drawn from a title-specific bell curve so composites
        look sane); choice titles get one of the listed symbols. Deterministic given ``seed``.
        """
        prompt = messages[-1]["content"] if messages else ""
        # only look at the item section (after the last "ITEMS" marker if present)
        marker = prompt.rfind("BEGIN SURVEY ITEMS")
        section = prompt[marker:] if marker >= 0 else prompt
        with self._dummy_lock:
            rng = random.Random(seed if seed is not None else self._dummy_rng.random())
        out = []
        blocks = re.split(r"\n(?=[A-Z][A-Z0-9_]+:\s*\n)", section)
        for blk in blocks:
            m = self._TITLE_RE.match(blk.strip("\n") + "\n")
            if not m:
                continue
            title = m.group(1)
            symbols = self._SYMBOL_RE.findall(blk)
            if symbols:  # choice item -> pick one symbol
                ans = rng.choice(symbols)[0]
            else:  # slider item -> integer 0-100
                mu = 65 if "TRUST" in title or "INST" in title else 55
                ans = str(int(min(100, max(0, rng.gauss(mu, 22)))))
            spec = int(min(100, max(0, rng.gauss(55, 20))))
            out.append(f"**title: {title}**\n**explanation: dummy client - random answer.**\n"
                       f"**answer: {ans}**\n**speculation: {spec}**")
        text = "\n\n".join(out) if out else "**title: NONE**\n**answer: 0**\n**speculation: 100**"
        return ChatResult(text=text, model="dummy", provider="dummy",
                          input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)


def make_client(cfg: dict, usage: UsageCounter | None = None) -> ChatClient:
    return ChatClient(cfg, usage=usage)


def add_endpoint_args(ap) -> None:
    """Attach the 05_infer_demographics_llm.py-style endpoint flags to an ArgumentParser."""
    ap.add_argument("--base-url", default=None,
                    help="model server URL, e.g. http://<node-ip>:<nodePort>/v1 "
                         "(overrides the config's base_url / base_url_env)")
    ap.add_argument("--api-key-env", default=None,
                    help="name of the env var holding the API key, e.g. QWEN_API_KEY "
                         "(overrides the config's api_key_env)")
    ap.add_argument("--model", default=None,
                    help="served model name (overrides the config's model for answer + cleanup)")


def apply_endpoint_args(cfg: dict, args) -> None:
    """Apply --base-url / --api-key-env / --model onto a run config, in place.

    Both the answer model and the cleanup model are overridden — pilots against a
    different endpoint should never silently mix two servers.
    """
    for section in ("model", "cleanup_model"):
        m = cfg.get(section)
        if not isinstance(m, dict):
            continue
        if getattr(args, "base_url", None):
            m["base_url"] = args.base_url
            m.pop("base_url_env", None)
        if getattr(args, "api_key_env", None):
            m["api_key_env"] = args.api_key_env
            m.pop("api_key", None)
        if getattr(args, "model", None):
            m["model"] = args.model
