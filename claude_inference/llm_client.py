"""Provider-agnostic LLM calls: the same pipeline prompt runs on Claude or on GPT.

The pipeline was written directly against the Anthropic SDK. This module is the one
place that knows how a provider is called, so every Claude call in
`research_pipeline.py` (and the query decomposer in `retrieve_papers.py`) can run on
an OpenAI model instead.

**Provider is inferred from the model id** -- `gpt-*` / `o3-*` / `chatgpt-*` go to
OpenAI, everything else to Anthropic. So switching providers is just
`--model gpt-5.6-terra`, and no caller has to thread a provider argument through the
three subprocess layers (`research_loop` -> `research_pipeline_parallel` ->
`research_pipeline`) that already forward `--model`. `set_provider_override()` backs
the `--provider` escape hatch for ids inference cannot classify -- a self-hosted
OpenAI-compatible endpoint, say.

Two differences between the providers are handled here rather than at the call sites,
because getting either wrong is silent:

1. **Token budgets.** OpenAI reasoning models spend `max_completion_tokens` on hidden
   reasoning *as well as* visible output, so the pipeline's 800-3000 budgets -- sized
   for Claude, where they only cover the answer -- truncate to empty content. Budgets
   are scaled up (see `budget_tokens`). Output is billed on tokens actually produced,
   so a generous ceiling costs nothing.

2. **What `input_tokens` means.** Anthropic reports cache reads *separately* from
   `input_tokens`; OpenAI's `prompt_tokens` *includes* its `cached_tokens`. Normalizing
   to Anthropic's convention (see `normalize_usage`) is what lets one `price_call`
   formula bill both -- without the subtraction, cached OpenAI input would be charged
   at both the full and the discounted rate.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

ANTHROPIC = "anthropic"
OPENAI = "openai"
PROVIDERS = (ANTHROPIC, OPENAI)

# Per-million-token pricing in USD (input, output).
# Claude: Anthropic pricing pages, verified for the 4.x family.
# GPT: the gpt-5.6 tiers, same numbers filter_queries.py bills against (it imports
# this table). Cache writes are billed at 1.25x input, cache hits at 0.10x -- both
# providers discount cached input at 0.10x; only Anthropic charges to write it.
# Extend this table when adding new models.
MODEL_PRICING = {
    "claude-opus-4-8":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-7":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-6":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-5":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-1":   {"input": 15.00, "output": 75.00},
    "claude-opus-4-0":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00},
    "claude-sonnet-5":   {"input": 3.00,  "output": 15.00},
    "gpt-5.6-luna":      {"input": 0.20,  "output": 1.20},
    "gpt-5.6-terra":     {"input": 2.00,  "output": 12.00},
    "gpt-5.6-sol":       {"input": 4.00,  "output": 20.00},
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

# Model ids that route to OpenAI. Anything unmatched falls through to Anthropic,
# which keeps every pre-existing invocation behaving exactly as it did.
_OPENAI_MODEL_RE = re.compile(r"^(gpt|o[1-9](-|$)|chatgpt|davinci|babbage)", re.I)

# OpenAI reasoning-token headroom (see the module docstring). A budget sized for
# Claude's visible output is multiplied by HEADROOM and floored at MIN, because
# reasoning alone can consume several thousand tokens before a single visible one.
OPENAI_TOKEN_HEADROOM = 4.0
OPENAI_MIN_MAX_TOKENS = 8000

_provider_override: Optional[str] = None
_reasoning_effort: Optional[str] = None
_PRICING_WARNED: set = set()


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def set_provider_override(provider: Optional[str]) -> None:
    """Force a provider for every subsequent call, regardless of model id.

    Set once from a CLI's `--provider` (`auto`/None clears it). Process-global on
    purpose: the parallel driver isolates seeds in subprocesses, and the threaded
    entry points (`verify_questions.py`) use one provider for all their workers.
    """
    global _provider_override
    if provider in (None, "auto"):
        _provider_override = None
        return
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
    _provider_override = provider


def set_reasoning_effort(effort: Optional[str]) -> None:
    """Set `reasoning_effort` for OpenAI calls. None leaves the model default."""
    global _reasoning_effort
    _reasoning_effort = effort or None


def resolve_provider(model: str, provider: Optional[str] = None) -> str:
    """Which provider serves `model`.

    Precedence: explicit `provider` argument, then the `--provider` override, then
    the model id. Unrecognized ids resolve to Anthropic.
    """
    if provider not in (None, "auto"):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
        return provider
    if _provider_override is not None:
        return _provider_override
    return OPENAI if _OPENAI_MODEL_RE.match(model or "") else ANTHROPIC


API_KEY_ENV = {ANTHROPIC: "ANTHROPIC_API_KEY", OPENAI: "OPENAI_API_KEY"}


def api_key_env(model: str, provider: Optional[str] = None) -> str:
    """Name of the env var whose key is needed to serve `model`."""
    return API_KEY_ENV[resolve_provider(model, provider)]


def make_client(model: str, provider: Optional[str] = None):
    """Construct the SDK client for `model`'s provider.

    Both SDKs read their key from the environment, so the returned client carries no
    explicit credential. `call_text`/`call_json_schema` need the same provider string
    that produced the client -- pass `resolve_provider(model)` to both.
    """
    p = resolve_provider(model, provider)
    if p == ANTHROPIC:
        from anthropic import Anthropic
        return Anthropic()
    from openai import OpenAI
    return OpenAI()


def budget_tokens(provider: str, max_tokens: int) -> int:
    """Token budget for `provider`, given a budget sized for Claude's visible output.

    Anthropic's `max_tokens` caps visible output only, so it passes through. OpenAI's
    `max_completion_tokens` also has to cover hidden reasoning; see the module
    docstring for why passing the Claude number through there yields empty responses.
    """
    if provider == ANTHROPIC:
        return max_tokens
    return max(OPENAI_MIN_MAX_TOKENS, int(max_tokens * OPENAI_TOKEN_HEADROOM))


# ---------------------------------------------------------------------------
# Usage & pricing
# ---------------------------------------------------------------------------

def normalize_usage(resp, provider: str) -> dict:
    """Token usage from a provider response, in Anthropic's field names.

    Anthropic reports cache reads outside `input_tokens`; OpenAI folds `cached_tokens`
    into `prompt_tokens`. The cached count is subtracted out here so `input_tokens`
    means "uncached input" for both, which is what `price_call`'s formula assumes.
    `reasoning_tokens` is carried through for OpenAI (billed as output, but worth
    seeing separately in the log when a call comes back truncated).
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return {}

    if provider == ANTHROPIC:
        return {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        }

    prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
    usage = {
        # max(0, ...) guards the (unexpected) case of cached > prompt rather than
        # letting a negative token count credit the run's cost.
        "input_tokens": max(0, prompt_tokens - cached),
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "cache_creation_input_tokens": 0,   # OpenAI caches implicitly and bills no write
        "cache_read_input_tokens": cached,
    }
    if reasoning:
        usage["reasoning_tokens"] = reasoning
    return usage


def price_call(model: str, usage: dict) -> tuple[float, dict]:
    """Compute cost in USD for one LLM call. Returns (cost, normalized_usage).

    `usage` must already be in Anthropic's field names -- pass it through
    `normalize_usage` first for an OpenAI response. Unknown models are priced at $0
    with a one-time warning printed to stderr.
    """
    rates = MODEL_PRICING.get(model)
    if rates is None:
        if model not in _PRICING_WARNED:
            print(
                f"[cost] WARNING: no pricing entry for model {model!r}; "
                "cost will be reported as $0. Add it to llm_client.MODEL_PRICING.",
                file=sys.stderr,
            )
            _PRICING_WARNED.add(model)
        rates = {"input": 0.0, "output": 0.0}

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)

    cost = (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_write * rates["input"] * CACHE_WRITE_MULTIPLIER
        + cache_read * rates["input"] * CACHE_READ_MULTIPLIER
    ) / 1_000_000

    normalized = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
    }
    if usage.get("reasoning_tokens"):
        normalized["reasoning_tokens"] = int(usage["reasoning_tokens"])
    return cost, normalized


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def _openai_messages(system: Optional[str], messages: list) -> list:
    """Anthropic's separate `system` argument becomes a leading system message."""
    return ([{"role": "system", "content": system}] if system is not None else []) + list(messages)


def _openai_kwargs(model: str, max_tokens: int) -> dict:
    kwargs = {"model": model, "max_completion_tokens": budget_tokens(OPENAI, max_tokens)}
    if _reasoning_effort:
        kwargs["reasoning_effort"] = _reasoning_effort
    return kwargs


def _openai_text(resp, model: str) -> str:
    """Visible text from a chat completion, with the failure modes named.

    A reasoning model that burns its whole budget thinking returns
    finish_reason="length" and empty content; saying so beats "empty response".
    """
    choice = resp.choices[0]
    if getattr(choice.message, "refusal", None):
        raise RuntimeError(f"{model} refused: {choice.message.refusal}")
    text = choice.message.content or ""
    if not text.strip():
        if choice.finish_reason == "length":
            reasoning = getattr(
                getattr(resp.usage, "completion_tokens_details", None), "reasoning_tokens", 0
            ) or 0
            raise RuntimeError(
                f"{model} produced no visible output: the token budget "
                f"({resp.usage.completion_tokens} used, {reasoning} on reasoning) went "
                f"entirely to reasoning. Raise llm_client.OPENAI_MIN_MAX_TOKENS or lower "
                f"--reasoning-effort."
            )
        raise RuntimeError(
            f"empty response from {model} (finish_reason={choice.finish_reason!r})"
        )
    return text


class EmptyResponse(RuntimeError):
    """A response arrived but carried no usable text. Holds the usage already billed."""

    def __init__(self, message: str, usage: Optional[dict] = None):
        super().__init__(message)
        self.usage = usage or {}


def call_text(
    client,
    provider: str,
    *,
    model: str,
    system: Optional[str],
    messages: list,
    max_tokens: int,
) -> tuple[str, dict]:
    """One completion. Returns (text, usage-in-Anthropic-field-names).

    Raises on a refusal or an empty/truncated response, so a caller that goes on to
    parse JSON gets a diagnosable error instead of a parse failure. Usage is returned
    even on those raises' behalf only when the response arrived -- a transport error
    propagates with nothing, as before.
    """
    if provider == ANTHROPIC:
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            # Cache the (static, reused) system prompt so repeated calls within the
            # 5-min TTL read it at 0.10x instead of full input price. Only the harder-
            # question generator passes a system prompt; the judge passes None.
            kwargs["system"] = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        resp = client.messages.create(**kwargs)
        usage = normalize_usage(resp, ANTHROPIC)
        # `content` is empty on a refusal and can lead with a non-text block, so join the
        # text blocks rather than indexing: content[0].text raised IndexError on refusals
        # and buried the cause as "list index out of range".
        text = "".join(getattr(b, "text", "") for b in resp.content)
        if not text.strip():
            raise EmptyResponse(
                f"empty response from {model} "
                f"(stop_reason={getattr(resp, 'stop_reason', None)!r})",
                usage=usage,
            )
        return text, usage

    if provider == OPENAI:
        # No explicit cache_control: OpenAI caches prompt prefixes over ~1024 tokens
        # automatically and reports the hits as `cached_tokens`, which normalize_usage
        # bills at the same 0.10x. Putting `system` first keeps that prefix stable.
        resp = client.chat.completions.create(
            messages=_openai_messages(system, messages),
            **_openai_kwargs(model, max_tokens),
        )
        usage = normalize_usage(resp, OPENAI)
        try:
            return _openai_text(resp, model), usage
        except RuntimeError as e:
            raise EmptyResponse(str(e), usage=usage) from None

    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def call_json_schema(
    client,
    provider: str,
    *,
    model: str,
    system: Optional[str],
    user: str,
    schema: dict,
    schema_name: str,
    max_tokens: int,
) -> tuple[str, dict]:
    """One completion constrained to `schema`. Returns (json_text, usage).

    `schema` must be strict-mode compatible for both providers: every property listed
    in `required`, and `additionalProperties: false`.
    """
    if provider == ANTHROPIC:
        req = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        output_config = {"format": {"type": "json_schema", "schema": schema}}
        try:
            resp = client.messages.create(output_config=output_config, **req)
        except TypeError:
            # SDKs older than ~0.6x have no output_config kwarg (anthropic 0.54 raises
            # "unexpected keyword argument"); extra_body puts it in the raw request body,
            # which the API accepts either way.
            resp = client.messages.create(extra_body={"output_config": output_config}, **req)
        # Every failure below raises EmptyResponse rather than a bare RuntimeError, so a
        # caller with a fallback path (retrieve_papers' decomposer) can still bill the
        # tokens the request already spent.
        usage = normalize_usage(resp, ANTHROPIC)
        if resp.stop_reason == "refusal":
            raise EmptyResponse(f"{model} refused the structured request", usage=usage)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        if not text.strip():
            raise EmptyResponse(
                f"no text block in {model}'s structured response "
                f"(stop_reason={resp.stop_reason!r})", usage=usage,
            )
        return text, usage

    if provider == OPENAI:
        resp = client.chat.completions.create(
            messages=_openai_messages(system, [{"role": "user", "content": user}]),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            **_openai_kwargs(model, max_tokens),
        )
        usage = normalize_usage(resp, OPENAI)
        try:
            return _openai_text(resp, model), usage
        except RuntimeError as e:
            raise EmptyResponse(str(e), usage=usage) from None

    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def add_provider_arg(parser, *, dest_help_extra: str = "") -> None:
    """Add the shared `--provider` / `--reasoning-effort` flags to an argparse parser."""
    parser.add_argument(
        "--provider", choices=["auto", ANTHROPIC, OPENAI], default="auto",
        help="Which API serves --model. 'auto' (default) infers it from the model id: "
             "gpt-*/o3-* -> openai, everything else -> anthropic. Set it explicitly only "
             "for an id the inference cannot classify." + dest_help_extra,
    )
    parser.add_argument(
        "--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None,
        help="OpenAI models only: reasoning_effort for every call (default: the model's "
             "own default). Ignored by Anthropic models.",
    )


def configure_from_args(args) -> None:
    """Apply `--provider` / `--reasoning-effort` from a parsed argparse namespace."""
    set_provider_override(getattr(args, "provider", None))
    set_reasoning_effort(getattr(args, "reasoning_effort", None))


def decomposer_hint(model: str, decomposer: str) -> str:
    """Advice appended when the decomposer's provider differs from the main model's.

    Only reachable via an explicit --decomposer-model, since the default follows --model
    (retrieve_papers.default_decomposer_model). The user chose this deliberately, so the
    advice is about the missing key, not about picking a different decomposer.
    """
    want, have = resolve_provider(decomposer), resolve_provider(model)
    if want == have:
        return ""
    return (f" --decomposer-model {decomposer!r} is on {want} while --model {model!r} is "
            f"on {have}, so this run needs both keys — export the one above, or drop "
            f"--decomposer-model to keep the run on {have} alone.")


def forward_provider_args(args) -> list[str]:
    """The provider flags on `args` as CLI tokens, for forwarding to a subprocess.

    research_loop -> research_pipeline_parallel -> research_pipeline, and
    research_loop -> verify_questions, each re-forward this same set, so the shape
    lives here instead of in three copies that can drift -- which they had, over
    whether a `--provider auto` is worth sending at all.

    Defaults are omitted rather than passed explicitly, which keeps the subprocess
    command lines short and makes a non-default choice visible in them. Read with
    getattr so a parser that declares only some of these (eval_other_model.py has no
    decomposer) still works.
    """
    out: list[str] = []
    provider = getattr(args, "provider", None)
    if provider and provider != "auto":
        out += ["--provider", provider]
    if getattr(args, "reasoning_effort", None):
        out += ["--reasoning-effort", args.reasoning_effort]
    if getattr(args, "decomposer_model", None):
        out += ["--decomposer-model", args.decomposer_model]
    return out


def require_api_key(model: str, provider: Optional[str] = None) -> Optional[str]:
    """Return an error message if the key for `model`'s provider is missing, else None."""
    env = api_key_env(model, provider)
    if os.environ.get(env):
        return None
    return f"{env} environment variable is not set (needed for model {model!r})."
