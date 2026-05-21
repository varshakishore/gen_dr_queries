#!/usr/bin/env python3
"""
Pipeline for generating hard research questions, evaluating answers, and iterating.

Flow per seed question:
  1. Use Claude to harden the seed question (with prior attempts as context after round 1).
  2. Send the new question to a drtulu server at localhost:8007/ask.
  3. Use Claude to judge the answer against the verification criterion.
  4. If the answer FAILED, stop (we found a question that breaks the system).
     Otherwise, loop back to step 1 with feedback, up to 5 attempts total.

Logging:
  Every external call (Claude + research server) is appended as a JSONL record to
  the run log file. Each record includes timestamps, latency, full request/response,
  token usage, and per-call cost so the run can be inspected after the fact.

Cost tracking:
  Token usage from each Claude call is multiplied by per-model rates and accumulated.
  A running total is kept per seed and globally, and is included in the final summary.
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import asyncio
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The workflow lives in this repo path; we add it to sys.path so we can import it.
WORKFLOW_REPO_PATH = "/weka/nora-default/varshak/dr-tulu/agent"
WORKFLOW_CONFIG_PATH = (
    "/weka/nora-default/varshak/dr-tulu/agent/workflows/"
    "auto_search_sft_s2_only_hamish.yaml"
)
DATASET_NAME = "sqav2"

if WORKFLOW_REPO_PATH not in sys.path:
    sys.path.insert(0, WORKFLOW_REPO_PATH)

# Imported lazily inside _get_workflow_class() so `--help` and unit tests work
# even when the workflow repo isn't on this machine.
_AutoReasonSearchWorkflow = None


def _get_workflow_class():
    global _AutoReasonSearchWorkflow
    if _AutoReasonSearchWorkflow is None:
        from workflows.auto_search_sft import AutoReasonSearchWorkflow  # type: ignore
        _AutoReasonSearchWorkflow = AutoReasonSearchWorkflow
    return _AutoReasonSearchWorkflow

CLAUDE_MODEL = "claude-opus-4-5"
MAX_ATTEMPTS = 5

# Per-million-token pricing in USD (input, output).
# Sources: Anthropic pricing pages, verified for the 4.x family.
# Cache writes are billed at 1.25x input; cache hits at 0.10x input.
# Extend this table when adding new models.
MODEL_PRICING = {
    "claude-opus-4-7":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-6":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-5":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-1":   {"input": 15.00, "output": 75.00},
    "claude-opus-4-0":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00},
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


HARDENING_PROMPT = """You are an expert in constructing challenging research questions.

Given a seed question, produce an updated question designed to expose weaknesses in deep research systems.

RULES:
- Avoid "what" questions that merely retrieve information.
- The updated question MUST require higher-order thinking: analysis, comparison, evaluation, or synthesis.
- It must NOT be answerable with a definition, summary, or single-source response.
- The updated question length should change by fewer than 10 words from the seed.

EXAMPLE STRATEGIES TO CONSIDER:
1. Require synthesis across 5+ sources or clearly disjoint domains (e.g., science + economics).
2. Require synthesis across differing viewpoints, stakeholder incentives, or theoretical frameworks.
3. Require multi-step reasoning, structured argumentation, or hierarchical planning.
4. Require handling conflicting, incomplete, or low-quality evidence.
5. Require correcting a hidden misconception or establishing key knowns before answering.
6. Ask for a specific "moment of truth" — a concrete case highlighting consequences and lessons learned.
7. Embed a specific context that changes the answer (e.g., "explain to a policymaker with no ML background").

OUTPUT FORMAT (valid JSON, no extra text):
{
  "brainstorming": "<think about distinct strategies and reason about why they may or may not work>",
  "chosen_strategy": "<name and justify the single most promising strategy>",
  "updated_question": "<the rewritten question>",
  "why_harder": "<explanation of why this question might be hard for a deep research system>",
  "verification_criterion": "<one concrete, testable criterion for checking whether the answer is good>"
}"""

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of deep research system outputs.

You will be given:
- A hardened research question
- The verification criterion that defines what a good answer must do
- The answer the research system produced

Your job is to judge whether the answer satisfies the verification criterion, and to flag other issues you notice (factual errors, hallucinations, evasion, missing reasoning, structural problems, etc.) even if those issues are not part of the criterion.

QUESTION:
{question}

VERIFICATION CRITERION:
{criterion}

ANSWER:
{answer}

OUTPUT FORMAT (valid JSON, no extra text):
{{
  "criterion_satisfied": <true | false>,
  "criterion_reasoning": "<why the answer does or does not satisfy the criterion>",
  "other_issues": ["<issue 1>", "<issue 2>", ...],
  "summary": "<2-4 sentence overall summary of how the answer performed and what to push on next time>",
  "verdict": "<PASSED | FAILED>"
}}

PASSED means: the criterion is satisfied AND there are no critical issues.
FAILED means: the criterion is not satisfied OR there are serious problems (hallucinations, refusals, off-topic).
"""


# ---------------------------------------------------------------------------
# Logger — JSONL append-only file, one record per external call
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class RunLogger:
    """Append-only JSONL logger. Every call_* method writes a single line."""

    def __init__(self, log_path: Path, run_id: str):
        self.log_path = log_path
        self.run_id = run_id
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write({
            "kind": "run_start",
            "run_id": run_id,
            "timestamp": _now_iso(),
        })

    def _write(self, record: dict) -> None:
        record.setdefault("run_id", self.run_id)
        record.setdefault("timestamp", _now_iso())
        with self.log_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_claude_call(
        self,
        *,
        seed: str,
        attempt: int,
        purpose: str,           # "harden" or "judge"
        model: str,
        system: Optional[str],
        messages: list,
        response_text: str,
        usage: dict,
        cost_usd: float,
        latency_s: float,
        error: Optional[str] = None,
    ) -> None:
        self._write({
            "kind": "claude_call",
            "seed": seed,
            "attempt": attempt,
            "purpose": purpose,
            "model": model,
            "system": system,
            "messages": messages,
            "response_text": response_text,
            "usage": usage,
            "cost_usd": cost_usd,
            "latency_s": latency_s,
            "error": error,
        })

    def log_research_call(
        self,
        *,
        seed: str,
        attempt: int,
        url: str,
        request_json: dict,
        response_status: Optional[int],
        response_body,  # str | dict | None — depends on transport
        latency_s: float,
        error: Optional[str] = None,
    ) -> None:
        self._write({
            "kind": "research_call",
            "seed": seed,
            "attempt": attempt,
            "url": url,
            "request_json": request_json,
            "response_status": response_status,
            "response_body": response_body,
            "latency_s": latency_s,
            "error": error,
        })

    def log_verdict(self, *, seed: str, attempt: int, verdict: str, summary: str) -> None:
        self._write({
            "kind": "verdict",
            "seed": seed,
            "attempt": attempt,
            "verdict": verdict,
            "summary": summary,
        })

    def log_run_end(self, totals: dict) -> None:
        self._write({"kind": "run_end", "totals": totals})


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------

@dataclass
class CostBucket:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: "CostBucket") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cost_usd += other.cost_usd
        self.calls += other.calls


_PRICING_WARNED: set = set()


def price_call(model: str, usage: dict) -> tuple[float, dict]:
    """Compute cost in USD for one Claude call. Returns (cost, normalized_usage).

    `usage` mirrors the fields on the Anthropic SDK's Usage object:
      input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens.
    Unknown models are priced at $0 with a one-time warning printed to stderr.
    """
    rates = MODEL_PRICING.get(model)
    if rates is None:
        if model not in _PRICING_WARNED:
            print(
                f"[cost] WARNING: no pricing entry for model {model!r}; "
                "cost will be reported as $0. Add it to MODEL_PRICING.",
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

    return cost, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HardenedQuestion:
    brainstorming: str
    chosen_strategy: str
    updated_question: str
    why_harder: str
    verification_criterion: str
    raw: str = ""


@dataclass
class Judgment:
    criterion_satisfied: bool
    criterion_reasoning: str
    other_issues: list
    summary: str
    verdict: str
    raw: str = ""


@dataclass
class AttemptRecord:
    attempt: int
    hardened: HardenedQuestion
    answer: str
    judgment: Judgment


@dataclass
class SeedResult:
    seed: str
    attempts: list = field(default_factory=list)
    final_status: str = ""
    error: Optional[str] = None
    cost: CostBucket = field(default_factory=CostBucket)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a string, tolerating code fences."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in text:\n{text}")
        candidate = text[start : end + 1]
    return json.loads(candidate)


# ---------------------------------------------------------------------------
# Claude calls
# ---------------------------------------------------------------------------

def _call_claude(
    client: Anthropic,
    *,
    model: str,
    system: Optional[str],
    messages: list,
    max_tokens: int,
    logger: RunLogger,
    seed: str,
    attempt: int,
    purpose: str,
) -> tuple[str, CostBucket]:
    """Single Claude call wrapped with logging + cost accounting."""
    t0 = time.perf_counter()
    err: Optional[str] = None
    response_text = ""
    usage: dict = {}
    cost_usd = 0.0

    try:
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        response_text = resp.content[0].text
        usage_obj = getattr(resp, "usage", None)
        if usage_obj is not None:
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", 0),
                "output_tokens": getattr(usage_obj, "output_tokens", 0),
                "cache_creation_input_tokens": getattr(
                    usage_obj, "cache_creation_input_tokens", 0
                ),
                "cache_read_input_tokens": getattr(
                    usage_obj, "cache_read_input_tokens", 0
                ),
            }
        cost_usd, usage = price_call(model, usage)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        logger.log_claude_call(
            seed=seed, attempt=attempt, purpose=purpose, model=model,
            system=system, messages=messages, response_text=response_text,
            usage=usage, cost_usd=cost_usd, latency_s=latency, error=err,
        )
        raise

    latency = time.perf_counter() - t0
    logger.log_claude_call(
        seed=seed, attempt=attempt, purpose=purpose, model=model,
        system=system, messages=messages, response_text=response_text,
        usage=usage, cost_usd=cost_usd, latency_s=latency, error=None,
    )

    bucket = CostBucket(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cost_usd=cost_usd,
        calls=1,
    )
    return response_text, bucket


def harden_question(
    client: Anthropic,
    model: str,
    seed: str,
    prior_attempts: list,
    logger: RunLogger,
    attempt: int,
) -> tuple[HardenedQuestion, CostBucket]:
    user_content = f"Seed question: {seed}"
    if prior_attempts:
        feedback_blocks = []
        for rec in prior_attempts:
            feedback_blocks.append(
                f"--- Previous attempt {rec.attempt} ---\n"
                f"Question tried: {rec.hardened.updated_question}\n"
                f"Strategy: {rec.hardened.chosen_strategy}\n"
                f"Verification criterion: {rec.hardened.verification_criterion}\n"
                f"Judge verdict: {rec.judgment.verdict}\n"
                f"Judge summary: {rec.judgment.summary}\n"
                f"Other issues flagged: {rec.judgment.other_issues}\n"
            )
        user_content += (
            "\n\nThe research system PASSED the previous hardened versions of this seed, "
            "which means those questions were not hard enough. MAKE THE QUESTION HARDER "
            "this time. Pick a meaningfully different strategy — do not just rephrase a "
            "prior attempt — and target a weakness the previous attempts did not exploit. "
            "Below is what was tried:\n\n"
            + "\n".join(feedback_blocks)
        )

    messages = [{"role": "user", "content": user_content}]
    raw, bucket = _call_claude(
        client, model=model, system=HARDENING_PROMPT, messages=messages,
        max_tokens=2000, logger=logger, seed=seed, attempt=attempt, purpose="harden",
    )
    data = extract_json(raw)
    return (
        HardenedQuestion(
            brainstorming=data.get("brainstorming", ""),
            chosen_strategy=data.get("chosen_strategy", ""),
            updated_question=data["updated_question"],
            why_harder=data.get("why_harder", ""),
            verification_criterion=data["verification_criterion"],
            raw=raw,
        ),
        bucket,
    )


def judge_answer(
    client: Anthropic,
    model: str,
    question: str,
    criterion: str,
    answer: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
) -> tuple[Judgment, CostBucket]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, criterion=criterion, answer=answer
    )
    messages = [{"role": "user", "content": prompt}]
    raw, bucket = _call_claude(
        client, model=model, system=None, messages=messages,
        max_tokens=2000, logger=logger, seed=seed, attempt=attempt, purpose="judge",
    )
    data = extract_json(raw)
    verdict = str(data.get("verdict", "")).upper()
    if verdict not in ("PASSED", "FAILED"):
        verdict = "PASSED" if data.get("criterion_satisfied") else "FAILED"
    return (
        Judgment(
            criterion_satisfied=bool(data.get("criterion_satisfied", False)),
            criterion_reasoning=data.get("criterion_reasoning", ""),
            other_issues=data.get("other_issues", []),
            summary=data.get("summary", ""),
            verdict=verdict,
            raw=raw,
        ),
        bucket,
    )


# ---------------------------------------------------------------------------
# Research workflow call
# ---------------------------------------------------------------------------

# Cache the workflow instance — config loading isn't free, and reusing the
# instance across seeds is the same pattern the user's snippet implies.
_workflow_instance = None


def _get_workflow():
    global _workflow_instance
    if _workflow_instance is None:
        cls = _get_workflow_class()
        _workflow_instance = cls(configuration=WORKFLOW_CONFIG_PATH)
    return _workflow_instance


async def _run_workflow_async(question: str, dataset_name: str) -> dict:
    workflow = _get_workflow()
    return await workflow(
        problem=question,
        dataset_name=dataset_name,
        verbose=False,
    )


def _run_workflow_sync(question: str, dataset_name: str) -> dict:
    """Run the workflow on a fresh event loop, then drain pending async
    resources before closing.

    asyncio.run() closes the loop immediately after the coroutine returns,
    which leaves aiohttp connectors and client sessions inside long-lived
    MCP tools (e.g. Crawl4AIBrowseTool) un-awaited and produces "Unclosed
    connector" warnings. The fix is to (1) keep the loop alive long enough
    to run shutdown_asyncgens, (2) give pending close callbacks one tick
    to flush, and (3) close cleanly.
    """
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run_workflow_async(question, dataset_name))
        # Drain async generators created by the workflow's HTTP clients.
        loop.run_until_complete(loop.shutdown_asyncgens())
        # One extra tick lets any deferred close callbacks (TCPConnector, etc.)
        # complete before the loop is torn down.
        loop.run_until_complete(asyncio.sleep(0))
        return result
    finally:
        loop.close()


def query_research_system(
    question: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
    dataset_name: str = DATASET_NAME,
) -> str:
    """Run the AutoReasonSearchWorkflow and return the final response string.

    The full trace and the final response are logged via log_research_call.
    """
    request_meta = {
        "problem": question,
        "dataset_name": dataset_name,
        "configuration": WORKFLOW_CONFIG_PATH,
    }
    t0 = time.perf_counter()
    err: Optional[str] = None
    final_response = ""
    trace_dump = None

    try:
        result = _run_workflow_sync(question, dataset_name)
        final_response = result["final_response"]
        ft = result.get("full_traces")
        trace_dump = ft.model_dump() if ft is not None and hasattr(ft, "model_dump") else ft
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        logger.log_research_call(
            seed=seed, attempt=attempt, url=f"workflow:{WORKFLOW_CONFIG_PATH}",
            request_json=request_meta, response_status=None,
            response_body=None, latency_s=latency, error=err,
        )
        raise

    latency = time.perf_counter() - t0
    # We pack final_response + trace into response_body so a single log record
    # captures everything the workflow returned. trace_dump can be large; we
    # serialize to JSON-compatible structures (the logger handles dataclasses
    # via default=str, but model_dump() already returns plain dicts).
    logger.log_research_call(
        seed=seed, attempt=attempt, url=f"workflow:{WORKFLOW_CONFIG_PATH}",
        request_json=request_meta, response_status=None,
        response_body={"final_response": final_response, "trace": trace_dump},
        latency_s=latency, error=None,
    )
    return final_response


# ---------------------------------------------------------------------------
# Main per-seed loop
# ---------------------------------------------------------------------------

def process_seed(
    client: Anthropic,
    model: str,
    seed: str,
    logger: RunLogger,
    max_attempts: int = MAX_ATTEMPTS,
    verbose: bool = True,
    dataset_name: str = DATASET_NAME,
) -> SeedResult:
    result = SeedResult(seed=seed)

    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"SEED: {seed!r}  |  Attempt {attempt}/{max_attempts}")
            print("=" * 70)

        # Step 1 — harden
        try:
            hardened, bucket = harden_question(
                client, model, seed, result.attempts, logger, attempt
            )
            result.cost.add(bucket)
        except Exception as e:
            result.final_status = "ERROR"
            result.error = f"Hardening failed: {e}"
            if verbose:
                print(f"[ERROR] Hardening failed: {e}")
            return result

        if verbose:
            print(f"[1] Hardened question: {hardened.updated_question}")
            print(f"    Strategy: {hardened.chosen_strategy}")
            print(f"    Criterion: {hardened.verification_criterion}")

        # Step 2 — query research system
        try:
            answer = query_research_system(
                hardened.updated_question, logger, seed, attempt,
                dataset_name=dataset_name,
            )
        except Exception as e:
            result.final_status = "ERROR"
            result.error = f"Research workflow call failed: {e}"
            if verbose:
                print(f"[ERROR] Research workflow call failed: {e}")
            return result

        if verbose:
            preview = answer.replace("\n", " ")
            preview = preview[:240] + ("…" if len(preview) > 240 else "")
            print(f"[2] Answer (preview): {preview}")

        # Step 3 — judge
        try:
            judgment, bucket = judge_answer(
                client, model, hardened.updated_question,
                hardened.verification_criterion, answer, logger, seed, attempt,
            )
            result.cost.add(bucket)
        except Exception as e:
            result.final_status = "ERROR"
            result.error = f"Judging failed: {e}"
            if verbose:
                print(f"[ERROR] Judging failed: {e}")
            return result

        logger.log_verdict(
            seed=seed, attempt=attempt,
            verdict=judgment.verdict, summary=judgment.summary,
        )

        if verbose:
            print(f"[3] Verdict: {judgment.verdict}")
            print(f"    Summary: {judgment.summary}")
            if judgment.other_issues:
                print(f"    Other issues: {judgment.other_issues}")
            print(
                f"    [cost so far on this seed: ${result.cost.cost_usd:.4f} "
                f"across {result.cost.calls} Claude calls]"
            )

        result.attempts.append(
            AttemptRecord(
                attempt=attempt, hardened=hardened, answer=answer, judgment=judgment,
            )
        )

        if judgment.verdict == "FAILED":
            result.final_status = "FAILED_FOUND"
            if verbose:
                print(f"\n>>> FAILED answer found on attempt {attempt}. Stopping.")
            return result

    result.final_status = "EXHAUSTED"
    if verbose:
        print(f"\n>>> Exhausted {max_attempts} attempts without producing a failing answer.")
    return result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def result_to_dict(result: SeedResult) -> dict:
    return {
        "seed": result.seed,
        "final_status": result.final_status,
        "error": result.error,
        "cost": asdict(result.cost),
        "attempts": [
            {
                "attempt": a.attempt,
                "hardened": asdict(a.hardened),
                "answer": a.answer,
                "judgment": asdict(a.judgment),
            }
            for a in result.attempts
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global WORKFLOW_CONFIG_PATH
    parser = argparse.ArgumentParser(
        description="Harden seed questions, query a research server, and judge answers."
    )
    parser.add_argument("seeds", nargs="*",
                        help="Seed questions. If omitted, reads from --seeds-file or stdin.")
    parser.add_argument("--seeds-file", help="Path to a file with one seed per line.")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                        help=f"Max hardening attempts per seed (default: {MAX_ATTEMPTS}).")
    parser.add_argument("--model", default=CLAUDE_MODEL,
                        help=f"Anthropic model (default: {CLAUDE_MODEL}).")
    parser.add_argument("--output", help="Optional path to write final JSON results.")
    parser.add_argument(
        "--log-dir", default="./logs",
        help="Directory for the JSONL run log (default: ./logs). "
             "A file named run-<run_id>.jsonl will be created.",
    )
    parser.add_argument("--run-id", help="Optional run identifier; auto-generated if omitted.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-attempt output.")
    parser.add_argument(
        "--dataset-name", default=DATASET_NAME,
        help=f"Dataset name passed to the workflow (default: {DATASET_NAME}).",
    )
    parser.add_argument(
        "--workflow-config", default=WORKFLOW_CONFIG_PATH,
        help="Path to the AutoReasonSearchWorkflow YAML config.",
    )
    args = parser.parse_args()

    # Allow CLI override of the workflow config before the workflow is built
    WORKFLOW_CONFIG_PATH = args.workflow_config

    seeds = list(args.seeds)
    if args.seeds_file:
        with open(args.seeds_file) as f:
            seeds.extend(line.strip() for line in f if line.strip())
    if not seeds and not sys.stdin.isatty():
        seeds.extend(line.strip() for line in sys.stdin if line.strip())
    if not seeds:
        parser.error("No seed questions provided.")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY environment variable is not set.")

    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log_path = Path(args.log_dir) / f"run-{run_id}.jsonl"
    logger = RunLogger(log_path, run_id)
    print(f"Run ID: {run_id}")
    print(f"Logging every call to: {log_path}")

    client = Anthropic()

    all_results: list[SeedResult] = []
    grand_total = CostBucket()

    for seed in seeds:
        result = process_seed(
            client, args.model, seed, logger,
            max_attempts=args.max_attempts, verbose=not args.quiet,
            dataset_name=args.dataset_name,
        )
        all_results.append(result)
        grand_total.add(result.cost)

    print(f"\n{'#' * 70}")
    print("FINAL SUMMARY")
    print("#" * 70)
    for r in all_results:
        n = len(r.attempts)
        print(
            f"- [{r.final_status:14}] ({n} attempts, ${r.cost.cost_usd:.4f}, "
            f"{r.cost.calls} Claude calls) {r.seed}"
        )
        if r.error:
            print(f"    error: {r.error}")
        elif r.final_status == "FAILED_FOUND":
            last = r.attempts[-1]
            print(f"    failing question: {last.hardened.updated_question}")
            print(f"    judge summary: {last.judgment.summary}")

    print(
        f"\nTotal cost: ${grand_total.cost_usd:.4f} "
        f"({grand_total.calls} Claude calls, "
        f"{grand_total.input_tokens:,} input + {grand_total.output_tokens:,} output tokens"
        + (f", {grand_total.cache_read_tokens:,} cache reads" if grand_total.cache_read_tokens else "")
        + (f", {grand_total.cache_creation_tokens:,} cache writes" if grand_total.cache_creation_tokens else "")
        + ")"
    )

    logger.log_run_end({
        "per_seed": [
            {"seed": r.seed, "status": r.final_status, "cost": asdict(r.cost)}
            for r in all_results
        ],
        "grand_total": asdict(grand_total),
    })
    print(f"Run log written to: {log_path}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "model": args.model,
                    "grand_total_cost": asdict(grand_total),
                    "results": [result_to_dict(r) for r in all_results],
                },
                f,
                indent=2,
            )
        print(f"Final results written to: {args.output}")


if __name__ == "__main__":
    main()