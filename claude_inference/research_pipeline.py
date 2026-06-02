#!/usr/bin/env python3
"""
Pipeline for generating hard research questions, evaluating answers, and iterating.

Flow per seed question:
  0. Test the seed question as-is against the research server. If the answer
     already FAILS the judge, stop and report the seed is already difficult.
  1. Use Claude to create a harder version of the seed question (with prior attempts as context after round 1).
  2. Send the harder question to a local research server at localhost:8007/ask.
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
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESEARCH_SERVER_URL = "http://localhost:8007/ask"
RESEARCH_TIMEOUT_S = 600  # generous: deep-research calls can be slow

CLAUDE_MODEL = "claude-sonnet-4-5"
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


PROMPT_TO_MAKE_HARDER_QUESTION = """You are an expert in constructing challenging research questions.

Given a seed question, produce an updated question designed to expose weaknesses in deep research systems.

ANSWERING SYSTEM PROFILE:
The system retrieves from a corpus of academic papers and synthesizes a cited report. Difficulty must not 
come from requiring sources outside this corpus.

RULES:
- Avoid "what" questions that merely retrieve information.
- The updated question MUST require higher-order thinking: analysis, comparison, evaluation, or synthesis.
- It must NOT be answerable with a definition, summary, or single-source response.
- The updated question length should change by fewer than 10 words from the seed.
- The question must be NATURAL and something a researcher might actually ask. It should ONLY have one main 
  component (no "and" or multiple sub-questions). It is better to keep it simple.

EXAMPLE STRATEGIES TO CONSIDER:
1. Require synthesis across 5+ sources or clearly disjoint domains (e.g., political science + economics).
2. Require synthesis across differing viewpoints, stakeholder incentives, or theoretical frameworks.
3. Require multi-step reasoning, structured argumentation, or hierarchical planning.
4. Require handling conflicting, incomplete, or low-quality evidence.
5. Require universal quantification ("for all X, is Y true?") or reasoning about edge cases and exceptions.
6. Require correcting a hidden misconception or establishing key knowns before answering.
7. Embed a specific context that changes the answer (e.g., "explain to a policymaker with no ML background").
8. Make a question that is unanswerable by current research, no existing work is available.
9. Something else you think of that would be effective at exposing weaknesses in research systems!

VERIFICATION CRITERIA:
Along with the updated question, also produce up to 3 simple, ATOMIC verification criteria. Each criterion
must test exactly ONE concrete property of a good answer — not a combination. Bad (compound): "is well-cited
and synthesizes multiple viewpoints". Good (atomic): "cites at least 3 distinct sources"; separately,
"presents at least two opposing viewpoints". Use fewer criteria if the question is narrow.

OUTPUT FORMAT (valid JSON, no extra text):
{
  "brainstorming": "<think about distinct strategies and reason about why they may or may not work>",
  "chosen_strategy": "<name and justify the single most promising strategy>",
  "updated_question": "<the rewritten question>",
  "why_harder": "<explanation of why this question might be hard for a deep research system>",
  "verification_criteria": ["<atomic criterion 1>", "<atomic criterion 2 (optional)>", "<atomic criterion 3 (optional)>"]
}"""

PROMPT_FOR_SEED_CRITERION = """You are an expert evaluator of deep research systems.

Given a research question, produce up to 3 simple, ATOMIC verification criteria. Each criterion must test
exactly ONE concrete property of a good answer — not a combination. Bad (compound): "is well-cited and
synthesizes multiple viewpoints". Good (atomic): "cites at least 3 distinct sources"; separately, "presents
at least two opposing viewpoints". Use fewer criteria if the question is narrow.

ANSWERING SYSTEM PROFILE:
The system retrieves from a corpus of academic papers and synthesizes a cited report.

QUESTION:
{question}

OUTPUT FORMAT (valid JSON, no extra text):
{{
  "verification_criteria": ["<atomic criterion 1>", "<atomic criterion 2 (optional)>", "<atomic criterion 3 (optional)>"]
}}
"""

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of deep research system outputs.

You will be given:
- A research question
- A list of verification criteria, each one defining a single property a good answer must have
- The answer the research system produced

Judge EACH criterion independently — for each, decide whether the answer satisfies it and explain briefly.
Also flag any other serious issues you notice (factual errors, hallucinations, evasion, refusals, off-topic).

The overall verdict is FAILED if ANY criterion is not satisfied, OR if there are serious other issues.
The overall verdict is PASSED only if every criterion is satisfied and there are no serious issues.
A higher number of failed criteria indicates a more difficult question for the research system.

QUESTION:
{question}

VERIFICATION CRITERIA:
{criteria}

ANSWER:
{answer}

OUTPUT FORMAT (valid JSON, no extra text):
{{
  "criterion_results": [
    {{"criterion": "<criterion 1 text>", "satisfied": <true | false>, "reasoning": "<brief>"}},
    {{"criterion": "<criterion 2 text>", "satisfied": <true | false>, "reasoning": "<brief>"}}
  ],
  "criteria_failed_count": <integer count of criteria with satisfied=false>,
  "other_issues": ["<issue 1>", "<issue 2>", ...],
  "summary": "<2-4 sentence overall summary of how the answer performed and what to push on next time>",
  "verdict": "<PASSED | FAILED>"
}}
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
        purpose: str,           # "make harder" or "judge"
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
class HarderQuestion:
    brainstorming: str
    chosen_strategy: str
    updated_question: str
    why_harder: str
    verification_criteria: list  # up to 3 atomic criteria
    raw: str = ""


@dataclass
class Judgment:
    criterion_results: list  # list of {criterion, satisfied, reasoning}
    criteria_failed_count: int
    other_issues: list
    summary: str
    verdict: str
    raw: str = ""


@dataclass
class AttemptRecord:
    attempt: int
    harder: HarderQuestion
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


def generate_seed_criterion(
    client: Anthropic,
    model: str,
    seed: str,
    logger: RunLogger,
) -> tuple[list, CostBucket]:
    """Generate up to 3 atomic verification criteria for the seed question (round 0)."""
    prompt = PROMPT_FOR_SEED_CRITERION.format(question=seed)
    messages = [{"role": "user", "content": prompt}]
    raw, bucket = _call_claude(
        client, model=model, system=None, messages=messages,
        max_tokens=800, logger=logger, seed=seed, attempt=0, purpose="seed_criterion",
    )
    data = extract_json(raw)
    criteria = data.get("verification_criteria") or []
    if not isinstance(criteria, list):
        criteria = [str(criteria)]
    return [c for c in criteria if c], bucket


def harder_question_gen(
    client: Anthropic,
    model: str,
    seed: str,
    prior_attempts: list,
    logger: RunLogger,
    attempt: int,
) -> tuple[HarderQuestion, CostBucket]:
    user_content = f"Seed question: {seed}"
    if prior_attempts:
        feedback_blocks = []
        for rec in prior_attempts:
            feedback_blocks.append(
                f"--- Previous attempt {rec.attempt} ---\n"
                f"Question tried: {rec.harder.updated_question}\n"
                f"Strategy: {rec.harder.chosen_strategy}\n"
                f"Verification criteria: {rec.harder.verification_criteria}\n"
                f"Judge verdict: {rec.judgment.verdict}\n"
                f"Criteria failed: {rec.judgment.criteria_failed_count}\n"
                f"Judge summary: {rec.judgment.summary}\n"
                f"Other issues flagged: {rec.judgment.other_issues}\n"
            )
        user_content += (
            "\n\nThe research system PASSED the previous harder versions of this seed, "
            "which means those questions were not hard enough. MAKE THE QUESTION HARDER "
            "this time. Pick a meaningfully different strategy — do not just rephrase a "
            "prior attempt — and target a weakness the previous attempts did not exploit. "
            "Below is what was tried:\n\n"
            + "\n".join(feedback_blocks)
        )

    messages = [{"role": "user", "content": user_content}]
    raw, bucket = _call_claude(
        client, model=model, system=PROMPT_TO_MAKE_HARDER_QUESTION, messages=messages,
        max_tokens=2000, logger=logger, seed=seed, attempt=attempt, purpose="harder",
    )
    data = extract_json(raw)
    criteria = data.get("verification_criteria") or []
    if not isinstance(criteria, list):
        criteria = [str(criteria)]
    criteria = [c for c in criteria if c]
    return (
        HarderQuestion(
            brainstorming=data.get("brainstorming", ""),
            chosen_strategy=data.get("chosen_strategy", ""),
            updated_question=data["updated_question"],
            why_harder=data.get("why_harder", ""),
            verification_criteria=criteria,
            raw=raw,
        ),
        bucket,
    )


def judge_answer(
    client: Anthropic,
    model: str,
    question: str,
    criteria: list,
    answer: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
) -> tuple[Judgment, CostBucket]:
    criteria_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria)) or "(none)"
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, criteria=criteria_block, answer=answer
    )
    messages = [{"role": "user", "content": prompt}]
    raw, bucket = _call_claude(
        client, model=model, system=None, messages=messages,
        max_tokens=2000, logger=logger, seed=seed, attempt=attempt, purpose="judge",
    )
    data = extract_json(raw)

    criterion_results = data.get("criterion_results") or []
    if not isinstance(criterion_results, list):
        criterion_results = []
    failed_count = data.get("criteria_failed_count")
    if not isinstance(failed_count, int):
        failed_count = sum(1 for r in criterion_results if not r.get("satisfied", True))

    verdict = str(data.get("verdict", "")).upper()
    if verdict not in ("PASSED", "FAILED"):
        verdict = "FAILED" if failed_count > 0 else "PASSED"

    return (
        Judgment(
            criterion_results=criterion_results,
            criteria_failed_count=int(failed_count),
            other_issues=data.get("other_issues", []),
            summary=data.get("summary", ""),
            verdict=verdict,
            raw=raw,
        ),
        bucket,
    )


# ---------------------------------------------------------------------------
# Research server call
# ---------------------------------------------------------------------------

def query_research_system(
    question: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
    url: str = RESEARCH_SERVER_URL,
    timeout_s: float = RESEARCH_TIMEOUT_S,
) -> str:
    """POST the question to the research server and return the answer string.

    The server is expected to respond with JSON of the form {"answer": "..."}
    (and may include additional fields, which are logged verbatim).
    """
    request_json = {"question": question}
    t0 = time.perf_counter()
    err: Optional[str] = None
    status: Optional[int] = None
    body = None
    answer = ""

    try:
        resp = requests.post(url, json=request_json, timeout=timeout_s)
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        resp.raise_for_status()
        if not isinstance(body, dict) or "answer" not in body:
            raise ValueError(
                f"Research server response missing 'answer' field; got: {body!r}"
            )
        answer = body["answer"]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        logger.log_research_call(
            seed=seed, attempt=attempt, url=url, request_json=request_json,
            response_status=status, response_body=body, latency_s=latency, error=err,
        )
        raise

    latency = time.perf_counter() - t0
    logger.log_research_call(
        seed=seed, attempt=attempt, url=url, request_json=request_json,
        response_status=status, response_body=body, latency_s=latency, error=None,
    )
    return answer


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
    server_url: str = RESEARCH_SERVER_URL,
) -> SeedResult:
    result = SeedResult(seed=seed)

    # Round 0 tests the seed as-is; rounds 1..N test harder rewrites.
    for attempt in range(0, max_attempts + 1):
        if verbose:
            print(f"\n{'=' * 70}")
            label = "Round 0 (testing seed as-is)" if attempt == 0 else f"Attempt {attempt}/{max_attempts}"
            print(f"SEED: {seed!r}  |  {label}")
            print("=" * 70)

        # Step 1 — produce the question + criterion to test.
        # Round 0: use the seed and generate a criterion for it.
        # Round 1+: ask Claude to rewrite the seed into a harder question,
        #           using only the prior harder attempts (not round 0) as feedback.
        try:
            if attempt == 0:
                criteria, bucket = generate_seed_criterion(client, model, seed, logger)
                result.cost.add(bucket)
                harder = HarderQuestion(
                    brainstorming="",
                    chosen_strategy="seed (no modification)",
                    updated_question=seed,
                    why_harder="",
                    verification_criteria=criteria,
                )
            else:
                prior_harder = [a for a in result.attempts if a.attempt > 0]
                harder, bucket = harder_question_gen(
                    client, model, seed, prior_harder, logger, attempt
                )
                result.cost.add(bucket)
        except Exception as e:
            result.final_status = "ERROR"
            result.error = (
                f"Generating seed criterion failed: {e}" if attempt == 0
                else f"Making harder question failed: {e}"
            )
            if verbose:
                print(f"[ERROR] {result.error}")
            return result

        if verbose:
            print(f"[1] Question: {harder.updated_question}")
            if attempt > 0:
                print(f"    Strategy: {harder.chosen_strategy}")
            print(f"    Criteria ({len(harder.verification_criteria)}):")
            for i, c in enumerate(harder.verification_criteria, 1):
                print(f"      {i}. {c}")

        # Step 2 — query research system
        try:
            answer = query_research_system(
                harder.updated_question, logger, seed, attempt, url=server_url,
            )
        except Exception as e:
            result.final_status = "ERROR"
            result.error = f"Research server call failed: {e}"
            if verbose:
                print(f"[ERROR] Research server call failed: {e}")
            return result

        if verbose:
            preview = answer.replace("\n", " ")
            preview = preview[:240] + ("…" if len(preview) > 240 else "")
            print(f"[2] Answer (preview): {preview}")

        # Step 3 — judge
        try:
            judgment, bucket = judge_answer(
                client, model, harder.updated_question,
                harder.verification_criteria, answer, logger, seed, attempt,
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
            total_criteria = len(harder.verification_criteria)
            print(
                f"[3] Verdict: {judgment.verdict} "
                f"({judgment.criteria_failed_count}/{total_criteria} criteria failed)"
            )
            print(f"    Summary: {judgment.summary}")
            if judgment.other_issues:
                print(f"    Other issues: {judgment.other_issues}")
            print(
                f"    [cost so far on this seed: ${result.cost.cost_usd:.4f} "
                f"across {result.cost.calls} Claude calls]"
            )

        result.attempts.append(
            AttemptRecord(
                attempt=attempt, harder=harder, answer=answer, judgment=judgment,
            )
        )

        if judgment.verdict == "FAILED":
            total = len(harder.verification_criteria)
            failed = judgment.criteria_failed_count
            if attempt == 0:
                result.final_status = "ALREADY_HARD"
                if verbose:
                    print(
                        f"\n>>> Seed question is already difficult — the research system "
                        f"failed {failed}/{total} criteria without any modification. Stopping."
                    )
            else:
                result.final_status = "FAILED_FOUND"
                if verbose:
                    print(
                        f"\n>>> FAILED answer found on attempt {attempt} "
                        f"({failed}/{total} criteria failed). Stopping."
                    )
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
                "harder": asdict(a.harder),
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
    parser = argparse.ArgumentParser(
        description="Make seed questions harder, query a research server, and judge answers."
    )
    parser.add_argument("seeds", nargs="*",
                        help="Seed questions. If omitted, reads from --seeds-file or stdin.")
    parser.add_argument("--seeds-file", help="Path to a file with one seed per line.")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                        help=f"Max attempts per seed (default: {MAX_ATTEMPTS}).")
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
        "--server-url", default=RESEARCH_SERVER_URL,
        help=f"Research server endpoint (default: {RESEARCH_SERVER_URL}).",
    )
    args = parser.parse_args()

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
            server_url=args.server_url,
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
            print(f"    failing question: {last.harder.updated_question}")
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