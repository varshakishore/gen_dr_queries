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
    "claude-opus-4-8":   {"input": 5.00,  "output": 25.00},
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
<<<<<<< HEAD
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
=======
The answering system is an open "deep research" model that is trained to produce attributed long-form answers and whose ONLY tool is Semantic Scholar ("S2") search over academic papers. The system retrieves from a corpus of academic papers and synthesizes a cited report. Difficulty must not come from requiring sources outside this corpus. The system is good at surveying a single well-studied topic and producing a long well-structured report. It is bad at complex reasoning. 
>>>>>>> f5d2f76 (updated prompt, added script for parallelization, added caching)

OUTPUT FORMAT (valid JSON, no extra text):
{
  "brainstorming": "<think about distinct strategies and reason about why they may or may not work>",
  "chosen_strategy": "<name and justify the single most promising strategy>",
  "updated_question": "<the rewritten question>",
  "why_harder": "<explanation of why this question might be hard for a deep research system>",
<<<<<<< HEAD
  "verification_criteria": ["<atomic criterion 1>", "<atomic criterion 2 (optional)>", "<atomic criterion 3 (optional)>"]
}"""
=======
  "verification_criterion": "<one concrete, testable criterion for checking whether the answer is good>"
}

RULES:
- Avoid questions that can easily be answered by retrieving information.
- The updated question should be hard to answer correctly, not just hard to retrieve — via higher-order thinking (analysis, comparison, evaluation, synthesis), a reasoning trap the system must catch (false premise, misconception, unanswerable claim), or an embedded constraint that changes what a correct answer must contain.
- The updated question length should change by fewer than 15 words from the seed.
- The verification criterion should be specific and checkable, not vague or aspirational. The criterion is checked by a judge who sees ONLY the question, the answer, and the answer's own cited sources — there is NO external answer key. So don't use hollow existence-counts like "identify at least three implicit assumptions" or "name four categories of evidence." Anchor it to THIS question by naming the actual entities/claims at issue — never a generic template. 
- Select whichever strategy best fits THIS seed; do not default to the strategies shown in the examples below.
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

Here are a few examples:
Seed Question: What is pretraining-data deduplication?
{
"brainstorming": "The current question can easily be answered by retrieving deduplication literature broadly. To make it harder, let's make the question unanswerable by asking a question that isn't answered by current literature.",
"chosen_strategy": "Make the question unanswerable by asking a question that hasn't been resolved by current research.",
"updated_question": "What is the causal contribution of pretraining-data deduplication to downstream reasoning, holding all else constant?",
"why_harder": "A system can easily define deduplication, but it is much harder to determine its isolated causal effect on reasoning because existing studies do not cleanly vary only deduplication while holding other training factors fixed.",
"verification_criterion": "Because no controlled study isolates this effect, the answer must EXPLICITLY state the question is unresolved by current research, name the specific missing evidence, and qualify any partial findings as correlational not causal. Fails if it asserts a confident causal answer or implies the literature resolves it."
}

Seed Question: What is the role of attention sparsity in efficient transformers?
{
"brainstorming": "Surveying efficient-transformer literature would let the system define sparsity and list methods, so a pure synthesis question is too easy. One option is multi-step reasoning about FLOPs tradeoffs, but those numbers can be retrieved and quoted directly. A stronger option exploits that benchmark results for sparse attention genuinely conflict across papers: the system can locate both 'sparse wins' and 'sparse loses' results, but is bad at the reasoning needed to reconcile them via confounds.",
"chosen_strategy": "Require reconciliation of conflicting evidence: force the system to explain WHY cited papers disagree rather than just report their results.",
"updated_question": "When do sparse-attention transformers underperform dense baselines, and why do reported results conflict?",
"why_harder": "A survey can enumerate sparse-attention methods and their headline numbers, but reconciling contradictory sparse-vs-dense comparisons requires identifying confounds (sequence length, task type, matched compute) that the papers themselves rarely make explicit, which is a reasoning task rather than a retrieval task.",
"verification_criterion": "The answer must show that its OWN cited sources disagree (some reporting sparse >= dense, others sparse < dense) and attribute the conflict to at least one concrete confound such as sequence length, task type (long-range vs short-context), or matched compute budget. Fails if it issues a single uniform verdict, or if the papers it cites do not actually report conflicting sparse-vs-dense comparisons."
}

Seed Question: How does brown adipose tissue produce heat?
{
"brainstorming": "The seed is a clean survey: retrieve BAT/UCP1 literature and summarize thermogenesis. A single-topic false premise (e.g. mislocating a function) fails, because if the corpus already frames it as a known misconception the system just retrieves the debunking. So embed a false CONJUNCTION whose refutation is not packaged anywhere: assert that (a) UCP1 drives ATP synthesis and (b) this powers shivering thermogenesis. Each underlying fact (UCP1 uncouples to make heat not ATP; BAT mediates NON-shivering thermogenesis) is documented separately as background, but no source refutes this composite because no one proposes it. A survey-strong, reasoning-weak system retrieves the facts yet writes fluently around the premise without noticing the contradiction.",
"chosen_strategy": "False premise via conjunction of separately-documented facts.",
"updated_question": "How does UCP1-driven ATP synthesis power shivering thermogenesis?",
"why_harder": "Surveying BAT thermogenesis returns UCP1=uncoupling=heat and BAT=non-shivering as separate background facts, but nothing in the corpus is framed as refuting 'ATP-powered shivering.' A non-reasoning synthesis can therefore produce a fluent answer that silently honors the premise. Rejecting it requires conjoining two facts the literature never assembles against this claim: that UCP1 bypasses ATP synthase, and that it is the non-shivering pathway.",
"verification_criterion": "The answer must reject BOTH embedded errors: (1) state that UCP1 uncouples oxidative phosphorylation and dissipates the proton gradient as heat rather than synthesizing ATP, i.e. it bypasses/short-circuits ATP synthase; and (2) state that UCP1/BAT mediates NON-shivering thermogenesis, which is distinct from and an alternative to shivering thermogenesis (skeletal-muscle contraction). Fails if it describes UCP1 as producing ATP, treats BAT/UCP1 as the mechanism of shivering, or answers fluently as though the premise were coherent."
}
"""
>>>>>>> f5d2f76 (updated prompt, added script for parallelization, added caching)

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

PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE = """You are an expert in constructing challenging research questions.

Given a seed question, produce an updated question designed to expose weaknesses in deep research systems. 

ANSWERING SYSTEM PROFILE:
The answering system is an open "deep research" model that is trained to produce attributed long-form answers and whose ONLY tool is Semantic Scholar ("S2") search over academic papers. The system retrieves from a corpus of academic papers and synthesizes a cited report. Difficulty must not come from requiring sources outside this corpus. The system is good at surveying a single well-studied topic and producing a long well-structured report. It is bad at complex reasoning. 

OUTPUT FORMAT (valid JSON, no extra text):
{
  "brainstorming": "<think about distinct strategies and reason about why they may or may not work>",
  "chosen_strategy": "<name and justify the single most promising strategy>",
  "updated_question": "<the rewritten question>",
  "why_harder": "<explanation of why this question might be hard for a deep research system>",
  "verification_criterion": "<one concrete, testable criterion for checking whether the answer is good>"
}

RULES:
- Avoid questions that can easily be answered by retrieving information.
- The updated question should be hard to answer correctly, not just hard to retrieve — via higher-order thinking (analysis, comparison, evaluation, synthesis), a reasoning trap the system must catch (false premise, misconception, unanswerable claim), or an embedded constraint that changes what a correct answer must contain.
- The updated question length should change by fewer than 15 words from the seed.
- The verification criterion should be specific and checkable, not vague or aspirational. The criterion is checked by a judge who sees ONLY the question, the answer, and the answer's own cited sources — there is NO external answer key. So don't use hollow existence-counts like "identify at least three implicit assumptions" or "name four categories of evidence." Anchor it to THIS question by naming the actual entities/claims at issue — never a generic template. 
- Select whichever strategy best fits THIS seed; do not default to the strategies shown in the examples below.
- DO NOT USE THE SAME STRATEGIES AS THE EXAMPLES BELOW.Be creative and come up with your own strategy. 

Here are a few examples:
Seed Question: What is pretraining-data deduplication?
{
"brainstorming": "The current question can easily be answered by retrieving deduplication literature broadly. To make it harder, let's make the question unanswerable by asking a question that isn't answered by current literature.",
"chosen_strategy": "Make the question unanswerable by asking a question that hasn't been resolved by current research.",
"updated_question": "What is the causal contribution of pretraining-data deduplication to downstream reasoning, holding all else constant?",
"why_harder": "A system can easily define deduplication, but it is much harder to determine its isolated causal effect on reasoning because existing studies do not cleanly vary only deduplication while holding other training factors fixed.",
"verification_criterion": "Because no controlled study isolates this effect, the answer must EXPLICITLY state the question is unresolved by current research, name the specific missing evidence, and qualify any partial findings as correlational not causal. Fails if it asserts a confident causal answer or implies the literature resolves it."
}

Seed Question: What is the role of attention sparsity in efficient transformers?
{
"brainstorming": "Surveying efficient-transformer literature would let the system define sparsity and list methods, so a pure synthesis question is too easy. One option is multi-step reasoning about FLOPs tradeoffs, but those numbers can be retrieved and quoted directly. A stronger option exploits that benchmark results for sparse attention genuinely conflict across papers: the system can locate both 'sparse wins' and 'sparse loses' results, but is bad at the reasoning needed to reconcile them via confounds.",
"chosen_strategy": "Require reconciliation of conflicting evidence: force the system to explain WHY cited papers disagree rather than just report their results.",
"updated_question": "When do sparse-attention transformers underperform dense baselines, and why do reported results conflict?",
"why_harder": "A survey can enumerate sparse-attention methods and their headline numbers, but reconciling contradictory sparse-vs-dense comparisons requires identifying confounds (sequence length, task type, matched compute) that the papers themselves rarely make explicit, which is a reasoning task rather than a retrieval task.",
"verification_criterion": "The answer must show that its OWN cited sources disagree (some reporting sparse >= dense, others sparse < dense) and attribute the conflict to at least one concrete confound such as sequence length, task type (long-range vs short-context), or matched compute budget. Fails if it issues a single uniform verdict, or if the papers it cites do not actually report conflicting sparse-vs-dense comparisons."
}

Seed Question: How does brown adipose tissue produce heat?
{
"brainstorming": "The seed is a clean survey: retrieve BAT/UCP1 literature and summarize thermogenesis. A single-topic false premise (e.g. mislocating a function) fails, because if the corpus already frames it as a known misconception the system just retrieves the debunking. So embed a false CONJUNCTION whose refutation is not packaged anywhere: assert that (a) UCP1 drives ATP synthesis and (b) this powers shivering thermogenesis. Each underlying fact (UCP1 uncouples to make heat not ATP; BAT mediates NON-shivering thermogenesis) is documented separately as background, but no source refutes this composite because no one proposes it. A survey-strong, reasoning-weak system retrieves the facts yet writes fluently around the premise without noticing the contradiction.",
"chosen_strategy": "False premise via conjunction of separately-documented facts.",
"updated_question": "How does UCP1-driven ATP synthesis power shivering thermogenesis?",
"why_harder": "Surveying BAT thermogenesis returns UCP1=uncoupling=heat and BAT=non-shivering as separate background facts, but nothing in the corpus is framed as refuting 'ATP-powered shivering.' A non-reasoning synthesis can therefore produce a fluent answer that silently honors the premise. Rejecting it requires conjoining two facts the literature never assembles against this claim: that UCP1 bypasses ATP synthase, and that it is the non-shivering pathway.",
"verification_criterion": "The answer must reject BOTH embedded errors: (1) state that UCP1 uncouples oxidative phosphorylation and dissipates the proton gradient as heat rather than synthesizing ATP, i.e. it bypasses/short-circuits ATP synthase; and (2) state that UCP1/BAT mediates NON-shivering thermogenesis, which is distinct from and an alternative to shivering thermogenesis (skeletal-muscle contraction). Fails if it describes UCP1 as producing ATP, treats BAT/UCP1 as the mechanism of shivering, or answers fluently as though the premise were coherent."
}

Reminder: Do not use the same strategies as the examples above.
"""

PROMPT_FOR_SEED_CRITERION = """You are an expert evaluator of deep research systems.

Given a research question, produce a simple, ATOMIC verification criterion. The criterion must test
exactly ONE concrete property of a good answer — not a combination.

QUESTION:
{question}

OUTPUT FORMAT (valid JSON, no extra text):
{{
  "verification_criterion": "<one concrete, testable criterion for checking whether the answer is good>"
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

VERIFICATION CRITERION:
{criterion}

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
    verification_criterion: str  # single atomic criterion
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
    harder: HarderQuestion
    answer: str
    judgment: Judgment
    trace: object = None


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
    log_messages: Optional[list] = None,
) -> tuple[str, CostBucket]:
    """Single Claude call wrapped with logging + cost accounting.

    `log_messages`, if given, is logged instead of the real `messages` — used to
    redact the bulky research answer from the judge prompt in the log.
    """
    logged_messages = log_messages if log_messages is not None else messages
    t0 = time.perf_counter()
    err: Optional[str] = None
    response_text = ""
    usage: dict = {}
    cost_usd = 0.0

    try:
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
            system=system, messages=logged_messages, response_text=response_text,
            usage=usage, cost_usd=cost_usd, latency_s=latency, error=err,
        )
        raise

    latency = time.perf_counter() - t0
    logger.log_claude_call(
        seed=seed, attempt=attempt, purpose=purpose, model=model,
        system=system, messages=logged_messages, response_text=response_text,
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
) -> tuple[str, CostBucket]:
    """Generate one atomic verification criterion for the seed question (round 0)."""
    prompt = PROMPT_FOR_SEED_CRITERION.format(question=seed)
    messages = [{"role": "user", "content": prompt}]
    raw, bucket = _call_claude(
        client, model=model, system=None, messages=messages,
        max_tokens=800, logger=logger, seed=seed, attempt=0, purpose="seed_criterion",
    )
    data = extract_json(raw)
    return str(data.get("verification_criterion") or ""), bucket


def harder_question_gen(
    client: Anthropic,
    model: str,
    seed: str,
    prior_attempts: list,
    logger: RunLogger,
    attempt: int,
    harder_prompt: str = PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE,
) -> tuple[HarderQuestion, CostBucket]:
    user_content = f"Seed question: {seed}"
    if prior_attempts:
        feedback_blocks = []
        for rec in prior_attempts:
            feedback_blocks.append(
                f"--- Previous attempt {rec.attempt} ---\n"
                f"Question tried: {rec.harder.updated_question}\n"
                f"Strategy: {rec.harder.chosen_strategy}\n"
                f"Verification criterion: {rec.harder.verification_criterion}\n"
                f"Judge verdict: {rec.judgment.verdict}\n"
                f"Judge summary: {rec.judgment.summary}\n"
                f"Other issues flagged: {rec.judgment.other_issues}\n"
            )
        user_content += (
            "\n\nThe research system PASSED the previous harder versions of this seed, "
            "which means those questions were not hard enough. MAKE THE QUESTION HARDER "
            "this time. Do not just rephrase a prior attempt."
            "You can use a different strategy than before, and you can also use the feedback on prior attempts."
            "Below is what was tried:\n\n"
            + "\n".join(feedback_blocks)
        )

    messages = [{"role": "user", "content": user_content}]
    raw, bucket = _call_claude(
        client, model=model, system=harder_prompt, messages=messages,
        max_tokens=2000, logger=logger, seed=seed, attempt=attempt, purpose="harder",
    )
    data = extract_json(raw)
    return (
        HarderQuestion(
            brainstorming=data.get("brainstorming", ""),
            chosen_strategy=data.get("chosen_strategy", ""),
            updated_question=data["updated_question"],
            why_harder=data.get("why_harder", ""),
            verification_criterion=data.get("verification_criterion", ""),
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
    # Log the prompt with the (bulky) answer redacted; it lives in the results file.
    log_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, criterion=criterion,
        answer=f"<answer omitted: {len(answer)} chars — see results file>",
    )
    log_messages = [{"role": "user", "content": log_prompt}]
    raw, bucket = _call_claude(
        client, model=model, system=None, messages=messages,
        max_tokens=2000, logger=logger, seed=seed, attempt=attempt, purpose="judge",
        log_messages=log_messages,
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
# Research server call
# ---------------------------------------------------------------------------

def _redact_research_body(body):
    """Strip the bulky `answer`/`trace` from a response body before logging.

    The full answer and trace live only in the results file; the log keeps just
    lightweight metadata (sizes/presence) plus any other fields verbatim.
    """
    if not isinstance(body, dict):
        return body
    redacted = {k: v for k, v in body.items() if k not in ("answer", "trace")}
    if "answer" in body:
        redacted["answer_chars"] = len(body.get("answer") or "")
    if "trace" in body:
        redacted["trace_present"] = body.get("trace") is not None
    return redacted


def query_research_system(
    question: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
    url: str = RESEARCH_SERVER_URL,
    timeout_s: float = RESEARCH_TIMEOUT_S,
) -> tuple[str, object]:
    """POST the question to the research server and return (answer, trace).

    The server is expected to respond with JSON of the form
    {"answer": "...", "trace": ...}. The full body is logged verbatim; `trace`
    is None if the server did not include it.
    """
    request_json = {"question": question}
    t0 = time.perf_counter()
    err: Optional[str] = None
    status: Optional[int] = None
    body = None
    answer = ""
    trace = None

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
        trace = body.get("trace")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        logger.log_research_call(
            seed=seed, attempt=attempt, url=url, request_json=request_json,
            response_status=status, response_body=_redact_research_body(body),
            latency_s=latency, error=err,
        )
        raise

    latency = time.perf_counter() - t0
    logger.log_research_call(
        seed=seed, attempt=attempt, url=url, request_json=request_json,
        response_status=status, response_body=_redact_research_body(body),
        latency_s=latency, error=None,
    )
    return answer, trace


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
    harder_prompt: str = PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE,
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
                criterion, bucket = generate_seed_criterion(client, model, seed, logger)
                result.cost.add(bucket)
                harder = HarderQuestion(
                    brainstorming="",
                    chosen_strategy="seed (no modification)",
                    updated_question=seed,
                    why_harder="",
                    verification_criterion=criterion,
                )
            else:
                prior_harder = [a for a in result.attempts if a.attempt > 0]
                harder, bucket = harder_question_gen(
                    client, model, seed, prior_harder, logger, attempt,
                    harder_prompt=harder_prompt,
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
            print(f"    Criterion: {harder.verification_criterion}")

        # Step 2 — query research system
        try:
            answer, trace = query_research_system(
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
                harder.verification_criterion, answer, logger, seed, attempt,
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
                attempt=attempt, harder=harder, answer=answer, judgment=judgment,
                trace=trace,
            )
        )

        if judgment.verdict == "FAILED":
            if attempt == 0:
                result.final_status = "ALREADY_HARD"
                if verbose:
                    print(
                        "\n>>> Seed question is already difficult — the research system "
                        "failed it without any modification. Stopping."
                    )
            else:
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
                "harder": asdict(a.harder),
                "answer": a.answer,
                "trace": a.trace,
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
    parser.add_argument(
        "--prompt", choices=["explore", "original"], default="explore",
        help="Which make-harder system prompt to use (default: explore). "
             "'explore' forbids reusing the example strategies; 'original' keeps the "
             "in-context strategy menu.",
    )
    args = parser.parse_args()

    harder_prompt = (PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE if args.prompt == "explore"
                     else PROMPT_TO_MAKE_HARDER_QUESTION)

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
            server_url=args.server_url, harder_prompt=harder_prompt,
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