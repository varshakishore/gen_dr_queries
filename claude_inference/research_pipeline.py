#!/usr/bin/env python3
"""
Pipeline for generating hard research questions, evaluating answers, and iterating.

Flow per seed question:
  0. Test the seed question as-is against the research server. If the answer
     already FAILS the judge, stop and report the seed is already difficult.
  1. Use Claude to create a harder version of the seed question (with prior attempts as context after round 1).
  1b. (--verify-criterion) Retrieve S2 papers for the harder question and ask Claude
     whether the verification criterion is itself factually correct. Continue on
     "correct"/"partly_correct", swapping in the meta-judge's rewrite when it supplies
     one; stop the seed on "incorrect"/"insufficient_evidence" before paying for a
     research call.
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

from cite_utils import build_doc_index, numbered_plaintext, references_block

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
    "claude-sonnet-5":  {"input": 3.00,  "output": 15.00},
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


# ---------------------------------------------------------------------------
# Answering-system profiles
# ---------------------------------------------------------------------------
# Describe the system whose weaknesses we're probing. Selected via --profile and
# injected into the make-harder prompts in place of the "{ANSWERING_SYSTEM_PROFILE}"
# sentinel. Add new entries here as you target new answering systems.
ANSWERING_SYSTEM_PROFILES = {
    "drtulu": (
        'The answering system is an open "deep research" model that is trained to produce '
        'attributed long-form answers and whose ONLY tool is Semantic Scholar ("S2") search '
        "over academic papers. The system retrieves from a corpus of academic papers and "
        "synthesizes a cited report. Difficulty must not come from requiring sources outside "
        "this corpus. The system is good at surveying a single well-studied topic and "
        "producing a long well-structured report. It is bad at complex reasoning."
    ),
    "tongyi": (
        'The answering system is an open "deep research" model that is trained to produce '
        'attributed long-form answers and whose ONLY tool is search '
        "over academic papers. The system retrieves from a corpus of papers and "
        "synthesizes a report. Difficulty must not come from requiring sources outside "
        "this corpus. The system is good at surveying a single well-studied topic and "
        "producing a long well-structured report. It is bad at complex reasoning."
    ),
}
DEFAULT_PROFILE = "drtulu"


# ---------------------------------------------------------------------------
# Example-strategy menus
# ---------------------------------------------------------------------------
# Each entry is a list of strategy descriptions shown to Claude under
# "EXAMPLE STRATEGIES TO CONSIDER" in PROMPT_TO_MAKE_HARDER_QUESTION. Selected via
# --strategies and numbered automatically at render time. Add new menus here to
# steer generation toward a particular class of difficulty.
STRATEGY_LISTS = {
    # The original mixed menu: a bit of everything.
    "default": [
        "Require synthesis across 5+ sources or clearly disjoint domains (e.g., political science + economics).",
        "Require synthesis across differing viewpoints, stakeholder incentives, or theoretical frameworks.",
        "Require multi-step reasoning, structured argumentation, or hierarchical planning.",
        "Require handling conflicting, incomplete, or low-quality evidence.",
        'Require universal quantification ("for all X, is Y true?") or reasoning about edge cases and exceptions. '
        "However, keep the scope reasonably bounded so that an answer could adequately address it, and not so "
        "broad that any answer would necessarily be incomplete.",
        "Require correcting a hidden misconception or establishing key knowns before answering.",
        'Embed a specific context that changes the answer (e.g., "explain to a policymaker with no ML background").',
        "Make a question that is unanswerable by current research, no existing work is available.",
        "Something else you think of that would be effective at exposing weaknesses in research systems!",
    ],
    "jena_cog_biases":  [
    # 1. Survivorship & selection
    "Frame the question around a filtered sample of evidence as though it represented the full population, requiring the answer to identify missing cases and explain how selection affects the inference.",
    # 2. Base-rate & magnitude neglect
    "Ignore or obscure a relevant base rate, denominator, sample size, effect size, or population magnitude, requiring the answer to restore the omitted quantity and explain its implications.",
    # 3. Spurious pattern & causation
    "Treat a correlation, temporal coincidence, or apparent cluster in noisy evidence as an established relationship, requiring the answer to consider randomness, confounding, reverse causation, or coincidence.",
    # 4. Source & consensus
    "Treat apparent agreement, popularity, or endorsement by a dominant authority as independent corroboration, requiring the answer to determine whether the sources genuinely converge or share data, methods, citations, or institutional origins.",
    # 5. Measurement & proxy
    "Treat a measurable proxy, such as citations, test scores, or statistical significance, as identical to the underlying construct, requiring the answer to examine whether the metric validly captures what matters.",
    # 6. Confirmation & motivated testing
    "Presuppose a favored conclusion and request only confirming evidence or a one-sided test, requiring the answer to consider alternative hypotheses and disconfirming evidence.",
    # 7. Anchoring, framing & substitution
    "Introduce an anchor, framing manipulation, salient detail, or substituted proxy question that distorts the target judgment, requiring the answer to identify the distortion before addressing the underlying issue.",
    # 8. Information & action
    "Presume that gathering more information, taking action, or eliminating a small residual risk is inherently worthwhile, requiring the answer to determine whether it could materially change the decision or outcome.",
    # 9. Paradigm resistance & belief updating
    "Frame a prevailing paradigm as settled and invite the answer to discount contradictory evidence, requiring it instead to weigh the new evidence on its merits and update the prior appropriately.",
    # 10. Hindsight & outcome
    "Present the outcome of a past prediction, decision, or study in a way that invites hindsight or outcome bias, requiring the answer to assess the reasoning using only the evidence available at the time.",
    # 11. Temporal distortion
    "Presume that a phenomenon is recent, increasingly common, newly important, or declining because of recent attention, requiring the answer to evaluate it against the appropriate historical record.",
    # 12. Overconfidence & illusion of understanding
    "Assume that a phenomenon or mechanism is well understood despite limited or primarily descriptive evidence, requiring the answer to calibrate confidence to what the literature actually establishes.",
    # 13. Illusory truth & availability
    "Assert that a claim is well established or self-evident because it is repeated, familiar, salient, or easy to retrieve, requiring the answer to separate familiarity and availability from evidential support.",
    ],
    "merged_v1": [
        "Require synthesis across 5+ sources or clearly disjoint domains (e.g., political science + economics).",
        "Require synthesis across differing viewpoints, stakeholder incentives, or theoretical frameworks.",
        "Require multi-step reasoning, structured argumentation, or hierarchical planning.",
        "Require handling conflicting, incomplete, or low-quality evidence.",
        "Require universal quantification ('for all X, is Y true?') or reasoning about edge cases and exceptions. "
        "However, keep the scope reasonably bounded so that an answer could adequately address it, and not so broad that any answer would necessarily be incomplete.",
        "Require correcting a hidden misconception or establishing key knowns before answering.",
        "Embed a specific context that changes the answer (e.g., 'explain to a policymaker with no ML background').",
        "Make a question that is unanswerable by current research, no existing work is available.",
        "Require careful quantitative reasoning about base rates, denominators, magnitudes, probabilities, effect sizes, or levels of aggregation rather than merely quoting headline figures.",
        "Frame the available evidence as though what was selected, measured, or recorded perfectly represented the underlying population or construct. A correct answer must identify the relevant selection effect, missing cases, or measurement limitation.", 
        "Treat a correlation, temporal coincidence, or apparent empirical pattern as though it established a causal relationship. A correct answer must consider confounding, reverse causation, selection, or chance and reject unsupported causal framing.", 
        "Frame the question around a misleading assumption, false dichotomy, anchor, or substituted question. A correct answer must identify and repair the framing rather than simply answer within it.", 
        "Use a concept whose meaning, boundary, or classification is ambiguous, contested, field-dependent, or genuinely continuous. A correct answer must clarify the relevant definition or explain why no unique cutoff exists.", 
        "Make the apparent evidential support misleading because of source provenance or evidence quality, such as false attribution, shared source ancestry, retraction, failed replication, publication bias, or unequal methodological strength. A correct answer must evaluate the evidence rather than count citations or repeat the claim.", 
        "Require reconciliation of apparently conflicting findings by examining differences in methods, populations, settings, time periods, or experimental conditions rather than simply choosing the majority result.", 
        "Ask about a past event, decision, prediction, or apparent trend in a way that invites hindsight, outcome bias, or distortion from recent attention. A correct answer must use the information and historical baseline appropriate to the time.", 
        "Present a familiar, dominant, or seemingly well-understood explanation as settled despite meaningful uncertainty or contradictory evidence. A correct answer must update beliefs according to evidence quality and calibrate confidence to what is actually established.", 
        "Presume that obtaining more information, increasing precision, taking action, or eliminating residual uncertainty is inherently valuable. A correct answer must assess whether it could materially change the relevant decision or outcome.", 
        "Something else you think of that would be effective at exposing weaknesses in research systems!",
    ]
}
DEFAULT_STRATEGIES = "default"


# ---------------------------------------------------------------------------
# Banned-strategy menus
# ---------------------------------------------------------------------------
# Banning forces the search wider.
BANNED_STRATEGY_LISTS = {
    # The three strategies the prompt's own few-shot examples demonstrate. The prompt
    # already says not to reuse them; naming them makes that enforceable.
    "default": [
        "Make a question that is unanswerable by current research, no existing work is available.",
        "Require reconciliation of conflicting evidence: force the system to explain WHY "
        "retrieved papers disagree rather than just report their results.",
        "Require recognition of an underlying false premise in the question.",
    ],
    # Ban exactly the menu the 'original' prompt is shown, so an explore run is
    # guaranteed to go off-menu (drops the open-ended "something else" line).
    "seed_menu": [s for s in STRATEGY_LISTS["default"] if not s.startswith("Something else")],
    # Ban the widest menu we have: 'default' plus the cognitive-bias traps.
    "merged_v1": [s for s in STRATEGY_LISTS["merged_v1"] if not s.startswith("Something else")],
    # Nothing off-limits: the "STRATEGIES TO NOT USE" block is removed entirely.
    "none": [],
}
DEFAULT_BANNED_STRATEGIES = "default"

_PROFILE_SENTINEL = "{ANSWERING_SYSTEM_PROFILE}"
_STRATEGIES_SENTINEL = "{EXAMPLE_STRATEGIES}"
_BANNED_STRATEGIES_SENTINEL = "{BANNED_STRATEGIES}"


def format_strategies(strategies: list) -> str:
    """Render a strategy list as the numbered menu block used in the prompt."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(strategies, start=1))


def with_profile(template: str, profile_text: str) -> str:
    """Inject the answering-system profile into a make-harder prompt template.

    Uses str.replace (not str.format) so the literal JSON braces in the template are
    left untouched.
    """
    return template.replace(_PROFILE_SENTINEL, profile_text)


def with_strategies(template: str, strategies: list) -> str:
    """Inject an example-strategy menu into a make-harder prompt template.

    Templates without the sentinel (e.g. the explore prompt, which forbids reusing
    the example strategies) are returned unchanged.
    """
    return template.replace(_STRATEGIES_SENTINEL, format_strategies(strategies))


def with_banned_strategies(template: str, strategies: list) -> str:
    """Inject a banned-strategy menu into a make-harder prompt template.

    An empty list removes the whole "STRATEGIES TO NOT USE" block rather than leaving a
    dangling header. Templates without the sentinel (e.g. the original prompt, which
    shows a menu to use instead of one to avoid) are returned unchanged.
    """
    block = f"STRATEGIES TO NOT USE:\n{_BANNED_STRATEGIES_SENTINEL}\n\n"
    if not strategies:
        return template.replace(block, "")
    return template.replace(_BANNED_STRATEGIES_SENTINEL, format_strategies(strategies))


PROMPT_TO_MAKE_HARDER_QUESTION = """You are an expert in constructing challenging research questions.

Given a seed question, produce an updated question designed to expose weaknesses in deep research systems. 

ANSWERING SYSTEM PROFILE:
{ANSWERING_SYSTEM_PROFILE}

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
- The verification criterion should be specific and checkable, not vague or aspirational. The criterion is checked by a judge who sees ONLY the question, and the answer — there is NO external answer key. So don't use hollow existence-counts like "identify at least three implicit assumptions" or "name four categories of evidence." Anchor it to THIS question by naming the actual entities/claims at issue — never a generic template. 
- Select whichever strategy best fits THIS seed; do not default to the strategies shown in the examples below.
- The question must be NATURAL and something a researcher might actually ask. It should ONLY have one main component (no "and" or multiple sub-questions). It is better to keep it simple.
- The question should be in English.

EXAMPLE STRATEGIES TO CONSIDER:
{EXAMPLE_STRATEGIES}

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
"chosen_strategy": "Require reconciliation of conflicting evidence: force the system to explain WHY retrieved papers disagree rather than just report their results.",
"updated_question": "When do sparse-attention transformers underperform dense baselines, and why do reported results conflict?",
"why_harder": "A survey can enumerate sparse-attention methods and their headline numbers, but reconciling contradictory sparse-vs-dense comparisons requires identifying confounds (sequence length, task type, matched compute) that the papers themselves rarely make explicit, which is a reasoning task rather than a retrieval task.",
"verification_criterion": "The answer must show that its retrieved sources disagree (some reporting sparse >= dense, others sparse < dense) and attribute the conflict to at least one concrete confound such as sequence length, task type (long-range vs short-context), or matched compute budget. Fails if it issues a single uniform verdict, or if the papers it retrieves do not actually report conflicting sparse-vs-dense comparisons."
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

PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE = """You are an expert in constructing challenging research questions.

Given a seed question, produce an updated question designed to expose weaknesses in deep research systems. 

ANSWERING SYSTEM PROFILE:
{ANSWERING_SYSTEM_PROFILE}

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
- The verification criterion should be specific and checkable, not vague or aspirational. The criterion is checked by a judge who sees ONLY the question, the answer, and the answer's own sources — there is NO external answer key. So don't use hollow existence-counts like "identify at least three implicit assumptions" or "name four categories of evidence." Anchor it to THIS question by naming the actual entities/claims at issue — never a generic template. 
- Think creatively and come up with a strategy that will result in a hard question for THIS seed.
- DO NOT USE THE STRATEGIES in the list below.
- DO NOT USE THE SAME STRATEGIES AS THE EXAMPLES BELOW. Be creative and come up with your own strategy. 

STRATEGIES TO NOT USE:
{BANNED_STRATEGIES}

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
"chosen_strategy": "Require reconciliation of conflicting evidence: force the system to explain WHY retrieved papers disagree rather than just report their results.",
"updated_question": "When do sparse-attention transformers underperform dense baselines, and why do reported results conflict?",
"why_harder": "A survey can enumerate sparse-attention methods and their headline numbers, but reconciling contradictory sparse-vs-dense comparisons requires identifying confounds (sequence length, task type, matched compute) that the papers themselves rarely make explicit, which is a reasoning task rather than a retrieval task.",
"verification_criterion": "The answer must show that its retrieved sources disagree (some reporting sparse >= dense, others sparse < dense) and attribute the conflict to at least one concrete confound such as sequence length, task type (long-range vs short-context), or matched compute budget. Fails if it issues a single uniform verdict, or if the papers it retrieves do not actually report conflicting sparse-vs-dense comparisons."
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

Given a research question, produce a simple, ATOMIC verification criterion. The criterion must test exactly ONE concrete property of a good answer — not a combination. If the question is underspecified, then the verification_criterion should say "Any non-empty answer is acceptable."

A question is underspecified when:
  - It states no ask at all (a bare entity, ID, name, or URL with no verb).
  - The ask has multiple non-overlapping readings and nothing selects one
    (e.g. "Tesla 2024" — sales? stock? litigation? model releases?).
  - The success condition depends on unstated context the evaluator does not
    have (e.g. "is this dosage safe for my patient?").

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
- The verification criterion that defines what a good answer must do
- The answer the research system produced, with inline [n] citation markers and a References section listing each cited paper snippet

Your job is to judge whether the answer satisfies the verification criterion, and to flag other issues you notice (factual errors, hallucinations, evasion, missing reasoning, structural problems, etc.) even if those issues are not part of the criterion. If the verification criterion is "Any non-empty answer is acceptable", then the verdict should be PASSED.

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

PROMPT_TO_VERIFY_VERIFICATION_CRITERIA = """You are an expert meta-evaluator for a deep-research benchmark with difficult questions. Your only task is to judge whether the VERIFICATION CRITERION itself is correct as an evaluation standard for the given question.

Do NOT reward or penalize style, atomicity, verbosity, or formatting except where those affect whether the criterion states a correct requirement. Determine whether the criterion's factual expectations, premises, causal claims, comparisons, mechanisms, entities, time frames, required distinctions, and absence/uncertainty claims are true, evidence-supported, and fairly required by the harder question.

BENCHMARK CONTEXT:
- The answering system being tested uses Semantic Scholar / academic-paper search.
- A verification criterion is supposed to define one property that a correct answer to the question should satisfy.
- A criterion is correct only if its required content is true.

EVIDENCE RULES:
- Use the provided local Semantic Scholar search results.
- If the criterion embeds a specific expected fact, verify that fact directly.
- If the criterion requires an answer to reject a false premise, verify that the premise is actually false or misleading.
- If the criterion requires uncertainty, no causal isolation, no consensus, or absence of evidence, verify that this is a fair characterization of the available evidence rather than an unsupported negative claim.
- If the combined evidence is not adequate to verify the criterion's factual expectation, use insufficient_evidence.
- In checked_claims, include ONLY factual claims made by or required by the verification criterion itself. Alternative hypotheses, possible counterexamples, and anything else surfaced by the search belong in reasoning, not in checked_claims.

REQUESTING ADDITIONAL SEARCHES:
The provided search results are normally what you must work with. Use additional_queries SPARINGLY — only when a claim genuinely cannot be settled from the context given. Leave the list empty in every other case.
- Always return your best provisional correctness_label from the evidence you already have.
- If a search would merely add corroborating detail, do not request it.

LABEL DEFINITIONS (for judging the verification criterion itself):
- correct: The criterion's factual expectations are supported, accurately framed, and fairly required by the question. It can be used as-is to judge an answer.
- partly_correct: The core expectation is directionally right, but some wording, scope, certainty, causal framing, entity mapping, or required distinction is materially imprecise. It should be revised before use.
- incorrect: A factual expectation, premise, mechanism, comparison, required conclusion, or absence/uncertainty claim in the criterion is contradicted, unsupported, or unfairly required by the harder question.
- insufficient_evidence: The given evidence is not adequate to verify whether the criterion itself is correct.

Question:
{question}

Generator's rationale for why the question is difficult:
{why_harder}

Verification criterion being evaluated:
{criterion}

Local Semantic Scholar search results:
{search_results_context}

OUTPUT FORMAT — return valid JSON only, with this exact shape:
{{
  "checked_claims": [
    {{
      "claim": "<factual claim made by or required by the verification criterion itself>",
      "verdict": "supported | contradicted | not_found | not_checkable",
      "evidence": "<evidence for whether this criterion claim is true>",
      "sources": ["local S2 title or URL...", "https://..."]
    }}
  ],
  "correctness_label": "correct | partly_correct | incorrect | insufficient_evidence",
  "main_correctness_problem": "<one sentence naming the single most serious defect in the criterion, e.g. the specific false expectation, mis-scoped requirement, or overstated absence claim. Empty string if and only if correctness_label is 'correct'. For insufficient_evidence, name the specific criterion claim that could not be verified.>",
  "reasoning": "<the decision process behind correctness_label: connect the criterion's own factual requirements to the combined evidence, weigh any counterexamples or alternative hypotheses your search surfaced, and explain why the selected label is justified.>",
  "rewrite": "<one corrected verification criterion for the given question, if the criterion is not correct but can be fixed. Empty string if correctness_label is 'correct' or if the evidence is insufficient to write a corrected version.>",
  "additional_queries": [
    {{
      "query": "<a specific search whose results would settle a claim you could not settle from the provided context>",
      "targets_claim": "<which checked_claims entry this would resolve>",
    }}
  ]
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

    def log_criterion_check(
        self,
        *,
        seed: str,
        attempt: int,
        question: str,
        criterion: str,
        retrieval: dict,
        check: Optional[dict],
        latency_s: float,
        error: Optional[str] = None,
    ) -> None:
        """Log the meta-judge's verdict on the verification criterion itself.

        `check` is the parsed output JSON of PROMPT_TO_VERIFY_VERIFICATION_CRITERIA,
        stored verbatim. `retrieval` holds the S2 retrieval metadata (counts, queries,
        filters) — the retrieved paper text itself is not logged, only its size.
        """
        self._write({
            "kind": "criterion_check",
            "seed": seed,
            "attempt": attempt,
            "question": question,
            "criterion": criterion,
            "retrieval": retrieval,
            "check": check,
            "latency_s": latency_s,
            "error": error,
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
    verification_criterion: str  # single atomic criterion; may be replaced by a meta-judge rewrite
    raw: str = ""
    # Set only when the criterion check supplied a rewrite: the criterion as originally
    # generated, before verification_criterion was replaced by it.
    verification_criterion_original: str = ""


@dataclass
class CriterionCheck:
    """Meta-judge verdict on the verification criterion itself (round 1+ only)."""
    correctness_label: str
    main_correctness_problem: str
    reasoning: str
    rewrite: str
    checked_claims: list = field(default_factory=list)
    additional_queries: list = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)
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
    # None when the attempt stopped before the research server was queried, i.e. the
    # criterion check rejected the criterion.
    judgment: Optional[Judgment] = None
    trace: object = None
    answer_model: object = None   # model the research server self-reported
    criterion_check: Optional[CriterionCheck] = None


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


def format_answer_for_judge(answer: str, trace: object) -> str:
    """Render an answer the way the HTML viewer does, in plain text.

    DR-Tulu's opaque `<cite id="...">` tags become inline [n] markers backed by a
    References section carrying each source's title, authors, and retrieved snippet,
    so the judge can check a claim against the text it cites rather than against the
    system's own paraphrase of it. Answers with no `<cite>` tags (Tongyi, or any
    response without a trace) pass through unchanged.
    """
    doc_index = build_doc_index(trace if isinstance(trace, dict) else {})
    marked, refs = numbered_plaintext(answer, doc_index)
    return marked + references_block(refs, include_snippets=True)


def judge_answer(
    client: Anthropic,
    model: str,
    question: str,
    criterion: str,
    answer: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
    trace: object = None,
) -> tuple[Judgment, CostBucket]:
    judge_answer_text = format_answer_for_judge(answer, trace)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, criterion=criterion, answer=judge_answer_text
    )
    messages = [{"role": "user", "content": prompt}]
    # Log the prompt with the (bulky) answer redacted; it lives in the results file.
    log_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, criterion=criterion,
        answer=f"<answer + references omitted: {len(judge_answer_text)} chars — "
               f"see results file>",
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
# Criterion verification (retrieval + meta-judge)
# ---------------------------------------------------------------------------

# Defaults for the S2 retrieval that grounds the criterion check. n_rerank is left at
# retrieve_papers' own default.
VERIFY_N_PAPERS = 15
VERIFY_MAX_CHARS_PER_PAPER = 4000


def format_search_results_context(
    papers: list,
    max_papers: int = VERIFY_N_PAPERS,
    max_chars_per_paper: int = VERIFY_MAX_CHARS_PER_PAPER,
) -> str:
    """Render retrieve_papers() output as the {search_results_context} block.

    Uses `relevance_judgment_input_expanded` — the same per-paper markdown blob
    ScholarQA feeds to its quote-extraction step — capped per paper so a few
    long full-text papers cannot crowd out the rest.
    """
    if not papers:
        return "(no papers were retrieved for this question)"
    blocks = []
    for i, paper in enumerate(papers[:max_papers], start=1):
        body = paper.get("relevance_judgment_input_expanded") or ""
        if max_chars_per_paper and len(body) > max_chars_per_paper:
            body = body[:max_chars_per_paper] + "\n...[truncated]"
        blocks.append(
            f"--- Paper {i} {paper.get('reference_string', '')} ---\n{body}"
        )
    return "\n\n".join(blocks)


def verify_criterion(
    client: Anthropic,
    model: str,
    question: str,
    why_harder: str,
    criterion: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
    retrieval_kwargs: Optional[dict] = None,
    n_context_papers: int = VERIFY_N_PAPERS,
    max_chars_per_paper: int = VERIFY_MAX_CHARS_PER_PAPER,
) -> tuple[CriterionCheck, CostBucket]:
    """Retrieve papers for `question`, then judge whether `criterion` is itself correct.

    retrieve_papers() runs its own Claude call for query decomposition using its own
    client. That call is not logged as a claude_call, but its usage is reported back and
    folded into the returned CostBucket (and into the criterion_check log record).
    """
    # Imported lazily so the pipeline still runs without the retrieval stack installed.
    from retrieve_papers import retrieve_papers

    t0 = time.perf_counter()
    retrieval_meta: dict = {}
    decompose_bucket = CostBucket()
    try:
        retrieved = retrieve_papers(question, **(retrieval_kwargs or {}))
        papers = retrieved.get("papers") or []
        context = format_search_results_context(
            papers, max_papers=n_context_papers, max_chars_per_paper=max_chars_per_paper
        )
        retrieval_meta = {
            "n_papers": len(papers),
            "n_context_papers": min(len(papers), n_context_papers),
            "context_chars": len(context),
            "rewritten_query": retrieved.get("rewritten_query"),
            "keyword_query": retrieved.get("keyword_query"),
            "search_filters": retrieved.get("search_filters"),
            "n_snippets": retrieved.get("n_snippets"),
            "n_keyword_papers": retrieved.get("n_keyword_papers"),
            "elapsed_s": retrieved.get("elapsed_s"),
        }
        # retrieve_papers' query-decomposition call bills to us; price it here since it
        # never passes through _call_claude.
        decompose_usage = retrieved.get("decompose_usage")
        if decompose_usage:
            decompose_model = retrieved.get("decomposer_model") or model
            cost, usage_n = price_call(decompose_model, decompose_usage)
            decompose_bucket = CostBucket(
                input_tokens=usage_n["input_tokens"],
                output_tokens=usage_n["output_tokens"],
                cache_creation_tokens=usage_n["cache_creation_input_tokens"],
                cache_read_tokens=usage_n["cache_read_input_tokens"],
                cost_usd=cost,
                calls=1,
            )
            retrieval_meta["decomposer_model"] = decompose_model
            retrieval_meta["decompose_usage"] = usage_n
            retrieval_meta["decompose_cost_usd"] = cost
    except Exception as e:
        logger.log_criterion_check(
            seed=seed, attempt=attempt, question=question, criterion=criterion,
            retrieval=retrieval_meta, check=None,
            latency_s=time.perf_counter() - t0, error=f"retrieval failed: {type(e).__name__}: {e}",
        )
        raise

    prompt = PROMPT_TO_VERIFY_VERIFICATION_CRITERIA.format(
        question=question, why_harder=why_harder, criterion=criterion,
        search_results_context=context,
    )
    messages = [{"role": "user", "content": prompt}]
    # Keep the bulky retrieved context out of the claude_call log record; the
    # criterion_check record carries its size and the queries that produced it.
    log_messages = [{"role": "user", "content": PROMPT_TO_VERIFY_VERIFICATION_CRITERIA.format(
        question=question, why_harder=why_harder, criterion=criterion,
        search_results_context=(
            f"<{retrieval_meta['n_context_papers']} papers omitted: "
            f"{retrieval_meta['context_chars']} chars>"
        ),
    )}]

    try:
        raw, bucket = _call_claude(
            client, model=model, system=None, messages=messages,
            max_tokens=3000, logger=logger, seed=seed, attempt=attempt,
            purpose="verify_criterion", log_messages=log_messages,
        )
        bucket.add(decompose_bucket)
        data = extract_json(raw)
    except Exception as e:
        logger.log_criterion_check(
            seed=seed, attempt=attempt, question=question, criterion=criterion,
            retrieval=retrieval_meta, check=None,
            latency_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}",
        )
        raise

    logger.log_criterion_check(
        seed=seed, attempt=attempt, question=question, criterion=criterion,
        retrieval=retrieval_meta, check=data,
        latency_s=time.perf_counter() - t0, error=None,
    )

    return (
        CriterionCheck(
            correctness_label=str(data.get("correctness_label") or "").strip().lower(),
            main_correctness_problem=data.get("main_correctness_problem", ""),
            reasoning=data.get("reasoning", ""),
            rewrite=(data.get("rewrite") or "").strip(),
            checked_claims=data.get("checked_claims", []),
            additional_queries=data.get("additional_queries", []),
            retrieval=retrieval_meta,
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
) -> dict:
    """POST the question to the research server and return
    {"answer", "trace", "model", "usage"}.

    The server is expected to respond with JSON containing at least "answer"; "trace",
    "model", and "usage" are passed through if present (else None). The full body is
    logged (redacted).
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
    # Pass through answer + any of trace/model/usage the server included (None if absent).
    return {
        "answer": answer,
        "trace": body.get("trace"),
        "model": body.get("model"),
        "usage": body.get("usage"),
    }


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
    timeout_s: float = RESEARCH_TIMEOUT_S,
    verify_criteria: bool = False,
    retrieval_kwargs: Optional[dict] = None,
    n_context_papers: int = VERIFY_N_PAPERS,
    max_chars_per_paper: int = VERIFY_MAX_CHARS_PER_PAPER,
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
                prior_harder = [
                    a for a in result.attempts if a.attempt > 0 and a.judgment is not None
                ]
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

        # Step 1b — check the criterion itself before spending a research-server call on
        # it. Rounds 1+ only: round 0's criterion comes from generate_seed_criterion, has
        # no why_harder to supply, and is often the "any non-empty answer" escape hatch.
        criterion_check = None
        if verify_criteria and attempt > 0:
            try:
                criterion_check, bucket = verify_criterion(
                    client, model, harder.updated_question, harder.why_harder,
                    harder.verification_criterion, logger, seed, attempt,
                    retrieval_kwargs=retrieval_kwargs,
                    n_context_papers=n_context_papers,
                    max_chars_per_paper=max_chars_per_paper,
                )
                result.cost.add(bucket)
            except Exception as e:
                result.final_status = "ERROR"
                result.error = f"Criterion verification failed: {e}"
                if verbose:
                    print(f"[ERROR] {result.error}")
                return result

            if verbose:
                print(
                    f"[1b] Criterion check: {criterion_check.correctness_label} "
                    f"({criterion_check.retrieval.get('n_context_papers', 0)} papers)"
                )
                if criterion_check.main_correctness_problem:
                    print(f"     Problem: {criterion_check.main_correctness_problem}")
                if criterion_check.additional_queries:
                    print(f"     Requested searches: {criterion_check.additional_queries}")

            if criterion_check.correctness_label not in ("correct", "partly_correct"):
                result.final_status = "CRITERION_INVALID"
                result.attempts.append(
                    AttemptRecord(
                        attempt=attempt, harder=harder, answer="",
                        criterion_check=criterion_check,
                    )
                )
                if verbose:
                    print(
                        f"\n>>> Criterion judged {criterion_check.correctness_label} on "
                        f"attempt {attempt}; not worth a research call. Stopping."
                    )
                return result

            if criterion_check.rewrite:
                harder.verification_criterion_original = harder.verification_criterion
                harder.verification_criterion = criterion_check.rewrite
                if verbose:
                    print(f"     Using rewritten criterion: {harder.verification_criterion}")

        # Step 2 — query research system
        try:
            research = query_research_system(
                harder.updated_question, logger, seed, attempt, url=server_url,
                timeout_s=timeout_s,
            )
            answer, trace, answer_model = (
                research["answer"], research["trace"], research["model"])
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
                trace=trace,
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
                trace=trace, answer_model=answer_model,
                criterion_check=criterion_check,
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
                "answer_model": a.answer_model,
                "trace": a.trace,
                "judgment": asdict(a.judgment) if a.judgment is not None else None,
                "criterion_check": (
                    asdict(a.criterion_check) if a.criterion_check is not None else None
                ),
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
        "--timeout", type=float, default=RESEARCH_TIMEOUT_S,
        help=f"Read timeout (s) for each research-server call (default: {RESEARCH_TIMEOUT_S}). "
             "Raise it for slow models, e.g. 7200 for Tongyi.",
    )
    parser.add_argument(
        "--prompt", choices=["explore", "original"], default="explore",
        help="Which make-harder system prompt to use (default: explore). "
             "'explore' forbids reusing the example strategies; 'original' keeps the "
             "in-context strategy menu.",
    )
    parser.add_argument(
        "--profile", choices=sorted(ANSWERING_SYSTEM_PROFILES), default=DEFAULT_PROFILE,
        help=f"Answering-system profile injected into the make-harder prompt "
             f"(default: {DEFAULT_PROFILE}).",
    )
    parser.add_argument(
        "--strategies", choices=sorted(STRATEGY_LISTS), default=DEFAULT_STRATEGIES,
        help=f"Which example-strategy menu to inject into the '--prompt original' "
             f"make-harder prompt (default: {DEFAULT_STRATEGIES}). Ignored by "
             f"'--prompt explore', which has no strategy menu.",
    )
    parser.add_argument(
        "--banned-strategies", choices=sorted(BANNED_STRATEGY_LISTS),
        default=DEFAULT_BANNED_STRATEGIES,
        help=f"Which banned-strategy menu to inject under 'STRATEGIES TO NOT USE' in the "
             f"'--prompt explore' make-harder prompt (default: {DEFAULT_BANNED_STRATEGIES}; "
             f"'none' drops the block). Ignored by '--prompt original'.",
    )
    parser.add_argument(
        "--verify-criterion", action="store_true",
        help="Before each research-server call (rounds 1+), retrieve papers for the harder "
             "question and ask Claude whether the verification criterion is itself correct. "
             "Continues on 'correct'/'partly_correct' (applying any rewrite) and stops the "
             "seed on 'incorrect'/'insufficient_evidence'. Requires retrieve_papers.py and "
             "S2_API_KEY.",
    )
    parser.add_argument(
        "--verify-n-papers", type=int, default=VERIFY_N_PAPERS,
        help=f"Papers to include in the criterion-check context (default: {VERIFY_N_PAPERS}).",
    )
    parser.add_argument(
        "--verify-max-chars-per-paper", type=int, default=VERIFY_MAX_CHARS_PER_PAPER,
        help=f"Truncate each paper's text to this many chars in the criterion-check "
             f"context (default: {VERIFY_MAX_CHARS_PER_PAPER}).",
    )
    parser.add_argument(
        "--reranker", default="auto", choices=["auto", "none", "vllm"],
        help="Reranker for criterion-check retrieval; 'auto' uses a remote vLLM server if "
             "one is configured, else no reranking.",
    )
    parser.add_argument(
        "--reranker-url", default=None,
        help="Base URL of the vLLM reranker, e.g. http://gpu-host:8000 (env: VLLM_RERANK_URL).",
    )
    args = parser.parse_args()

    profile_text = ANSWERING_SYSTEM_PROFILES[args.profile]

    retrieval_kwargs = {
        "reranker": args.reranker,
        "reranker_url": args.reranker_url,
    }

    base_template = (PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE if args.prompt == "explore"
                     else PROMPT_TO_MAKE_HARDER_QUESTION)
    harder_prompt = with_banned_strategies(
        with_strategies(
            with_profile(base_template, profile_text), STRATEGY_LISTS[args.strategies]
        ),
        BANNED_STRATEGY_LISTS[args.banned_strategies],
    )

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

    if args.verify_criterion and not os.environ.get("S2_API_KEY"):
        print(
            "[verify] WARNING: S2_API_KEY is not set; criterion-check retrieval will be "
            "rate limited hard by the Semantic Scholar API.",
            file=sys.stderr,
        )

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
            timeout_s=args.timeout,
            verify_criteria=args.verify_criterion,
            retrieval_kwargs=retrieval_kwargs,
            n_context_papers=args.verify_n_papers,
            max_chars_per_paper=args.verify_max_chars_per_paper,
        )
        all_results.append(result)
        grand_total.add(result.cost)

    print(f"\n{'#' * 70}")
    print("FINAL SUMMARY")
    print("#" * 70)
    for r in all_results:
        n = len(r.attempts)
        print(
            f"- [{r.final_status:17}] ({n} attempts, ${r.cost.cost_usd:.4f}, "
            f"{r.cost.calls} Claude calls) {r.seed}"
        )
        if r.error:
            print(f"    error: {r.error}")
        elif r.final_status == "FAILED_FOUND":
            last = r.attempts[-1]
            print(f"    failing question: {last.harder.updated_question}")
            print(f"    judge summary: {last.judgment.summary}")
        elif r.final_status == "CRITERION_INVALID":
            last = r.attempts[-1]
            print(f"    rejected question: {last.harder.updated_question}")
            print(f"    criterion: {last.harder.verification_criterion}")
            print(f"    label: {last.criterion_check.correctness_label}")
            print(f"    problem: {last.criterion_check.main_correctness_problem}")

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