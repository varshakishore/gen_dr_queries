#!/usr/bin/env python3
"""
Pipeline for generating hard research questions, evaluating answers, and iterating.

Flow per seed question:
  0. Test the seed question as-is against the research server. If the answer
     already FAILS the judge, stop and report the seed is already difficult.
  1. Use Claude to create a harder version of the seed question (with prior attempts as context after round 1).
  1b. (--verify-criterion) Retrieve S2 papers for the harder question and ask Claude
     whether the verification criterion is itself factually correct. Continue on
     "correct"/"almost_correct", swapping in the meta-judge's rewrite when it supplies
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

import llm_client
from cite_utils import (
    build_doc_index, has_inline_citations, numbered_plaintext, references_block,
    selected_refs,
)
from llm_client import price_call, resolve_provider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESEARCH_SERVER_URL = "http://localhost:8007/ask"
RESEARCH_TIMEOUT_S = 600  # generous: deep-research calls can be slow

# Default generator/judge model. Any id llm_client routes to OpenAI (gpt-*, o3-*)
# works here too -- see llm_client for how the provider is picked from the id.
# The default generator/judge model for every entry point. Its PROVIDER cascades: the
# query decomposer (retrieve_papers.DEFAULT_DECOMPOSER_MODELS) and the strategy clusterer
# (--cluster-provider auto) both follow whichever side --model lands on, so one default
# moves the whole run and only OPENAI_API_KEY is needed. The decomposer defaults are
# tier-matched (gpt-5.6-terra $2/$12 vs claude-sonnet-4-5 $3/$15), so this does not
# quietly drop a capability tier on the step that picks the papers.
DEFAULT_MODEL = "gpt-5.6-terra"
# Pre-rename alias: eval_other_model.py and test_providers.py import this name.
CLAUDE_MODEL = DEFAULT_MODEL
MAX_ATTEMPTS = 5

# MODEL_PRICING and the cache multipliers now live in llm_client, so both providers
# bill off one table -- add a new model's rates there. `price_call` stays re-exported
# from here for importers (compare_claude.py, research_loop.py) that took it from here.


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
            'long-form answers and whose ONLY tool is search. '
            'The target questions are academic research questions, so the '
            'system should rely primarily on academic sources such as research papers, '
            'scholarly articles, and other research-oriented search results. '
            "Difficulty must not come from requiring sources outside "
            "this set. The system retrieves information and then synthesizes a report. "
            "The system is good at surveying a single well-studied topic and "
            "producing a long well-structured report. It is bad at complex reasoning. The system cannot produce in-line citations."
    ),
    "webthinker": (
            'The answering system is an open "deep research" model that is trained to produce '
            'long-form answers and whose ONLY tool is search. '
            'The target questions are academic research questions, so the '
            'system should rely primarily on academic sources such as research papers, '
            'scholarly articles, and other research-oriented search results. '
            "Difficulty must not come from requiring sources outside "
            "this set. The system retrieves information and then synthesizes a report. "
            "The system is good at surveying a single well-studied topic and "
            "producing a long well-structured report. It is bad at complex reasoning. The system cannot produce in-line citations."
        ),
}
DEFAULT_PROFILE = "drtulu"


# ---------------------------------------------------------------------------
# Example-strategy menus
# ---------------------------------------------------------------------------
# Each entry is a list of strategy descriptions shown to Claude under
# "EXAMPLE STRATEGIES TO CONSIDER" in PROMPT_TO_MAKE_HARDER_QUESTION_EXPLOIT. Selected via
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
    ]
}
# merged_v1 (18) rather than default (9): the per-seed menu draws
# --max-example-strategies (6) without replacement, and that draw only does real work
# when the cap is comfortably below the pool -- at 6 of 9 the same strategies appear in
# most menus, at 6 of 18 they genuinely vary. See research_loop.py's menu sampling.
DEFAULT_STRATEGIES = "merged_v1"


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
    # Ban exactly the menu the 'exploit' prompt is shown, so an explore run is
    # guaranteed to go off-menu (drops the open-ended "something else" line).
    "seed_menu": [s for s in STRATEGY_LISTS["default"] if not s.startswith("Something else")],
    # Ban the widest menu we have: 'default' plus the cognitive-bias traps.
    "merged_v1": [s for s in STRATEGY_LISTS["merged_v1"] if not s.startswith("Something else")],
    # Nothing off-limits: the "STRATEGIES TO NOT USE" block is removed entirely.
    "none": [],
}
DEFAULT_BANNED_STRATEGIES = "default"

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------
# The worked examples shown under "Here are a few examples:" in BOTH make-harder prompts
# (they are identical in the two templates). Round 0 of research_loop.py uses these; later
# rounds swap in demonstrations drawn from the run itself via --few-shots-file.
# `brainstorming` and `why_harder` are optional -- a shot missing them renders without
# those keys -- but keep them where possible: they are what teaches the reasoning, not
# just the output shape.
FEW_SHOT_KEYS = ["brainstorming", "chosen_strategy", "updated_question", "why_harder",
                 "verification_criterion"]

DEFAULT_FEW_SHOTS = [
    {
        "seed_question": "What is pretraining-data deduplication?",
        "brainstorming": "The current question can easily be answered by retrieving deduplication literature broadly. To make it harder, let's make the question unanswerable by asking a question that isn't answered by current literature.",
        "chosen_strategy": "Make the question unanswerable by asking a question that hasn't been resolved by current research.",
        "updated_question": "What is the causal contribution of pretraining-data deduplication to downstream reasoning, holding all else constant?",
        "why_harder": "A system can easily define deduplication, but it is much harder to determine its isolated causal effect on reasoning because existing studies do not cleanly vary only deduplication while holding other training factors fixed.",
        "verification_criterion": "Because no controlled study isolates this effect, the answer must EXPLICITLY state the question is unresolved by current research, name the specific missing evidence, and qualify any partial findings as correlational not causal. Fails if it asserts a confident causal answer or implies the literature resolves it."
    },
    {
        "seed_question": "What is the role of attention sparsity in efficient transformers?",
        "brainstorming": "Surveying efficient-transformer literature would let the system define sparsity and list methods, so a pure synthesis question is too easy. One option is multi-step reasoning about FLOPs tradeoffs, but those numbers can be retrieved and quoted directly. A stronger option exploits that benchmark results for sparse attention genuinely conflict across papers: the system can locate both 'sparse wins' and 'sparse loses' results, but is bad at the reasoning needed to reconcile them via confounds.",
        "chosen_strategy": "Require reconciliation of conflicting evidence: force the system to explain WHY retrieved papers disagree rather than just report their results.",
        "updated_question": "When do sparse-attention transformers underperform dense baselines, and why do reported results conflict?",
        "why_harder": "A survey can enumerate sparse-attention methods and their headline numbers, but reconciling contradictory sparse-vs-dense comparisons requires identifying confounds (sequence length, task type, matched compute) that the papers themselves rarely make explicit, which is a reasoning task rather than a retrieval task.",
        "verification_criterion": "The answer must show that its retrieved sources disagree (some reporting sparse >= dense, others sparse < dense) and link the conflict to at least one concrete confound such as sequence length, task type (long-range vs short-context), or matched compute budget. Fails if it issues a single uniform verdict, or if the papers it retrieves do not actually report conflicting sparse-vs-dense comparisons."
    },
    {
        "seed_question": "How does brown adipose tissue produce heat?",
        "brainstorming": "The seed is a clean survey: retrieve BAT/UCP1 literature and summarize thermogenesis. A single-topic false premise (e.g. mislocating a function) fails, because if the corpus already frames it as a known misconception the system just retrieves the debunking. So embed a false CONJUNCTION whose refutation is not packaged anywhere: assert that (a) UCP1 drives ATP synthesis and (b) this powers shivering thermogenesis. Each underlying fact (UCP1 uncouples to make heat not ATP; BAT mediates NON-shivering thermogenesis) is documented separately as background, but no source refutes this composite because no one proposes it. A survey-strong, reasoning-weak system retrieves the facts yet writes fluently around the premise without noticing the contradiction.",
        "chosen_strategy": "False premise via conjunction of separately-documented facts.",
        "updated_question": "How does UCP1-driven ATP synthesis power shivering thermogenesis?",
        "why_harder": "Surveying BAT thermogenesis returns UCP1=uncoupling=heat and BAT=non-shivering as separate background facts, but nothing in the corpus is framed as refuting 'ATP-powered shivering.' A non-reasoning synthesis can therefore produce a fluent answer that silently honors the premise. Rejecting it requires conjoining two facts the literature never assembles against this claim: that UCP1 bypasses ATP synthase, and that it is the non-shivering pathway.",
        "verification_criterion": "The answer must reject BOTH embedded errors: (1) state that UCP1 uncouples oxidative phosphorylation and dissipates the proton gradient as heat rather than synthesizing ATP, i.e. it bypasses/short-circuits ATP synthase; and (2) state that UCP1/BAT mediates NON-shivering thermogenesis, which is distinct from and an alternative to shivering thermogenesis (skeletal-muscle contraction). Fails if it describes UCP1 as producing ATP, treats BAT/UCP1 as the mechanism of shivering, or answers fluently as though the premise were coherent."
    }
]


_PROFILE_SENTINEL = "{ANSWERING_SYSTEM_PROFILE}"
_STRATEGIES_SENTINEL = "{EXAMPLE_STRATEGIES}"
_BANNED_STRATEGIES_SENTINEL = "{BANNED_STRATEGIES}"
_FEW_SHOTS_SENTINEL = "{FEW_SHOT_EXAMPLES}"


def format_few_shots(shots: list) -> str:
    """Render few-shot examples as the `Seed Question: ...` + JSON blocks the prompts use."""
    out = []
    for s in shots:
        body = {k: s[k] for k in FEW_SHOT_KEYS if s.get(k)}
        out.append(f"Seed Question: {s.get('seed_question', '').strip()}\n"
                   + json.dumps(body, indent=0, ensure_ascii=False))
    return "\n\n".join(out)


def with_few_shots(template: str, shots: list) -> str:
    """Inject the worked examples into a make-harder prompt template."""
    return template.replace(_FEW_SHOTS_SENTINEL, format_few_shots(shots))


def load_few_shots_file(path: Path | str) -> list:
    """Read few-shot examples from a JSON file: a list of dicts keyed like DEFAULT_FEW_SHOTS."""
    shots = json.loads(Path(path).read_text())
    if not isinstance(shots, list) or not all(isinstance(s, dict) for s in shots):
        raise ValueError(f"{path}: expected a JSON list of example objects")
    missing = [i for i, s in enumerate(shots)
               if not s.get("seed_question") or not s.get("updated_question")]
    if missing:
        raise ValueError(f"{path}: example(s) {missing} lack seed_question/updated_question")
    return shots


def load_strategy_file(path: Path | str) -> list:
    """Read a strategy menu from a file: one strategy per line, `#` comments ignored.

    Lets a caller (e.g. research_loop.py) supply a menu computed at runtime instead of
    picking a hard-coded one from STRATEGY_LISTS / BANNED_STRATEGY_LISTS.
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


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
    dangling header. Templates without the sentinel (e.g. the exploit prompt, which
    shows a menu to use instead of one to avoid) are returned unchanged.
    """
    block = f"STRATEGIES TO NOT USE:\n{_BANNED_STRATEGIES_SENTINEL}\n\n"
    if not strategies:
        return template.replace(block, "")
    return template.replace(_BANNED_STRATEGIES_SENTINEL, format_strategies(strategies))


PROMPT_TO_MAKE_HARDER_QUESTION_EXPLOIT = """You are an expert in constructing challenging research questions.

Given a seed question, produce an updated question designed to expose weaknesses in deep research systems. 

ANSWERING SYSTEM PROFILE:
{ANSWERING_SYSTEM_PROFILE}

OUTPUT FORMAT (valid JSON, no extra text):
{
  "brainstorming": "<in under 4 sentences, reason about a couple strategies and why they might or might not work for THIS seed>",
  "chosen_strategy": "<name and explain the single most promising strategy>",
  "updated_question": "<the rewritten question>",
  "why_harder": "<explanation of why this question might be hard for a deep research system>",
  "verification_criterion": "<one concrete, testable criterion for checking whether the answer is good>"
}

RULES:
- Avoid questions that can easily be answered by retrieving information.
- The updated question should be hard to answer correctly, not just hard to retrieve — via higher-order thinking (analysis, comparison, evaluation, synthesis), a reasoning trap the system must catch (false premise, misconception, unanswerable claim), or an embedded constraint that changes what a correct answer must contain.
- The updated question length should change by fewer than 15 words from the seed.
- The verification criterion should be specific and checkable, not vague or aspirational. The criterion is checked by a judge who sees ONLY the question, the answer, and the answer's own sources — there is NO external answer key. So don't use hollow existence-counts like "identify at least three implicit assumptions" or "name four categories of evidence." Anchor it to THIS question by naming the actual entities/claims at issue — never a generic template. 
- Select whichever strategy works best for THIS seed from the list below. You can use variations of the strategies listed below.
- The question must be NATURAL and something a researcher might actually ask. It should ONLY have one main component (no "and" or multiple sub-questions). It is better to keep it simple.
- The question should be in English.

EXAMPLE STRATEGIES TO CONSIDER:
{EXAMPLE_STRATEGIES}

Here are a few examples:
{FEW_SHOT_EXAMPLES}
"""

PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE = """You are an expert in constructing challenging research questions.

Given a seed question, produce an updated question designed to expose weaknesses in deep research systems. 

ANSWERING SYSTEM PROFILE:
{ANSWERING_SYSTEM_PROFILE}

OUTPUT FORMAT (valid JSON, no extra text):
{
  "brainstorming": "<in under 4 sentences, reason about a couple strategies and why they might or might not work for THIS seed>",
  "chosen_strategy": "<name and explain the single most promising strategy>",
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
{FEW_SHOT_EXAMPLES}

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

# Same task as JUDGE_PROMPT_TEMPLATE, for answering systems that cannot produce
# inline citations (WebThinker, Tongyi). The Sources section lists what the system
# retrieved, in trace order -- it is NOT a claim-level attribution, and nothing in
# the answer points into it. Without saying so, the judge reads every claim as
# uncited and fails the answer on format instead of substance, which would make
# the FAILED_FOUND rate a property of the output format rather than the question.
JUDGE_PROMPT_NO_INLINE_CITES = """You are an expert evaluator of deep research system outputs.

You will be given:
- A research question
- The verification criterion that defines what a good answer must do
- The answer the research system produced, followed by a SOURCES section

IMPORTANT — how to read the SOURCES section: this answering system cannot produce inline citations. The SOURCES section lists the pages the system retrieved while researching, in the order it found them. It is a record of what the system consulted, NOT a claim-by-claim attribution: no sentence in the answer points to any particular source, and a source being listed does not mean it supports any specific claim. Some listed sources may not have been used at all.

Therefore:
- Do NOT penalise the answer for lacking inline citations or a bibliography; this system structurally cannot produce them.
- Do NOT treat a claim as unsupported merely because no source is attached to it.
- DO use the sources as context for spotting claims that contradict the retrieved material, or that concern entities absent from every source (a likely fabrication).

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

VERIFY_MAX_EXTRA_QUERIES = 3

# PROMPT_FOR_SEED_CRITERION hands underspecified seeds this escape hatch, and
# JUDGE_PROMPT_TEMPLATE passes anything carrying it. 18% of round-0 criteria on disk are
# this, so --include-seed-round skips them: there is no factual claim to check.
SEED_CRITERION_ESCAPE_HATCH = "any non-empty answer is acceptable"


def is_escape_hatch_criterion(criterion: str) -> bool:
    return SEED_CRITERION_ESCAPE_HATCH in (criterion or "").strip().lower()

PROMPT_TO_PROPOSE_VERIFY_QUERIES = """You are generating targeted literature-search queries to help fact-check a VERIFICATION CRITERION for a research question.

Search results are already retrieved for the QUESTION itself. Your job is to propose only additional searches that would help determine whether the VERIFICATION CRITERION is factually correct.

RULES:
- Propose 0 to {max_queries} queries. Propose NONE if searching the question itself is likely to surface enough evidence to evaluate the criterion. An empty list is valid and should be common.
- Each query must target a specific factual claim or necessary assumption made by the criterion; do not generate near-duplicate searches.
- Prefer queries that could DISCRIMINATE between the criterion being correct and incorrect, rather than queries that merely provide general background.
- Write queries that would help SETTLE the claim either way. For claims about absence, consensus, uniqueness, causality, or lack of evidence, actively consider searches for contradictory evidence.
- Do not assume the criterion is correct. The purpose of the search is to verify it, not to find support for it.
- Queries are run verbatim against a literature search engine. Use short, natural-language keyword phrases, not Boolean syntax or full questions.
- Do not restate or paraphrase the QUESTION. 

QUESTION:
{question}

VERIFICATION CRITERION:
{criterion}

OUTPUT FORMAT (valid JSON, no extra text):
{{
  "queries": [
    {{
      "query": "<keyword phrase to search>"
    }}
  ]
}}
"""


PROMPT_TO_VERIFY_VERIFICATION_CRITERIA = """You are an expert meta-evaluator for a deep-research benchmark with difficult questions. Your only task is to judge whether the VERIFICATION CRITERION itself is correct and valid as an evaluation standard for the given question.

A verification criterion is a gatekeeper requirement: EVERY fully correct answer to the question must satisfy it. If there exists a plausible fully correct answer that would fail the criterion, then the criterion is too restrictive and is not correct as written.

Your task has two components:

1. FACTUAL VALIDITY:
   Determine whether the criterion's factual expectations, premises, causal claims, comparisons, mechanisms, entities, time frames, required distinctions, and absence/uncertainty claims are true and evidence-supported.

2. GATEKEEPER NECESSITY:
   Determine whether every requirement imposed by the criterion is strictly necessary for a fully correct answer to the question. Judge necessity from the question itself. A criterion must capture something that a fully correct answer cannot omit, contradict, or frame differently while still correctly answering the question. When a criterion names specific examples, mechanisms, subcases, or pieces of evidence, determine whether they are merely illustrative of the core requirement or are being imposed as mandatory answer content. If the criterion makes them mandatory but a fully correct answer could satisfy the core requirement without mentioning those specific items, treat that part of the criterion as over-restrictive and unfair.

A true or useful statement is not automatically a valid verification criterion; it must be required for correctness.

BENCHMARK CONTEXT:

* The answering system being tested uses Semantic Scholar / academic-paper search.
* A verification criterion is supposed to define one property that every fully correct answer to the question should satisfy.
* The criterion is used as a gatekeeper: an answer that fails it may be judged incorrect.

EVIDENCE RULES:

* Use the provided local Semantic Scholar search results.
* If the criterion embeds a specific expected fact, verify that fact directly.
* If the criterion requires an answer to reject a false premise, verify that the premise is actually false.
* If the criterion requires uncertainty, no causal isolation, no consensus, lack of evidence, or absence of studies, verify that this is a fair characterization of the available evidence rather than an unsupported negative claim.
* If the criterion requires a particular interpretation of an ambiguous question, check whether other reasonable interpretations could also yield fully correct answers.
* If a factual claim central to evaluating the criterion cannot be verified from the provided evidence, use insufficient_evidence when appropriate.
* In checked_claims, include ONLY factual claims made by or required by the verification criterion itself. Alternative hypotheses, possible counterexamples, and anything else surfaced by the search belong in reasoning, not in checked_claims.

UNFAIR REQUIREMENTS:

Identify any part of the criterion that could cause an accurate, relevant, and fully correct answer to be rejected.

Examples include requiring:

* one specific framing when multiple valid framings exist,
* requiring a categorical conclusion when the justified conclusion should depend on the strength, type, or scope of the evidence or method available,
* an explicit caveat that is useful but not necessary,
* a particular mechanism when the question can be correctly answered at a higher level,
* rejection of a premise that is not actually required to answer the question,
* one interpretation of an ambiguous term when another reasonable interpretation is valid,
* an unsupported level of certainty,
* an unnecessarily narrow population, time frame, mechanism, comparison, or evidence/study type.

REQUESTING ADDITIONAL SEARCHES:

The provided search results are normally what you must work with. Use additional_queries SPARINGLY — only when a claim genuinely cannot be settled from the context given. Leave the list empty in every other case.

* Always return your best provisional correctness_label from the evidence you already have.
* If a search would merely add corroborating detail, do not request it.

LABEL DEFINITIONS:
* correct: The criterion is factually supported, accurately framed, and valid as a gatekeeper. Every fully correct answer to the question should satisfy it. The criterion can be used as-is.
* almost_correct: The criterion's core requirement is valid and necessary, but some wording, scope, certainty, causal framing, entity mapping, or required distinction is materially imprecise or over-restrictive. It can be fixed with a local revision while preserving what the criterion is fundamentally testing.
* incorrect: The criterion's core requirement is false, unsupported, or not necessary for correctness. Also choose this if a plausible fully correct answer could fail the criterion because the central requirement is optional, overly specific, or tied to only one valid interpretation or framing. Fixing it would require removing or substantially replacing what the criterion is fundamentally testing.
* insufficient_evidence: The provided evidence is not adequate to verify a factual claim central to determining whether the criterion is valid.

Question:
{question}

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
"evidence": "<concise evidence for whether this criterion claim is true>",
"sources": ["<local S2 title or URL>", "https://..."]
}}
],
"unfair_requirements": "<any requirement that could cause a correct answer to be rejected. Empty string if there are none.>",
"reasoning": "<reason about both factual validity and gatekeeper necessity>",
"correctness_label": "correct | almost_correct | incorrect | insufficient_evidence",
"main_correctness_problem": "<one sentence naming the single most serious defect in the criterion. For over-restrictive criteria, state the requirement that a fully correct answer need not satisfy. Empty string if and only if correctness_label is 'correct'. For insufficient_evidence, name the specific criterion claim that could not be verified.>",
"rewrite": "<one corrected verification criterion for the given question if the criterion is not correct but can be fixed. The rewrite must itself be something every fully correct answer should satisfy. Empty string if correctness_label is 'correct' or if the criterion cannot be easily corrected without changing what it fundamentally tests.>",
"additional_queries": [
{{
"query": "<a specific search whose results would settle a claim you could not settle from the provided context>",
"targets_claim": "<which checked_claims entry this would resolve>"
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
        provider: Optional[str] = None,
    ) -> None:
        # `kind` stays "claude_call" whichever provider served it: summarize_run.py and
        # loop_report.py key off that string, and renaming it would orphan every log
        # already on disk. `provider` is the field to read.
        self._write({
            "kind": "claude_call",
            "seed": seed,
            "attempt": attempt,
            "purpose": purpose,
            "model": model,
            "provider": provider,
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


# Labels that mean almost_correct. "partly_correct" is the pre-rename name and
# appears in ~60 stored criterion checks on disk; "partially_correct" is what a
# model writes when it paraphrases the enum. Both normalise to the canonical
# value so old runs stay readable and a paraphrase does not silently become an
# unrecognised label (which would stop the seed instead of applying the rewrite).
ALMOST_CORRECT = "almost_correct"
CRITERION_LABEL_ALIASES = {
    "partly_correct": ALMOST_CORRECT,
    "partially_correct": ALMOST_CORRECT,
}
# Labels whose seeds continue (applying any rewrite) rather than stopping.
KEEP_LABELS = ("correct", ALMOST_CORRECT)


def normalize_criterion_label(label: str) -> str:
    """Canonical criterion-check label, mapping legacy/paraphrased spellings."""
    clean = str(label or "").strip().lower()
    return CRITERION_LABEL_ALIASES.get(clean, clean)


@dataclass
class CriterionCheck:
    """Meta-judge verdict on the verification criterion itself (round 1+ only)."""
    correctness_label: str
    main_correctness_problem: str
    reasoning: str
    rewrite: str
    # Requirements the criterion imposes that could cause a fully correct answer to
    # be rejected. Empty string when there are none.
    unfair_requirements: str = ""
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

def effective_decomposer_model(decomposer_model: Optional[str],
                               model: str) -> Optional[str]:
    """Which model retrieve_papers will decompose with, given the flag and --model.

    An explicit --decomposer-model wins. Otherwise it follows `model`'s provider, so
    `--model gpt-5.6-terra` yields an all-OpenAI run needing only OPENAI_API_KEY rather
    than reaching Anthropic for this one step. Callers pass the result to retrieval
    explicitly, so the choice is visible in the log rather than resolved down inside
    retrieve_papers.

    Note the decomposer picks the papers, so its provider shifts retrieval and hence what
    the meta-judge sees -- see retrieve_papers.DEFAULT_DECOMPOSER_MODELS.

    Imported inside the function to keep the retrieval stack optional for runs without
    --verify-criterion; returns None if it is not installed, leaving the check to skip.
    """
    if decomposer_model:
        return decomposer_model
    try:
        from retrieve_papers import default_decomposer_model
    except ImportError:
        return None
    return default_decomposer_model(resolve_provider(model))


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
# LLM calls
# ---------------------------------------------------------------------------

def _call_llm(
    client,
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
    provider: Optional[str] = None,
) -> tuple[str, CostBucket]:
    """Single Claude-or-GPT call wrapped with logging + cost accounting.

    `provider` defaults to whatever `model`'s id resolves to (see llm_client); the
    request shape, the reasoning-token headroom, and the usage field names all live
    there, so nothing above this function branches on provider.

    `log_messages`, if given, is logged instead of the real `messages` — used to
    redact the bulky research answer from the judge prompt in the log.
    """
    provider = resolve_provider(model, provider)
    logged_messages = log_messages if log_messages is not None else messages
    t0 = time.perf_counter()
    err: Optional[str] = None
    response_text = ""
    usage: dict = {}
    cost_usd = 0.0

    try:
        response_text, usage = llm_client.call_text(
            client, provider, model=model, system=system,
            messages=messages, max_tokens=max_tokens,
        )
        cost_usd, usage = price_call(model, usage)
    except Exception as e:
        # A response that arrived but carried no usable text was still billed, so
        # recover its usage: otherwise a refusal or a reasoning-truncated call drops
        # its (already-paid-for) input tokens from both the log and the cost total.
        if isinstance(e, llm_client.EmptyResponse) and e.usage:
            cost_usd, usage = price_call(model, e.usage)
        err = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        logger.log_claude_call(
            seed=seed, attempt=attempt, purpose=purpose, model=model,
            system=system, messages=logged_messages, response_text=response_text,
            usage=usage, cost_usd=cost_usd, latency_s=latency, error=err,
            provider=provider,
        )
        raise

    latency = time.perf_counter() - t0
    logger.log_claude_call(
        seed=seed, attempt=attempt, purpose=purpose, model=model,
        system=system, messages=logged_messages, response_text=response_text,
        usage=usage, cost_usd=cost_usd, latency_s=latency, error=None,
        provider=provider,
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
    client,
    model: str,
    seed: str,
    logger: RunLogger,
) -> tuple[str, CostBucket]:
    """Generate one atomic verification criterion for the seed question (round 0)."""
    prompt = PROMPT_FOR_SEED_CRITERION.format(question=seed)
    messages = [{"role": "user", "content": prompt}]
    raw, bucket = _call_llm(
        client, model=model, system=None, messages=messages,
        max_tokens=800, logger=logger, seed=seed, attempt=0, purpose="seed_criterion",
    )
    data = extract_json(raw)
    return str(data.get("verification_criterion") or ""), bucket


def harder_question_gen(
    client,
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
    raw, bucket = _call_llm(
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


def format_answer_for_judge(answer: str, trace: object) -> tuple[str, bool]:
    """Render an answer the way the HTML viewer does, in plain text.

    Returns (text, inline_cites) -- the flag picks the judge prompt, because the
    two cases ask the judge for different things.

    DR-Tulu's opaque `<cite id="...">` tags become inline [n] markers backed by a
    References section carrying each source's title, authors, and retrieved snippet,
    so the judge can check a claim against the text it cites rather than against the
    system's own paraphrase of it.

    WebThinker emits no inline citations at all -- its refined article contains
    zero URLs and zero [n] markers -- so there is nothing to resolve per claim.
    Instead the sources its trace shows it consulted are appended as a flat list.
    That is provenance, not attribution: it says what informed the report, never
    which source backs a given sentence. JUDGE_PROMPT_NO_INLINE_CITES says so,
    so the judge does not mark every claim uncited and fail the answer on a
    formatting mismatch rather than on substance.

    Answers with no citations and no usable trace (Tongyi) pass through unchanged.
    """
    doc_index = build_doc_index(trace)
    if has_inline_citations(answer):
        marked, refs = numbered_plaintext(answer, doc_index)
        return marked + references_block(refs, include_snippets=True), True
    refs = selected_refs(doc_index)
    return (answer or "") + references_block(
        refs, include_snippets=True,
        title="SOURCES (retrieved while researching; not claim-level citations)",
    ), False


def judge_answer(
    client,
    model: str,
    question: str,
    criterion: str,
    answer: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
    trace: object = None,
) -> tuple[Judgment, CostBucket]:
    judge_answer_text, inline_cites = format_answer_for_judge(answer, trace)
    template = JUDGE_PROMPT_TEMPLATE if inline_cites else JUDGE_PROMPT_NO_INLINE_CITES
    prompt = template.format(
        question=question, criterion=criterion, answer=judge_answer_text
    )
    messages = [{"role": "user", "content": prompt}]
    # Log the prompt with the (bulky) answer redacted; it lives in the results file.
    log_prompt = template.format(
        question=question, criterion=criterion,
        answer=f"<answer + references omitted: {len(judge_answer_text)} chars — "
               f"see results file>",
    )
    log_messages = [{"role": "user", "content": log_prompt}]
    raw, bucket = _call_llm(
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


def propose_verify_queries(
    client,
    model: str,
    question: str,
    criterion: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
    max_queries: int = VERIFY_MAX_EXTRA_QUERIES,
) -> tuple[list[str], CostBucket]:
    """Criterion-aware search queries to run alongside the question's own retrieval.

    Retrieval for the criterion check is driven by `retrieve_papers(question)`, so the
    criterion's own requirements -- a named entity, a required distinction, a demanded
    study design -- never reach the search engine. This asks for up to `max_queries`
    searches that target them. The queries widen the candidate pool only: everything is
    reranked together and cut to n_context_papers, so the verify prompt does not grow.

    Returns `(queries, bucket)`, the query strings in prompt order (possibly empty -- the
    prompt says an empty list should be common). A failure here is non-fatal: the check
    falls back to question-only retrieval rather than losing the seed.
    """
    prompt = PROMPT_TO_PROPOSE_VERIFY_QUERIES.format(
        question=question, criterion=criterion, max_queries=max_queries)
    raw, bucket = _call_llm(
        client, model=model, system=None,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200, logger=logger, seed=seed, attempt=attempt,
        purpose="propose_verify_queries",
    )
    data = extract_json(raw)
    out: list[str] = []
    for q in (data.get("queries") or []):
        # the schema asks for {"query": ...}; a bare string is accepted too
        text = q.get("query") if isinstance(q, dict) else q
        text = text.strip() if isinstance(text, str) else ""
        if text and text not in out:
            out.append(text)
    # cap AFTER filtering, so a blank or duplicate entry does not eat a query slot
    return out[:max_queries], bucket


def verify_criterion(
    client,
    model: str,
    question: str,
    criterion: str,
    logger: RunLogger,
    seed: str,
    attempt: int,
    retrieval_kwargs: Optional[dict] = None,
    n_context_papers: int = VERIFY_N_PAPERS,
    max_chars_per_paper: int = VERIFY_MAX_CHARS_PER_PAPER,
    propose_queries: bool = False,
    max_extra_queries: int = VERIFY_MAX_EXTRA_QUERIES,
) -> tuple[CriterionCheck, CostBucket]:
    """Retrieve papers for `question`, then judge whether `criterion` is itself correct.

    retrieve_papers() runs its own LLM call for query decomposition, with its own client
    and its own (independently selectable) model — so the decomposer can sit on a
    different provider than the meta-judge. That call is not logged as a claude_call,
    but its usage is reported back and folded into the returned CostBucket (and into
    the criterion_check log record).
    """
    # Imported lazily so the pipeline still runs without the retrieval stack installed.
    from retrieve_papers import retrieve_papers

    t0 = time.perf_counter()
    retrieval_meta: dict = {}
    decompose_bucket = CostBucket()

    # Criterion-aware queries, if enabled. Non-fatal: a failure here degrades to
    # question-only retrieval rather than dropping the seed.
    proposed: list[str] = []
    propose_bucket = CostBucket()
    propose_error = None
    if propose_queries and max_extra_queries > 0:
        try:
            proposed, propose_bucket = propose_verify_queries(
                client, model, question, criterion, logger, seed, attempt,
                max_queries=max_extra_queries,
            )
        except Exception as e:
            propose_error = f"{type(e).__name__}: {e}"

    try:
        retrieved = retrieve_papers(
            question,
            extra_queries=proposed or None,
            **(retrieval_kwargs or {}),
        )
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
            # None = retrieval ran WITHOUT reranking (see retrieve_papers.build_reranker)
            "reranker": retrieved.get("reranker"),
            "n_snippets": retrieved.get("n_snippets"),
            "n_keyword_papers": retrieved.get("n_keyword_papers"),
            "elapsed_s": retrieved.get("elapsed_s"),
            # criterion-aware pre-retrieval: the queries asked for and what they added.
            # Empty list = the step ran and proposed nothing (the prompt says that should
            # be common); None = the step was off.
            "proposed_queries": proposed if propose_queries else None,
            "extra_query_results": retrieved.get("extra_queries"),
            "n_extra_papers": retrieved.get("n_extra_papers"),
            "propose_error": propose_error,
        }
        # retrieve_papers' query-decomposition call bills to us; price it here since it
        # never passes through _call_llm.
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
        question=question, criterion=criterion,
        search_results_context=context,
    )
    messages = [{"role": "user", "content": prompt}]
    # Keep the bulky retrieved context out of the claude_call log record; the
    # criterion_check record carries its size and the queries that produced it.
    log_messages = [{"role": "user", "content": PROMPT_TO_VERIFY_VERIFICATION_CRITERIA.format(
        question=question, criterion=criterion,
        search_results_context=(
            f"<{retrieval_meta['n_context_papers']} papers omitted: "
            f"{retrieval_meta['context_chars']} chars>"
        ),
    )}]

    try:
        raw, bucket = _call_llm(
            client, model=model, system=None, messages=messages,
            max_tokens=3000, logger=logger, seed=seed, attempt=attempt,
            purpose="verify_criterion", log_messages=log_messages,
        )
        bucket.add(decompose_bucket)
        bucket.add(propose_bucket)
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
            correctness_label=normalize_criterion_label(data.get("correctness_label")),
            main_correctness_problem=data.get("main_correctness_problem", ""),
            unfair_requirements=str(data.get("unfair_requirements") or "").strip(),
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
    client,
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
    propose_queries: bool = False,
    max_extra_queries: int = VERIFY_MAX_EXTRA_QUERIES,
    include_seed_round: bool = False,
    skip_seed_round: bool = False,
) -> SeedResult:
    result = SeedResult(seed=seed)

    # Round 0 tests the seed as-is; rounds 1..N test harder rewrites. skip_seed_round
    # starts at 1, so ALREADY_HARD cannot occur and every seed yields a generated rewrite
    # -- which is what feeds strategy clustering, since round-0 attempts are excluded from
    # it (they carry no strategy). The cost is losing the "already hard" signal: a seed the
    # system already fails gets hardened anyway, and may overshoot into unanswerable.
    for attempt in range(1 if skip_seed_round else 0, max_attempts + 1):
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
        # it. Rounds 1+ by default; --include-seed-round extends it to round 0, whose
        # criterion comes from generate_seed_criterion. The escape hatch is skipped either
        # way: "any non-empty answer is acceptable" states no fact to verify, and the judge
        # passes it unconditionally.
        criterion_check = None
        check_this_attempt = verify_criteria and (
            attempt > 0
            or (include_seed_round
                and not is_escape_hatch_criterion(harder.verification_criterion))
        )
        if check_this_attempt:
            try:
                criterion_check, bucket = verify_criterion(
                    client, model, harder.updated_question,
                    harder.verification_criterion, logger, seed, attempt,
                    retrieval_kwargs=retrieval_kwargs,
                    n_context_papers=n_context_papers,
                    max_chars_per_paper=max_chars_per_paper,
                    propose_queries=propose_queries,
                    max_extra_queries=max_extra_queries,
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
                if criterion_check.unfair_requirements:
                    print(f"     Unfair requirement: {criterion_check.unfair_requirements}")
                if criterion_check.additional_queries:
                    print(f"     Requested searches: {criterion_check.additional_queries}")

            if criterion_check.correctness_label not in KEEP_LABELS:
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
                        help=f"Generator/judge model (default: {CLAUDE_MODEL}). Accepts a "
                             f"Claude id or an OpenAI one (e.g. gpt-5.6-terra); the "
                             f"provider is inferred from the id unless --provider says "
                             f"otherwise.")
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
        "--prompt", choices=["explore", "exploit"], default="explore",
        help="Which make-harder system prompt to use (default: explore). "
             "'explore' forbids reusing the example strategies; 'exploit' keeps the "
             "in-context strategy menu.",
    )
    parser.add_argument(
        "--profile", choices=sorted(ANSWERING_SYSTEM_PROFILES), default=DEFAULT_PROFILE,
        help=f"Answering-system profile injected into the make-harder prompt "
             f"(default: {DEFAULT_PROFILE}).",
    )
    parser.add_argument(
        "--strategies", choices=sorted(STRATEGY_LISTS), default=DEFAULT_STRATEGIES,
        help=f"Which example-strategy menu to inject into the '--prompt exploit' "
             f"make-harder prompt (default: {DEFAULT_STRATEGIES}). Ignored by "
             f"'--prompt explore', which has no strategy menu.",
    )
    parser.add_argument(
        "--banned-strategies", choices=sorted(BANNED_STRATEGY_LISTS),
        default=DEFAULT_BANNED_STRATEGIES,
        help=f"Which banned-strategy menu to inject under 'STRATEGIES TO NOT USE' in the "
             f"'--prompt explore' make-harder prompt (default: {DEFAULT_BANNED_STRATEGIES}; "
             f"'none' drops the block). Ignored by '--prompt exploit'.",
    )
    parser.add_argument(
        "--few-shots-file", default=None,
        help="JSON file of worked examples (list of objects keyed like DEFAULT_FEW_SHOTS: "
             "seed_question, brainstorming, chosen_strategy, updated_question, why_harder, "
             "verification_criterion) shown under 'Here are a few examples:' INSTEAD of the "
             "built-in three. Applies to both prompt variants.",
    )
    parser.add_argument(
        "--strategies-file", default=None,
        help="File of example strategies (one per line, '#' comments ok) used INSTEAD of "
             "--strategies. For feeding a menu computed at runtime, e.g. by research_loop.py.",
    )
    parser.add_argument(
        "--banned-strategies-file", default=None,
        help="File of banned strategies (one per line, '#' comments ok) used INSTEAD of "
             "--banned-strategies. An empty file drops the 'STRATEGIES TO NOT USE' block.",
    )
    parser.add_argument(
        "--verify-criterion", action="store_true",
        help="Before each research-server call (rounds 1+), retrieve papers for the harder "
             "question and ask Claude whether the verification criterion is itself correct. "
             "Continues on 'correct'/'almost_correct' (applying any rewrite) and stops the "
             "seed on 'incorrect'/'insufficient_evidence'. Requires retrieve_papers.py and "
             "S2_API_KEY.",
    )
    parser.add_argument(
        "--verify-n-papers", type=int, default=VERIFY_N_PAPERS,
        help=f"Papers to include in the criterion-check context (default: {VERIFY_N_PAPERS}).",
    )
    parser.add_argument(
        "--skip-seed-round", action="store_true",
        help="Start at attempt 1 instead of testing the unmodified seed first. No seed can "
             "come back ALREADY_HARD, every seed produces a generated rewrite (round-0 "
             "attempts carry no strategy, so only rewrites feed strategy clustering), and "
             "one research call plus one seed-criterion call per seed are saved. In "
             "exchange a seed the system already fails is hardened anyway and may overshoot "
             "into unanswerable. Mutually exclusive with --include-seed-round.",
    )
    parser.add_argument(
        "--include-seed-round", action="store_true",
        help="Also verify the ROUND-0 criterion (the unmodified seed), which is skipped by "
             "default. Criteria that are just 'Any non-empty answer is acceptable' are "
             "still skipped — there is nothing to check. Note this can turn a seed that "
             "would have been ALREADY_HARD into CRITERION_INVALID: the check runs before "
             "the research call, so a rejected criterion stops the seed before the "
             "unmodified question is ever tested. Only meaningful with --verify-criterion.",
    )
    parser.add_argument(
        "--verify-propose-queries", action="store_true",
        help="Before retrieving for the criterion check, ask the model for up to "
             "--verify-max-extra-queries searches targeting what the CRITERION requires "
             "but the question does not say (a named entity, a required distinction, a "
             "demanded study design). Retrieval is otherwise driven by the question alone, "
             "so those claims never reach the search engine. The extra hits widen the "
             "candidate pool only -- everything is reranked together and cut to "
             "--verify-n-papers -- so the check's prompt does not grow. Off by default: it "
             "changes which papers the meta-judge sees, so runs with and without it are "
             "not comparable. Wants a reranker (see --reranker-url); unreranked, the extra "
             "candidates are merged in arbitrary order.",
    )
    parser.add_argument(
        "--verify-max-extra-queries", type=int, default=VERIFY_MAX_EXTRA_QUERIES,
        help=f"Cap on criterion-aware queries per check (default: "
             f"{VERIFY_MAX_EXTRA_QUERIES}); 0 disables them even with "
             f"--verify-propose-queries.",
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
    parser.add_argument(
        "--decomposer-model", default=None,
        help="Model for retrieve_papers' query decomposition during --verify-criterion "
             "(default: retrieve_papers' own default). Independent of --model, so the "
             "cheap structured-extraction step can stay on a cheap model.",
    )
    llm_client.add_provider_arg(parser)
    args = parser.parse_args()
    llm_client.configure_from_args(args)

    profile_text = ANSWERING_SYSTEM_PROFILES[args.profile]

    decomposer = effective_decomposer_model(args.decomposer_model, args.model)
    retrieval_kwargs = {
        "reranker": args.reranker,
        "reranker_url": args.reranker_url,
    }
    if decomposer:
        retrieval_kwargs["decomposer_model"] = decomposer

    base_template = (PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE if args.prompt == "explore"
                     else PROMPT_TO_MAKE_HARDER_QUESTION_EXPLOIT)
    example_strategies = (load_strategy_file(args.strategies_file) if args.strategies_file
                          else STRATEGY_LISTS[args.strategies])
    banned_strategies = (load_strategy_file(args.banned_strategies_file)
                         if args.banned_strategies_file
                         else BANNED_STRATEGY_LISTS[args.banned_strategies])
    few_shots = (load_few_shots_file(args.few_shots_file) if args.few_shots_file
                 else DEFAULT_FEW_SHOTS)
    harder_prompt = with_few_shots(
        with_banned_strategies(
            with_strategies(
                with_profile(base_template, profile_text), example_strategies
            ),
            banned_strategies,
        ),
        few_shots,
    )

    seeds = list(args.seeds)
    if args.seeds_file:
        with open(args.seeds_file) as f:
            seeds.extend(line.strip() for line in f if line.strip())
    if not seeds and not sys.stdin.isatty():
        seeds.extend(line.strip() for line in sys.stdin if line.strip())
    if not seeds:
        parser.error("No seed questions provided.")

    if args.skip_seed_round and args.include_seed_round:
        parser.error("--skip-seed-round and --include-seed-round contradict: one removes round 0, "
             "the other verifies its criterion. Pick one.")
    if args.skip_seed_round and args.max_attempts < 1:
        parser.error("--skip-seed-round needs --max-attempts >= 1, or no attempt runs at all.")

    missing_key = llm_client.require_api_key(args.model)
    if missing_key:
        parser.error(missing_key)
    if args.verify_criterion and decomposer:
        if missing_key := llm_client.require_api_key(decomposer):
            parser.error(missing_key + llm_client.decomposer_hint(args.model, decomposer))

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

    provider = resolve_provider(args.model)
    print(f"Model: {args.model} (provider: {provider}"
          + (f", reasoning_effort: {args.reasoning_effort}"
             if provider == llm_client.OPENAI and args.reasoning_effort else "")
          + ")")
    if args.verify_criterion and decomposer:
        print(f"Decomposer: {decomposer} "
              f"(provider: {resolve_provider(decomposer)})")
    client = llm_client.make_client(args.model)

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
            propose_queries=args.verify_propose_queries,
            max_extra_queries=args.verify_max_extra_queries,
            include_seed_round=args.include_seed_round,
            skip_seed_round=args.skip_seed_round,
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
                    "prompt_variant": args.prompt,
                    "example_strategies": example_strategies,
                    "banned_strategies": banned_strategies,
                    "few_shot_seeds": [s.get("seed_question", "") for s in few_shots],
                    "grand_total_cost": asdict(grand_total),
                    "results": [result_to_dict(r) for r in all_results],
                },
                f,
                indent=2,
            )
        print(f"Final results written to: {args.output}")


if __name__ == "__main__":
    main()