#!/usr/bin/env python3
"""
Final verification of a finished run's harvest: is each question benchmark-worthy, and is
its verification criterion (VC) appropriate for it?

Two independent reviews per harvested question, both by a strong model (default
gpt-5.6-sol), in this order:

  1. QUESTION review -- sees ONLY the question. PASS / MINOR_REVISION / REJECT.
  2. VC review       -- sees the question and the VC, nothing from review 1. When review 1
                        proposed a rewritten question, the VC is judged against that
                        rewrite, since it is the question that would ship. Only the VC is
                        judged (vc_verdict); the pair verdict is the worse of the two.

The harvest is the deciding attempt of every FAILED_FOUND seed, with the VC the judge
actually graded against (i.e. after any inline criterion-check rewrite). Nothing else from
the run -- the answer, the judge's reasoning, the inline criterion check, the seed -- is
shown to either reviewer, so the reviews are independent of how the question was made.

Rejected without any model call (`skip_reason`), since no review could rescue them:

    criterion_satisfied  the run's judge found the answer MET the VC and failed it only for
                         other issues -- the question never actually broke the system
    too_long             the question is over --max-question-words (default 100)

and the VC review is skipped when the question review is REJECT (`vc_skipped`): the pair
cannot be better than the question, so the verdict is already decided.

With --phrase-check, each question first gets a phrase check: one LLM call extracts its
proper nouns and multi-word technical terms, each is looked up on Semantic Scholar as an
exact phrase (title/abstract, /paper/search/bulk), and a term with fewer than
--phrase-min-hits papers (default 1, i.e. zero hits) rejects the question before either
review runs (`phrase_rejected`, details in `phrase_check`). It targets what a reviewer
reading only the question cannot see: fluent coinages and unidentifiable names. A failed
search is recorded as unknown and never rejects. Set S2_API_KEY to avoid rate limits.
Terms over 4 words or containing an acronym/code are counted but never reject, and dashes
are searched as hyphens. --phrase-mode evidence never rejects on the check; it appends the
counts to the question reviewer's input instead.
--phrase-mode mixed rejects on a zero-hit single word and passes multi-word phrases
to the reviewer as evidence.

Final decision per question:

    pass    question PASS and pair PASS             -> usable as written
    revise  worst verdict is MINOR_REVISION         -> proposed rewrites recorded, NOT applied
    reject  any REJECT, or a skip rule above        -> drop
    error   a review failed                         -> retried on the next run

Rewrites are never applied automatically: the question's FAILED label was earned by the
original question against the original VC. A rewritten question was never answered, and a
rewritten VC was never graded against, so a `revise` row needs a re-answer / re-judge
before it can join the benchmark as a failing question.

Outputs, in <run_dir>/final_verification/ (or --out-dir):
    results.jsonl   one row per question: both reviews, final decision, cost
    summary.json    counts by verdict and by system / prompt / round, total cost
    pass.jsonl      the benchmark-ready set (question + VC as written)
    revise.jsonl    rows with proposed rewrites, flagged needs_rejudge

Resumable: rows already in results.jsonl are skipped, errored rows are retried.

Examples:
  # pilot on 30 questions first to check verdict rates and cost
  python final_verification.py runs/final_loop_600 --limit 30

  python final_verification.py runs/final_loop_600 --concurrency 8 --budget-usd 100

Requires OPENAI_API_KEY (or ANTHROPIC_API_KEY for a Claude --model).
"""

import argparse
import datetime as dt
import json
import random
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import llm_client

DEFAULT_MODEL = "gpt-5.6-sol"
# Sized for visible output; llm_client scales it for OpenAI reasoning (x4, floor 8000),
# so a strong model thinking hard does not truncate to an empty response.
MAX_TOKENS = 4000

SAMPLE_RE = re.compile(r"^sample_\d+\.json$")   # not sample_NNN.compare.json etc.
VERDICTS = ("PASS", "MINOR_REVISION", "REJECT")
SEVERITY = {v: i for i, v in enumerate(VERDICTS)}

QUESTION_FILTER_PROMPT = """
You are reviewing a candidate research question for inclusion in a benchmark of hard research questions. The benchmark is for questions that can be answered by finding, comparing, and evaluating existing published research. Reject questions that primarily ask the answerer to conduct or design a study, experiment or coding project.

Judge ONLY the question itself.

The benchmark should contain questions that are difficult because they require strong research, evidence evaluation, or reasoning—not because they are vague, malformed, underspecified, or artificially written.


A good question should:
- be understandable on its own
- identify key terms, entities, programs, datasets, frameworks, sources, or referents
- use terminology that is standard in the relevant field or defined in the question
- sound reasonably natural for a knowledgeable researcher to ask
- contain one coherent research task, even if technically difficult or complex
- not require the reader to invent substantial missing context or assumptions

Do NOT penalize a question merely because:
- it is difficult
- its premise may be false
- the answer may be "no"
- no qualifying evidence may exist
- it asks whether a causal claim, comparison, ranking, mechanism, or inference is valid
- it uses specialized technical terminology that has an established meaning in the relevant field
- it uses formal academic language

Flag only MATERIAL issues, including:

- undefined or nonstandard terminology, including non-existent or unrecognized compound phrases
- non-existent words or unknown entities
- missing referent, such as "this program", "the framework", "the dataset", or "the intervention"
- a specific paper, source, dataset, method, program, framework, or other entity that the question depends on is not identifiable from the wording
- the comparison, ranking, superiority claim, or prioritization lacks a basis that is necessary to determine what is being compared or what outcome would establish the claim
- multiple substantially separable research tasks combined into one question
- wording that is so synthetic, compressed, or awkward that it materially reduces clarity
- unnecessary formatting requirements that primarily test instruction-following rather than research ability
- cannot be answered by consulting and using academic papers from semantic scholar

Important rules:
1. Do not flag specialized terminology merely because it is unfamiliar to a general reader. Flag it only when it is undefined, nonstandard, materially ambiguous, or not reasonably identifiable within the relevant field.

2. A question may intentionally test whether evidence supports a claim. Do not penalize it because the likely answer is negative.

3. Most well-formed questions should PASS. Don't try to hunt for a problem if there is no obvious issues.

4. Brevity is not underspecification. A short question should PASS if a knowledgeable researcher can identify the research task and the key construct or comparison from the wording as written. Do not require background context, formal definitions, named theories, named sources, dates, populations, mechanisms, or other details merely because adding them would make the question more specific. Research questions commonly permit judgment about which studies, populations, measures, time periods, and methods are most relevant. That is part of the research task, not a defect in the question.

5. REJECT a question if it is structurally overloaded, with too many clauses, constraints, entities, methods, comparisons, qualifiers, or inferential steps packed together. Signs include nested clauses, stacked technical modifiers, multiple conditions or comparisons in a single sentence, long chains of qualifications, or wording that must be mentally reorganized or broken apart to identify the core research task. 

6. REJECT or require revision for invented or weakly established compound terminology. A phrase is not acceptable merely because each individual word is standard or because its intended meaning can be inferred compositionally. If a multi-word technical expression is not an established term, conventional formulation, or natural phrase that specialists in the relevant literature would plausibly use, flag it. In particular, be suspicious of adjective+noun or multi-modifier constructions that appear to combine legitimate concepts into a novel label, such as “Bayesian mechanism density,” “validity friction,” “cross-sectional treatment effect stability,” or “temporal role discovery.” Apply this check to technical phrases anywhere in the question, including phrases embedded within much longer, otherwise well-formed sentences. 

Classify the question as:

PASS
- if it is suitable as written

MINOR_REVISION
- if the intended research task is clear and a few small local edits would fix the problem
- if the underlying research task does not need to change

REJECT
- if it is hard to understand what the question is asking
- if the question has undefined or nonstandard terminology or compound phrases that researchers don't naturally use
- if there are missing referants in the question
- if the question would need major restructuring
- if it is overloaded, jargon-stacked, or word-salad-like: it packs so many technical terms, conditions, methods, comparisons, qualifications, or inferential steps that the wording becomes unnatural, difficult to parse, or hard to follow
- if it is a design or coding task instead of a deep research QA task

Use MINOR_REVISION sparingly only when required and when the problem can be fixed without materially redesigning the question. Do not expand abbreviations, add formal names, introduce new technical terminology, add background context, make implicit concepts explicit, or otherwise improve the question beyond what is necessary to fix the identified defect.

Return JSON only:

{
  "issues": [],
  "verdict": "PASS | MINOR_REVISION | REJECT",
  "explanation": "",
  "rewritten_question": null
}

If verdict is MINOR_REVISION, provide a minimally edited rewritten_question. Change only the wording necessary to correct the specific issue identified in issues. Preserve all other wording wherever reasonably possible. Do not use the rewrite as an opportunity to make the question more precise, formal, comprehensive, self-contained, or technically sophisticated. Do not add new entities, information or facts as part of the rewrite.
If verdict is PASS or REJECT, rewritten_question should normally be null.
"""

VC_FILTER_PROMPT = """
You are reviewing whether a verification criterion (VC) is appropriate for a research question in a benchmark of hard research questions. A VC is ONE necessary condition for a strong answer, not a checklist of the whole question. The question has already been evaluated separately. The VC's factual claims were also checked separately against retrieved literature. 

Your task is to judge whether the VC itself is appropriate for the question as written

Do not use the VC to excuse, reinterpret, or repair defects in the question.

A good VC should:
- follow directly from the question as written
- test something genuinely necessary for a strong answer
- every good answer should pass the VC
- preserve the question's scope, framing, population, setting, timeframe, comparator, outcome, and level of analysis
- require evidence strength appropriate to the claim made in the question
- be specific enough that a verifier can determine whether an answer satisfies it
- allow multiple valid answers when the evidence permits them

Flag MATERIAL problems such as:

- the VC repairs ambiguity or missing context in the question
- the VC chooses one interpretation when the question reasonably allows several
- the VC adds an unstated comparator, population, setting, timeframe, outcome, mechanism, or level of analysis
- the VC has a missing referent, such as "this program", "the framework", "the dataset", or "the intervention"
- the VC turns a descriptive question into a causal one
- the VC requires a stronger inference than the question asks for
- the VC requires one specific method or study design when multiple valid methods could establish the relevant claim
- the VC forces a preferred conclusion rather than testing an evidentiary requirement
- the VC requires unnecessary completeness
- the VC treats desirable content as mandatory
- the VC uses vague or unverifiable requirements
- the VC adds formatting requirements that the question does not require
- the VC merely paraphrases the question without identifying a concrete, independently checkable requirement for a strong answer

Important principles:

1. For every substantive VC requirement, ask:
"What language in the question makes this requirement necessary?"

If there is no clear answer, the VC may be adding a hidden requirement.

2. Use this critical test:
"Could a knowledgeable researcher give a strong, accurate, responsive answer to the question as written and still fail this VC?"

If yes, the VC is probably too restrictive, too narrow, or adding something the question did not require.

3. Prefer requiring an evidentiary property rather than a specific method.

For example:
GOOD:
"The answer must distinguish evidence capable of supporting causal attribution from descriptive association."

TOO NARROW:
"The answer must use an instrumental-variable study."

unless the question specifically requires that method.

Classify the VC as:

PASS
- if it is appropriate and usable for the question as written

MINOR_REVISION
- if it is already substantively appropriate, with one or a few localized changes
- if there is a specific, localized defect that can be fixed with a small edit to the existing wording

REJECT
- if it is substantially mismatched, overly restrictive, or fundamentally unsuitable for the question, or would need substantial rewriting

Return JSON only:

{
  "issues": [],
  "vc_verdict": "PASS | MINOR_REVISION | REJECT",
  "explanation": "",
  "rewritten_vc": null
}

If vc_verdict is MINOR_REVISION, provide a rewritten_vc. Do not add new requirements, examples, exceptions, distinctions, evidentiary standards, methods, concepts, or subcriteria. Use MINOR_REVISION only for local rephrasing or for removing or weakening a specific requirement that is too strong, while preserving the rest of the VC as much as possible. A rewritten VC must stay checkable by a judge without an answer key, and should remain substantively meaningful and nontrivial.
Otherwise rewritten_vc should normally be null.
"""

# Strict-mode schemas mirroring the JSON the prompts ask for. Enforcing them at the API
# means an out-of-vocabulary verdict or a malformed escape can never reach the parser --
# the pipeline lost a seed to "Invalid \escape" from a free-form JSON reply.
_VERDICT = {"type": "string", "enum": list(VERDICTS)}
_ISSUES = {"type": "array", "items": {"type": "string"}}
_NULLABLE_STR = {"type": ["string", "null"]}

# Property order matches the prompts (issues before verdict): strict-mode output is
# generated in schema order, so the verdict comes after the issues it rests on.
QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"issues": _ISSUES, "verdict": _VERDICT, "explanation": {"type": "string"},
                   "rewritten_question": _NULLABLE_STR},
    "required": ["issues", "verdict", "explanation", "rewritten_question"],
    "additionalProperties": False,
}
VC_SCHEMA = {
    "type": "object",
    "properties": {"issues": _ISSUES, "vc_verdict": _VERDICT,
                   "explanation": {"type": "string"}, "rewritten_vc": _NULLABLE_STR},
    "required": ["issues", "vc_verdict", "explanation", "rewritten_vc"],
    "additionalProperties": False,
}


def _worst(*verdicts) -> str:
    return max((v for v in verdicts if v), key=SEVERITY.__getitem__)


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------


def collect_harvest(run_dir: Path) -> list:
    """The deciding attempt of every FAILED_FOUND seed under `run_dir`, with its metadata.

    Read straight from the sample files rather than through load_examples_from_runs,
    which dedups across runs and drops the per-file identity (round / system / prompt /
    sample) that the output needs to point back at the run.
    """
    items = []
    for path in sorted(run_dir.rglob("sample_*.json")):
        if not SAMPLE_RE.match(path.name):
            continue
        try:
            res = (json.loads(path.read_text()).get("results") or [{}])[0]
        except (ValueError, OSError):
            continue                     # truncated file: not part of the harvest
        if res.get("final_status") != "FAILED_FOUND" or not res.get("attempts"):
            continue
        last = res["attempts"][-1]
        harder = last.get("harder") or {}
        rel = path.relative_to(run_dir).with_suffix("")
        parts = rel.parts                # round_KK/<system>/<prompt>/sample_NNN, or shorter
        judgment = last.get("judgment") or {}
        items.append({
            "id": str(rel),
            "round": next((int(p[6:]) for p in parts if re.fullmatch(r"round_\d+", p)), None),
            "system": parts[-3] if len(parts) >= 4 else None,
            "prompt": parts[-2] if len(parts) >= 2 else None,
            "seed": res.get("seed", ""),
            "question": harder.get("updated_question", ""),
            # the VC the judge graded against -- post inline-check rewrite, if any
            "verification_criterion": harder.get("verification_criterion", ""),
            "vc_rewritten_inline": bool(harder.get("verification_criterion_original")),
            "strategy": harder.get("chosen_strategy", ""),
            "attempt": last.get("attempt"),
            "judge_criterion_satisfied": judgment.get("criterion_satisfied"),
        })
    return items


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------


def _review(client, provider, model, system, user, schema, name):
    text, usage = llm_client.call_json_schema(
        client, provider, model=model, system=system, user=user,
        schema=schema, schema_name=name, max_tokens=MAX_TOKENS)
    cost, usage = llm_client.price_call(model, usage)
    return json.loads(text), cost, usage


# ---------------------------------------------------------------------------
# Phrase check (--phrase-check): reject questions built on terms the literature never uses
# ---------------------------------------------------------------------------
# The question reviewer reads only the question, so a fluent coinage ("temporal role
# discovery") or an unidentifiable name ("Gandzalu") looks fine to it -- every version of
# the prompt passed those. An exact-phrase count from Semantic Scholar is evidence it cannot
# produce by reasoning: on the annotation study the made-up terms had 0 papers while their
# real sub-phrases had 140-149.

PHRASE_EXTRACT_PROMPT = """Extract the key terms in the given question. We are trying to find uncommon terms and phrases. Each term will be searched as an exact phrase in Semantic Scholar, so extract the terms most likely to reveal wording the literature does not use.

Return at most 5 items. Prioritize, in this order:
1. compound_phrase: a named concept of 2-4 words that the question uses as if it were an established term in its field — the kind of phrase that could be a paper keyword, or the name of a method, model, measure, or construct. Extract the whole term, including any modifier that is part of its name (e.g. "hierarchical contrastive graph pooling", not "graph pooling"; "adaptive multi-site wastewater surveillance", not "wastewater surveillance"). Do not split a term into parts, and do not extract a part of it on its own.
   Do NOT treat these as compound phrases:
   - chains of hyphenated descriptive modifiers that describe this question's particular measure or setting rather than name a concept (e.g. "satellite-derived nighttime-light intensity measure", "parent-reported sleep-onset latency")
   - lists joined by commas, "and", or "or" (e.g. "dose, timing, and adherence variation")
   - anything longer than 4 words
2. uncommon_word: a single rare, technical, or unusual word, including possible misspellings or transliterations (e.g. "palynomorph", "eutectoid", "Hodgekin").
3. proper_noun: a named person, group, place, organization, program, dataset, model, or method (e.g. "Okavango Delta", "UK Biobank", "Framingham Heart Study").

Do NOT extract:
- common words, or general academic vocabulary ("methodology", "usefulness", "importance", "aspects")
- descriptive phrases that just state conditions of the question rather than naming a concept ("under typical clinical conditions", "across low- and high-income settings")
- author-year citations such as "Smith et al. (2019)"
- numbers, units, dates, or quantities

Copy each term from the question: keep its words, spelling, and word order. Drop possessives and use the singular form, but do not correct, expand, shorten, or replace the wording — the point is to test the wording the question actually uses.

If the question contains no such terms, return an empty list.

Return JSON only:

{
  "phrases": [
    {"phrase": "<term copied from the question>", "kind": "compound_phrase | uncommon_word | proper_noun"}
  ]
}
"""

PHRASE_SCHEMA = {
    "type": "object",
    "properties": {"phrases": {"type": "array", "items": {
        "type": "object",
        "properties": {"phrase": {"type": "string"},
                       "kind": {"type": "string",
                                "enum": ["compound_phrase", "uncommon_word", "proper_noun"]}},
        "required": ["phrase", "kind"], "additionalProperties": False}}},
    "required": ["phrases"],
    "additionalProperties": False,
}

S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_s2_cache: dict = {}
_s2_lock = threading.Lock()
_s2_last = [0.0]
S2_MIN_INTERVAL_S = 1.1            # the public and default-key rate limit is ~1 request/second


def s2_phrase_count(phrase: str):
    """Papers whose title or abstract contains `phrase` exactly, or None if the search
    failed. Throttled across threads and cached for the run."""
    import os
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    key = phrase.strip().lower()
    with _s2_lock:
        if key in _s2_cache:
            return _s2_cache[key]
    query = '"' + phrase.strip().replace('"', " ") + '"'
    url = S2_BULK_URL + "?" + urllib.parse.urlencode({"query": query, "fields": "title"})
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    total = None
    for attempt in range(6):
        with _s2_lock:                     # one request at a time, spaced out
            wait = _s2_last[0] + S2_MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _s2_last[0] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=30) as r:
                total = json.load(r).get("total")
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            break
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(2 * (attempt + 1))
    with _s2_lock:
        if total is not None:
            _s2_cache[key] = total
    return total


PHRASE_MAX_WORDS = 4


def _can_reject(term: str) -> bool:
    """Whether a zero count for `term` may reject the question. Long terms and terms with
    an acronym or a code (ML, RSN, FCPXML, A7-0201L) miss exact-phrase search for reasons
    unrelated to the question being bad -- an abbreviation or code a paper would spell
    out, or a long phrase no paper repeats verbatim -- so they are counted but never
    reject."""
    if len(term.split()) > PHRASE_MAX_WORDS:
        return False
    return not re.search(r"\b[A-Z]{2,}\b|\b[A-Z]+\d|\d", term)


def phrase_evidence(pc: dict) -> str:
    """The phrase check as evidence for the question reviewer (--phrase-mode evidence)."""
    lines = [f'- "{p["phrase"]}": ' + ("search failed" if p["hits"] is None else
                                      f'{p["hits"]} paper{"s" if p["hits"] != 1 else ""}')
             for p in pc["phrases"]]
    if not lines:
        return ""
    return ("\n\nLITERATURE CHECK -- exact-phrase matches in Semantic Scholar titles and "
            "abstracts for terms in this question:\n" + "\n".join(lines) + "\n"
            "A term with 0 papers may be coined, misspelled, or unidentifiable -- or simply "
            "unusual wording of a real concept. Treat the counts as signal that the wording might be overloaded, artificial or wrong, it is not a direct verdict.")


def phrase_check(item: dict, client, provider: str, model: str, min_hits: int) -> dict:
    """Extract the question's key terms and count each on Semantic Scholar.

    `zero` lists the terms below `min_hits` papers. A failed search counts as unknown,
    never as zero: a Semantic Scholar outage must not reject the harvest.
    """
    out, cost, usage = _review(client, provider, model, PHRASE_EXTRACT_PROMPT,
                               f"QUESTION:\n{item['question']}", PHRASE_SCHEMA,
                               "phrase_extract")
    phrases = []
    for p in out.get("phrases", [])[:5]:
        text = p["phrase"].strip()
        if not text or text.lower() in {x["phrase"].lower() for x in phrases}:
            continue
        # Search en/em dashes as hyphens, the more common spelling in titles/abstracts. (On
        # the annotation study this changed no counts -- the dashed zeros were rare wording.)
        searched = re.sub(r"\s*[–—]\s*", "-", text)
        phrases.append({"phrase": text, "searched_as": searched, "kind": p["kind"],
                        "hits": s2_phrase_count(searched),
                        "can_reject": _can_reject(text)})
    zero = [p["phrase"] for p in phrases
            if p["can_reject"] and p["hits"] is not None and p["hits"] < min_hits]
    return {"phrases": phrases, "zero": zero,
            "search_failed": [p["phrase"] for p in phrases if p["hits"] is None],
            "min_hits": min_hits, "cost_usd": cost, "usage": usage}


def skip_reason(item: dict, max_words: int):
    """Why `item` is rejected without review, or None. Cheap checks, no model call."""
    if item.get("judge_criterion_satisfied") is True:
        return "criterion_satisfied"
    if max_words and len(item["question"].split()) > max_words:
        return "too_long"
    return None


def skipped_row(item: dict, reason: str) -> dict:
    return {**item, "model": None, "cost_usd": 0.0, "skip_reason": reason,
            "pair_verdict": "REJECT", "decision": "reject"}


def review_one(item: dict, client, provider: str, model: str,
               phrase_min_hits: int = 0, phrase_mode: str = "reject") -> dict:
    """Both reviews for one question. Never raises: failures become decision 'error'.

    With phrase_min_hits > 0 (--phrase-check), the question's key terms are looked up on
    Semantic Scholar first. phrase_mode "reject": a term with fewer hits rejects the
    question before either review runs. "evidence": nothing is rejected here; the counts
    are appended to the question reviewer's input instead. "mixed": a single word (no
    spaces) with fewer hits rejects, as in "reject"; multi-word phrases never reject and
    go to the reviewer as evidence, as in "evidence".
    """
    row = {**item, "model": model, "cost_usd": 0.0}
    try:
        if phrase_min_hits > 0:
            pc = phrase_check(item, client, provider, model, phrase_min_hits)
            row["cost_usd"] += pc["cost_usd"]
            row["phrase_check"] = pc
            # which zero-hit terms may reject before review, by mode
            rejecting = pc["zero"] if phrase_mode == "reject" else \
                [t for t in pc["zero"] if len(t.split()) == 1] if phrase_mode == "mixed" else []
            if rejecting:
                pc["rejected_on"] = rejecting
                row.update(question_review=None, vc_review=None, vc_skipped=True,
                           phrase_rejected=True, pair_verdict="REJECT", decision="reject")
                return row
        # The prompt goes in `system` so it forms a stable prefix the provider can cache
        # across all ~600 calls; only the item varies in `user`.
        evidence = phrase_evidence(row["phrase_check"]) \
            if phrase_mode in ("evidence", "mixed") and row.get("phrase_check") else ""
        q, cost, usage = _review(client, provider, model, QUESTION_FILTER_PROMPT,
                                 f"QUESTION:\n{item['question']}{evidence}",
                                 QUESTION_SCHEMA, "question_review")
        row["cost_usd"] += cost
        row["question_review"] = {**q, "usage": usage}
        if q["verdict"] == "REJECT":
            # The pair can be no better than the question, so the outcome is already
            # REJECT and a VC review would only cost money.
            row.update(vc_review=None, vc_skipped=True, pair_verdict="REJECT",
                       decision="reject")
            return row

        # The VC reviewer gets no verdict or issues from the question review, only the
        # question it should judge the VC against: the proposed rewrite when there is
        # one (that is the question that would ship, and any rewritten VC must fit it),
        # else the question as written.
        rewritten_q = (q.get("rewritten_question") or "").strip() \
            if q["verdict"] == "MINOR_REVISION" else ""
        vc_question = rewritten_q or item["question"]
        row["vc_judged_against"] = "rewritten_question" if rewritten_q else "original_question"
        vc_user = (f"QUESTION:\n{vc_question}\n\n"
                   f"VERIFICATION CRITERION:\n{item['verification_criterion']}")
        v, cost, usage = _review(client, provider, model, VC_FILTER_PROMPT, vc_user,
                                 VC_SCHEMA, "vc_review")
        row["cost_usd"] += cost
        row["vc_review"] = {**v, "usage": usage}
    except Exception as e:                      # one bad item must not sink the batch
        row["decision"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        return row

    # The pair is as good as its weaker part: PASS only if both passed.
    pair = _worst(q["verdict"], v["vc_verdict"])
    row["pair_verdict"] = pair
    row["decision"] = {"PASS": "pass", "MINOR_REVISION": "revise", "REJECT": "reject"}[pair]
    if row["decision"] == "revise":
        row["proposed_question"] = rewritten_q or None
        row["proposed_vc"] = v.get("rewritten_vc") if v["vc_verdict"] == "MINOR_REVISION" else None
        # A MINOR_REVISION with no rewrite needs a human edit rather than a swap-in.
        row["missing_rewrite"] = bool(
            (q["verdict"] == "MINOR_REVISION" and not row["proposed_question"])
            or (v["vc_verdict"] == "MINOR_REVISION" and not row["proposed_vc"]))
        row["needs_rejudge"] = True
    return row


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def load_prior(path: Path) -> dict:
    """Latest row per id from an append-only results file; a success beats a later error."""
    prior = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue                 # a line cut off by a kill mid-write
            if r.get("decision") != "error" or prior.get(r["id"], {}).get("decision") in (None, "error"):
                prior[r["id"]] = r
    return prior


def summarize(rows: list) -> dict:
    ok = [r for r in rows if r.get("decision") != "error"]

    def by(key):
        out = {}
        for r in ok:
            out.setdefault(str(r.get(key)), Counter())[r["decision"]] += 1
        return {k: dict(v) for k, v in sorted(out.items())}

    return {
        "checked": len(rows),
        "errors": len(rows) - len(ok),
        "decision": dict(Counter(r["decision"] for r in rows)),
        "skipped": dict(Counter(r["skip_reason"] for r in ok if r.get("skip_reason"))),
        "vc_skipped_after_question_reject": sum(bool(r.get("vc_skipped")) and not r.get("phrase_rejected")
                                                for r in ok),
        "phrase_rejected": sum(bool(r.get("phrase_rejected")) for r in ok),
        "question_verdict": dict(Counter(r["question_review"]["verdict"]
                                         for r in ok if r.get("question_review"))),
        "vc_verdict": dict(Counter(r["vc_review"]["vc_verdict"]
                                   for r in ok if r.get("vc_review"))),
        "pair_verdict": dict(Counter(r["pair_verdict"] for r in ok)),
        "vc_judged_against_rewrite": sum(r.get("vc_judged_against") == "rewritten_question"
                                         for r in ok),
        "revise_missing_rewrite": sum(bool(r.get("missing_rewrite")) for r in ok),
        "by_system": by("system"),
        "by_prompt": by("prompt"),
        "by_round": by("round"),
        # the judge's own "criterion met but failed for other issues" flag, crossed with
        # the outcome: those rows are mislabelled harvest whatever this review says
        "judge_criterion_satisfied": by("judge_criterion_satisfied"),
        "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 4),
    }


def write_jsonl(path: Path, rows: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\nFinal decision")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path, help="A research_loop.py --out-dir (or any run dir).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write results (default: <run_dir>/final_verification).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Reviewer model (default: {DEFAULT_MODEL}).")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--shuffle", action="store_true",
                   help="Review questions in random order instead of file order, so a "
                        "--limit pilot samples every round, system, and prompt.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --shuffle (default: 0).")
    p.add_argument("--limit", type=int, default=0,
                   help="Review at most N not-yet-reviewed questions (0 = all). For pilots.")
    p.add_argument("--budget-usd", type=float, default=0.0,
                   help="Stop submitting once this much has been spent (0 = no limit); "
                        "in-flight reviews still finish.")
    p.add_argument("--max-question-words", type=int, default=100,
                   help="Reject questions longer than this without review (default: 100; "
                        "0 = no limit).")
    p.add_argument("--phrase-check", action="store_true",
                   help="Before the reviews, extract the question's proper nouns and technical "
                        "terms (one LLM call) and look each up on Semantic Scholar as an exact "
                        "phrase; reject the question if any has fewer than --phrase-min-hits "
                        "papers. Uses S2_API_KEY if set (otherwise slower, rate-limited).")
    p.add_argument("--phrase-min-hits", type=int, default=1,
                   help="With --phrase-check: minimum papers a term needs (default: 1, i.e. "
                        "reject only on zero hits).")
    p.add_argument("--phrase-mode", choices=["reject", "evidence", "mixed"], default="reject",
                   help="With --phrase-check: 'reject' (default) rejects on a zero-hit term "
                        "before review; 'evidence' never rejects on its own but appends the "
                        "counts to the question reviewer's input; 'mixed' rejects on a "
                        "zero-hit single word and passes multi-word phrases as evidence.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be reviewed and exit without calling the model.")
    llm_client.add_provider_arg(p)
    args = p.parse_args()
    llm_client.configure_from_args(args)

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir / "final_verification"
    items = collect_harvest(run_dir)
    if not items:
        p.error(f"no FAILED_FOUND questions under {run_dir}")

    results_path = out_dir / "results.jsonl"
    # Skip rules are re-derived every run rather than read back, so changing
    # --max-question-words takes effect on a resume; a skip also wins over an old review.
    skipped = {it["id"]: skipped_row(it, why) for it in items
               if (why := skip_reason(it, args.max_question_words))}
    prior = {k: r for k, r in load_prior(results_path).items()
             if not r.get("skip_reason") and k not in skipped}
    done = {k: r for k, r in prior.items() if r.get("decision") != "error"}
    todo = [it for it in items if it["id"] not in done and it["id"] not in skipped]
    if args.shuffle:
        # A fixed seed gives one fixed ordering, so successive --limit runs keep drawing
        # fresh questions from the same random sequence instead of overlapping.
        random.Random(args.seed).shuffle(todo)
    if args.limit:
        todo = todo[:args.limit]
    provider = llm_client.resolve_provider(args.model)
    why = Counter(r["skip_reason"] for r in skipped.values())
    print(f"Reviewer: {args.model} ({provider})  |  harvest {len(items)}, rejected without "
          f"review {len(skipped)} {dict(why)}, already reviewed {len(done)}, "
          f"to review {len(todo)} -> {out_dir}/")
    if args.dry_run:
        return
    if todo and (missing := llm_client.require_api_key(args.model)):
        p.error(missing)

    out_dir.mkdir(parents=True, exist_ok=True)
    client = llm_client.make_client(args.model) if todo else None
    lock = threading.Lock()
    # prior errors stay in the results until retried, even if --limit skips them this run
    rows = {**prior, **skipped}
    spent = 0.0
    stopped = False

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool, \
            results_path.open("a") as log:
        min_hits = args.phrase_min_hits if args.phrase_check else 0
        futures = {pool.submit(review_one, it, client, provider, args.model, min_hits,
                               args.phrase_mode): it
                   for it in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            with lock:
                # append-only while running, so a crash keeps every review already paid for
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                rows[row["id"]] = row
                spent += row.get("cost_usd", 0.0)
            q = (row.get("question_review") or {}).get("verdict", "-")
            vc = (row.get("vc_review") or {}).get("vc_verdict", "-")
            print(f"[{i:>3}/{len(todo)}] {row['decision']:<6} Q={q:<14} VC={vc:<14} "
                  f"${spent:.2f}  {row['question'][:60]}", flush=True)
            if args.budget_usd and spent >= args.budget_usd and not stopped:
                stopped = True
                cancelled = sum(1 for f in futures if f.cancel())
                print(f"\n[budget] ${spent:.2f} spent, limit ${args.budget_usd:.2f} — "
                      f"{cancelled} review(s) cancelled", file=sys.stderr, flush=True)

    # Rewrite the log as one row per question (latest wins), in harvest order.
    order = {it["id"]: n for n, it in enumerate(items)}
    final = sorted(rows.values(), key=lambda r: order.get(r["id"], 1 << 30))
    write_jsonl(results_path, final)
    write_jsonl(out_dir / "pass.jsonl", [
        {k: r[k] for k in ("id", "round", "system", "prompt", "seed", "question",
                           "verification_criterion", "strategy")}
        for r in final if r["decision"] == "pass"])
    write_jsonl(out_dir / "revise.jsonl", [r for r in final if r["decision"] == "revise"])
    summary = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "run_dir": str(run_dir), "model": args.model,
               "reasoning_effort": args.reasoning_effort, "harvest_size": len(items),
               "phrase_check": args.phrase_check,
               "phrase_min_hits": args.phrase_min_hits if args.phrase_check else None,
               "phrase_mode": args.phrase_mode if args.phrase_check else None,
               "stopped_early": stopped, **summarize(final)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    s = summary
    print(f"\n{'#' * 70}\nFINAL VERIFICATION SUMMARY\n{'#' * 70}")
    print(f"reviewed {s['checked']}/{len(items)}  errors {s['errors']}  ${s['cost_usd']:.2f}")
    for key in ("decision", "question_verdict", "vc_verdict", "pair_verdict"):
        print(f"  {key:<18} {s[key]}")
    print(f"\npass set:   {out_dir / 'pass.jsonl'}\nrevise set: {out_dir / 'revise.jsonl'}"
          f"\nall rows:   {results_path}\nsummary:    {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
