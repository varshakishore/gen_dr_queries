"""
Question difficulty scorer.

Pipeline:
  1. Run registered input providers to build a context dict
  2. Score question difficulty 1-10 via LLM using that context

All signal providers live in difficulty_signals.py.
Set up providers before scoring:

  from difficulty_signals import register_default_signals, register_all_signals

  scorer = DifficultyScorer()
  register_default_signals(scorer)          # papers + rubric (standard setup)
  register_all_signals(scorer)              # papers + rubric + all extended signals

Add/remove individual providers:
  scorer.register_input("my_input", my_provider_fn)   # add
  scorer.unregister_input("my_input")                  # remove

Each provider: (question: str, context: dict) -> dict
Its returned dict is merged into the scoring context passed to the LLM.
"""

import os
import re
import json
import logging
from typing import Optional, Callable
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table
# ---------------------------------------------------------------------------
_DEFAULT_COST_PER_1M = (2.50, 10.0)
_MODEL_COST_PER_1M = {
    "gpt-4.1": (2, 8),
    "gpt-5-mini": (0.25, 2),
    "gpt-5-nano": (0.05, 0.4),
}


def _cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    in_1m, out_1m = _MODEL_COST_PER_1M.get(model_name, _DEFAULT_COST_PER_1M)
    return (input_tokens / 1e6 * in_1m) + (output_tokens / 1e6 * out_1m)


# ---------------------------------------------------------------------------
# Scoring prompt
# ---------------------------------------------------------------------------

_SCORE_SYSTEM = """\
You are a difficulty scorer for research questions that require long-form reports with citations.
Score 1 (trivial) to 10 (at the research frontier). Base your score entirely on the signals provided.

Signal interpretation guide:

LITERATURE signals
  s2_result_count         low (<20) → sparse literature → harder
  s2_field_count          high (>3) → cross-disciplinary synthesis required → harder
  s2_evidence_heterogeneity  >2 distinct field buckets → methodologically diverse sources → harder
  s2_pct_last_5yr         high (>0.7) → rapidly evolving, evidence unsettled → harder
  s2_year_spread          wide AND recent skew → paradigm still shifting → harder
  s2_meta_analysis_present  True → primary studies disagree, contested evidence → harder
  s2_avg_citation_count   low → niche/emerging area → harder; high → well-established → easier

REASONING signals
  causal_chain_depth      0–1 = direct lookup; 3–5 = deep multi-hop causal chain → harder
  inference_type          deductive = easier; abductive or mixed = harder
  sub_question_count      each additional sub-question adds synthesis burden → harder
  scope_anchor_count      0 = unconstrained question, reporter must define scope → harder

CONCEPT signals
  concept_embedding_avg_distance   >0.5 → concepts span different semantic domains → harder
  concept_rarity_ratio             <0.05 → concepts rarely studied together → novel intersection → harder

RUBRIC criteria weights reflect domain-specific difficulty dimensions — higher-weighted criteria
that are structurally hard (multi_hop, evidence_scarcity) push the score up.

Scoring process:
1. Read each signal value and note whether it pushes difficulty up or down.
2. Weight signals that are present more heavily than absent ones.
3. Reason about how difficult this question is and write` a rationale .
4. Produce a single integer score that best reflects the aggregate signal evidence.


Respond with ONLY valid JSON:
{"reasoning": "<rationale for score>", "score": <integer 1-10>}"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_S2_META_KEYS = frozenset({
    "s2_result_count", "s2_max_year", "s2_avg_year", "s2_pct_last_5yr",
    "s2_year_spread", "s2_fields", "s2_field_count", "s2_evidence_heterogeneity",
    "s2_meta_analysis_present", "s2_avg_citation_count", "s2_min_citation_count",
    "s2_max_citation_count",
})
_LLM_ANALYSIS_KEYS = frozenset({
    "causal_chain_depth", "inference_type", "sub_question_count",
    "scope_anchors", "scope_anchor_count",
})
_CONCEPT_KEYS = frozenset({
    "key_concepts", "concept_embedding_avg_distance", "concept_embedding_max_distance",
    "concept_individual_counts", "concept_combined_count", "concept_rarity_ratio",
})
_KNOWN_KEYS = frozenset({"rubric", "retrieved_papers", "reranker_scores"}) \
    | _S2_META_KEYS | _LLM_ANALYSIS_KEYS | _CONCEPT_KEYS


def _build_score_prompt(question: str, context: dict) -> str:
    lines = [f"Question: {question}\n"]

    # Rubric
    rubric = context.get("rubric")
    if rubric and rubric.get("criteria"):
        lines.append("Rubric criteria:")
        for c in rubric["criteria"]:
            lines.append(f"  - {c['name']} (weight {c['weight']}): {c['description']}")
        lines.append("")

    # Retrieved papers
    papers = context.get("retrieved_papers")
    if papers:
        lines.append(f"Retrieved papers ({len(papers)}):")
        for i, p in enumerate(papers[:5], 1):
            authors = ", ".join(p.get("authors", [])[:3])
            fields = ", ".join((p.get("fieldsOfStudy") or [])[:3])
            suffix = f" [{fields}]" if fields else ""
            lines.append(f"  {i}. [{p.get('year', '?')}] {p.get('title', '')} ({authors}){suffix}")
        lines.append("")

    # Reranker
    reranker = context.get("reranker_scores")
    if reranker is not None:
        lines.append(f"Reranker top score: {reranker:.4f}\n")

    # S2 metadata signals
    if any(k in context for k in _S2_META_KEYS):
        lines.append("Literature signals:")
        lines.append(f"  result_count: {context.get('s2_result_count', '?')}")
        if context.get("s2_avg_year") is not None:
            pct = context.get("s2_pct_last_5yr")
            pct_str = f"{pct:.0%}" if pct is not None else "?"
            lines.append(
                f"  publication_years: avg={context['s2_avg_year']}, "
                f"max={context.get('s2_max_year')}, "
                f"spread={context.get('s2_year_spread')}, "
                f"pct_last_5yr={pct_str}"
            )
        if context.get("s2_fields"):
            lines.append(
                f"  fields_of_study ({context.get('s2_field_count')}): "
                f"{', '.join(context['s2_fields'][:8])}"
            )
            lines.append(
                f"  evidence_heterogeneity_buckets: {context.get('s2_evidence_heterogeneity')}"
            )
        lines.append(f"  meta_analysis_present: {context.get('s2_meta_analysis_present')}")
        if context.get("s2_avg_citation_count") is not None:
            lines.append(
                f"  citation_counts: avg={context['s2_avg_citation_count']:.0f}, "
                f"min={context.get('s2_min_citation_count')}, "
                f"max={context.get('s2_max_citation_count')}"
            )
        lines.append("")

    # LLM analysis signals
    if any(k in context for k in _LLM_ANALYSIS_KEYS):
        lines.append("Reasoning signals:")
        if "causal_chain_depth" in context:
            lines.append(f"  causal_chain_depth: {context['causal_chain_depth']}/5")
        if "inference_type" in context:
            lines.append(f"  inference_type: {context['inference_type']}")
        if "sub_question_count" in context:
            lines.append(f"  sub_question_count: {context['sub_question_count']}")
        anchors = context.get("scope_anchors") or []
        anchor_count = context.get("scope_anchor_count", len(anchors))
        anchor_str = ", ".join(anchors) if anchors else "none — broad/unconstrained"
        lines.append(f"  scope_anchors ({anchor_count}): {anchor_str}")
        lines.append("")

    # Concept signals
    if any(k in context for k in _CONCEPT_KEYS):
        lines.append("Concept signals:")
        concepts = context.get("key_concepts") or []
        if concepts:
            lines.append(f"  key_concepts: {', '.join(concepts)}")
        avg_d = context.get("concept_embedding_avg_distance")
        max_d = context.get("concept_embedding_max_distance")
        if avg_d is not None:
            lines.append(
                f"  embedding_distance: avg={avg_d}, max={max_d} "
                "(0=same domain, 2=maximally cross-domain)"
            )
        rarity = context.get("concept_rarity_ratio")
        if rarity is not None:
            lines.append(
                f"  concept_rarity_ratio: {rarity} "
                "(low=rarely studied together, high=well-established intersection)"
            )
        lines.append("")

    # Any remaining custom inputs not handled above
    for key, val in context.items():
        if key not in _KNOWN_KEYS:
            lines.append(f"{key}: {val}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class DifficultyScorer:
    """
    Scores a question's difficulty on a 1-10 scale.

    Providers are registered externally via register_input().
    Use difficulty_signals.register_default_signals() or register_all_signals()
    to set up the standard provider stack.

    Each provider: (question: str, context: dict) -> dict
    The returned dict is merged into the context before scoring.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5-mini",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        self._client: Optional[OpenAI] = None
        self._input_providers: dict[str, Callable] = {}

    def register_input(self, name: str, provider: Callable) -> None:
        """Register a named input provider. Providers run in registration order."""
        self._input_providers[name] = provider
        logger.info("Registered input provider: %r", name)

    def unregister_input(self, name: str) -> None:
        if name in self._input_providers:
            del self._input_providers[name]
            logger.info("Unregistered input provider: %r", name)

    def list_inputs(self) -> list[str]:
        return list(self._input_providers.keys())

    def _get_client(self) -> OpenAI:
        if self._client is None:
            kwargs: dict = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def score(self, question: str, extra_context: Optional[dict] = None) -> dict:
        """
        Score question difficulty.

        Args:
            question: The question to evaluate.
            extra_context: Optional pre-computed context values. Providers whose
                           key is already present are still called (they may add
                           other keys); pass a key to short-circuit a provider
                           by having it check context before doing work.

        Returns:
            {"score": int, "reasoning": str, "context": dict,
             "usage": dict | None, "cost_usd": float}
        """
        context: dict = dict(extra_context or {})

        for name, provider in self._input_providers.items():
            try:
                result = provider(question, context)
                context.update(result)
            except Exception as e:
                logger.warning("Input provider %r failed: %s", name, e)
        client = self._get_client()
        prompt = _build_score_prompt(question, context)
        try:
            resp = client.responses.create(
                model=self.model_name,
                input=_SCORE_SYSTEM + "\n" + prompt,
            )
            usage = getattr(resp, "usage", None)
            usage_dict = None
            cost = 0.0
            if usage:
                pt = getattr(usage, "input_tokens", 0) or 0
                ct = getattr(usage, "output_tokens", 0) or 0
                usage_dict = {"input_tokens": pt, "output_tokens": ct}
                cost = _cost_usd(self.model_name, pt, ct)
            text = resp.output_text.strip()
            try:
                parsed = json.loads(text)
                raw_score = parsed.get("score", -1)
                reasoning = parsed.get("reasoning", "")
            except json.JSONDecodeError:
                m = re.search(r"\b([1-9]|10)\b", text)
                raw_score = int(m.group(1)) if m else 5
                reasoning = text
            score = max(1, min(10, int(raw_score)))
            logger.info("Difficulty score=%d for question: %r", score, question[:60])
            logger.warning("dcost=%.6f", cost)
            return {
                "score": score,
                "reasoning": reasoning,
                "context": context,
                "usage": usage_dict,
                "cost_usd": cost,
            }
        except Exception as e:
            logger.warning("DifficultyScorer LLM call failed: %s. Returning default.", e)
            return {"score": 5, "reasoning": "fallback", "context": context, "usage": None, "cost_usd": 0.0}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from difficulty_signals import register_default_signals, register_all_signals

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Score question difficulty 1-10.")
    parser.add_argument("question", nargs="?", help="Question to evaluate (omit when using --batch)")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-retriever", action="store_true")
    parser.add_argument("--no-rubric", action="store_true")
    parser.add_argument("--reranker-score", type=float, default=None)
    parser.add_argument("--input_file", metavar="FILE",
                        help="JSON file with [{seed_question, updated_questions: [{updated_question, ...}]}]")
    parser.add_argument("--output", metavar="FILE", default=None,
                        help="Write batch results to this JSON file (default: print only)")
    parser.add_argument("--extended-signals", action="store_true",
                        help="Enable all extended signals (S2 metadata, LLM analysis, "
                             "concept embeddings, co-occurrence)")
    args = parser.parse_args()

    scorer = DifficultyScorer(model_name=args.model)

    if args.extended_signals:
        register_all_signals(
            scorer,
            model_name=args.model,
            top_k=args.top_k,
            use_retriever=not args.no_retriever,
            use_rubric=not args.no_rubric,
        )
    else:
        register_default_signals(
            scorer,
            model_name=args.model,
            top_k=args.top_k,
            use_retriever=not args.no_retriever,
            use_rubric=not args.no_rubric,
        )

    extra = {}
    if args.reranker_score is not None:
        extra["reranker_scores"] = args.reranker_score

    if args.input_file and args.input_file.endswith(".json"):
        with open(args.input_file) as f:
            items = json.load(f)

        results = []
        total_cost = 0.0
        for item in items:
            seed = item["seed_question"]
            seed_result = scorer.score(seed, extra_context=extra or None)
            total_cost += seed_result["cost_usd"]
            entry = {
                "seed_question": seed,
                "seed_score": seed_result["score"],
                "seed_reasoning": seed_result["reasoning"],
                "avg_updated_score": None,
                "updated_questions": [],
            }
            for uq in item.get("updated_questions", []):
                q = uq["updated_question"]
                r = scorer.score(q, extra_context=extra or None)
                total_cost += r["cost_usd"]
                entry["updated_questions"].append({
                    "updated_question": q,
                    "score": r["score"],
                    "reasoning": r["reasoning"],
                })
            results.append(entry)
            uq_scores = [uq["score"] for uq in entry["updated_questions"]]
            avg = sum(uq_scores) / len(uq_scores) if uq_scores else 0.0
            entry["avg_updated_score"] = round(avg, 2)
            print(f"\n[{seed_result['score']}/10] SEED: {seed}")
            for uq in entry["updated_questions"]:
                print(f"  [{uq['score']}/10] {uq['updated_question']}")
            print(f"  avg updated: {avg:.1f}/10")

        print(f"\nTotal cost: ${total_cost:.6f} USD")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results written to {args.output}")

    elif args.input_file and args.input_file.endswith(".jsonl"):
        total_cost = 0.0
        results = []
        with open(args.input_file) as f:
            for line in f:
                item = json.loads(line)
                easy_question = item["easy"]
                hard_question = item["hard"]
                result_easy = scorer.score(easy_question, extra_context=extra or None)
                total_cost += result_easy["cost_usd"]
                result_hard = scorer.score(hard_question, extra_context=extra or None)
                total_cost += result_hard["cost_usd"]
                print(f"\n[{result_easy['score']}/10] {easy_question}")
                print(f"[{result_hard['score']}/10] {hard_question}")
                results.append({
                    "easy_question": easy_question,
                    "easy_score": result_easy["score"],
                    "easy_reasoning": result_easy["reasoning"],
                    "hard_question": hard_question,
                    "hard_score": result_hard["score"],
                    "hard_reasoning": result_hard["reasoning"],
                })
                if len(results) == 4:
                    break
        
        print(f"\nTotal cost: ${total_cost:.6f} USD")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results written to {args.output}")

    elif args.question:
        result = scorer.score(args.question, extra_context=extra or None)
        print(f"\nScore: {result['score']}/10")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Cost: ${result['cost_usd']:.6f} USD")
        print(f"Active inputs: {scorer.list_inputs()}")
    else:
        parser.error("Provide a question or --batch FILE")
