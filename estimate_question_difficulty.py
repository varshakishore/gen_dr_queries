"""
Question difficulty scorer.

Pipeline:
  1. Retrieve Semantic Scholar papers via snippet search (S2SnippetRetriever)
  2. Generate evaluation rubrics via GPT-5 (RubricGenerator)
  3. Score question difficulty 1-10 (DifficultyScorer) with pluggable inputs

Adding/removing inputs to the scorer:
  scorer.register_input("my_input", my_provider_fn)   # add
  scorer.unregister_input("my_input")                  # remove

Each provider is a callable: (question: str, context: dict) -> dict
Its returned dict is merged into the scoring context passed to the LLM.
"""

import os
import re
import json
import logging
import requests
from typing import Optional, Callable
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table (same pattern as judge.py)
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
# 1. Semantic Scholar snippet retrieval
# ---------------------------------------------------------------------------

class S2SnippetRetriever:
    """
    Retrieves papers from Semantic Scholar using the snippet/relevance search endpoint.
    Docs: https://api.semanticscholar.org/graph/v1
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: Optional[str] = None, top_k: int = 5):
        self.api_key = api_key or os.getenv("S2_API_KEY")
        self.top_k = top_k

    def _headers(self) -> dict:
        h = {"x-api-key": self.api_key} if self.api_key else {}
        return h

    def retrieve(self, query: str) -> list[dict]:
        """
        Search S2 for `query` and return up to `top_k` papers.

        Each result dict has keys: paperId, title, year, authors, url.
        """
        params = {
            "query": query,
            "limit": self.top_k,
            "fields": "paperId,title,year,authors,url",
        }
        try:
            resp = requests.get(
                self.BASE_URL,
                params=params,
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            papers = []
            for item in data.get("data", []):
                authors = [a["name"] for a in item.get("authors", [])]
                papers.append({
                    "paperId": item.get("paperId", ""),
                    "title": item.get("title", ""),
                    "year": item.get("year"),
                    "authors": authors,
                    "url": item.get("url", ""),
                })
            logger.info("S2 retrieved %d papers for query: %r", len(papers), query[:60])
            return papers
        except Exception as e:
            logger.warning("S2 retrieval failed: %s", e)
            return []


# ---------------------------------------------------------------------------
# 2. Rubric generator
# ---------------------------------------------------------------------------

_RUBRIC_SYSTEM = """\
You are an expert question analyst. Given a question, produce a JSON rubric for \
evaluating how hard it is to answer. The rubric should list 4-6 criteria, each \
with a name, a one-sentence description, and a weight (0-1, weights must sum to 1).

Respond ONLY with valid JSON of the form:
{
  "criteria": [
    {"name": "...", "description": "...", "weight": 0.2},
    ...
  ]
}"""


class RubricGenerator:
    """Uses GPT-5 to produce a difficulty rubric for a question."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5-mini",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, question: str) -> dict:
        """
        Returns a rubric dict: {"criteria": [...], "usage": {...}, "cost_usd": float}
        Falls back to a simple default rubric on failure.
        """
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _RUBRIC_SYSTEM},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                response_format={"type": "json_object"},
            )
            usage = getattr(resp, "usage", None)
            usage_dict = None
            cost = 0.0
            if usage:
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                usage_dict = {"prompt_tokens": pt, "completion_tokens": ct}
                cost = _cost_usd(self.model_name, pt, ct)
            text = (resp.choices[0].message.content or "").strip()
            rubric = json.loads(text)
            rubric["usage"] = usage_dict
            rubric["cost_usd"] = cost
            return rubric
        except Exception as e:
            logger.warning("RubricGenerator failed: %s. Using default rubric.", e)
            return self._default_rubric()

    @staticmethod
    def _default_rubric() -> dict:
        return {
            "criteria": [
                {"name": "domain_depth", "description": "Requires deep domain knowledge.", "weight": 0.3},
                {"name": "multi_hop", "description": "Requires reasoning across multiple facts.", "weight": 0.3},
                {"name": "ambiguity", "description": "Question contains ambiguous or nuanced phrasing.", "weight": 0.2},
                {"name": "evidence_scarcity", "description": "Supporting evidence is rare or hard to find.", "weight": 0.2},
            ],
            "usage": None,
            "cost_usd": 0.0,
        }


# ---------------------------------------------------------------------------
# 3. Difficulty scorer with pluggable inputs
# ---------------------------------------------------------------------------

_SCORE_SYSTEM = """\
You are an expert question difficulty evaluator for research and academic settings.

Score the question on a 1–10 scale using these criteria:
  1–2  Factual recall, single concept, answer in one sentence (e.g. "What is DNA?")
  3–4  Requires basic domain knowledge or simple comparison (e.g. "What are LLMs?")
  5–6  Multi-concept synthesis, some domain depth required
  7–8  Cross-domain reasoning, contested/incomplete evidence, or rare specialist knowledge
  9–10 Open research frontier, requires integrating findings across multiple sub-fields,
       answer is actively debated or unknown

Scoring signals to consider:
- Specificity: vague questions are easier; highly specific mechanistic questions are harder
- Breadth: does answering require knowledge from multiple fields?
- Evidence availability: is the answer well-established or at the research frontier?
- Reasoning depth: single-hop lookup vs. multi-step causal/mechanistic reasoning
- Ambiguity: is there a clear correct answer, or is it contested?

Anchor examples (do NOT copy these scores blindly, calibrate relative to the question):
  "What is the capital of France?" → 1
  "What does RNA stand for?" → 2
  "How does attention work in transformers?" → 5
  "What are the trade-offs between RLHF and DPO for aligning language models?" → 7
  "What molecular mechanisms link mitochondrial dysfunction to tau hyperphosphorylation in Alzheimer's disease?" → 9

Respond with ONLY valid JSON (no markdown, no explanation outside the JSON):
{"score": <integer 1-10>, "reasoning": "<one concise sentence explaining the score>"}"""


def _build_score_prompt(question: str, context: dict) -> str:
    lines = [f"Question: {question}\n"]
    rubric = context.get("rubric")
    if rubric and rubric.get("criteria"):
        lines.append("Rubric criteria:")
        for c in rubric["criteria"]:
            lines.append(f"  - {c['name']} (weight {c['weight']}): {c['description']}")
        lines.append("")
    papers = context.get("retrieved_papers")
    if papers:
        lines.append(f"Retrieved papers ({len(papers)}):")
        for i, p in enumerate(papers[:5], 1):
            authors = ", ".join(p.get("authors", [])[:3])
            lines.append(f"  {i}. [{p.get('year','?')}] {p.get('title','')} ({authors})")
        lines.append("")
    reranker = context.get("reranker_scores")
    if reranker is not None:
        lines.append(f"Reranker top score: {reranker:.4f}\n")
    # Any extra inputs registered by user
    for key, val in context.items():
        if key not in ("rubric", "retrieved_papers", "reranker_scores"):
            lines.append(f"{key}: {val}\n")
    return "\n".join(lines)


class DifficultyScorer:
    """
    Scores a question's difficulty on a 1-10 scale.

    Built-in inputs: rubric, retrieved_papers, reranker_scores.
    Add custom inputs via register_input(); remove via unregister_input().

    Each input provider is a callable:
        provider(question: str, context: dict) -> dict
    The returned dict is merged into the context dict before scoring.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5-mini",
        base_url: Optional[str] = None,
        retriever: Optional[S2SnippetRetriever] = None,
        rubric_generator: Optional[RubricGenerator] = None,
        use_retriever: bool = True,
        use_rubric: bool = True,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        self._client = None

        self.retriever = retriever or S2SnippetRetriever()
        self.rubric_gen = rubric_generator or RubricGenerator(
            api_key=api_key, model_name=model_name, base_url=base_url
        )

        # Registry of pluggable input providers: name -> callable
        self._input_providers: dict[str, Callable] = {}
        if use_retriever:
            self.register_input("retrieved_papers", self._provide_papers)
        if use_rubric:
            self.register_input("rubric", self._provide_rubric)

    # ------------------------------------------------------------------
    # Input provider registry
    # ------------------------------------------------------------------

    def register_input(self, name: str, provider: Callable) -> None:
        """
        Register a named input provider.

        provider(question, context) -> dict
        The dict is merged into context before the LLM scoring call.
        """
        self._input_providers[name] = provider
        logger.info("Registered input provider: %r", name)

    def unregister_input(self, name: str) -> None:
        """Remove a previously registered input provider."""
        if name in self._input_providers:
            del self._input_providers[name]
            logger.info("Unregistered input provider: %r", name)

    def list_inputs(self) -> list[str]:
        return list(self._input_providers.keys())

    # ------------------------------------------------------------------
    # Built-in providers
    # ------------------------------------------------------------------

    def _provide_papers(self, question: str, context: dict) -> dict:
        papers = self.retriever.retrieve(question)
        return {"retrieved_papers": papers}

    def _provide_rubric(self, question: str, context: dict) -> dict:
        rubric = self.rubric_gen.generate(question)
        return {"rubric": rubric}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {}
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
            extra_context: Optional pre-computed context to seed the pipeline
                           (e.g. {"reranker_scores": 0.87}). Providers listed
                           in the registry are still called unless you pass their
                           key here to short-circuit them.

        Returns:
            {
              "score": int (1-10),
              "reasoning": str,
              "context": dict,    # full context fed to LLM
              "usage": dict | None,
              "cost_usd": float,
            }
        """
        context: dict = dict(extra_context or {})

        # Run each registered provider in order (skip if key already in context)
        for name, provider in self._input_providers.items():
            key = name  # provider is expected to return {name: value} or any dict
            try:
                result = provider(question, context)
                context.update(result)
            except Exception as e:
                logger.warning("Input provider %r failed: %s", name, e)

        client = self._get_client()
        prompt = _build_score_prompt(question, context)
        try:
            # print(_SCORE_SYSTEM+"\n"+prompt)
            resp = client.responses.create(
                model=self.model_name,
                input=_SCORE_SYSTEM+"\n"+prompt
            )
            usage = getattr(resp, "usage", None)
            usage_dict = None
            cost = 0.0
            if usage:
                pt = getattr(usage, "input_tokens", 0) or 0
                ct = getattr(usage, "output_tokens", 0) or 0
                usage_dict = {"input_tokens": pt, "output_tokens": ct}
                cost = _cost_usd(self.model_name, pt, ct)
            text = (resp.output_text).strip()
            # Try JSON first, fall back to regex extraction of the score number
            try:
                parsed = json.loads(text)
                raw_score = parsed.get("score", -1)
                reasoning = parsed.get("reasoning", "")
            except json.JSONDecodeError:
                import pdb; pdb.set_trace()
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
# Quick CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Score question difficulty 1-10.")
    parser.add_argument("question", nargs="?", help="Question to evaluate (omit when using --batch)")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-retriever", action="store_true")
    parser.add_argument("--no-rubric", action="store_true")
    parser.add_argument("--reranker-score", type=float, default=None)
    parser.add_argument("--batch", metavar="FILE",
                        help="JSON file with [{seed_question, updated_questions: [{updated_question, ...}]}]")
    parser.add_argument("--output", metavar="FILE", default=None,
                        help="Write batch results to this JSON file (default: print only)")
    args = parser.parse_args()

    scorer = DifficultyScorer(
        model_name=args.model,
        retriever=S2SnippetRetriever(top_k=args.top_k),
        use_retriever=not args.no_retriever,
        use_rubric=not args.no_rubric,
    )

    extra = {}
    if args.reranker_score is not None:
        extra["reranker_scores"] = args.reranker_score

    if args.batch:
        with open(args.batch) as f:
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
            # Print as we go
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

    elif args.question:
        result = scorer.score(args.question, extra_context=extra or None)
        print(f"\nScore: {result['score']}/10")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Cost: ${result['cost_usd']:.6f} USD")
        print(f"Active inputs: {scorer.list_inputs()}")
    else:
        parser.error("Provide a question or --batch FILE")
