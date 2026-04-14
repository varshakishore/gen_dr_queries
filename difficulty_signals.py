"""
All difficulty signal providers for DifficultyScorer.

Every provider follows the (question: str, context: dict) -> dict interface.
The returned dict is merged into the scoring context.

Providers:
  PaperRetrievalSignals      – fetches papers from Semantic Scholar
  RubricSignals              – generates a difficulty rubric via LLM
  S2MetadataSignals          – derives signals from already-retrieved papers
  LLMAnalysisSignals         – causal chain depth, inference type, sub-question count, scope anchoring
  ConceptEmbeddingSignals    – cross-field concept embedding distance
  ConceptCooccurrenceSignals – S2 co-occurrence rarity across key concepts

Setup helpers:
  register_default_signals(scorer, ...)   – papers + rubric (standard setup)
  register_all_signals(scorer, ...)       – default + all extended signals
"""

import os
import json
import math
import logging
import datetime
import requests
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared cost helper
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


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


# ---------------------------------------------------------------------------
# S2 retrieval
# ---------------------------------------------------------------------------

class S2SnippetRetriever:
    """
    Retrieves papers from Semantic Scholar using a two-step approach:
      1. snippet/search  – semantic relevance search, returns paperId + basic fields
      2. paper/batch     – enriches results with fieldsOfStudy and citationCount
    """

    SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
    BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

    def __init__(self, api_key: Optional[str] = None, top_k: int = 10):
        self.api_key = api_key or os.getenv("S2_API_KEY")
        self.top_k = top_k

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def retrieve(self, query: str) -> list[dict]:
        # Step 1: snippet search — no fields param, returns corpusId, title, authors per paper
        try:
            resp = requests.get(
                self.SNIPPET_URL,
                params={"query": query, "limit": self.top_k},
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            # Preserve snippet-search ranking order via an ordered dict
            papers: dict[str, dict] = {}
            for item in resp.json().get("data", []):
                if (item.get("score") or 0) < 0.5:
                    continue
                p = item.get("paper", {})
                corpus_id = p.get("corpusId")
                if corpus_id is None:
                    continue
                papers[str(corpus_id)] = {
                    "paperId": "",
                    "title": p.get("title", ""),
                    "year": None,
                    "authors": p.get("authors", []),
                    "url": "",
                    "fieldsOfStudy": [],
                    "citationCount": None,
                }
        except Exception as e:
            logger.warning("S2 snippet search failed: %s", e)
            return []

        if not papers:
            return []
        
        # Step 2: batch fetch everything snippet/search doesn't provide
        try:
            batch_resp = requests.post(
                self.BATCH_URL,
                params={"fields": "corpusId,paperId,year,url,fieldsOfStudy,citationCount"},
                json={"ids": [f"CorpusId:{cid}" for cid in papers]},
                headers=self._headers(),
                timeout=30,
            )
            batch_resp.raise_for_status()
            for item in batch_resp.json():
                if not item:
                    continue
                cid = str(item.get("corpusId", ""))
                if cid in papers:
                    papers[cid]["paperId"] = item.get("paperId", "")
                    papers[cid]["year"] = item.get("year")
                    papers[cid]["url"] = item.get("url", "")
                    papers[cid]["fieldsOfStudy"] = item.get("fieldsOfStudy") or []
                    papers[cid]["citationCount"] = item.get("citationCount")
        except Exception as e:
            logger.warning("S2 batch metadata fetch failed: %s", e)

        result = list(papers.values())
        logger.info("S2 retrieved %d papers for query: %r", len(result), query[:60])
        return result


# ---------------------------------------------------------------------------
# 1. Paper retrieval signal
# ---------------------------------------------------------------------------

class PaperRetrievalSignals:
    """
    Fetches relevant papers from Semantic Scholar.

    Output:
      retrieved_papers: list[dict]  – paperId, title, year, authors, url,
                                       fieldsOfStudy, citationCount
    """

    def __init__(self, s2_api_key: Optional[str] = None, top_k: int = 5):
        self.retriever = S2SnippetRetriever(api_key=s2_api_key, top_k=top_k)

    def __call__(self, question: str, context: dict) -> dict:
        return {"retrieved_papers": self.retriever.retrieve(question)}


# ---------------------------------------------------------------------------
# 2. Rubric signal
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

_DEFAULT_RUBRIC = {
    "criteria": [
        {"name": "domain_depth", "description": "Requires deep domain knowledge.", "weight": 0.3},
        {"name": "multi_hop", "description": "Requires reasoning across multiple facts.", "weight": 0.3},
        {"name": "ambiguity", "description": "Question contains ambiguous or nuanced phrasing.", "weight": 0.2},
        {"name": "evidence_scarcity", "description": "Supporting evidence is rare or hard to find.", "weight": 0.2},
    ],
    "usage": None,
    "cost_usd": 0.0,
}


class RubricSignals:
    """
    Generates a difficulty rubric for the question via LLM.

    Output:
      rubric: dict  – criteria list with name/description/weight, plus usage and cost_usd
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

    def _get_client(self) -> OpenAI:
        if self._client is None:
            kwargs: dict = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def __call__(self, question: str, context: dict) -> dict:
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
            rubric = json.loads(resp.choices[0].message.content or "")
            rubric["usage"] = usage_dict
            rubric["cost_usd"] = cost
            return {"rubric": rubric}
        except Exception as e:
            logger.warning("RubricSignals failed: %s. Using default rubric.", e)
            return {"rubric": _DEFAULT_RUBRIC}


# ---------------------------------------------------------------------------
# 3. S2 metadata signals  (derived from retrieved_papers in context)
# ---------------------------------------------------------------------------

_FIELD_BUCKETS: dict[str, str] = {
    "Medicine": "biomedical", "Biology": "biomedical",
    "Chemistry": "biomedical", "Neuroscience": "biomedical",
    "Psychology": "social", "Sociology": "social",
    "Economics": "social", "Political Science": "social", "History": "social",
    "Computer Science": "technical", "Engineering": "technical",
    "Environmental Science": "technical",
    "Mathematics": "formal", "Physics": "formal",
    "Philosophy": "formal", "Linguistics": "formal",
    "Art": "other", "Business": "other", "Geography": "other", "Law": "other",
}

_META_KEYWORDS = frozenset({
    "meta-analysis", "systematic review", "umbrella review", "scoping review",
})


class S2MetadataSignals:
    """
    Derives signals from 'retrieved_papers' already in context.
    Must be registered AFTER PaperRetrievalSignals.

    Outputs (all prefixed s2_):
      s2_result_count            int
      s2_max_year                int
      s2_avg_year                float
      s2_pct_last_5yr            float
      s2_year_spread             int
      s2_fields                  list[str]
      s2_field_count             int
      s2_evidence_heterogeneity  int   – count of distinct high-level field buckets
      s2_meta_analysis_present   bool
      s2_avg_citation_count      float  (omitted if no citation data)
      s2_min_citation_count      int
      s2_max_citation_count      int
    """

    def __call__(self, question: str, context: dict) -> dict:
        papers = context.get("retrieved_papers")
        if not papers:
            logger.warning("S2MetadataSignals: no retrieved_papers in context")
            return {}

        # Recency
        years = [p["year"] for p in papers if p.get("year") is not None]
        if years:
            current_year = datetime.date.today().year
            max_year, min_year = max(years), min(years)
            avg_year = round(sum(years) / len(years), 1)
            pct_last_5yr = round(
                sum(1 for y in years if y >= current_year - 5) / len(years), 2
            )
            year_spread = max_year - min_year
        else:
            max_year = min_year = avg_year = pct_last_5yr = year_spread = None
        
        # Field diversity
        all_fields: list[str] = []
        for p in papers:
            all_fields.extend(p.get("fieldsOfStudy") or [])
        unique_fields = sorted(set(all_fields))
        buckets = {_FIELD_BUCKETS.get(f, "other") for f in unique_fields}

        # Meta-analysis presence
        meta_present = any(
            kw in (p.get("title") or "").lower()
            for p in papers
            for kw in _META_KEYWORDS
        )

        # Citation stats
        citation_counts = [
            p["citationCount"] for p in papers if p.get("citationCount") is not None
        ]

        result: dict = {
            "s2_result_count": len(papers),
            "s2_max_year": max_year,
            "s2_avg_year": avg_year,
            "s2_pct_last_5yr": pct_last_5yr,
            "s2_year_spread": year_spread,
            "s2_fields": unique_fields,
            "s2_field_count": len(unique_fields),
            "s2_evidence_heterogeneity": len(buckets),
            "s2_meta_analysis_present": meta_present,
        }
        if citation_counts:
            result["s2_avg_citation_count"] = round(
                sum(citation_counts) / len(citation_counts), 1
            )
            result["s2_min_citation_count"] = min(citation_counts)
            result["s2_max_citation_count"] = max(citation_counts)

        logger.info(
            "S2MetadataSignals: %d papers, %d fields, heterogeneity=%d, meta=%s",
            len(papers), len(unique_fields), len(buckets), meta_present,
        )
        return result


# ---------------------------------------------------------------------------
# 4. LLM analysis signals
# ---------------------------------------------------------------------------

_LLM_ANALYSIS_SYSTEM = """\
You are an expert at decomposing research questions. Given a question, respond ONLY
with a valid JSON object containing exactly these keys:

{
  "causal_chain_depth": <int 0-5>,
  "inference_type": "<deductive|inductive|abductive|mixed>",
  "sub_question_list": ["<sub-question 1>", "<sub-question 2>", ...],
  "sub_question_count": <int>,
  "scope_anchors": ["<constraint>", ...]
}

Definitions:
  causal_chain_depth  – number of causal hops implicit in a complete answer
                        0 = no causal reasoning, 5 = five or more causal steps
  inference_type      – dominant reasoning mode required
                        deductive  = applying established laws/rules to a specific case
                        inductive  = generalising from specific cases to a pattern
                        abductive  = inferring the best explanation for observations
                        mixed      = answer requires multiple inference types
  sub_question_list  – list of distinct sub-questions a complete long-form answer must address
  sub_question_count – count of distinct sub-questions (derived from sub_question_list)
  scope_anchors       – explicit constraints stated IN the question
                        (e.g. "in elderly patients", "after 2010", "in the US")
                        Empty array [] if the question has no explicit scope constraints.

No markdown fences. No extra keys."""

_VALID_INFERENCE_TYPES = frozenset({"deductive", "inductive", "abductive", "mixed"})


class LLMAnalysisSignals:
    """
    Single LLM call covering causal chain depth, inference type,
    sub-question count, and scope anchoring.

    Outputs:
      causal_chain_depth  int (0-5)
      inference_type      str
      sub_question_count  int
      scope_anchors       list[str]
      scope_anchor_count  int
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

    def _get_client(self) -> OpenAI:
        if self._client is None:
            kwargs: dict = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def __call__(self, question: str, context: dict) -> dict:
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _LLM_ANALYSIS_SYSTEM},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                response_format={"type": "json_object"},
            )
            usage = getattr(resp, "usage", None)
            if usage:
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                logger.warning("llm_analysis cost=%.6f", _cost_usd(self.model_name, pt, ct))

            data = json.loads(resp.choices[0].message.content)
            anchors: list[str] = data.get("scope_anchors") or []
            inference = data.get("inference_type", "mixed")
            if inference not in _VALID_INFERENCE_TYPES:
                inference = "mixed"

            result = {
                "causal_chain_depth": max(0, min(5, int(data.get("causal_chain_depth", 0)))),
                "inference_type": inference,
                "sub_question_count": max(1, int(data.get("sub_question_count", 1))),
                "sub_question_list": data.get("sub_question_list") or [],
                "scope_anchors": anchors,
                "scope_anchor_count": len(anchors),
            }
            logger.info(
                "LLMAnalysisSignals: causal=%d type=%s sub_qs=%d anchors=%d",
                result["causal_chain_depth"], result["inference_type"],
                result["sub_question_count"], result["scope_anchor_count"],
            )
            return result
        except Exception as e:
            logger.warning("LLMAnalysisSignals failed: %s", e)
            return {}


# ---------------------------------------------------------------------------
# Shared concept extraction helper
# ---------------------------------------------------------------------------

_CONCEPT_EXTRACT_SYSTEM = """\
Extract 3-7 key domain concepts from the question. Focus on technical terms,
named entities, and domain-specific ideas. Exclude generic words like "role",
"effect", "relationship", or "impact".
Respond ONLY with valid JSON: {"concepts": ["concept1", "concept2", ...]}"""


def _extract_concepts(
    question: str, context: dict, client: OpenAI, model_name: str
) -> list[str]:
    """
    Return key concepts, using context["key_concepts"] as a cache so multiple
    providers sharing this helper only trigger one LLM call.
    """
    cached = context.get("key_concepts")
    if cached and isinstance(cached, list) and len(cached) >= 1:
        return cached
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _CONCEPT_EXTRACT_SYSTEM},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        concepts = [
            str(c) for c in (json.loads(resp.choices[0].message.content).get("concepts") or [])
            if c
        ]
        logger.info("Extracted %d concepts: %s", len(concepts), concepts)
        context["key_concepts"] = concepts  # cache for subsequent providers
        return concepts
    except Exception as e:
        logger.warning("Concept extraction failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# 5. Concept embedding signals
# ---------------------------------------------------------------------------

class ConceptEmbeddingSignals:
    """
    Embeds each key concept and computes pairwise cosine distances.
    High average distance → concepts span different semantic domains → harder.

    Outputs:
      key_concepts                    list[str]
      concept_embedding_avg_distance  float  (0–2, higher = more cross-domain)
      concept_embedding_max_distance  float
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5-mini",
        embedding_model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.base_url = base_url
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            kwargs: dict = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def __call__(self, question: str, context: dict) -> dict:
        client = self._get_client()
        concepts = _extract_concepts(question, context, client, self.model_name)
        base = {"key_concepts": concepts}

        if len(concepts) < 2:
            return {**base, "concept_embedding_avg_distance": None, "concept_embedding_max_distance": None}

        try:
            resp = client.embeddings.create(model=self.embedding_model, input=concepts)
            vectors = [item.embedding for item in resp.data]
            distances = [
                _cosine_distance(vectors[i], vectors[j])
                for i in range(len(vectors))
                for j in range(i + 1, len(vectors))
            ]
            avg_dist = round(sum(distances) / len(distances), 4)
            max_dist = round(max(distances), 4)
            logger.info("Concept embedding distances: avg=%.4f max=%.4f", avg_dist, max_dist)
            return {
                **base,
                "concept_embedding_avg_distance": avg_dist,
                "concept_embedding_max_distance": max_dist,
            }
        except Exception as e:
            logger.warning("ConceptEmbeddingSignals failed: %s", e)
            return base


# ---------------------------------------------------------------------------
# 6. Concept co-occurrence signals
# ---------------------------------------------------------------------------

class ConceptCooccurrenceSignals:
    """
    Measures how rarely key concepts co-occur in the literature via S2 searches.

    For each concept: S2 total result count (individual search).
    For all concepts combined: S2 total result count (joint search).
    rarity_ratio = combined_total / geometric_mean(individual_totals)

    Low ratio → concepts rarely appear together → novel intersection → harder.
    Note: S2 'total' may be approximate for large result sets.

    Outputs:
      key_concepts                list[str]  (cached from ConceptEmbeddingSignals if run first)
      concept_individual_counts   dict[str, int]
      concept_combined_count      int
      concept_rarity_ratio        float
    """

    S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(
        self,
        s2_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5-mini",
        base_url: Optional[str] = None,
    ):
        self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            kwargs: dict = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _s2_total(self, query: str) -> int:
        headers = {"x-api-key": self.s2_api_key} if self.s2_api_key else {}
        try:
            resp = requests.get(
                self.S2_SEARCH_URL,
                params={"query": query, "limit": 1, "fields": "paperId"},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            body = resp.json()
            return body.get("total") or len(body.get("data", []))
        except Exception as e:
            logger.warning("S2 total count failed for %r: %s", query[:50], e)
            return 0

    def __call__(self, question: str, context: dict) -> dict:
        client = self._get_client()
        concepts = _extract_concepts(question, context, client, self.model_name)
        if len(concepts) < 2:
            return {"key_concepts": concepts}

        individual_counts = {c: self._s2_total(c) for c in concepts}
        combined_count = self._s2_total(" ".join(concepts))

        log_sum = sum(math.log(max(c, 1)) for c in individual_counts.values())
        geo_mean = math.exp(log_sum / len(individual_counts))
        rarity_ratio = round((combined_count + 1) / (geo_mean + 1), 4)

        logger.info(
            "Co-occurrence: combined=%d geo_mean=%.0f rarity_ratio=%.4f",
            combined_count, geo_mean, rarity_ratio,
        )
        return {
            "key_concepts": list(concepts),
            "concept_individual_counts": individual_counts,
            "concept_combined_count": combined_count,
            "concept_rarity_ratio": rarity_ratio,
        }


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def register_default_signals(
    scorer,
    api_key: Optional[str] = None,
    model_name: str = "gpt-5-mini",
    s2_api_key: Optional[str] = None,
    top_k: int = 5,
    base_url: Optional[str] = None,
    use_retriever: bool = True,
    use_rubric: bool = True,
) -> None:
    """Register the standard paper retrieval and rubric providers."""
    if use_retriever:
        scorer.register_input(
            "retrieved_papers",
            PaperRetrievalSignals(s2_api_key=s2_api_key, top_k=top_k),
        )
    if use_rubric:
        scorer.register_input(
            "rubric",
            RubricSignals(api_key=api_key, model_name=model_name, base_url=base_url),
        )


def register_all_signals(
    scorer,
    api_key: Optional[str] = None,
    model_name: str = "gpt-5-mini",
    s2_api_key: Optional[str] = None,
    top_k: int = 5,
    base_url: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    use_retriever: bool = True,
    use_rubric: bool = True,
) -> None:
    """
    Register all providers: default (papers + rubric) plus all extended signals.

    Provider order matters — S2MetadataSignals reads retrieved_papers from context,
    and ConceptCooccurrenceSignals reuses the key_concepts cache set by
    ConceptEmbeddingSignals.
    """
    register_default_signals(
        scorer,
        api_key=api_key,
        model_name=model_name,
        s2_api_key=s2_api_key,
        top_k=top_k,
        base_url=base_url,
        use_retriever=use_retriever,
        use_rubric=use_rubric,
    )
    scorer.register_input("s2_metadata", S2MetadataSignals())
    scorer.register_input(
        "llm_analysis",
        LLMAnalysisSignals(api_key=api_key, model_name=model_name, base_url=base_url),
    )
    # scorer.register_input(
    #     "concept_embedding",
    #     ConceptEmbeddingSignals(
    #         api_key=api_key,
    #         model_name=model_name,
    #         embedding_model=embedding_model,
    #         base_url=base_url,
    #     ),
    # )
    # scorer.register_input(
    #     "concept_cooccurrence",
    #     ConceptCooccurrenceSignals(
    #         s2_api_key=s2_api_key,
    #         api_key=api_key,
    #         model_name=model_name,
    #         base_url=base_url,
    #     ),
    # )
