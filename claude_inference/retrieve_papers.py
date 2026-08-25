#!/usr/bin/env python3
r"""
Standalone paper retrieval, mirroring the RAG stage of ai2-scholarqa-lib.

Reference implementation:
  https://github.com/allenai/ai2-scholarqa-lib/tree/main/api/scholarqa/rag
  (retriever_base.py, retrieval.py, reranker/reranker_base.py,
   plus preprocess/query_preprocessor.py for the query decomposition)

Pipeline (same order and semantics as ScholarQA):

  1. Query decomposition / rewriting (Claude, via the QUERY_DECOMPOSER_PROMPT
     lifted verbatim from scholarqa/llms/prompts.py). Produces:
       - rewritten_query          -> used for snippet (full text) search
       - keyword_query            -> used for paper/search (keyword) search
       - search_filters           -> year / venue / fieldsOfStudy
  2. S2 `snippet/search` for up to n_retrieval passages on the rewritten query;
     passages with <= 20 whitespace tokens are dropped.
  3. S2 `paper/search` for up to n_keyword_srch abstracts on the keyword query;
     papers already surfaced by snippet search are dropped. These come back with
     full metadata, so they double as a metadata cache.
  4. Cross-encoder rerank over (query, passage) pairs, truncated to n_rerank.
  5. `paper/batch` metadata fetch for anything not already covered.
  6. Aggregate passages up to the paper level, keep papers whose best score is
     >= context_threshold, and format each paper into the markdown blob
     ScholarQA feeds to the quote-extraction step
     (`relevance_judgment_input_expanded`) plus its `reference_string` citation key.

Reranking:
  The default is a remote vLLM reranker (Cohere-compatible /v1/rerank endpoint)
  serving Qwen/Qwen3-Reranker-4B. The score template is NOT shipped in the pip
  wheel (it only exists in a vLLM git checkout), so fetch it first and pass an
  absolute path -- a repo-relative path fails with "appears path-like, but
  doesn't exist". On the GPU box:

      curl -sL -o ~/vllm-templates/qwen3_reranker.jinja \
        https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/examples/pooling/score/template/qwen3_reranker.jinja

      vllm serve Qwen/Qwen3-Reranker-4B   --runner pooling   --trust-remote-code   --max-model-len 4096   --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'   --chat-template ~/vllm-templates/qwen3_reranker.jinja --port 8017 --gpu-memory-utilization 0.7

  --chat-template is REQUIRED and its absence is silent: without it vLLM falls
  back to concatenating query+document, the model never sees the judge prompt it
  was trained on, and scores collapse into a narrow band (verified: 0.86 vs 0.83
  for a relevant/irrelevant pair, versus 0.993 vs 0.00003 with the template).
  To confirm a running server is applying it, rerank two documents of obviously
  different relevance -- scores should be near 1.0 and near 0.0, and usage
  should report ~85-110 prompt tokens per pair rather than ~16.

  The server is assumed to be unauthenticated, so keep it on a trusted network
  or behind an SSH tunnel. --reranker none skips reranking entirely.

Environment:
  S2_API_KEY         Semantic Scholar API key (optional but strongly recommended;
                     without it you will get rate limited hard).
  ANTHROPIC_API_KEY  Only needed for query decomposition (--no-decompose skips it).
  VLLM_RERANK_URL    Base URL of the vLLM reranker, e.g. http://gpu-host:8000

Usage:
  export S2_API_KEY=...
  export ANTHROPIC_API_KEY=...

  python retrieve_papers.py "how do LLM agents plan over long horizons?" \
      --reranker-url http://gpu-host:8000 \
      --out papers.json

Importable:
  from retrieve_papers import retrieve_papers
  result = retrieve_papers("my question", n_rerank=25)
  for paper in result["papers"]:
      print(paper["reference_string"], paper["title"])
"""

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger("retrieve_papers")

# ---------------------------------------------------------------------------
# S2 API configuration (scholarqa/utils.py)
# ---------------------------------------------------------------------------

S2_API_BASE_URL = "https://api.semanticscholar.org/graph/v1/"

NUMERIC_META_FIELDS = {"year", "citationCount", "referenceCount", "influentialCitationCount"}
CATEGORICAL_META_FIELDS = {
    "title",
    "abstract",
    "corpusId",
    "authors",
    "venue",
    "publicationVenue",
    "isOpenAccess",
    "openAccessPdf",
    "s2FieldsOfStudy",
    "externalIds",
}
METADATA_FIELDS = ",".join(sorted(CATEGORICAL_META_FIELDS.union(NUMERIC_META_FIELDS)))

# Snippet subfields requested from snippet/search (scholarqa/rag/retrieval.py)
SNIPPET_SEARCH_FIELDS = [
    "text",
    "snippetKind",
    "snippetOffset",
    "section",
    "annotations.refMentions",
    "annotations.sentences.start",
    "annotations.sentences.end",
]

# paper/batch caps out at 500 ids per POST.
METADATA_BATCH_SIZE = 500

# Query decomposition model. ScholarQA uses anthropic/claude-sonnet-4-20250514 (via litellm);
# claude-sonnet-4-5 is the nearest live equivalent -- that snapshot is deprecated (retires
# 2026-06-15). Decomposition is a small structured-extraction task, so this is not a
# capability-sensitive choice; override with --decomposer-model.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"
# Default served model: Qwen3-Reranker-4B, via vLLM's Qwen3ForSequenceClassification conversion.
# Needs --hf_overrides AND the qwen3 score template server-side -- see the module docstring.
# Scores are a softmax over the no/yes token logits, i.e. [0, 1], so context_threshold compares
# on the same scale as the mxbai v2 model this replaced.
#
# Alternatives (whatever you serve, pass the same string via --reranker-model):
#   tomaarsen/Qwen3-Reranker-4B-seq-cls   same weights, pre-converted -- no --hf_overrides needed
#   mixedbread-ai/mxbai-rerank-large-v2   1.5B, ~2.6x faster, needs its own overrides + template
#   BAAI/bge-reranker-v2-m3               568M, no overrides or template at all
#
# Not servable on vLLM: mixedbread-ai/mxbai-rerank-large-v1, the model AI2 runs (DeBERTa-v3 --
# there is no DebertaV2 implementation in vLLM's model registry).
DEFAULT_VLLM_RERANKER = "Qwen/Qwen3-Reranker-4B"

# Drop papers whose best passage scores below this (paper-level max, applied after rerank
# truncation). 0.0 = no filtering, which is what ScholarQA actually runs: run_configs/default.json
# passes paper_finder_args.context_threshold = 0.0, overriding the 0.5 default in the
# PaperFinderWithReranker signature. n_rerank alone bounds the result set.
#
# That 0.5 was calibrated for their reranker (mxbai-rerank-large-v1, DeBERTa), not the model
# served here, whose scores have a different distribution -- so it does not transfer. Tune this
# by inspecting `relevance_judgement` on real runs before raising it.
#
# Note a positive threshold also drops every keyword-search paper when --reranker none is used:
# those keep score 0.0 until a reranker scores them.
DEFAULT_CONTEXT_THRESHOLD = 0.0


def query_s2_api(
    end_pt: str = "paper/batch",
    params: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    method: str = "get",
    max_retries: int = 4,
    timeout: int = 60,
) -> Dict[str, Any]:
    """GET/POST against the S2 graph API with backoff on 429/5xx."""
    url = S2_API_BASE_URL + end_pt
    headers = {}
    api_key = os.getenv("S2_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    req_method = requests.get if method == "get" else requests.post
    backoff = 2.0
    last_status, last_body = None, ""
    for attempt in range(max_retries):
        response = req_method(url, headers=headers, params=params, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        last_status, last_body = response.status_code, response.text[:500]
        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
            wait = backoff * (2 ** attempt)
            logger.warning(
                "S2 %s returned %s, retrying in %.0fs (attempt %d/%d)",
                end_pt, response.status_code, wait, attempt + 1, max_retries,
            )
            time.sleep(wait)
            continue
        break
    raise RuntimeError(
        "S2 API request to {} failed with status {}: {}".format(end_pt, last_status, last_body)
    )


def make_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def get_paper_metadata(corpus_ids: Sequence[str], fields: str = METADATA_FIELDS) -> Dict[str, Any]:
    """Fetch metadata for corpus ids, keyed by str(corpusId)."""
    corpus_ids = [cid for cid in dict.fromkeys(str(c) for c in corpus_ids) if cid]
    if not corpus_ids:
        return {}

    paper_metadata: Dict[str, Any] = {}
    for start in range(0, len(corpus_ids), METADATA_BATCH_SIZE):
        chunk = corpus_ids[start:start + METADATA_BATCH_SIZE]
        paper_data = query_s2_api(
            end_pt="paper/batch",
            params={"fields": fields},
            payload={"ids": ["CorpusId:{0}".format(cid) for cid in chunk]},
            method="post",
        )
        for pdata in paper_data:
            if not pdata or "corpusId" not in pdata:
                continue
            pmeta = {
                k: make_int(v) if k in NUMERIC_META_FIELDS else pdata.get(k)
                for k, v in pdata.items()
            }
            if pmeta.get("s2FieldsOfStudy"):
                pmeta["s2FieldsOfStudy"] = [
                    f["category"] for f in pmeta["s2FieldsOfStudy"] if f.get("source") == "s2-fos-model"
                ]
            paper_metadata[str(pdata["corpusId"])] = pmeta
    return paper_metadata


def to_ascii(text: str) -> str:
    """anyascii() if available, otherwise a best-effort unicode -> ascii fold."""
    try:
        from anyascii import anyascii  # type: ignore

        return anyascii(text)
    except ImportError:
        import unicodedata

        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def get_ref_author_str(authors: Optional[List[Dict[str, str]]]) -> str:
    if not authors:
        return "NULL"
    first = (authors[0].get("name") or "").split()
    f_author_lname = first[-1] if first else "NULL"
    return f_author_lname if len(authors) == 1 else "{} et al.".format(f_author_lname)


# ---------------------------------------------------------------------------
# 1. Query decomposition (scholarqa/preprocess/query_preprocessor.py)
# ---------------------------------------------------------------------------

# `usage` carries the token counts of the decomposition call so callers can bill it;
# it defaults to None (no call made, or the call never returned a usage object).
LLMProcessedQuery = namedtuple(
    "LLMProcessedQuery", ["rewritten_query", "keyword_query", "search_filters", "usage"],
    defaults=(None,),
)

# Verbatim from scholarqa/llms/prompts.py::QUERY_DECOMPOSER_PROMPT.
QUERY_DECOMPOSER_PROMPT = """
<task>
Your task is to analyze a query issued by a user of an academic question-answering system and break it down into parts relevant for searching and retrieving high-quality, well-cited answers. The goal is to create a structured JSON output that can be used by an academic search engine API, which supports filters like publication years, venues, authors, and fields of study.

Your output should decompose the query into the following components:
1. Publication years: If the query specifies a time range (e.g., "recent" or "last five years"), convert it to the relevant year range. If no time range is mentioned, leave these fields blank.
   - The current year is 2025. Interpret "recent" as 2022-2025 and adjust other relative terms (e.g., "last decade," "since 2018") accordingly.
2. Venues: Include any journals, conferences, or publishers mentioned explicitly in the query as a comma separated string. Use their exact names.
3. Authors: List any authors mentioned explicitly in the query. Each author name should appear as a separate entry in an array.
4. Fields of study: Use only the following fields of study. If the query includes subfields or ambiguous terms, map them to the closest match from this list:
   - Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics.
   Concatenate multiple fields with commas and no spaces (e.g., `Physics,Mathematics`).
5. Rewritten query: Simplify the remaining query into a concise, natural-language phrase, excluding any information already extracted into `year`, `venues`, `authors`, or `fields_of_study`.
6. Rewritten query for keyword search: Remove unnecessary stop words and connectors to create a keyword-friendly version of the remaining query, excluding any information already extracted into other fields.
7. Complex, multi-sentence queries should still be complete in terms of content when rewritten. The goal of the rewritten query is to remove the metadata like year, venues, authors and fields_of_study, but keep all of the important topical content that needs to be addressed in an answer.

<note about handling ambiguous terms and missing information>
- If a field cannot be inferred from the query (e.g., no authors are mentioned), leave it empty.
- For terms not matching the fields of study list, map them to the closest matching field(s). For instance, "machine learning" should map to `Computer Science`, while "neuroscience" might map to `Biology` or `Psychology` based on context.
</note about handling ambiguous terms and missing information>
</task>

<examples>

<example input #1>
What are the latest papers by Andrew Ng on deep reinforcement learning?
</example input #1>

<example output #1>
```json
{
    "earliest_search_year": "2022",
    "latest_search_year": "2025",
    "venues": "",
    "authors": ["Andrew Ng"],
    "field_of_study": "Computer Science",
    "rewritten_query": "Deep reinforcement learning.",
    "rewritten_query_for_keyword_search": "deep reinforcement learning"
}
```
</example output #1>

<example input #2>
Summarize the findings on climate policy impacts in articles published in Nature or Science from the last five years.
</example input #2>

<example output #2>
```json
{
    "earliest_search_year": "2020",
    "latest_search_year": "2025",
    "venues": "Nature,Science",
    "authors": [],
    "field_of_study": "Environmental Science,Political Science",
    "rewritten_query": "Findings on climate policy impacts.",
    "rewritten_query_for_keyword_search": "climate policy impacts"
}
```
</example output #2>

<example input #3>
Discuss recent contributions by Noam Chomsky to the study of linguistics and cognitive science.
</example input #3>

<example output #3>
```json
{
    "earliest_search_year": "2022",
    "latest_search_year": "2025",
    "venues": "",
    "authors": ["Noam Chomsky"],
    "field_of_study": "Linguistics,Psychology",
    "rewritten_query": "Contributions to linguistics and cognitive science.",
    "rewritten_query_for_keyword_search": "linguistics cognitive science"
}
```
</example output #3>

<example input #4>
What are the effects of climate change on agricultural productivity in Sub-Saharan Africa in recent studies?
</example input #4>

<example output #4>
```json
{
    "earliest_search_year": "2022",
    "latest_search_year": "2025",
    "venues": "",
    "authors": [],
    "field_of_study": "Environmental Science,Agricultural and Food Sciences",
    "rewritten_query": "Effects of climate change on agricultural productivity in Sub-Saharan Africa.",
    "rewritten_query_for_keyword_search": "climate change agricultural productivity Sub-Saharan Africa"
}
```
</example output #4>

<example input #5>
Explore the role of neural networks in solving mathematical optimization problems.
</example input #5>

<example output #5>
```json
{
    "earliest_search_year": "",
    "latest_search_year": "",
    "venues": "",
    "authors": [],
    "field_of_study": "Computer Science,Mathematics",
    "rewritten_query": "Role of neural networks in solving mathematical optimization problems.",
    "rewritten_query_for_keyword_search": "neural networks mathematical optimization problems"
}
```
</example output #5>

<example input #6>
Discuss the historical significance of the Renaissance period on modern art and philosophy.
</example input #6>

<example output #6>
```json
{
    "earliest_search_year": "",
    "latest_search_year": "",
    "venues": "",
    "authors": [],
    "field_of_study": "Art,Philosophy",
    "rewritten_query": "Historical significance of Renaissance on modern art and philosophy.",
    "rewritten_query_for_keyword_search": "Renaissance modern art philosophy"
}
```
</example output #6>

<example input #7>
Review papers by Andrew Ng and Yann LeCun on neural networks since 2010.
</example input #7>

<example output #7>
```json
{
    "earliest_search_year": "2010",
    "latest_search_year": "2025",
    "venues": "",
    "authors": ["Andrew Ng", "Yann LeCun"],
    "field_of_study": "Computer Science",
    "rewritten_query": "Neural networks.",
    "rewritten_query_for_keyword_search": "neural networks"
}
```
</example output #7>
</examples>
"""

# The upstream prompt hardcodes 2025; keep the prompt verbatim and correct the
# clock with a trailing note so relative dates ("recent") resolve correctly.
CURRENT_YEAR_NOTE = (
    "\n<current_date>\nThe current year is {year}. Override any other year mentioned above: "
    "interpret \"recent\" as {recent_start}-{year} and scale other relative time expressions "
    "(\"last five years\", \"since 2018\", \"last decade\") to end at {year}. The example "
    "outputs were written in an earlier year; follow their format, not their absolute years.\n"
    "</current_date>\n"
)

DECOMPOSED_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "earliest_search_year": {
            "type": "string",
            "description": "The earliest year to search for papers, or empty string",
        },
        "latest_search_year": {
            "type": "string",
            "description": "The latest year to search for papers, or empty string",
        },
        "venues": {
            "type": "string",
            "description": "Comma separated list of venues, or empty string",
        },
        "authors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of author names mentioned in the query",
        },
        "field_of_study": {
            "type": "string",
            "description": "Comma separated list of fields of study, or empty string",
        },
        "rewritten_query": {
            "type": "string",
            "description": "The rewritten simplified query, used for full text snippet search",
        },
        "rewritten_query_for_keyword_search": {
            "type": "string",
            "description": "The rewritten query for keyword search",
        },
    },
    "required": [
        "earliest_search_year",
        "latest_search_year",
        "venues",
        "authors",
        "field_of_study",
        "rewritten_query",
        "rewritten_query_for_keyword_search",
    ],
    "additionalProperties": False,
}


def decompose_query(
    query: str,
    model: str = DEFAULT_CLAUDE_MODEL,
    current_year: Optional[int] = None,
) -> LLMProcessedQuery:
    """Ask Claude to split the query into a rewritten query, a keyword query, and S2 filters.

    Falls back to (query, "", {}) on any failure, exactly like ScholarQA does. The
    returned `usage` holds the call's token counts (None if no call completed) so the
    caller can account for its cost.
    """
    year = current_year or dt.date.today().year
    system_prompt = QUERY_DECOMPOSER_PROMPT + CURRENT_YEAR_NOTE.format(
        year=year, recent_start=year - 3
    )

    search_filters: Dict[str, str] = {}
    # Captured as soon as the API returns, so tokens are still billed to the caller when
    # a later step (refusal, JSON parse) sends us down the fallback path.
    usage: Optional[Dict[str, int]] = None
    try:
        import anthropic

        client = anthropic.Anthropic()
        req = dict(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )
        output_config = {"format": {"type": "json_schema", "schema": DECOMPOSED_QUERY_SCHEMA}}
        try:
            response = client.messages.create(output_config=output_config, **req)
        except TypeError:
            # SDKs older than ~0.6x have no output_config kwarg (anthropic 0.54 raises
            # "unexpected keyword argument"); extra_body puts it in the raw request body,
            # which the API accepts either way.
            response = client.messages.create(extra_body={"output_config": output_config}, **req)
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(
                    usage_obj, "cache_creation_input_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(
                    usage_obj, "cache_read_input_tokens", 0) or 0,
            }
        if response.stop_reason == "refusal":
            raise RuntimeError("query decomposition refused by the model")
        text = next(b.text for b in response.content if b.type == "text")
        decomposed = json.loads(text)
        logger.info("Decomposed query: %s", decomposed)

        rewritten_query = decomposed.get("rewritten_query") or query
        keyword_query = decomposed.get("rewritten_query_for_keyword_search") or ""
        earliest = str(decomposed.get("earliest_search_year") or "")
        latest = str(decomposed.get("latest_search_year") or "")
        if earliest or latest:
            search_filters["year"] = "{}-{}".format(earliest, latest)
        if decomposed.get("venues"):
            search_filters["venue"] = decomposed["venues"]
        if decomposed.get("field_of_study"):
            search_filters["fieldsOfStudy"] = decomposed["field_of_study"]
    except Exception as e:
        logger.warning("Error while decomposing query (%s); falling back to the raw query", e)
        return LLMProcessedQuery(
            rewritten_query=query, keyword_query=query, search_filters={}, usage=usage,
        )

    return LLMProcessedQuery(
        rewritten_query=rewritten_query,
        keyword_query=keyword_query,
        search_filters=search_filters,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# 2/3. Retrieval (scholarqa/rag/retriever_base.py::FullTextRetriever)
# ---------------------------------------------------------------------------


class FullTextRetriever:
    """S2 snippet search (full text passages) + paper search (abstracts)."""

    def __init__(self, n_retrieval: int = 256, n_keyword_srch: int = 20):
        self.n_retrieval = n_retrieval
        self.n_keyword_srch = n_keyword_srch

    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        snippets_list = self.snippet_search(query, **filter_kwargs)
        # Upstream drops very short passages -- they are almost never usable evidence.
        return [s for s in snippets_list if len(s["text"].split(" ")) > 20]

    def snippet_search(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        if not self.n_retrieval:
            return []
        query_params = {k: v for k, v in filter_kwargs.items() if v}
        query_params.update({"query": query, "limit": self.n_retrieval})
        logger.info("snippet/search params: %s", query_params)

        snippets = query_s2_api(end_pt="snippet/search", params=query_params, method="get")
        snippets_list = []
        for fields in snippets.get("data") or []:
            snippet, paper = fields["snippet"], fields["paper"]
            res_map: Dict[str, Any] = {
                "corpus_id": str(paper["corpusId"]),
                "title": paper.get("title"),
                "text": snippet.get("text"),
                "score": fields.get("score", 0.0),
                "section_title": snippet.get("snippetKind"),
            }
            if snippet.get("snippetKind") == "body" and snippet.get("section"):
                res_map["section_title"] = snippet["section"]

            offset = snippet.get("snippetOffset") or {}
            res_map["char_start_offset"] = offset.get("start") or 0

            annotations = snippet.get("annotations") or {}
            res_map["sentence_offsets"] = annotations.get("sentences") or []
            res_map["ref_mentions"] = [
                rm
                for rm in (annotations.get("refMentions") or [])
                if rm.get("matchedPaperCorpusId") and rm.get("start") and rm.get("end")
            ]
            res_map["pdf_hash"] = snippet.get("extractionPdfHash", "")
            res_map["stype"] = "vespa"
            snippets_list.append(res_map)
        return snippets_list

    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        return self.keyword_search(query, **filter_kwargs) if self.n_keyword_srch else []

    def keyword_search(self, kquery: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """S2 paper/search. Returns full metadata, so these papers skip the batch fetch."""
        query_params = {k: v for k, v in filter_kwargs.items() if v}
        query_params.update(
            {"query": kquery, "limit": self.n_keyword_srch, "fields": METADATA_FIELDS}
        )
        logger.info("paper/search params: %s", {k: v for k, v in query_params.items() if k != "fields"})

        res = query_s2_api(end_pt="paper/search", params=query_params, method="get")
        paper_data = [
            pd
            for pd in (res.get("data") or [])
            if pd.get("corpusId") and pd.get("title") and pd.get("abstract")
        ]
        normalized = []
        for pd in paper_data:
            pdn = {k: make_int(v) if k in NUMERIC_META_FIELDS else v for k, v in pd.items()}
            pdn["corpus_id"] = str(pdn["corpusId"])
            pdn["text"] = pdn["abstract"]
            pdn["section_title"] = "abstract"
            pdn["char_start_offset"] = 0
            pdn["sentence_offsets"] = []
            pdn["ref_mentions"] = []
            pdn["score"] = 0.0
            pdn["stype"] = "public_api"
            pdn["pdf_hash"] = ""
            if pdn.get("s2FieldsOfStudy"):
                pdn["s2FieldsOfStudy"] = [
                    f["category"] for f in pdn["s2FieldsOfStudy"] if f.get("source") == "s2-fos-model"
                ]
            normalized.append(pdn)
        return normalized


# ---------------------------------------------------------------------------
# 4. Rerankers (scholarqa/rag/reranker/reranker_base.py; upstream's Modal-hosted
#    reranker is replaced here by a remote vLLM server)
# ---------------------------------------------------------------------------


class VLLMReranker:
    """Remote reranker served by vLLM's Cohere-compatible rerank endpoint.

    Launch on the GPU host:
        vllm serve BAAI/bge-reranker-v2-m3 --host 0.0.0.0 --port 8000 --max-model-len 2048

    Request:  POST {url}/v1/rerank  {"model", "query", "documents", ...}
    Response: {"results": [{"index": i, "relevance_score": s, "document": {...}}, ...]}
    Results come back sorted by score, so `index` maps them back to input order.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model: str = DEFAULT_VLLM_RERANKER,
        batch_size: int = 64,
        timeout: int = 120,
        max_doc_tokens: Optional[int] = None,
        max_retries: int = 3,
    ):
        url = url or os.getenv("VLLM_RERANK_URL", "")
        if not url:
            raise ValueError(
                "No reranker URL. Pass --reranker-url http://host:8000 or set VLLM_RERANK_URL."
            )
        self.url = url.rstrip("/")
        # Accept a bare host:port or a full endpoint path.
        self.endpoint = self.url if self.url.endswith("rerank") else self.url + "/v1/rerank"
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_doc_tokens = max_doc_tokens
        self.max_retries = max_retries
        logger.info("Using remote vLLM reranker %s at %s", self.model, self.endpoint)

    def _post(self, query: str, documents: List[str]) -> List[float]:
        headers = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if self.max_doc_tokens:
            payload["max_tokens_per_doc"] = self.max_doc_tokens

        last_err = ""
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=self.timeout
                )
                if resp.status_code == 200:
                    results = resp.json().get("results") or []
                    scores = [0.0] * len(documents)
                    for item in results:
                        scores[item["index"]] = float(item["relevance_score"])
                    return scores
                last_err = "HTTP {}: {}".format(resp.status_code, resp.text[:300])
                if resp.status_code < 500:
                    break  # 4xx will not fix itself
            except requests.RequestException as e:
                last_err = str(e)
            if attempt < self.max_retries - 1:
                time.sleep(2.0 * (2 ** attempt))
        raise RuntimeError("vLLM rerank request to {} failed -- {}".format(self.endpoint, last_err))

    def get_scores(self, query: str, passages: List[str]) -> List[float]:
        # Chunked so one request never carries all 256 passages; cross-encoder scores are
        # computed per (query, passage) pair, so they stay comparable across chunks.
        scores: List[float] = []
        for start in range(0, len(passages), self.batch_size):
            scores.extend(self._post(query, passages[start:start + self.batch_size]))
        return scores

    def health_check(self) -> bool:
        try:
            resp = requests.get(self.url + "/health", timeout=10)
            return resp.status_code == 200
        except requests.RequestException:
            return False


def build_reranker(
    kind: str,
    model_name: Optional[str] = None,
    url: Optional[str] = None,
    batch_size: int = 64,
    timeout: int = 120,
    max_doc_tokens: Optional[int] = None,
):
    """Instantiate a reranker.

    'auto' -> remote vLLM if a URL is configured, else no reranking.
    """
    if kind == "none":
        return None
    if kind == "auto":
        if not (url or os.getenv("VLLM_RERANK_URL")):
            logger.warning(
                "No reranker URL (--reranker-url / VLLM_RERANK_URL); running WITHOUT a reranker "
                "(results stay in S2 relevance order)."
            )
            return None
        kind = "vllm"
    if kind != "vllm":
        raise ValueError("Unknown reranker: {}".format(kind))

    return VLLMReranker(
        url=url,
        model=model_name or DEFAULT_VLLM_RERANKER,
        batch_size=batch_size,
        timeout=timeout,
        max_doc_tokens=max_doc_tokens,
    )


# ---------------------------------------------------------------------------
# 5/6. Paper finder: rerank + aggregate (scholarqa/rag/retrieval.py)
# ---------------------------------------------------------------------------


class PaperFinder:
    def __init__(
        self,
        retriever: FullTextRetriever,
        reranker=None,
        context_threshold: float = 0.0,
        n_rerank: int = 50,
        max_date: Optional[str] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.context_threshold = context_threshold
        self.n_rerank = n_rerank
        self.max_date = max_date  # "YYYY-MM"

    # -- retrieval -----------------------------------------------------------

    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        filter_kwargs = dict(filter_kwargs)
        filter_kwargs["fields"] = ",".join("snippet.{}".format(f) for f in SNIPPET_SEARCH_FIELDS)
        if self.max_date:
            filter_kwargs["insertedBefore"] = self.max_date
        return self.retriever.retrieve_passages(query, **filter_kwargs)

    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        filter_kwargs = dict(filter_kwargs)
        if self.max_date:
            # paper/search has no insertedBefore, so fold max_date into publicationDateOrYear.
            year_filter = filter_kwargs.pop("year", None)
            max_year = self.max_date.split("-")[0]
            if year_filter:
                date_start, _, date_end = year_filter.partition("-")
                date_start = min(date_start, max_year) if date_start else ""
                date_end = date_end if date_end and date_end < max_year else self.max_date
                filter_kwargs["publicationDateOrYear"] = "{}:{}".format(date_start, date_end)
            else:
                filter_kwargs["publicationDateOrYear"] = ":{}".format(self.max_date)
        return self.retriever.retrieve_additional_papers(query, **filter_kwargs)

    # -- rerank --------------------------------------------------------------

    def rerank(self, query: str, retrieved_ctxs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not retrieved_ctxs:
            return []
        if self.reranker is None:
            sorted_ctxs = sorted(retrieved_ctxs, key=lambda x: x["score"], reverse=True)
            return sorted_ctxs[: self.n_rerank] if self.n_rerank > 0 else sorted_ctxs

        passages = [
            "{} {}".format(doc["title"], doc["text"]) if doc.get("title") else doc["text"]
            for doc in retrieved_ctxs
        ]
        rerank_scores = self.reranker.get_scores(query, passages)
        for doc, score in zip(retrieved_ctxs, rerank_scores):
            doc["rerank_score"] = score
        sorted_ctxs = sorted(retrieved_ctxs, key=lambda x: x["rerank_score"], reverse=True)
        sorted_ctxs = sorted_ctxs[: self.n_rerank] if self.n_rerank > 0 else sorted_ctxs
        logger.info("Done reranking: %d passages remain", len(sorted_ctxs))
        return sorted_ctxs

    # -- aggregate -----------------------------------------------------------

    def aggregate_into_papers(
        self, snippets_list: List[Dict[str, Any]], paper_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        snippets_list = [
            s for s in snippets_list if s["corpus_id"] in paper_metadata and s.get("text")
        ]
        aggregated = self.aggregate_snippets_to_papers(snippets_list, paper_metadata)
        aggregated = [a for a in aggregated if a["relevance_judgement"] >= self.context_threshold]
        return self.format_retrieval_response(aggregated)

    @staticmethod
    def aggregate_snippets_to_papers(
        snippets_list: List[Dict[str, Any]], paper_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        logger.info("Aggregating %d passages at paper level with metadata", len(snippets_list))
        paper_snippets: Dict[str, Dict[str, Any]] = {}
        for snippet in snippets_list:
            corpus_id = snippet["corpus_id"]
            if corpus_id not in paper_snippets:
                paper_snippets[corpus_id] = dict(paper_metadata[corpus_id])
                paper_snippets[corpus_id]["corpus_id"] = corpus_id
                paper_snippets[corpus_id]["sentences"] = []
            # Abstract-only (keyword search) hits carry no passage-level evidence.
            if snippet["stype"] != "public_api":
                paper_snippets[corpus_id]["sentences"].append(snippet)
            paper_snippets[corpus_id].pop("paperId", None)
            paper_snippets[corpus_id]["relevance_judgement"] = max(
                paper_snippets[corpus_id].get("relevance_judgement", -1e9),
                snippet.get("rerank_score", snippet["score"]),
            )
            if not paper_snippets[corpus_id].get("abstract") and snippet["section_title"] == "abstract":
                paper_snippets[corpus_id]["abstract"] = snippet["text"]
        sorted_ctxs = sorted(
            paper_snippets.values(), key=lambda x: x["relevance_judgement"], reverse=True
        )
        logger.info("Scores after aggregation: %s", [round(s["relevance_judgement"], 4) for s in sorted_ctxs])
        return sorted_ctxs

    @staticmethod
    def format_sections_to_markdown(sentences: List[Dict[str, Any]]) -> str:
        """Stitch a paper's passages into '## <section>\\n<text>' blocks, in document order."""
        if not sentences:
            return ""
        ordered = sorted(sentences, key=lambda s: s.get("char_start_offset", 0))
        grouped: Dict[str, List[str]] = {}
        for s in ordered:
            title = s.get("section_title") or ""
            if title in ("abstract", "title"):
                continue
            grouped.setdefault(title, []).append(s["text"])
        return "\n\n".join(
            "## {}\n{}".format(title, "\n...\n".join(texts)) for title, texts in grouped.items()
        )

    def format_retrieval_response(
        self, agg_reranked_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        papers = []
        for cand in agg_reranked_candidates:
            if cand.get("sentences") is None or not cand.get("year"):
                continue
            authors = cand.get("authors") or []
            author_str = ", ".join(a.get("name", "") for a in authors)
            abstract = cand.get("abstract") or ""
            prepend_text = (
                "# Title: {}\n# Venue: {}\n# Authors: {}\n## Abstract\n{}\n".format(
                    cand.get("title"), cand.get("venue"), author_str, abstract
                )
            )
            section_text = self.format_sections_to_markdown(cand["sentences"])
            reference_string = to_ascii(
                "[{} | {} | {} | Citations: {}]".format(
                    make_int(cand["corpus_id"]),
                    get_ref_author_str(authors),
                    make_int(cand.get("year")),
                    make_int(cand.get("citationCount")),
                )
            )
            papers.append(
                {
                    "corpus_id": make_int(cand["corpus_id"]),
                    "title": cand.get("title"),
                    "year": make_int(cand.get("year")),
                    "venue": cand.get("venue"),
                    "authors": authors,
                    "abstract": abstract,
                    "citation_count": make_int(cand.get("citationCount")),
                    "reference_count": make_int(cand.get("referenceCount")),
                    "influential_citation_count": make_int(cand.get("influentialCitationCount")),
                    "is_open_access": cand.get("isOpenAccess"),
                    "open_access_pdf": (cand.get("openAccessPdf") or {}).get("url"),
                    "external_ids": cand.get("externalIds"),
                    "fields_of_study": cand.get("s2FieldsOfStudy"),
                    "relevance_judgement": cand["relevance_judgement"],
                    "n_snippets": len(cand["sentences"]),
                    "snippets": [
                        {
                            "text": s["text"],
                            "section_title": s.get("section_title"),
                            "char_start_offset": s.get("char_start_offset", 0),
                            "score": s.get("score"),
                            "rerank_score": s.get("rerank_score"),
                        }
                        for s in sorted(
                            cand["sentences"], key=lambda x: x.get("char_start_offset", 0)
                        )
                    ],
                    "reference_string": reference_string,
                    "relevance_judgment_input_expanded": prepend_text + section_text,
                }
            )
        return papers


# ---------------------------------------------------------------------------
# Top level entry point
# ---------------------------------------------------------------------------


def retrieve_papers(
    query: str,
    n_retrieval: int = 256,
    n_keyword_srch: int = 20,
    n_rerank: int = 50,
    context_threshold: float = DEFAULT_CONTEXT_THRESHOLD,
    reranker: str = "auto",
    reranker_model: Optional[str] = None,
    reranker_url: Optional[str] = None,
    rerank_batch_size: int = 64,
    rerank_timeout: int = 120,
    rerank_max_doc_tokens: Optional[int] = None,
    decompose: bool = True,
    decomposer_model: str = DEFAULT_CLAUDE_MODEL,
    max_date: Optional[str] = None,
    extra_filters: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run the full ScholarQA retrieval stage and return papers ranked by relevance.

    Returns a dict with keys: query, rewritten_query, keyword_query, search_filters,
    n_snippets, n_keyword_papers, n_reranked_passages, papers.
    """
    t0 = time.time()

    if decompose:
        llm_query = decompose_query(query, model=decomposer_model)
    else:
        llm_query = LLMProcessedQuery(
            rewritten_query=query, keyword_query=query, search_filters={}, usage=None,
        )

    search_filters = dict(llm_query.search_filters)
    if extra_filters:
        search_filters.update({k: v for k, v in extra_filters.items() if v})

    # Hoisted out of the PaperFinder call so the choice can be reported back to the
    # caller: whether reranking actually ran is otherwise invisible downstream.
    reranker_obj = build_reranker(
        reranker,
        model_name=reranker_model,
        url=reranker_url,
        batch_size=rerank_batch_size,
        timeout=rerank_timeout,
        max_doc_tokens=rerank_max_doc_tokens,
    )
    paper_finder = PaperFinder(
        retriever=FullTextRetriever(n_retrieval=n_retrieval, n_keyword_srch=n_keyword_srch),
        reranker=reranker_obj,
        context_threshold=context_threshold,
        n_rerank=n_rerank,
        max_date=max_date,
    )

    # 2. full text snippet search on the rewritten query
    snippet_results = paper_finder.retrieve_passages(
        llm_query.rewritten_query, **search_filters
    )
    snippet_corpus_ids = {s["corpus_id"] for s in snippet_results}
    logger.info("Retrieved %d passages from snippet search", len(snippet_results))

    # 3. keyword search on the keyword query, minus anything already retrieved
    if llm_query.keyword_query:
        search_api_results = paper_finder.retrieve_additional_papers(
            llm_query.keyword_query, **search_filters
        )
        search_api_results = [
            r for r in search_api_results if r["corpus_id"] not in snippet_corpus_ids
        ]
    else:
        search_api_results = []
    logger.info("Retrieved %d additional papers from keyword search", len(search_api_results))

    # keyword-search hits already carry metadata
    paper_metadata = {r["corpus_id"]: dict(r) for r in search_api_results}

    # 4. rerank passages + abstracts together against the ORIGINAL user query
    candidates = snippet_results + search_api_results
    reranked = paper_finder.rerank(query, candidates)

    # 5. metadata for whatever the keyword search did not cover
    remaining_ids = {s["corpus_id"] for s in reranked if s["corpus_id"] not in paper_metadata}
    if remaining_ids:
        paper_metadata.update(get_paper_metadata(sorted(remaining_ids)))

    # 6. aggregate to the paper level
    papers = paper_finder.aggregate_into_papers(reranked, paper_metadata)
    logger.info("Found %d papers in %.2fs", len(papers), time.time() - t0)

    return {
        "query": query,
        "rewritten_query": llm_query.rewritten_query,
        "keyword_query": llm_query.keyword_query,
        "search_filters": search_filters,
        "n_snippets": len(snippet_results),
        "n_keyword_papers": len(search_api_results),
        "n_reranked_passages": len(reranked),
        # None when nothing reranked (kind 'none', or 'auto' with no URL configured):
        # unreranked keyword papers keep score 0.0 and sort last, which biases a
        # downstream judge toward "the literature does not cover X".
        "reranker": ({"model": getattr(reranker_obj, "model", None),
                      "endpoint": getattr(reranker_obj, "endpoint", None)}
                     if reranker_obj is not None else None),
        "papers": papers,
        "elapsed_s": round(time.time() - t0, 2),
        # Token usage of the decomposition call (None when --no-decompose, or when the
        # call failed before returning), so callers can fold it into their own billing.
        "decompose_usage": llm_query.usage,
        "decomposer_model": decomposer_model if decompose else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve relevant papers via S2 snippet + keyword search with reranking "
                    "(port of ai2-scholarqa-lib's rag pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", help="the research question")
    parser.add_argument("--n-retrieval", type=int, default=256, help="snippet search limit")
    parser.add_argument("--n-keyword", type=int, default=20, help="keyword (paper/search) limit")
    parser.add_argument("--n-rerank", type=int, default=50,
                        help="keep this many passages after reranking (<=0 keeps all)")
    parser.add_argument("--context-threshold", type=float, default=DEFAULT_CONTEXT_THRESHOLD,
                        help="drop papers whose best passage score is below this "
                             "(0 = keep everything, as ScholarQA runs it)")
    parser.add_argument("--reranker", default="auto", choices=["auto", "none", "vllm"],
                        help="auto = remote vllm if a URL is configured, else none")
    parser.add_argument("--reranker-model", default=None,
                        help="model name as served by vLLM (default: {})".format(DEFAULT_VLLM_RERANKER))
    parser.add_argument("--reranker-url", default=None,
                        help="base URL of the vLLM reranker, e.g. http://gpu-host:8000 "
                             "(env: VLLM_RERANK_URL)")
    parser.add_argument("--rerank-batch-size", type=int, default=64,
                        help="passages per rerank HTTP request")
    parser.add_argument("--rerank-timeout", type=int, default=120,
                        help="per-request timeout in seconds for the rerank server")
    parser.add_argument("--rerank-max-doc-tokens", type=int, default=None,
                        help="truncate each passage to this many tokens server-side "
                             "(maps to max_tokens_per_doc)")
    parser.add_argument("--no-decompose", action="store_true",
                        help="skip the Claude query decomposition/rewriting step")
    parser.add_argument("--decomposer-model", default=DEFAULT_CLAUDE_MODEL,
                        help="Claude model used for query decomposition")
    parser.add_argument("--max-date", default=None,
                        help="only papers inserted/published before this date, YYYY-MM")
    parser.add_argument("--year", default=None, help="override year filter, e.g. 2020-2025")
    parser.add_argument("--venue", default=None, help="override venue filter, comma separated")
    parser.add_argument("--fields-of-study", default=None,
                        help="override fieldsOfStudy filter, e.g. Computer Science")
    parser.add_argument("--out", default=None, help="write JSON results to this path")
    parser.add_argument("--quiet", action="store_true", help="suppress progress logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if not os.getenv("S2_API_KEY"):
        logger.warning("S2_API_KEY is not set -- expect aggressive rate limiting from the S2 API.")

    extra_filters = {}
    if args.year:
        extra_filters["year"] = args.year
    if args.venue:
        extra_filters["venue"] = args.venue
    if args.fields_of_study:
        extra_filters["fieldsOfStudy"] = args.fields_of_study

    result = retrieve_papers(
        args.query,
        n_retrieval=args.n_retrieval,
        n_keyword_srch=args.n_keyword,
        n_rerank=args.n_rerank,
        context_threshold=args.context_threshold,
        reranker=args.reranker,
        reranker_model=args.reranker_model,
        reranker_url=args.reranker_url,
        rerank_batch_size=args.rerank_batch_size,
        rerank_timeout=args.rerank_timeout,
        rerank_max_doc_tokens=args.rerank_max_doc_tokens,
        decompose=not args.no_decompose,
        decomposer_model=args.decomposer_model,
        max_date=args.max_date,
        extra_filters=extra_filters,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote %s", args.out)

    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
