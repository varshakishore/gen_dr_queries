#!/usr/bin/env python3
"""
Answer a research question with an OpenAI model + the Responses API's web_search tool,
restricted (enforced, filters.allowed_domains) to arxiv.org and semanticscholar.org, and
return it in the shape the rest of the pipeline expects from a research server:

    {"answer", "trace", "model", "usage", "cost_usd"}

What the API returns, and how it is converted (checked on openai_probe.json):
  - web_search_call items: action "search" (several queries per call, plus the sources it
    returned, requested with include=web_search_call.action.sources), "open_page" or
    "find_in_page" (a URL). Each becomes a tool call in the trace;
  - the answer cites inline as markdown links -- "([arxiv.org](https://arxiv.org/abs/...))" --
    and each url_citation annotation (character offsets, with the paper's title) covers
    exactly that link. The link is replaced by a <cite id="c-<n>"></cite> marker, which
    cite_utils renders as " [n]" with the title and URL in the references; an annotation
    that does not cover a link keeps its text and gets the marker at its end;
  - no evidence text comes back for sources, so references carry title and URL only (as
    for Gemini).

The domain audit checks every returned source, opened page and cited URL against the
allowed domains, so the enforced restriction is verified per answer.

Used by answer_cross_systems.py for a systems-file entry with "kind": "openai":

  {"name": "gpt56_sol_s2", "kind": "openai", "model": "gpt-5.6-sol",
   "allowed_domains": ["arxiv.org", "semanticscholar.org"], "max_searches": 20,
   "effort": "high", "timeout": 1800, "concurrency": 4, "price_per_1k_searches": 10,
   "search_scope": "openai web_search, filters.allowed_domains arxiv.org + semanticscholar.org"}

Standalone:
  python openai_search.py "What is ...?" --out gpt_test.json
  python openai_search.py x --from-raw openai_probe.json --out gpt_conv.json   # convert only
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import llm_client

DEFAULT_DOMAINS = ["arxiv.org", "semanticscholar.org"]
SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's research question using the web "
    "search tool to find and read the relevant scientific literature. Search as much as the "
    "question needs, base every substantive claim on what you found, and cite your sources. Say "
    "plainly where the evidence is missing, mixed or does not support a confident conclusion. "
    "Write the final answer as a self-contained report.")
LINK_RE = re.compile(r"\s*\(?\s*\[[^\]]*\]\([^)]*\)\s*\)?\s*")     # "([arxiv.org](url))"


def system_prompt(conf: dict) -> str:
    """Same base prompt as Claude's, plus a sentence stating the (enforced) restriction."""
    prompt = conf.get("system_prompt", SYSTEM_PROMPT)
    domains = conf.get("allowed_domains", DEFAULT_DOMAINS)
    if not conf.get("state_domains", True) or not domains:
        return prompt
    listed = domains[0] if len(domains) == 1 else ", ".join(domains[:-1]) + " and " + domains[-1]
    return f"{prompt} Your web search tool only returns results from {listed}."


def on_domain(url: str, domains) -> bool:
    h = urlparse(url or "").netloc.lower().split(":")[0]
    return any(h == d or h.endswith("." + d) for d in domains)


def norm(url: str) -> str:
    """A matching key that ignores scheme, host case, a trailing slash, the fragment and
    tracking parameters (OpenAI appends ?utm_source=openai to some cited URLs)."""
    from urllib.parse import parse_qsl, urlencode
    p = urlparse((url or "").strip())
    q = urlencode([(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")])
    return f"{p.netloc.lower()}{p.path.rstrip('/')}" + (f"?{q}" if q else "")


def to_trace(raw: dict, domains=DEFAULT_DOMAINS) -> tuple[str, dict]:
    """(answer with <cite> markers, DR-Tulu-shaped trace) from a Responses API response."""
    out = raw.get("output") or []
    last_tool = max((k for k, o in enumerate(out) if o.get("type") == "web_search_call"), default=-1)
    calls, opened, interim, parts = [], [], [], []
    for k, o in enumerate(out):
        if o.get("type") == "web_search_call":
            a = o.get("action") or {}
            kind = a.get("type") or "search"
            if kind == "search":
                qs = a.get("queries") or ([a["query"]] if a.get("query") else [])
                calls.append({"call_id": f"s{len(calls) + 1}", "tool_name": "web_search",
                              "query": " | ".join(qs), "queries": qs, "status": o.get("status"),
                              "documents": [{"title": "", "url": s.get("url") or "", "snippet": ""}
                                            for s in a.get("sources") or [] if isinstance(s, dict)],
                              "error": None if o.get("status") in (None, "completed") else o.get("status")})
            else:                                        # open_page / find_in_page
                opened.append(a.get("url") or "")
                calls.append({"call_id": f"s{len(calls) + 1}", "tool_name": kind,
                              "query": a.get("pattern") or "", "url": a.get("url") or "",
                              "documents": [], "status": o.get("status"), "error": None})
        elif o.get("type") == "message":
            for c in o.get("content") or []:
                if c.get("type") == "output_text":
                    (parts if k > last_tool else interim).append(c)
                elif c.get("type") == "refusal":
                    interim.append({"text": f"[refusal] {c.get('refusal', '')}"})

    docs, by_url, answer = [], {}, []
    for c in parts:
        text = c.get("text") or ""
        edits = []                                       # (start, end, replacement)
        for a in c.get("annotations") or []:
            if a.get("type") != "url_citation" or not a.get("url"):
                continue
            key = norm(a["url"])
            if key not in by_url:
                by_url[key] = len(docs)
                clean = re.sub(r"[?&]utm_[^&#]*", "", a["url"]).rstrip("?&")
                docs.append({"title": a.get("title") or clean, "url": clean, "cited_url": a["url"],
                             "snippet": ""})
            s0, s1 = int(a.get("start_index", 0)), int(a.get("end_index", 0))
            marker = f'<cite id="c-{by_url[key]}"></cite>'
            if LINK_RE.fullmatch(text[s0:s1] or "x"):   # the inline markdown link itself
                edits.append((s0, s1, marker))
            else:                                        # claim text: keep it, mark its end
                edits.append((s1, s1, marker))
        for s0, s1, rep in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
            text = text[:s0] + rep + text[s1:]
        answer.append(text)
    calls.append({"call_id": "c", "tool_name": "citation", "query": "", "documents": docs, "error": None})

    sources = [d["url"] for c in calls if c["tool_name"] == "web_search" for d in c["documents"]]
    src_keys = {norm(u) for u in sources}
    audit = {
        "searches": sum(c["tool_name"] == "web_search" for c in calls),
        "queries": sum(len(c.get("queries") or []) for c in calls),
        "search_results": len(sources),
        "off_domain_results": sorted({u for u in sources if not on_domain(u, domains)}),
        "pages_opened": len(opened),
        "off_domain_opened": sorted({u for u in opened if u and not on_domain(u, domains)}),
        "citations": sum(1 for c in parts for a in c.get("annotations") or [] if a.get("type") == "url_citation"),
        "cited_sources": len(docs),
        "off_domain_citations": sorted({d["url"] for d in docs if not on_domain(d["url"], domains)}),
        "cited_not_in_results": sorted({d["url"] for d in docs if norm(d["url"]) not in src_keys}),
    }
    return "".join(answer).strip(), {"tool_calls": calls,
                                     "interim_text": [c.get("text") for c in interim],
                                     "domain_audit": audit}


def price(model: str, conf: dict, usage: dict) -> float:
    rates = llm_client.MODEL_PRICING.get(model, {})
    p_in, p_out = conf.get("price_input", rates.get("input")), conf.get("price_output", rates.get("output"))
    searches = usage.get("searches", 0) * conf.get("price_per_1k_searches", 10.0) / 1000
    if p_in is None or p_out is None:
        return searches
    cached = usage.get("cached_tokens", 0)
    return ((usage.get("input_tokens", 0) - cached) * p_in
            + cached * p_in * llm_client.CACHE_READ_MULTIPLIER
            + usage.get("output_tokens", 0) * p_out) / 1e6 + searches     # output includes reasoning


def is_priced(conf: dict) -> bool:
    rates = llm_client.MODEL_PRICING.get(conf["model"], {})
    return (conf.get("price_input", rates.get("input")) is not None
            and conf.get("price_output", rates.get("output")) is not None)


def query_gpt(question: str, conf: dict, client=None) -> dict:
    """Answer `question` per `conf` (see module docstring). Raises on API failure."""
    from openai import OpenAI
    model = conf["model"]
    client = client or OpenAI(timeout=conf.get("timeout", 1800), max_retries=3)
    domains = conf.get("allowed_domains", DEFAULT_DOMAINS)
    kwargs = {"model": model, "instructions": system_prompt(conf), "input": question,
              "tools": [{"type": "web_search", "filters": {"allowed_domains": domains}}],
              "include": ["web_search_call.action.sources"],
              "max_tool_calls": conf.get("max_searches", 20)}
    if conf.get("effort"):
        kwargs["reasoning"] = {"effort": conf["effort"]}
    t0 = time.time()
    raw = client.responses.create(**kwargs).model_dump(mode="json")
    answer, trace = to_trace(raw, domains)
    u = raw.get("usage") or {}
    n_search = ((raw.get("tool_usage") or {}).get("web_search") or {}).get("num_requests")
    usage = {"input_tokens": u.get("input_tokens", 0),
             "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
             "output_tokens": u.get("output_tokens", 0),
             "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
             "searches": n_search if n_search is not None else trace["domain_audit"]["searches"]}
    status = raw.get("status")
    trace.update({"raw_output_types": dict(Counter(o.get("type") for o in raw.get("output") or [])),
                  "status": status, "incomplete_details": raw.get("incomplete_details"),
                  "complete": status == "completed", "response_id": raw.get("id"),
                  "system_prompt": kwargs["instructions"], "allowed_domains": domains,
                  "restriction": "enforced (filters.allowed_domains)",
                  "max_searches": kwargs["max_tool_calls"], "reasoning": raw.get("reasoning"),
                  "tool_call_count": trace["domain_audit"]["searches"],
                  "elapsed_s": round(time.time() - t0, 1)})
    if not answer:
        raise ValueError(f"empty answer (status {status}; {trace['interim_text'][:1]})")
    if status != "completed":
        print(f"[warn] incomplete answer: status {status} {raw.get('incomplete_details')}", file=sys.stderr)
    return {"answer": answer, "trace": trace, "model": model, "usage": usage,
            "cost_usd": round(price(model, conf, usage), 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\nUsed by")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-searches", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("gpt_test.json"))
    ap.add_argument("--from-raw", type=Path, help="Convert a saved raw response instead of calling the API.")
    args = ap.parse_args()
    conf = {"model": args.model, "allowed_domains": args.domains, "effort": args.effort,
            "max_searches": args.max_searches}
    if args.from_raw:
        answer, trace = to_trace(json.load(open(args.from_raw)), args.domains)
        res = {"answer": answer, "trace": trace, "model": args.model}
    else:
        if (missing := llm_client.require_api_key(args.model)):
            sys.exit(missing)
        res = query_gpt(args.question, conf)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    import research_pipeline as rp
    text, inline = rp.format_answer_for_judge(res["answer"], res["trace"])
    a = res["trace"]["domain_audit"]
    print(f"searches {a['searches']} ({a['queries']} queries), results {a['search_results']}, pages opened "
          f"{a['pages_opened']} | off-domain: results {len(a['off_domain_results'])}, opened "
          f"{len(a['off_domain_opened'])}, cited {len(a['off_domain_citations'])} | citations {a['citations']} "
          f"-> {a['cited_sources']} sources (not among results: {len(a['cited_not_in_results'])})"
          + (f"\ncomplete {res['trace'].get('complete')} | {res['trace'].get('elapsed_s')}s | "
             f"${res['cost_usd']:.3f} | usage {res['usage']}" if "usage" in res else "")
          + f"\ninline cites {inline} | judge rendering {len(text)} chars -> {args.out}\n")
    print(text[:2500])


if __name__ == "__main__":
    main()
