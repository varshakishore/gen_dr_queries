#!/usr/bin/env python3
"""
Answer a research question with Gemini + Google Search (Interactions API), INSTRUCTED --
not enforced -- to use only arxiv.org and semanticscholar.org, and return it in the shape
the rest of the pipeline expects from a research server:

    {"answer", "trace", "model", "usage", "cost_usd"}

Gemini's search tool has no domain allow-list, so the restriction lives in the system
instruction (search with site: filters, cite only those sites); the trace audits how well
it was followed (queries without a site: filter, cited URLs off the allowed domains).

What Gemini returns, and how it is converted (checked on gemini_probe.json):
  - search steps carry the queries but NO results (only Google's suggestion widget), so the
    sources are known only from the answer's citations;
  - citations are url_citation annotations whose start/end are UTF-8 BYTE offsets into the
    answer, often several sources per span and spans overlapping. Each is turned into a
    <cite id="g-<n> ..."></cite> marker at the END of its span (cite_utils renders it " [n]"),
    so overlap needs no special handling;
  - citation URLs are Google redirect links whose `title` is just the domain. Each is
    resolved (one request, no redirect following) to the real page URL, and the paper's
    title is looked up -- one Semantic Scholar batch call (arXiv ids as ARXIV:<id>; set
    S2_API_KEY, the public rate limit is tight), the arXiv API as a fallback, with backoff --
    so references are identifiable. Documents carry NO snippet: Gemini does
    not say what text it used, and an abstract would be a guess at its evidence.

Used by answer_cross_systems.py for a systems-file entry with "kind": "gemini":

  {"name": "gemini_flash_prompted_s2", "kind": "gemini", "model": "gemini-3.8-flash",
   "allowed_domains": ["arxiv.org", "semanticscholar.org"], "timeout": 900,
   "concurrency": 4, "price_per_1k_searches": 14,
   "search_scope": "google_search, unrestricted; instructed to use only ..."}

Standalone, for one question:
  python gemini_search.py "What is ...?" --out gemini_test.json
"""

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests

import llm_client

DEFAULT_DOMAINS = ["arxiv.org", "semanticscholar.org"]
REDIRECT_HOST = "vertexaisearch.cloud.google.com"
ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})")
S2_ID_RE = re.compile(r"semanticscholar\.org/(?:paper|reader)/(?:[^/]+/)?([0-9a-f]{40})")
S2_PDF_RE = re.compile(r"pdfs\.semanticscholar\.org/([0-9a-f]{4})/([0-9a-f]{36})")


def system_instruction(conf: dict) -> str:
    domains = conf.get("allowed_domains", DEFAULT_DOMAINS)
    listed = " and ".join(domains) if len(domains) <= 2 else ", ".join(domains[:-1]) + " and " + domains[-1]
    sites = " OR ".join(f"site:{d}" for d in domains)
    return conf.get("system_prompt") or (
        "You are a careful research assistant. Answer the user's research question using the "
        f"Google Search tool to find and read the relevant scientific literature. Use and cite ONLY "
        f"results from {listed}: add \"{sites}\" to your searches and ignore results from any "
        "other website. Search as much as the question needs, base every substantive claim on "
        "what you found, and cite your sources. Say plainly where the evidence is missing, mixed "
        "or does not support a confident conclusion. Write the final answer as a self-contained "
        "report.")


def on_domain(url: str, domains) -> bool:
    host = urlparse(url or "").netloc.lower().split(":")[0]
    return any(host == d or host.endswith("." + d) for d in domains)


# ---------------------------------------------------------------------------
# Citation sources: resolve redirects, look up titles
# ---------------------------------------------------------------------------

_redirects, _titles = {}, {}                    # process-wide caches (answers share papers)
_s2_lock, _s2_last = threading.Lock(), [0.0]


def resolve(url: str) -> str | None:
    """The real page behind a Google grounding redirect link (None if it will not say)."""
    if REDIRECT_HOST not in (urlparse(url).netloc or ""):
        return url
    if url in _redirects:
        return _redirects[url]
    final = None
    for attempt in range(3):
        try:
            r = requests.get(url, allow_redirects=False, timeout=20)
            final = r.headers.get("Location")
            break
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    _redirects[url] = final
    return final


def _get_with_backoff(do, tries=4, base=3.0):
    """Call do() -> requests.Response, retrying on 429 / 5xx / network errors with backoff."""
    for k in range(tries):
        try:
            r = do()
            if r.status_code != 429 and r.status_code < 500:
                return r
        except requests.RequestException:
            pass
        time.sleep(base * (2 ** k))
    return None


def paper_ids(url: str) -> tuple[str | None, str | None]:
    """(arXiv id, Semantic Scholar paper id) recognisable in a resolved URL."""
    m = ARXIV_ID_RE.search(url or "")
    if m:
        return m.group(1), None
    m = S2_ID_RE.search(url or "") or S2_PDF_RE.search(url or "")
    if m:
        return None, "".join(m.groups())
    return None, None


def lookup_titles(docs: list) -> None:
    """Fill in real paper titles, in place: one Semantic Scholar batch call for every
    recognisable id (arXiv ids as ARXIV:<id>), then the arXiv API for arXiv ids S2 missed.
    Both are rate limited (S2 without S2_API_KEY especially); a doc whose title cannot be
    found keeps its URL as the title."""
    import os
    keyed = {}                                       # doc index -> S2 batch id
    for i, d in enumerate(docs):
        ax, s2 = paper_ids(d["url"])
        if ax or s2:
            keyed[i] = f"ARXIV:{ax}" if ax else s2
            d["arxiv_id"], d["s2_id"] = ax, s2
    want = [x for x in dict.fromkeys(keyed.values()) if x not in _titles]
    if want:
        with _s2_lock:                               # ~1 request/s across worker threads
            wait = 1.1 - (time.time() - _s2_last[0])
            if wait > 0:
                time.sleep(wait)
            _s2_last[0] = time.time()
        headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
        r = _get_with_backoff(lambda: requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch", params={"fields": "title"},
            json={"ids": want[:500]}, headers=headers, timeout=30))
        if r is not None and r.ok:
            for pid, paper in zip(want, r.json()):
                if paper and paper.get("title"):
                    _titles[pid] = (paper["title"], "s2_api")
    missing_ax = [x[6:] for x in dict.fromkeys(keyed.values())
                  if x.startswith("ARXIV:") and x not in _titles]
    if missing_ax:
        r = _get_with_backoff(lambda: requests.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": ",".join(missing_ax), "max_results": len(missing_ax)}, timeout=30), base=5.0)
        if r is not None and r.ok:
            try:
                ns = {"a": "http://www.w3.org/2005/Atom"}
                for e in ET.fromstring(r.text).findall("a:entry", ns):
                    base = re.sub(r"v\d+$", "", (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1])
                    title = " ".join((e.findtext("a:title", "", ns) or "").split())
                    if base and title and title != "Error":
                        _titles[f"ARXIV:{base}"] = (title, "arxiv_api")
            except ET.ParseError:
                pass
    for i, pid in keyed.items():
        if pid in _titles:
            docs[i]["title"], docs[i]["title_source"] = _titles[pid]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def byte_to_char(text: str):
    """f(byte offset) -> char offset, for UTF-8 text."""
    starts = []                                     # byte offset at which each char starts
    pos = 0
    for ch in text:
        starts.append(pos)
        pos += len(ch.encode("utf-8"))
    starts.append(pos)

    def f(b: int) -> int:
        import bisect
        return bisect.bisect_right(starts, b) - 1
    return f


def to_trace(raw: dict, domains=DEFAULT_DOMAINS) -> tuple[str, dict]:
    """(answer with <cite> markers, DR-Tulu-shaped trace) from a Gemini interaction."""
    steps = raw.get("steps") or []
    calls, interim = [], []
    last_search = max((k for k, s in enumerate(steps) if s.get("type") == "google_search_call"),
                      default=-1)
    texts = []                                      # (text, annotations) of the final report
    for k, s in enumerate(steps):
        if s.get("type") == "google_search_call":
            args = s.get("arguments") or {}
            qs = args.get("queries") if isinstance(args, dict) else None
            calls.append({"call_id": f"q{len(calls) + 1}", "tool_name": "google_search",
                          "query": " | ".join(qs or []), "queries": qs or [], "documents": [],
                          "error": None})
        elif s.get("type") == "google_search_result" and str(s.get("is_error")).lower() == "true":
            if calls:
                calls[-1]["error"] = "is_error"
        elif s.get("type") == "model_output":
            for c in s.get("content") or []:
                if c.get("type") == "text":
                    (texts if k > last_search else interim).append(c)

    # sources: one document per real (resolved) URL
    docs, by_url, marks = [], {}, defaultdict(list)     # marks: char end -> doc indices
    answer_parts, offset = [], 0
    for c in texts:
        text = c.get("text") or ""
        to_char = byte_to_char(text)
        for a in c.get("annotations") or []:
            if a.get("type") != "url_citation" or not a.get("url"):
                continue
            real = resolve(a["url"])
            key = real or a["url"]
            if key not in by_url:
                by_url[key] = len(docs)
                docs.append({"title": a.get("title") or "", "url": real or "", "redirect_url": a["url"],
                             "resolved": real is not None, "domain_hint": a.get("title") or "",
                             "snippet": "", "title_source": "domain"})
            i = by_url[key]
            end = offset + to_char(int(a.get("end_index", 0)))
            if i not in marks[end]:
                marks[end].append(i)
        answer_parts.append(text)
        offset += len(text)
    answer = "".join(answer_parts)

    lookup_titles(docs)                             # real paper titles where possible
    for d in docs:
        if d["title_source"] == "domain":
            d["title"] = f"{d['url'] or d['redirect_url']}"

    # markers, inserted from the end so earlier offsets stay valid
    for end in sorted(marks, reverse=True):
        ids = " ".join(f"g-{i}" for i in marks[end])
        answer = answer[:end] + f'<cite id="{ids}"></cite>' + answer[end:]
    calls.append({"call_id": "g", "tool_name": "citation", "query": "", "documents": docs,
                  "error": None})

    queries = [q for c in calls if c["tool_name"] == "google_search" for q in c["queries"]]
    audit = {
        "queries": len(queries),
        "queries_without_site_filter": [q for q in queries
                                        if not any(f"site:{d}" in q for d in domains)],
        "citations": sum(len(v) for v in marks.values()),
        "cited_sources": len(docs),
        "off_domain_citations": sorted({d["url"] for d in docs if d["resolved"] and not on_domain(d["url"], domains)}),
        "unresolved_citations": [d["redirect_url"] for d in docs if not d["resolved"]],
        "titles_found": sum(d["title_source"] != "domain" for d in docs),
    }
    return answer, {"tool_calls": calls, "interim_text": [c.get("text") for c in interim],
                    "domain_audit": audit}


def price(model: str, conf: dict, usage: dict) -> float:
    rates = llm_client.MODEL_PRICING.get(model, {})
    p_in, p_out = conf.get("price_input", rates.get("input")), conf.get("price_output", rates.get("output"))
    searches = usage.get("searches", 0) * conf.get("price_per_1k_searches", 14.0) / 1000
    if p_in is None or p_out is None:
        return searches
    cached = usage.get("cached_tokens", 0)
    tokens = ((usage.get("input_tokens", 0) - cached) * p_in
              + cached * p_in * llm_client.CACHE_READ_MULTIPLIER
              + (usage.get("output_tokens", 0) + usage.get("thought_tokens", 0)) * p_out) / 1e6
    return tokens + searches


def is_priced(conf: dict) -> bool:
    rates = llm_client.MODEL_PRICING.get(conf["model"], {})
    return (conf.get("price_input", rates.get("input")) is not None
            and conf.get("price_output", rates.get("output")) is not None)


def _dump(obj) -> dict:
    try:
        return obj.model_dump(mode="json")
    except (AttributeError, TypeError):
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))


def _strip(raw: dict) -> dict:
    """The raw interaction without the suggestion widgets' HTML and opaque signatures."""
    out = json.loads(json.dumps(raw))
    for s in out.get("steps") or []:
        s.pop("signature", None)
        if s.get("type") == "google_search_result":
            s["result"] = "(search suggestions widget removed)"
    return out


def query_gemini(question: str, conf: dict, client=None) -> dict:
    """Answer `question` per `conf` (see module docstring). Raises on API failure."""
    from google import genai
    from google.genai import types
    model = conf["model"]
    client = client or genai.Client(http_options=types.HttpOptions(timeout=int(conf.get("timeout", 900) * 1000)))
    sysi = system_instruction(conf)
    t0 = time.time()
    raw = _dump(client.interactions.create(model=model, input=question,
                                           tools=[{"type": "google_search"}],
                                           system_instruction=sysi))
    answer, trace = to_trace(raw, conf.get("allowed_domains", DEFAULT_DOMAINS))
    u = raw.get("usage") or {}
    usage = {"input_tokens": u.get("total_input_tokens", 0), "cached_tokens": u.get("total_cached_tokens", 0),
             "output_tokens": u.get("total_output_tokens", 0), "thought_tokens": u.get("total_thought_tokens", 0),
             "searches": sum(g.get("count", 0) for g in u.get("grounding_tool_count") or []
                             if g.get("type") == "google_search")}
    status = raw.get("status")
    trace.update({"raw": _strip(raw), "status": status, "complete": status == "completed",
                  "interaction_id": raw.get("id"), "system_prompt": sysi,
                  "allowed_domains": conf.get("allowed_domains", DEFAULT_DOMAINS),
                  "restriction": "prompted (not enforced)",
                  "tool_call_count": len([c for c in trace["tool_calls"] if c["tool_name"] == "google_search"]),
                  "elapsed_s": round(time.time() - t0, 1)})
    if not answer.strip():
        raise ValueError(f"empty answer (status {status})")
    if status != "completed":
        print(f"[warn] incomplete answer: status {status}", file=sys.stderr)
    return {"answer": answer, "trace": trace, "model": model, "usage": usage,
            "cost_usd": round(price(model, conf, usage), 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\nUsed by")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    ap.add_argument("--out", type=Path, default=Path("gemini_test.json"))
    ap.add_argument("--from-raw", type=Path,
                    help="Convert a saved raw interaction (e.g. gemini_probe.json) instead of calling the API.")
    args = ap.parse_args()
    conf = {"model": args.model, "allowed_domains": args.domains}
    if args.from_raw:
        raw = json.load(open(args.from_raw))
        answer, trace = to_trace(raw, args.domains)
        res = {"answer": answer, "trace": trace, "model": args.model}
    else:
        res = query_gemini(args.question, conf)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    import research_pipeline as rp
    text, inline = rp.format_answer_for_judge(res["answer"], res["trace"])
    a = res["trace"]["domain_audit"]
    print(f"queries {a['queries']} (without site: filter {len(a['queries_without_site_filter'])}) | "
          f"citations {a['citations']} -> {a['cited_sources']} sources, titles found {a['titles_found']} | "
          f"off-domain cited {len(a['off_domain_citations'])} | unresolved {len(a['unresolved_citations'])}"
          + (f"\ncomplete {res['trace'].get('complete')} | {res['trace'].get('elapsed_s')}s | "
             f"${res['cost_usd']:.3f} | usage {res['usage']}" if "usage" in res else "")
          + f"\ninline cites {inline} | judge rendering {len(text)} chars -> {args.out}\n")
    print(text[:3000])


if __name__ == "__main__":
    main()
