#!/usr/bin/env python3
"""
Answer a research question with Claude + Anthropic's server-side web search, restricted to
given domains (default arxiv.org and semanticscholar.org), and return it in the shape the
rest of the pipeline expects from a research server:

    {"answer", "trace", "model", "usage", "cost_usd"}

The trace is DR-Tulu's shape, so cite_utils / format_answer_for_judge render it with no
changes: every web search becomes a tool call (`call_id`, `query`, `documents`), and every
cited span of Claude's answer is wrapped as <cite id="<call_id>-<i> ...">span</cite>, the
ids pointing at the search results it cites. A document's `snippet` is the text Claude
quoted from it (`cited_text`, joined when cited several times): Anthropic returns search
result bodies only as encrypted_content, so the cited passages, titles and URLs are the
readable evidence -- what the judge sees for this system. The raw response blocks (minus
the encrypted payloads) are kept under trace["raw_blocks"].

Used by answer_cross_systems.py for a systems-file entry with "kind": "anthropic":

  {"name": "claude_opus_s2", "kind": "anthropic", "model": "claude-opus-5-5",
   "allowed_domains": ["arxiv.org", "semanticscholar.org"], "max_searches": 20,
   "max_tokens": 32000, "effort": "high", "timeout": 1800, "concurrency": 4,
   "price_input": <$/M>, "price_output": <$/M>, "price_per_1k_searches": 10,
   "search_scope": "anthropic web_search, allowed_domains arxiv.org + semanticscholar.org"}

Prices default to llm_client.MODEL_PRICING; a model missing there needs price_input and
price_output, or its cost is reported as $0 and --budget-usd cannot stop it.

Standalone, for one question:
  python frontier_search.py "What is ...?" --model claude-opus-5-5 --out one.json
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import llm_client

DEFAULT_DOMAINS = ["arxiv.org", "semanticscholar.org"]
SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's research question using the "
    "web search tool to find and read the relevant scientific literature. Search as much as "
    "the question needs, base every substantive claim on what you found, and cite your "
    "sources. Say plainly where the evidence is missing, mixed or does not support a "
    "confident conclusion. Write the final answer as a self-contained report."
)
WEB_SEARCH_TOOL = "web_search_20250305"
MAX_CONTINUATIONS = 10          # pause_turn resumptions before giving up


def _strip_encrypted(x):
    """A response block without its (large, opaque) encrypted fields."""
    if isinstance(x, dict):
        if x.get("type") == "redacted_thinking":
            return {"type": "redacted_thinking"}
        return {k: _strip_encrypted(v) for k, v in x.items()
                if k not in ("encrypted_content", "encrypted_index", "signature")}
    if isinstance(x, list):
        return [_strip_encrypted(v) for v in x]
    return x


def norm_url(url: str) -> str:
    """A URL key that ignores scheme, host case, a trailing slash and the fragment, so a
    citation and the search result it came from match despite cosmetic differences."""
    p = urlparse((url or "").strip())
    return f"{p.netloc.lower()}{p.path.rstrip('/')}" + (f"?{p.query}" if p.query else "")


def on_domain(url: str, domains) -> bool:
    host = urlparse(url or "").netloc.lower().split(":")[0]
    return any(host == d or host.endswith("." + d) for d in domains)


def to_trace(blocks: list, domains=DEFAULT_DOMAINS) -> tuple[str, dict]:
    """(answer with <cite> tags, DR-Tulu-shaped trace) from Claude's content blocks.

    The answer is the text AFTER the last search -- Claude's final report, like DR-Tulu's
    <answer> block; any notes it wrote between searches ("Let me look for...") go to
    trace["interim_text"] instead. Only cited documents carry a snippet (the passages
    Claude quoted); uncited ones show title and URL only, since Anthropic returns result
    bodies encrypted -- so the judge has less evidence per source than for DR-Tulu.
    """
    calls, by_tool_id, url_to_ref = [], {}, {}
    for b in blocks:
        if b.get("type") == "server_tool_use" and b.get("name") == "web_search":
            call = {"call_id": f"s{len(calls) + 1}", "tool_name": "web_search",
                    "query": (b.get("input") or {}).get("query", ""), "documents": [],
                    "error": None}
            calls.append(call)
            by_tool_id[b.get("id")] = call
        elif b.get("type") == "web_search_tool_result":
            call = by_tool_id.get(b.get("tool_use_id"))
            if call is None:
                continue
            content = b.get("content")
            if isinstance(content, dict):          # web_search_tool_result_error
                call["error"] = content.get("error_code") or "error"
                continue
            for r in content or []:
                if r.get("type") != "web_search_result" or not r.get("url"):
                    continue
                i = len(call["documents"])
                call["documents"].append({"title": r.get("title") or "", "url": r["url"],
                                          "page_age": r.get("page_age"), "snippet": ""})
                url_to_ref.setdefault(norm_url(r["url"]), (call, i))

    last_tool = max((k for k, b in enumerate(blocks)
                     if b.get("type") in ("server_tool_use", "web_search_tool_result")), default=-1)
    cited = defaultdict(list)                       # cite id -> cited passages
    extra = None                                    # citations to URLs no search returned
    parts, interim, cited_urls = [], [], []
    for k, b in enumerate(blocks):
        if b.get("type") != "text":
            continue
        text = b.get("text") or ""
        if k < last_tool:                           # narration between searches
            interim.append(text)
            continue
        ids = []
        for c in b.get("citations") or []:
            url = c.get("url")
            if not url:
                continue
            key = norm_url(url)
            if key not in url_to_ref:
                if extra is None:
                    extra = {"call_id": "cited", "tool_name": "citation", "query": "",
                             "documents": [], "error": None}
                url_to_ref[key] = (extra, len(extra["documents"]))
                extra["documents"].append({"title": c.get("title") or "", "url": url,
                                           "page_age": None, "snippet": ""})
            call, i = url_to_ref[key]
            cid = f"{call['call_id']}-{i}"
            if cid not in ids:
                ids.append(cid)
            cited_urls.append(url)
            quote = (c.get("cited_text") or "").strip()
            if quote and quote not in cited[cid]:
                cited[cid].append(quote)
        parts.append(f'<cite id="{" ".join(ids)}">{text}</cite>' if ids and text.strip() else text)
    if extra is not None:
        calls.append(extra)
    for call in calls:
        for i, d in enumerate(call["documents"]):
            d["snippet"] = " … ".join(cited.get(f"{call['call_id']}-{i}", []))

    results = [d["url"] for c in calls if c["tool_name"] == "web_search" for d in c["documents"]]
    audit = {
        "search_results": len(results),
        "off_domain_results": sorted({u for u in results if not on_domain(u, domains)}),
        "citations": len(cited_urls),
        "off_domain_citations": sorted({u for u in cited_urls if not on_domain(u, domains)}),
        # cited URLs no search returned: a URL-matching problem, not a search leak
        "cited_not_in_results": [d["url"] for d in (extra or {}).get("documents", [])],
    }
    return "".join(parts).strip(), {"tool_calls": calls, "interim_text": interim,
                                    "domain_audit": audit}


def price(model: str, conf: dict, usage: dict) -> float:
    rates = llm_client.MODEL_PRICING.get(model, {})
    p_in = conf.get("price_input", rates.get("input"))
    p_out = conf.get("price_output", rates.get("output"))
    searches = usage["web_search_requests"] * conf.get("price_per_1k_searches", 10.0) / 1000
    if p_in is None or p_out is None:
        return searches                   # token prices unknown; the search fee still counts
    tokens = (usage["input_tokens"] * p_in
              + usage["cache_creation_input_tokens"] * p_in * llm_client.CACHE_WRITE_MULTIPLIER
              + usage["cache_read_input_tokens"] * p_in * llm_client.CACHE_READ_MULTIPLIER
              + usage["output_tokens"] * p_out) / 1e6
    return tokens + searches


def is_priced(conf: dict) -> bool:
    rates = llm_client.MODEL_PRICING.get(conf["model"], {})
    return (conf.get("price_input", rates.get("input")) is not None
            and conf.get("price_output", rates.get("output")) is not None)


def system_prompt(conf: dict) -> str:
    """The system prompt, plus a sentence stating the search restriction -- built from the
    same allowed_domains the tool enforces, so the two cannot drift apart. Set
    "state_domains": false in the entry to leave the restriction unstated."""
    prompt = conf.get("system_prompt", SYSTEM_PROMPT)
    domains = conf.get("allowed_domains", DEFAULT_DOMAINS)
    if not conf.get("state_domains", True) or not domains:
        return prompt
    listed = domains[0] if len(domains) == 1 else ", ".join(domains[:-1]) + " and " + domains[-1]
    return f"{prompt} Your web search tool only returns results from {listed}."


def query_claude(question: str, conf: dict, client=None) -> dict:
    """Answer `question` per `conf` (see module docstring). Raises on API failure."""
    from anthropic import Anthropic
    model = conf["model"]
    client = client or Anthropic(timeout=conf.get("timeout", 1800), max_retries=3)
    # search_tool: web_search_20250305 (basic) or web_search_20260209 / _20260318, where Claude
    # filters results with code before reading them ("dynamic filtering"). response_inclusion
    # is left at its default "full": "excluded" would drop the results code consumed, and
    # with them the domain audit.
    tool = {"type": conf.get("search_tool", WEB_SEARCH_TOOL), "name": "web_search",
            "max_uses": conf.get("max_searches", 20),
            "allowed_domains": conf.get("allowed_domains", DEFAULT_DOMAINS)}
    if conf.get("allowed_callers"):                 # e.g. ["direct"] to disable filtering
        tool["allowed_callers"] = conf["allowed_callers"]
    kwargs = {"model": model, "max_tokens": conf.get("max_tokens", 32000),
              "system": system_prompt(conf), "tools": [tool]}
    # Opus 5.5 rejects a fixed thinking budget ({"type": "enabled", "budget_tokens"}): it
    # takes adaptive thinking, steered by output_config.effort (needs anthropic SDK >= 1.x).
    if conf.get("effort"):
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": conf["effort"]}

    base = [{"role": "user", "content": question}]
    messages, so_far = base, []                     # so_far: the unfinished assistant turn
    blocks, stops = [], []
    repaired, delta_kinds = [], Counter()       # stream-vs-final-message text check
    usage = defaultdict(int)
    t0 = time.time()
    for _ in range(MAX_CONTINUATIONS + 1):
        # streamed, so a long turn (many searches, high effort) is not one fragile request.
        # Every text_delta is also collected here: under anthropic 1.13 a full-settings run
        # came back with all CITED text blocks empty (a small run did not reproduce it), so
        # the final message is checked against what actually streamed in, and repaired.
        streamed = defaultdict(list)
        with client.messages.stream(messages=messages, **kwargs) as stream:
            for ev in stream:
                if getattr(ev, "type", None) == "content_block_delta":
                    delta_kinds[getattr(ev.delta, "type", "?")] += 1
                    if getattr(ev.delta, "type", None) == "text_delta":
                        streamed[ev.index].append(ev.delta.text)
            resp = stream.get_final_message()
        for i, blk in enumerate(resp.content):
            got = "".join(streamed.get(i, []))
            if getattr(blk, "type", None) == "text" and got and got != (blk.text or ""):
                repaired.append({"request": len(stops), "block": i,
                                 "final_message_chars": len(blk.text or ""), "streamed_chars": len(got),
                                 "cited": bool(getattr(blk, "citations", None))})
                blk.text = got
        u = resp.usage
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                  "cache_read_input_tokens"):
            usage[k] += getattr(u, k, 0) or 0
        stu = getattr(u, "server_tool_use", None)
        usage["web_search_requests"] += (getattr(stu, "web_search_requests", 0) or 0) if stu else 0
        blocks += [b.model_dump() for b in resp.content]
        stops.append(resp.stop_reason)
        if resp.stop_reason != "pause_turn":
            break
        # A paused server-tool turn is ONE unfinished assistant turn: resend all of it so far
        # as a single assistant message (thinking blocks and signatures unchanged), never as
        # several assistant messages in a row.
        so_far = so_far + list(resp.content)
        messages = base + [{"role": "assistant", "content": so_far}]
    answer, trace = to_trace(blocks, tool["allowed_domains"])
    usage = dict(usage)
    final = stops[-1]
    cap = tool["max_uses"]
    trace.update({"raw_blocks": _strip_encrypted(blocks), "stop_reasons": stops,
                  # end_turn = finished. max_tokens (thinking shares the budget), a
                  # pause_turn left after MAX_CONTINUATIONS, or a refusal = a partial report
                  "complete": final == "end_turn", "final_stop_reason": final,
                  "tool_call_count": sum(1 for c in trace["tool_calls"] if c["tool_name"] == "web_search"),
                  "max_searches": cap,
                  # max_uses is per request; whether searches in earlier continuations count
                  # toward it is undocumented, so flag (do not silently allow) a run past it
                  "searches_over_cap": usage.get("web_search_requests", 0) > cap,
                  # text blocks whose final-message text differed from the streamed deltas
                  # (repaired from the deltas), and the delta types seen, for diagnosis
                  "stream_text_repaired": repaired, "stream_delta_kinds": dict(delta_kinds),
                  "allowed_domains": tool["allowed_domains"], "search_tool": tool["type"],
                  "system_prompt": kwargs["system"],
                  "elapsed_s": round(time.time() - t0, 1)})
    if not answer:
        raise ValueError(f"empty answer (stop reasons {stops})")
    if final != "end_turn":
        print(f"[warn] incomplete answer: final stop_reason {final}", file=sys.stderr)
    if repaired:
        print(f"[warn] {len(repaired)} text block(s) differed from the streamed text; repaired "
              f"({sum(r['cited'] for r in repaired)} cited)", file=sys.stderr)
    empty_cited = sum(1 for b in blocks if b.get("type") == "text" and b.get("citations")
                      and not (b.get("text") or "").strip())
    trace["empty_cited_blocks"] = empty_cited
    if empty_cited:
        print(f"[warn] {empty_cited} cited text block(s) have no text even in the stream -- the "
              f"cited sentences are missing from this answer", file=sys.stderr)
    if trace["searches_over_cap"]:
        print(f"[warn] {usage['web_search_requests']} searches > max_searches {cap}", file=sys.stderr)
    return {"answer": answer, "trace": trace, "model": model, "usage": usage,
            "cost_usd": round(price(model, conf, usage), 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\nUsed by")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    ap.add_argument("--max-searches", type=int, default=20)
    ap.add_argument("--search-tool", default=WEB_SEARCH_TOOL,
                    help="web_search_20250305 (basic) or web_search_20260318 (dynamic filtering)")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--effort", default="high",
                    help="Adaptive-thinking effort (output_config.effort); '' = no thinking")
    ap.add_argument("--price-input", type=float, help="$ per M input tokens (if not in llm_client)")
    ap.add_argument("--price-output", type=float, help="$ per M output tokens")
    ap.add_argument("--out", type=Path, default=Path("frontier_search_test.json"))
    args = ap.parse_args()
    conf = {"model": args.model, "allowed_domains": args.domains, "max_searches": args.max_searches,
            "search_tool": args.search_tool,
            "max_tokens": args.max_tokens, "effort": args.effort}
    if args.price_input is not None:
        conf["price_input"] = args.price_input
    if args.price_output is not None:
        conf["price_output"] = args.price_output
    if (missing := llm_client.require_api_key(args.model)):
        sys.exit(missing)
    if not is_priced(conf):
        print(f"[warn] no price for {args.model}; cost will show $0 (pass --price-input/--price-output)",
              file=sys.stderr)
    res = query_claude(args.question, conf)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    import research_pipeline as rp
    text, inline = rp.format_answer_for_judge(res["answer"], res["trace"])
    t, a = res["trace"], res["trace"]["domain_audit"]
    doms = defaultdict(int)
    for c in t["tool_calls"]:
        if c["tool_name"] == "web_search":           # real searches only, not the "cited" bucket
            for d in c["documents"]:
                doms[urlparse(d["url"]).netloc.lower()] += 1
    print(f"searches {res['usage'].get('web_search_requests')} | results {a['search_results']} | "
          f"domains {dict(doms)}\noff-domain: {len(a['off_domain_results'])} result URLs, "
          f"{len(a['off_domain_citations'])} cited | cited URLs not in any result: "
          f"{len(a['cited_not_in_results'])}\ncomplete {t['complete']} (stop reasons {t['stop_reasons']}) | "
          f"interim text blocks {len(t['interim_text'])} | stream repairs {len(t['stream_text_repaired'])} | "
          f"empty cited blocks {t['empty_cited_blocks']} | {t['elapsed_s']}s | ${res['cost_usd']:.3f} | "
          f"usage {res['usage']}\nsearch tool {t['search_tool']} | block types "
          f"{dict(Counter(b.get('type') for b in t['raw_blocks']))}\ninline cites {inline} | judge rendering {len(text)} chars -> {args.out}\n")
    print(text[:3000])


if __name__ == "__main__":
    main()
