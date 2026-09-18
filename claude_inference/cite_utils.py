#!/usr/bin/env python3
"""
Shared helpers for resolving DR-Tulu inline citations against an attempt's trace.

DR-Tulu answers contain inline citations like:  <cite id="5209281c-1">claim</cite>
where the id is "<tool_call_id>-<doc_index>". The id indexes into
trace.tool_calls[*].documents (enriched with raw_output.data[*].paper for authors
and corpusId). Both view_answer.py and summarize_run.py import from here so the
overview report and the per-answer viewer resolve citations identically.
"""

import html
import json
import re

CITE_RE = re.compile(r'<cite\s+id="([^"]+)">(.*?)</cite>', re.DOTALL)

# --- WebThinker -------------------------------------------------------------
# WebThinker's trace is a LIST of explorer calls, each
#   {search_query, Input, Output, Extracted_info}
# where Input embeds the call's search results as numbered JSON blocks:
#   ***Web Page 3:***
#   {"id": 3, "title": ..., "url": ..., "snippet": ..., "page_info": ...}
# One block == one search result == one URL. The answer carries NO inline
# citations, so "which sources were used" is recovered from Extracted_info --
# the only text passed back to the main reasoner -- which references pages as
# "(Web Pages 3 and 5)". Calls that reference none get all their pages (a
# superset: the real source is in there, we just cannot tell which).
WEBTHINKER_PAGE_RE = re.compile(r'\*\*\*Web Page (\d+):\*\*\*\s*(\{.*?\n\})', re.DOTALL)
WEBTHINKER_REF_RE = re.compile(r'Web Pages?\s+([0-9]+(?:\s*(?:,|and|&)\s*[0-9]+)*)')
# Per-source cap on page_info. Uncapped, back-referenced pages run ~129k chars
# (~32k tokens) -- 1.7x the largest DR-Tulu judge prompt observed. Capped, the
# reference block lands at ~61k chars (~15k tokens), inside DR-Tulu's
# median..p95 band. Truncation, not ranking: every selected page is kept.
WEBTHINKER_PAGE_CHARS = 2000


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def s2_url(corpus_id) -> str:
    return f"https://api.semanticscholar.org/CorpusID:{corpus_id}" if corpus_id else ""


def abridge(text: str, words: int = 40) -> str:
    """First `words` words of a snippet, with an ellipsis if truncated."""
    toks = (text or "").split()
    return " ".join(toks[:words]) + (" …" if len(toks) > words else "")


def build_doc_index(trace) -> dict:
    """Map cite-id -> document dict, dispatching on the trace's shape.

    DR-Tulu traces are dicts ({"tool_calls": [...]}); WebThinker's is a list of
    explorer calls. Anything else (Tongyi's chat-message list carries no
    retrievable documents, None, a bare string) yields {} rather than raising --
    the viewers pass whatever the server returned straight through.
    """
    if isinstance(trace, dict):
        return _build_doc_index_drtulu(trace)
    if isinstance(trace, list) and trace and isinstance(trace[0], dict) \
            and "Input" in trace[0]:
        return build_doc_index_webthinker(trace)
    return {}


def _build_doc_index_drtulu(trace: dict) -> dict:
    """Map cite-id ('<call_id>-<i>') -> merged document dict."""
    out = {}
    for tc in (trace or {}).get("tool_calls", []) or []:
        call_id = tc.get("call_id")
        docs = tc.get("documents") or []
        data = (tc.get("raw_output") or {}).get("data") or []
        for i, doc in enumerate(docs):
            paper = data[i].get("paper", {}) if i < len(data) else {}
            out[f"{call_id}-{i}"] = {
                "title": doc.get("title") or paper.get("title") or "(untitled)",
                "authors": paper.get("authors") or [],
                "corpus_id": paper.get("corpusId"),
                "url": doc.get("url") or "",
                "snippet": doc.get("snippet") or doc.get("text") or "",
                "query": tc.get("query", ""),
            }
    return out


def build_doc_index_webthinker(trace: list, page_chars: int = WEBTHINKER_PAGE_CHARS) -> dict:
    """Map '<call_idx>-<page_id>' -> document dict, in build_doc_index's shape.

    Each doc's `snippet` is the evidence text the references block should show:
    the fetched page text (`page_info`, truncated to `page_chars`) for pages the
    model back-referenced, and the search-result blurb for the rest. Serper's
    blurb runs 95-200 chars -- provenance, not evidence -- while page_info's
    median 2.5k is comparable to a DR-Tulu snippet (median 1.9k), so spending
    the tokens only on back-referenced pages keeps the prompt in range. A
    back-referenced page whose fetch failed has no page_info and degrades to its
    blurb. `page_chars=0` disables truncation.
    """
    out = {}
    for ci, call in enumerate(trace or []):
        if not isinstance(call, dict):
            continue
        docs = {}
        for pid, raw in WEBTHINKER_PAGE_RE.findall(call.get("Input") or ""):
            try:
                docs[pid] = json.loads(raw)
            except (ValueError, TypeError):
                continue                      # a malformed block loses one page, not the call
        if not docs:
            continue
        cited = set()
        for group in WEBTHINKER_REF_RE.findall(call.get("Extracted_info") or ""):
            cited |= set(re.findall(r"\d+", group))
        cited &= set(docs)
        for pid, doc in docs.items():
            backref = pid in cited
            blurb = doc.get("snippet") or ""
            page = doc.get("page_info") or ""
            if backref and page:
                evidence = page[:page_chars] if page_chars else page
            else:
                evidence = blurb
            out[f"{ci}-{pid}"] = {
                "title": doc.get("title") or "(untitled)",
                "authors": [],                # WebThinker carries no author metadata
                "corpus_id": None,            # so dedupe falls back to url/title
                "url": doc.get("url") or "",
                "snippet": evidence,
                "query": call.get("search_query") or "",
                # Selection metadata, ignored by the shared renderers.
                "backref": backref,
                "selected": backref or not cited,
                "search_snippet": blurb,
            }
    return out


def selected_refs(doc_index: dict):
    """[(n, doc)] for a system with no inline citations, deduped by url/title.

    Numbering follows trace order. When the same URL turns up in several calls
    (21 of 84 did in a measured run), a back-referenced copy wins over a
    fallback one so the richer page text is the one shown.
    """
    order, by_key = [], {}
    for key in doc_index:
        doc = doc_index[key]
        if not doc.get("selected"):
            continue
        dedupe_key = doc.get("url") or doc.get("title")
        if dedupe_key not in by_key:
            by_key[dedupe_key] = doc
            order.append(dedupe_key)
        elif doc.get("backref") and not by_key[dedupe_key].get("backref"):
            by_key[dedupe_key] = doc
    return [(i + 1, by_key[k]) for i, k in enumerate(order)]


def has_inline_citations(answer: str) -> bool:
    return bool(CITE_RE.search(answer or ""))


def resolve_answer(answer: str, trace):
    """(html_body, refs, missing_ids) for any answering system.

    Systems that emit inline <cite> tags resolve them; systems that do not
    (WebThinker) get the trace-selected sources as a flat reference list, so the
    viewer shows what informed the answer even though no claim points at it.
    """
    doc_index = build_doc_index(trace)
    body, refs, missing = render_answer(answer or "", doc_index)
    if not refs and not has_inline_citations(answer):
        refs = selected_refs(doc_index)
    return body, refs, missing


def render_answer(answer: str, doc_index: dict):
    """Return (html_body, ordered_refs, missing_ids).

    Citations become numbered [n] links with a hover card showing the paper title
    and an abridged snippet. Reference numbers are assigned by first appearance and
    deduped on corpusId/title so the same paper reuses one number.
    """
    refs = []          # list of (num, doc)
    key_to_num = {}    # corpusId or title -> num
    missing = set()

    def ref_num_for(cid):
        doc = doc_index.get(cid)
        if doc is None:
            missing.add(cid)
            return None, None
        key = doc.get("corpus_id") or doc.get("title")
        if key not in key_to_num:
            key_to_num[key] = len(refs) + 1
            refs.append((key_to_num[key], doc))
        return key_to_num[key], doc

    pieces, last = [], 0
    for m in CITE_RE.finditer(answer):
        pieces.append(esc(answer[last:m.start()]))
        claim = esc(m.group(2))
        badges = []
        for cid in m.group(1).split():
            num, doc = ref_num_for(cid)
            if num is None:
                badges.append('<sup class="cite missing">[?]</sup>')
            else:
                snip = abridge(doc["snippet"]) or "(no snippet available)"
                tip = (f'<span class="tip"><b>{esc(doc["title"])}</b>'
                       f'<span class="tsnip">{esc(snip)}</span></span>')
                badges.append(
                    f'<span class="cw"><sup class="cite">'
                    f'<a href="#ref{num}">[{num}]</a></sup>{tip}</span>'
                )
        pieces.append(f'<span class="claim">{claim}</span>{"".join(badges)}')
        last = m.end()
    pieces.append(esc(answer[last:]))
    body = "".join(pieces).replace("\n", "<br>\n")
    return body, refs, missing


def _ref_line(n, doc) -> str:
    """One '[n] Title — url/id' reference line (Title + URL/source-id detail)."""
    title = doc.get("title") or "(untitled)"
    url = doc.get("url") or (s2_url(doc.get("corpus_id")) if doc.get("corpus_id") else "")
    return f"[{n}] {title}" + (f" — {url}" if url else "")


def references_block(refs, include_snippets: bool = False, snippet_words: int = 0,
                     title: str = "References") -> str:
    """Plain-text 'References' section for a list of (num, doc) tuples ('' if empty).

    With include_snippets, each entry also carries the paper's authors and the
    retrieved snippet — the same detail the HTML reference list shows — so a reader
    (or a judge) can check a claim against the text it cites. Pass snippet_words to
    abridge those snippets; 0 keeps them whole.
    """
    if not refs:
        return ""
    entries = []
    for n, doc in refs:
        lines = [_ref_line(n, doc)]
        if include_snippets:
            authors = doc.get("authors") or []
            if authors:
                who = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
                lines.append(f"    {who}")
            snippet = doc.get("snippet") or ""
            if snippet and snippet_words:
                snippet = abridge(snippet, snippet_words)
            if snippet:
                lines.append(f"    snippet: {snippet}")
        entries.append("\n".join(lines))
    sep = "\n\n" if include_snippets else "\n"
    return f"\n\n{title}\n{sep.join(entries)}"


def numbered_plaintext(answer: str, doc_index: dict):
    """Convert a DR-Tulu <cite id="...">claim</cite> answer to plain text with inline
    [n] markers, returning (marked_text, refs). Refs are deduped by paper (corpusId/title)
    and numbered by first appearance — the same scheme as render_answer, in plain text."""
    refs, key_to_num = [], {}

    def num_for(cid):
        doc = doc_index.get(cid)
        if doc is None:
            return None
        key = doc.get("corpus_id") or doc.get("title")
        if key not in key_to_num:
            key_to_num[key] = len(refs) + 1
            refs.append((key_to_num[key], doc))
        return key_to_num[key]

    def repl(m):
        nums = [n for n in (num_for(c) for c in m.group(1).split()) if n]
        marks = "".join(f"[{n}]" for n in nums)
        return f"{m.group(2)} {marks}".rstrip() if marks else m.group(2)

    return CITE_RE.sub(repl, answer), refs


def render_refs(refs) -> str:
    if not refs:
        return "<p><em>No resolved references.</em></p>"
    items = []
    for num, doc in refs:
        authors = ", ".join(doc["authors"][:6]) + (" et al." if len(doc["authors"]) > 6 else "")
        link = doc["url"] or s2_url(doc["corpus_id"])
        title = esc(doc["title"])
        title_html = f'<a href="{esc(link)}" target="_blank">{title}</a>' if link else title
        authors_html = f'<div class="ra">{esc(authors)}</div>' if authors else ""
        snippet_html = f'<div class="rs">{esc(doc["snippet"])}</div>' if doc["snippet"] else ""
        items.append(
            f'<li id="ref{num}"><span class="rn">[{num}]</span> '
            f'<span class="rt">{title_html}</span>{authors_html}{snippet_html}</li>'
        )
    return f'<ol class="refs">{"".join(items)}</ol>'


def render_searches(trace) -> str:
    """Searches-run table for either trace shape (empty string for neither)."""
    if isinstance(trace, dict):
        rows = [(tc.get("tool_name"), tc.get("query"), len(tc.get("documents") or []))
                for tc in (trace.get("tool_calls") or [])]
    elif isinstance(trace, list):
        rows = [("web_search", c.get("search_query"),
                 len(WEBTHINKER_PAGE_RE.findall(c.get("Input") or "")))
                for c in trace if isinstance(c, dict) and "Input" in c]
    else:
        rows = []
    calls = rows
    if not calls:
        return ""
    rows = "".join(
        f'<tr><td>{esc(tool)}</td><td>{esc(query)}</td><td>{n}</td></tr>'
        for tool, query, n in calls
    )
    return ('<details class="searches"><summary>Searches run '
            f'({len(calls)})</summary><table>'
            '<tr><th>tool</th><th>query</th><th>#docs</th></tr>'
            f'{rows}</table></details>')


# CSS shared by the viewer and the report's answer panels.
CITE_CSS = """
 sup.cite a { text-decoration:none; color:#0969da; font-weight:bold; }
 sup.cite.missing { color:#cf222e; }
 .cw { position:relative; }
 .cw .tip { display:none; position:absolute; left:0; top:1.4em; z-index:20; width:340px;
            background:#1f2328; color:#fff; padding:8px 10px; border-radius:6px;
            font:13px/1.45 -apple-system,system-ui,sans-serif; box-shadow:0 4px 16px rgba(0,0,0,.25); }
 .cw:hover .tip { display:block; }
 .cw .tip b { display:block; margin-bottom:4px; color:#9ec5ff; }
 .cw .tip .tsnip { color:#e6edf3; }
 .refs { font-family:-apple-system,system-ui,sans-serif; font-size:13.5px; padding-left:0; list-style:none; }
 .refs li { margin:10px 0; padding:6px 0 6px 12px; border-left:3px solid #d0d7de; }
 .refs li:target { border-left-color:#0969da; background:#ddf4ff; }
 .rn { color:#57606a; font-weight:bold; margin-right:4px; }
 .rt a { color:#0969da; text-decoration:none; } .rt a:hover { text-decoration:underline; }
 .ra { color:#57606a; font-style:italic; margin-top:2px; }
 .rs { color:#3a3f45; margin-top:4px; font-size:12.5px; background:#f6f8fa; padding:6px 8px; border-radius:5px; }
 .searches { font-family:-apple-system,system-ui,sans-serif; font-size:13px; margin:10px 0; }
 .searches table { border-collapse:collapse; width:100%; margin-top:6px; }
 .searches td, .searches th { border:1px solid #d0d7de; padding:3px 7px; text-align:left; }
 .warn { color:#9a6700; font-family:-apple-system,system-ui,sans-serif; font-size:13px;
         background:#fff8c5; border:1px solid #eac54f; border-radius:6px; padding:6px 10px; margin:8px 0; }
"""
