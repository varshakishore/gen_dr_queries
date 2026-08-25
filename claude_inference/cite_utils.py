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
import re

CITE_RE = re.compile(r'<cite\s+id="([^"]+)">(.*?)</cite>', re.DOTALL)


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def s2_url(corpus_id) -> str:
    return f"https://api.semanticscholar.org/CorpusID:{corpus_id}" if corpus_id else ""


def abridge(text: str, words: int = 40) -> str:
    """First `words` words of a snippet, with an ellipsis if truncated."""
    toks = (text or "").split()
    return " ".join(toks[:words]) + (" …" if len(toks) > words else "")


def build_doc_index(trace: dict) -> dict:
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


def references_block(refs, include_snippets: bool = False, snippet_words: int = 0) -> str:
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
    return f"\n\nReferences\n{sep.join(entries)}"


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


def render_searches(trace: dict) -> str:
    calls = (trace or {}).get("tool_calls", []) or []
    if not calls:
        return ""
    rows = "".join(
        f'<tr><td>{esc(tc.get("tool_name"))}</td><td>{esc(tc.get("query"))}</td>'
        f'<td>{len(tc.get("documents") or [])}</td></tr>'
        for tc in calls
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
