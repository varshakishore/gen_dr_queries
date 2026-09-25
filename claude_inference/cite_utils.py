#!/usr/bin/env python3
"""
Shared helpers for turning an answering system's trace into a reference list.

Three trace shapes are supported, dispatched on by `build_doc_index`:

  DR-Tulu    a dict, {"tool_calls": [...]}. The answer carries inline citations
             (<cite id="5209281c-1">claim</cite>, the id being
             "<tool_call_id>-<doc_index>") which index into
             trace.tool_calls[*].documents, enriched with raw_output.data[*].paper
             for authors and corpusId.
  WebThinker a list of explorer calls; sources are embedded in each call's prompt
             string and selected via back-references in Extracted_info.
  Tongyi     a ReAct message list; sources come from search/scholar responses and
             the selection signal is which pages the model chose to `visit`.

The latter two emit NO inline citations, so their reference list is provenance --
what informed the answer -- rather than claim-level attribution, and the judge is
told so (JUDGE_PROMPT_NO_INLINE_CITES). Every shape produces the same doc dict, so
view_answer.py, summarize_run.py, annotate_answer.py, the annotation app and the
judge all number references identically.
"""

import html
import json
import re

CITE_RE = re.compile(r'<cite\s+id="([^"]+)">(.*?)</cite>', re.DOTALL)
# DR-Tulu joins several ids in one tag with EITHER spaces or commas -- id="a-2 a-3" and
# id="a-2,a-3" both occur, sometimes mixed. Splitting on whitespace alone left the comma
# form as one unresolvable token: measured over 1,478 stored DR-Tulu answers, 917 of the
# 1,686 unresolved ids (54%) were comma-joined, and 84% of their parts resolve once split.
# Safe because no id ever legitimately contains a comma -- 0 of 26,072 doc-index keys have
# one, and no comma-joined token resolved as a whole.
CITE_ID_SEP = re.compile(r"[,\s]+")

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
# The back-reference list. The separator repeats (`(?:\s*(?:,|and|&|or))+`) so an Oxford
# comma -- ", and 8" -- reads as one separator rather than ending the list: with a single
# separator token "Web Pages 2, 7, and 8" stopped at 7, and the dropped page vanished from
# the references entirely, because a call that back-references anything selects ONLY the
# pages it named. Measured over 15 stored traces: 12 ids were being lost this way, 10 to the
# Oxford comma and 2 to "or"; 4 distinct pages present in their call were recovered, all 4
# of them previously excluded. Ranges ("Web Pages 1-4") are expanded by
# _webthinker_ref_ids. Every number has to sit in the run that starts immediately after
# "Web Page(s)", which is what keeps prose out: in "Web Page 4 notes ... 3-5 years" the run
# ends at 4, since " notes" is not a separator.
WEBTHINKER_REF_SEP = r'(?:\s*(?:,|and|&|or))+\s*'
WEBTHINKER_REF_NUM = r'\d+(?:\s*[-\u2013\u2014]\s*\d+)?'
WEBTHINKER_REF_RE = re.compile(
    rf'Web Pages?\s+({WEBTHINKER_REF_NUM}(?:{WEBTHINKER_REF_SEP}{WEBTHINKER_REF_NUM})*)',
    re.I)
_WEBTHINKER_RANGE_RE = re.compile(r'(\d+)\s*[-\u2013\u2014]\s*(\d+)|(\d+)')
# A page list is a handful of ids; anything wider is prose that happens to look like a
# range, so only the endpoints are taken.
WEBTHINKER_MAX_REF_RANGE = 20
# Per-source cap on page_info. Uncapped, back-referenced pages run ~129k chars
# (~32k tokens) -- 1.7x the largest DR-Tulu judge prompt observed. Capped, the
# reference block lands at ~61k chars (~15k tokens), inside DR-Tulu's
# median..p95 band. Truncation, not ranking: every selected page is kept.
WEBTHINKER_PAGE_CHARS = 2000

# --- Tongyi DeepResearch -----------------------------------------------------
# Tongyi's trace is a ReAct message list -- {role, content} where assistant turns
# carry <tool_call> and user turns carry <tool_response>. Three response shapes
# produce sources:
#   search   "A Google search for '...' found N results:"  -> 1. [Title](url) + snippet
#   scholar  "A Google scholar for '...' found N results:" -> 1. [Title](pdfUrl: url)
#                                                             + publicationInfo/citedBy
#   visit    "The useful information in <url> for user goal <goal> as follows:"
#            followed by "Evidence in page:" and the extracted text
# `visit` is the selection signal and it is unambiguous: the model chose to read
# that page. Measured over 11 traces, visits pick 6-8% of the search pool (105 of
# 1278; 18 of 278 live), so unlike WebThinker there is no back-reference parsing
# and no superset fallback -- the used set is simply what was visited.
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)
TONGYI_SEARCH_HEAD = re.compile(r"^A Google (search|scholar) for ", re.I)
# Sources to show when a trace searched but never visited anything (not seen in 11
# traces; the minimum was 1 visit). Keeps the judge from getting an empty list.
TONGYI_POOL_FALLBACK = 10
TONGYI_HIT_RE = re.compile(
    r"^\s*\d+\.\s*\[([^\]]+)\]\(\s*(?:pdfUrl:\s*)?(https?://[^\)\s]+)\s*\)\s*\n(.*?)"
    r"(?=\n\s*\d+\.\s*\[|\Z)",
    re.S | re.M,
)
TONGYI_VISIT_RE = re.compile(
    r"^The useful information in (\S+) for user goal (.*?) as follows:\s*(.*)", re.S)
TONGYI_PUBINFO_RE = re.compile(r"^publicationInfo:\s*(.+)$", re.M)
TONGYI_HEADING_RE = re.compile(r"^#{1,4}\s*(.+)$|^Title:\s*(.+)$", re.M)
# Every visit body opens with this line (127 of 127 measured); it carries nothing
# and would otherwise be the first thing the judge reads in every reference.
TONGYI_EVIDENCE_LEAD_RE = re.compile(r"^\s*Evidence in page:\s*", re.I)
# A visit that did not actually reach the page still comes back as a normal
# response -- an apology, or the text of a bot wall -- and supports nothing, so it
# must not become a reference. 26 of 127 measured visits (20%) are one of these,
# mostly CAPTCHA interstitials on researchgate / openreview / pmc. Length is not a
# usable signal: the longest such body is 20,550 chars of reCAPTCHA page furniture,
# well above the 2,182-char median of a real visit. Matched against the opening of
# the body only, since a wall announces itself immediately while a page that merely
# discusses CAPTCHAs would not.
TONGYI_FETCH_FAIL_RE = re.compile(
    r"captcha|complete the check|security check required|just a moment|"
    r"checking your browser|cloudflare|verify you are human|"
    r"access denied|403 forbidden|not authorized|permission denied|"
    r"could not be accessed|not directly accessible|unable to (?:access|retrieve)|"
    r"requires user interaction|failed to (?:fetch|load)|error (?:fetching|loading)|"
    r"no HTML elements|content is empty|no (?:useful|relevant) information",
    re.I)
TONGYI_FAIL_WINDOW = 400
TONGYI_MIN_EVIDENCE = 200
# Tongyi visits a lot of PDFs, and roughly half its visited URLs never appeared in
# a search result, so there is no title to inherit and the body has no heading.
# These URLs are still identifiable, so build a label from the id rather than
# showing a raw PDF link.
TONGYI_URL_LABELS = (
    (re.compile(r"arxiv\.org/(?:abs|pdf|html|e-print)/([0-9]{4}\.[0-9]{4,5})"), "arXiv:{}"),
    (re.compile(r"aclanthology\.org/([0-9A-Za-z.\-]+?)(?:\.pdf|/|$)"), "ACL Anthology {}"),
    (re.compile(r"doi\.org/(10\.[^\s/]+/[^\s?#]+)"), "doi:{}"),
    (re.compile(r"(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/(PMC[0-9]+)"), "PMC {}"),
)
# Visit text is already condensed -- Tongyi fetches a page and keeps ~6% of it,
# selected against its own stated goal -- so every character survived a relevance
# pass and truncation cuts chosen content, not boilerplate. Measured over 127
# visits: median 2201 chars, p90 6682, max 20569. A 2000-char cap would truncate
# 53% of references and drop 47% of the text; 4000 truncates 18% and keeps 75%,
# clipping only the long tail. Uncapped the block runs median 35k / max 58k,
# already inside DR-Tulu's range (median 16k, p95 37k, max 77k), so the cap is
# tail insurance rather than a budget necessity. 0 disables truncation.
TONGYI_PAGE_CHARS = 4000


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
    explorer calls ("Input" per entry); Tongyi's is a ReAct message list
    ("role"/"content" per entry). Anything else (None, a bare string, a list of
    something else) yields {} rather than raising -- the viewers pass whatever the
    server returned straight through.
    """
    if isinstance(trace, dict):
        return _build_doc_index_drtulu(trace)
    if isinstance(trace, list) and trace and isinstance(trace[0], dict):
        if "Input" in trace[0]:
            return build_doc_index_webthinker(trace)
        if "role" in trace[0] and "content" in trace[0]:
            return build_doc_index_tongyi(trace)
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


def _webthinker_ref_ids(extracted_info: str) -> set:
    """Page ids back-referenced in one explorer call's Extracted_info, ranges expanded."""
    ids = set()
    for group in WEBTHINKER_REF_RE.findall(extracted_info or ""):
        for lo, hi, single in _WEBTHINKER_RANGE_RE.findall(group):
            if single:
                ids.add(int(single))
                continue
            lo, hi = int(lo), int(hi)
            if 0 < hi - lo <= WEBTHINKER_MAX_REF_RANGE:
                ids.update(range(lo, hi + 1))
            else:
                ids.update((lo, hi))
    return {str(n) for n in ids}


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
        cited = _webthinker_ref_ids(call.get("Extracted_info") or "")
        cited &= set(docs)       # an id the model invented, or a range overshooting, drops out
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


def _tongyi_tool_responses(trace: list):
    """Yield (message_index, response_text) for every <tool_response> in the trace."""
    for i, msg in enumerate(trace or []):
        if not isinstance(msg, dict):
            continue
        for body in TOOL_RESPONSE_RE.findall(msg.get("content") or ""):
            yield i, body.strip()


def _tongyi_title_from_evidence(text: str) -> str:
    """First markdown heading / 'Title:' line in a visit's extracted text, if any."""
    m = TONGYI_HEADING_RE.search(text or "")
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").strip()[:200]


def _tongyi_label_from_url(url: str) -> str:
    """'arXiv:2604.01657' etc. for a URL whose id is recognisable, else ''."""
    for pattern, fmt in TONGYI_URL_LABELS:
        m = pattern.search(url or "")
        if m:
            return fmt.format(m.group(1))
    return ""


def build_doc_index_tongyi(trace: list, page_chars: int = TONGYI_PAGE_CHARS) -> dict:
    """Map '<msg_idx>-<n>' -> document dict, in build_doc_index's shape.

    Search and scholar hits build a candidate pool keyed by URL; `visit` responses
    are the sources the model actually read, and only those are marked `selected`.
    A visited URL takes its title, blurb and authors from the pool entry when the
    search results carried one -- only about half do, since Tongyi visits URLs it
    reached by other means -- and otherwise falls back to a heading inside the
    extracted text, then to the URL itself.

    Each doc's `snippet` is the evidence the references block shows: the visit's
    extracted text (truncated to `page_chars`; 0 disables) for visited pages, and
    the search blurb for the rest.
    """
    pool = {}                       # url -> {title, snippet, authors, query}
    visits = {}                     # url -> (msg_idx, goal, evidence)
    for idx, body in _tongyi_tool_responses(trace):
        if TONGYI_SEARCH_HEAD.match(body):
            query = ""
            qm = re.match(r"^A Google (?:search|scholar) for '([^']*)'", body)
            if qm:
                query = qm.group(1)
            for title, url, tail in TONGYI_HIT_RE.findall(body):
                if url in pool:
                    continue
                authors = []
                pm = TONGYI_PUBINFO_RE.search(tail)
                if pm:                       # "AR Cornelius - 2012 - digitalcommons…"
                    authors = [pm.group(1).split(" - ")[0].strip()]
                blurb = "\n".join(
                    ln for ln in tail.strip().splitlines()
                    if not re.match(r"^(publicationInfo|Date published|citedBy):", ln)
                ).strip()
                pool[url] = {"title": title.strip(), "snippet": blurb,
                             "authors": authors, "query": query}
            continue
        vm = TONGYI_VISIT_RE.match(body)
        if vm:
            url, goal, evidence = vm.groups()
            evidence = TONGYI_EVIDENCE_LEAD_RE.sub("", evidence.strip()).strip()
            # A failed fetch supports nothing -- do not make it a reference.
            if len(evidence) < TONGYI_MIN_EVIDENCE \
                    or TONGYI_FETCH_FAIL_RE.search(evidence[:TONGYI_FAIL_WINDOW]):
                continue
            visits.setdefault(url, (idx, goal.strip(), evidence))

    out = {}
    for n, (url, (idx, goal, evidence)) in enumerate(visits.items()):
        hit = pool.get(url) or {}
        title = (hit.get("title")
                 or _tongyi_title_from_evidence(evidence)
                 or _tongyi_label_from_url(url)
                 or url)
        out[f"{idx}-{n}"] = {
            "title": title,
            "authors": hit.get("authors") or [],
            "corpus_id": None,
            "url": url,
            "snippet": (evidence[:page_chars] if page_chars else evidence),
            "query": goal or hit.get("query") or "",
            "backref": True,          # a visit is an explicit read, not an inference
            "selected": True,
            "search_snippet": hit.get("snippet", ""),
        }
    if out:
        return out
    # Degenerate case: the model searched but never visited anything. Fall back to
    # the search pool so the judge still sees what was consulted.
    for n, (url, hit) in enumerate(list(pool.items())[:TONGYI_POOL_FALLBACK]):
        out[f"pool-{n}"] = {
            "title": hit["title"], "authors": hit["authors"], "corpus_id": None,
            "url": url, "snippet": hit["snippet"], "query": hit["query"],
            "backref": False, "selected": True, "search_snippet": hit["snippet"],
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


def split_cite_ids(raw: str) -> list:
    """The ids inside one `<cite id="...">`, split on commas as well as whitespace."""
    return [c for c in CITE_ID_SEP.split(raw.strip()) if c]


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
        for cid in split_cite_ids(m.group(1)):
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
        nums = [n for n in (num_for(c) for c in split_cite_ids(m.group(1))) if n]
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
    elif isinstance(trace, list) and trace and isinstance(trace[0], dict) \
            and "role" in trace[0]:
        rows = []
        for _, body in _tongyi_tool_responses(trace):
            if TONGYI_SEARCH_HEAD.match(body):
                qm = re.match(r"^A Google (search|scholar) for '([^']*)'", body)
                rows.append((f"google_{qm.group(1)}" if qm else "google_search",
                             qm.group(2) if qm else "",
                             len(TONGYI_HIT_RE.findall(body))))
            else:
                vm = TONGYI_VISIT_RE.match(body)
                if vm:
                    rows.append(("visit", vm.group(1), 1))
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
