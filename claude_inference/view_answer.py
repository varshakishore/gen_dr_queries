#!/usr/bin/env python3
"""
Render DR-Tulu answers as readable HTML pages with resolved, linked references.

Citations (<cite id="...">) are resolved against each attempt's trace and rendered
as numbered [n] links; hovering a citation shows the paper title and an abridged
snippet. A numbered References list and a collapsible "Searches run" trail follow.

Usage:
  python view_answer.py runs/test_sqa50/sample_001.json          # all attempts
  python view_answer.py runs/test_sqa50/sample_001.json --attempt 2
  python view_answer.py runs/test_sqa50                           # whole run -> answers/

summarize_run.py reuses render_sample() to generate these pages and link to them.
"""

import argparse
import json
from pathlib import Path

import re

from cite_utils import (
    CITE_CSS, abridge, build_doc_index, esc, render_answer, render_refs, render_searches,
)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font:16px/1.65 Georgia, 'Times New Roman', serif; max-width:820px; margin:0 auto;
        padding:30px 24px 80px; color:#1f2328; }}
 h1 {{ font-size:22px; font-family:-apple-system,system-ui,sans-serif; }}
 h3 {{ font-size:15px; font-family:-apple-system,system-ui,sans-serif; color:#57606a;
       text-transform:uppercase; letter-spacing:.04em; }}
 details.attempt {{ border-top:2px solid #d0d7de; margin-top:20px; padding-top:8px; }}
 details.attempt > summary {{ cursor:pointer; list-style:none; font-family:-apple-system,system-ui,sans-serif;
       display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; padding:8px 0; }}
 details.attempt > summary::-webkit-details-marker {{ display:none; }}
 .atitle {{ font-size:17px; font-weight:700; }}
 .verdict {{ font-weight:bold; font-size:14px; }}
 .aq {{ color:#57606a; font-size:13px; flex:1; min-width:200px; }}
 .seed {{ color:#57606a; font-family:-apple-system,system-ui,sans-serif; font-size:14px; }}
 .qbox {{ background:#f6f8fa; border:1px solid #d0d7de; border-radius:8px; padding:12px 14px;
          font-family:-apple-system,system-ui,sans-serif; font-size:14px; }}
 .qbox div {{ margin:4px 0; }} .qbox b {{ color:#57606a; }}
 .answer {{ margin:18px 0; }}
 .badge {{ color:#fff; border-radius:10px; padding:2px 9px; font-size:12px;
           font-family:-apple-system,system-ui,sans-serif; }}
{cite_css}
</style></head><body>
<h1>{heading}</h1>
<div class="seed">Seed: {seed}</div>
{attempts}
</body></html>"""


def render_attempt(att: dict, is_last: bool = False) -> str:
    h = att.get("harder") or {}
    # judgment is None when the attempt stopped before the research server was queried,
    # i.e. the criterion check rejected the criterion.
    j = att.get("judgment") or {}
    cc = att.get("criterion_check") or {}
    trace = att.get("trace") or {}
    doc_index = build_doc_index(trace)
    body, refs, missing = render_answer(att.get("answer") or "", doc_index)
    if att.get("judgment") is None and cc:
        verdict = cc.get("correctness_label", "") or "NO VERDICT"
        vcolor = "#9a6700"
    else:
        verdict = j.get("verdict", "")
        vcolor = "#cf222e" if verdict == "FAILED" else "#1a7f37"
    # Show what the criterion check said whenever it rewrote or rejected the criterion.
    cc_rows = ""
    if cc and (att.get("judgment") is None or h.get("verification_criterion_original")):
        if h.get("verification_criterion_original"):
            cc_rows += (f'<div><b>Criterion (original)</b> '
                        f'{esc(h["verification_criterion_original"])}</div>')
        cc_rows += f'<div><b>Criterion check</b> {esc(cc.get("correctness_label"))}</div>'
        if cc.get("main_correctness_problem"):
            cc_rows += f'<div><b>Problem</b> {esc(cc["main_correctness_problem"])}</div>'
    miss = (f'<div class="warn">{len(missing)} citation id(s) could not be resolved '
            f'from the trace.</div>' if missing else "")
    open_attr = " open" if is_last else ""
    return f"""
<details class="attempt"{open_attr}>
  <summary><span class="atitle">Attempt {att.get('attempt')}</span>
      <span class="verdict" style="color:{vcolor}">{esc(verdict)}</span>
      <span class="aq">{esc(h.get('updated_question'))}</span></summary>
  <div class="qbox">
    <div><b>Question</b> {esc(h.get('updated_question'))}</div>
    <div><b>Criterion</b> {esc(h.get('verification_criterion'))}</div>
    {cc_rows}
    <div><b>Judge</b> {esc(j.get('summary'))}</div>
  </div>
  {render_searches(trace)}
  {miss}
  <div class="answer">{body}</div>
  <h3>References</h3>
  {render_refs(refs)}
</details>"""


def render_sample(sample_path: Path, only_attempt=None) -> str:
    data = json.loads(sample_path.read_text())
    res = (data.get("results") or [None])[0]
    if res is None:
        return PAGE.format(title="(empty)", heading="(no result)", seed="",
                           attempts="", cite_css=CITE_CSS)
    atts = res.get("attempts") or []
    if only_attempt is not None:
        atts = [a for a in atts if a.get("attempt") == only_attempt]
    bodies = "".join(
        render_attempt(a, is_last=(i == len(atts) - 1)) for i, a in enumerate(atts)
    )
    return PAGE.format(
        title=f"Answer viewer — {sample_path.stem}",
        heading=f"{sample_path.stem} · {res.get('final_status', '')}",
        seed=esc(res.get("seed", "")), attempts=bodies, cite_css=CITE_CSS,
    )


VERDICT_COLOR = {"claude": "#1a7f37", "dr_tulu": "#cf222e", "tie": "#9a6700",
                 "both_bad": "#6e7781", "inconsistent": "#57606a"}
VERDICT_LABEL = {"claude": "Claude+search wins", "dr_tulu": "DR-Tulu wins",
                 "tie": "Tie", "both_bad": "Both bad",
                 "inconsistent": "Inconsistent (position bias)"}


def _verdict_chip(v) -> str:
    return (f'<span class="badge" style="background:{VERDICT_COLOR.get(v, "#57606a")}">'
            f'{esc(VERDICT_LABEL.get(v, v))}</span>')


_MARK_RE = re.compile(r"\[(\d+)\]")


def _source_snippet(snippet: str, norm_answer: str) -> str:
    """Hover preview text: the cited source passage, but only when it is NOT just a copy
    of the answer (we don't echo the answer's own text back as a 'source snippet')."""
    s = " ".join((snippet or "").split())
    return s if s and s.lower() not in norm_answer else ""


def _render_marked_answer(marked: str, sources: list, answer_text: str):
    """Render Claude's inline-[n] answer with hover cite links + a numbered References
    list — mirroring the DR-Tulu viewer. `sources[n-1]` is reference [n]. The references
    list shows Title + URL only; the hover preview shows the source snippet when it adds
    text beyond the answer."""
    norm_answer = " ".join((answer_text or "").split()).lower()
    body = esc(marked)

    def repl(m):
        n = int(m.group(1))
        if not (1 <= n <= len(sources)):
            return m.group(0)  # not a real citation number; leave as-is
        s = sources[n - 1]
        support = _source_snippet(s.get("snippet"), norm_answer)
        tsnip = f'<span class="tsnip">{esc(abridge(support))}</span>' if support else ""
        tip = (f'<span class="tip"><b>{esc(s.get("title") or s.get("url"))}</b>{tsnip}</span>')
        return (f'<span class="cw"><sup class="cite">'
                f'<a href="#cref{n}">[{n}]</a></sup>{tip}</span>')

    body = _MARK_RE.sub(repl, body).replace("\n", "<br>\n")

    # References list: Title + URL only (no snippet — the cited passage lives in the
    # answer/hover, not duplicated here).
    items = []
    for i, s in enumerate(sources, 1):
        url, title = s.get("url", ""), (s.get("title") or s.get("url") or "(untitled)")
        title_html = f'<a href="{esc(url)}" target="_blank">{esc(title)}</a>' if url else esc(title)
        items.append(f'<li id="cref{i}"><span class="rn">[{i}]</span> '
                     f'<span class="rt">{title_html}</span></li>')
    refs_html = f'<ol class="refs">{"".join(items)}</ol>' if items else "<p><em>No sources.</em></p>"
    return body, refs_html


def render_compare(compare_path: Path) -> str:
    """Render a Claude+web-search comparison (sample_NNN.compare.json) as HTML."""
    c = json.loads(compare_path.read_text())
    sources = c.get("claude_sources") or []
    cited = c.get("claude_sources_kind") == "cited" and c.get("claude_marked")
    if cited:
        ans, src_block = _render_marked_answer(
            c["claude_marked"], sources, c.get("claude_answer", ""))
        src_heading = f"References ({len(sources)})"
    else:
        # Fallback: plain answer + flat consulted-source list (no inline citations).
        ans = esc(c.get("claude_answer") or "").replace("\n", "<br>\n")
        items = "".join(
            f'<li><a href="{esc(s.get("url"))}" target="_blank">'
            f'{esc(s.get("title") or s.get("url"))}</a></li>' for s in sources)
        src_block = f'<ol class="refs">{items}</ol>'
        src_heading = f"Sources ({len(sources)}) — consulted (no inline citations)"
    raw = c.get("judge_raw", {})
    ov, cr = raw.get("overall", {}), raw.get("criterion", {})
    ov_conf = f" (confidence {ov.get('confidence')}/5)" if ov.get("confidence") is not None else ""
    cr_conf = f" (confidence {cr.get('confidence')}/5)" if cr.get("confidence") is not None else ""
    v = c.get("verdicts", {})
    cost = c.get("cost", {})
    order = "DR-Tulu shown as A" if c.get("judge_order") == "dr_tulu_first" else "Claude shown as A"
    judge_model = c.get("judge_model", "judge")
    return PAGE.format(
        title=f"Comparison — {compare_path.stem}",
        heading=f"{compare_path.stem.replace('.compare', '')} · DR-Tulu vs Claude+web-search",
        seed=esc(c.get("seed", "")),
        cite_css=CITE_CSS,
        attempts=f"""
<section class="attempt" open>
  <div class="qbox">
    <div><b>Question</b> {esc(c.get('question'))}</div>
    <div><b>Criterion</b> {esc(c.get('criterion'))}</div>
  </div>
  <h3>Verdicts ({esc(judge_model)} judge · {esc(order)})</h3>
  <div class="qbox">
    <div><b>Overall quality</b> {_verdict_chip(v.get('overall'))}{esc(ov_conf)}
         &nbsp;<span style="color:#57606a">— {esc(ov.get('reasoning'))}</span></div>
    <div><b>Criterion satisfaction</b> {_verdict_chip(v.get('criterion'))}{esc(cr_conf)}
         &nbsp;<span style="color:#57606a">— {esc(cr.get('reasoning'))}</span></div>
  </div>
  <h3>Claude + web-search answer
      <span style="font-weight:400;color:#57606a">({c.get('claude_num_searches', 0)} searches · ${cost.get('cost_usd', 0):.4f})</span></h3>
  <div class="answer">{ans}</div>
  <h3>{src_heading}</h3>
  {src_block}
</section>""",
    )


def main():
    p = argparse.ArgumentParser(description="Render DR-Tulu answers with resolved references.")
    p.add_argument("path", help="A sample_NNN.json file, or a run directory.")
    p.add_argument("--attempt", type=int, help="Only render this attempt number.")
    p.add_argument("-o", "--output", help="Output HTML path (single-file mode).")
    args = p.parse_args()

    path = Path(args.path)
    if path.is_dir():
        out_dir = path / "answers"
        out_dir.mkdir(exist_ok=True)
        links = []
        for fp in sorted(path.glob("sample_*.json")):
            (out_dir / f"{fp.stem}.html").write_text(render_sample(fp, args.attempt))
            links.append(f'<li><a href="{fp.stem}.html">{fp.stem}</a></li>')
        (out_dir / "index.html").write_text(
            f"<!doctype html><meta charset=utf-8><title>Answers</title>"
            f"<h1>Answer viewers — {path.name}</h1><ul>{''.join(links)}</ul>"
        )
        print(f"Wrote {len(links)} answer pages to {out_dir}/ (open {out_dir/'index.html'})")
    else:
        out = Path(args.output) if args.output else path.with_suffix(".answer.html")
        out.write_text(render_sample(path, args.attempt))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
