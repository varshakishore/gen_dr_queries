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

from cite_utils import (
    CITE_CSS, build_doc_index, esc, render_answer, render_refs, render_searches,
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
{cite_css}
</style></head><body>
<h1>{heading}</h1>
<div class="seed">Seed: {seed}</div>
{attempts}
</body></html>"""


def render_attempt(att: dict, is_last: bool = False) -> str:
    h = att.get("harder", {})
    j = att.get("judgment", {})
    doc_index = build_doc_index(att.get("trace") or {})
    body, refs, missing = render_answer(att.get("answer") or "", doc_index)
    verdict = j.get("verdict", "")
    vcolor = "#cf222e" if verdict == "FAILED" else "#1a7f37"
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
    <div><b>Judge</b> {esc(j.get('summary'))}</div>
  </div>
  {render_searches(att.get('trace') or {{}})}
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
