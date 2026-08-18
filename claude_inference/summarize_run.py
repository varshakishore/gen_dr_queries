#!/usr/bin/env python3
"""
Summarize a research_pipeline_parallel.py run directory.

Reads <run_dir>/index.json + sample_NNN.json files and produces:
  - a console summary (status counts, cost, attempt histogram, error breakdown)
  - a <run_dir>/report.html dashboard with one expandable card per seed: the
    original seed, the final question that broke the system, the strategy, the
    verification criterion, and the judge's reasoning.
  - <run_dir>/answers/sample_NNN.html, one rich answer page per seed (rendered via
    view_answer) with citations resolved to linked references; each report card
    links out to its answer page.

Usage:
  python summarize_run.py runs/test_sqa50
"""

import html
import json
import sys
from collections import Counter
from pathlib import Path

from view_answer import render_sample, render_compare, VERDICT_COLOR, VERDICT_LABEL


def load_run(run_dir: Path):
    index = json.loads((run_dir / "index.json").read_text())
    samples = []
    for row in index["samples"]:
        rec = dict(row)
        fpath = run_dir / row["file"]
        rec["result"] = None
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text())
                rec["result"] = (data.get("results") or [None])[0]
            except (ValueError, OSError):
                pass
        # Attach the Claude+web-search comparison if it exists.
        rec["compare"] = None
        cpath = run_dir / row["file"].replace(".json", ".compare.json")
        if cpath.exists():
            try:
                rec["compare"] = json.loads(cpath.read_text())
            except (ValueError, OSError):
                pass
        samples.append(rec)
    return index, samples


def console_summary(index, samples):
    print("#" * 70)
    print(f"RUN: model={index.get('model')}  max_attempts={index.get('max_attempts')}")
    print("#" * 70)
    counts = Counter(s["status"] for s in samples)
    for st, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {st:16} {n:3}  ({100*n/len(samples):.0f}%)")

    print(f"\n  total cost     ${index.get('total_cost_usd', 0):.4f}")
    print(f"  total LLM calls {index.get('total_claude_calls', 0)}")
    avg = index.get("avg_make_harder_calls_failed_found")
    if avg is not None:
        print(f"  avg make-harder calls / FAILED_FOUND seed: {avg:.2f} "
              f"(over {index.get('num_failed_found')} seeds)")

    # Attempt histogram over FAILED_FOUND (how many tries it took to break the system).
    ff = [s["attempts"] for s in samples if s["status"] == "FAILED_FOUND"]
    if ff:
        print("\n  attempts-to-break (FAILED_FOUND):")
        hist = Counter(ff)
        for k in sorted(hist):
            bar = "█" * hist[k]
            print(f"    {k} attempt(s): {bar} {hist[k]}")

    errs = [s for s in samples if s["status"] == "ERROR"]
    if errs:
        print(f"\n  {len(errs)} ERROR seed(s):")
        ecounts = Counter()
        for s in errs:
            msg = (s["result"] or {}).get("error", "") or ""
            # bucket by the leading phrase before the colon/details
            ecounts[msg.split(":")[0][:50] or "unknown"] += 1
        for msg, n in ecounts.most_common():
            print(f"    {n}x  {msg}")

    # Claude+web-search comparison stats, if any comparisons have been run.
    compared = [s for s in samples if s.get("compare")]
    if compared:
        print(f"\n  vs Claude+web-search ({len(compared)} compared):")
        for axis in ("overall", "criterion"):
            c = Counter(s["compare"].get("verdicts", {}).get(axis) for s in compared)
            parts = "  ".join(f"{k}={c[k]}" for k in
                              ("claude", "dr_tulu", "tie", "inconsistent") if c[k])
            print(f"    {axis:9} {parts}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

STATUS_COLOR = {
    "FAILED_FOUND": "#1a7f37",   # green: we broke the system (goal)
    "EXHAUSTED": "#9a6700",      # amber: never broke it
    "ERROR": "#cf222e",          # red: pipeline error
}


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _verdict_chip(v) -> str:
    return (f'<span class="badge" style="background:{VERDICT_COLOR.get(v, "#57606a")}">'
            f'{esc(VERDICT_LABEL.get(v, v))}</span>')


def card_html(s) -> str:
    status = s["status"]
    color = STATUS_COLOR.get(status, "#57606a")
    res = s["result"] or {}
    attempts = res.get("attempts") or []
    seed = res.get("seed", s.get("seed", ""))

    head = (
        f'<div class="head">'
        f'<span class="badge" style="background:{color}">{esc(status)}</span>'
        f'<span class="idx">#{s["index"]:03d}</span>'
        f'<span class="meta">{s.get("attempts", 0)} attempt(s) · '
        f'${s.get("cost_usd", 0):.4f}</span>'
        f'<span class="seed">{esc(seed)}</span>'
        f"</div>"
    )

    if status == "ERROR":
        body = f'<div class="err">{esc(res.get("error", "(no result file)"))}</div>'
        return f'<details class="card"><summary>{head}</summary>{body}</details>'

    # The decisive attempt is the last one (the FAILED one for FAILED_FOUND).
    parts = []
    if attempts:
        last = attempts[-1]
        h = last.get("harder") or {}
        # judgment is None when the criterion check rejected the criterion, so the
        # attempt never reached the research server.
        j = last.get("judgment") or {}
        cc = last.get("criterion_check") or {}
        parts.append(f'<div class="field"><b>Final question</b><div>{esc(h.get("updated_question"))}</div></div>')
        parts.append(f'<div class="field"><b>Strategy</b><div>{esc(h.get("chosen_strategy"))}</div></div>')
        parts.append(f'<div class="field"><b>Verification criterion</b><div>{esc(h.get("verification_criterion"))}</div></div>')
        if h.get("verification_criterion_original"):
            parts.append(f'<div class="field"><b>Criterion before rewrite</b>'
                         f'<div>{esc(h["verification_criterion_original"])}</div></div>')
        if last.get("judgment") is None:
            parts.append(f'<div class="field"><b>Criterion check</b>'
                         f'<div>{esc(cc.get("correctness_label"))} — '
                         f'{esc(cc.get("main_correctness_problem"))}</div></div>')
            if cc.get("reasoning"):
                parts.append(f'<div class="field"><b>Why the criterion was rejected</b>'
                             f'<div>{esc(cc["reasoning"])}</div></div>')
            if cc.get("rewrite"):
                parts.append(f'<div class="field"><b>Suggested rewrite</b>'
                             f'<div>{esc(cc["rewrite"])}</div></div>')
        else:
            parts.append(f'<div class="field"><b>Judge verdict</b><div>{esc(j.get("verdict"))} — {esc(j.get("summary"))}</div></div>')
            parts.append(f'<div class="field"><b>Why it {("failed" if j.get("verdict")=="FAILED" else "passed")}</b><div>{esc(j.get("criterion_reasoning"))}</div></div>')
            issues = j.get("other_issues") or []
            if issues:
                lis = "".join(f"<li>{esc(i)}</li>" for i in issues)
                parts.append(f'<div class="field"><b>Other issues</b><ul>{lis}</ul></div>')
        ans = last.get("answer") or ""
        stem = Path(s["file"]).stem
        if ans:
            parts.append(
                f'<div class="field"><a class="viewlink" href="answers/{stem}.html" '
                f'target="_blank">📄 Open full answer with linked references '
                f'({len(ans):,} chars)</a></div>'
            )

        # Claude + web-search comparison, if available.
        cmp = s.get("compare")
        if cmp:
            v = cmp.get("verdicts", {})
            chips = (f'overall: {_verdict_chip(v.get("overall"))} &nbsp; '
                     f'criterion: {_verdict_chip(v.get("criterion"))}')
            parts.append(f'<div class="field"><b>vs Claude+web-search</b><div>{chips}</div></div>')
            parts.append(
                f'<div class="field"><a class="viewlink" href="answers/{stem}.compare.html" '
                f'target="_blank">⚖️ Open Claude answer + judge reasoning</a></div>'
            )

    # Per-attempt trail (verdict + question for each round).
    if len(attempts) > 1:
        rows = "".join(
            f'<tr><td>{a.get("attempt")}</td>'
            f'<td>{esc((a.get("judgment") or {}).get("verdict") or (a.get("criterion_check") or {}).get("correctness_label"))}</td>'
            f'<td>{esc((a.get("harder") or {}).get("updated_question"))}</td></tr>'
            for a in attempts
        )
        parts.append(
            '<details class="trail"><summary>All attempts</summary>'
            f'<table><tr><th>#</th><th>verdict</th><th>question</th></tr>{rows}</table></details>'
        )

    return f'<details class="card"><summary>{head}</summary>{"".join(parts)}</details>'


def build_html(index, samples) -> str:
    counts = Counter(s["status"] for s in samples)
    chips = " ".join(
        f'<span class="chip" style="background:{STATUS_COLOR.get(st,"#57606a")}">{esc(st)}: {n}</span>'
        for st, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    avg = index.get("avg_make_harder_calls_failed_found")
    avg_txt = f"{avg:.2f}" if avg is not None else "n/a"

    # order: FAILED_FOUND first, then EXHAUSTED, then ERROR; by index within group
    order = {"FAILED_FOUND": 0, "EXHAUSTED": 1, "ERROR": 2}
    ordered = sorted(samples, key=lambda s: (order.get(s["status"], 9), s["index"]))
    cards = "\n".join(card_html(s) for s in ordered)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Run report</title>
<style>
 body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 0; background:#f6f8fa; color:#1f2328; }}
 header {{ position:sticky; top:0; background:#fff; border-bottom:1px solid #d0d7de; padding:14px 20px; }}
 h1 {{ font-size:18px; margin:0 0 8px; }}
 .chip, .badge {{ color:#fff; border-radius:10px; padding:2px 8px; font-size:12px; }}
 .stats {{ color:#57606a; font-size:13px; margin-top:6px; }}
 .wrap {{ padding:16px 20px; }}
 .card {{ background:#fff; border:1px solid #d0d7de; border-radius:8px; margin:8px 0; padding:8px 12px; }}
 .card > summary {{ cursor:pointer; list-style:none; }}
 .head {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
 .idx {{ color:#57606a; font-variant-numeric:tabular-nums; }}
 .meta {{ color:#57606a; font-size:12px; }}
 .seed {{ font-weight:600; flex:1; min-width:200px; }}
 .field {{ margin:10px 0; }} .field b {{ color:#57606a; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
 .field > div {{ margin-top:2px; }}
 .err {{ color:#cf222e; font-family:ui-monospace,monospace; margin-top:8px; white-space:pre-wrap; }}
 .answer pre {{ white-space:pre-wrap; background:#f6f8fa; border:1px solid #d0d7de; border-radius:6px; padding:10px; max-height:480px; overflow:auto; }}
 .viewlink {{ display:inline-block; color:#0969da; text-decoration:none; font-weight:600; background:#ddf4ff; border:1px solid #54aeff; border-radius:6px; padding:6px 10px; }}
 .viewlink:hover {{ background:#b6e3ff; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; }}
 th, td {{ border:1px solid #d0d7de; padding:4px 8px; text-align:left; vertical-align:top; }}
 details details {{ margin-top:8px; }} summary {{ outline:none; }}
</style></head><body>
<header>
 <h1>Run report — {esc(index.get('model'))} · {len(samples)} seeds</h1>
 <div>{chips}</div>
 <div class="stats">total ${index.get('total_cost_usd',0):.4f} · {index.get('total_claude_calls',0)} LLM calls ·
   avg make-harder calls / FAILED_FOUND = {avg_txt}</div>
</header>
<div class="wrap">{cards}</div>
</body></html>"""


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python summarize_run.py <run_dir>")
    run_dir = Path(sys.argv[1])
    index, samples = load_run(run_dir)
    console_summary(index, samples)

    # Per-answer pages with resolved references (linked from the report cards).
    ans_dir = run_dir / "answers"
    ans_dir.mkdir(exist_ok=True)
    n_pages = 0
    n_compare = 0
    for s in samples:
        if s.get("result") is None:
            continue
        fp = run_dir / s["file"]
        (ans_dir / f"{fp.stem}.html").write_text(render_sample(fp))
        n_pages += 1
        if s.get("compare"):
            cpath = run_dir / s["file"].replace(".json", ".compare.json")
            (ans_dir / f"{fp.stem}.compare.html").write_text(render_compare(cpath))
            n_compare += 1

    out = run_dir / "report.html"
    out.write_text(build_html(index, samples))
    print(f"\nHTML report written to: {out}")
    print(f"Answer pages ({n_pages}) written to: {ans_dir}/")
    if n_compare:
        print(f"Claude comparison pages ({n_compare}) written to: {ans_dir}/")


if __name__ == "__main__":
    main()
