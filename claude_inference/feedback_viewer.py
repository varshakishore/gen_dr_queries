"""Render a strategy_feedback JSON (from build_strategy_feedback.py) as standalone HTML.

Shows, in one self-contained page (no server, no external assets):
  * a cluster-comparison table — every cluster (in seeded mode: each seed strategy plus
    novel clusters) with its FAIL_RATE broken down per source_run, so generation prompts
    (e.g. original vs explore) can be compared strategy by strategy; and
  * every strategy cluster as a card, in `rank_by` order, with its statistics (failure rate,
    cluster size, failed/not-failed split, share of the round, seed vs new), its per-subset
    rates and its expandable few-shot failures. Nothing is filtered out — the whole ranked
    list is shown, weakest clusters included.

Colour language matches EvalTree/build_viewer.py: green = higher failure rate (a strategy
that stumps the agent, which is GOOD here). Use standalone or let build_strategy_feedback
emit `<out>.html` automatically.

    python feedback_viewer.py runs/compare/strategy_feedback_seeded.json   # writes .html alongside
    python feedback_viewer.py in.json --out view.html
"""

import argparse
import html
import json
import math
from pathlib import Path

# Stable identity colours per source_run (assigned by order); used for pie slices + legend.
_SUBSET_PALETTE = ["#58a6ff", "#a371f7", "#e3b341", "#3fb950", "#f778ba", "#f0883e"]

# Pie sizing: AREA ∝ cluster size (so radius ∝ sqrt(size)); biggest cluster -> _R_MAX.
# _R_MIN is only a legibility floor for near-empty clusters, not an additive offset, so
# sizes stay honestly comparable (a 2x-area pie means a ~2x-bigger cluster).
_R_MAX = 26.0
_R_MIN = 4.0


def _hue(rate: float) -> float:
    """0.0 -> red (0), 1.0 -> green (120): higher failure reads greener."""
    return max(0.0, min(1.0, rate)) * 120.0


def _short(run: str) -> str:
    return run.rsplit("_", 1)[-1] if "_" in run else run


def _subset_colors(source_runs: list[str]) -> dict:
    return {run: _SUBSET_PALETTE[i % len(_SUBSET_PALETTE)] for i, run in enumerate(source_runs)}


def _radius(size: int, max_size: int) -> float:
    if size <= 0 or max_size <= 0:
        return _R_MIN
    return _R_MIN + (_R_MAX - _R_MIN) * math.sqrt(size / max_size)


def _slice_path(cx: float, cy: float, r: float, a0: float, a1: float) -> str:
    x0, y0 = cx + r * math.sin(a0), cy - r * math.cos(a0)
    x1, y1 = cx + r * math.sin(a1), cy - r * math.cos(a1)
    large = 1 if (a1 - a0) > math.pi else 0
    return f"M{cx:.2f},{cy:.2f} L{x0:.2f},{y0:.2f} A{r:.2f},{r:.2f} 0 {large} 1 {x1:.2f},{y1:.2f} Z"


_FAIL_COLOR = "hsl(120,62%,46%)"   # green = failed (stumps the agent — good)
_PASS_COLOR = "#2a2f3a"            # grey  = passed


def _pie_cell(size: int, failed: int, max_size: int, scope: str = "", href: str | None = None) -> str:
    """One pie: AREA ∝ `size` (comparable across the whole table), green slice = fraction
    FAILED, grey = passed. Used per scope (overall / original / explore), so subset pies
    can be compared to each other and to the overall pie. If `href` is given (and the scope
    is non-empty), the pie links to a static detail page listing its instances."""
    box = _R_MAX * 2
    cx = cy = box / 2
    if size <= 0:
        return (f'<span class="pie empty" title="{html.escape(scope)} 0 questions">'
                f'<svg width="{box:.0f}" height="{box:.0f}" viewBox="0 0 {box:.0f} {box:.0f}">'
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{_R_MIN:.1f}" fill="none" '
                f'stroke="#3a414c" stroke-dasharray="2 2"/></svg><span class="n">0</span></span>')
    r = _radius(size, max_size)
    rate = failed / size
    if failed == 0:
        shape = f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{_PASS_COLOR}"/>'
    elif failed >= size:
        shape = f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{_FAIL_COLOR}"/>'
    else:
        a = 2 * math.pi * rate
        shape = (f'<path d="{_slice_path(cx, cy, r, 0.0, a)}" fill="{_FAIL_COLOR}"/>'
                 f'<path d="{_slice_path(cx, cy, r, a, 2 * math.pi)}" fill="{_PASS_COLOR}"/>')
    title = f"{scope + ': ' if scope else ''}{size} q, {failed} failed ({rate:.0%})"
    if href:
        title += " — click to inspect"
    inner = (
        f'<svg width="{box:.0f}" height="{box:.0f}" viewBox="0 0 {box:.0f} {box:.0f}">{shape}</svg>'
        f'<span class="lab"><span class="num" style="color:hsl({_hue(rate)},62%,62%)">{rate:.2f}</span>'
        f'<span class="n">{failed}/{size}</span></span>'
    )
    cls = "pie" + (" link" if href else "")
    tag = "a" if href else "span"
    hattr = f' href="{html.escape(href)}"' if href else ""
    return f'<{tag} class="{cls}"{hattr} title="{html.escape(title)}">{inner}</{tag}>'


def _legend(source_runs: list[str], colors: dict) -> str:
    return ('<div class="legend">Each pie: AREA ∝ #questions (compare cluster sizes across '
            'rows and across <b>overall / ' + " / ".join(html.escape(_short(r)) for r in source_runs) +
            '</b>); <i>green</i> = fraction FAILED (stumps the agent — good), grey = passed. '
            'Number = FAIL_RATE.</div>')


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)


def _page_name(cluster_id: str, scope_key: str) -> str:
    return f"{_safe(cluster_id)}__{_safe(scope_key)}.html"


def _scope_instances(instances: list[dict], scope_key: str) -> list[dict]:
    """All instances (overall) or just those from one source_run subset."""
    if scope_key == "overall":
        return instances
    return [x for x in instances if x.get("source_run") == scope_key]


def _detail_page(meta: dict, cluster: dict, scope_key: str, scope_label: str,
                 instances: list[dict], back_href: str) -> str:
    """A standalone HTML page listing every instance for one (cluster, scope)."""
    failed = sum(1 for x in instances if str(x.get("drtulu_verdict", "")).upper() == "FAILED")
    n = len(instances)
    rate = failed / n if n else 0.0
    rows = []
    for x in instances:
        verdict = str(x.get("drtulu_verdict", "")).upper()
        vcls = "failed" if verdict == "FAILED" else "passed"
        rnd = x.get("round")
        meta_line = f'{html.escape(x.get("source_run", ""))}'
        if rnd is not None:
            meta_line += f' · round {html.escape(str(rnd))}'
        meta_line += f' · #{x.get("index")}'
        rows.append(
            '<div class="inst">'
            f'<div class="ihead"><span class="badge {vcls}">{html.escape(verdict or "?")}</span>'
            f'<span class="imeta">{meta_line}</span></div>'
            f'<div class="q"><span class="lbl">seed</span>{html.escape(x.get("seed_question", ""))}</div>'
            f'<div class="q hard"><span class="lbl">→ harder</span>{html.escape(x.get("updated_question", ""))}</div>'
            f'<div class="m"><span class="lbl">strategy</span>{html.escape(x.get("strategy", ""))}</div>'
            f'<div class="m"><span class="lbl">criterion</span>{html.escape(x.get("verification_criterion", ""))}</div>'
            "</div>"
        )
    body = "".join(rows) or '<p class="muted">No instances.</p>'
    title = f'{cluster["cluster_id"]} — {scope_label}'
    return _PAGE_TEMPLATE.format(
        title=html.escape(title),
        back=html.escape(back_href),
        cluster_id=html.escape(cluster["cluster_id"]),
        scope=html.escape(scope_label),
        desc=html.escape(cluster.get("description", "")),
        stats=f"{n} questions · {failed} FAILED · {rate:.0%} fail rate",
        rows=body,
    )


def build_pages(feedback: dict, pages_dirname: str, main_filename: str):
    """Build one detail page per (cluster, scope). Returns (href_map, files).

    href_map: {(cluster_id, scope_key): href-relative-to-main}. files: {filename: html}.
    scope_key is 'overall' or a full source_run name; only non-empty scopes get a page.
    """
    meta = feedback.get("meta", {})
    source_runs = meta.get("source_runs", [])
    back_href = f"../{main_filename}"
    href_map: dict = {}
    files: dict = {}
    for c in feedback.get("cluster_comparison", []):
        cid = c["cluster_id"]
        instances = c.get("instances", [])
        for scope_key, scope_label in [("overall", "all questions")] + [(r, _short(r)) for r in source_runs]:
            scoped = _scope_instances(instances, scope_key)
            if not scoped:
                continue
            fname = _page_name(cid, scope_key)
            files[fname] = _detail_page(meta, c, scope_key, scope_label, scoped, back_href)
            href_map[(cid, scope_key)] = f"{pages_dirname}/{fname}"
    return href_map, files


def _comparison_table(feedback: dict, source_runs: list[str], colors: dict, href_map: dict) -> str:
    rows = feedback.get("cluster_comparison", [])
    if not rows:
        return "<p class='muted'>No cluster_comparison in this feedback file.</p>"
    # One shared scale so every pie (overall and per-subset) is size-comparable.
    max_size = max((c["num_questions"] for c in rows), default=1)
    head = "".join(f"<th>{html.escape(_short(r))}</th>" for r in source_runs)
    body = []
    for c in rows:
        cid = c["cluster_id"]
        overall = _pie_cell(c["num_questions"], c.get("num_failed", 0), max_size, "overall",
                            href_map.get((cid, "overall")))
        subset_cells = []
        for run in source_runs:
            b = c.get("by_source_run", {}).get(run, {})
            subset_cells.append(
                f'<td class="pcell">{_pie_cell(b.get("num_questions", 0), b.get("num_failed", 0), max_size, _short(run), href_map.get((cid, run)))}</td>'
            )
        body.append(
            "<tr>"
            f'<td class="cid">{html.escape(cid)}</td>'
            f'<td class="desc">{html.escape(c["description"])}</td>'
            f'<td class="pcell">{overall}</td>'
            + "".join(subset_cells)
            + "</tr>"
        )
    return (
        '<table class="cmp"><thead><tr>'
        "<th>cluster</th><th>strategy</th><th>overall</th>" + head +
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def _examples(examples: list[dict]) -> str:
    if not examples:
        return ""
    items = []
    for ex in examples:
        items.append(
            '<div class="ex">'
            f'<div class="q"><span class="lbl">seed</span>{html.escape(ex.get("seed_question", ""))}</div>'
            f'<div class="q hard"><span class="lbl">→ harder</span>{html.escape(ex.get("updated_question", ""))}</div>'
            f'<div class="m"><span class="lbl">strategy</span>{html.escape(ex.get("strategy", ""))}</div>'
            f'<div class="m"><span class="lbl">criterion</span>{html.escape(ex.get("verification_criterion", ""))}</div>'
            "</div>"
        )
    return (
        f"<details><summary>{len(examples)} few-shot failure example(s)</summary>"
        + "".join(items) + "</details>"
    )


def _cluster_cards(feedback: dict, source_runs: list[str], colors: dict, href_map: dict) -> str:
    """Every strategy cluster as a card, in the order the feedback ranked them.

    Nothing is filtered: unused clusters (0 questions this round) and clusters that never
    stumped the agent are shown too, just at the bottom of the ranking. Pies are scaled
    against cluster_comparison so they stay comparable with the table above.
    """
    clusters = feedback.get("strategy_clusters", [])
    if not clusters:
        return "<p class='muted'>No strategy_clusters in this feedback file.</p>"
    max_size = max((c["num_questions"] for c in feedback.get("cluster_comparison", [])),
                   default=max((s["num_questions"] for s in clusters), default=1))
    cards = []
    for s in clusters:
        cid = s["cluster_id"]
        size, failed = s["num_questions"], s.get("num_failed", 0)
        not_failed = s.get("num_not_failed", size - failed)
        hue = _hue(s["failure_rate"])
        pie = _pie_cell(size, failed, max_size, "overall", href_map.get((cid, "overall")))
        subset_bits = []
        for run in source_runs:
            b = s.get("by_source_run", {}).get(run, {})
            subset_bits.append(
                f'<span class="sub"><b>{html.escape(_short(run))}</b> '
                f'{_pie_cell(b.get("num_questions", 0), b.get("num_failed", 0), max_size, _short(run), href_map.get((cid, run)))}</span>'
            )
        kind = "seed" if s.get("is_seed") else "new"
        stats = (f'{size} question(s) · {failed} failed / {not_failed} not failed · '
                 f'{s["share"] * 100:.0f}% of the round · score {s["score"]:.2f}'
                 if size else
                 "Unused — the generator did not try this strategy at all this round.")
        cards.append(
            f'<div class="card{"" if size else " unused"}">'
            '<div class="chead">'
            f'<span class="rank">#{s["rank"]}</span>'
            f'<span class="cid">{html.escape(cid)}</span>'
            f'<span class="tag {kind}">{kind}</span>'
            f'<span class="big" style="color:hsl({hue},62%,62%)">{s["failure_rate"]:.2f} FAIL</span>'
            f'<span class="frac">{failed}/{size} · {round(s["share"] * 100)}% of round</span>'
            f'<span class="cardpie">{pie}</span>'
            "</div>"
            f'<div class="cdesc">{html.escape(s["description"])}</div>'
            f'<div class="subs">{"".join(subset_bits)}</div>'
            f'<div class="rat">{html.escape(stats)}</div>'
            + _examples(s.get("few_shot_failures", []))
            + "</div>"
        )
    return "".join(cards)


def _meta_html(meta: dict) -> str:
    bits = [
        ("mode", meta.get("clustering", {}).get("cluster_mode", "?")),
        ("dataset", Path(meta.get("dataset_dir", "")).name),
        ("model", meta.get("model", "")),
        ("instances", meta.get("num_instances", "")),
        ("overall FAIL", f'{meta.get("overall_failure_rate", 0):.2f}'),
        ("ranked by", meta.get("ranking", {}).get("rank_by", "?")),
    ]
    cl = meta.get("clustering", {})
    if cl.get("cluster_mode") == "seeded":
        bits += [
            ("assign", cl.get("assign_method", "?")),
            ("seeds used", f'{cl.get("num_seed_clusters_used", "?")}/{cl.get("num_seeds", "?")}'),
            ("new clusters", cl.get("num_new_clusters", "?")),
        ]
        if cl.get("assign_method") == "llm":
            bits.append(("cluster model", cl.get("cluster_model", "?")))
        else:
            bits += [
                ("max dist", cl.get("seed_max_distance", "?")),
                ("nearest-seed dist (min/mean/max)",
                 f'{cl.get("nearest_seed_distance_min")}/{cl.get("nearest_seed_distance_mean")}/{cl.get("nearest_seed_distance_max")}'),
            ]
    return " ".join(
        f'<span class="mi"><b>{html.escape(str(k))}</b> {html.escape(str(v))}</span>' for k, v in bits
    )


def build_html(feedback: dict, href_map: dict | None = None) -> str:
    href_map = href_map or {}
    meta = feedback.get("meta", {})
    source_runs = meta.get("source_runs", [])
    colors = _subset_colors(source_runs)
    title = "Strategy feedback — " + meta.get("clustering", {}).get("cluster_mode", "")
    clusters = feedback.get("strategy_clusters", [])
    rank_by = meta.get("ranking", {}).get("rank_by", "")
    return _TEMPLATE.format(
        title=html.escape(title),
        meta=_meta_html(meta),
        legend=_legend(source_runs, colors),
        comparison=_comparison_table(feedback, source_runs, colors, href_map),
        clusters_count=f"{len(clusters)} cluster(s)"
                       + (f", ranked by {html.escape(rank_by)}" if rank_by else ""),
        clusters=_cluster_cards(feedback, source_runs, colors, href_map),
        notes=html.escape(meta.get("notes", "")),
    )


def write_viewer(feedback: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Static detail pages (one per cluster × scope), written to a sibling directory; the
    # main page's pies link into them so any pie can be clicked to inspect its instances.
    pages_dirname = out_path.stem + "_pages"
    href_map, files = build_pages(feedback, pages_dirname, out_path.name)
    if files:
        pages_dir = out_path.with_name(pages_dirname)
        pages_dir.mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            (pages_dir / fname).write_text(content, encoding="utf-8")
    out_path.write_text(build_html(feedback, href_map), encoding="utf-8")
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
  :root {{
    --bg:#0e1116; --panel:#161b22; --text:#e6edf3; --muted:#8b949e;
    --line:#30363d; --accent:#58a6ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
    background:var(--bg); color:var(--text); }}
  header {{ padding:16px 22px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }}
  header h1 {{ margin:0 0 8px; font-size:17px; }}
  .meta {{ color:var(--muted); font-size:12px; display:flex; flex-wrap:wrap; gap:6px 16px; }}
  .mi b {{ color:var(--accent); font-weight:600; }}
  main {{ padding:22px; max-width:1200px; margin:0 auto; }}
  h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:26px 0 12px; }}
  .legend {{ color:var(--muted); font-size:12px; margin:-6px 0 14px; }}
  .legend i {{ color:hsl(120,62%,55%); font-style:normal; }}
  table.cmp {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.cmp th {{ text-align:left; color:var(--muted); font-weight:600; padding:6px 10px; border-bottom:1px solid var(--line); position:sticky; top:73px; background:var(--bg); }}
  table.cmp td {{ padding:7px 10px; border-bottom:1px solid #1c222b; vertical-align:top; }}
  table.cmp tr:hover td {{ background:#12171e; }}
  .cid {{ font-family:ui-monospace,Menlo,monospace; color:var(--accent); white-space:nowrap; }}
  .desc {{ max-width:520px; color:#c9d1d9; }}
  .cell {{ display:inline-flex; align-items:center; gap:6px; }}
  .cell.empty {{ color:#3a414c; }}
  .num {{ font-variant-numeric:tabular-nums; font-weight:600; }}
  .frac {{ color:var(--muted); font-size:11px; }}
  .pcell {{ text-align:left; white-space:nowrap; }}
  .pie {{ display:inline-flex; align-items:center; gap:7px; }}
  .pie svg {{ display:block; overflow:visible; flex:0 0 auto; }}
  .pie .lab {{ display:inline-flex; flex-direction:column; line-height:1.15; }}
  .pie .lab .num {{ font-size:12px; }}
  .pie .n {{ color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }}
  .pie.empty .lab, .pie.empty .n {{ color:#3a414c; }}
  a.pie.link {{ text-decoration:none; color:inherit; border-radius:8px; padding:2px 4px; cursor:pointer; transition:background .1s ease; }}
  a.pie.link:hover {{ background:#20262f; outline:1px solid var(--accent); }}
  .cardpie {{ margin-left:auto; }}
  .legend b {{ color:var(--accent); }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:12px; }}
  .chead {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
  .rank {{ font-weight:700; color:var(--muted); }}
  .tag {{ font-size:11px; padding:2px 8px; border-radius:999px; background:#1f6feb22; color:var(--accent); }}
  .tag.seed {{ background:#1f6feb22; color:var(--accent); }}
  .tag.new {{ background:#f0883e22; color:#f0883e; }}
  .card.unused {{ opacity:.62; }}
  h2 .count {{ text-transform:none; letter-spacing:0; font-weight:400; color:#6e7681; margin-left:8px; }}
  .big {{ font-weight:700; }}
  .cdesc {{ margin:10px 0; color:#c9d1d9; }}
  .subs {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin:8px 0; }}
  .sub b {{ color:var(--muted); font-weight:600; margin-right:5px; }}
  .rat {{ color:var(--muted); font-size:12px; font-style:italic; margin:6px 0 4px; }}
  details {{ margin-top:8px; }}
  summary {{ cursor:pointer; color:var(--accent); font-size:12px; }}
  .ex {{ border-left:2px solid var(--line); padding:6px 0 6px 12px; margin:10px 0; }}
  .ex .q {{ margin:2px 0; }}
  .ex .q.hard {{ color:#e6edf3; }}
  .ex .m {{ color:var(--muted); font-size:12px; margin:2px 0; }}
  .lbl {{ display:inline-block; min-width:64px; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; margin-right:6px; }}
  .muted {{ color:var(--muted); }}
</style></head>
<body>
<header><h1>{title}</h1><div class="meta">{meta}</div></header>
<main>
  <h2>Cluster comparison</h2>
  {legend}
  {comparison}
  <h2>Strategy clusters <span class="count">{clusters_count}</span></h2>
  <div class="legend">Every cluster, nothing filtered out — the ranking sets the order only.
  <b>seed</b> = off the seed strategy menu, <b>new</b> = discovered as novel. Clusters the
  generator never tried this round sit at the bottom with 0 questions.</div>
  {clusters}
  <h2>Notes</h2>
  <p class="muted">{notes}</p>
</main>
</body></html>
"""


_PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
  :root {{ --bg:#0e1116; --panel:#161b22; --text:#e6edf3; --muted:#8b949e; --line:#30363d; --accent:#58a6ff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; background:var(--bg); color:var(--text); }}
  header {{ padding:16px 22px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }}
  header a {{ color:var(--accent); text-decoration:none; font-size:12px; }}
  header h1 {{ margin:6px 0 4px; font-size:16px; }}
  header .cid {{ font-family:ui-monospace,Menlo,monospace; color:var(--accent); }}
  header .desc {{ color:#c9d1d9; margin:4px 0; max-width:900px; }}
  header .stats {{ color:var(--muted); font-size:12px; }}
  main {{ padding:20px; max-width:960px; margin:0 auto; }}
  .inst {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:12px; }}
  .ihead {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
  .badge {{ font-size:11px; font-weight:700; padding:2px 8px; border-radius:999px; }}
  .badge.failed {{ background:#3fb95022; color:hsl(120,62%,62%); }}
  .badge.passed {{ background:#8b949e22; color:var(--muted); }}
  .imeta {{ color:var(--muted); font-size:12px; }}
  .q {{ margin:3px 0; }}
  .q.hard {{ color:#e6edf3; }}
  .m {{ color:var(--muted); font-size:12.5px; margin:3px 0; }}
  .lbl {{ display:inline-block; min-width:70px; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; margin-right:6px; }}
  .muted {{ color:var(--muted); }}
</style></head>
<body>
<header>
  <a href="{back}">← back to comparison</a>
  <h1><span class="cid">{cluster_id}</span> · {scope}</h1>
  <div class="desc">{desc}</div>
  <div class="stats">{stats}</div>
</header>
<main>{rows}</main>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feedback_json", type=Path, help="a strategy_feedback*.json from build_strategy_feedback.py")
    ap.add_argument("--out", type=Path, default=None, help="output HTML (default: <input>.html)")
    args = ap.parse_args()
    feedback = json.loads(args.feedback_json.read_text())
    out_path = args.out or args.feedback_json.with_suffix(".html")
    write_viewer(feedback, out_path)
    print(f"wrote viewer -> {out_path}")


if __name__ == "__main__":
    main()
