#!/usr/bin/env python3
"""
One page for a whole research_loop.py run: rounds, strategy evolution, verification.

The per-round pages already exist -- summarize_run.py writes report.html per prompt side
and feedback_viewer.py writes feedback.html per round. What was missing is the view ACROSS
rounds: how the menu and ban list changed, what each round yielded per prompt, and which
harvested questions survived criterion verification (and why the rest did not). This page
is that view, and links out to the per-round pages rather than duplicating them.

    python loop_report.py runs/loop_250
    open runs/loop_250/loop_report.html

Reads (all optional except loop.json):
    loop.json                          the manifest
    round_KK/example_strategies.txt    the menu that round ran with
    round_KK/banned_strategies.txt     the ban list that round ran with
    round_KK/few_shots.{side}.json     the worked examples each prompt was shown
    round_KK/{side}/index.json         per-side status counts
    round_KK/feedback.html             linked if feedback_viewer.py has been run
    round_KK/{side}/report.html        linked if summarize_run.py has been run
    verified.json                      the post-hoc criterion check

The per-round pages are generated automatically when missing (summarize_run.py per prompt
side, feedback_viewer.py per round -- both idempotent and API-free). Pass --no-subpages to
skip that, or --regenerate-subpages to rebuild them all.
"""

import argparse
import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

SIDES = ("explore", "exploit")
HERE = Path(__file__).resolve().parent

CSS = """
:root { --bg:#0e1116; --panel:#161b22; --text:#e6edf3; --muted:#8b949e;
        --line:#30363d; --accent:#58a6ff; --good:hsl(120,62%,58%); --bad:hsl(6,72%,62%);
        --warn:hsl(38,80%,60%); }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       background:var(--bg); color:var(--text); }
header { padding:16px 22px; background:var(--panel); border-bottom:1px solid var(--line); }
header h1 { margin:0 0 10px; font-size:17px; }
.meta { color:var(--muted); font-size:12px; display:flex; flex-wrap:wrap; gap:6px 18px; }
.meta b { color:var(--accent); font-weight:600; }
main { padding:22px; max-width:1180px; margin:0 auto; }
h2 { font-size:13px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
     margin:32px 0 12px; border-bottom:1px solid var(--line); padding-bottom:6px; }
.note { color:var(--muted); font-size:12px; margin:-6px 0 14px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; color:var(--muted); font-weight:600; padding:6px 10px;
     border-bottom:1px solid var(--line); }
td { padding:7px 10px; border-bottom:1px solid #1c222b; vertical-align:top; }
tr:hover td { background:#12171e; }
.num { font-variant-numeric:tabular-nums; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
        padding:14px 16px; margin-bottom:14px; }
.card h3 { margin:0 0 4px; font-size:14px; }
.card .sub { color:var(--muted); font-size:12px; margin-bottom:10px; }
ul.strat { list-style:none; padding:0; margin:6px 0; }
ul.strat li { padding:4px 8px; border-left:3px solid var(--line); margin-bottom:3px;
              font-size:12.5px; color:#c9d1d9; }
ul.strat li.added { border-left-color:var(--good); background:#3fb95012; }
ul.strat li.dropped { border-left-color:var(--bad); background:#f8514912; color:var(--muted);
                      text-decoration:line-through; }
ul.strat li.novel { border-left-color:var(--accent); background:#58a6ff12; }
.tag { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
       padding:1px 6px; border-radius:999px; margin-right:6px; }
.tag.added { background:#3fb95022; color:var(--good); }
.tag.dropped { background:#f8514922; color:var(--bad); }
.tag.novel { background:#58a6ff22; color:var(--accent); }
.badge { font-size:11px; font-weight:700; padding:2px 8px; border-radius:999px; white-space:nowrap; }
.badge.correct { background:#3fb95022; color:var(--good); }
.badge.almost_correct, .badge.partly_correct { background:#d2992222; color:var(--warn); }
.badge.incorrect { background:#f8514922; color:var(--bad); }
.badge.insufficient_evidence { background:#8b949e22; color:var(--muted); }
.badge.error { background:#f8514922; color:var(--bad); }
details { margin:4px 0; }
summary { cursor:pointer; color:var(--muted); font-size:12px; }
summary:hover { color:var(--accent); }
.crit { font-size:12.5px; color:#c9d1d9; margin:5px 0; }
.lbl { display:inline-block; min-width:78px; color:var(--muted); font-size:11px;
       text-transform:uppercase; letter-spacing:.04em; margin-right:6px; }
.bar { display:inline-block; height:9px; border-radius:2px; vertical-align:middle; }
.q { color:#e6edf3; font-size:13px; margin:2px 0 6px; }
.q a { color:#e6edf3; border-bottom:1px dotted var(--accent); }
.q a:hover { color:var(--accent); text-decoration:none; }
.mono { font-family:ui-monospace,Menlo,monospace; }
.side-by-side { display:flex; gap:14px; flex-wrap:wrap; }
.side-by-side > div { flex:1 1 320px; }
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def read_lines(path: Path) -> list:
    if not path.exists():
        return []
    return [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def hms(seconds) -> str:
    s = int(round(seconds or 0))
    if s < 60:
        return f"{s}s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m" if h else f"{m}m {sec}s"


def side_stats(round_dir: Path) -> dict:
    """Per-prompt status counts for one round, from each side's index.json."""
    out = {}
    for side in SIDES:
        idx = round_dir / side / "index.json"
        if not idx.exists():
            continue
        rows = json.loads(idx.read_text()).get("samples") or []
        counts = Counter(r.get("status") for r in rows)
        ff = counts.get("FAILED_FOUND", 0)
        att = [r["attempts"] for r in rows if r.get("status") == "FAILED_FOUND"]
        out[side] = {
            "n": len(rows), "ff": ff,
            "rate": ff / len(rows) if rows else 0.0,
            "attempts": (sum(att) / len(att)) if att else None,
            "cost": sum(r.get("cost_usd", 0.0) for r in rows),
            "counts": dict(sorted(counts.items())),
            "report": (round_dir / side / "report.html") if (round_dir / side / "report.html").exists() else None,
        }
    return out


def ensure_round_pages(loop_dir: Path, python: str, force: bool = False) -> None:
    """Generate the per-round pages this report links to, if they are missing.

    summarize_run.py writes report.html + answers/ per prompt side, and feedback_viewer.py
    writes feedback.html per round. Both are cheap, idempotent, and make no API calls, so
    running them here removes the ordering trap: without answers/ pages the verification
    rows silently degrade to unclickable text.
    """
    for side_dir in sorted(loop_dir.glob("round_*/*/")):
        if not (side_dir / "index.json").exists():
            continue                      # not a prompt-side run dir
        if not force and (side_dir / "answers").is_dir() and (side_dir / "report.html").exists():
            continue
        print(f"[pages] summarize_run.py {side_dir}", flush=True)
        r = subprocess.run([python, str(HERE / "summarize_run.py"), str(side_dir)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[pages] WARNING failed ({r.returncode}): "
                  f"{(r.stderr or r.stdout).strip().splitlines()[-1:]}", file=sys.stderr)

    for fb in sorted(loop_dir.glob("round_*/feedback.json")):
        if not force and fb.with_suffix(".html").exists():
            continue
        print(f"[pages] feedback_viewer.py {fb}", flush=True)
        r = subprocess.run([python, str(HERE / "feedback_viewer.py"), str(fb)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[pages] WARNING failed ({r.returncode}): "
                  f"{(r.stderr or r.stdout).strip().splitlines()[-1:]}", file=sys.stderr)


def index_samples(loop_dir: Path) -> dict:
    """Map every question text -> where its seed's answer page and attempt trail live.

    verified.json records source_run and the attempt index but not which round dir or
    sample file a question came from, so recover it by scanning the samples. The value
    carries the answers/ page written by summarize_run.py (via view_answer.render_sample,
    which renders EVERY attempt with its answer and resolved citations) plus the attempt
    trail, so a verified question can be traced back to what the system actually replied.
    """
    idx = {}
    for sample in sorted(loop_dir.glob("round_*/*/sample_*.json")):
        try:
            data = json.loads(sample.read_text())
        except (OSError, ValueError):
            continue
        side_dir = sample.parent
        page = side_dir / "answers" / f"{sample.stem}.html"
        for res in data.get("results") or []:
            trail = [{"attempt": a.get("attempt"),
                      "verdict": ((a.get("judgment") or {}).get("verdict")
                                  if isinstance(a.get("judgment"), dict) else None),
                      "question": (a.get("harder") or {}).get("updated_question") or res.get("seed"),
                      "strategy": (a.get("harder") or {}).get("chosen_strategy") or ""}
                     for a in res.get("attempts") or []]
            entry = {
                "round": side_dir.parent.name,
                "side": side_dir.name,
                "sample": sample.stem,
                "status": res.get("final_status"),
                "page": (page.relative_to(loop_dir).as_posix() if page.exists() else None),
                "trail": trail,
            }
            for key in [res.get("seed")] + [t["question"] for t in trail]:
                if key:
                    idx.setdefault(key, entry)
    return idx


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def header_html(lm: dict, loop_dir: Path) -> str:
    v = lm.get("verification") or {}
    gen = sum(r.get("cost_usd", 0.0) for r in lm["rounds"])
    clu = sum(r.get("clustering_cost_usd") or 0.0 for r in lm["rounds"])
    ff = sum(r.get("num_failed_found", 0) for r in lm["rounds"])
    n = sum(r.get("num_seeds", 0) for r in lm["rounds"])
    bits = [
        f'<span><b>{ff}</b>/{n} broke the system</span>',
        f'<span><b>{v.get("kept", "?")}</b> verified usable</span>' if v else "",
        (f'<span>1 round, no feedback</span>' if len(lm["rounds"]) == 1
         else f'<span>{len(lm["rounds"])} rounds x {lm.get("feedback_every")}</span>'),
        (f'<span>both prompts per seed</span>' if lm.get("both_prompts")
         else f'<span>mix {lm.get("prompt_mix")} explore</span>'),
        # older manifests carry feedback_scope instead; the loop is always cumulative now
        (f'<span>quota {lm["ban_after_failures"]} failures</span>'
         if lm.get("ban_after_failures") is not None
         else f'<span>scope {lm.get("feedback_scope")}</span>'),
        f'<span>{esc(lm.get("model"))}</span>',
        f'<span>gen <b>${gen:.2f}</b> + cluster <b>${clu:.2f}</b>'
        + (f' + verify <b>${v["cost_usd"]:.2f}</b>' if v else "")
        + f' = <b>${lm.get("total_cost_usd", 0):.2f}</b></span>',
        f'<span>{hms(lm.get("elapsed_s"))}</span>',
        f'<span>started {esc((lm.get("started_at") or "")[:19])}</span>',
    ]
    if lm.get("stopped_early"):
        bits.append(f'<span style="color:hsl(6,72%,62%)">STOPPED EARLY: {esc(lm["stopped_early"])}</span>')
    return (f'<header><h1>{esc(loop_dir.name)} &mdash; loop report</h1>'
            f'<div class="meta">{"".join(b for b in bits if b)}</div></header>')


def rounds_table(lm: dict, loop_dir: Path) -> str:
    rows = []
    for r in lm["rounds"]:
        rd = loop_dir / f"round_{r['round']:02d}"
        st = side_stats(rd)
        cells = []
        for side in SIDES:
            s = st.get(side)
            if not s:
                cells.append('<td class="num">&mdash;</td>')
                continue
            link = (f'<a href="{esc(s["report"].relative_to(loop_dir))}">{s["ff"]}/{s["n"]}</a>'
                    if s["report"] else f'{s["ff"]}/{s["n"]}')
            w = max(2, round(s["rate"] * 90))
            cells.append(
                f'<td class="num">{link} <span style="color:var(--muted)">({s["rate"]:.0%})</span>'
                f'<br><span class="bar" style="width:{w}px;background:var(--good)"></span>'
                f'<br><span style="color:var(--muted);font-size:11px">'
                f'{"att %.1f" % s["attempts"] if s["attempts"] else ""} &middot; ${s["cost"]:.2f}</span></td>')
        fb = rd / "feedback.html"
        fb_cell = (f'<a href="{esc(fb.relative_to(loop_dir))}">clusters</a>'
                   if fb.exists() else '<span style="color:var(--muted)">not generated</span>')
        rows.append(
            f'<tr><td class="mono">r{r["round"]}</td>'
            f'<td class="num">{r.get("num_seeds", 0)}</td>'
            + "".join(cells)
            + f'<td class="num">{len(r.get("example_strategy_pool") or r.get("example_strategies") or [])} / '
              f'{len(r.get("banned_strategies") or [])}</td>'
              f'<td class="num">${r.get("cost_usd", 0):.2f}'
            + (f' + ${r["clustering_cost_usd"]:.2f}' if r.get("clustering_cost_usd") else "")
            + f'</td><td class="num">{hms(r.get("generation_elapsed_s"))}'
            + (f' + {hms(r["feedback_elapsed_s"])}' if r.get("feedback_elapsed_s") else "")
            + f'</td><td>{fb_cell}</td></tr>')
    return (
        '<h2>Rounds</h2>'
        '<p class="note">FAILED_FOUND per prompt (the objective: the answering system failed). '
        'Links open the per-seed report written by summarize_run.py.</p>'
        '<table><tr><th>round</th><th>seeds</th><th>explore</th><th>exploit</th>'
        '<th>menu / bans</th><th>cost</th><th>time</th><th>feedback</th></tr>'
        + "".join(rows) + '</table>')


def strategies_html(lm: dict, loop_dir: Path) -> str:
    """Per round: the menu it ran with, marked up against the previous round's menu."""
    blocks, prev = [], None
    for r in lm["rounds"]:
        k = r["round"]
        rd = loop_dir / f"round_{k:02d}"
        menu = (r.get("example_strategy_pool") or r.get("example_strategies")
                or read_lines(rd / "example_strategy_pool.txt")
                or read_lines(rd / "example_strategies.txt"))
        bans = r.get("banned_strategies") or read_lines(rd / "banned_strategies.txt")
        items = []
        for s in menu:
            # a lowercase first letter marks an LLM-named novel cluster promoted onto the menu
            novel = s[:1].islower()
            cls = "added" if prev is not None and s not in prev else ("novel" if novel else "")
            tag = ('<span class="tag added">new this round</span>' if cls == "added" else "")
            if novel and cls != "added":
                tag = '<span class="tag novel">discovered</span>'
            items.append(f'<li class="{cls}">{tag}{esc(s)}</li>')
        if prev is not None:
            for s in prev:
                if s not in menu:
                    items.append(f'<li class="dropped"><span class="tag dropped">dropped</span>{esc(s)}</li>')
        shots = []
        for side in SIDES:
            f = rd / f"few_shots.{side}.json"
            if not f.exists():
                continue
            ss = json.loads(f.read_text())
            rows = "".join(
                f'<div class="crit"><span class="lbl">{i + 1}</span>{esc(s.get("chosen_strategy"))}'
                f'<br><span class="lbl">question</span>{esc(s.get("updated_question"))}</div>'
                for i, s in enumerate(ss))
            shots.append(f'<div><details><summary>{side} few-shots ({len(ss)})</summary>{rows}</details></div>')
        nm = r.get("next_menus") or {}
        prov = ""
        if nm:
            fresh = nm.get("fresh_bans") or []
            prov = ('<div class="sub">feedback &rarr; next round: '
                    f'pool of {nm.get("exploit_pool_size", nm.get("num_example_strategies", "?"))}, '
                    f'{nm.get("num_banned_strategies", "?")} bans'
                    + (f', {len(nm.get("quota_bans") or fresh)} over quota'
                       if (nm.get("quota_bans") or fresh) else ', none over quota')
                    + '</div>')
        blocks.append(
            f'<div class="card"><h3>round {k} &mdash; {len(menu)} pool, {len(bans)} banned</h3>'
            + prov
            + f'<ul class="strat">{"".join(items)}</ul>'
            + '<details><summary>ban list (shown to explore)</summary>'
            + "".join(f'<div class="crit">{esc(b)}</div>' for b in bans) + '</details>'
            + f'<div class="side-by-side">{"".join(shots)}</div></div>')
        prev = menu
    return ('<h2>Strategy evolution</h2>'
            '<p class="note">The menu is shown to the exploit prompt; the ban list to explore. '
            'Green = added by the previous round\'s feedback, red = dropped, '
            'blue = an off-menu strategy the clustering discovered and promoted.</p>'
            + "".join(blocks))


def verification_html(loop_dir: Path, samples: dict) -> str:
    path = loop_dir / "verified.json"
    if not path.exists():
        return ('<h2>Verification</h2><p class="note">No verified.json &mdash; run '
                '<span class="mono">verify_questions.py</span> on this directory.</p>')
    v = json.loads(path.read_text())
    qs = v.get("questions") or []
    t = v.get("totals") or {}
    by_side = {}
    for side in SIDES:
        sub = [q for q in qs if q.get("source_run") == side]
        if sub:
            by_side[side] = (sum(1 for q in sub if q.get("kept")), len(sub))
    lab = Counter(q.get("label") for q in qs)
    chips = " ".join(f'<span class="badge {esc(k)}">{esc(k)} {n}</span>' for k, n in lab.most_common())
    sides = " &middot; ".join(f"{s}: <b>{k}</b>/{n} ({k / n:.0%})" for s, (k, n) in by_side.items())

    rows = []
    for q in sorted(qs, key=lambda x: (bool(x.get("kept")), x.get("label") or "")):
        label = q.get("label") or "?"
        crit = q.get("criterion_original") or ""
        extra = ""
        if q.get("criterion_rewritten"):
            extra = (f'<div class="crit"><span class="lbl">rewritten</span>{esc(q.get("criterion"))}</div>')
        why = q.get("main_correctness_problem") or ""
        unfair = q.get("unfair_requirements") or ""
        reasoning = q.get("reasoning") or q.get("error") or ""
        hit = samples.get(q.get("question")) or samples.get(q.get("seed_question")) or {}
        qtext = esc(q.get("question"))
        qcell = (f'<a href="{esc(hit["page"])}">{qtext}</a>' if hit.get("page") else qtext)
        where = ""
        if hit:
            where = (f'<div class="crit"><span class="lbl">from</span>'
                     f'<span class="mono">{esc(hit["round"])}/{esc(hit["side"])}/'
                     f'{esc(hit["sample"])}</span> &middot; {esc(hit.get("status"))}'
                     + (f' &middot; <a href="{esc(hit["page"])}">answer + citations, '
                        f'all {len(hit["trail"])} attempt(s)</a>' if hit.get("page")
                        else ' &middot; <span style="color:var(--muted)">no answer page: run '
                             'summarize_run.py on that side</span>')
                     + '</div>')
            if hit.get("trail"):
                tr = "".join(
                    f'<tr><td class="num">{esc(t["attempt"])}</td>'
                    f'<td><span class="badge {"correct" if t["verdict"] == "FAILED" else "insufficient_evidence"}">'
                    f'{esc(t["verdict"] or "-")}</span></td>'
                    f'<td>{esc((t["question"] or "")[:130])}'
                    + (f'<br><span style="color:var(--muted);font-size:11px">'
                       f'{esc(t["strategy"][:120])}</span>' if t["strategy"] else "")
                    + '</td></tr>' for t in hit["trail"])
                where += ('<details><summary>attempt trail on this seed '
                          f'({len(hit["trail"])})</summary><table>'
                          '<tr><th>attempt</th><th>judge</th><th>question / strategy</th></tr>'
                          f'{tr}</table></details>')
        rows.append(
            f'<tr><td><span class="badge {esc(label)}">{esc(label)}</span></td>'
            f'<td style="color:var(--muted);font-size:11px">{esc(q.get("source_run"))}<br>'
            f'r{esc(q.get("round"))}</td>'
            f'<td><div class="q">{qcell}</div>'
            + (f'<div class="crit" style="color:var(--bad)"><span class="lbl">problem</span>{esc(why)}</div>' if why else "")
            + (f'<div class="crit" style="color:var(--warn)"><span class="lbl">unfair</span>'
               f'{esc(unfair)}</div>' if unfair else "")
            + extra
            + where
            + f'<details><summary>criterion + judge reasoning</summary>'
              f'<div class="crit"><span class="lbl">criterion</span>{esc(crit)}</div>'
              f'<div class="crit"><span class="lbl">reasoning</span>{esc(reasoning)}</div>'
              f'<div class="crit"><span class="lbl">seed</span>{esc(q.get("seed_question"))}</div>'
              f'</details></td></tr>')
    return (
        '<h2>Verification</h2>'
        f'<p class="note">Kept <b>{t.get("kept")}</b>/{t.get("checked")} '
        f'({(t.get("keep_rate") or 0):.0%}), {t.get("criteria_rewritten")} with a rewritten '
        f'criterion, ${(t.get("cost_usd") or 0):.2f}. {sides}<br>{chips}</p>'
        '<p class="note">Dropped questions first &mdash; those are the criteria the meta-judge '
        'found unsupported, i.e. the generator\'s own failure modes.</p>'
        '<table><tr><th>label</th><th>from</th><th>question</th></tr>'
        + "".join(rows) + '</table>')


def build(loop_dir: Path, out: Path) -> Path:
    lm = json.loads((loop_dir / "loop.json").read_text())
    samples = index_samples(loop_dir)
    body = (header_html(lm, loop_dir)
            + '<main>'
            + rounds_table(lm, loop_dir)
            + strategies_html(lm, loop_dir)
            + verification_html(loop_dir, samples)
            + '</main>')
    out.write_text(
        f'<!doctype html><meta charset="utf-8"><title>{esc(loop_dir.name)} loop report</title>'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<style>{CSS}</style>{body}')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("loop_dir", type=Path, help="A research_loop.py --out-dir (holds loop.json).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output HTML (default: <loop_dir>/loop_report.html).")
    p.add_argument("--no-subpages", action="store_true",
                   help="Do not generate missing per-round pages; link only what already "
                        "exists (rows without an answer page stay unclickable).")
    p.add_argument("--regenerate-subpages", action="store_true",
                   help="Rebuild every per-round page even if it already exists.")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter used for the per-round generators.")
    args = p.parse_args()
    if not (args.loop_dir / "loop.json").exists():
        p.error(f"{args.loop_dir}/loop.json not found — is that a research_loop.py output dir?")
    if not args.no_subpages:
        ensure_round_pages(args.loop_dir, args.python, force=args.regenerate_subpages)
    out = build(args.loop_dir, args.out or (args.loop_dir / "loop_report.html"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
