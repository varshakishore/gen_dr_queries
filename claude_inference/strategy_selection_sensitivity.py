"""Sensitivity analysis of `strategy_feedback_module.select_focus` on a built feedback JSON.

Re-runs ONLY the selection stage (stage 3) of an existing feedback file over a grid of
filtering/ranking parameters and reports how many clusters land in `focus_strategies`
("good": fed back into the next round) vs `ineligible_strategies` ("bad": dropped, with a
reason). Nothing is re-clustered, re-embedded or re-scored: the cluster nodes are rebuilt
verbatim from `cluster_comparison`, so no API calls and no new numbers are computed.

    python strategy_selection_sensitivity.py round1_strategy_feedback.json \
        --out-html round1_selection_sensitivity.html

Emits a standalone dark HTML report (no server, no external assets) whose palette matches
feedback_viewer.py, plus the raw sweep as JSON. Contents: one-at-a-time sweeps of each
knob, a min_failure_rate x max_share grid, the always_include_new interaction, and a
rank_by ordering comparison.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from strategy_feedback_module import QuestionExample, few_shot_failures, select_focus

# Module defaults for `select_focus` / `build_feedback`, held fixed while one knob moves.
DEFAULTS = {
    "min_failure_rate": 0.3,
    "max_share": 0.5,
    "min_cluster_size": 0,
    "max_cluster_size": None,
    "rank_by": "underrepresented",
    "always_include_new": True,
    "examples_per_strategy": 5,
}

SWEEPS = [
    ("min_failure_rate", [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]),
    ("max_share", [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.25, 0.5, 1.0]),
    ("min_cluster_size", [0, 1, 2, 3, 4, 5, 6, 8, 10]),
    ("max_cluster_size", [None, 3, 4, 5, 6, 8, 9, 12]),
    ("always_include_new", [True, False]),
    ("rank_by", ["underrepresented", "diverse", "failure_rate", "volume"]),
    ("examples_per_strategy", [1, 2, 3, 5, 8, 10]),
]

SWEEP_NOTES = {
    "min_failure_rate": "How often a cluster must stump the agent to be eligible. "
                        "Failure is GOOD here, so this is a quality floor.",
    "max_share": "Drops over-represented catch-all clusters. Only bites once it falls "
                 "below the largest cluster's share.",
    "min_cluster_size": "Drops clusters with too few questions. At 0, unused seeds "
                        "(0 questions) survive as 'untried'.",
    "max_cluster_size": "Optional upper size cap — the inverse gate, dropping the "
                        "biggest themes rather than the smallest.",
    "always_include_new": "Force-includes every novel (off-menu) cluster with >=1 failure "
                          "and share <= max_share, bypassing the size/rate gates.",
    "rank_by": "Ordering only — it never changes which clusters are eligible.",
    "examples_per_strategy": "Few-shot failures attached per focus strategy. Deduped "
                             "globally, so it caps out at the number of failures.",
}

# --- palette ---------------------------------------------------------------
# Surface + ink match feedback_viewer.py (GitHub-dark). The good/bad pair is the one
# categorical decision here and is validated on the #0e1116 surface: adjacent CVD
# separation deutan/protan dE 29.3, tritan 30.4, normal 33.5, all six checks PASS.
# The obvious green/red pair (#3fb950/#f85149) FAILS at deutan dE 2.2 and is not used.
C_GOOD = "#388bfd"   # kept -> focus_strategies
C_BAD = "#db6d28"    # dropped -> ineligible_strategies
# Sequential single-hue ramp for failure_rate / heatmap magnitude (dim -> bright blue,
# monotonic in lightness; never a rainbow).
RAMP_LO = (0x14, 0x22, 0x33)
RAMP_HI = (0x79, 0xc0, 0xff)


def ramp(t: float) -> str:
    """Sequential blue for magnitude t in [0,1]: dim (low) -> bright (high)."""
    t = max(0.0, min(1.0, t))
    return "#" + "".join(f"{round(lo + (hi - lo) * t):02x}"
                         for lo, hi in zip(RAMP_LO, RAMP_HI))


def ink_on(t: float) -> tuple[str, str]:
    """Readable text on a ramp(t) fill: light ink on the dim end, dark on the bright end.

    The ramp spans a wide lightness range, so a single ink colour would fail contrast at
    one end or the other. Returns (primary, secondary).
    """
    return ("#0b1117", "#0b1117b0") if t > 0.52 else ("#ffffff", "#ffffffa6")


def load_round(path: Path) -> tuple[list[dict], list[QuestionExample], dict]:
    """Rebuild (nodes, examples, meta) from a feedback JSON written by build_feedback."""
    fb = json.loads(Path(path).read_text())
    clusters = fb["cluster_comparison"]
    if not clusters or "instances" not in clusters[0]:
        raise SystemExit(f"{path} was written with include_instances=False; "
                         "re-run the feedback build without --no-instances")

    by_index: dict[int, QuestionExample] = {}
    nodes: list[dict] = []
    for c in clusters:
        leaves = []
        for inst in c["instances"]:
            i = inst["index"]
            leaves.append(i)
            by_index[i] = QuestionExample(
                updated_question=inst.get("updated_question", ""),
                strategy=inst.get("strategy", ""),
                failed=bool(inst.get("failed")),
                source_run=inst.get("source_run", ""),
                seed_question=inst.get("seed_question", ""),
                verification_criterion=inst.get("verification_criterion", ""),
                round=inst.get("round"),
            )
        nodes.append({
            "node_id": c["cluster_id"],
            "description": c["description"],
            "num_questions": c["num_questions"],
            "num_failed": c["num_failed"],
            "failure_rate": c["failure_rate"],
            "leaves": leaves,
        })

    n = max(by_index) + 1 if by_index else 0
    blank = QuestionExample(updated_question="", strategy="", failed=False)
    examples = [by_index.get(i, blank) for i in range(n)]
    return nodes, examples, fb["meta"]


def run_one(nodes, examples, total, params: dict) -> dict:
    """One selection run -> the counts and the picks, exactly as build_feedback would."""
    p = {**DEFAULTS, **params}
    focus, ineligible = select_focus(
        nodes, total,
        min_failure_rate=p["min_failure_rate"], max_share=p["max_share"],
        min_cluster_size=p["min_cluster_size"], max_cluster_size=p["max_cluster_size"],
        rank_by=p["rank_by"], always_include_new=p["always_include_new"],
    )
    # few-shot examples are deduped globally in rank order, so count them the same way
    seen: set[str] = set()
    picks = []
    n_examples = 0
    for rank, node in enumerate(focus, start=1):
        shots = few_shot_failures(node, examples,
                                 per_strategy=p["examples_per_strategy"], seen=seen)
        n_examples += len(shots)
        picks.append({
            "rank": rank,
            "cluster_id": node["node_id"],
            "description": node["description"],
            "num_questions": node["num_questions"],
            "num_failed": node["num_failed"],
            "failure_rate": round(node["failure_rate"], 4),
            "share": round(node["share"], 4),
            "score": round(node["score"], 4),
            "selected_for": node.get("selected_for", ""),
            "num_few_shot": len(shots),
        })

    tags: dict[str, int] = {}
    for pick in picks:
        # select_focus only tags picks under rank_by="diverse"; the other three ranking
        # modes leave selected_for empty, so surface that rather than showing a blank.
        tag = pick["selected_for"] or "(untagged)"
        tags[tag] = tags.get(tag, 0) + 1
    reasons: dict[str, int] = {}
    for node in ineligible:
        for r in node["excluded_for"]:
            key = r.split(" (")[0].split(" —")[0]
            reasons[key] = reasons.get(key, 0) + 1

    return {
        "params": p,
        "num_focus": len(focus),
        "num_ineligible": len(ineligible),
        "num_few_shot_examples": n_examples,
        "focus_questions": sum(n["num_questions"] for n in focus),
        "focus_failures": sum(n["num_failed"] for n in focus),
        "selected_for_counts": tags,
        "exclusion_reason_counts": reasons,
        "focus_ids": [n["node_id"] for n in focus],
        "ineligible_ids": [n["node_id"] for n in ineligible],
        "picks": picks,
        "dropped": [{"cluster_id": n["node_id"], "num_questions": n["num_questions"],
                     "failure_rate": round(n["failure_rate"], 4),
                     "excluded_for": n["excluded_for"]} for n in ineligible],
    }


# ---------------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------------


def e(x) -> str:
    return html.escape(str(x), quote=True)


def fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def meter(good: int, bad: int, total: int, tip: str = "") -> str:
    """Stacked good/bad split of a fixed total — comparable across every row."""
    gw = 100.0 * good / total if total else 0.0
    bw = 100.0 * bad / total if total else 0.0
    t = tip or f"{good} good / {bad} bad of {total} clusters"
    return (f'<span class="meter" data-tip="{e(t)}">'
            f'<span class="seg g" style="width:{gw:.2f}%"></span>'
            f'<span class="seg b" style="width:{bw:.2f}%"></span></span>')


def bar(value: float, vmax: float, color: str, label: str, tip: str) -> str:
    """A single magnitude bar with its number alongside in text ink (never series color)."""
    w = 100.0 * value / vmax if vmax else 0.0
    return (f'<span class="barwrap" data-tip="{e(tip)}">'
            f'<span class="bartrack"><span class="barfill" '
            f'style="width:{w:.2f}%;background:{color}"></span></span>'
            f'<span class="barnum">{e(label)}</span></span>')


def chips(ids: list[str], kind: str) -> str:
    if not ids:
        return '<span class="none">—</span>'
    return " ".join(f'<span class="chip {kind}">{e(i)}</span>' for i in ids)


def range_strip(lo: int, hi: int, total: int, base: int) -> str:
    """The span of `good` counts a parameter reaches, against the fixed cluster total."""
    left = 100.0 * lo / total if total else 0.0
    width = 100.0 * (hi - lo) / total if total else 0.0
    bpos = 100.0 * base / total if total else 0.0
    inert = "" if hi > lo else " inert"
    return (f'<span class="barwrap" data-tip="good ranges {lo}–{hi} of {total} clusters '
            f'(baseline {base})">'
            f'<span class="bartrack range">'
            f'<span class="rangefill{inert}" style="left:{left:.2f}%;width:{max(width, 1.2):.2f}%">'
            f'</span><span class="basetick" style="left:{bpos:.2f}%"></span></span>'
            f'<span class="barnum">{lo}–{hi}</span></span>')


def section(title: str, note: str = "") -> str:
    n = f'<p class="note">{note}</p>' if note else ""
    return f"<h2>{e(title)}</h2>{n}"


def build_report(path: Path) -> tuple[str, dict]:
    nodes, examples, meta = load_round(path)
    total_q = meta["num_instances"]
    n_clusters = len(nodes)
    baseline = run_one(nodes, examples, total_q, {})

    results = {"source": str(path), "meta": meta, "defaults": DEFAULTS,
               "baseline": baseline, "at_a_glance": [], "sweeps": {}, "grid": [],
               "interaction_new_rescue": [], "rank_orderings": {}}

    B: list[str] = []
    A = B.append

    # --- header + KPI tiles -------------------------------------------------
    A('<header><h1>Selection sensitivity</h1>'
      f'<div class="meta">'
      f'<span class="mi">source <b>{e(path.name)}</b></span>'
      f'<span class="mi">clustering <b>{e(meta["clustering"]["assign_method"])}</b>, '
      f'{meta["clustering"]["num_seeds"]} seeds + '
      f'{meta["clustering"]["num_new_clusters"]} novel</span>'
      f'<span class="mi">source_runs <b>{e(", ".join(meta["source_runs"]))}</b></span>'
      f'<span class="mi">built <b>{e(meta["generated_at"][:19])}</b></span>'
      '</div></header><main>')

    A('<p class="lede">Only the <b>selection</b> stage is re-run. Clustering, failure '
      'counts and shares are taken verbatim from the existing feedback file — nothing is '
      'recomputed and no model is called. <b>Good</b> = <code>focus_strategies</code> '
      '(fed back into the next round); <b>bad</b> = <code>ineligible_strategies</code> '
      '(dropped, with <code>excluded_for</code>). Failure is <i>good</i> here: a FAILED '
      'question is one the generator successfully made hard.</p>')

    tiles = [
        (total_q, "questions", f"{meta['num_failed']} FAILED "
                               f"({meta['overall_failure_rate']:.2f} overall)"),
        (n_clusters, "clusters", f"{meta['clustering']['num_seeds']} seed + "
                                 f"{meta['clustering']['num_new_clusters']} novel"),
        (baseline["num_focus"], "good at defaults", "fed back next round"),
        (baseline["num_ineligible"], "bad at defaults", "dropped"),
        (baseline["num_few_shot_examples"], "few-shot examples", "deduped across clusters"),
    ]
    A('<div class="tiles">')
    for value, label, sub in tiles:
        cls = " good" if label.startswith("good") else (" bad" if label.startswith("bad") else "")
        A(f'<div class="tile{cls}"><div class="big">{value}</div>'
          f'<div class="lbl">{e(label)}</div><div class="sub">{e(sub)}</div></div>')
    A("</div>")

    # --- cluster inventory --------------------------------------------------
    A(section("Cluster inventory",
              "The fixed input to every run below. Bar length encodes the value; the "
              "number beside it is the value itself."))
    # share bars are drawn on a 0–max_share scale, so their shortness IS the finding:
    # no cluster comes anywhere near the default catch-all gate.
    A('<table><thead><tr><th>cluster</th><th class="r">n</th><th class="r">failed</th>'
      '<th>failure_rate <span class="scale">0–1</span></th>'
      '<th>share <span class="scale">0–0.5, the max_share gate</span></th>'
      '<th>description</th></tr></thead><tbody>')
    for n in sorted(nodes, key=lambda x: -x["failure_rate"]):
        share = n["num_questions"] / total_q if total_q else 0.0
        novel = n["node_id"].startswith("new.")
        desc = n["description"].replace("[novel strategy not in seed menu] ", "")
        rate, nq, nf = n["failure_rate"], n["num_questions"], n["num_failed"]
        rate_bar = bar(rate, 1.0, ramp(rate), f"{rate:.2f}",
                       f"{nf}/{nq} questions FAILED")
        share_bar = bar(share, 0.5, ramp(share / 0.5), f"{share:.2f}",
                        f"{nq} of {total_q} questions this round")
        A(f'<tr><td><span class="chip {"new" if novel else "seed"}">{e(n["node_id"])}</span></td>'
          f'<td class="r num">{nq}</td>'
          f'<td class="r num">{nf}</td>'
          f"<td>{rate_bar}</td>"
          f"<td>{share_bar}</td>"
          f'<td class="desc">{e(desc)}'
          + (' <span class="tagnew">novel</span>' if novel else "")
          + "</td></tr>")
    A("</tbody></table>")

    # --- baseline params ----------------------------------------------------
    A(section("Baseline — all module defaults"))
    A('<table class="params"><thead><tr>'
      + "".join(f"<th><code>{e(k)}</code></th>" for k in DEFAULTS)
      + "</tr></thead><tbody><tr>"
      + "".join(f'<td class="num">{e(fmt(v))}</td>' for v in DEFAULTS.values())
      + "</tr></tbody></table>")

    # --- at a glance --------------------------------------------------------
    A(section("At a glance: which knobs move the good/bad split",
              "Span of <code>good</code> counts each parameter reaches across the values "
              "swept below. The tick marks the baseline count."))
    A('<table><thead><tr><th>parameter</th><th>good range</th><th class="r">verdict</th>'
      '<th>values swept</th></tr></thead><tbody>')
    for name, values in SWEEPS:
        counts = [run_one(nodes, examples, total_q, {name: v})["num_focus"] for v in values]
        lo, hi = min(counts), max(counts)
        moves = hi > lo
        verdict = (f'<span class="verd yes">moves {hi - lo}</span>' if moves
                   else '<span class="verd no">inert</span>')
        A(f'<tr><td><code>{e(name)}</code></td>'
          f'<td>{range_strip(lo, hi, n_clusters, baseline["num_focus"])}</td>'
          f'<td class="r">{verdict}</td>'
          f'<td class="vals">{e(", ".join(fmt(v) for v in values))}</td></tr>')
        results["at_a_glance"].append({"parameter": name,
                                       "values": [fmt(v) for v in values],
                                       "good_min": lo, "good_max": hi, "counts": counts})
    A("</tbody></table>")

    # --- legend -------------------------------------------------------------
    A('<div class="legend"><span class="key"><i style="background:%s"></i>good — kept in '
      'focus_strategies</span><span class="key"><i style="background:%s"></i>bad — dropped '
      'to ineligible_strategies</span><span class="key note-in">every meter is the same '
      '%d-cluster total, so rows are directly comparable</span></div>'
      % (C_GOOD, C_BAD, n_clusters))

    # --- one-at-a-time sweeps ----------------------------------------------
    A(section("One-at-a-time sweeps",
              "Each table varies one parameter with the others at their defaults."))
    for name, values in SWEEPS:
        A(f'<h3><code>{e(name)}</code></h3><p class="note">{SWEEP_NOTES[name]}</p>')
        A('<table><thead><tr><th>value</th><th>good / bad</th><th class="r">good</th>'
          '<th class="r">bad</th><th class="r">few-shots</th><th class="r">q covered</th>'
          '<th class="r">fails covered</th><th>selected_for</th>'
          '<th>dropped clusters</th></tr></thead><tbody>')
        rows = []
        for v in values:
            r = run_one(nodes, examples, total_q, {name: v})
            rows.append(r)
            is_def = v == DEFAULTS[name] and type(v) is type(DEFAULTS[name])
            tags = " ".join(
                f'<span class="tag {e(k.strip("()"))}">{e(k)}&nbsp;{c}</span>'
                for k, c in sorted(r["selected_for_counts"].items()))
            tip = (f"{name}={fmt(v)}: {r['num_focus']} good / "
                   f"{r['num_ineligible']} bad of {n_clusters} clusters")
            A(f'<tr class="{"isdef" if is_def else ""}">'
              f'<td class="num vcell">{e(fmt(v))}'
              + ('<span class="defmark">default</span>' if is_def else "")
              + "</td>"
              f'<td>{meter(r["num_focus"], r["num_ineligible"], n_clusters, tip)}</td>'
              f'<td class="r num g">{r["num_focus"]}</td>'
              f'<td class="r num b">{r["num_ineligible"]}</td>'
              f'<td class="r num">{r["num_few_shot_examples"]}</td>'
              f'<td class="r num">{r["focus_questions"]}</td>'
              f'<td class="r num">{r["focus_failures"]}</td>'
              f"<td>{tags}</td>"
              f'<td>{chips(r["ineligible_ids"], "drop")}</td></tr>')
        A("</tbody></table>")
        results["sweeps"][name] = [
            {"value": r["params"][name], "num_focus": r["num_focus"],
             "num_ineligible": r["num_ineligible"],
             "num_few_shot_examples": r["num_few_shot_examples"],
             "focus_ids": r["focus_ids"], "ineligible_ids": r["ineligible_ids"],
             "selected_for_counts": r["selected_for_counts"],
             "exclusion_reason_counts": r["exclusion_reason_counts"]}
            for r in rows
        ]

    # --- 2-D grid -----------------------------------------------------------
    A(section("Grid: min_failure_rate x max_share",
              "Cell shows the <code>good</code> count; shade encodes the same number "
              "(dim = few kept, bright = most kept)."))
    rates = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    shares = [0.03, 0.05, 0.08, 0.10, 0.12, 0.5]
    A('<table class="heat"><thead><tr><th class="corner">min_failure_rate \\ max_share</th>'
      + "".join(f'<th class="r">{s:g}</th>' for s in shares) + "</tr></thead><tbody>")
    for mfr in rates:
        A(f'<tr><th class="rowh">{mfr:g}</th>')
        for ms in shares:
            r = run_one(nodes, examples, total_q,
                        {"min_failure_rate": mfr, "max_share": ms})
            g = r["num_focus"]
            t = g / n_clusters
            fg, fg2 = ink_on(t)
            is_def = mfr == DEFAULTS["min_failure_rate"] and ms == DEFAULTS["max_share"]
            A(f'<td class="cell{" isdef" if is_def else ""}" '
              f'style="background:{ramp(t)}" '
              f'data-tip="min_failure_rate={mfr:g}, max_share={ms:g} &rarr; '
              f'{g} good / {r["num_ineligible"]} bad">'
              f'<span class="cn" style="color:{fg}">{g}</span>'
              f'<span class="cd" style="color:{fg2}">/{r["num_ineligible"]}</span></td>')
            results["grid"].append({"min_failure_rate": mfr, "max_share": ms,
                                    "num_focus": g,
                                    "num_ineligible": r["num_ineligible"],
                                    "focus_ids": r["focus_ids"]})
        A("</tr>")
    A("</tbody></table>")
    A('<p class="note">Numbers read <b>good</b>/<span class="dim">bad</span>. The outlined '
      'cell is the shipped default.</p>')

    # --- interaction --------------------------------------------------------
    A(section("Interaction: always_include_new x min_failure_rate",
              "<code>always_include_new</code> looks inert one-at-a-time, but only because "
              "at the default <code>min_failure_rate</code> every novel cluster with a "
              "failure already passes on its own. Tighten the gate and the rescue starts "
              "doing the work."))
    A('<table><thead><tr><th class="r">min_failure_rate</th><th class="r">good, rescue ON</th>'
      '<th class="r">good, rescue OFF</th><th class="r">delta</th>'
      '<th>clusters rescued</th></tr></thead><tbody>')
    for mfr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        on = run_one(nodes, examples, total_q,
                     {"min_failure_rate": mfr, "always_include_new": True})
        off = run_one(nodes, examples, total_q,
                      {"min_failure_rate": mfr, "always_include_new": False})
        rescued = [c for c in on["focus_ids"] if c not in off["focus_ids"]]
        d = on["num_focus"] - off["num_focus"]
        A(f'<tr><td class="r num">{mfr:g}</td>'
          f'<td class="r num g">{on["num_focus"]}</td>'
          f'<td class="r num">{off["num_focus"]}</td>'
          f'<td class="r num">{"+" + str(d) if d else "—"}</td>'
          f'<td>{chips(rescued, "new")}</td></tr>')
        results["interaction_new_rescue"].append(
            {"min_failure_rate": mfr, "good_on": on["num_focus"],
             "good_off": off["num_focus"], "rescued": rescued})
    A("</tbody></table>")

    # --- ranking comparison -------------------------------------------------
    A(section("rank_by: same set, different order",
              "All four modes keep the same clusters. But order decides which cluster "
              "claims a shared few-shot example first (the dedup is global), and what the "
              "generator sees at the top of its list."))
    orderings = {}
    for mode in ["underrepresented", "diverse", "failure_rate", "volume"]:
        r = run_one(nodes, examples, total_q, {"rank_by": mode})
        orderings[mode] = r["picks"]
        results["rank_orderings"][mode] = [
            {k: p[k] for k in ("rank", "cluster_id", "score", "failure_rate",
                               "num_questions", "selected_for", "num_few_shot")}
            for p in r["picks"]
        ]
    base_order = {p["cluster_id"]: i for i, p in enumerate(orderings["underrepresented"])}
    A('<table class="rankcmp"><thead><tr><th class="r">rank</th>'
      + "".join(f'<th><code>{e(m)}</code>'
                + ('<span class="defmark">default</span>' if m == DEFAULTS["rank_by"] else "")
                + "</th>" for m in orderings)
      + "</tr></thead><tbody>")
    depth = max(len(v) for v in orderings.values())
    for i in range(depth):
        A(f'<tr><td class="r num">{i + 1}</td>')
        for mode in orderings:
            picks = orderings[mode]
            if i >= len(picks):
                A('<td><span class="none">—</span></td>')
                continue
            p = picks[i]
            shift = base_order.get(p["cluster_id"], i) - i
            mv = ""
            if mode != "underrepresented" and shift:
                mv = (f'<span class="shift {"up" if shift > 0 else "down"}">'
                      f'{"▲" if shift > 0 else "▼"}{abs(shift)}</span>')
            novel = p["cluster_id"].startswith("new.")
            A(f'<td><span class="chip {"new" if novel else "seed"}">{e(p["cluster_id"])}</span>'
              f'{mv}<span class="pmeta">{p["failure_rate"]:.2f} · n={p["num_questions"]} · '
              f'{p["num_few_shot"]}ex</span></td>')
        A("</tr>")
    A("</tbody></table>")
    A('<p class="note">▲/▼ is the move relative to the default '
      '<code>underrepresented</code> order.</p>')

    A("</main>")
    return PAGE.format(title=e(f"Selection sensitivity — {path.name}"),
                       good=C_GOOD, bad=C_BAD, body="\n".join(B)), results


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg:#0e1116; --panel:#161b22; --text:#e6edf3; --muted:#8b949e;
  --line:#30363d; --accent:#58a6ff; --good:{good}; --bad:{bad};
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  background:var(--bg); color:var(--text); }}
header {{ padding:16px 22px; background:var(--panel); border-bottom:1px solid var(--line); }}
h1 {{ font-size:17px; margin:0 0 6px; }}
.meta {{ color:var(--muted); font-size:12px; display:flex; flex-wrap:wrap; gap:6px 18px; }}
.mi b {{ color:var(--accent); font-weight:600; }}
main {{ padding:20px 22px 60px; max-width:1280px; }}
.lede {{ color:#c9d1d9; max-width:78ch; margin:4px 0 20px; }}
h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
  margin:34px 0 10px; padding-top:14px; border-top:1px solid var(--line); }}
h3 {{ font-size:13px; margin:22px 0 4px; font-weight:600; }}
h3 code {{ color:var(--accent); }}
.note {{ color:var(--muted); font-size:12px; margin:0 0 10px; max-width:82ch; }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; }}
.dim {{ color:var(--muted); }}

/* KPI tiles: a hero number needs no chart */
.tiles {{ display:flex; flex-wrap:wrap; gap:10px; margin:0 0 8px; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:12px 16px; min-width:132px; }}
.tile .big {{ font-size:26px; font-weight:650; font-variant-numeric:tabular-nums;
  line-height:1.1; }}
.tile .lbl {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--muted); margin-top:3px; }}
.tile .sub {{ font-size:11px; color:#6e7681; margin-top:2px; }}
.tile.good {{ border-left:3px solid var(--good); }}
.tile.bad {{ border-left:3px solid var(--bad); }}

table {{ border-collapse:collapse; width:100%; margin:6px 0 4px; font-size:13px; }}
th {{ text-align:left; color:var(--muted); font-weight:600; font-size:11px;
  text-transform:uppercase; letter-spacing:.04em; padding:6px 10px;
  border-bottom:1px solid var(--line); white-space:nowrap; }}
td {{ padding:6px 10px; border-bottom:1px solid #1c2129; vertical-align:middle; }}
tr:hover td {{ background:#12171e; }}
.r {{ text-align:right; }}
.num {{ font-variant-numeric:tabular-nums; font-family:ui-monospace,Menlo,monospace;
  font-size:12px; white-space:nowrap; }}
td.g {{ color:var(--text); }} td.b {{ color:var(--text); }}
.desc {{ color:#c9d1d9; max-width:460px; }}
.vals {{ color:var(--muted); font-family:ui-monospace,Menlo,monospace; font-size:11px; }}
.scale {{ color:#6e7681; font-weight:400; text-transform:none; letter-spacing:0;
  font-size:10px; margin-left:5px; }}
tr.isdef td {{ background:#131a24; }}
tr.isdef:hover td {{ background:#182130; }}
.defmark {{ font-size:9px; text-transform:uppercase; letter-spacing:.06em;
  color:var(--accent); background:#1f6feb22; border-radius:3px; padding:1px 5px;
  margin-left:6px; vertical-align:middle; }}
.vcell {{ white-space:nowrap; }}
table.params td {{ font-size:12px; }}

/* stacked good/bad meter — 4px rounded outer ends, 2px surface gap between fills */
.meter {{ display:inline-flex; width:132px; height:9px; background:#1c2129;
  border-radius:4px; overflow:hidden; vertical-align:middle; gap:2px; }}
.meter .seg {{ height:100%; }}
.meter .seg.g {{ background:var(--good); border-radius:4px 0 0 4px; }}
.meter .seg.b {{ background:var(--bad); border-radius:0 4px 4px 0; }}
.meter .seg.g:only-child, .meter .seg.b:only-child {{ border-radius:4px; }}

/* single magnitude bar + its number in text ink */
.barwrap {{ display:inline-flex; align-items:center; gap:8px; }}
.bartrack {{ position:relative; width:84px; height:9px; background:#1c2129;
  border-radius:4px; overflow:hidden; }}
.barfill {{ display:block; height:100%; border-radius:4px; }}
.barnum {{ font-family:ui-monospace,Menlo,monospace; font-size:12px;
  font-variant-numeric:tabular-nums; color:var(--text); min-width:34px; }}
.bartrack.range {{ width:120px; overflow:visible; }}
.rangefill {{ position:absolute; top:0; height:100%; background:var(--accent);
  border-radius:4px; }}
.rangefill.inert {{ background:#3d444d; }}
.basetick {{ position:absolute; top:-3px; width:2px; height:15px; background:var(--text);
  opacity:.55; }}
.verd {{ font-size:11px; padding:2px 8px; border-radius:999px; white-space:nowrap; }}
.verd.yes {{ background:#1f6feb22; color:var(--accent); }}
.verd.no {{ background:#8b949e1f; color:var(--muted); }}

.chip {{ font-family:ui-monospace,Menlo,monospace; font-size:11px; padding:1px 6px;
  border-radius:4px; white-space:nowrap; }}
.chip.seed {{ background:#1f6feb1f; color:var(--accent); }}
.chip.new {{ background:#db6d281f; color:#f0883e; }}
.chip.drop {{ background:#8b949e1a; color:var(--muted); }}
.tagnew {{ font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:#f0883e;
  background:#db6d281f; border-radius:3px; padding:1px 5px; margin-left:4px; }}
.tag {{ font-size:11px; padding:2px 7px; border-radius:999px; background:#8b949e1f;
  color:var(--muted); white-space:nowrap; display:inline-block; margin:1px 2px 1px 0; }}
.tag.new-underrepresented {{ background:#db6d281f; color:#f0883e; }}
.tag.failure_rate {{ background:#1f6feb22; color:var(--accent); }}
.tag.volume {{ background:#a371f722; color:#a371f7; }}
.none {{ color:#3a414c; }}

/* heatmap: one hue, dim -> bright; every cell also prints its number */
table.heat {{ width:auto; }}
table.heat th.corner {{ text-transform:none; letter-spacing:0; font-size:11px; }}
table.heat th.rowh {{ text-align:right; font-family:ui-monospace,Menlo,monospace;
  font-size:12px; text-transform:none; color:var(--text); border-bottom:1px solid #1c2129; }}
td.cell {{ text-align:right; font-family:ui-monospace,Menlo,monospace; font-size:12px;
  min-width:62px; border:2px solid var(--bg); border-radius:4px; }}
tr:hover td.cell {{ background-blend-mode:normal; }}
td.cell .cn {{ font-weight:650; }}   /* ink set per-cell: see ink_on() */
td.cell .cd {{ font-size:11px; }}
td.cell.isdef {{ outline:2px solid var(--accent); outline-offset:-3px; }}

.legend {{ display:flex; flex-wrap:wrap; gap:8px 20px; align-items:center;
  color:var(--muted); font-size:12px; margin:14px 0 2px; }}
.legend .key {{ display:inline-flex; align-items:center; gap:7px; }}
.legend i {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
.legend .note-in {{ color:#6e7681; font-style:italic; }}

.pmeta {{ color:var(--muted); font-size:11px; font-family:ui-monospace,Menlo,monospace;
  margin-left:7px; }}
/* rank movement is polarity: the validated blue/orange pair, and the glyph carries the
   direction so it never reads by colour alone */
.shift {{ font-size:10px; margin-left:5px; font-family:ui-monospace,Menlo,monospace; }}
.shift.up {{ color:var(--accent); }}
.shift.down {{ color:#f0883e; }}
table.rankcmp td {{ white-space:nowrap; }}

#tip {{ position:fixed; z-index:20; pointer-events:none; opacity:0;
  transition:opacity .08s ease; background:#1c2229; color:var(--text);
  border:1px solid var(--line); border-radius:6px; padding:5px 9px; font-size:12px;
  max-width:320px; box-shadow:0 6px 20px #0008; }}
</style></head>
<body data-palette="{good},{bad}">
{body}
<div id="tip"></div>
<script>
// hover layer: every bar / meter / heat cell explains itself on hover
(function () {{
  var tip = document.getElementById('tip');
  document.addEventListener('mouseover', function (ev) {{
    var el = ev.target.closest('[data-tip]');
    if (!el) return;
    tip.innerHTML = el.getAttribute('data-tip');
    tip.style.opacity = 1;
    var r = el.getBoundingClientRect();
    var t = tip.getBoundingClientRect();
    var x = Math.min(Math.max(8, r.left), window.innerWidth - t.width - 8);
    var y = r.top - t.height - 8;
    tip.style.left = x + 'px';
    tip.style.top = (y < 8 ? r.bottom + 8 : y) + 'px';
  }});
  document.addEventListener('mouseout', function (ev) {{
    if (ev.target.closest('[data-tip]')) tip.style.opacity = 0;
  }});
}})();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("feedback_json", type=Path,
                    help="a feedback file written by strategy_feedback_module (with instances)")
    ap.add_argument("--out-html", type=Path, default=None,
                    help="output HTML (default: <input stem>_selection_sensitivity.html)")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="raw sweep results (default: alongside the HTML)")
    args = ap.parse_args()

    out_html = args.out_html or Path(f"{args.feedback_json.stem}_selection_sensitivity.html")
    out_json = args.out_json or out_html.with_suffix(".json")

    page, results = build_report(args.feedback_json)
    out_html.write_text(page)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"wrote -> {out_html}\nwrote -> {out_json}")


if __name__ == "__main__":
    main()
