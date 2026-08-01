"""Turn an EvalTree strategy run into a feedback JSON for the claude inference pipeline.

`run_pipeline.drchallenge.strategy.sh` clusters the generated (hardened) questions by
their `strategy` field and computes, per cluster node, how often the answering agent
FAILED (results.json uses 1 = FAILED). A high failure rate is GOOD here: it means the
generator successfully produced questions that stump the agent.

This script reads that strategy tree together with its confidence_interval.json and the
dataset.json, and emits a single JSON file describing:

  * focus_strategies  — a size-diverse set of clusters that work (decent failure rate over
                        a real theme). Selection interleaves a failure-rate ranking (small,
                        strongly-failing strategies) with a volume ranking (bigger, moderate
                        strategies) so the mix spans cluster sizes instead of only tiny ones.
  * few_shot_failures — concrete examples where the generator produced a question that
                        FAILED verification (drawn from the focus clusters first), to be
                        dropped into the generation prompt as demonstrations.
  * cluster_comparison — EVERY cluster (in seeded mode: each seed strategy + any novel
                        clusters) with its failure rate broken down per source_run, so the
                        prompts can be compared (e.g. original vs explore) strategy by
                        strategy, not just for the selected focus set.

The output is meant to be fed back into the generation prompt for the NEXT round, so the
generator emphasises strategies that work but are currently rare.

Repeatable across rounds: point --dataset-dir / --tree at whatever round's EvalTree
outputs you built (each round is a dataset.json + its own tree), and --out at a
round-specific file. Same code, different inputs -> different feedback. Selection
thresholds are all flags so a round can be tuned without editing code.

Usage:
    # after run_one_round.sh has built the strategy tree for a round's dataset:
    python build_strategy_feedback.py \
        --tree "EvalTree/Datasets/DRChallenge/EvalTree/stage3-RecursiveClustering/[split=full]_[annotation=strategy]_..._[stage4-CapabilityDescription-model=gpt-4o-mini].json" \
        --out runs/round1/strategy_feedback.json

    # or let it auto-detect the strategy tree in the default dataset dir:
    python build_strategy_feedback.py --annotation strategy --latest \
        --out runs/round1/strategy_feedback.json
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# Reuse the tree/CI locators the CSV extractor already uses, so both tools agree on
# which files a given (dataset, annotation, model) maps to.
from extract_promising_categories import (
    DEFAULT_DATASET_DIR,
    find_ci_json,
    find_tree_json,
)


def collect_nodes(tree, ci, nodes: list[dict], depth: int = 0, node_id: str = "root") -> list[int]:
    """Parallel-walk the description tree and the CI tree, recording one node per cluster.

    Returns the dataset-index leaves under `tree` so each node knows exactly which
    instances (and thus which questions) belong to it.
    """
    # Leaves are bare instance indices in both trees; they carry no description.
    if isinstance(tree, int):
        return [tree]

    t_sub, c_sub = tree["subtrees"], ci["subtrees"]
    leaves: list[int] = []
    if isinstance(t_sub, dict):
        for k in t_sub:
            leaves += collect_nodes(t_sub[k], c_sub[k], nodes, depth + 1, f"{node_id}.{k}")
    elif isinstance(t_sub, list):
        for i, (t, c) in enumerate(zip(t_sub, c_sub)):
            leaves += collect_nodes(t, c, nodes, depth + 1, f"{node_id}[{i}]")
    elif isinstance(t_sub, int):  # single leaf child (int index) — nothing more to describe
        leaves.append(t_sub)

    size = ci["size"]
    nodes.append({
        "node_id": node_id,
        "depth": depth,
        "description": tree.get("description", ""),
        "num_questions": size,
        "num_failed": ci["sum_metrics"],
        "failure_rate": (ci["sum_metrics"] / size) if size else 0.0,
        "leaves": list(leaves),
    })
    return leaves


def select_focus(nodes: list[dict], total: int, *, min_failure_rate: float,
                 max_share: float, min_cluster_size: int, max_cluster_size: int | None,
                 max_strategies: int, rank_by: str = "underrepresented",
                 always_include_new: bool = True) -> list[dict]:
    """Pick clusters that work (stump the agent) and are worth leaning into next round.

    A cluster is eligible when it is a real theme (>= min_cluster_size questions, and no
    larger than max_cluster_size if set), it stumps the agent often enough
    (failure_rate >= min_failure_rate), and it is not an over-represented catch-all
    (share <= max_share).

    `rank_by` chooses how the eligible clusters are ordered:
      * underrepresented (default) — score = failure_rate * (1 - share): reward strategies
        that fail a lot yet are RARE in this round, so the generator leans into working
        strategies it currently under-uses. Lower --max-share to focus harder on rare ones.
      * diverse — interleave a failure_rate ranking (small, strongly-failing) with a volume
        ranking (bigger, medium-rate) for a spread of cluster sizes.
      * failure_rate — pure failure_rate, highest first.
      * volume — pure absolute number of failures, highest first.
    Each picked cluster is tagged `selected_for` with the reason it was chosen.

    When `always_include_new`, every NEW (off-menu, novel) cluster that is underrepresented
    (share <= max_share) and stumped the agent at least once is ALWAYS included, even if it
    falls below the size/failure thresholds or the ranking cutoff — novel working strategies
    are exactly what we want the generator to explore more. They are appended (beyond
    max_strategies if needed) and tagged selected_for='new-underrepresented'.
    """
    def enrich(n: dict) -> dict:
        share = n["num_questions"] / total if total else 0.0
        return {**n, "share": share, "score": n["failure_rate"] * (1.0 - share)}

    eligible = []
    for n in nodes:
        if n["node_id"] == "root":
            continue  # the whole dataset is not a "strategy"
        size = n["num_questions"]
        if size < min_cluster_size:
            continue
        if max_cluster_size is not None and size > max_cluster_size:
            continue
        share = size / total if total else 0.0
        if n["failure_rate"] < min_failure_rate:
            continue
        if share > max_share:
            continue
        eligible.append(enrich(n))

    if rank_by == "diverse":
        # Round-robin two rankings for a spread of cluster sizes.
        rankings = [
            ("failure_rate", sorted(eligible, key=lambda n: (n["failure_rate"], n["num_failed"]), reverse=True)),
            ("volume", sorted(eligible, key=lambda n: (n["num_failed"], n["failure_rate"]), reverse=True)),
        ]
        picked: list[dict] = []
        picked_ids: set[str] = set()
        cursors, turn = [0, 0], 0
        while len(picked) < max_strategies and len(picked_ids) < len(eligible):
            tag, ranked = rankings[turn]
            i = cursors[turn]
            while i < len(ranked) and ranked[i]["node_id"] in picked_ids:
                i += 1
            cursors[turn] = i
            if i < len(ranked):
                picked.append({**ranked[i], "selected_for": tag})
                picked_ids.add(ranked[i]["node_id"])
            turn = 1 - turn
    else:
        # Single-score rankings.
        if rank_by == "failure_rate":
            key = lambda n: (n["failure_rate"], -n["share"], n["num_failed"])
        elif rank_by == "volume":
            key = lambda n: (n["num_failed"], n["failure_rate"])
        else:  # underrepresented (default): high failure but low share first
            rank_by = "underrepresented"
            key = lambda n: (n["score"], n["failure_rate"], -n["share"])
        ranked = sorted(eligible, key=key, reverse=True)
        picked = [{**n, "selected_for": rank_by} for n in ranked[:max_strategies]]

    # Guarantee underrepresented NEW clusters that work, even below thresholds/cutoff.
    if always_include_new:
        have = {p["node_id"] for p in picked}
        extra = [enrich(n) for n in nodes
                 if n["node_id"].startswith("new.")
                 and n["node_id"] not in have
                 and n["num_failed"] >= 1
                 and (n["num_questions"] / total if total else 0.0) <= max_share]
        extra.sort(key=lambda n: (n["score"], n["failure_rate"]), reverse=True)
        picked.extend({**n, "selected_for": "new-underrepresented"} for n in extra)

    return picked


def dedup_descendant_clusters(focus: list[dict]) -> list[dict]:
    """Drop a focus cluster whose leaves are fully contained in a higher-ranked one.

    The tree is nested, so a strong parent and its equally-strong child can both qualify;
    keep only the first (higher-ranked) of any ancestor/descendant pair to avoid emitting
    the same strategy twice at different granularities.
    """
    kept: list[dict] = []
    for n in focus:
        n_leaves = set(n["leaves"])
        redundant = any(
            n_leaves <= set(k["leaves"]) or set(k["leaves"]) <= n_leaves
            for k in kept
        )
        if not redundant:
            kept.append(n)
    return kept


def instance_view(dataset: list[dict], i: int) -> dict:
    """The fields of one dataset instance the viewer's detail pages display."""
    inst = dataset[i] if 0 <= i < len(dataset) else {}
    return {
        "index": i,
        "seed_question": inst.get("seed_question", ""),
        "updated_question": inst.get("updated_question", ""),
        "strategy": inst.get("strategy", ""),
        "verification_criterion": inst.get("verification_criterion", ""),
        "drtulu_verdict": inst.get("drtulu_verdict", ""),
        "source_run": inst.get("source_run", ""),
        "round": inst.get("round"),
    }


def source_run_breakdown(leaves: list[int], dataset: list[dict], results: list[int]) -> dict:
    """Per-source_run {num_questions, num_failed, failure_rate} for a cluster's leaves.

    Lets a cluster be compared across generation subsets — e.g. whether the 'original'
    prompt or the 'explore' prompt hit this strategy more, and which one's questions
    stumped the agent more often.
    """
    from collections import defaultdict

    agg: dict[str, dict] = defaultdict(lambda: {"num_questions": 0, "num_failed": 0})
    for i in leaves:
        run = dataset[i].get("source_run") or "unknown"
        agg[run]["num_questions"] += 1
        if 0 <= i < len(results) and results[i] == 1:
            agg[run]["num_failed"] += 1
    out: dict[str, dict] = {}
    for run in sorted(agg):
        n, f = agg[run]["num_questions"], agg[run]["num_failed"]
        out[run] = {"num_questions": n, "num_failed": f,
                    "failure_rate": round(f / n, 4) if n else 0.0}
    return out


def examples_for_cluster(node: dict, dataset: list[dict], results: list[int],
                         *, per_strategy: int, seen: set[str]) -> list[dict]:
    """Up to `per_strategy` failure examples for ONE focus cluster.

    Only questions the agent FAILED (results[i] == 1) — the generator succeeded at making
    them hard. Deduped by updated_question against `seen` (shared across strategies so the
    same question is never shown twice, even if clusters happen to overlap).
    """
    examples: list[dict] = []
    for i in node["leaves"]:
        if len(examples) >= per_strategy:
            break
        if not (0 <= i < len(results)) or results[i] != 1:
            continue
        inst = dataset[i]
        q = inst.get("updated_question", "")
        if not q or q in seen:
            continue
        seen.add(q)
        examples.append({
            "seed_question": inst.get("seed_question", ""),
            "updated_question": q,
            "strategy": inst.get("strategy", ""),
            "verification_criterion": inst.get("verification_criterion", ""),
            "drtulu_verdict": inst.get("drtulu_verdict", "FAILED"),
        })
    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                    help=f"DRChallenge dataset dir (default: {DEFAULT_DATASET_DIR})")
    ap.add_argument("--tree", type=Path, default=None,
                    help="explicit stage-4 strategy tree JSON (bypasses auto-detection); the exact "
                         "path run_one_round.sh reports as 'Strategy tree:'")
    ap.add_argument("--annotation", default="strategy",
                    help="leaf-label variant for auto-detection (default: strategy)")
    ap.add_argument("--model", default="drtulu",
                    help="agent name under eval_results/real/ (default: drtulu)")
    ap.add_argument("--latest", action="store_true",
                    help="if several trees share the annotation, use the most recently built one")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON path (default: <dataset-dir>/strategy_feedback[_<annotation>].json)")
    # --- clustering source ---
    ap.add_argument("--cluster-mode", choices=("evaltree", "seeded"), default="evaltree",
                    help="'evaltree': walk the stage-4 recursive-clustering tree (default). "
                         "'seeded': assign each question to a canonical seed strategy "
                         "(research_pipeline.py menu) or a new cluster.")
    ap.add_argument("--assign-method", choices=("llm", "embedding"), default="llm",
                    help="[seeded] how to assign strategies to seed/new clusters. 'llm' (default): "
                         "prompt --cluster-model to pick a seed or create a new cluster per strategy "
                         "(no embeddings needed). 'embedding': cosine nearest-seed with a distance "
                         "threshold (needs the cached stage-2 embeddings + OpenAI).")
    ap.add_argument("--cluster-model", default="claude-opus-4-1",
                    help="[seeded/llm] Anthropic model that assigns strategies to clusters (default: claude-opus-4-1)")
    ap.add_argument("--no-merge-new-clusters", action="store_true",
                    help="[seeded/llm] skip the consolidation pass that merges near-duplicate NEW "
                         "clusters into broad themes (by default merging is ON so few new clusters emerge)")
    ap.add_argument("--seeds-file", type=Path, default=None,
                    help="[seeded] file of seed strategies, one per line (default: built-in 8-item menu)")
    ap.add_argument("--embedding-model", default="text-embedding-3-small",
                    help="[seeded/embedding] embedding model; must match the cached question embeddings (default: text-embedding-3-small)")
    ap.add_argument("--seed-max-distance", type=float, default=0.5,
                    help="[seeded/embedding] cosine distance beyond which a question spawns a NEW "
                         "cluster instead of joining a seed (default: 0.5)")
    # --- selection knobs (per round; failures are GOOD, so higher failure_rate is better) ---
    ap.add_argument("--min-failure-rate", type=float, default=0.3,
                    help="a strategy must stump the agent at least this often to count as working (default: 0.3)")
    ap.add_argument("--max-share", type=float, default=0.5,
                    help="skip over-represented catch-all clusters covering more than this share of "
                         "the round's questions (default: 0.5)")
    ap.add_argument("--min-cluster-size", type=int, default=2,
                    help="ignore clusters with fewer questions than this; keep at 2 so small "
                         "strongly-failing clusters stay in the mix (default: 2)")
    ap.add_argument("--max-cluster-size", type=int, default=None,
                    help="optionally ignore clusters larger than this many questions (default: no cap)")
    ap.add_argument("--rank-by", choices=("underrepresented", "diverse", "failure_rate", "volume"),
                    default="underrepresented",
                    help="how to rank focus strategies: 'underrepresented' (default) favours "
                         "working-but-rare strategies (failure_rate*(1-share)); 'diverse' mixes "
                         "cluster sizes; 'failure_rate' / 'volume' rank by that alone.")
    ap.add_argument("--max-strategies", type=int, default=4,
                    help="max focus strategies to emit by ranking (default: 4); underrepresented "
                         "NEW clusters may be added on top of this — see --no-always-include-new")
    ap.add_argument("--no-always-include-new", action="store_true",
                    help="do NOT force-include underrepresented novel (new.*) clusters that stumped "
                         "the agent; by default they are always added to the focus strategies")
    ap.add_argument("--examples-per-strategy", type=int, default=5,
                    help="max few-shot failure examples attached to EACH focus strategy (default: 5)")
    ap.add_argument("--no-viewer", action="store_true",
                    help="skip writing the standalone HTML viewer next to the JSON output")
    args = ap.parse_args()

    dataset = json.loads((args.dataset_dir / "dataset.json").read_text())
    nodes: list[dict] = []
    # `cluster_source` describes where the clustering came from, recorded in meta.
    if args.cluster_mode == "seeded":
        import seed_cluster

        results_path = (args.dataset_dir / "eval_results" / "real" / args.model / "results.json")
        if not results_path.is_file():
            sys.exit(f"no results.json for model={args.model} at {results_path}")
        results = json.loads(results_path.read_text())

        seeds = seed_cluster.load_seed_strategies(args.seeds_file)
        strategies = [inst.get("strategy", "") for inst in dataset]

        if args.assign_method == "llm":
            assignment = seed_cluster.assign_llm(
                strategies, seeds, args.cluster_model,
                merge=not args.no_merge_new_clusters)
        else:
            q_embs = seed_cluster.load_question_embeddings(
                args.dataset_dir, args.annotation, args.embedding_model)
            if len(q_embs) != len(dataset):
                sys.exit(f"cached embeddings ({len(q_embs)}) != dataset size ({len(dataset)}); "
                         f"re-run the EvalTree strategy stage-2 for this dataset")
            seed_embs = seed_cluster.embed_seeds(seeds, args.embedding_model)
            assignment = seed_cluster.assign(q_embs, seed_embs, args.seed_max_distance)

        nodes = seed_cluster.build_nodes(dataset, results, seeds, assignment)

        n_new = assignment["n_clusters"] - assignment["n_seeds"]
        # Every seed now emits a node (empties included); "used" = seeds that got questions.
        n_seed_used = sum(1 for n in nodes
                          if n["node_id"].startswith("seed.") and n["num_questions"] > 0)
        cluster_source = {
            "cluster_mode": "seeded",
            "assign_method": args.assign_method,
            "num_seeds": len(seeds),
            "num_seed_clusters_used": n_seed_used,
            "num_new_clusters": n_new,
        }
        if args.assign_method == "llm":
            cluster_source["cluster_model"] = args.cluster_model
            cluster_source["merged_new_clusters"] = not args.no_merge_new_clusters
            cluster_source["new_cluster_labels"] = [
                assignment.get("new_labels", {}).get(assignment["n_seeds"] + k)
                for k in range(n_new)
            ]
            source_label = f"seeded[llm={args.cluster_model}]"
        else:
            dists = assignment["nearest_seed_dist"]
            cluster_source["embedding_model"] = args.embedding_model
            cluster_source["seed_max_distance"] = args.seed_max_distance
            cluster_source["nearest_seed_distance_min"] = round(min(dists), 4) if dists else None
            cluster_source["nearest_seed_distance_mean"] = round(sum(dists) / len(dists), 4) if dists else None
            cluster_source["nearest_seed_distance_max"] = round(max(dists), 4) if dists else None
            source_label = f"seeded[embed={args.embedding_model}, max_dist={args.seed_max_distance}]"
        ci_label = str(results_path)
    else:
        if args.tree is not None:
            tree_path = args.tree
            if not tree_path.is_file():
                sys.exit(f"--tree not found: {tree_path}")
        else:
            tree_path = find_tree_json(args.dataset_dir, args.annotation, latest=args.latest)
        ci_path = find_ci_json(args.dataset_dir, args.model, tree_path.name)
        tree = json.loads(tree_path.read_text())
        ci = json.loads(ci_path.read_text())
        results = json.loads((ci_path.parents[2] / "results.json").read_text())
        collect_nodes(tree, ci, nodes)
        cluster_source = {"cluster_mode": "evaltree"}
        source_label = tree_path.name
        ci_label = str(ci_path)

    total = len(dataset)
    num_failed = sum(1 for r in results if r == 1)

    focus = select_focus(
        nodes, total,
        min_failure_rate=args.min_failure_rate,
        max_share=args.max_share,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        max_strategies=args.max_strategies,
        rank_by=args.rank_by,
        always_include_new=not args.no_always_include_new,
    )
    # dedup nested clusters (evaltree only; seeded clusters are disjoint). No length cap
    # here so guaranteed underrepresented-new clusters beyond max_strategies are kept.
    focus = dedup_descendant_clusters(focus)

    # Up to N failure examples per focus strategy; `seen` is shared so no question repeats.
    seen_examples: set[str] = set()

    source_runs = sorted({inst.get("source_run") or "unknown" for inst in dataset})

    # Full per-cluster comparison across generation subsets (e.g. original vs explore),
    # covering EVERY cluster — not just the selected focus strategies. In seeded mode this
    # is the failure rate of each seed strategy broken down per prompt. Ordered seeds first
    # (by menu number), then any novel clusters.
    def cluster_sort_key(n: dict):
        nid = n["node_id"]
        kind, _, num = nid.partition(".")
        order = {"seed": 0, "new": 1}.get(kind, 2)
        try:
            return (order, int(num))
        except ValueError:
            return (order, 0)

    # Seeded mode has a flat, interpretable cluster set (seed strategies + novel clusters),
    # so compare every one. The evaltree tree is deep/nested — its own EvalTree HTML viewer
    # already overlays per-source_run rates — so we skip the full-tree dump here.
    comparison_nodes = nodes if args.cluster_mode == "seeded" else []
    cluster_comparison = [
        {
            "cluster_id": n["node_id"],
            "description": n["description"],
            "num_questions": n["num_questions"],
            "num_failed": n["num_failed"],
            "failure_rate": round(n["failure_rate"], 4),
            "by_source_run": source_run_breakdown(n["leaves"], dataset, results),
            # Full instance list so the viewer can build a click-through detail page per
            # cluster (and per source_run scope) showing every corresponding question.
            "instances": [instance_view(dataset, i) for i in n["leaves"]],
        }
        for n in sorted(comparison_nodes, key=cluster_sort_key)
    ]

    def strategy_row(rank: int, n: dict) -> dict:
        examples = examples_for_cluster(
            n, dataset, results,
            per_strategy=args.examples_per_strategy, seen=seen_examples,
        )
        return {
            "rank": rank,
            "cluster_id": n["node_id"],
            "description": n["description"],
            "num_questions": n["num_questions"],
            "num_failed": n["num_failed"],
            "failure_rate": round(n["failure_rate"], 4),
            "share": round(n["share"], 4),
            "score": round(n["score"], 4),
            "selected_for": n.get("selected_for", ""),
            "rationale": (
                f"Works ({n['num_failed']}/{n['num_questions']} stumped the agent, "
                f"failure rate {n['failure_rate']:.2f}) but under-used — only {n['share'] * 100:.0f}% "
                f"of this round's questions. Under-represented, so lean into it next round."
            ),
            "by_source_run": source_run_breakdown(n["leaves"], dataset, results),
            "few_shot_failures": examples,
        }

    out = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_dir": str(args.dataset_dir),
            "clustering": cluster_source,
            "tree": source_label,
            "results": ci_label,
            "model": args.model,
            "source_runs": source_runs,
            "num_instances": total,
            "num_failed": num_failed,
            "overall_failure_rate": round(num_failed / total, 4) if total else 0.0,
            "selection": {
                "rank_by": args.rank_by,
                "min_failure_rate": args.min_failure_rate,
                "max_share": args.max_share,
                "min_cluster_size": args.min_cluster_size,
                "max_cluster_size": args.max_cluster_size,
                "max_strategies": args.max_strategies,
                "examples_per_strategy": args.examples_per_strategy,
            },
            "notes": "failure_rate is the fraction of questions in the cluster that FAILED "
                     "verification; higher is better (the generator succeeded at making a hard "
                     "question). share is the cluster's fraction of this round's questions. "
                     "focus_strategies are ranked by rank_by='" + args.rank_by + "'; the default "
                     "'underrepresented' favours strategies that work (high failure_rate) but are "
                     "RARE (low share) — working approaches the generator under-uses. each carries "
                     "up to examples_per_strategy few_shot_failures — questions that FAILED "
                     "verification — to demonstrate it.",
        },
        "focus_strategies": [strategy_row(i + 1, n) for i, n in enumerate(focus)],
        "cluster_comparison": cluster_comparison,
    }

    if args.out is not None:
        out_path = args.out
    else:
        suffix = f"_{args.annotation}" if args.annotation else ""
        if args.cluster_mode == "seeded":
            suffix += "_seeded"
        out_path = args.dataset_dir / f"strategy_feedback{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    viewer_path = None
    if not args.no_viewer:
        import feedback_viewer
        viewer_path = feedback_viewer.write_viewer(out, out_path.with_suffix(".html"))

    print(f"clustering: {source_label}")
    if args.cluster_mode == "seeded":
        cs = cluster_source
        line = (f"  seeds: {cs['num_seed_clusters_used']}/{cs['num_seeds']} used, "
                f"{cs['num_new_clusters']} new cluster(s)")
        if cs.get("assign_method") == "embedding":
            line += (f" | nearest-seed dist min/mean/max = {cs['nearest_seed_distance_min']}/"
                     f"{cs['nearest_seed_distance_mean']}/{cs['nearest_seed_distance_max']}")
        print(line)
        for lbl in (cs.get("new_cluster_labels") or []):
            if lbl:
                print(f"    new cluster: {lbl}")
    print(f"round:  {total} questions ({num_failed} FAILED, "
          f"overall failure rate {num_failed / total:.3f})" if total else "round: empty")
    print(f"source_runs: {', '.join(source_runs)}")
    rows = out["focus_strategies"]
    total_examples = sum(len(r["few_shot_failures"]) for r in rows)
    print(f"selected {len(rows)} focus strategies, {total_examples} few-shot failure examples total")
    for r in rows:
        print(f"  [{r['failure_rate']:.2f} fail | {r['num_questions']:>2} q | "
              f"{r['share'] * 100:>4.0f}% | {len(r['few_shot_failures'])} ex | "
              f"{r['selected_for']:>12}] {r['description'][:60]}")

    # Per-cluster comparison table: failure rate of each cluster, split by source_run.
    if len(source_runs) > 1:
        print(f"\ncluster comparison (fail_rate n=num_questions, per source_run):")
        header = "  " + "cluster".ljust(9) + "overall".rjust(12)
        header += "".join(run[-18:].rjust(20) for run in source_runs)
        print(header)
        for c in cluster_comparison:
            line = "  " + c["cluster_id"].ljust(9)
            line += f"{c['failure_rate']:.2f} n={c['num_questions']:<3}".rjust(12)
            for run in source_runs:
                b = c["by_source_run"].get(run)
                cell = f"{b['failure_rate']:.2f} n={b['num_questions']:<3}" if b else "-"
                line += cell.rjust(20)
            print(line)
    print(f"\nwrote -> {out_path}")
    if viewer_path is not None:
        print(f"viewer -> {viewer_path}")


if __name__ == "__main__":
    main()
