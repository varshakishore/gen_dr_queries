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
                 max_strategies: int) -> list[dict]:
    """Pick a size-DIVERSE set of clusters that work (stump the agent often enough).

    A cluster is eligible when it is a real theme (>= min_cluster_size questions, and no
    larger than max_cluster_size if set), it stumps the agent often enough
    (failure_rate >= min_failure_rate), and it is not an over-represented catch-all
    (share <= max_share).

    Small clusters are NOT excluded. To get a diverse mix rather than only the tiny
    perfect-failure clusters (what a pure failure_rate ranking gives) or only the biggest
    ones (what a pure volume ranking gives), we interleave two rankings and pull from each
    in turn:
      * by_rate — highest failure_rate first (surfaces small, strongly-failing strategies)
      * by_volume — most absolute failures first (surfaces bigger, medium-rate strategies)
    Each picked cluster is tagged `selected_for` = "failure_rate" or "volume" so the mix
    is visible. `score` (num_failed + failure_rate) is reported for reference only.
    """
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
        eligible.append({**n, "share": share, "score": n["num_failed"] + n["failure_rate"]})

    rankings = [
        ("failure_rate", sorted(eligible, key=lambda n: (n["failure_rate"], n["num_failed"]), reverse=True)),
        ("volume", sorted(eligible, key=lambda n: (n["num_failed"], n["failure_rate"]), reverse=True)),
    ]

    # Round-robin between the two rankings: take the next not-yet-picked cluster from each
    # in turn, until we have max_strategies or both rankings are exhausted.
    picked: list[dict] = []
    picked_ids: set[str] = set()
    cursors = [0, 0]
    turn = 0
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
    ap.add_argument("--max-strategies", type=int, default=4,
                    help="max focus strategies to emit (default: 4)")
    ap.add_argument("--examples-per-strategy", type=int, default=5,
                    help="max few-shot failure examples attached to EACH focus strategy (default: 5)")
    args = ap.parse_args()

    if args.tree is not None:
        tree_path = args.tree
        if not tree_path.is_file():
            sys.exit(f"--tree not found: {tree_path}")
    else:
        tree_path = find_tree_json(args.dataset_dir, args.annotation, latest=args.latest)
    ci_path = find_ci_json(args.dataset_dir, args.model, tree_path.name)

    tree = json.loads(tree_path.read_text())
    ci = json.loads(ci_path.read_text())
    dataset = json.loads((args.dataset_dir / "dataset.json").read_text())
    results = json.loads((ci_path.parents[2] / "results.json").read_text())

    total = len(dataset)
    num_failed = sum(1 for r in results if r == 1)

    nodes: list[dict] = []
    collect_nodes(tree, ci, nodes)

    focus = select_focus(
        nodes, total,
        min_failure_rate=args.min_failure_rate,
        max_share=args.max_share,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        max_strategies=args.max_strategies,
    )
    focus = dedup_descendant_clusters(focus)[:args.max_strategies]

    # Up to N failure examples per focus strategy; `seen` is shared so no question repeats.
    seen_examples: set[str] = set()

    source_runs = sorted({inst.get("source_run") or "unknown" for inst in dataset})

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
                f"Stumped the agent on {n['num_failed']}/{n['num_questions']} questions "
                f"(failure rate {n['failure_rate']:.2f}, {n['share'] * 100:.0f}% of this round) — "
                f"a strategy that works with room to use more, so lean into it next round."
            ),
            "few_shot_failures": examples,
        }

    out = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_dir": str(args.dataset_dir),
            "tree": tree_path.name,
            "confidence_interval": str(ci_path),
            "model": args.model,
            "source_runs": source_runs,
            "num_instances": total,
            "num_failed": num_failed,
            "overall_failure_rate": round(num_failed / total, 4) if total else 0.0,
            "selection": {
                "min_failure_rate": args.min_failure_rate,
                "max_share": args.max_share,
                "min_cluster_size": args.min_cluster_size,
                "max_cluster_size": args.max_cluster_size,
                "max_strategies": args.max_strategies,
                "examples_per_strategy": args.examples_per_strategy,
            },
            "notes": "failure_rate is the fraction of questions in the cluster that FAILED "
                     "verification; higher is better (the generator succeeded at making a hard "
                     "question). focus_strategies are a size-diverse mix: some picked for high "
                     "failure_rate (selected_for=failure_rate, often small clusters), some for "
                     "volume of failures (selected_for=volume, larger/medium clusters). each "
                     "carries up to examples_per_strategy few_shot_failures — questions that "
                     "FAILED verification — to demonstrate it.",
        },
        "focus_strategies": [strategy_row(i + 1, n) for i, n in enumerate(focus)],
    }

    if args.out is not None:
        out_path = args.out
    else:
        suffix = f"_{args.annotation}" if args.annotation else ""
        out_path = args.dataset_dir / f"strategy_feedback{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"tree:   {tree_path.name}")
    print(f"CI:     {ci_path.parent.name}")
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
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
