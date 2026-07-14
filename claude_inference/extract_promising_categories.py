"""Extract promising (high-failure-rate) capability categories from an EvalTree run.

`EvalTree/run_pipeline.drchallenge.sh` produces, for the DRChallenge dataset, two
parallel trees over the same clustering:

  * the stage-4 tree JSON  -> a `description` (capability / strategy phrase) per node
  * confidence_interval.json -> `size` (# questions under the node) and `sum_metrics`
                                (# the agent FAILED, since results.json uses 1 = FAILED)

This script walks both together and emits one row per capability node (every internal
cluster description down to the individual leaf), with the node's failure rate, the
total number of questions under it, and how many failed. Rows are sorted from highest
failure rate to lowest — the top rows are the research capabilities that most reliably
stump the agent, i.e. the promising categories to subsample and dig into.

Usage:
    python extract_promising_categories.py                       # auto-detect the tree
    python extract_promising_categories.py --annotation strategy # pick a variant
    python extract_promising_categories.py --out categories.csv --min-questions 3

Output CSV columns:
    failure_rate, num_questions, num_failed, num_passed, depth, node_id, description
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Default dataset dir: the DRChallenge dataset shipped alongside this script.
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "EvalTree" / "Datasets" / "DRChallenge"


def find_tree_json(dataset_dir: Path, annotation: str | None, latest: bool = False) -> Path:
    """Locate the stage-4 tree JSON (optionally constrained to an annotation variant)."""
    pattern = "stage3-RecursiveClustering/*[stage4-CapabilityDescription-model=*.json"
    candidates = sorted(p for p in (dataset_dir / "EvalTree").glob(pattern))
    if annotation is not None:
        candidates = [p for p in candidates if f"[annotation={annotation}]" in p.name]
    if not candidates:
        sys.exit(f"no stage-4 tree JSON found under {dataset_dir/'EvalTree'} "
                 f"(annotation={annotation}); run the pipeline first")
    if len(candidates) > 1:
        if latest:
            # e.g. a capped and a fully-recursive tree share an annotation; take the
            # freshest, which is the one the pipeline just built.
            return max(candidates, key=lambda p: p.stat().st_mtime)
        names = "\n  ".join(p.name for p in candidates)
        sys.exit(f"multiple trees found; pass --annotation (and/or --latest) to choose:\n  {names}")
    return candidates[0]


def find_ci_json(dataset_dir: Path, model: str, tree_name: str) -> Path:
    """Find the confidence_interval.json whose TREE=[...] matches the given tree file."""
    # The CI dir encodes the same tree params as `TREE=[stage3-RecursiveClustering]_<...>`,
    # minus the trailing `_[stage4-CapabilityDescription-model=...].json` suffix.
    stem = tree_name[: tree_name.index("_[stage4-CapabilityDescription-model=")]
    tree_key = f"TREE=[stage3-RecursiveClustering]_{stem}"
    ci_path = dataset_dir / "eval_results" / "real" / model / "EvalTree" / tree_key / "confidence_interval.json"
    if not ci_path.is_file():
        sys.exit(f"no confidence_interval.json for model={model} at:\n  {ci_path}\n"
                 f"run the CI step of the pipeline (or check --model)")
    return ci_path


def walk(tree, ci, rows: list[dict], depth: int = 0, node_id: str = "root") -> None:
    """Parallel-walk the description tree and the CI tree, collecting one row per node."""
    # Leaves are bare instance indices in both trees; they carry no description.
    if isinstance(tree, int):
        return
    rows.append({
        "failure_rate": ci["sum_metrics"] / ci["size"] if ci["size"] else 0.0,
        "num_questions": ci["size"],
        "num_failed": ci["sum_metrics"],
        "num_passed": ci["size"] - ci["sum_metrics"],
        "depth": depth,
        "node_id": node_id,
        "description": tree.get("description", ""),
    })
    t_sub, c_sub = tree["subtrees"], ci["subtrees"]
    if isinstance(t_sub, dict):
        for k in t_sub:
            walk(t_sub[k], c_sub[k], rows, depth + 1, f"{node_id}.{k}")
    elif isinstance(t_sub, list):
        for i, (t, c) in enumerate(zip(t_sub, c_sub)):
            walk(t, c, rows, depth + 1, f"{node_id}[{i}]")
    else:  # single leaf child (int index) — nothing more to describe
        return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                    help=f"DRChallenge dataset dir (default: {DEFAULT_DATASET_DIR})")
    ap.add_argument("--tree", type=Path, default=None,
                    help="explicit stage-4 tree JSON to read (bypasses auto-detection), resolved "
                         "against CWD; the exact path the pipeline reports as '<...> tree:'")
    ap.add_argument("--annotation", default=None,
                    help="leaf-label variant, e.g. 'strategy' or 'gpt-4o-mini' (default: auto if unambiguous)")
    ap.add_argument("--model", default="drtulu", help="agent name under eval_results/real/ (default: drtulu)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV path (default: <dataset-dir>/promising_categories[_<annotation>].csv)")
    ap.add_argument("--min-questions", type=int, default=1,
                    help="drop categories covering fewer than this many questions (default: 1)")
    ap.add_argument("--latest", action="store_true",
                    help="if several trees share the annotation, use the most recently built one")
    args = ap.parse_args()

    if args.tree is not None:
        tree_path = args.tree  # resolved against CWD like any file path (absolute also fine)
        if not tree_path.is_file():
            sys.exit(f"--tree not found: {tree_path}")
    else:
        tree_path = find_tree_json(args.dataset_dir, args.annotation, latest=args.latest)
    ci_path = find_ci_json(args.dataset_dir, args.model, tree_path.name)
    tree = json.loads(tree_path.read_text())
    ci = json.loads(ci_path.read_text())

    rows: list[dict] = []
    walk(tree, ci, rows)
    rows = [r for r in rows if r["num_questions"] >= args.min_questions]
    # Highest failure rate first; break ties toward larger, more actionable categories.
    rows.sort(key=lambda r: (r["failure_rate"], r["num_questions"]), reverse=True)

    if args.out is not None:
        out_path = args.out
    else:
        ann = next((p.split("=")[1].rstrip("]") for p in tree_path.name.split("_[")
                    if p.startswith("annotation=")), None)
        suffix = f"_{ann}" if ann else ""
        out_path = args.dataset_dir / f"promising_categories{suffix}.csv"

    fields = ["failure_rate", "num_questions", "num_failed", "num_passed", "depth", "node_id", "description"]
    with open(out_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "failure_rate": f"{r['failure_rate']:.4f}"})

    print(f"tree: {tree_path.name}")
    print(f"CI:   {ci_path.parent.name}")
    print(f"wrote {len(rows)} categories -> {out_path}")
    if rows:
        top = rows[0]
        print(f"top: failure_rate={top['failure_rate']:.3f} "
              f"({top['num_failed']}/{top['num_questions']}) — {top['description'][:80]}")


if __name__ == "__main__":
    main()
