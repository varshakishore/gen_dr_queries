#!/usr/bin/env python3
"""
Draw a fixed, stratified random subset of a benchmark directory's questions, for systems
too expensive to run on all of them (frontier models, deep-research products).

Stratified by target system (the system each question was generated against), because
fail rates differ by target: each target gets a share of the subset proportional to its
share of the benchmark (largest-remainder rounding, so the shares sum to --n), and its
questions are drawn at random with --seed. Nothing in the benchmark is changed; the
subset is a new file:

    <bench_dir>/subsets/subset_<n>_seed<seed>.json
      {"name", "n", "seed", "method", "strata": {target: count}, "created_at",
       "source_index", "source_index_sha256", "ids": [...]}

An existing subset file is never overwritten -- systems already run on it would no
longer match it. Use it with answer_cross_systems.py --ids-file.

Also prints a representativeness check: each answering system's fail rate on the subset
next to its rate on the whole benchmark (for a system's own target questions, which it
failed by construction, those questions are left out).

  python make_subset.py runs/final_loop_600_full_benchmark --n 100 --seed 0
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def allocate(sizes: dict, n: int) -> dict:
    """Split n across groups in proportion to their sizes (largest remainder)."""
    total = sum(sizes.values())
    exact = {g: n * s / total for g, s in sizes.items()}
    alloc = {g: math.floor(x) for g, x in exact.items()}
    for g in sorted(exact, key=lambda g: (-(exact[g] - alloc[g]), g))[:n - sum(alloc.values())]:
        alloc[g] += 1
    return alloc


def fail_rates(rows: list) -> dict:
    """{system: (failed, answered)} -- a system's own target questions excluded."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        for s, v in r["verdicts"].items():
            if v is None or s == r["target_system"]:
                continue
            out[s][1] += 1
            out[s][0] += v == "FAILED"
    return dict(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\nAlso prints")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_dir", type=Path, help="An export_benchmark.py directory.")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: <bench_dir>/subsets/subset_<n>_seed<seed>.json")
    ap.add_argument("--dry-run", action="store_true", help="Draw and report; write nothing.")
    args = ap.parse_args()

    index_path = args.bench_dir / "index.jsonl"
    rows = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
    if not 0 < args.n <= len(rows):
        ap.error(f"--n must be between 1 and {len(rows)}")
    out = args.out or args.bench_dir / "subsets" / f"subset_{args.n}_seed{args.seed}.json"
    if out.exists():
        ap.error(f"{out} already exists; a subset is never overwritten (choose another --seed/--out)")

    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target_system"]].append(r["id"])
    alloc = allocate({t: len(v) for t, v in by_target.items()}, args.n)
    rng = random.Random(args.seed)
    ids = []
    for t in sorted(by_target):                          # fixed order -> reproducible draw
        ids += rng.sample(sorted(by_target[t]), alloc[t])
    ids.sort()

    print(f"{len(rows)} questions -> subset of {len(ids)} (seed {args.seed})")
    print("  by target system:", {t: f"{alloc[t]}/{len(by_target[t])}" for t in sorted(alloc)})
    picked = set(ids)
    sub = [r for r in rows if r["id"] in picked]
    dom_all, dom_sub = Counter(r.get("domain") for r in rows), Counter(r.get("domain") for r in sub)
    if any(dom_all):
        print("  by domain (subset/all):", {d: f"{dom_sub[d]}/{dom_all[d]}" for d, _ in dom_all.most_common()})
    full, part = fail_rates(rows), fail_rates(sub)
    print("  fail rate on others' questions, subset vs all:")
    for s in sorted(full):
        fs, ns = part.get(s, (0, 0))
        fa, na = full[s]
        print(f"    {s:<16} subset {fs:>3}/{ns:<3} = {fs / ns if ns else float('nan'):5.1%}   "
              f"all {fa:>3}/{na:<3} = {fa / na if na else float('nan'):5.1%}")
    if args.dry_run:
        print(f"dry run: would write {out}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "name": out.stem, "n": len(ids), "seed": args.seed,
        "method": "stratified by target_system, proportional (largest remainder); "
                  "random.Random(seed).sample within each target, targets in sorted order",
        "strata": {t: alloc[t] for t in sorted(alloc)},
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_index": str(index_path),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "ids": ids}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
