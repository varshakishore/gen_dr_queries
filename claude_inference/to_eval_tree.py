"""Convert claude_inference run output into EvalTree input for the DRChallenge dataset.

The research pipeline writes, per sample, a `sample_NNN.json` file holding a list of
`results`. Each result records a `seed` question and a list of `attempts`; each attempt
carries the hardened question it posed (`harder.updated_question`, plus the attack
`chosen_strategy` and `verification_criterion`) and the grader's `judgment.verdict`
("PASSED" / "FAILED"). The final attempt is the one that determined `final_status`:

  * ALREADY_HARD  -> attempt 0 FAILED; seed used unmodified.
  * FAILED_FOUND  -> a rewritten (harder) attempt FAILED; the rewrite is "found".
  * EXHAUSTED     -> no attempt ever failed; the hardest (last) attempt is kept.
  * ERROR         -> no usable attempt; skipped.

This script assembles those runs into the three artifacts the EvalTree DRChallenge
pipeline consumes (see Datasets/DRChallenge/README.md):

  <out-dir>/dataset.json                          # one instance per graded question
  <out-dir>/eval_results/real/<model>/results.json  # 0/1 per instance (1 = FAILED)
  <out-dir>/splits/train-test.json                # [0 .. N-1]

Each dataset instance is:
  { seed_question, updated_question, strategy, verification_criterion, drtulu_verdict,
    source_run }
where `source_run` names the run folder the instance came from (so the combined dataset
can be subsampled per run; EvalTree ignores the extra field). `results.json[i]` is 1
when instance i's verdict is FAILED, else 0 (dataset order).

By default every graded attempt across all runs becomes an instance (the deciding /
"challenge" attempt of each result first, then the other attempts the agent also
answered — mirroring the dataset's failing-then-passing ordering), deduplicated by
`updated_question`. Use --deciding-only to keep just the one challenge question per
result (the original behaviour), or --keep-run-duplicates to let the same question
appear once per run it came from.

Round 0 (the seed tested as-is) is OMITTED by default — those instances are the
original seed questions, not generated/hardened ones. Pass --include-seed-round to
keep them. Each instance records its `round` (0 = seed, 1..N = harder rewrites).

Usage:
    # Build a fresh DRChallenge dataset dir from any run outputs:
    python to_eval_tree.py runs/sqa_50_100_explore runs/sqa_50_100_original \
        --out-dir EvalTree/Datasets/DRChallenge

    # A run path may be a run dir, a parent of run dirs, or a single sample_*.json.
"""

import argparse
import json
import sys
from pathlib import Path


def strategy_label(status: str, chosen: str) -> str:
    """Human-readable strategy for the *deciding* attempt of a result."""
    if status == "ALREADY_HARD":
        return "ALREADY_HARD (seed already hard; no rewrite)"
    if status == "EXHAUSTED":
        return f"EXHAUSTED (no failing answer found; hardest attempt) | {chosen}"
    # FAILED_FOUND (and any other rewrite-based status): use the chosen strategy.
    return chosen


def verdict_of(attempt: dict) -> str | None:
    """The grader's verdict for an attempt, or None if it wasn't (validly) graded."""
    judgment = attempt.get("judgment")
    if isinstance(judgment, dict):
        verdict = judgment.get("verdict")
        if verdict in ("PASSED", "FAILED"):
            return verdict
    return None


def attempt_to_entry(result: dict, attempt: dict, is_deciding: bool, round_idx: int) -> dict | None:
    """Turn one graded attempt into a dataset instance, or None if unusable."""
    verdict = verdict_of(attempt)
    harder = attempt.get("harder", {})
    updated = harder.get("updated_question", "")
    if verdict is None or not updated:
        return None
    if is_deciding:
        strategy = strategy_label(result.get("final_status", ""), harder.get("chosen_strategy", ""))
    else:
        strategy = harder.get("chosen_strategy", "")
    return {
        "seed_question": result.get("seed", ""),
        "updated_question": updated,
        "strategy": strategy,
        "verification_criterion": harder.get("verification_criterion", ""),
        "drtulu_verdict": verdict,
        # Round index within the seed's run: 0 = seed tested as-is, 1..N = harder
        # rewrites. Retained so round 0 (the seed itself) can be filtered downstream.
        "round": round_idx,
        # Originating run folder, retained so the combined dataset can be
        # subsampled per run (EvalTree ignores unknown fields).
        "source_run": "",
    }


def iter_sample_files(path: Path):
    """Yield sample_*.json files under `path` (a file, run dir, or parent of run dirs)."""
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        print(f"[skip] not a file or directory: {path}", file=sys.stderr)
        return
    direct = sorted(path.glob("sample_*.json"))
    if direct:
        yield from direct
    else:
        # Treat `path` as a parent of run dirs and search one level deeper.
        yield from sorted(path.glob("*/sample_*.json"))


def load_results(sample_path: Path) -> list[dict]:
    """Return the `results` list from a sample file (tolerant of a bare list top level)."""
    data = json.loads(sample_path.read_text())
    if isinstance(data, list):
        return data
    return data.get("results", [])


def collect_entries(run_paths: list[Path], *, deciding_only: bool, keep_run_dups: bool,
                    include_seed_round: bool) -> list[dict]:
    """Assemble dataset instances from all runs, deciding attempts first, then deduped."""
    deciding: list[dict] = []
    other: list[dict] = []
    for run_path in run_paths:
        # A run's identity (for --keep-run-duplicates) is the dir holding its samples.
        for sample_path in iter_sample_files(run_path):
            run_key = str(sample_path.parent)
            for result in load_results(sample_path):
                if result.get("final_status") == "ERROR":
                    continue
                attempts = result.get("attempts", [])
                if not attempts:
                    continue
                last_i = len(attempts) - 1
                for i, attempt in enumerate(attempts):
                    round_idx = attempt.get("attempt", i)  # 0 = seed as-is
                    # Skip the seed round (round 0) by default: those are the original
                    # seed questions, not generated/hardened ones.
                    if round_idx == 0 and not include_seed_round:
                        continue
                    is_deciding = i == last_i
                    if deciding_only and not is_deciding:
                        continue
                    entry = attempt_to_entry(result, attempt, is_deciding, round_idx)
                    if entry is None:
                        continue
                    entry["source_run"] = Path(run_key).name
                    entry["_run_key"] = run_key
                    (deciding if is_deciding else other).append(entry)

    # Deciding/failing questions first, then the remaining attempts — mirrors the
    # dataset's "selected questions, then intermediate passing cases" index order.
    seen: set = set()
    entries: list[dict] = []
    for entry in deciding + other:
        dedup_key = (entry["_run_key"], entry["updated_question"]) if keep_run_dups else entry["updated_question"]
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        entries.append({k: v for k, v in entry.items() if k != "_run_key"})
    return entries


def print_subset_summary(entries: list[dict]) -> None:
    """Print a per-source_run breakdown so subset differences are visible at build time."""
    from collections import Counter

    runs = sorted({e.get("source_run") or "unknown" for e in entries})
    if len(runs) <= 1:
        return  # single subset — nothing to compare
    failed = Counter()
    total = Counter()
    for e in entries:
        run = e.get("source_run") or "unknown"
        total[run] += 1
        if e["drtulu_verdict"] == "FAILED":
            failed[run] += 1
    width = max(len(r) for r in runs)
    print("per-subset breakdown (source_run):")
    print(f"  {'subset'.ljust(width)}  {'n':>5}  {'FAILED':>6}  {'PASSED':>6}  {'fail_rate':>9}")
    for run in runs:
        n, f = total[run], failed[run]
        rate = f"{f / n:.3f}" if n else "-"
        print(f"  {run.ljust(width)}  {n:>5}  {f:>6}  {n - f:>6}  {rate:>9}")


def write_dataset(out_dir: Path, entries: list[dict], model_name: str) -> None:
    """Write dataset.json, results.json, and splits/train-test.json under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    results = [1 if e["drtulu_verdict"] == "FAILED" else 0 for e in entries]
    results_dir = out_dir / "eval_results" / "real" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(results))

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train-test.json").write_text(json.dumps(list(range(len(entries)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_paths", nargs="+", help="run dirs, parents of run dirs, or sample_*.json files")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="output dataset dir (e.g. EvalTree/Datasets/DRChallenge)")
    ap.add_argument("--model-name", default="drtulu",
                    help="agent name for eval_results/real/<model>/results.json (default: drtulu)")
    ap.add_argument("--deciding-only", action="store_true",
                    help="keep only the one challenge question per result (skip other attempts)")
    ap.add_argument("--keep-run-duplicates", action="store_true",
                    help="dedup per (run, question) instead of globally, keeping cross-run repeats")
    ap.add_argument("--include-seed-round", action="store_true",
                    help="also include round-0 (seed-as-is) questions; by default they are omitted")
    args = ap.parse_args()

    entries = collect_entries(
        [Path(p) for p in args.run_paths],
        deciding_only=args.deciding_only,
        keep_run_dups=args.keep_run_duplicates,
        include_seed_round=args.include_seed_round,
    )
    if not entries:
        print("no usable graded attempts found in the given run paths", file=sys.stderr)
        sys.exit(1)

    write_dataset(args.out_dir, entries, args.model_name)

    failed = sum(1 for e in entries if e["drtulu_verdict"] == "FAILED")
    print(f"wrote {len(entries)} instances ({failed} FAILED / {len(entries) - failed} PASSED) -> {args.out_dir}")
    print_subset_summary(entries)
    print(f"  {args.out_dir / 'dataset.json'}")
    print(f"  {args.out_dir / 'eval_results' / 'real' / args.model_name / 'results.json'}")
    print(f"  {args.out_dir / 'splits' / 'train-test.json'}")


if __name__ == "__main__":
    main()
