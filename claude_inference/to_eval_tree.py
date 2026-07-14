"""Convert research_pipeline run output into the eval-tree input format.

Each run dir (e.g. runs/sqa_50_100_explore) holds sample_NNN.json files, each
with a list of `results`; every result records the seed and a list of attempts.
The final attempt is the one that determined `final_status`:

  * ALREADY_HARD  -> attempt 0 FAILED; seed used unmodified.
  * FAILED_FOUND  -> a rewritten (harder) attempt FAILED; the rewrite is "found".
  * EXHAUSTED     -> no attempt ever failed; we keep the hardest (last) attempt.
  * ERROR         -> no usable attempt; skipped.

For each kept result we emit:
  { seed_question, updated_question, strategy, verification_criterion }

Usage:
    python to_eval_tree.py runs/sqa_50_100_explore runs/sqa_50_100_original
    # writes <dir>_eval_tree.json next to each run dir
"""

import argparse
import json
import sys
from pathlib import Path


def strategy_label(status: str, chosen: str) -> str:
    if status == "ALREADY_HARD":
        return "ALREADY_HARD (seed already hard; no rewrite)"
    if status == "EXHAUSTED":
        return f"EXHAUSTED (no failing answer found; hardest attempt) | {chosen}"
    # FAILED_FOUND (and any other rewrite-based status): use the chosen strategy.
    return chosen


def result_to_entry(result: dict) -> dict | None:
    status = result.get("final_status", "")
    attempts = result.get("attempts", [])
    if status == "ERROR" or not attempts:
        return None
    # The attempt that set final_status is the last one recorded.
    last = attempts[-1]
    harder = last.get("harder", {})
    return {
        "seed_question": result.get("seed", ""),
        "updated_question": harder.get("updated_question", ""),
        "strategy": strategy_label(status, harder.get("chosen_strategy", "")),
        "verification_criterion": harder.get("verification_criterion", ""),
    }


def convert_dir(run_dir: Path) -> list[dict]:
    entries = []
    for sample_path in sorted(run_dir.glob("sample_*.json")):
        data = json.loads(sample_path.read_text())
        for result in data.get("results", []):
            entry = result_to_entry(result)
            if entry is not None:
                entries.append(entry)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="run directories to convert")
    args = ap.parse_args()

    for d in args.run_dirs:
        run_dir = Path(d)
        if not run_dir.is_dir():
            print(f"[skip] not a directory: {run_dir}", file=sys.stderr)
            continue
        entries = convert_dir(run_dir)
        out_path = run_dir.parent / f"{run_dir.name}_eval_tree.json"
        out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
        print(f"{run_dir.name}: wrote {len(entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
