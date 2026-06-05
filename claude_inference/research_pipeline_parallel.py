#!/usr/bin/env python3
"""
Run research_pipeline.py over many seed questions in parallel.

Each seed is processed by its own `research_pipeline.py` subprocess, N at a time
(default 5). For seed i (1-based) the wrapper writes, inside --out-dir:
  - sample_NNN.json        the pipeline's --output result for that seed
  - run-sample_NNN.jsonl   the pipeline's per-call JSONL log for that seed
  - sample_NNN.console.txt the subprocess's stdout/stderr, ONLY if the run failed
                           (non-zero exit or non FAILED_FOUND/EXHAUSTED status)
and an index.json mapping each index -> seed -> status (since the numbered
filenames are not self-describing), plus per-seed cost/attempts and run totals.

Seeds come from CLI args, a .txt --seeds-file, or (if neither is given) the HF
dataset allenai/asta-user-interactions (optin_queries/train, tool=sqa), capped
at --limit. Seeds whose sample_NNN.json already exists are skipped, so an
interrupted run resumes by re-running the same command (--no-skip-existing forces).

Examples:
  # seeds from the HF dataset (default source), 10 seeds, 5 in parallel
  python research_pipeline_parallel.py --out-dir runs/exp1 --limit 10 --concurrency 5

  # one explicit seed
  python research_pipeline_parallel.py "external memory in LLMs" --out-dir runs/exp1

  # a list of seeds from a file
  python research_pipeline_parallel.py --seeds-file seeds.txt --out-dir runs/exp1
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent / "research_pipeline.py"


def load_seeds(args) -> list[str]:
    seeds = list(args.seeds)
    if args.seeds_file and args.seeds_file.endswith(".txt"):
        with open(args.seeds_file) as f:
            seeds.extend(line.strip() for line in f if line.strip())
    elif not seeds:
        seeds.extend(load_hf_seeds(args))
    return seeds


def load_hf_seeds(args) -> list[str]:
    """Stream unique `query` strings (tool=sqa) from allenai/asta-user-interactions."""
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/asta-user-interactions", "optin_queries",
        split="train", streaming=True,
    )
    seen: set[str] = set()
    out: list[str] = []
    for row in ds:
        if row.get("tool") != "sqa":
            continue
        q = (row.get("query") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if args.limit and len(out) >= args.limit:
            break
    print(f"Loaded {len(out)} unique seed(s) from allenai/asta-user-interactions "
          f"[optin_queries/train, tool=sqa]")
    return out


def run_one(idx: int, seed: str, args, out_dir: Path) -> dict:
    """Run the pipeline for one seed in its own subprocess. Returns a manifest row."""
    tag = f"sample_{idx:03d}"
    result_path = out_dir / f"{tag}.json"
    console_path = out_dir / f"{tag}.console.txt"

    if args.skip_existing and result_path.exists():
        print(f"[skip {idx:>3}/{args._n}] {result_path.name} exists — {seed}", flush=True)
        return {"index": idx, "seed": seed, "file": result_path.name,
                "status": "SKIPPED", "returncode": None}

    cmd = [
        args.python, str(PIPELINE), seed,
        "--output", str(result_path),
        "--run-id", tag,             # -> log file run-sample_NNN.jsonl in --log-dir
        "--log-dir", str(out_dir),
        "--max-attempts", str(args.max_attempts),
        "--model", args.model,
        "--server-url", args.server_url,
        "--quiet",                   # avoid interleaved per-attempt output across workers
    ]

    print(f"[start {idx:>3}/{args._n}] {seed}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # The pipeline catches its own errors and still writes --output; read status back.
    status = "UNKNOWN"
    cost_usd = 0.0
    calls = 0
    attempts = 0  # = number of make-harder prompt calls for this seed
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            results = data.get("results") or []
            if results:
                status = results[0].get("final_status", "UNKNOWN")
                attempts = len(results[0].get("attempts") or [])
            cost = data.get("grand_total_cost") or {}
            cost_usd = float(cost.get("cost_usd") or 0.0)
            calls = int(cost.get("calls") or 0)
        except (ValueError, OSError):
            status = "BAD_OUTPUT"
    elif proc.returncode != 0:
        status = "SUBPROCESS_FAILED"

    # Keep the console capture only when something went wrong (it holds the traceback);
    # a clean run's stdout is redundant with the result + log files.
    failed = proc.returncode != 0 or status not in ("FAILED_FOUND", "EXHAUSTED")
    console_name = None
    if failed:
        console_path.write_text(
            proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
        )
        console_name = console_path.name

    print(f"[done  {idx:>3}/{args._n}] [{status}] ${cost_usd:.4f} "
          f"rc={proc.returncode} — {seed}", flush=True)
    row = {"index": idx, "seed": seed, "file": result_path.name,
           "log": f"run-{tag}.jsonl", "status": status,
           "returncode": proc.returncode, "cost_usd": cost_usd,
           "claude_calls": calls, "attempts": attempts}
    if console_name:
        row["console"] = console_name
    return row


def main():
    p = argparse.ArgumentParser(description="Run research_pipeline.py over seeds in parallel.")
    p.add_argument("seeds", nargs="*",
                   help="Seed questions. If none given (and no .txt --seeds-file), "
                        "seeds are pulled from the allenai/asta-user-interactions HF dataset.")
    p.add_argument("--seeds-file", help="A .txt file with one seed per line.")
    p.add_argument("--out-dir", required=True, help="Folder for per-seed results, logs, and index.json.")
    p.add_argument("--concurrency", type=int, default=5, help="Max seeds in flight (default: 5).")
    # HF dataset source (used when no explicit seeds / .txt file are given)
    p.add_argument("--limit", type=int, default=100,
                   help="Max seeds to pull from the HF dataset (default: 100; 0 = all).")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--model", default="claude-sonnet-4-5")
    p.add_argument("--server-url", default="http://localhost:8007/ask")
    p.add_argument("--python", default=sys.executable, help="Python interpreter for subprocesses.")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Re-run seeds even if their sample_NNN.json already exists.")
    p.set_defaults(skip_existing=True)
    args = p.parse_args()

    seeds = load_seeds(args)
    if not seeds:
        p.error("No seed questions provided.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args._n = len(seeds)
    workers = max(1, min(args.concurrency, len(seeds)))
    print(f"Running {len(seeds)} seed(s), {workers} at a time -> {out_dir}/")

    rows: list[dict] = [None] * len(seeds)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one, i + 1, seed, args, out_dir): i
                   for i, seed in enumerate(seeds)}
        for fut in as_completed(futures):
            rows[futures[fut]] = fut.result()

    total_cost = sum(r.get("cost_usd", 0.0) for r in rows)
    total_calls = sum(r.get("claude_calls", 0) for r in rows)

    # Average make-harder calls (= attempts) over FAILED_FOUND seeds only, where the
    # count is exact (the failing attempt's index).
    ff_attempts = [r["attempts"] for r in rows if r.get("status") == "FAILED_FOUND"]
    avg_attempts_ff = (sum(ff_attempts) / len(ff_attempts)) if ff_attempts else None

    (out_dir / "index.json").write_text(json.dumps(
        {"model": args.model, "max_attempts": args.max_attempts,
         "total_cost_usd": total_cost, "total_claude_calls": total_calls,
         "avg_make_harder_calls_failed_found": avg_attempts_ff,
         "num_failed_found": len(ff_attempts),
         "samples": rows}, indent=2
    ))

    print(f"\n{'#' * 70}\nSUMMARY\n{'#' * 70}")
    from collections import Counter
    counts = Counter(r["status"] for r in rows)
    for status, n in sorted(counts.items()):
        print(f"  {status:18} {n}")
    print(f"\nTotal cost: ${total_cost:.4f} across {total_calls} Claude calls "
          f"({len(rows)} seed(s))")
    if avg_attempts_ff is not None:
        print(f"Avg make-harder calls per FAILED_FOUND seed: {avg_attempts_ff:.2f} "
              f"(over {len(ff_attempts)} seed(s))")
    else:
        print("Avg make-harder calls per FAILED_FOUND seed: n/a (no FAILED_FOUND seeds)")
    print(f"Index written to: {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
