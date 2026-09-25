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
dataset varshak1/asta-user-interactions-filtered (filter_queries.py's output),
keeping only usable=true rows, capped at --limit. Seeds whose sample_NNN.json
already exists are skipped, so an interrupted run resumes by re-running the same
command (--no-skip-existing forces).

Examples:
  # seeds from the HF dataset (default source), 10 seeds, 5 in parallel
  python research_pipeline_parallel.py --out-dir runs/exp1 --limit 10 --concurrency 5

  # one explicit seed
  python research_pipeline_parallel.py "external memory in LLMs" --out-dir runs/exp1

  # a list of seeds from a file
  python research_pipeline_parallel.py --seeds-file seeds.txt --out-dir runs/exp1

Usage:
python research_pipeline_parallel.py --out-dir runs/sqa_50_100_explore --start 50 --limit 50 --concurrency 10 --prompt explore && python summarize_run.py runs/sqa_50_100_explore && python research_pipeline_parallel.py --out-dir runs/sqa_50_100_exploit --start 50 --limit 50 --concurrency 10 --prompt exploit && python summarize_run.py runs/sqa_50_100_exploit
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import llm_client
import research_pipeline as RP

PIPELINE = Path(__file__).resolve().parent / "research_pipeline.py"

# Rubric-labeled seeds from filter_queries.py (tool=sqa by construction).
SEEDS_DATASET = "varshak1/asta-user-interactions-filtered"


def load_seeds(args) -> list[str]:
    """Seeds from CLI args, a --seeds-file, or (failing both) the HF dataset.

    A `.txt` file is one seed per line, so it CANNOT carry a seed containing newlines --
    a multi-line seed silently fragments into several. Use `.json` (a list of strings)
    whenever the seeds are machine-generated; research_loop.py always does.
    """
    seeds = list(args.seeds)
    if args.seeds_file:
        path = Path(args.seeds_file)
        if path.suffix == ".json":
            loaded = json.loads(path.read_text())
            if not isinstance(loaded, list) or not all(isinstance(x, str) for x in loaded):
                raise ValueError(f"{path}: expected a JSON list of seed strings")
            seeds.extend(x.strip() for x in loaded if x.strip())
        elif path.suffix == ".txt":
            seeds.extend(ln.strip() for ln in path.read_text().splitlines() if ln.strip())
        else:
            raise ValueError(f"{path}: --seeds-file must be .txt (one per line) or .json")
    elif not seeds:
        seeds.extend(load_hf_seeds(args))
    return seeds


def load_hf_seeds(args) -> list[str]:
    """Stream unique `query` strings (usable=true) from the filter_queries.py dataset."""
    from datasets import load_dataset

    ds = load_dataset(SEEDS_DATASET, split="train", streaming=True)
    start = getattr(args, "start", 0) or 0
    need = (start + args.limit) if args.limit else None  # how many to collect before slicing
    seen: set[str] = set()
    out: list[str] = []
    for row in ds:
        if not row.get("usable"):
            continue
        q = (row.get("query") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if need and len(out) >= need:
            break
    sliced = out[start: (start + args.limit) if args.limit else None]
    print(f"Loaded {len(sliced)} unique seed(s) from {SEEDS_DATASET} "
          f"[train, usable=true, seeds {start}:{start + len(sliced)}]")
    return sliced


def scan_existing(seeds: list[str], out_dir: Path) -> tuple[dict, list]:
    """Recover manifest rows for seeds already on disk; flag any seed/file mispairing.

    Two jobs, both of which used to be skipped along with the seed:

    * **Recover the row.** Skipping once returned `status: "SKIPPED"` with no cost and no
      attempts, and `index.json` is rewritten wholesale on every invocation -- so resuming
      a run REPLACED its real statuses with SKIPPED and its cost with 0. Every reporter
      reads status off the index row (`summarize_run.load_run`, `loop_report.side_counts`),
      so a resumed run under-reported its own harvest while the results sat intact on disk.
      Re-reading the file costs one parse per finished seed and keeps the index truthful.
    * **Check the pairing.** The skip is by FILENAME -- `sample_007.json` exists, so seed 7
      is done -- and the numbering is positional. Change anything that shifts the split
      (`--limit`, `--start`, `--feedback-every`, `--prompt-mix`, dropping a system from
      `--systems-file`) and seed 7 is a different question than the one whose result is in
      that file. Nothing noticed: the index would record the NEW seed text beside the OLD
      result, and the new seed would never run at all. Results carry their own seed, so the
      mispairing is detectable -- cheaply, and before anything is spent.

    Returns `({index: row}, [(index, wanted_seed, stored_seed), ...])`.
    """
    rows: dict = {}
    mismatches: list = []
    for idx, seed in enumerate(seeds, start=1):
        path = out_dir / f"sample_{idx:03d}.json"
        if not path.exists():
            continue
        row = {"index": idx, "seed": seed, "file": path.name,
               "returncode": None, "resumed": True}
        try:
            data = json.loads(path.read_text())
            res = (data.get("results") or [{}])[0]
        except (ValueError, OSError):
            # Truncated by a kill mid-write: the pipeline's json.dump is not atomic, so a
            # half-file looks "done" to the skip. Surface it instead of counting it.
            row["status"] = "BAD_OUTPUT"
            rows[idx] = row
            continue
        stored = (res.get("seed") or "").strip()
        if stored and stored != seed.strip():
            mismatches.append((idx, seed, stored))
            continue
        cost = data.get("grand_total_cost") or {}
        row["status"] = res.get("final_status", "UNKNOWN")
        row["attempts"] = len(res.get("attempts") or [])
        row["cost_usd"] = float(cost.get("cost_usd") or 0.0)
        row["claude_calls"] = int(cost.get("calls") or 0)
        rows[idx] = row
    return rows, mismatches


def run_one(idx: int, seed: str, args, out_dir: Path) -> dict:
    """Run the pipeline for one seed in its own subprocess. Returns a manifest row."""
    tag = f"sample_{idx:03d}"
    result_path = out_dir / f"{tag}.json"
    console_path = out_dir / f"{tag}.console.txt"

    done = args._resumed.get(idx)
    if done is not None:
        print(f"[skip {idx:>3}/{args._n}] {result_path.name} exists "
              f"({done['status']}) — {seed}", flush=True)
        return done

    cmd = [
        args.python, str(PIPELINE),
        "--output", str(result_path),
        "--run-id", tag,             # -> log file run-sample_NNN.jsonl in --log-dir
        "--log-dir", str(out_dir),
        "--max-attempts", str(args.max_attempts),
        "--model", args.model,
        "--prompt", args.prompt,
        "--server-url", args.server_url,
        "--quiet",                   # avoid interleaved per-attempt output across workers
    ]
    trailing = ["--", seed]          # after `--`: a seed starting with '-' is not a flag
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.strategies:
        cmd += ["--strategies", args.strategies]
    if args.banned_strategies:
        cmd += ["--banned-strategies", args.banned_strategies]
    if args.few_shots_file:
        cmd += ["--few-shots-file", str(args.few_shots_file)]
    # A per-seed menu wins over the round-wide one: research_loop samples a fresh example
    # menu for every seed, so each seed sees its own draw from the strategy pool. Falls
    # back to the shared file when this seed has no menu (explore side, or round 0).
    seed_menu = Path(args.strategies_dir) / f"{tag}.txt" if args.strategies_dir else None
    if seed_menu and seed_menu.exists():
        cmd += ["--strategies-file", str(seed_menu)]
    elif args.strategies_file:
        cmd += ["--strategies-file", str(args.strategies_file)]
    if args.banned_strategies_file:
        cmd += ["--banned-strategies-file", str(args.banned_strategies_file)]
    if args.timeout:
        cmd += ["--timeout", str(args.timeout)]
    cmd += llm_client.forward_provider_args(args)
    if args.verify_criterion:
        cmd += ["--verify-criterion",
                "--verify-n-papers", str(args.verify_n_papers),
                "--verify-max-extra-queries", str(args.verify_max_extra_queries),
                "--verify-max-chars-per-paper", str(args.verify_max_chars_per_paper),
                "--reranker", args.reranker]
        if args.verify_propose_queries:
            cmd += ["--verify-propose-queries"]
        if args.include_seed_round:
            cmd += ["--include-seed-round"]
        if args.reranker_url:
            cmd += ["--reranker-url", args.reranker_url]
    if args.skip_seed_round:                    # independent of --verify-criterion
        cmd += ["--skip-seed-round"]

    print(f"[start {idx:>3}/{args._n}] {seed[:160]}", flush=True)
    proc = subprocess.run(cmd + trailing, capture_output=True, text=True)

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
    failed = proc.returncode != 0 or status not in (
        "FAILED_FOUND", "EXHAUSTED", "CRITERION_INVALID", "ALREADY_HARD"
    )
    console_name = None
    if failed:
        console_path.write_text(
            proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
        )
        console_name = console_path.name

    print(f"[done  {idx:>3}/{args._n}] [{status}] ${cost_usd:.4f} "
          f"rc={proc.returncode} — {seed[:160]}", flush=True)
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
    p.add_argument("--seeds-file",
                   help="A .txt file with one seed per line, or a .json list of seed "
                        "strings (use .json if a seed can contain newlines).")
    p.add_argument("--out-dir", required=True, help="Folder for per-seed results, logs, and index.json.")
    p.add_argument("--concurrency", type=int, default=5, help="Max seeds in flight (default: 5).")
    # HF dataset source (used when no explicit seeds / .txt file are given)
    p.add_argument("--limit", type=int, default=100,
                   help="Max seeds to pull from the HF dataset (default: 100; 0 = all).")
    p.add_argument("--start", type=int, default=0,
                   help="Skip the first N HF seeds before taking --limit "
                        "(e.g. --start 50 --limit 50 = seeds 50-99).")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--model", default=RP.DEFAULT_MODEL,
                   help=f"Generator/judge model forwarded to the pipeline (default: "
                        f"{RP.DEFAULT_MODEL}). Accepts a Claude "
                        "id or an OpenAI one (e.g. gpt-5.6-terra); the provider is inferred "
                        "from the id unless --provider says otherwise.")
    p.add_argument("--prompt", choices=["explore", "exploit"], default="explore",
                   help="make-harder prompt variant passed to the pipeline (default: explore).")
    p.add_argument("--profile", help="Answering-system profile name passed to the pipeline "
                                     "(e.g. drtulu, tongyi).")
    p.add_argument("--banned-strategies",
                   help="Banned-strategy menu name passed to the pipeline "
                        "(--banned-strategies); only used with --prompt explore.")
    p.add_argument("--strategies", help="Example-strategy menu name passed to the pipeline "
                                        "(e.g. default, jena_cog_biases). Only affects "
                                        "--prompt exploit.")
    p.add_argument("--few-shots-file",
                   help="JSON file of worked examples passed to the pipeline, shown instead "
                        "of the built-in three. Applies to both prompt variants.")
    p.add_argument("--strategies-file",
                   help="File of example strategies (one per line) passed to the pipeline, "
                        "used instead of --strategies. Only affects --prompt exploit.")
    p.add_argument("--strategies-dir",
                   help="Directory of per-seed example-strategy menus, named sample_NNN.txt "
                        "to match this run's seed numbering. A seed with a menu here uses it "
                        "instead of --strategies-file, so every seed can be shown its own "
                        "sample of the strategy pool. Only affects --prompt exploit.")
    p.add_argument("--banned-strategies-file",
                   help="File of banned strategies (one per line) passed to the pipeline, "
                        "used instead of --banned-strategies. Only affects --prompt explore.")
    p.add_argument("--timeout", type=float,
                   help="Read timeout (s) per research-server call passed to the pipeline "
                        "(e.g. 7200 for slow models like Tongyi).")
    p.add_argument("--server-url", default="http://localhost:8007/ask")
    # Criterion verification, forwarded to the pipeline (see research_pipeline.py).
    p.add_argument("--verify-criterion", action="store_true",
                   help="Verify each harder question's criterion against retrieved S2 papers "
                        "before querying the research server. Needs S2_API_KEY.")
    p.add_argument("--skip-seed-round", action="store_true",
                   help="Start at attempt 1 rather than testing the unmodified seed. "
                        "Forwarded to the pipeline; see its --skip-seed-round.")
    p.add_argument("--include-seed-round", action="store_true",
                   help="Also verify the round-0 criterion (the unmodified seed). Forwarded "
                        "to the pipeline; see its --include-seed-round.")
    p.add_argument("--verify-propose-queries", action="store_true",
                   help="Ask the model for criterion-aware search queries before retrieving for the criterion check, on top of the question's own retrieval. Forwarded to the pipeline; see its --verify-propose-queries.")
    p.add_argument("--verify-max-extra-queries", type=int, default=RP.VERIFY_MAX_EXTRA_QUERIES,
                   help="Cap on criterion-aware queries per check (default: "
                        f"{RP.VERIFY_MAX_EXTRA_QUERIES}); 0 disables them.")
    p.add_argument("--verify-n-papers", type=int, default=15,
                   help="Papers in the criterion-check context (default: 15).")
    p.add_argument("--verify-max-chars-per-paper", type=int, default=4000,
                   help="Per-paper char cap in the criterion-check context (default: 4000).")
    p.add_argument("--reranker", default="auto", choices=["auto", "none", "vllm"],
                   help="Reranker for criterion-check retrieval (default: auto).")
    p.add_argument("--provider", choices=["auto", "anthropic", "openai"], default="auto",
                   help="Provider for --model, forwarded to the pipeline. 'auto' (default) "
                        "infers it from the model id.")
    p.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"],
                   default=None,
                   help="OpenAI models only: reasoning_effort forwarded to the pipeline.")
    p.add_argument("--decomposer-model", default=None,
                   help="Model for retrieve_papers' query decomposition under "
                        "--verify-criterion; independent of --model.")
    p.add_argument("--reranker-url", default=None,
                   help="vLLM reranker base URL (env: VLLM_RERANK_URL).")
    p.add_argument("--skip-server-check", action="store_true",
                   help="Skip the startup probe of --server-url / --reranker-url.")
    p.add_argument("--python", default=sys.executable, help="Python interpreter for subprocesses.")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Re-run seeds even if their sample_NNN.json already exists.")
    p.set_defaults(skip_existing=True)
    args = p.parse_args()
    llm_client.configure_from_args(args)
    provider = llm_client.resolve_provider(args.model)
    if missing := llm_client.require_api_key(args.model):
        p.error(f"{missing} Needed for generation.")
    if args.verify_criterion:
        # Resolve what retrieval will really use -- it follows --model unless overridden.
        args.decomposer_model = RP.effective_decomposer_model(args.decomposer_model,
                                                              args.model)
        if args.decomposer_model and (
                missing := llm_client.require_api_key(args.decomposer_model)):
            p.error(missing + llm_client.decomposer_hint(args.model, args.decomposer_model))
        # The pipeline raises both of these per seed, but this driver runs each seed with
        # capture_output=True and keeps the output only when the run FAILS -- so on a
        # healthy run they are discarded. Repeat them here, once, where they are visible.
        if not os.environ.get("S2_API_KEY"):
            print("[verify] WARNING: S2_API_KEY is not set; criterion-check retrieval will "
                  "be rate limited hard by Semantic Scholar.", file=sys.stderr)
        if not (args.reranker_url or os.environ.get("VLLM_RERANK_URL")) \
                and args.reranker != "none":
            print("[verify] WARNING: no --reranker-url / VLLM_RERANK_URL; criterion-check "
                  "retrieval will run WITHOUT reranking, which biases the meta-judge toward "
                  "accepting 'the literature does not cover X' claims.", file=sys.stderr)
    if not args.skip_server_check:
        problems = [f"  answering server: {why}"
                    for why in [RP.check_server_reachable(args.server_url)] if why]
        if args.reranker_url and args.reranker != "none":
            why = RP.check_server_reachable(args.reranker_url)
            if why:
                problems.append(f"  reranker: {why}")
        if problems:
            p.error("cannot reach every server this run needs:\n" + "\n".join(problems)
                    + "\n\nStart them, fix the URL, or pass --skip-server-check.")

    seeds = load_seeds(args)
    if not seeds:
        p.error("No seed questions provided.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args._n = len(seeds)

    # Resume pre-pass: read what is already on disk ONCE, before anything is spent, so
    # a shifted seed->file numbering is caught here rather than silently pairing stale
    # results with new seeds (see scan_existing). run_one then just returns these rows.
    args._resumed = {}
    if args.skip_existing:
        args._resumed, mismatches = scan_existing(seeds, out_dir)
        if mismatches:
            shown = "\n".join(
                f"  sample_{i:03d}.json holds {stored[:70]!r}\n"
                f"      but seed {i} of this run is {want[:70]!r}"
                for i, want, stored in mismatches[:5])
            more = (f"\n  ... and {len(mismatches) - 5} more"
                    if len(mismatches) > 5 else "")
            p.error(
                f"{len(mismatches)} seed(s) do not match the results already in "
                f"{out_dir}/:\n{shown}{more}\n\n"
                "Seeds are matched to files by POSITION, so this means the seed list "
                "shifted since that run -- a different --start/--limit, --feedback-every, "
                "--prompt-mix, or a system dropped from --systems-file. Resuming would "
                "pair stale results with the wrong questions. Use the original seed "
                "arguments, pick a fresh --out-dir, or pass --no-skip-existing to "
                "overwrite.")
        if args._resumed:
            done = Counter(r["status"] for r in args._resumed.values())
            print(f"Resuming: {len(args._resumed)}/{len(seeds)} seed(s) already done "
                  f"({', '.join(f'{k} {v}' for k, v in sorted(done.items()))}) — "
                  f"re-reading their results, not re-running them.")

    workers = max(1, min(args.concurrency, len(seeds)))
    print(f"Running {len(seeds)} seed(s), {workers} at a time -> {out_dir}/  "
          f"[{args.model} via {provider}]")

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
        {"model": args.model, "provider": provider, "max_attempts": args.max_attempts,
         "total_cost_usd": total_cost, "total_claude_calls": total_calls,
         "avg_make_harder_calls_failed_found": avg_attempts_ff,
         "num_failed_found": len(ff_attempts),
         "samples": rows}, indent=2
    ))

    print(f"\n{'#' * 70}\nSUMMARY\n{'#' * 70}")
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
