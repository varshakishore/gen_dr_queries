#!/usr/bin/env python3
"""
Post-hoc criterion verification: filter the questions a run harvested.

The same check as `research_pipeline.py --verify-criterion`, run AFTER generation instead
of during it. For each harvested question it retrieves S2 papers and asks Claude whether
the verification criterion is itself factually correct, then labels the question:

    correct              -> KEEP, criterion unchanged
    partly_correct       -> KEEP, criterion replaced by the meta-judge's rewrite
                            (the original is preserved as `criterion_original`)
    incorrect            -> DROP
    insufficient_evidence-> DROP

Why post-hoc rather than inline: inline, a rejected criterion ends the seed
(`CRITERION_INVALID`) before the pipeline has found a failing question, so the seed yields
nothing and the research calls already spent on it are wasted. Run afterwards, the check
costs the same per question but is paid only on questions worth keeping, nothing is lost
mid-flight, and it is re-runnable with different thresholds against a finished run.

By default only the questions that BROKE the answering system are checked -- the deciding
attempt of each FAILED_FOUND seed, i.e. the actual harvest. `--all-questions` checks every
graded rewrite instead.

Inputs are any mix of loop dirs, round dirs, run dirs, or `sample_*.json` files; each is
searched recursively, so `runs/loop1` covers every round and both prompts.

Outputs (next to `--out`):
    <out>            every question checked, with label, reasoning, retrieval meta, cost
    <out>.kept.json  just the survivors, criteria already swapped in -- the filtered set
    <out>.jsonl      the per-call log (retrieval + meta-judge), one record per call

Resumable: re-running skips questions already labelled in `--out`, and retries any that
errored (`--refresh` re-checks everything).

Examples:
  python verify_questions.py runs/loop1 --out runs/loop1/verified.json \\
      --concurrency 8 --verify-n-papers 25 --reranker-url http://spark-9076:8017

  # everything, not just the questions that broke the system
  python verify_questions.py runs/sqa_50_100_explore --out verified.json --all-questions

Requires ANTHROPIC_API_KEY, and S2_API_KEY (without it S2 retrieval is heavily rate
limited). Reranking needs --reranker-url or VLLM_RERANK_URL; without one it silently
degrades to no reranking, which biases the meta-judge toward accepting
"the literature does not cover X" claims.
"""

import argparse
import datetime as dt
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from anthropic import Anthropic

import research_pipeline as RP
from strategy_feedback_module import load_examples_from_runs

KEEP_LABELS = ("correct", "partly_correct")


class LockedRunLogger(RP.RunLogger):
    """RunLogger guarded by a lock: every log method funnels through _write, and the
    checks run concurrently, so without this two records can interleave mid-line."""

    def __init__(self, *a, **kw):
        self._lock = threading.Lock()   # set first: RunLogger.__init__ already logs run_start
        super().__init__(*a, **kw)

    def _write(self, record: dict) -> None:
        with self._lock:
            super()._write(record)


def collect_questions(roots, *, all_questions: bool, include_seed_round: bool) -> list:
    """The questions to check: deciding attempts (default: only the ones that FAILED)."""
    files = []
    for root in roots:
        p = Path(root)
        files.extend([p] if p.is_file() else sorted(p.rglob("sample_*.json")))
    if not files:
        return []
    examples = load_examples_from_runs(
        files, deciding_only=not all_questions, include_seed_round=include_seed_round
    )
    if not all_questions:
        examples = [e for e in examples if e.failed]
    return examples


def check_one(ex, client, logger, args) -> dict:
    """Verify one question's criterion. Never raises: errors become label 'error'."""
    row = {
        "seed_question": ex.seed_question,
        "question": ex.updated_question,
        "strategy": ex.strategy,
        "source_run": ex.source_run,
        "round": ex.round,
        "criterion_original": ex.verification_criterion,
    }
    if not ex.verification_criterion:
        return {**row, "label": "error", "kept": False,
                "error": "no verification_criterion recorded for this attempt"}
    try:
        check, bucket = RP.verify_criterion(
            client,
            model=args.model,
            question=ex.updated_question,
            why_harder=ex.extra.get("why_harder", ""),
            criterion=ex.verification_criterion,
            logger=logger,
            seed=ex.seed_question,
            attempt=ex.round if ex.round is not None else 1,
            retrieval_kwargs={"reranker": args.reranker, "reranker_url": args.reranker_url},
            n_context_papers=args.verify_n_papers,
            max_chars_per_paper=args.verify_max_chars_per_paper,
        )
    except Exception as e:                      # one bad question must not sink the batch
        return {**row, "label": "error", "kept": False, "error": f"{type(e).__name__}: {e}"}

    label = check.correctness_label
    kept = label in KEEP_LABELS
    # partly_correct is kept only with the rewrite applied; that is what "partly" buys.
    criterion = (check.rewrite if label == "partly_correct" and check.rewrite
                 else ex.verification_criterion)
    return {
        **row,
        "label": label,
        "kept": kept,
        "criterion": criterion if kept else "",
        "criterion_rewritten": bool(kept and criterion != ex.verification_criterion),
        "main_correctness_problem": check.main_correctness_problem,
        "reasoning": check.reasoning,
        "rewrite": check.rewrite,
        "checked_claims": check.checked_claims,
        "additional_queries": check.additional_queries,
        "retrieval": check.retrieval,
        "cost": asdict(bucket),
    }


def summarize(rows: list) -> dict:
    labels = Counter(r.get("label", "error") for r in rows)
    cost = sum((r.get("cost") or {}).get("cost_usd", 0.0) for r in rows)
    kept = [r for r in rows if r.get("kept")]
    return {
        "checked": len(rows),
        "kept": len(kept),
        "dropped": len(rows) - len(kept),
        "keep_rate": round(len(kept) / len(rows), 4) if rows else 0.0,
        "criteria_rewritten": sum(1 for r in kept if r.get("criterion_rewritten")),
        "by_label": dict(sorted(labels.items())),
        "cost_usd": round(cost, 4),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\nWhy post-hoc")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("runs", nargs="+",
                   help="Loop dirs, round dirs, run dirs, or sample_*.json files "
                        "(searched recursively).")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument("--all-questions", action="store_true",
                   help="Check every graded rewrite, not just the ones that broke the "
                        "answering system.")
    p.add_argument("--include-seed-round", action="store_true",
                   help="Also check round-0 questions (ALREADY_HARD seeds, tested as-is).")
    p.add_argument("--limit", type=int, default=0, help="Check at most N questions (0 = all).")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Checks in flight (default: 5). Each takes ~1 min, most of it "
                        "waiting on retrieval.")
    p.add_argument("--budget-usd", type=float, default=0.0,
                   help="Stop once this much has been spent (0 = no limit). Checked as each "
                        "check completes, so in-flight checks still finish and the actual "
                        "spend can exceed it by up to --concurrency checks.")
    p.add_argument("--refresh", action="store_true",
                   help="Re-check questions already labelled in --out.")
    p.add_argument("--model", default="claude-sonnet-4-5")
    p.add_argument("--verify-n-papers", type=int, default=RP.VERIFY_N_PAPERS)
    p.add_argument("--verify-max-chars-per-paper", type=int,
                   default=RP.VERIFY_MAX_CHARS_PER_PAPER)
    p.add_argument("--reranker", default="auto", choices=["auto", "none", "vllm"])
    p.add_argument("--reranker-url", default=None)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = collect_questions(args.runs, all_questions=args.all_questions,
                                 include_seed_round=args.include_seed_round)
    if not examples:
        p.error(f"no questions found under {', '.join(map(str, args.runs))}")

    # Resume: keep prior rows, re-check only what is missing or errored.
    done: dict = {}
    if out_path.exists() and not args.refresh:
        prior = json.loads(out_path.read_text())
        done = {r["question"]: r for r in prior.get("questions", [])
                if r.get("label") not in (None, "error")}
        if done:
            print(f"[resume] {len(done)} question(s) already verified in {out_path}")

    todo = [e for e in examples if e.updated_question not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Verifying {len(todo)} of {len(examples)} question(s), "
          f"{args.concurrency} at a time -> {out_path}")
    if not todo:
        print("Nothing to do.")

    logger = LockedRunLogger(out_path.with_suffix(".jsonl"), run_id=out_path.stem)
    client = Anthropic()
    rows = list(done.values())
    lock = threading.Lock()

    def flush():
        """Write after every completion so a crash keeps the checks already paid for."""
        payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": args.model,
            "source_dirs": [str(r) for r in args.runs],
            "selection": {"all_questions": args.all_questions,
                          "include_seed_round": args.include_seed_round,
                          "verify_n_papers": args.verify_n_papers,
                          "verify_max_chars_per_paper": args.verify_max_chars_per_paper,
                          "reranker": args.reranker,
                          "budget_usd": args.budget_usd or None},
            "totals": summarize(rows),
            "questions": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    stopped_early = False
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex_pool:
        futures = {ex_pool.submit(check_one, e, client, logger, args): e for e in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            with lock:
                rows.append(row)
                payload = flush()
            mark = "keep" if row.get("kept") else "DROP"
            print(f"[{i:>3}/{len(todo)}] {mark:<4} {row.get('label',''):<22} "
                  f"{row['question'][:70]}", flush=True)
            spent = payload["totals"]["cost_usd"]
            if args.budget_usd and spent >= args.budget_usd and not stopped_early:
                stopped_early = True
                pending = sum(1 for f in futures if f.cancel())
                print(f"\n[budget] ${spent:.2f} spent, limit ${args.budget_usd:.2f} — "
                      f"stopping; {pending} check(s) cancelled, "
                      f"{len(todo) - i - pending} already in flight will finish",
                      file=sys.stderr, flush=True)

    payload = flush()
    if stopped_early:
        payload["totals"]["stopped_early"] = True
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    kept = [{"seed_question": r["seed_question"], "question": r["question"],
             "criterion": r["criterion"], "strategy": r["strategy"],
             "source_run": r["source_run"], "criterion_original": r["criterion_original"],
             "criterion_rewritten": r.get("criterion_rewritten", False)}
            for r in rows if r.get("kept")]
    kept_path = out_path.with_suffix(".kept.json")
    kept_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False))

    t = payload["totals"]
    print(f"\n{'#' * 70}\nVERIFICATION SUMMARY\n{'#' * 70}")
    for label, n in t["by_label"].items():
        print(f"  {label:<22} {n:>4}  ({n / t['checked']:.0%})")
    print(f"\nKept {t['kept']}/{t['checked']} ({t['keep_rate']:.0%}), "
          f"{t['criteria_rewritten']} with a rewritten criterion. ${t['cost_usd']:.4f}")
    print(f"Full report:    {out_path}")
    print(f"Filtered set:   {kept_path}")
    print(f"Call log:       {out_path.with_suffix('.jsonl')}")


if __name__ == "__main__":
    main()
