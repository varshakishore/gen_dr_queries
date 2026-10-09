#!/usr/bin/env python3
"""
Re-answer and re-judge the `revise` rows of a final_verification.py run, so a rewritten
question / VC carries a FAILED label it actually earned.

final_verification.py never applies its rewrites: the original FAILED verdict was earned
by the original question against the original VC, and says nothing about the rewritten
pair. This script earns the label again, per revise row:

    criterion-only rewrite   question unchanged, so the stored answer still answers it:
                             re-judge that answer (from the run's sample file) against
                             the rewritten VC. No research call.
    question rewrite         the stored answer is to a different question: send the
                             rewritten question to the SAME answering system that failed
                             it (server from --systems-file), then judge the new answer
                             against the rewritten VC (or the original VC if only the
                             question changed).
    missing rewrite          a MINOR_REVISION came back without its rewrite: skipped,
                             it needs a human edit.

Judging reuses research_pipeline's judge (same prompts, same citation rendering via
cite_utils) and, by default, the judge model the run itself used.

Outcome per row, mirroring final_verification's criterion_satisfied skip rule:

    still_fails           FAILED and the VC is unmet      -> keep, with the rewrites
    now_passes            PASSED                          -> drop: no longer breaks it
    failed_other_issues   FAILED although the VC is met   -> drop
    error                 a research or judge call failed -> retried on the next run

Outputs, in <verification_dir>/rejudge/ (or --out-dir). The run dir is only read; an output
dir inside it is refused.
    results.jsonl        one row per revise row: rewrites, new judgment, outcome
    still_failing.jsonl  the rewritten pairs that still break their system
    benchmark.jsonl      the verification's pass rows (as written) + still_failing rows
    summary.json         counts by kind and outcome, judge cost
    answers/<id>.json    full new answer + trace for each re-answered question
    rejudge.log.jsonl    per-call log (research + judge), as research_pipeline writes

Resumable: rows already in results.jsonl are skipped, errored rows are retried.

Examples:
  python final_verification_rejudge.py ../annotation_app/studies/<run>_final_verification_v14 \\
      --run-dir runs/final_loop_600 --dry-run
  python final_verification_rejudge.py runs/final_loop_600_final_verification --budget-usd 20

Requires the judge model's API key, and the research servers in --systems-file for any
question rewrite.
"""

import argparse
import datetime as dt
import json
import sys
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import llm_client
import research_pipeline as rp

DEFAULT_SYSTEMS_FILE = Path(__file__).parent / "systems.json"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue                     # a line cut off by a kill mid-write
    return rows


def deciding_attempt(run_dir: Path, item_id: str) -> dict:
    """The attempt final_verification reviewed: the last one of the sample file."""
    data = json.loads((run_dir / f"{item_id}.json").read_text())
    return data["results"][0]["attempts"][-1], data.get("model")


def plan(row: dict) -> dict:
    """What to do with one revise row: the pair to judge and how to get the answer."""
    new_q = (row.get("proposed_question") or "").strip()
    new_vc = (row.get("proposed_vc") or "").strip()
    if row.get("missing_rewrite"):
        kind = "missing_rewrite"
    elif new_q:
        kind = "question_rewrite"
    elif new_vc:
        kind = "criterion_only"
    else:
        kind = "missing_rewrite"         # revise with no rewrite recorded at all
    return {"kind": kind,
            "question": new_q or row["question"],
            "criterion": new_vc or row["verification_criterion"],
            "question_changed": bool(new_q), "criterion_changed": bool(new_vc)}


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


def outcome_of(judgment: dict) -> str:
    if judgment["verdict"] == "PASSED":
        return "now_passes"
    return "failed_other_issues" if judgment["criterion_satisfied"] else "still_fails"


def rejudge_one(row: dict, *, run_dir: Path, systems: dict, client, model: str,
                logger, semaphores: dict, answers_dir: Path) -> dict:
    """Re-answer (if needed) and re-judge one revise row. Never raises."""
    p = plan(row)
    out = {"id": row["id"], "system": row.get("system"), "prompt": row.get("prompt"),
           "round": row.get("round"), "seed": row.get("seed"), "kind": p["kind"],
           "original_question": row["question"],
           "original_criterion": row["verification_criterion"],
           "question": p["question"], "criterion": p["criterion"],
           "question_changed": p["question_changed"],
           "criterion_changed": p["criterion_changed"],
           "judge_model": model, "cost_usd": 0.0,
           "rejudged_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if p["kind"] == "missing_rewrite":
        out["outcome"] = "skipped_missing_rewrite"
        return out
    try:
        attempt, _ = deciding_attempt(run_dir, row["id"])
        if p["kind"] == "criterion_only":
            # The stored answer still answers the (unchanged) question.
            if attempt["harder"]["updated_question"].strip() != row["question"].strip():
                raise ValueError("sample file's question differs from the verified one")
            answer, trace = attempt.get("answer") or "", attempt.get("trace")
            out["answer_source"] = "stored"
        else:
            sysconf = systems.get(row.get("system"))
            if not sysconf:
                raise ValueError(f"no server configured for system {row.get('system')!r}")
            with semaphores[row["system"]]:
                res = rp.query_research_system(
                    p["question"], logger, seed=row["id"], attempt=0,
                    url=sysconf["server_url"],
                    timeout_s=sysconf.get("timeout", rp.RESEARCH_TIMEOUT_S))
            answer, trace = res["answer"], res["trace"]
            out["answer_source"] = "re-answered"
            out["answer_model"] = res.get("model")
            path = answers_dir / f"{row['id'].replace('/', '__')}.json"
            path.write_text(json.dumps({"id": row["id"], "question": p["question"],
                                        "answer": answer, "trace": trace,
                                        "model": res.get("model"),
                                        "usage": res.get("usage")}, ensure_ascii=False))
            out["answer_file"] = str(path)
        judgment, bucket = rp.judge_answer(
            client, model, p["question"], p["criterion"], answer, logger,
            seed=row["id"], attempt=0, trace=trace)
        out["cost_usd"] = bucket.cost_usd
        out["judgment"] = {k: getattr(judgment, k) for k in
                           ("verdict", "criterion_satisfied", "criterion_reasoning",
                            "other_issues", "summary")}
        out["outcome"] = outcome_of(out["judgment"])
    except Exception as e:
        out["outcome"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\nOutcome per row")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verification_dir", type=Path,
                    help="A final_verification.py --out-dir (has results.jsonl, summary.json).")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Where the sample files live (default: run_dir in summary.json).")
    ap.add_argument("--systems-file", type=Path, default=DEFAULT_SYSTEMS_FILE,
                    help="JSON list of {name, server_url, timeout, concurrency} "
                         f"(default: {DEFAULT_SYSTEMS_FILE.name}).")
    ap.add_argument("--model", default=None,
                    help="Judge model (default: the model the run used, from its sample files).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <verification_dir>/rejudge.")
    ap.add_argument("--only", choices=["all", "criterion_only", "question_rewrite"],
                    default="all", help="Restrict to one kind of revise row.")
    ap.add_argument("--judge-concurrency", type=int, default=8,
                    help="Parallel criterion-only re-judges (no research call).")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all).")
    ap.add_argument("--budget-usd", type=float, default=0.0,
                    help="Stop submitting once judge spend reaches this (0 = no limit). "
                         "Research-server calls are not priced.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be re-answered / re-judged and exit.")
    llm_client.add_provider_arg(ap)
    args = ap.parse_args()
    llm_client.configure_from_args(args)

    vdir = args.verification_dir
    rows = load_jsonl(vdir / "results.jsonl")
    if not rows:
        ap.error(f"no results.jsonl rows in {vdir}")
    summary_in = json.loads((vdir / "summary.json").read_text()) \
        if (vdir / "summary.json").exists() else {}
    run_dir = args.run_dir or Path(summary_in.get("run_dir", ""))
    if not run_dir or not run_dir.exists():
        ap.error(f"run dir not found ({run_dir}); pass --run-dir")
    out_dir = args.out_dir or vdir / "rejudge"
    # the run dir (sample files) is read-only input: never write into it
    if out_dir.resolve().is_relative_to(run_dir.resolve()):
        ap.error(f"output dir {out_dir} is inside the run dir {run_dir}; pass --out-dir "
                 f"somewhere else (the verification dir itself is inside the run dir)")
    systems = {s["name"]: s for s in json.loads(args.systems_file.read_text())}

    revise = [r for r in rows if r.get("decision") == "revise"]
    plans = {r["id"]: plan(r) for r in revise}
    prior = {r["id"]: r for r in load_jsonl(out_dir / "results.jsonl")}
    done = {k for k, r in prior.items() if r.get("outcome") != "error"}
    todo = [r for r in revise if r["id"] not in done
            and (args.only == "all" or plans[r["id"]]["kind"] == args.only)]
    if args.limit:
        todo = todo[:args.limit]

    # judge model: the run's own unless overridden
    model = args.model
    if not model:
        _, model = deciding_attempt(run_dir, revise[0]["id"]) if revise else (None, None)
        model = model or rp.DEFAULT_MODEL
    kinds = Counter(plans[r["id"]]["kind"] for r in todo)
    by_sys = Counter(r.get("system") for r in todo if plans[r["id"]]["kind"] == "question_rewrite")
    print(f"revise rows {len(revise)} | already done {len(done)} | to process {len(todo)} "
          f"{dict(kinds)} | re-answers by system {dict(by_sys)} | judge {model} -> {out_dir}/")
    if args.dry_run or not todo:
        return
    if (missing := llm_client.require_api_key(model)):
        ap.error(missing)
    for name in by_sys:
        if name not in systems:
            ap.error(f"system {name!r} has question rewrites but no entry in {args.systems_file}")
        if (err := rp.check_server_reachable(systems[name]["server_url"])):
            ap.error(f"{name} server unreachable: {err}")

    out_dir.mkdir(parents=True, exist_ok=True)
    answers_dir = out_dir / "answers"
    answers_dir.mkdir(exist_ok=True)
    logger = rp.RunLogger(out_dir / "rejudge.log.jsonl", f"rejudge-{uuid.uuid4().hex[:8]}")
    client = llm_client.make_client(model)
    semaphores = {n: threading.Semaphore(max(1, int(s.get("concurrency", 1))))
                  for n, s in systems.items()}
    workers = args.judge_concurrency + sum(int(s.get("concurrency", 1)) for s in systems.values())

    results = dict(prior)
    spent, stopped, lock = 0.0, False, threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, \
            (out_dir / "results.jsonl").open("a") as log:
        futures = {pool.submit(rejudge_one, r, run_dir=run_dir, systems=systems,
                               client=client, model=model, logger=logger,
                               semaphores=semaphores, answers_dir=answers_dir): r
                   for r in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            with lock:
                log.write(json.dumps(res, ensure_ascii=False) + "\n")   # crash-safe
                log.flush()
                results[res["id"]] = res
                spent += res.get("cost_usd", 0.0)
            print(f"[{i:>3}/{len(todo)}] {res['outcome']:<22} {res['kind']:<17} "
                  f"{res.get('system') or '-':<10} ${spent:.2f}  {res['question'][:55]}",
                  flush=True)
            if args.budget_usd and spent >= args.budget_usd and not stopped:
                stopped = True
                cancelled = sum(1 for f in futures if f.cancel())
                print(f"\n[budget] ${spent:.2f} spent — {cancelled} row(s) cancelled",
                      file=sys.stderr, flush=True)

    order = {r["id"]: n for n, r in enumerate(rows)}
    final = sorted(results.values(), key=lambda r: order.get(r["id"], 1 << 30))
    write_jsonl(out_dir / "results.jsonl", final)
    still = [r for r in final if r.get("outcome") == "still_fails"]
    keep_cols = ("id", "round", "system", "prompt", "seed")
    write_jsonl(out_dir / "still_failing.jsonl", [
        {**{k: r.get(k) for k in keep_cols}, "question": r["question"],
         "verification_criterion": r["criterion"], "source": "rewritten",
         "rewrite_kind": r["kind"]} for r in still])
    passes = [r for r in rows if r.get("decision") == "pass"]
    write_jsonl(out_dir / "benchmark.jsonl", [
        {**{k: r.get(k) for k in keep_cols}, "question": r["question"],
         "verification_criterion": r["verification_criterion"], "source": "passed_as_written"}
        for r in passes] + [
        {**{k: r.get(k) for k in keep_cols}, "question": r["question"],
         "verification_criterion": r["criterion"], "source": "rewritten",
         "rewrite_kind": r["kind"]} for r in still])
    summary = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "verification_dir": str(vdir), "run_dir": str(run_dir), "judge_model": model,
               "reasoning_effort": args.reasoning_effort, "stopped_early": stopped,
               "revise_rows": len(revise), "processed": len(final),
               "by_kind": dict(Counter(r["kind"] for r in final)),
               "outcome": dict(Counter(r["outcome"] for r in final)),
               "outcome_by_kind": {k: dict(Counter(r["outcome"] for r in final if r["kind"] == k))
                                   for k in sorted({r["kind"] for r in final})},
               "benchmark_size": len(passes) + len(still),
               "judge_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in final), 4)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'#' * 70}\nREJUDGE SUMMARY\n{'#' * 70}")
    for k in ("by_kind", "outcome", "outcome_by_kind"):
        print(f"  {k:<16} {summary[k]}")
    print(f"  benchmark: {len(passes)} passed as written + {len(still)} rewritten & still "
          f"failing = {summary['benchmark_size']}  |  judge cost ${summary['judge_cost_usd']:.2f}")
    print(f"\nresults:   {out_dir / 'results.jsonl'}\nbenchmark: {out_dir / 'benchmark.jsonl'}")


if __name__ == "__main__":
    main()
