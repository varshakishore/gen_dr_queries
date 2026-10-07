#!/usr/bin/env python3
"""
Answer every benchmark question with the answering systems it was NOT generated against,
judge those answers, and store them in the sample files next to the target system's.

Input is a directory built by export_benchmark.py:

    <bench_dir>/samples/<id with / -> __>.json   systems.<name> = {answer, judgment} | null
    <bench_dir>/index.jsonl                       one row per question, verdicts per system

For every sample and every system in --systems whose slot is still null:

    1. send the final `question` to that system's server (--systems-file), as research_pipeline
       does for the target system;
    2. judge the answer against the final `verification_criterion` with research_pipeline's
       judge (same prompts, same citation rendering via cite_utils), by default with the judge
       model the target system was judged with;
    3. write {answer, judgment} into systems.<name> of the sample file (atomic rewrite).

The target system's entry is never touched, and nothing outside <bench_dir> is written.
Filled slots are skipped, so the script is resumable; a failed research or judge call leaves
the slot null (and is logged), so re-running retries it. index.jsonl's verdicts are rebuilt
from the sample files at the end.

Also written, in <bench_dir>/cross_system/:
    results.jsonl     one row per (sample, system) attempt: verdict or error, timing, cost
    calls.log.jsonl   per-call log (research + judge), as research_pipeline writes
    summary.json      verdicts by system and by target system, cost

Examples:
  python answer_cross_systems.py runs/final_loop_600_full_benchmark --dry-run
  python answer_cross_systems.py runs/final_loop_600_full_benchmark --limit 3      # smoke test
  python answer_cross_systems.py runs/final_loop_600_full_benchmark --budget-usd 30

Requires the judge model's API key and the research servers in --systems-file.
"""

import argparse
import datetime as dt
import json
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import llm_client
import research_pipeline as rp

DEFAULT_SYSTEMS_FILE = Path(__file__).parent / "systems.json"
JUDGMENT_FIELDS = ("verdict", "criterion_satisfied", "criterion_reasoning", "other_issues",
                   "summary")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(path)


def answer_one(path: Path, system: str, *, sysconf: dict, client, model: str, logger,
               semaphore, file_lock) -> dict:
    """Answer + judge one (sample, system) and store it in the sample file. Never raises."""
    sample = json.loads(path.read_text())
    out = {"id": sample["id"], "system": system, "target_system": sample["target_system"],
           "judge_model": model, "cost_usd": 0.0, "started_at": now()}
    try:
        t0 = time.time()
        with semaphore:
            res = rp.query_research_system(
                sample["question"], logger, seed=sample["id"], attempt=0,
                url=sysconf["server_url"],
                timeout_s=sysconf.get("timeout", rp.RESEARCH_TIMEOUT_S))
        out["answer_seconds"] = round(time.time() - t0, 1)
        answer = {"text": res["answer"] or "", "trace": res["trace"], "model": res.get("model"),
                  "usage": res.get("usage"), "answer_source": "cross_system",
                  "answered_at": now()}
        judgment, bucket = rp.judge_answer(
            client, model, sample["question"], sample["verification_criterion"],
            answer["text"], logger, seed=sample["id"], attempt=0, trace=answer["trace"])
        out["cost_usd"] = bucket.cost_usd
        judgment = {**{k: getattr(judgment, k) for k in JUDGMENT_FIELDS},
                    "judge_model": model, "judgment_source": "cross_system",
                    "judged_at": now()}
        # re-read under the lock: another system may have written this file meanwhile
        with file_lock:
            sample = json.loads(path.read_text())
            if sample["systems"].get(system) is not None:
                raise RuntimeError(f"slot {system} was filled while this call ran")
            sample["systems"][system] = {"answer": answer, "judgment": judgment}
            write_json_atomic(path, sample)
        out["verdict"] = judgment["verdict"]
        out["criterion_satisfied"] = judgment["criterion_satisfied"]
    except Exception as e:
        out["verdict"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
    out["finished_at"] = now()
    return out


def rebuild_index(bench_dir: Path) -> list:
    """Refresh the per-system verdicts of index.jsonl from the sample files."""
    index_path = bench_dir / "index.jsonl"
    rows = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
    for row in rows:
        sample = json.loads((bench_dir / row["file"]).read_text())
        row["verdicts"] = {k: (v["judgment"]["verdict"] if v else None)
                           for k, v in sample["systems"].items()}
    tmp = index_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(index_path)
    return rows


def summarize(rows: list, systems: list) -> dict:
    by_sys = {s: dict(Counter(r["verdicts"][s] for r in rows if r["target_system"] != s))
              for s in systems}
    cross = defaultdict(dict)                     # target -> other system -> verdict counts
    for t in sorted({r["target_system"] for r in rows}):
        for s in systems:
            if s != t:
                cross[t][s] = dict(Counter(r["verdicts"][s] for r in rows
                                           if r["target_system"] == t))
    all_fail = sum(1 for r in rows if all(v == "FAILED" for v in r["verdicts"].values()))
    complete = sum(1 for r in rows if all(v is not None for v in r["verdicts"].values()))
    return {"questions": len(rows), "complete": complete, "failed_by_all_systems": all_fail,
            "verdicts_by_system": by_sys, "verdicts_by_target_system": dict(cross)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\nAlso written")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_dir", type=Path, help="Directory built by export_benchmark.py.")
    ap.add_argument("--systems", nargs="+", default=None,
                    help="Systems to fill (default: every system in the sample files).")
    ap.add_argument("--systems-file", type=Path, default=DEFAULT_SYSTEMS_FILE,
                    help="JSON list of {name, server_url, timeout, concurrency} "
                         f"(default: {DEFAULT_SYSTEMS_FILE.name}).")
    ap.add_argument("--model", default=None,
                    help="Judge model (default: the one the target systems were judged with).")
    ap.add_argument("--limit", type=int, default=0,
                    help="At most N (sample, system) pairs (0 = all).")
    ap.add_argument("--budget-usd", type=float, default=0.0,
                    help="Stop submitting once judge spend reaches this (0 = no limit). "
                         "Research-server calls are not priced.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be answered and exit.")
    llm_client.add_provider_arg(ap)
    args = ap.parse_args()
    llm_client.configure_from_args(args)

    bench = args.bench_dir
    if not (bench / "index.jsonl").exists() or not (bench / "samples").is_dir():
        ap.error(f"{bench} is not an export_benchmark.py directory (no index.jsonl / samples/)")
    index = [json.loads(l) for l in (bench / "index.jsonl").read_text().splitlines() if l.strip()]
    all_systems = sorted({s for r in index for s in r["verdicts"]})
    wanted = args.systems or all_systems
    if (unknown := set(wanted) - set(all_systems)):
        ap.error(f"unknown system(s) {sorted(unknown)}; the samples have {all_systems}")

    # the work list comes from the sample files themselves, not the (possibly stale) index
    todo, judge_models = [], Counter()
    for r in index:
        path = bench / r["file"]
        sample = json.loads(path.read_text())
        tgt = sample["systems"][sample["target_system"]]
        judge_models[tgt["judgment"].get("judge_model")] += 1
        for s in wanted:
            if s != sample["target_system"] and sample["systems"].get(s) is None:
                todo.append((path, s))
    if args.limit:
        todo = todo[:args.limit]

    model = args.model or judge_models.most_common(1)[0][0] or rp.DEFAULT_MODEL
    if len(judge_models) > 1 and not args.model:
        print(f"[warn] target judgments used several judge models {dict(judge_models)}; "
              f"using {model}", file=sys.stderr)
    by_sys = Counter(s for _, s in todo)
    print(f"{len(index)} questions | systems {wanted} | to answer {len(todo)} "
          f"{dict(by_sys)} | judge {model}")
    if args.dry_run or not todo:
        return

    if (missing := llm_client.require_api_key(model)):
        ap.error(missing)
    systems = {s["name"]: s for s in json.loads(args.systems_file.read_text())}
    for name in by_sys:
        if name not in systems:
            ap.error(f"system {name!r} has no entry in {args.systems_file}")
        if (err := rp.check_server_reachable(systems[name]["server_url"])):
            ap.error(f"{name} server unreachable: {err}")

    out_dir = bench / "cross_system"
    out_dir.mkdir(exist_ok=True)
    logger = rp.RunLogger(out_dir / "calls.log.jsonl", f"cross-{uuid.uuid4().hex[:8]}")
    client = llm_client.make_client(model)
    semaphores = {n: threading.Semaphore(max(1, int(systems[n].get("concurrency", 1))))
                  for n in by_sys}
    file_locks = defaultdict(threading.Lock)
    workers = sum(int(systems[n].get("concurrency", 1)) for n in by_sys)

    spent, stopped, counts = 0.0, False, Counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, \
            (out_dir / "results.jsonl").open("a") as log:
        futures = {pool.submit(answer_one, path, s, sysconf=systems[s], client=client,
                               model=model, logger=logger, semaphore=semaphores[s],
                               file_lock=file_locks[path]): (path, s)
                   for path, s in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            log.write(json.dumps(res, ensure_ascii=False) + "\n")
            log.flush()
            spent += res.get("cost_usd", 0.0)
            counts[(res["system"], res["verdict"])] += 1
            print(f"[{i:>4}/{len(todo)}] {res['verdict']:<7} {res['system']:<10} "
                  f"(target {res['target_system']:<10}) {res.get('answer_seconds', '-'):>7}s "
                  f"${spent:.2f}  {res['id']}"
                  + (f"  {res['error'][:80]}" if res.get("error") else ""), flush=True)
            if args.budget_usd and spent >= args.budget_usd and not stopped:
                stopped = True
                cancelled = sum(1 for f in futures if f.cancel())
                print(f"\n[budget] ${spent:.2f} spent — {cancelled} pair(s) cancelled",
                      file=sys.stderr, flush=True)

    rows = rebuild_index(bench)
    summary = {"updated_at": now(), "judge_model": model, "systems": wanted,
               "this_run": {"attempted": sum(counts.values()), "stopped_early": stopped,
                            "judge_cost_usd": round(spent, 4),
                            "by_system_verdict": {f"{s}:{v}": n for (s, v), n in counts.items()}},
               **summarize(rows, all_systems)}
    prev = out_dir / "summary.json"
    if prev.exists():
        summary["judge_cost_usd_total"] = round(
            json.loads(prev.read_text()).get("judge_cost_usd_total", 0.0) + spent, 4)
    else:
        summary["judge_cost_usd_total"] = round(spent, 4)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    errors = sum(n for (s, v), n in counts.items() if v == "error")
    print(f"\n{'#' * 70}\nCROSS-SYSTEM SUMMARY\n{'#' * 70}")
    print(f"  this run: {dict(counts)}  |  judge ${spent:.2f}  |  errors {errors}"
          + ("  (re-run to retry)" if errors else ""))
    print(f"  complete questions {summary['complete']}/{summary['questions']}  |  "
          f"failed by all systems {summary['failed_by_all_systems']}")
    for s, v in summary["verdicts_by_system"].items():
        print(f"  {s:<10} on others' questions: {v}")
    print(f"\nindex:   {bench / 'index.jsonl'}\nsummary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
