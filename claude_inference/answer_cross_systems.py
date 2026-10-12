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

--add-systems NAME ... answers with systems that are not in the samples yet -- e.g. a
variant of an existing system (`tongyi_s2`: Tongyi with its Serper web search restricted
to arxiv.org and semanticscholar.org) -- on EVERY question, the target's included; its slot is created on write. Each
answer records the server it came from and the systems-file entry's `search_scope`.

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
  python answer_cross_systems.py runs/final_loop_600_full_benchmark \\
      --add-systems tongyi_s2 webthinker_s2 --systems-file systems_s2.json --budget-usd 40

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
               semaphore, file_lock, cache_dir: Path) -> dict:
    """Answer + judge one (sample, system) and store it in the sample file. Never raises.

    The answer is cached in cache_dir before it is judged and the cache is removed once the
    sample is written, so a judge failure (or a crash) between the two does not throw away
    a paid-for answer: the next run judges the cached answer instead of asking again."""
    sample = json.loads(path.read_text())
    out = {"id": sample["id"], "system": system, "target_system": sample["target_system"],
           "judge_model": model, "cost_usd": 0.0, "started_at": now()}
    cache = cache_dir / system / f"{sample['id'].replace('/', '__')}.json"
    try:
        t0 = time.time()
        if cache.exists():                              # answered before, judging failed
            res = json.loads(cache.read_text())
            out["answer_reused"] = True
            out["answer_cost_usd"] = 0.0                # paid in the run that produced it
        else:
            with semaphore:
                if sysconf.get("kind") == "anthropic":  # Claude + server-side web search
                    import frontier_search
                    res = frontier_search.query_claude(sample["question"], sysconf)
                else:                                   # a research server's /ask
                    res = rp.query_research_system(
                        sample["question"], logger, seed=sample["id"], attempt=0,
                        url=sysconf["server_url"],
                        timeout_s=sysconf.get("timeout", rp.RESEARCH_TIMEOUT_S))
            out["answer_cost_usd"] = res.get("cost_usd", 0.0)
            cache.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(cache, res)
        out["answer_seconds"] = round(time.time() - t0, 1)
        if isinstance(res.get("trace"), dict):
            for key, field in (("complete", "answer_complete"),
                               ("cited_text_missing", "cited_text_missing")):
                if key in res["trace"]:
                    out[field] = res["trace"][key]     # partial report / blank cited text
        answer = {"text": res["answer"] or "", "trace": res["trace"], "model": res.get("model"),
                  "usage": res.get("usage"), "answer_source": "cross_system",
                  "answered_at": now(), "server_url": sysconf.get("server_url"),
                  "cost_usd": res.get("cost_usd"),
                  "search_scope": sysconf.get("search_scope")}
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
        cache.unlink(missing_ok=True)                   # stored with its judgment now
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
    # .get: a system added with --add-systems has no key in samples it has not reached yet
    v = lambda r, s: r["verdicts"].get(s)
    by_sys = {s: dict(Counter(v(r, s) for r in rows if r["target_system"] != s))
              for s in systems}
    cross = defaultdict(dict)                     # target -> other system -> verdict counts
    for t in sorted({r["target_system"] for r in rows}):
        for s in systems:
            if s != t:
                cross[t][s] = dict(Counter(v(r, s) for r in rows if r["target_system"] == t))
    # A system run only on a subset (--ids-file) would make every question outside it
    # "incomplete" and "not failed by all", so those two counts use the systems that answered
    # at least half the questions; the others are listed under partial_systems.
    coverage = {s: sum(v(r, s) is not None for r in rows) for s in systems}
    full = [s for s in systems if coverage[s] >= len(rows) / 2]
    all_fail = sum(1 for r in rows if full and all(v(r, s) == "FAILED" for s in full))
    complete = sum(1 for r in rows if full and all(v(r, s) is not None for s in full))
    return {"questions": len(rows), "complete": complete, "failed_by_all_systems": all_fail,
            "counted_systems": full,
            "partial_systems": {s: coverage[s] for s in systems if s not in full},
            "verdicts_by_system": by_sys, "verdicts_by_target_system": dict(cross)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\nAlso written")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_dir", type=Path, help="Directory built by export_benchmark.py.")
    ap.add_argument("--systems", nargs="+", default=None,
                    help="Systems to fill (default: every system in the sample files).")
    ap.add_argument("--add-systems", nargs="+", default=[],
                    help="New systems (not yet in the samples) to answer EVERY question with; "
                         "each needs an entry in --systems-file.")
    ap.add_argument("--systems-file", type=Path, default=DEFAULT_SYSTEMS_FILE,
                    help="JSON list of {name, server_url, timeout, concurrency} "
                         f"(default: {DEFAULT_SYSTEMS_FILE.name}).")
    ap.add_argument("--model", default=None,
                    help="Judge model (default: the one the target systems were judged with).")
    ap.add_argument("--ids-file", type=Path, default=None,
                    help="Only these questions: a make_subset.py file ({..., \"ids\": [...]}) "
                         "or a JSON list of ids. summary.json also gets results on the subset.")
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
    existing = sorted({s for r in index for s in r["verdicts"]})
    all_systems = existing + [s for s in args.add_systems if s not in existing]
    if args.add_systems:
        wanted = list(dict.fromkeys((args.systems or []) + args.add_systems))
    else:
        wanted = args.systems or existing
    if (unknown := set(wanted) - set(all_systems)):
        ap.error(f"unknown system(s) {sorted(unknown)}; the samples have {existing} "
                 f"(use --add-systems for a new one)")

    subset = None
    if args.ids_file:
        loaded = json.loads(args.ids_file.read_text())
        subset = set(loaded["ids"] if isinstance(loaded, dict) else loaded)
        if (missing := subset - {r["id"] for r in index}):
            ap.error(f"{len(missing)} id(s) in {args.ids_file} are not in this benchmark, "
                     f"e.g. {sorted(missing)[:3]}")
        print(f"subset: {len(subset)} questions from {args.ids_file}")

    # the work list comes from the sample files themselves, not the (possibly stale) index
    todo, judge_models = [], Counter()
    for r in index:
        if subset is not None and r["id"] not in subset:
            continue
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
        conf = systems[name]
        if conf.get("kind") == "anthropic":
            import frontier_search
            if (missing := llm_client.require_api_key(conf["model"])):
                ap.error(f"{name}: {missing}")
            if args.budget_usd and not frontier_search.is_priced(conf):
                ap.error(f"{name}: no price for {conf['model']} in llm_client.MODEL_PRICING; add "
                         f"price_input / price_output to its systems-file entry, or --budget-usd "
                         f"cannot count its spend")
        elif (err := rp.check_server_reachable(conf["server_url"])):
            ap.error(f"{name} server unreachable: {err}")

    out_dir = bench / "cross_system"
    out_dir.mkdir(exist_ok=True)
    logger = rp.RunLogger(out_dir / "calls.log.jsonl", f"cross-{uuid.uuid4().hex[:8]}")
    client = llm_client.make_client(model)
    semaphores = {n: threading.Semaphore(max(1, int(systems[n].get("concurrency", 1))))
                  for n in by_sys}
    file_locks = defaultdict(threading.Lock)
    workers = sum(int(systems[n].get("concurrency", 1)) for n in by_sys)

    spent, spent_answer, stopped, interrupted, counts = 0.0, 0.0, False, False, Counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, \
            (out_dir / "results.jsonl").open("a") as log:
        futures = {pool.submit(answer_one, path, s, sysconf=systems[s], client=client,
                               model=model, logger=logger, semaphore=semaphores[s],
                               file_lock=file_locks[path], cache_dir=out_dir / "answer_cache"):
                       (path, s)
                   for path, s in todo}
        logged = set()

        def record(fut):
            nonlocal spent, spent_answer, stopped
            logged.add(fut)
            res = fut.result()
            log.write(json.dumps(res, ensure_ascii=False) + "\n")
            log.flush()
            spent += res.get("cost_usd", 0.0) + res.get("answer_cost_usd", 0.0)
            spent_answer += res.get("answer_cost_usd", 0.0)
            counts[(res["system"], res["verdict"])] += 1
            flags = "".join(f" {tag}" for tag, on in (
                ("INCOMPLETE", res.get("answer_complete") is False),
                ("CITED-TEXT-MISSING", res.get("cited_text_missing")),
                ("(cached answer)", res.get("answer_reused"))) if on)
            print(f"[{len(logged):>4}/{len(todo)}] {res['verdict']:<7} {res['system']:<10} "
                  f"(target {res['target_system']:<10}) {res.get('answer_seconds', '-'):>7}s "
                  f"${spent:.2f}  {res['id']}{flags}"
                  + (f"  {res['error'][:80]}" if res.get("error") else ""), flush=True)
            if args.budget_usd and spent >= args.budget_usd and not stopped:
                stopped = True
                cancelled = sum(1 for f in futures if f.cancel())
                print(f"\n[budget] ${spent:.2f} spent — {cancelled} pair(s) cancelled",
                      file=sys.stderr, flush=True)

        try:
            for fut in as_completed(futures):
                record(fut)
        except KeyboardInterrupt:
            # Without this the pool would still run every queued pair on its way out --
            # Ctrl-C would silently spend the rest of the budget. Cancel what has not
            # started; let the few in progress finish (they are paid for) and log them.
            stopped = interrupted = True
            cancelled = sum(1 for f in futures if f.cancel())
            running = [f for f in futures if not f.cancelled() and f not in logged]
            print(f"\n[interrupt] {cancelled} queued pair(s) cancelled; waiting for the "
                  f"{len(running)} in progress to finish and be saved (Ctrl-C again to abandon "
                  f"them; their answers stay in the answer cache)", file=sys.stderr, flush=True)
            for fut in as_completed(running):
                record(fut)

    rows = rebuild_index(bench)
    summary = {"updated_at": now(), "judge_model": model, "systems": wanted,
               "this_run": {"attempted": sum(counts.values()), "stopped_early": stopped,
                            "interrupted": interrupted,
                            "judge_cost_usd": round(spent - spent_answer, 4),
                            "answer_cost_usd": round(spent_answer, 4),
                            "by_system_verdict": {f"{s}:{v}": n for (s, v), n in counts.items()}},
               **summarize(rows, all_systems)}
    prev = out_dir / "summary.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    summary["judge_cost_usd_total"] = round(
        old.get("judge_cost_usd_total", 0.0) + spent - spent_answer, 4)
    summary["answer_cost_usd_total"] = round(old.get("answer_cost_usd_total", 0.0) + spent_answer, 4)
    summary["subsets"] = dict(old.get("subsets", {}))     # results on each subset run so far
    if subset is not None:
        summary["subsets"][args.ids_file.stem] = {
            "ids_file": str(args.ids_file), "updated_at": now(),
            **summarize([r for r in rows if r["id"] in subset], all_systems)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    errors = sum(n for (s, v), n in counts.items() if v == "error")
    print(f"\n{'#' * 70}\nCROSS-SYSTEM SUMMARY\n{'#' * 70}")
    print(f"  this run: {dict(counts)}  |  judge ${spent - spent_answer:.2f}"
          + (f" + answering ${spent_answer:.2f}" if spent_answer else "") + f"  |  errors {errors}"
          + ("  (re-run to retry)" if errors else ""))
    print(f"  complete questions {summary['complete']}/{summary['questions']}  |  "
          f"failed by all systems {summary['failed_by_all_systems']}")
    if summary["partial_systems"]:
        print(f"  (counted over {summary['counted_systems']}; partly answered, not counted: "
              f"{summary['partial_systems']})")
    for s, v in summary["verdicts_by_system"].items():
        print(f"  {s:<10} on others' questions: {v}")
    if subset is not None:
        sub = summary["subsets"][args.ids_file.stem]
        print(f"\n  on subset {args.ids_file.stem} ({sub['questions']} questions): complete "
              f"{sub['complete']} | failed by all {sub['failed_by_all_systems']} "
              f"(over {sub['counted_systems']})")
        for s, v in sub["verdicts_by_system"].items():
            print(f"  {s:<10} on others' questions: {v}")
    print(f"\nindex:   {bench / 'index.jsonl'}\nsummary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
