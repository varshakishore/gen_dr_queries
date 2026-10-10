#!/usr/bin/env python3
"""
Assemble the benchmark into one directory, one JSON file per question.

Reads (never writes) the generation run and its final_verification + rejudge outputs, and
writes a new directory:

    <out_dir>/
    ├── README.md        what this is, how it was built, field definitions, counts
    ├── index.jsonl      one row per question: metadata, final question/VC, verdict per system
    └── samples/<round>__<system>__<prompt>__sample_NNN.json   one file per question

The benchmark is rejudge/benchmark.jsonl: every question that passed the filter as written,
plus every rewritten question that still fails after re-answering / re-judging. Each sample
file keeps the answer and judgment under `systems.<name>`; the system the question was
generated against (`target_system`) is filled in, the other answering systems are null, to
be filled by a later cross-system step in the same shape.

Where the target system's answer and judgment come from:

    passed_as_written           answer + judgment from the run's sample file (deciding attempt)
    rewritten / criterion_only  answer from the run's sample file, judgment from the rejudge
    rewritten / question_rewrite  answer + judgment from the rejudge (re-answered question)

Answers are stored raw, with their trace: the citations (DR-Tulu's <cite id> tags; the
sources Tongyi and WebThinker consulted) are resolved from the trace by cite_utils when
needed, exactly as the judge saw them.

The source directories are only read. The output is built in a temporary directory and
renamed into place at the end; an existing --out-dir is refused, as is one inside either
source directory.

  python export_benchmark.py                 # defaults below
  python export_benchmark.py --dry-run       # build in memory, report, write nothing
"""

import argparse
import datetime as dt
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

SYSTEMS = ("drtulu", "tongyi", "webthinker")
HERE = Path(__file__).parent


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def strip_usage(review):
    """A filter review without its token-usage bookkeeping."""
    if not review:
        return review
    return {k: v for k, v in review.items() if k not in ("usage", "cost_usd")}


def file_name(item_id: str) -> str:
    return item_id.replace("/", "__") + ".json"


def build_sample(b: dict, run_dir: Path, vdir: Path, filt: dict, rej: dict, bins: dict,
                 systems: tuple = SYSTEMS) -> dict:
    """One benchmark question as a self-contained record."""
    item_id = b["id"]
    sample_path = run_dir / f"{item_id}.json"
    data = json.loads(sample_path.read_text())
    res = data["results"][0]
    att = res["attempts"][-1]
    harder = att.get("harder") or {}
    f = filt[item_id]
    r = rej.get(item_id)
    target = b["system"]

    # --- the target system's answer and judgment, from wherever its label was earned
    if b["source"] == "passed_as_written":
        answer = {"text": att.get("answer") or "", "trace": att.get("trace"),
                  "model": att.get("answer_model"), "answer_source": "original_run",
                  "answered_at": None}
        j = att.get("judgment") or {}
        judgment = {k: j.get(k) for k in ("verdict", "criterion_satisfied",
                                          "criterion_reasoning", "other_issues", "summary")}
        judgment.update(judge_model=data.get("model"), judgment_source="original_run",
                        judged_at=None)
    else:
        if r is None or r.get("outcome") != "still_fails":
            raise ValueError(f"{item_id}: rewritten item without a still_fails rejudge row")
        if r["kind"] == "question_rewrite":
            a = json.loads((vdir / "rejudge" / "answers" / file_name(item_id)).read_text())
            answer = {"text": a.get("answer") or "", "trace": a.get("trace"),
                      "model": a.get("model"), "answer_source": "re_answered",
                      "answered_at": r.get("rejudged_at")}
        else:                                           # criterion_only: stored answer
            answer = {"text": att.get("answer") or "", "trace": att.get("trace"),
                      "model": att.get("answer_model"), "answer_source": "original_run",
                      "answered_at": None}
        judgment = {**r["judgment"], "judge_model": r.get("judge_model"),
                    "judgment_source": "rejudge", "judged_at": r.get("rejudged_at")}

    if judgment.get("verdict") != "FAILED" or judgment.get("criterion_satisfied") is not False:
        raise ValueError(f"{item_id}: target judgment is not a criterion failure: {judgment}")

    return {
        "id": item_id,
        "domain": bins.get(res.get("seed")),
        "prompt_variant": b.get("prompt"),
        "round": b.get("round"),
        "target_system": target,
        "user_query": res.get("seed"),
        "question": b["question"],
        "verification_criterion": b["verification_criterion"],
        "source": b["source"],
        "rewrite_kind": b.get("rewrite_kind"),
        "original_question": harder.get("updated_question"),
        "original_verification_criterion": harder.get("verification_criterion"),
        "generation": {
            "strategy": harder.get("chosen_strategy"),
            "why_harder": harder.get("why_harder"),
            "attempt": att.get("attempt"),
            "generator_model": data.get("model"),
            "prompt_variant": data.get("prompt_variant"),
            # the VC was already rewritten once during generation by the inline criterion check
            "vc_rewritten_inline": bool(harder.get("verification_criterion_original")),
        },
        "filter": {
            "decision": f.get("decision"),
            "pair_verdict": f.get("pair_verdict"),
            "question_review": strip_usage(f.get("question_review")),
            "vc_review": strip_usage(f.get("vc_review")),
            "vc_judged_against": f.get("vc_judged_against"),
            "phrase_check": strip_usage(f.get("phrase_check")),
            "model": f.get("model"),
        },
        "provenance": {
            "run_sample": str(sample_path),
            "filter_results": str(vdir / "results.jsonl"),
            "rejudge_results": str(vdir / "rejudge" / "results.jsonl") if r else None,
        },
        "systems": {s: ({"answer": answer, "judgment": judgment} if s == target else None)
                    for s in systems},
    }


def index_row(s: dict) -> dict:
    return {
        "id": s["id"], "file": f"samples/{file_name(s['id'])}",
        "domain": s["domain"], "target_system": s["target_system"],
        "prompt_variant": s["prompt_variant"], "round": s["round"],
        "source": s["source"], "rewrite_kind": s["rewrite_kind"],
        "question": s["question"], "verification_criterion": s["verification_criterion"],
        "verdicts": {k: (v["judgment"]["verdict"] if v else None)
                     for k, v in s["systems"].items()},
    }


def readme(samples: list, args) -> str:
    c = lambda key: Counter(s[key] for s in samples)
    table = lambda counter: "\n".join(f"| {k} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: -kv[1]))
    kinds = Counter(s["rewrite_kind"] for s in samples if s["source"] == "rewritten")
    others = len(args.systems) > 1
    return f"""# {args.out_dir.name}

{len(samples)} questions, each in its own file under `samples/`, with `index.jsonl` as a
one-line-per-question summary. Built {dt.date.today().isoformat()} by
`claude_inference/export_benchmark.py` from:

- run: `{args.run_dir}`
- filter + rejudge: `{args.verification_dir}` (questions = `rejudge/benchmark.jsonl`)

{"Not a release: the non-target systems' slots are null until answer_cross_systems.py fills them." if others else f"One answering system ({args.systems[0]}); no cross-system slots."}

## What is in it

Every question the target answering system **failed on its verification criterion** after the
final filter:

- **passed_as_written** ({c('source').get('passed_as_written', 0)}): passed final_verification unchanged;
  answer and judgment are the original run's.
- **rewritten** ({c('source').get('rewritten', 0)}): final_verification proposed a minimal rewrite, and the
  rewritten pair still fails after re-judging — criterion_only {kinds.get('criterion_only', 0)} (the stored
  answer re-judged against the new criterion), question_rewrite {kinds.get('question_rewrite', 0)} (the
  rewritten question re-answered by the same system and judged).

`question` / `verification_criterion` are the final versions; `original_*` are as generated.

| target system | n |
| --- | --- |
{table(c('target_system'))}

| domain | n |
| --- | --- |
{table(c('domain'))}

| prompt variant | n |
| --- | --- |
{table(c('prompt_variant'))}

## Sample file fields

| field | meaning |
| --- | --- |
| `id` | `round_KK/[<system>/]<prompt>/sample_NNN` — the run's sample path; the file name is the id with `/` → `__` |
| `domain`, `prompt_variant`, `round` | from `seed_bins.json` (null for seeds it does not cover) and the run layout |
| `target_system` | the answering system the question was generated against and failed |
| `user_query` | the original user query (seed) the question was generated from |
| `question`, `verification_criterion` | final versions |
| `source`, `rewrite_kind` | `passed_as_written` / `rewritten`; `criterion_only` / `question_rewrite` / null |
| `original_question`, `original_verification_criterion` | as generated by the run |
| `generation` | strategy, why_harder, attempt, generator model, whether the VC was rewritten inline during generation |
| `filter` | final_verification's question and VC reviews, phrase check (terms + Semantic Scholar counts), decision |
| `provenance` | paths to the run sample file, the filter results, the rejudge results |
| `systems.<name>` | `{{answer, judgment}}` per answering system; null until answered |
| `systems.<name>.answer` | `text` (raw), `trace`, `model`, `answer_source` (`original_run` / `re_answered` / later `cross_system`), `answered_at` |
| `systems.<name>.judgment` | `verdict`, `criterion_satisfied`, `criterion_reasoning`, `other_issues`, `summary`, `judge_model`, `judgment_source` (`original_run` / `rejudge`), `judged_at` |

`index.jsonl` has `id`, `file`, the metadata above, the final question and criterion, and
`verdicts` — one per system (null until answered).

## Notes

- **Answers are stored raw.** DR-Tulu's answers carry `<cite id="…">` tags that resolve only
  against the trace; Tongyi and WebThinker emit no citations, only a trace of what they read.
  `claude_inference/cite_utils.py` renders both (`format_answer_for_judge` in
  research_pipeline.py) into what the judge saw.
- `answer.model` is null for DR-Tulu: its server does not report a model name.
- **Judgments use the run's judge** (`gpt-5.6-terra`, research_pipeline's judge prompts). For
  passed_as_written items the judgment is from the original run, whose exact prompt was not
  stored and whose citation rendering used the cite_utils of that time.
- **Answer timing differs**: original-run answers predate the rejudge answers{", and cross-system answers are later still" if others else ""}. Answers vary from run to run, so a
  single FAILED is one sample, not a guarantee.
{"- Questions were generated against `target_system` with that system's profile in the" + chr(10) + "  generation prompt; a failure on another system is evidence the question is hard in general," + chr(10) + "  not something it was tuned for." + chr(10) if others else ""}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\nThe source directories")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=Path("runs/final_loop_600"))
    ap.add_argument("--verification-dir", type=Path,
                    default=Path("runs/final_loop_600_final_verification_v14"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/final_loop_600_full_benchmark"))
    ap.add_argument("--seed-bins", type=Path, default=HERE / "seed_bins.json",
                    help="seed -> domain map; seeds it does not cover get domain null.")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS),
                    help="Answering-system slots in each sample (default: %(default)s). The "
                         "target fills its own; the rest stay null for answer_cross_systems.py.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build every record in memory and report; write nothing.")
    args = ap.parse_args()

    run_dir, vdir, out = args.run_dir, args.verification_dir, args.out_dir
    for src in (run_dir, vdir):
        if not src.exists():
            ap.error(f"source dir not found: {src}")
        if out.resolve().is_relative_to(src.resolve()):
            ap.error(f"--out-dir {out} is inside source dir {src}; it must be separate")
    if out.exists():
        ap.error(f"{out} already exists; refusing to overwrite (move or delete it first)")

    bench = load_jsonl(vdir / "rejudge" / "benchmark.jsonl")
    filt = {r["id"]: r for r in load_jsonl(vdir / "results.jsonl")}
    rej = {r["id"]: r for r in load_jsonl(vdir / "rejudge" / "results.jsonl")}
    bins = json.loads(args.seed_bins.read_text()) if args.seed_bins.exists() else {}
    if len({b["id"] for b in bench}) != len(bench):
        sys.exit("duplicate ids in benchmark.jsonl")

    if (bad := {b["system"] for b in bench} - set(args.systems)):
        ap.error(f"target system(s) {sorted(bad)} not in --systems {args.systems}")
    samples = [build_sample(b, run_dir, vdir, filt, rej, bins, tuple(args.systems))
               for b in bench]
    missing_domain = [s["id"] for s in samples if s["domain"] is None]
    size = sum(len(json.dumps(s, ensure_ascii=False)) for s in samples)
    print(f"{len(samples)} samples | sources {dict(Counter(s['source'] for s in samples))} | "
          f"targets {dict(Counter(s['target_system'] for s in samples))} | "
          f"~{size / 1e6:.0f} MB | missing domain: {len(missing_domain)}")
    if args.dry_run:
        print(f"dry run: would write {out}/")
        return

    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)                      # leftover from an interrupted build of ours
    (tmp / "samples").mkdir(parents=True)
    for s in samples:
        (tmp / "samples" / file_name(s["id"])).write_text(
            json.dumps(s, ensure_ascii=False, indent=1))
    (tmp / "index.jsonl").write_text(
        "".join(json.dumps(index_row(s), ensure_ascii=False) + "\n" for s in samples))
    (tmp / "README.md").write_text(readme(samples, args))
    tmp.rename(out)
    print(f"wrote {out}/ ({len(samples)} sample files, index.jsonl, README.md)")


if __name__ == "__main__":
    main()
