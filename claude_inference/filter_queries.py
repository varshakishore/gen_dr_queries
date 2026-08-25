#!/usr/bin/env python3
"""
Label queries from allenai/asta-user-interactions with an LLM filter and push the
results to a new HF dataset.

Each query is sent to an OpenAI model with the fixed rubric prompt below and gets
back {English, Clarity, Research Question, Request Type, Usable} as JSON (enforced
server-side via strict structured outputs, so no parsing/retry loop is needed).

Results are appended to a local JSONL cache as they complete, so an interrupted
run resumes by re-running the same command. Queries already present in the cache
-- or already in the destination HF dataset -- are skipped.

Runs accumulate. `push_to_hub` replaces the split rather than appending to it, so
each push writes rows already in the dataset plus this run's new ones, keyed by
query. Widening --limit or moving --start therefore grows the dataset instead of
replacing it with the latest window.

Examples:
  # label 200 queries and push them to varshak1/asta-user-interactions-filtered
  python filter_queries.py --limit 200

  # resume/extend the same dataset (already-labeled queries are skipped)
  python filter_queries.py --limit 500

  # label only, don't push (inspect the JSONL first)
  python filter_queries.py --limit 50 --no-push

  # write somewhere else
  python filter_queries.py --out-repo me/scratch-filtered --limit 50

Spend is tracked from each response's token usage and capped by --max-cost (default
$25): once the cap is reached no new requests are sent, in-flight ones finish, and
everything labeled so far is still cached and pushed.

Requires OPENAI_API_KEY exported in the environment and, for --push, a HF token
(`huggingface-cli login` or HF_TOKEN).

Run command: python filter_queries.py --limit 2000 --concurrency 16 --max-cost 5
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

SOURCE_DATASET = "allenai/asta-user-interactions"
SOURCE_CONFIG = "optin_queries"
SOURCE_SPLIT = "train"

DEFAULT_OUT_REPO = "varshak1/asta-user-interactions-filtered"

PROMPT = """You will be given a user query. Evaluate it according to the criteria below.

Return only valid JSON in exactly this format:
{
  "English": <true | false>,
  "Clarity": <"clear" | "vague" | "needs clarification">,
  "Research Question": <true | false>,
  "Request Type": <"information seeking" | "design" | "review">,
  "Usable": <true | false>
}

Definitions:
- "English": true if the query is primarily written in English; otherwise false.
- "Clarity":
  - "clear": The user's request or question can be understood without needing additional information.
  - "vague": The general intent is understandable, but some details or scope are ambiguous.
  - "needs clarification": The intended request cannot be reliably determined without additional information.
- "Research Question": true if the query asks a question that could reasonably and meaningfully be answered by consulting academic or scholarly literature; otherwise false.
- "Request Type": what the query mainly asks the assistant to DO. Judge by the instruction, not
  by the topic.
  - "design": design or draft the user's own research -- study designs, protocols, methods or statistical analysis plans, survey/interview instruments, grant or thesis proposals, hypotheses to test, or sections of the user's paper.
  - "review": critique, edit, or summarize a text, draft, manuscript, protocol, dataset, or code that the user supplies or pastes in.
  - "information seeking": anything else -- the user wants an answer, evidence, or a synthesis of what the literature says. Asking for a review, survey, or overview of the published literature* on a topic belongs here, not in review.  
- "Usable": true only if all of the following are true:
  1. "English" is true,
  2. "Clarity" is "clear" or "vague",
  3. "Research Question" is true,
  4. "Request Type" is "information seeking".

Do not include explanations, reasoning, Markdown, or any text outside the JSON.

User query:"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "English": {"type": "boolean"},
        "Clarity": {"type": "string", "enum": ["clear", "vague", "needs clarification"]},
        "Research Question": {"type": "boolean"},
        "Request Type": {"type": "string",
                         "enum": ["information seeking", "design", "review"]},
        "Usable": {"type": "boolean"},
    },
    "required": ["English", "Clarity", "Research Question", "Request Type", "Usable"],
    "additionalProperties": False,
}

# Columns of the pushed dataset. Rows are projected onto this before the push, so a
# cache (or an existing dest dataset) written under an older column set still loads.
# Per-row token counts stay in the JSONL cache for auditing and are dropped here.
COLUMNS = ("query", "thread_id", "english", "clarity", "research_question",
           "request_type", "usable", "filter_model")

# USD per 1M tokens, (input, output). Override for other models with --price-in/--price-out.
PRICING = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-sol": (4.00, 20.00),
}

_write_lock = threading.Lock()


class BudgetExceeded(Exception):
    """Raised in a worker when the spend cap was hit before its request went out."""


class Budget:
    """Thread-safe spend tracker with a hard stop.

    Cached input tokens bill cheaper than fresh ones but are counted here at the full
    input rate, so the running total is an upper bound on real spend -- which is what
    you want a cap to be. Because up to --concurrency requests are already in flight
    when the cap trips, the final total can overshoot by that many queries.
    """

    def __init__(self, limit: float, price_in: float, price_out: float):
        self.limit = limit
        self.price_in = price_in
        self.price_out = price_out
        self.in_tokens = 0
        self.out_tokens = 0
        self.spent = 0.0
        self.stopped = False
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.stopped:
            raise BudgetExceeded()

    def record(self, in_tokens: int, out_tokens: int) -> float:
        with self._lock:
            self.in_tokens += in_tokens
            self.out_tokens += out_tokens
            self.spent = (self.in_tokens * self.price_in
                          + self.out_tokens * self.price_out) / 1e6
            if self.limit and self.spent >= self.limit and not self.stopped:
                self.stopped = True
                print(f"\n*** spend cap hit: ${self.spent:.2f} >= ${self.limit:.2f} — "
                      f"no new requests will be sent (in-flight ones will finish) ***\n",
                      flush=True)
            return self.spent


def load_source_queries(args) -> list[dict]:
    """Stream unique queries from the source dataset, honoring --tool/--start/--limit."""
    from datasets import load_dataset

    ds = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split=SOURCE_SPLIT, streaming=True)

    need = (args.start + args.limit) if args.limit else None
    seen: set[str] = set()
    rows: list[dict] = []
    n_long = 0
    for row in ds:
        if args.tool != "any" and row.get("tool") != args.tool:
            continue
        q = (row.get("query") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)  # before the length check, so a dropped query isn't recounted
        # Dropped here rather than in the rubric: a pasted manuscript costs thousands of
        # input tokens to label and is thrown away anyway.
        if args.max_words and len(q.split()) > args.max_words:
            n_long += 1
            continue
        rows.append({"query": q, "thread_id": row.get("thread_id")})
        if need and len(rows) >= need:
            break

    sliced = rows[args.start: (args.start + args.limit) if args.limit else None]
    print(f"Loaded {len(sliced)} unique query(ies) from {SOURCE_DATASET} "
          f"[{SOURCE_CONFIG}/{SOURCE_SPLIT}, tool={args.tool}, "
          f"window {args.start}:{args.start + len(sliced)}]")
    if n_long:
        print(f"  (skipped {n_long} query(ies) over {args.max_words} words)")
    return sliced


def load_cache(path: Path) -> dict[str, dict]:
    """Read previously labeled rows from the local JSONL cache, keyed by query."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line from a killed run
            if row.get("query"):
                out[row["query"]] = row
    return out


def load_existing_repo(repo: str, revision: str | None) -> dict[str, dict]:
    """Read already-labeled rows from the destination HF dataset, if it exists."""
    from datasets import load_dataset

    try:
        ds = load_dataset(repo, split="train", revision=revision)
    except Exception as e:  # not created yet, no access, empty, ...
        print(f"[dest] {repo}: no existing dataset to resume from ({type(e).__name__}: {e})")
        return {}
    rows = {r["query"]: r for r in ds if r.get("query")}
    print(f"[dest] {repo}: {len(rows)} already-labeled query(ies)")
    return rows


def label_one(client: OpenAI, args, item: dict, budget: Budget) -> dict:
    """Send one query through the rubric. Returns the item plus verdict fields."""
    budget.check()  # don't start a request we've already decided we can't afford

    # temperature is left unset: the model default, and reasoning models reject
    # non-default values anyway.
    resp = client.chat.completions.create(
        model=args.model,
        max_completion_tokens=args.max_tokens,
        messages=[{"role": "user", "content": f"{PROMPT}\n{item['query']}"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "query_filter", "strict": True,
                            "schema": VERDICT_SCHEMA},
        },
    )

    usage = resp.usage
    in_tokens = getattr(usage, "prompt_tokens", 0) or 0
    out_tokens = getattr(usage, "completion_tokens", 0) or 0
    budget.record(in_tokens, out_tokens)  # billed even if the content is unusable

    choice = resp.choices[0]
    if choice.message.refusal:
        raise RuntimeError(f"refused: {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise RuntimeError("truncated (raise --max-tokens)")
    if not choice.message.content:
        raise RuntimeError(f"empty content (finish_reason={choice.finish_reason})")
    verdict = json.loads(choice.message.content)

    return {
        **item,
        "english": verdict["English"],
        "clarity": verdict["Clarity"],
        "research_question": verdict["Research Question"],
        "request_type": verdict["Request Type"],
        "usable": verdict["Usable"],
        "filter_model": args.model,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
    }


def append_row(path: Path, row: dict) -> None:
    with _write_lock:
        with open(path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def push(args, rows: list[dict]) -> None:
    from datasets import Dataset

    if args.usable_only:
        rows = [r for r in rows if r.get("usable")]
    if not rows:
        print("[push] nothing to push")
        return
    rows = sorted(rows, key=lambda r: r["query"])
    ds = Dataset.from_list([{k: r.get(k) for k in COLUMNS} for r in rows])
    print(f"[push] pushing {len(ds)} row(s) to {args.out_repo} (private={args.private})")
    ds.push_to_hub(args.out_repo, private=args.private)
    print(f"[push] done: https://huggingface.co/datasets/{args.out_repo}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-repo", default=DEFAULT_OUT_REPO,
                   help=f"destination HF dataset repo id (default: {DEFAULT_OUT_REPO})")
    p.add_argument("--limit", type=int, default=100,
                   help="how many unique source queries to consider (0 = all). "
                        "This is a window over the source stream, not a count of new labels, "
                        "so re-running with the same value resumes deterministically.")
    p.add_argument("--start", type=int, default=0,
                   help="skip this many unique source queries before the window")
    p.add_argument("--tool", default="sqa",
                   help="source `tool` value to keep ('any' for no filter). Default: sqa")
    p.add_argument("--max-words", type=int, default=300,
                   help="skip source queries longer than this many words (0 = no maximum).")
    p.add_argument("--cache", default=None,
                   help="local JSONL of labeled rows (default: runs/filter_queries/<repo>.jsonl)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--model", default="gpt-5.6-luna",
                   help="OpenAI model id (must support strict structured outputs)")
    p.add_argument("--max-cost", type=float, default=25.0,
                   help="stop sending requests once estimated spend reaches this many "
                        "USD (0 = no cap)")
    p.add_argument("--price-in", type=float, default=None,
                   help="USD per 1M input tokens (default: looked up from --model)")
    p.add_argument("--price-out", type=float, default=None,
                   help="USD per 1M output tokens (default: looked up from --model)")
    p.add_argument("--max-tokens", type=int, default=2000,
                   help="max_completion_tokens; raise for reasoning models, which "
                        "spend this budget on reasoning tokens too")
    p.add_argument("--private", action="store_true", help="create the dataset as private")
    p.add_argument("--usable-only", action="store_true",
                   help="push only rows with usable=true (default: push all with labels). "
                        "The push replaces the split, so turning this on drops "
                        "already-pushed unusable rows from the dataset")
    p.add_argument("--no-push", dest="push", action="store_false",
                   help="label into the local JSONL only; skip the HF push")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="re-label queries even if they are already labeled")
    args = p.parse_args()

    cache_path = Path(args.cache) if args.cache else (
        Path("runs/filter_queries") / f"{args.out_repo.replace('/', '__')}.jsonl"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Always load what's already labeled: push_to_hub *replaces* the split, so these
    # rows have to be carried into the push or they'd be dropped from the dataset.
    # --no-skip-existing only means "re-label them", not "forget them".
    known: dict[str, dict] = dict(load_existing_repo(args.out_repo, revision=None))
    cached = load_cache(cache_path)
    print(f"[cache] {cache_path}: {len(cached)} already-labeled query(ies)")
    known.update(cached)

    source = load_source_queries(args)
    todo = source if not args.skip_existing else [r for r in source if r["query"] not in known]
    print(f"{len(todo)} query(ies) to label, {len(source) - len(todo)} skipped as already done")

    price_in, price_out = PRICING.get(args.model, (None, None))
    price_in = args.price_in if args.price_in is not None else price_in
    price_out = args.price_out if args.price_out is not None else price_out
    if price_in is None or price_out is None:
        print(f"error: no pricing for {args.model!r}; pass --price-in and --price-out "
              f"(USD per 1M tokens) so the spend cap can be enforced", file=sys.stderr)
        return 2
    budget = Budget(args.max_cost, price_in, price_out)
    print(f"[cost] {args.model}: ${price_in}/1M in, ${price_out}/1M out | "
          f"cap: {('$%.2f' % args.max_cost) if args.max_cost else 'none'}")

    client = OpenAI()
    labeled: list[dict] = []
    failed = 0
    skipped_budget = 0

    if todo:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(label_one, client, args, item, budget): item
                       for item in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                item = futures[fut]
                try:
                    row = fut.result()
                except BudgetExceeded:
                    skipped_budget += 1
                    continue
                except Exception as e:
                    failed += 1
                    print(f"[fail {i:>4}/{len(todo)}] {type(e).__name__}: {e} "
                          f"— {item['query'][:80]}", flush=True)
                    continue
                append_row(cache_path, row)  # written as we go, so the run is resumable
                labeled.append(row)
                print(f"[ok   {i:>4}/{len(todo)}] ${budget.spent:7.4f} "
                      f"usable={row['usable']!s:<5} clarity={row['clarity']:<20} "
                      f"type={row['request_type']:<19} "
                      f"{row['query'][:60]}", flush=True)

    # Accumulate: everything already labeled, plus this run's rows (which win on
    # re-label). The dataset therefore grows across runs with different --start/--limit
    # instead of each push replacing it with only the latest window.
    merged = dict(known)
    for row in labeled:
        merged[row["query"]] = row
    push_rows = list(merged.values())

    n_usable = sum(1 for r in push_rows if r.get("usable"))
    pct = f" ({n_usable / len(push_rows):.1%})" if push_rows else ""
    print(f"\nlabeled this run: {len(labeled)} | failed: {failed} | "
          f"skipped (spend cap): {skipped_budget} | "
          f"dataset total: {len(push_rows)} ({len(push_rows) - len(known)} new) | "
          f"usable: {n_usable}{pct}")
    print(f"[cost] {budget.in_tokens:,} in + {budget.out_tokens:,} out tokens "
          f"= ${budget.spent:.4f}"
          + (f" | ${budget.spent / len(labeled):.5f}/query" if labeled else ""))
    if budget.stopped:
        print(f"[cost] stopped at the ${args.max_cost:.2f} cap — re-run with a higher "
              f"--max-cost to continue where this left off")

    if args.push:
        push(args, push_rows)
    else:
        print(f"[push] skipped (--no-push); rows are in {cache_path}")

    return 1 if failed and not labeled else 0


if __name__ == "__main__":
    sys.exit(main())
