#!/usr/bin/env python3
"""Classify seed and updated questions by Bloom's taxonomy using an LLM.

Reads ``analysis/question_pairs.json`` (a list of ``[seed_question, meta, result]``
triples) and, for each pair, asks Claude to assign both the seed question and the
updated (hardened) question to a level of Bloom's revised taxonomy.

Output keeps, for every pair:
  - seed_question
  - updated_question
  - verification_criterion
  - result            (PASSED / FAILED)
  - strategy          (the chosen hardening strategy)
  - seed_bloom_level      + seed_bloom_rationale
  - updated_bloom_level   + updated_bloom_rationale

Results are written to JSON and CSV.

Auth: reads ANTHROPIC_API_KEY from the environment or from a .env file in the
repo root / current directory (no python-dotenv dependency).

Usage:
    python analysis/classify_blooms.py
    python analysis/classify_blooms.py --input analysis/question_pairs.json \
        --out-json analysis/blooms_classified.json \
        --out-csv  analysis/blooms_classified.csv \
        --model claude-sonnet-4-5 --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bloom's taxonomy reference (revised taxonomy, ordered low -> high)
# ---------------------------------------------------------------------------

BLOOM_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

BLOOM_REFERENCE = """Bloom's revised taxonomy (ordered from lowest to highest cognitive demand):

- Remember: recall facts and basic concepts (define, duplicate, list, memorize, repeat, state)
- Understand: explain ideas or concepts (classify, describe, discuss, explain, identify, locate, recognize, report, select, translate)
- Apply: use information in new situations (execute, implement, solve, use, demonstrate, interpret, operate, schedule, sketch)
- Analyze: draw connections among ideas (differentiate, organize, relate, compare, contrast, distinguish, examine, experiment, question, test)
- Evaluate: justify a stand or decision (appraise, argue, defend, judge, select, support, value, critique, weigh)
- Create: produce new or original work (design, assemble, construct, conjecture, develop, formulate, author, investigate)"""

SYSTEM_PROMPT = f"""You are an expert in educational assessment and Bloom's taxonomy. \
You classify a research question by the primary level of cognitive demand it requires of the answerer.

{BLOOM_REFERENCE}

You will be given ONE question. Classify it into exactly ONE level. Choose the single highest \
level of cognitive demand that the question genuinely requires to answer well (not merely the \
verbs that happen to appear). Provide a one-sentence rationale.

Record your answer by calling the record_bloom_classification tool."""

USER_TEMPLATE = """QUESTION:
{question}"""

# Forced structured output: the model MUST call this tool, and the level field
# is enum-constrained, which eliminates JSON-parse failures and field swaps.
CLASSIFY_TOOL = {
    "name": "record_bloom_classification",
    "description": "Record the Bloom's taxonomy level of the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bloom_level": {"type": "string", "enum": BLOOM_LEVELS},
            "bloom_rationale": {"type": "string", "description": "One sentence."},
        },
        "required": ["bloom_level", "bloom_rationale"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dotenv(repo_root: Path) -> None:
    """Minimal .env loader: populate os.environ from a .env file if present.

    Does not overwrite variables already set in the environment.
    """
    for candidate in (Path.cwd() / ".env", repo_root / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def classify_question(client: Anthropic, model: str, question: str, retries: int = 2) -> dict:
    """One Claude call classifying a single question, independently.

    Uses forced tool-use so the reply is schema-validated structured data.
    Retries on transient errors or truncated tool input.
    """
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,  # headroom so the tool-input JSON never truncates
                temperature=0,  # deterministic / reproducible classifications
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # static prompt -> cache it
                }],
                tools=[CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": CLASSIFY_TOOL["name"]},
                messages=[{
                    "role": "user",
                    "content": USER_TEMPLATE.format(question=question),
                }],
            )
            if resp.stop_reason == "refusal":
                # Deterministic at temperature=0 — retrying won't help; skip it.
                raise RuntimeError("model refused to classify this question")
            tool_use = next(b for b in resp.content if b.type == "tool_use")
            parsed = tool_use.input
            return {
                "level": parsed["bloom_level"],  # KeyError -> truncated; retry
                "rationale": parsed.get("bloom_rationale", "").strip(),
            }
        except RuntimeError:
            raise  # refusal: don't burn retries
        except Exception as e:
            last_err = e
    raise last_err


def build_row(pair: list) -> dict:
    """Extract the fields we want to keep from one [seed, meta, result] triple."""
    seed, meta, result = pair[0], pair[1], pair[2]
    return {
        "seed_question": seed,
        "updated_question": meta.get("updated_question", ""),
        "verification_criterion": meta.get("verification_criterion", ""),
        "result": result,
        "strategy": meta.get("chosen_strategy", ""),
        "source": "question_pairs",
    }


def build_round0_rows(path: Path) -> list[dict]:
    """Rows for the FAILED ('already hard') examples in round0_summary.json.

    These seeds defeated the research system as-is, so there is no rewritten
    question — the failed question IS the seed. We mirror it into
    ``updated_question`` so it lands in the FAILED group of the analysis.
    """
    entries = json.loads(path.read_text())
    rows = []
    for e in entries:
        if e.get("verdict") != "FAILED":
            continue
        seed = e["seed"]
        rows.append({
            "seed_question": seed,
            "updated_question": seed,  # already hard; no rewrite
            "verification_criterion": "; ".join(e.get("criteria", [])),
            "result": "FAILED",
            "strategy": "ALREADY_HARD (seed already hard; no rewrite)",
            "source": "round0_already_hard",
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(repo_root / "analysis" / "question_pairs.json"))
    parser.add_argument("--out-json", default=str(repo_root / "analysis" / "blooms_classified.json"))
    parser.add_argument("--out-csv", default=str(repo_root / "analysis" / "blooms_classified.csv"))
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--workers", type=int, default=6, help="concurrent API calls")
    parser.add_argument("--limit", type=int, default=None, help="classify only the first N pairs (for testing)")
    parser.add_argument("--round0-summary",
                        default=str(repo_root / "claude_inference" / "round0_summary.json"),
                        help="also include FAILED 'already hard' examples from this summary "
                             "(pass empty string to skip)")
    args = parser.parse_args()

    load_dotenv(repo_root)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY is not set (env or .env file).")

    pairs = json.loads(Path(args.input).read_text())
    if args.limit is not None:
        pairs = pairs[: args.limit]
    rows = [build_row(p) for p in pairs]
    print(f"Loaded {len(rows)} question pairs from {args.input}")

    if args.round0_summary:
        round0_rows = build_round0_rows(Path(args.round0_summary))
        rows.extend(round0_rows)
        print(f"Added {len(round0_rows)} FAILED 'already hard' examples from {args.round0_summary}")

    client = Anthropic()

    def classify_one(question: str, prefix: str) -> tuple[dict, bool]:
        """Classify a single question; on failure return an ERROR record."""
        if not question.strip():
            return {f"{prefix}_bloom_level": "ERROR",
                    f"{prefix}_bloom_rationale": "empty question"}, True
        try:
            cls = classify_question(client, args.model, question)
            return {f"{prefix}_bloom_level": cls["level"],
                    f"{prefix}_bloom_rationale": cls["rationale"]}, False
        except Exception as e:
            return {f"{prefix}_bloom_level": "ERROR",
                    f"{prefix}_bloom_rationale": f"{type(e).__name__}: {e}"}, True

    def work(idx_row):
        # The seed and updated questions are classified in two independent calls.
        idx, row = idx_row
        seed_cls, seed_err = classify_one(row["seed_question"], "seed")
        if row["updated_question"] == row["seed_question"]:
            # 'Already hard' rows: the updated question IS the seed — reuse the
            # one classification instead of paying for an identical second call.
            upd_cls = {"updated_bloom_level": seed_cls["seed_bloom_level"],
                       "updated_bloom_rationale": seed_cls["seed_bloom_rationale"]}
            upd_err = seed_err
        else:
            upd_cls, upd_err = classify_one(row["updated_question"], "updated")
        return idx, {**row, **seed_cls, **upd_cls}, (seed_err or upd_err)

    results: list[Optional[dict]] = [None] * len(rows)
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, (i, r)) for i, r in enumerate(rows)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="classifying"):
            idx, record, err = fut.result()
            results[idx] = record
            if err:
                errors += 1

    # Write JSON.
    Path(args.out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Write CSV.
    fieldnames = [
        "seed_question", "updated_question", "verification_criterion",
        "result", "strategy", "source",
        "seed_bloom_level", "seed_bloom_rationale",
        "updated_bloom_level", "updated_bloom_rationale",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in results:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    # Brief summary of the distribution.
    def dist(key):
        counts: dict[str, int] = {}
        for rec in results:
            counts[rec[key]] = counts.get(rec[key], 0) + 1
        order = BLOOM_LEVELS + [k for k in counts if k not in BLOOM_LEVELS]
        return {lvl: counts[lvl] for lvl in order if lvl in counts}

    print(f"\nWrote {len(results)} rows to:\n  {args.out_json}\n  {args.out_csv}")
    if errors:
        print(f"WARNING: {errors} pair(s) failed classification (level=ERROR).", file=sys.stderr)
    print("\nSeed question Bloom levels:   ", dist("seed_bloom_level"))
    print("Updated question Bloom levels:", dist("updated_bloom_level"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
