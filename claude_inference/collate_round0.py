"""Collate round-0 results from a research_pipeline output JSON.

Joins each pipeline result back to the original ID + metadata in test-lucy.jsonl
(matching by seed question text) and prints a table + writes a CSV/JSON summary.

Usage:
    python collate_round0.py \\
        --pipeline-output round0_test_lucy.json \\
        --source ../data/test-lucy.jsonl \\
        --out-csv round0_summary.csv \\
        --out-json round0_summary.json
"""

import argparse
import csv
import json
from pathlib import Path


def load_source(path: Path) -> dict:
    by_text = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            by_text[d["seed_question"].strip()] = d
    return by_text


def round0_from(result: dict) -> dict | None:
    for a in result.get("attempts", []):
        if a.get("attempt") == 0:
            return a
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-output", required=True,
                    help="JSON written by research_pipeline.py via --output")
    ap.add_argument("--source", required=True,
                    help="Original JSONL file with id/seed_question/metadata")
    ap.add_argument("--out-csv", help="Where to write the summary CSV")
    ap.add_argument("--out-json", help="Where to write the detailed summary JSON")
    args = ap.parse_args()

    src_by_text = load_source(Path(args.source))
    pipeline = json.loads(Path(args.pipeline_output).read_text())

    rows = []
    for r in pipeline["results"]:
        seed = r["seed"].strip()
        src = src_by_text.get(seed)
        rid = src["id"] if src else "<unknown>"
        original_critique = (src.get("metadata", {}).get("original_critique", "")
                             if src else "")
        attempt0 = round0_from(r)
        if attempt0 is None:
            rows.append({
                "id": rid, "seed": seed,
                "status": r.get("final_status", ""),
                "verdict": "",
                "criteria_failed": "",
                "criteria_total": "",
                "summary": r.get("error") or "(no round-0 attempt recorded)",
                "criteria": [],
                "criterion_results": [],
                "original_critique": original_critique,
                "pipeline_status": r.get("final_status", ""),
            })
            continue
        h = attempt0["harder"]
        j = attempt0["judgment"]
        crits = h.get("verification_criteria", []) or []
        rows.append({
            "id": rid,
            "seed": seed,
            "status": r.get("final_status", ""),
            "verdict": j.get("verdict", ""),
            "criteria_failed": j.get("criteria_failed_count", 0),
            "criteria_total": len(crits),
            "summary": j.get("summary", ""),
            "criteria": crits,
            "criterion_results": j.get("criterion_results", []),
            "original_critique": original_critique,
            "pipeline_status": r.get("final_status", ""),
        })

    # ---- Pretty table to stdout ----
    headers = ["id", "verdict", "fail/tot", "status", "seed (truncated)"]
    widths = [16, 8, 8, 14, 60]
    sep = "  "
    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep.join("-" * w for w in widths))
    for row in rows:
        ft = f"{row['criteria_failed']}/{row['criteria_total']}" \
            if row["criteria_total"] != "" else "—"
        cells = [
            row["id"][:widths[0]].ljust(widths[0]),
            (row["verdict"] or "—").ljust(widths[1]),
            ft.ljust(widths[2]),
            (row["pipeline_status"] or "—").ljust(widths[3]),
            row["seed"][:widths[4]],
        ]
        print(sep.join(cells))

    n = len(rows)
    n_failed = sum(1 for r in rows if r["verdict"] == "FAILED")
    n_passed = sum(1 for r in rows if r["verdict"] == "PASSED")
    n_err = sum(1 for r in rows if r["pipeline_status"] == "ERROR")
    print()
    print(f"Round-0 outcomes: {n_failed} FAILED (ALREADY_HARD), "
          f"{n_passed} PASSED, {n_err} ERROR, {n} total")
    if n_failed:
        avg_fail = sum(r["criteria_failed"] for r in rows
                       if r["verdict"] == "FAILED") / n_failed
        print(f"Average criteria failed (FAILED rows): {avg_fail:.2f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "id", "verdict", "criteria_failed", "criteria_total",
                "pipeline_status", "summary", "seed", "criteria_joined",
                "original_critique",
            ])
            for r in rows:
                w.writerow([
                    r["id"], r["verdict"], r["criteria_failed"], r["criteria_total"],
                    r["pipeline_status"], r["summary"], r["seed"],
                    " | ".join(r["criteria"]),
                    r["original_critique"],
                ])
        print(f"\nCSV written: {args.out_csv}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2))
        print(f"JSON written: {args.out_json}")


if __name__ == "__main__":
    main()
