#!/usr/bin/env python3
"""
Outer loop: generate M questions in rounds of N, re-deriving the strategy menus from
strategy feedback after every round.

    seeds[0:N]   ->  generate  ->  strategy feedback  ->  new menus
    seeds[N:2N]  ->  generate  ->  strategy feedback  ->  new menus
    ...

Each round:

  1. SPLIT   the round's seeds between the two make-harder prompts by an independent coin
             flip per seed (--prompt-mix, default 0.5 -> 50/50 explore vs exploit), and run
             research_pipeline_parallel.py once per side into
             <out-dir>/round_KK/explore/ and <out-dir>/round_KK/exploit/.
             Separate dirs keep the `source_run` labels that make the per-prompt comparison
             in the feedback work; strategy_feedback_module globs `*/sample_*.json` under a
             round dir, so both sides are picked up by passing the round dir alone.

  2. FEED    the round dirs (all of them so far, or just this one -- see --feedback-scope)
     BACK    through strategy_feedback_module.build_feedback, and rewrite the two menus:

               EXAMPLE STRATEGIES TO CONSIDER  <- focus_strategies
                   what works but is under-used, plus seeds never tried. Injected into
                   PROMPT_TO_MAKE_HARDER_QUESTION_EXPLOIT (the 'exploit' prompt).

               STRATEGIES TO NOT USE           <- ineligible_strategies
                   the ruts (over-represented) and the duds (tried >= --ban-min-evidence
                   times and still below min_failure_rate). Injected into
                   PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE (the 'explore' prompt).
                   Clusters excluded merely for being small are NOT banned -- too little
                   evidence to write them off.

               "Here are a few examples:"       <- failures from the run itself
                   Round 0 shows the built-in DEFAULT_FEW_SHOTS. Afterwards each prompt
                   gets --few-shots-per-round (default 3) worked examples resampled from
                   the set THAT prompt needs: 'exploit' from failures of the strategies
                   on its new example menu (demonstrating the menu it must work from),
                   'explore' from failures of the newly banned clusters (its examples are
                   negative examples -- "do not use the same strategies as the examples
                   above" -- so they must show what it is told to avoid). One example per
                   strategy before a second from any; short samples are padded with the
                   built-in three. --static-few-shots keeps round 0's examples throughout.

             So the two prompts get complementary halves of the same feedback: 'exploit'
             is told what to lean into, 'explore' is told what ground is already covered.
             The two lists are independent. 'exploit' is instructed to work from its
             menu (or a variation of an entry), so a ban only ever speaks to 'explore',
             and the same strategy may legitimately be recommended to one prompt while
             banned from the other.

             The three strategies the prompt's own few-shot examples demonstrate stay
             banned in every round; dynamic bans accumulate across rounds up to
             --max-banned-strategies (oldest dropped first), since a strategy stops
             looking over-represented as soon as it is banned, and un-banning it would
             just oscillate.

Round 0 starts from the built-in menus (--strategies / --banned-strategies). The last
round is not followed by a feedback call -- there is nothing left to feed it to.

  3. VERIFY  (--verify-after) once every round is done, verify_questions.py re-checks the
             harvest: for each question that broke the answering system, retrieve S2 papers
             and ask whether its verification criterion is itself correct, dropping the ones
             that are not. This is the same check as the pipeline's inline --verify-criterion
             but paid only on questions worth keeping, and it cannot end a seed mid-run.
             Writes <out-dir>/verified.json (+ .kept.json, the filtered set).

Each round dir holds the menus it was actually run with (example_strategies.txt,
banned_strategies.txt), the worked examples each prompt was shown (few_shots.exploit.json
/ few_shots.explore.json; round 0's are the built-in three), the seed split
(explore.seeds.json / exploit.seeds.json), and the
feedback it produced (feedback.json + feedback.txt). <out-dir>/loop.json is the manifest.

Resumability: the driver skips seeds whose sample_NNN.json exists, and a round's feedback
is reused if feedback.json is already there (--refresh-feedback recomputes it), so
re-running the same command continues an interrupted loop without re-paying for either.

Examples:
  # 50 questions in rounds of 10, feedback 4 times, 10 seeds in flight
  python research_loop.py --out-dir runs/loop1 --total 50 --feedback-every 10 --concurrency 10

  # only explore, and score the round on its own rather than cumulatively
  python research_loop.py --out-dir runs/loop2 --total 40 --feedback-every 10 \
      --prompt-mix 1.0 --feedback-scope last

Requires ANTHROPIC_API_KEY (generation + clustering) and a research server on --server-url.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import research_pipeline as RP
from research_pipeline_parallel import load_seeds
from strategy_feedback_module import (
    build_feedback,
    format_summary,
    load_examples_from_runs,
    write_feedback,
)

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "research_pipeline_parallel.py"
VERIFIER = HERE / "verify_questions.py"

# Kept as the tail of the example menu so the generator is never boxed into the list.
OPEN_ENDED_TAIL = ("Something else you think of that would be effective at exposing "
                   "weaknesses in research systems!")
NOVEL_PREFIX = "[novel strategy not in seed menu] "


# ---------------------------------------------------------------------------
# Menu files
# ---------------------------------------------------------------------------


def write_menu(path: Path, items: list, header: str) -> Path:
    """Write a strategy menu as the one-per-line format load_strategy_file() reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(items)
    path.write_text(f"# {header}\n" + (body + "\n" if body else ""))
    return path


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _clean(description: str) -> str:
    """A cluster description as a usable strategy line, or '' if it isn't one."""
    d = description.strip()
    if d.startswith(NOVEL_PREFIX):
        d = d[len(NOVEL_PREFIX):].strip()
    # Pipeline bookkeeping labels and clustering fallbacks are not strategies.
    if not d or d.startswith(("ALREADY_HARD", "EXHAUSTED", "unclassified")):
        return ""
    if d.startswith("e.g. "):          # new cluster with no LLM label, described by a member
        d = d[len("e.g. "):].strip()
    return d


def _dedup(items: list) -> list:
    seen, out = set(), []
    for it in items:
        k = _norm(it)
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _shot(rec: dict) -> dict:
    """One feedback record (a few_shot_failure or a cluster instance) as a prompt example."""
    strategy = rec.get("strategy", "")
    if "|" in strategy and strategy.startswith("EXHAUSTED"):
        strategy = strategy.split("|", 1)[1].strip()      # drop the pipeline's status prefix
    return {
        "seed_question": rec.get("seed_question", ""),
        "brainstorming": rec.get("brainstorming", ""),
        "chosen_strategy": strategy,
        "updated_question": rec.get("updated_question", ""),
        "why_harder": rec.get("why_harder", ""),
        "verification_criterion": rec.get("verification_criterion", ""),
    }


def _round_robin(buckets: list, n: int) -> list:
    """Take up to n items, one per bucket per pass, so the picks spread across strategies."""
    out = []
    while len(out) < n and any(buckets):
        for b in buckets:
            if b and len(out) < n:
                out.append(b.pop(0))
    return out


def _pad(shots: list, defaults: list, n: int) -> list:
    """Top the list up to n with built-in examples the derived ones do not already cover."""
    have = {_norm(s["updated_question"]) for s in shots}
    for d in defaults:
        if len(shots) >= n:
            break
        if _norm(d.get("updated_question", "")) not in have:
            shots.append(d)
    return shots[:n]


def derive_few_shots(feedback: dict, example_menu: list, banned_menu: list, *,
                     defaults: list, n: int = 3) -> tuple[list, list]:
    """The next round's worked examples, sampled per prompt from the set that prompt needs.

    The two prompts use their examples for opposite purposes, so they draw from opposite
    halves of the feedback:

      'exploit' ("EXAMPLE STRATEGIES TO CONSIDER" + "select ... from the list below")
          -> failures of the FOCUS strategies that made it onto the example menu. The
             examples demonstrate the menu it is being told to work from.

      'explore' ("STRATEGIES TO NOT USE" + "do not use the same strategies as the examples")
          -> failures of the BANNED clusters. Its examples are negative examples, so they
             must demonstrate exactly what it is being told to avoid; sampling focus
             strategies here would tell it to avoid the strategies we want it to find.

    One example per strategy before a second from any (round-robin over distinct clusters),
    so `n` examples show `n` strategies. Short lists are padded from `defaults`, so the
    prompt always carries n worked examples.
    """
    menu_keys = {_norm(e) for e in example_menu}
    focus_buckets = [
        [_shot(x) for x in (f.get("few_shot_failures") or [])]
        for f in feedback["focus_strategies"]
        if _norm(_clean(f["description"])) in menu_keys
    ]

    # `ineligible_strategies` carries no examples, so the banned clusters' failures come
    # from cluster_comparison; biggest failure counts first, as the clearest demonstrations.
    banned_keys = {_norm(b) for b in banned_menu}
    banned_clusters = [c for c in feedback.get("cluster_comparison", [])
                       if _norm(_clean(c["description"])) in banned_keys]
    banned_clusters.sort(key=lambda c: c.get("num_failed", 0), reverse=True)
    banned_buckets = [
        [_shot(x) for x in (c.get("instances") or []) if x.get("failed")]
        for c in banned_clusters
    ]

    exploit = _pad(_round_robin(focus_buckets, n), defaults, n)
    explore = _pad(_round_robin(banned_buckets, n), defaults, n)
    return exploit, explore


def derive_menus(feedback: dict, prev_banned: list, *, base_banned: list,
                 max_examples: int, max_banned: int, ban_min_evidence: int,
                 keep_open_ended: bool = True) -> tuple[list, list, dict]:
    """Turn one feedback dict into the next round's (example_menu, banned_menu, provenance).

    Examples come from `focus_strategies` (works-but-rare, plus never-tried seeds), bans
    from `ineligible_strategies` -- but only the clusters that are over-represented or that
    have been tried at least `ban_min_evidence` times and still under-perform. A cluster
    left out merely for being small is neither recommended nor banned.
    """
    examples = _dedup(
        c for c in (_clean(f["description"]) for f in feedback["focus_strategies"]) if c
    )[:max_examples]
    if keep_open_ended:
        # The tail is a seed line in the built-in menus, so it can come back through
        # focus_strategies; dedup again so it is not listed twice. It sits outside
        # max_examples -- it is an escape hatch, not one of the ranked picks.
        examples = _dedup(examples + [OPEN_ENDED_TAIL])

    fresh_bans, ban_reasons = [], {}
    for r in feedback["ineligible_strategies"]:
        d = _clean(r["description"])
        if not d:
            continue
        reasons = r.get("excluded_for") or []
        over_represented = any("over-represented" in x for x in reasons)
        unproductive = (any(x.startswith("failure_rate") for x in reasons)
                       and r["num_questions"] >= ban_min_evidence)
        if over_represented or unproductive:
            fresh_bans.append(d)
            ban_reasons[d] = "; ".join(reasons)

    # base bans are permanent; dynamic ones accumulate, oldest dropped when over the cap
    base_keys = {_norm(b) for b in base_banned}
    dynamic = _dedup([b for b in prev_banned if _norm(b) not in base_keys] + fresh_bans)
    keep = max(0, max_banned - len(base_banned))
    cut = max(0, len(dynamic) - keep)
    dropped, dynamic = dynamic[:cut], dynamic[cut:]
    banned = _dedup(list(base_banned) + dynamic)

    return examples, banned, {
        "num_focus_strategies": len(feedback["focus_strategies"]),
        "num_example_strategies": len(examples),
        "num_banned_strategies": len(banned),
        "fresh_bans": fresh_bans,
        "ban_reasons": ban_reasons,
        "bans_dropped_at_cap": dropped,
    }


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------


def split_by_prompt(seeds: list, rng: random.Random, p_explore: float) -> dict:
    """Independent coin flip per seed -> {'explore': [...], 'exploit': [...]}."""
    out = {"explore": [], "exploit": []}
    for s in seeds:
        out["explore" if rng.random() < p_explore else "exploit"].append(s)
    return out


def run_side(seeds: list, prompt: str, round_dir: Path, args,
             example_file: Path, banned_file: Path, few_shots_file: Path | None = None) -> dict:
    """Run the parallel driver for one prompt variant of one round. Returns its index.json."""
    out_dir = round_dir / prompt
    # JSON, not one-per-line: a seed containing newlines would otherwise fragment into
    # several seeds when the driver reads the file back.
    seeds_file = round_dir / f"{prompt}.seeds.json"
    round_dir.mkdir(parents=True, exist_ok=True)
    seeds_file.write_text(json.dumps(seeds, indent=2, ensure_ascii=False))

    cmd = [
        args.python, str(DRIVER),
        "--seeds-file", str(seeds_file),
        "--out-dir", str(out_dir),
        "--prompt", prompt,
        "--concurrency", str(args.concurrency),
        "--max-attempts", str(args.max_attempts),
        "--model", args.model,
        "--server-url", args.server_url,
        "--strategies-file", str(example_file),
        "--banned-strategies-file", str(banned_file),
    ]
    if few_shots_file:
        cmd += ["--few-shots-file", str(few_shots_file)]
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.timeout:
        cmd += ["--timeout", str(args.timeout)]
    if not args.skip_existing:
        cmd += ["--no-skip-existing"]
    if args.verify_criterion:
        cmd += ["--verify-criterion",
                "--verify-n-papers", str(args.verify_n_papers),
                "--verify-max-chars-per-paper", str(args.verify_max_chars_per_paper),
                "--reranker", args.reranker]
        if args.reranker_url:
            cmd += ["--reranker-url", args.reranker_url]

    print(f"\n--- {round_dir.name}/{prompt}: {len(seeds)} seed(s) ---", flush=True)
    proc = subprocess.run(cmd)
    index_path = out_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {"samples": [], "returncode": proc.returncode}


def round_stats(indexes: list) -> dict:
    """Roll per-prompt index.json files up into one round summary."""
    from collections import Counter
    rows = [r for ix in indexes for r in (ix.get("samples") or [])]
    counts = Counter(r.get("status", "UNKNOWN") for r in rows)
    return {
        "num_seeds": len(rows),
        "statuses": dict(sorted(counts.items())),
        "num_failed_found": counts.get("FAILED_FOUND", 0),
        "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 4),
        "claude_calls": sum(r.get("claude_calls", 0) for r in rows),
    }


def compute_feedback(run_dirs: list, round_dir: Path, example_file: Path, args) -> dict:
    """Cluster + score everything in `run_dirs`, writing feedback.json / feedback.txt."""
    fb_path = round_dir / "feedback.json"
    if fb_path.exists() and not args.refresh_feedback:
        print(f"[feedback] reusing {fb_path}", flush=True)
        cached = json.loads(fb_path.read_text())
        # already paid for in an earlier invocation: keep it out of this run's spend
        (cached.get("meta", {}).get("clustering") or {}).pop("cost_usd", None)
        return cached

    examples = load_examples_from_runs(run_dirs)
    if not examples:
        print("[feedback] no graded questions yet — keeping the current menus",
              file=sys.stderr, flush=True)
        return {}

    # Cluster against the menu this round actually ran with, so cluster ids line up with
    # what the generator was shown rather than with a stale built-in list.
    feedback = build_feedback(
        examples,
        RP.load_strategy_file(example_file),
        cluster_model=args.cluster_model,
        min_failure_rate=args.min_failure_rate,
        max_share=args.max_share,
        min_cluster_size=args.min_cluster_size,
        rank_by=args.rank_by,
        examples_per_strategy=args.examples_per_strategy,
    )
    usage = (feedback.get("meta", {}).get("clustering") or {}).get("usage") or {}
    model = (feedback.get("meta", {}).get("clustering") or {}).get("cluster_model")
    if usage and model:
        # strategy_feedback_module reports tokens only; pricing lives here.
        cost, _ = RP.price_call(model, dict(usage))
        feedback["meta"]["clustering"]["cost_usd"] = round(cost, 6)
    write_feedback(feedback, fb_path)
    summary = format_summary(feedback)
    (round_dir / "feedback.txt").write_text(summary + "\n")
    print(summary, flush=True)
    return feedback


def run_verification(out_dir: Path, args, remaining_usd: float | None = None) -> dict:
    """Post-filter the harvest with verify_questions.py. Returns its totals."""
    out = out_dir / "verified.json"
    cmd = [args.python, str(VERIFIER), str(out_dir), "--out", str(out),
           "--concurrency", str(args.concurrency), "--model", args.model,
           "--verify-n-papers", str(args.verify_n_papers),
           "--verify-max-chars-per-paper", str(args.verify_max_chars_per_paper),
           "--reranker", args.reranker]
    if remaining_usd is not None:
        cmd += ["--budget-usd", f"{remaining_usd:.4f}"]
    if args.reranker_url:
        cmd += ["--reranker-url", args.reranker_url]

    print(f"\n{'=' * 70}\nVERIFY: re-checking the harvest's criteria\n{'=' * 70}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0 or not out.exists():
        print(f"[verify] verification failed (rc={proc.returncode}); "
              f"the rounds themselves are unaffected", file=sys.stderr)
        return {}
    return {"report": str(out), "kept_set": str(out.with_suffix(".kept.json")),
            **json.loads(out.read_text()).get("totals", {})}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\nEach round:")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("seeds", nargs="*", help="Seed questions (default: the HF dataset).")
    p.add_argument("--out-dir", required=True, help="Folder for round dirs + loop.json.")
    p.add_argument("--total", "-m", type=int, default=50,
                   help="M: total questions (seeds) to generate across all rounds (default: 50).")
    p.add_argument("--feedback-every", "-n", type=int, default=10,
                   help="N: questions per round; feedback runs after each (default: 10).")
    p.add_argument("--prompt-mix", type=float, default=0.5,
                   help="P(explore) per seed; 0.5 = 50/50 explore vs exploit (default: 0.5).")
    p.add_argument("--random-seed", type=int, default=0,
                   help="RNG seed for the per-seed prompt coin flips (default: 0).")
    p.add_argument("--feedback-scope", choices=["all", "last"], default="all",
                   help="Score every round so far ('all', default) or only the round just "
                        "finished ('last'). 'all' gives stabler rates; 'last' reacts faster.")
    p.add_argument("--budget-usd", type=float, default=0.0,
                   help="Hard spend cap for the whole loop, generation + verification "
                        "(0 = no limit). Checked after each prompt side finishes, so the "
                        "actual spend can overshoot by up to one side of one round.")
    p.add_argument("--refresh-feedback", action="store_true",
                   help="Recompute a round's feedback even if feedback.json exists.")

    src = p.add_argument_group("seed source")
    src.add_argument("--seeds-file",
                     help="A .txt file with one seed per line, or a .json list of seeds.")
    src.add_argument("--start", type=int, default=0,
                    help="Skip the first N HF seeds before taking --total (default: 0).")

    men = p.add_argument_group("starting menus (round 0)")
    men.add_argument("--strategies", default=RP.DEFAULT_STRATEGIES,
                     choices=sorted(RP.STRATEGY_LISTS),
                     help=f"Example-strategy menu round 0 starts from "
                          f"(default: {RP.DEFAULT_STRATEGIES}).")
    men.add_argument("--banned-strategies", default=RP.DEFAULT_BANNED_STRATEGIES,
                     choices=sorted(RP.BANNED_STRATEGY_LISTS),
                     help=f"Banned-strategy menu round 0 starts from, and whose entries stay "
                          f"banned in every round (default: {RP.DEFAULT_BANNED_STRATEGIES}).")
    men.add_argument("--max-example-strategies", type=int, default=10,
                     help="Cap on the derived EXAMPLE STRATEGIES menu (default: 10).")
    men.add_argument("--max-banned-strategies", type=int, default=12,
                     help="Cap on the derived STRATEGIES TO NOT USE list, base bans included "
                          "(default: 12).")
    men.add_argument("--ban-min-evidence", type=int, default=3,
                     help="Questions a cluster needs before a low failure rate gets it banned "
                          "(default: 3).")
    men.add_argument("--few-shots-per-round", type=int, default=3,
                     help="Worked examples shown in each prompt (default: 3). Rounds 1+ "
                          "sample them from the run: 'exploit' from the failures of its "
                          "example-menu strategies, 'explore' from the failures of the "
                          "banned ones. Short samples are padded with the built-in three.")
    men.add_argument("--static-few-shots", action="store_true",
                     help="Keep the built-in few-shot examples in every round instead of "
                          "re-sampling them from the feedback.")
    men.add_argument("--no-open-ended-tail", dest="open_ended_tail", action="store_false",
                     help="Drop the trailing 'something else you think of' line from the "
                          "derived example menu.")
    p.set_defaults(open_ended_tail=True)

    fbk = p.add_argument_group("feedback (see strategy_feedback_module.py)")
    fbk.add_argument("--cluster-model", default=None,
                     help="Model for strategy clustering (default: the module's).")
    fbk.add_argument("--min-failure-rate", type=float, default=0.25)
    fbk.add_argument("--max-share", type=float, default=0.5)
    fbk.add_argument("--min-cluster-size", type=int, default=0)
    fbk.add_argument("--rank-by", default="underrepresented",
                     choices=["underrepresented", "diverse", "failure_rate", "volume"])
    fbk.add_argument("--examples-per-strategy", type=int, default=5)

    gen = p.add_argument_group("generation (forwarded to the pipeline)")
    gen.add_argument("--concurrency", type=int, default=5)
    gen.add_argument("--max-attempts", type=int, default=5)
    gen.add_argument("--model", default="claude-sonnet-4-5")
    gen.add_argument("--profile", help="Answering-system profile (e.g. drtulu, tongyi).")
    gen.add_argument("--server-url", default="http://localhost:8007/ask")
    gen.add_argument("--timeout", type=float)
    gen.add_argument("--verify-after", action="store_true",
                     help="After the last round, re-check every harvested question's "
                          "criterion with verify_questions.py and write the filtered set "
                          "to <out-dir>/verified.kept.json. Needs S2_API_KEY.")
    gen.add_argument("--verify-criterion", action="store_true",
                     help="INLINE verification during generation: checks the criterion "
                          "before each research call and kills the seed if it is rejected. "
                          "Expensive and lossy; prefer --verify-after.")
    gen.add_argument("--verify-n-papers", type=int, default=15)
    gen.add_argument("--verify-max-chars-per-paper", type=int, default=4000)
    gen.add_argument("--reranker", default="auto", choices=["auto", "none", "vllm"])
    gen.add_argument("--reranker-url", default=None)
    gen.add_argument("--python", default=sys.executable)
    gen.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                     help="Re-run seeds even if their sample_NNN.json exists.")
    p.set_defaults(skip_existing=True)
    args = p.parse_args()

    # Fail here rather than in every subprocess: without a key each seed dies on argparse
    # and the whole round comes back SUBPROCESS_FAILED with the reason buried in a
    # per-seed console capture.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        p.error("ANTHROPIC_API_KEY is not set (generation and clustering both need it).")
    if args.verify_after and not os.environ.get("S2_API_KEY"):
        print("[verify] WARNING: S2_API_KEY is not set; the post-hoc criterion check will "
              "be rate limited hard by Semantic Scholar.", file=sys.stderr)
    if args.verify_after and not (args.reranker_url or os.environ.get("VLLM_RERANK_URL")) \
            and args.reranker != "none":
        print("[verify] WARNING: no --reranker-url / VLLM_RERANK_URL; retrieval will run "
              "WITHOUT reranking.", file=sys.stderr)

    if args.feedback_every < 1:
        p.error("--feedback-every must be >= 1")
    if not 0.0 <= args.prompt_mix <= 1.0:
        p.error("--prompt-mix must be between 0 and 1")

    seeds = load_seeds(SimpleNamespace(seeds=args.seeds, seeds_file=args.seeds_file,
                                       start=args.start, limit=args.total))[:args.total]
    if not seeds:
        p.error("No seed questions provided.")

    # filter_queries.py screens pasted documents out of the dataset, so this is only a
    # backstop for hand-supplied --seeds-file input. Thresholds are deliberately far above
    # a long-but-real question (the dataset has legitimate 1900-char, 3-newline seeds).
    for i, seed in enumerate(seeds):
        if len(seed) > 5000 or seed.count("\n") >= 5:
            print(f"[seeds] WARNING seed {i} looks like a pasted document, not a question "
                  f"({len(seed)} chars, {seed.count(chr(10))} newlines): "
                  f"{seed.splitlines()[0][:80]}...", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = [seeds[i:i + args.feedback_every]
              for i in range(0, len(seeds), args.feedback_every)]
    base_banned = list(RP.BANNED_STRATEGY_LISTS[args.banned_strategies])
    example_menu = list(RP.STRATEGY_LISTS[args.strategies])
    banned_menu = list(base_banned)
    # Round 0 runs on the built-ins, but they are still written per round and passed
    # explicitly, so every round's artifacts are uniform and diffable.
    few_shots = {"explore": list(RP.DEFAULT_FEW_SHOTS), "exploit": list(RP.DEFAULT_FEW_SHOTS)}
    rng = random.Random(args.random_seed)

    print(f"{len(seeds)} question(s) in {len(rounds)} round(s) of "
          f"<= {args.feedback_every}, P(explore)={args.prompt_mix} -> {out_dir}/")

    manifest = {"total": len(seeds), "budget_usd": args.budget_usd or None,
                "feedback_every": args.feedback_every,
                "prompt_mix": args.prompt_mix, "random_seed": args.random_seed,
                "feedback_scope": args.feedback_scope, "model": args.model,
                "base_banned_strategies": base_banned, "rounds": []}
    round_dirs: list[Path] = []
    stopped_early = ""
    spent = 0.0          # running total, updated as each prompt side finishes

    for k, round_seeds in enumerate(rounds):
        round_dir = out_dir / f"round_{k:02d}"
        round_dirs.append(round_dir)
        example_file = write_menu(round_dir / "example_strategies.txt", example_menu,
                                  f"round {k}: EXAMPLE STRATEGIES TO CONSIDER "
                                  f"('exploit' prompt)")
        banned_file = write_menu(round_dir / "banned_strategies.txt", banned_menu,
                                 f"round {k}: STRATEGIES TO NOT USE ('explore' prompt)")

        split = split_by_prompt(round_seeds, rng, args.prompt_mix)
        print(f"\n{'=' * 70}\nROUND {k}: {len(round_seeds)} seed(s) — "
              f"{len(split['explore'])} explore / {len(split['exploit'])} exploit\n"
              f"  menus: {len(example_menu)} example, {len(banned_menu)} banned\n{'=' * 70}")

        shot_files = {}
        for prompt, shots in few_shots.items():
            if shots:
                path = round_dir / f"few_shots.{prompt}.json"
                path.write_text(json.dumps(shots, indent=2, ensure_ascii=False))
                shot_files[prompt] = path

        indexes = []
        for prompt, side_seeds in split.items():
            if not side_seeds:
                continue
            if args.budget_usd and spent >= args.budget_usd:
                stopped_early = (f"budget ${args.budget_usd:.2f} reached (${spent:.4f} spent) "
                                 f"before round {k}/{prompt}")
                break
            ix = run_side(side_seeds, prompt, round_dir, args,
                          example_file, banned_file, shot_files.get(prompt))
            indexes.append(ix)
            spent += round_stats([ix])["cost_usd"]
        stats = round_stats(indexes)
        print(f"\n[round {k}] {stats['num_failed_found']}/{stats['num_seeds']} FAILED_FOUND, "
              f"${stats['cost_usd']:.4f}, statuses {stats['statuses']}")

        record = {"round": k, "dir": str(round_dir),
                  "num_explore": len(split["explore"]),
                  "num_exploit": len(split["exploit"]),
                  "example_strategies": example_menu, "banned_strategies": banned_menu,
                  "few_shot_seeds": {k: [x["seed_question"] for x in v]
                                     for k, v in few_shots.items() if v},
                  **stats}

        if k < len(rounds) - 1:
            scope = round_dirs if args.feedback_scope == "all" else [round_dir]
            feedback = compute_feedback(scope, round_dir, example_file, args)
            if feedback:
                clustering_cost = ((feedback.get("meta", {}).get("clustering") or {})
                                   .get("cost_usd") or 0.0)
                spent += clustering_cost
                record["clustering_cost_usd"] = round(clustering_cost, 4)
                example_menu, banned_menu, provenance = derive_menus(
                    feedback, banned_menu,
                    base_banned=base_banned,
                    max_examples=args.max_example_strategies,
                    max_banned=args.max_banned_strategies,
                    ban_min_evidence=args.ban_min_evidence,
                    keep_open_ended=args.open_ended_tail,
                )
                if not args.static_few_shots:
                    few_shots["exploit"], few_shots["explore"] = derive_few_shots(
                        feedback, example_menu, banned_menu,
                        defaults=RP.DEFAULT_FEW_SHOTS, n=args.few_shots_per_round,
                    )
                record["feedback"] = str(round_dir / "feedback.json")
                record["feedback_scope_dirs"] = [str(d) for d in scope]
                record["next_menus"] = provenance
                print(f"[round {k}] next menus: {provenance['num_example_strategies']} example, "
                      f"{provenance['num_banned_strategies']} banned "
                      f"({len(provenance['fresh_bans'])} newly banned)")
                if not args.static_few_shots:
                    for prompt, shots in few_shots.items():
                        derived = sum(1 for x in shots if x not in RP.DEFAULT_FEW_SHOTS)
                        print(f"[round {k}] next {prompt} few-shots: {derived}/{len(shots)} "
                              f"from this run")

        if not indexes:
            record["skipped"] = stopped_early or "no seeds ran"
        manifest["rounds"].append(record)
        if stopped_early:
            manifest["stopped_early"] = stopped_early
        (out_dir / "loop.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        if stopped_early:
            ran = sum(1 for r in manifest["rounds"] if not r.get("skipped"))
            print(f"\n[budget] {stopped_early} — stopping after {ran} round(s), "
                  f"${spent:.4f} spent", file=sys.stderr)
            break

    if args.verify_after and not stopped_early:
        remaining = (args.budget_usd - spent) if args.budget_usd else None
        if remaining is not None and remaining <= 0:
            print(f"[budget] ${spent:.4f} of ${args.budget_usd:.2f} spent on generation — "
                  f"skipping verification. Raise --budget-usd and run verify_questions.py "
                  f"on this dir to pick it up.", file=sys.stderr)
            manifest["verification_skipped"] = "budget exhausted"
            verified = {}
        else:
            verified = run_verification(out_dir, args, remaining)
        if verified:
            manifest["verification"] = verified
            (out_dir / "loop.json").write_text(json.dumps(manifest, indent=2,
                                                          ensure_ascii=False))

    manifest["total_cost_usd"] = round(spent + (manifest.get("verification") or {})
                                       .get("cost_usd", 0.0), 4)
    (out_dir / "loop.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    total_cost = sum(r["cost_usd"] for r in manifest["rounds"])
    total_ff = sum(r["num_failed_found"] for r in manifest["rounds"])
    total_q = sum(r["num_seeds"] for r in manifest["rounds"])
    print(f"\n{'#' * 70}\nLOOP SUMMARY\n{'#' * 70}")
    for r in manifest["rounds"]:
        rate = (r["num_failed_found"] / r["num_seeds"]) if r["num_seeds"] else 0.0
        clu = r.get("clustering_cost_usd") or 0.0
        print(f"  round {r['round']:>2}  {r['num_failed_found']:>3}/{r['num_seeds']:<3} "
              f"FAILED_FOUND ({rate:.0%})  ${r['cost_usd']:.4f} gen"
              + (f" + ${clu:.4f} cluster" if clu else "")
              + f"  [{r['num_explore']}e/{r['num_exploit']}x]")
    total_clu = sum(r.get("clustering_cost_usd") or 0.0 for r in manifest["rounds"])
    print(f"\nTotal: {total_ff}/{total_q} FAILED_FOUND, ${total_cost:.4f} generation"
          + (f" + ${total_clu:.4f} clustering" if total_clu else ""))
    v = manifest.get("verification")
    if v:
        print(f"Verified: {v['kept']}/{v['checked']} criteria upheld "
              f"({v['keep_rate']:.0%}, +${v['cost_usd']:.4f}) -> {v['kept_set']}")
    print(f"Manifest: {out_dir / 'loop.json'}")


if __name__ == "__main__":
    main()
