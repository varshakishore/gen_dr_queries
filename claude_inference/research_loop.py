#!/usr/bin/env python3
"""
Outer loop: generate M questions in rounds of N, re-deriving the strategy menus from
strategy feedback after every round.

    seeds[0:N]   ->  generate  ->  strategy feedback  ->  new menus
    seeds[N:2N]  ->  generate  ->  strategy feedback  ->  new menus
    ...

Each round:

  1. SPLIT   the round's seeds between the two make-harder prompts by a balanced split
             (--prompt-mix, default 0.5 -> an even explore/exploit split each round, with
             which seeds go where randomised) -- or, with --both-prompts, run every seed
             through BOTH prompts for a paired comparison -- and run
             research_pipeline_parallel.py once per side into
             <out-dir>/round_KK/explore/ and <out-dir>/round_KK/exploit/.
             Separate dirs keep the `source_run` labels that make the per-prompt comparison
             in the feedback work; strategy_feedback_module globs `*/sample_*.json` under a
             round dir, so both sides are picked up by passing the round dir alone.

  2. FEED    every round dir so far through strategy_feedback_module.build_feedback --
     BACK    always cumulatively, since the quota below is a lifetime count -- and rewrite
             the two menus from its `strategy_clusters`:

               EXAMPLE STRATEGIES TO CONSIDER  <- a fresh sample PER SEED
                   --max-example-strategies drawn without replacement from every cluster
                   not over the quota, P(pick) monotone in a smoothed failure rate times
                   (1 - share): what works, favouring what the generator under-uses.
                   Sampling rather than taking the top N keeps a mid-ranked cluster from
                   starving -- it would otherwise never be tried again, never accumulate
                   evidence, and never climb. Drawn per SEED rather than per round
                   (--menu-sample), so one round exercises the whole pool: measured over 35
                   seeds with a pool of 11 and a cap of 6, every strategy appeared in some
                   seed's menu, at rates from 80% down to 20% by weight. Written to
                   round_KK/menus/sample_NNN.txt and injected into
                   PROMPT_TO_MAKE_HARDER_QUESTION_EXPLOIT (the 'exploit' prompt).

               STRATEGIES TO NOT USE           <- the harvest quota
                   the base bans, plus every cluster that has yielded more than
                   --ban-after-failures system-breaking questions: we have enough
                   questions of that type, so retire it and push the generator elsewhere.
                   Injected into PROMPT_TO_MAKE_HARDER_QUESTION_EXPLORE ('explore').

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

             The ban list is REBUILT from scratch every round, not accumulated. It can be,
             because the quota re-derives itself: num_failed only grows, so a retired
             strategy is still over the quota next round. That avoids storing ban strings,
             which would silently lapse -- novel cluster descriptions are LLM-written and
             churn completely between rounds (measured: 0 of 5 survive a round).

             The three strategies the prompt's own few-shot examples demonstrate stay
             banned for 'explore' in every round, but are NOT withheld from 'exploit' --
             that reason is specific to explore, and exploit exists to work known-good
             strategies. A base ban leaves the exploit pool only by crossing the quota.

             Feedback is clustered against cluster_seeds.txt (menu + bans), not the example
             menu, so a retired strategy keeps a stable seed.N identity instead of being
             rediscovered as a fresh new.N each round -- which is what keeps the quota able
             to see it at all.

Round 0 starts from the built-in menus (--strategies / --banned-strategies). The last
round is not followed by a feedback call -- there is nothing left to feed it to.

  3. VERIFY  (--verify-after) once every round is done, verify_questions.py re-checks the
             harvest: for each question that broke the answering system, retrieve S2 papers
             and ask whether its verification criterion is itself correct, dropping the ones
             that are not. This is the same check as the pipeline's inline --verify-criterion
             but paid only on questions worth keeping, and it cannot end a seed mid-run.
             Writes <out-dir>/verified.json (+ .kept.json, the filtered set).

Each round dir holds the menus it was actually run with (menus/sample_NNN.txt, one
per exploit seed; example_strategy_pool.txt, what they were drawn from;
example_strategies.txt, the single round-wide draw used by --menu-sample round and as a
fallback; banned_strategies.txt; cluster_seeds.txt), the worked examples each prompt was
shown
(few_shots.exploit.json
/ few_shots.explore.json; round 0's are the built-in three), the seed split
(explore.seeds.json / exploit.seeds.json), and the
feedback it produced (feedback.json + feedback.txt). <out-dir>/loop.json is the manifest.

Resumability: the driver skips seeds whose sample_NNN.json exists, and a round's feedback
is reused if feedback.json is already there (--refresh-feedback recomputes it), so
re-running the same command continues an interrupted loop without re-paying for either.

Examples:
  # 50 questions in rounds of 10, feedback 4 times, 10 seeds in flight
  python research_loop.py --out-dir runs/loop1 --total 50 --feedback-every 10 --concurrency 10

  # only explore, retiring a strategy after it has yielded 10 hard questions
  python research_loop.py --out-dir runs/loop2 --total 40 --feedback-every 10 \
      --prompt-mix 1.0 --ban-after-failures 10

Requires the API key for whichever provider serves --model / --cluster-model
(ANTHROPIC_API_KEY or OPENAI_API_KEY) and a research server on --server-url.
"""

import argparse
import datetime as dt
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import llm_client
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


def _hms(seconds: float) -> str:
    """Elapsed seconds as a compact h/m/s string."""
    s = int(round(seconds))
    if s < 60:
        return f"{seconds:.1f}s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return (f"{h}h {m}m {sec}s" if h else f"{m}m {sec}s")


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
        "required_reasoning_process": rec.get("required_reasoning_process") or [],
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


def derive_few_shots(feedback: dict, example_strategies: list, banned_menu: list, *,
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
    # `example_strategies` is the whole exploit POOL, not one seed's draw: any pool entry
    # may be the strategy a given seed is shown, so all of them are fair game to demo.
    menu_keys = {_norm(e) for e in example_strategies}
    focus_buckets = [
        [_shot(x) for x in (f.get("few_shot_failures") or [])]
        for f in feedback["strategy_clusters"]
        if _norm(_clean(f["description"])) in menu_keys
    ]

    # The banned clusters' failures come from cluster_comparison rather than from the
    # ban list, which carries no examples; biggest failure counts first, as the clearest
    # demonstrations. NOTE these are now the strategies that hit the harvest quota, i.e.
    # the BEST performers -- explore is shown them as negative examples, so watch for it
    # imitating them instead of avoiding them.
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


def _smoothed_weight(row: dict, total: int) -> float:
    """Sampling weight for one cluster: smoothed failure rate, damped by its share.

    Raw failure_rate is noise at small n -- a 1/1 cluster scores 1.00 and outranks a 20/23
    one -- and a never-tried cluster scores 0.0, which is an absorbing state: weight 0 means
    it is never sampled, so it stays untried forever. The Beta(1,1) posterior mean
    (f + 1) / (q + 2) fixes both ends: 1/1 -> 0.67, 20/23 -> 0.84, 0/0 -> 0.50, a real
    "unknown" prior. The (1 - share) factor throttles ruts continuously -- sampling a
    strategy raises its share and so lowers its own weight next round.
    """
    q = row.get("num_questions", 0) or 0
    f = row.get("num_failed", 0) or 0
    share = q / total if total else 0.0
    return max(((f + 1) / (q + 2)) * (1.0 - share), 1e-9)


def _weighted_sample(rows: list, k: int, rng: random.Random) -> list:
    """`k` rows sampled without replacement, P(pick) monotone in row['weight'].

    Efraimidis-Spirakis A-Res: key each row with u ** (1 / w) and keep the k largest. One
    pass, no renormalising as rows are drawn. Note that drawing k > 1 without replacement
    compresses the inclusion probabilities toward k/n -- higher weight is always likelier,
    but the ratio between two weights is not preserved.
    """
    keyed = [(rng.random() ** (1.0 / r["weight"]), r) for r in rows]
    keyed.sort(key=lambda kr: kr[0], reverse=True)
    return [r for _, r in keyed[:k]]


def sample_menu(pool: list, cap: int, rng: random.Random,
                keep_open_ended: bool = True) -> list:
    """One EXAMPLE STRATEGIES menu: `cap` strategies drawn from `pool` by weight.

    Called once per seed, so each seed is shown its own draw and the pool gets exercised
    across a round instead of one menu being frozen for all of it. A strategy left out of
    one seed's menu is very likely in another's, which is what makes a low-weight strategy
    get tried somewhere rather than nowhere.
    """
    picked = _weighted_sample(pool, cap, rng)
    picked.sort(key=lambda r: r.get("rank") or 0)        # present the menu in ranked order
    menu = _dedup([r["strategy"] for r in picked])
    if keep_open_ended:
        menu = _dedup(menu + [OPEN_ENDED_TAIL])
    return menu


def derive_menus(feedback: dict, *, base_banned: list, ban_after_failures: int,
                 keep_open_ended: bool = True) -> tuple[list, list, list, dict]:
    """Turn one feedback dict into the next round's ban list and strategy pool. Returns
    `(pool, banned_menu, cluster_seeds, provenance)`.

    `pool` is every strategy eligible for the exploit menu, each with the sampling weight
    `sample_menu` draws by -- this function selects nothing itself, it just decides what is
    retired and what remains available.

    The ban list is a HARVEST QUOTA, recomputed from scratch every round: a strategy is
    banned once it has yielded more than `ban_after_failures` system-breaking questions --
    we have enough questions of that type, so retire it and push the generator elsewhere.
    Recomputing rather than accumulating is what makes this safe: `num_failed` only ever
    grows (feedback is always scored over every round so far), so a quota ban re-derives
    identically next round without being stored anywhere. That matters because novel
    cluster names churn completely between rounds -- measured at 0 of 5 novel descriptions
    surviving a round -- so a stored ban string would stop matching and silently lapse.

    The base bans stay banned for `explore` every round (they are the strategies its own
    few-shot examples demonstrate) but are NOT withheld from `exploit`: that reason is
    specific to explore, and exploit exists to work known-good strategies. A base ban
    leaves the exploit pool only if it crosses the quota like any other cluster.

    `cluster_seeds` is what the NEXT round is clustered against, and is deliberately not the
    example menu: it keeps every strategy, bans included, so a retired strategy stays a
    stable `seed.N` with a fixed description instead of being re-discovered as a fresh
    `new.N` under a new name -- and novel clusters are where the churn lives.
    """
    total = (feedback.get("meta") or {}).get("num_instances", 0)
    rows = []
    for c in feedback.get("strategy_clusters") or []:
        d = _clean(c.get("description", ""))
        if not d:
            continue
        rows.append({**c, "_strategy": d, "_weight": _smoothed_weight(c, total)})

    over_quota = [r for r in rows if (r.get("num_failed") or 0) > ban_after_failures]
    quota_keys = {_norm(r["_strategy"]) for r in over_quota}
    banned = _dedup(list(base_banned) + [r["_strategy"] for r in over_quota])

    # exploit draws from everything not retired -- base bans included, since those are
    # banned only for explore. The open-ended tail is a fixed menu line, not a draw.
    tail_key = _norm(OPEN_ENDED_TAIL)
    pool = [{"strategy": r["_strategy"], "weight": r["_weight"], "rank": r.get("rank"),
             "num_failed": r.get("num_failed"), "num_questions": r.get("num_questions"),
             "failure_rate": r.get("failure_rate")}
            for r in rows
            if _norm(r["_strategy"]) not in quota_keys
            and _norm(r["_strategy"]) != tail_key]

    cluster_seeds = _dedup([r["_strategy"] for r in rows] + list(base_banned)
                           + ([OPEN_ENDED_TAIL] if keep_open_ended else []))

    base_keys = {_norm(b) for b in base_banned}
    return pool, banned, cluster_seeds, {
        "num_clusters": len(rows),
        "num_banned_strategies": len(banned),
        "exploit_pool_size": len(pool),
        "ban_after_failures": ban_after_failures,
        # recomputed each round, so the full set is recorded -- a quota ban that flickers
        # (its cluster dissolved and re-formed below the quota) is invisible otherwise.
        "quota_bans": [{"strategy": r["_strategy"], "num_failed": r.get("num_failed"),
                        "num_questions": r.get("num_questions"),
                        "failure_rate": r.get("failure_rate")} for r in over_quota],
        "base_bans_over_quota": [r["_strategy"] for r in over_quota
                                 if _norm(r["_strategy"]) in base_keys],
        "pool": [{"strategy": r["strategy"], "weight": round(r["weight"], 4),
                   "num_failed": r["num_failed"], "num_questions": r["num_questions"]}
                  for r in pool],
    }


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------


def split_by_prompt(seeds: list, rng: random.Random, p_explore: float) -> dict:
    """Split a round's seeds -> {'explore': [...], 'exploit': [...]}.

    Balanced, not an independent coin flip per seed: `p_explore` sets the SHARE of the
    round that goes to explore, and which seeds go where is randomised. An independent
    flip puts both seeds on one side half the time at 2 seeds/round, and that round then
    yields no evidence at all for the other prompt -- its by_source_run block is empty and
    the per-prompt comparison for that round is gone. Shares of 0.0 / 1.0 still send
    everything one way.
    """
    shuffled = list(seeds)
    rng.shuffle(shuffled)
    n_explore = int(round(len(shuffled) * p_explore))
    if seeds and 0.0 < p_explore < 1.0:      # never starve a side by rounding
        n_explore = min(max(n_explore, 1), len(shuffled) - 1) if len(shuffled) > 1 else n_explore
    return {"explore": shuffled[:n_explore], "exploit": shuffled[n_explore:]}


def run_side(seeds: list, prompt: str, round_dir: Path, args,
             example_file: Path, banned_file: Path, few_shots_file: Path | None = None,
             strategies_dir: Path | None = None) -> dict:
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
    cmd += llm_client.forward_provider_args(args)
    if strategies_dir:
        cmd += ["--strategies-dir", str(strategies_dir)]
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


def write_seed_menus(round_dir: Path, n_seeds: int, pool: list, args,
                     rng: random.Random) -> tuple[Path, dict]:
    """One sampled menu per exploit seed, named to match the driver's sample_NNN numbering.

    The driver indexes its seeds from 1, so seed i gets menus/sample_{i:03d}.txt. Returns
    the directory plus how many of the round's menus each strategy landed in -- the compact
    audit trail, since recording every menu in loop.json would swamp it (the files stay on
    disk for the full detail).
    """
    menu_dir = round_dir / "menus"
    menu_dir.mkdir(parents=True, exist_ok=True)
    counts: dict = {}
    for i in range(1, n_seeds + 1):
        menu = sample_menu(pool, args.max_example_strategies, rng,
                           keep_open_ended=args.open_ended_tail)
        write_menu(menu_dir / f"sample_{i:03d}.txt", menu,
                   f"exploit seed {i}: EXAMPLE STRATEGIES TO CONSIDER (per-seed sample)")
        for m in menu:
            counts[m] = counts.get(m, 0) + 1
    return menu_dir, counts


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


def compute_feedback(run_dirs: list, round_dir: Path, cluster_seed_file: Path, args) -> dict:
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

    # Cluster against every strategy in play, NOT the example menu -- a strategy retired
    # by the quota is off the menu but must stay a stable seed.N here, or its questions
    # re-cluster as a fresh new.N under a new LLM-written name every round and the quota
    # can no longer see them. See derive_menus.
    feedback = build_feedback(
        examples,
        RP.load_strategy_file(cluster_seed_file),
        cluster_provider=cluster_provider(args),
        cluster_model=args.cluster_model,
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


def cluster_provider(args) -> str:
    """Which provider clusters strategies.

    'auto' follows the model ids already chosen: --cluster-model if one was given, else
    --model, so `--model gpt-5.6-terra` moves clustering across with generation rather
    than leaving the loop half on each provider.
    """
    if args.cluster_provider != "auto":
        return args.cluster_provider
    return llm_client.resolve_provider(args.cluster_model or args.model)


def run_verification(out_dir: Path, args, remaining_usd: float | None = None) -> dict:
    """Post-filter the harvest with verify_questions.py. Returns its totals."""
    out = out_dir / "verified.json"
    cmd = [args.python, str(VERIFIER), str(out_dir), "--out", str(out),
           "--concurrency", str(args.concurrency), "--model", args.model,
           "--verify-n-papers", str(args.verify_n_papers),
           "--verify-max-chars-per-paper", str(args.verify_max_chars_per_paper),
           "--reranker", args.reranker]
    cmd += llm_client.forward_provider_args(args)
    if remaining_usd is not None:
        cmd += ["--budget-usd", f"{remaining_usd:.4f}"]
    if args.reranker_url:
        cmd += ["--reranker-url", args.reranker_url]

    print(f"\n{'=' * 70}\nVERIFY: re-checking the harvest's criteria\n{'=' * 70}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0 or not out.exists():
        print(f"[verify] verification failed (rc={proc.returncode}); "
              f"the rounds themselves are unaffected", file=sys.stderr)
        return {}
    return {"report": str(out), "kept_set": str(out.with_suffix(".kept.json")),
            "elapsed_s": round(elapsed, 1),
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
    p.add_argument("--both-prompts", action="store_true",
                   help="Run EVERY seed through both prompts instead of splitting the round "
                        "between them: a paired explore-vs-exploit comparison on identical "
                        "seeds (2x the runs). --prompt-mix is ignored.")
    p.add_argument("--no-feedback", action="store_true",
                   help="One round, no clustering and no menu updates -- just generate with "
                        "the starting menus (plus --verify-after if given). Same as setting "
                        "--feedback-every to the seed count.")
    p.add_argument("--prompt-mix", type=float, default=0.5,
                   help="Share of each round's seeds given to the explore prompt; "
                        "0.5 = an even split (default: 0.5).")
    p.add_argument("--random-seed", type=int, default=0,
                   help="RNG seed for the explore/exploit assignment (default: 0).")
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
    men.add_argument("--max-example-strategies", type=int, default=6,
                     help="How many strategies each EXAMPLE STRATEGIES menu holds "
                          "(default: 6). Drawn without replacement from the un-retired "
                          "pool, weighted by smoothed failure rate x (1 - share). Set it "
                          "BELOW the pool size or the draw is a no-op -- every seed then "
                          "sees the whole pool and the weights do nothing.")
    men.add_argument("--menu-sample", choices=("seed", "round"), default="seed",
                     help="Draw a fresh example menu per SEED (default) or once per round. "
                          "Per-seed exercises the whole pool within a round, so a "
                          "low-weight strategy gets tried somewhere rather than nowhere; "
                          "per-round freezes one draw for every seed in the round.")
    men.add_argument("--ban-after-failures", type=int, default=15,
                     help="Harvest quota: retire a strategy once it has produced MORE than "
                          "this many system-breaking questions (default: 15). It is then "
                          "banned from 'explore' and dropped from the 'exploit' pool. The "
                          "count is cumulative over the whole loop, so this is a lifetime "
                          "quota per strategy, not a per-round rate.")
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
                     help="Model for strategy clustering (default: the clustering "
                          "provider's own default).")
    fbk.add_argument("--cluster-provider", choices=["auto", "anthropic", "openai"],
                     default="auto",
                     help="Provider for strategy clustering. 'auto' (default) follows "
                          "--cluster-model's id, or --model's when no cluster model is set, "
                          "so the whole loop moves providers together.")
    fbk.add_argument("--rank-by", default="underrepresented",
                     choices=["underrepresented", "diverse", "failure_rate", "volume"])
    fbk.add_argument("--examples-per-strategy", type=int, default=5)

    gen = p.add_argument_group("generation (forwarded to the pipeline)")
    gen.add_argument("--concurrency", type=int, default=5)
    gen.add_argument("--max-attempts", type=int, default=5)
    gen.add_argument("--model", default="claude-sonnet-4-5",
                     help="Generator/judge model for every round. Accepts a Claude id or an "
                          "OpenAI one (e.g. gpt-5.6-terra); provider inferred from the id.")
    gen.add_argument("--provider", choices=["auto", "anthropic", "openai"], default="auto",
                     help="Provider for --model (default: auto, inferred from the model id).")
    gen.add_argument("--reasoning-effort",
                     choices=["minimal", "low", "medium", "high"], default=None,
                     help="OpenAI models only: reasoning_effort for every generation call.")
    gen.add_argument("--decomposer-model", default=None,
                     help="Model for retrieve_papers' query decomposition during "
                          "verification; independent of --model.")
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
    llm_client.configure_from_args(args)
    for label, model in (("generation", args.model),
                         ("clustering", args.cluster_model or args.model)):
        if missing := llm_client.require_api_key(model):
            p.error(f"{missing} Needed for {label}.")
    if args.verify_criterion or args.verify_after:
        # Resolve what retrieval will really use -- it follows --model unless overridden.
        args.decomposer_model = RP.effective_decomposer_model(args.decomposer_model,
                                                             args.model)
        if args.decomposer_model and (
                missing := llm_client.require_api_key(args.decomposer_model)):
            p.error(missing + llm_client.decomposer_hint(args.model, args.decomposer_model))
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
    every = len(seeds) if args.no_feedback else args.feedback_every
    rounds = [seeds[i:i + every] for i in range(0, len(seeds), every)]
    base_banned = list(RP.BANNED_STRATEGY_LISTS[args.banned_strategies])
    # Round 0 has no feedback, so every built-in strategy carries the same weight -- the
    # 0.5 Beta prior an untried cluster gets -- making the first round's per-seed draws a
    # uniform subset rather than a ranked one.
    # The tail is excluded from the pool and appended to every menu instead, matching what
    # derive_menus does, so it never consumes one of the --max-example-strategies slots.
    pool = [{"strategy": t, "weight": 0.5, "rank": i, "num_failed": None,
             "num_questions": None, "failure_rate": None}
            for i, t in enumerate(RP.STRATEGY_LISTS[args.strategies], start=1)
            if _norm(t) != _norm(OPEN_ENDED_TAIL)]
    banned_menu = list(base_banned)
    # What feedback clusters against: every strategy in play, menu + bans. Kept separate
    # from the menu so a quota-retired strategy keeps a stable cluster identity.
    cluster_seeds = _dedup([r["strategy"] for r in pool] + base_banned)
    # Round 0 runs on the built-ins, but they are still written per round and passed
    # explicitly, so every round's artifacts are uniform and diffable.
    few_shots = {"explore": list(RP.DEFAULT_FEW_SHOTS), "exploit": list(RP.DEFAULT_FEW_SHOTS)}
    rng = random.Random(args.random_seed)
    # A separate stream for menu sampling: drawing from `rng` would shift the
    # explore/exploit split's sequence, so the split would stop being reproducible
    # against runs made before sampling existed.
    menu_rng = random.Random(args.random_seed + 1_000_003)

    print(f"{len(seeds)} seed(s) in {len(rounds)} round(s) of <= {every}, "
          + ("both prompts on every seed" if args.both_prompts
             else f"explore share {args.prompt_mix}")
          + (", no feedback" if len(rounds) == 1 else "")
          + f" -> {out_dir}/")

    t_start = time.perf_counter()
    manifest = {"total": len(seeds), "budget_usd": args.budget_usd or None,
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "feedback_every": every, "both_prompts": args.both_prompts,
                "prompt_mix": args.prompt_mix, "random_seed": args.random_seed,
                "ban_after_failures": args.ban_after_failures, "model": args.model,
                "base_banned_strategies": base_banned, "rounds": []}
    round_dirs: list[Path] = []
    stopped_early = ""
    spent = 0.0          # running total, updated as each prompt side finishes

    for k, round_seeds in enumerate(rounds):
        round_dir = out_dir / f"round_{k:02d}"
        round_dirs.append(round_dir)
        pool_strategies = [r["strategy"] for r in pool]
        # The pool is recorded for reference; it is never a prompt menu itself.
        write_menu(round_dir / "example_strategy_pool.txt", pool_strategies,
                   f"round {k}: the pool per-seed menus are sampled from")
        # One round-wide draw: what exploit gets under --menu-sample round, and the
        # fallback for any seed without its own menu file.
        example_file = write_menu(
            round_dir / "example_strategies.txt",
            sample_menu(pool, args.max_example_strategies, menu_rng,
                        keep_open_ended=args.open_ended_tail),
            f"round {k}: EXAMPLE STRATEGIES TO CONSIDER ('exploit' prompt, one draw)")
        banned_file = write_menu(round_dir / "banned_strategies.txt", banned_menu,
                                 f"round {k}: STRATEGIES TO NOT USE ('explore' prompt)")
        cluster_seed_file = write_menu(round_dir / "cluster_seeds.txt", cluster_seeds,
                                       f"round {k}: strategies this round's feedback is "
                                       f"clustered against (menu + bans)")

        t_round = time.perf_counter()
        split = ({"explore": list(round_seeds), "exploit": list(round_seeds)}
                 if args.both_prompts else split_by_prompt(round_seeds, rng, args.prompt_mix))
        print(f"\n{'=' * 70}\nROUND {k}: {len(round_seeds)} seed(s) — "
              f"{len(split['explore'])} explore / {len(split['exploit'])} exploit\n"
              f"  menus: {len(pool)} in the example pool"
              f" (sample {args.max_example_strategies}/seed"
              f"{' — pool <= cap, so every seed sees all of it' if len(pool) <= args.max_example_strategies else ''}"
              f"), {len(banned_menu)} banned\n{'=' * 70}")

        menu_dir, menu_counts = None, {}
        if args.menu_sample == "seed" and split["exploit"]:
            menu_dir, menu_counts = write_seed_menus(
                round_dir, len(split["exploit"]), pool, args, menu_rng)
            print(f"  sampled {len(split['exploit'])} per-seed menus -> {menu_dir}/")

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
                          example_file, banned_file, shot_files.get(prompt),
                          strategies_dir=menu_dir if prompt == "exploit" else None)
            indexes.append(ix)
            spent += round_stats([ix])["cost_usd"]
        stats = round_stats(indexes)
        gen_elapsed = time.perf_counter() - t_round
        print(f"\n[round {k}] {stats['num_failed_found']}/{stats['num_seeds']} FAILED_FOUND, "
              f"${stats['cost_usd']:.4f}, {_hms(gen_elapsed)}, statuses {stats['statuses']}")

        record = {"round": k, "dir": str(round_dir),
                  "num_explore": len(split["explore"]),
                  "num_exploit": len(split["exploit"]),
                  "example_strategy_pool": pool_strategies,
                  "banned_strategies": banned_menu,
                  "cluster_seeds": cluster_seeds,
                  "menu_sample": args.menu_sample,
                  # how many of this round's per-seed menus each strategy landed in
                  "menu_draw_counts": dict(sorted(menu_counts.items(),
                                                  key=lambda kv: -kv[1])),
                  "few_shot_seeds": {k: [x["seed_question"] for x in v]
                                     for k, v in few_shots.items() if v},
                  "generation_elapsed_s": round(gen_elapsed, 1),
                  **stats}

        if k < len(rounds) - 1:
            scope = round_dirs          # always cumulative: the quota is a lifetime count
            t_fb = time.perf_counter()
            feedback = compute_feedback(scope, round_dir, cluster_seed_file, args)
            record["feedback_elapsed_s"] = round(time.perf_counter() - t_fb, 1)
            if feedback:
                clustering_cost = ((feedback.get("meta", {}).get("clustering") or {})
                                   .get("cost_usd") or 0.0)
                spent += clustering_cost
                record["clustering_cost_usd"] = round(clustering_cost, 4)
                pool, banned_menu, cluster_seeds, provenance = derive_menus(
                    feedback,
                    base_banned=base_banned,
                    ban_after_failures=args.ban_after_failures,
                    keep_open_ended=args.open_ended_tail,
                )
                if not args.static_few_shots:
                    # exploit's examples demonstrate the POOL, not one seed's draw: any
                    # pool strategy may be the one a given seed is shown.
                    few_shots["exploit"], few_shots["explore"] = derive_few_shots(
                        feedback, [r["strategy"] for r in pool], banned_menu,
                        defaults=RP.DEFAULT_FEW_SHOTS, n=args.few_shots_per_round,
                    )
                record["feedback"] = str(round_dir / "feedback.json")
                record["feedback_dirs"] = [str(d) for d in scope]
                record["next_menus"] = provenance
                print(f"[round {k}] next: pool of {provenance['exploit_pool_size']} un-retired "
                      f"(sample {args.max_example_strategies}/seed), "
                      f"{provenance['num_banned_strategies']} banned "
                      f"({len(provenance['quota_bans'])} over the "
                      f"{args.ban_after_failures}-failure quota)")
                if provenance['exploit_pool_size'] <= args.max_example_strategies:
                    print(f"[round {k}] NOTE pool "
                          f"({provenance['exploit_pool_size']}) <= "
                          f"--max-example-strategies ({args.max_example_strategies}): every "
                          f"seed will see the whole pool and the sampling weights do "
                          f"nothing. Lower the cap to make them bite.", file=sys.stderr)
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
    manifest["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    (out_dir / "loop.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    total_cost = sum(r["cost_usd"] for r in manifest["rounds"])
    total_ff = sum(r["num_failed_found"] for r in manifest["rounds"])
    total_q = sum(r["num_seeds"] for r in manifest["rounds"])
    print(f"\n{'#' * 70}\nLOOP SUMMARY\n{'#' * 70}")
    for r in manifest["rounds"]:
        rate = (r["num_failed_found"] / r["num_seeds"]) if r["num_seeds"] else 0.0
        clu = r.get("clustering_cost_usd") or 0.0
        gen_t = r.get("generation_elapsed_s") or 0.0
        fb_t = r.get("feedback_elapsed_s") or 0.0
        print(f"  round {r['round']:>2}  {r['num_failed_found']:>3}/{r['num_seeds']:<3} "
              f"FAILED_FOUND ({rate:.0%})  ${r['cost_usd']:.4f} gen"
              + (f" + ${clu:.4f} cluster" if clu else "")
              + f"  {_hms(gen_t)} gen" + (f" + {_hms(fb_t)} fb" if fb_t else "")
              + f"  [{r['num_explore']}e/{r['num_exploit']}x]")
    total_clu = sum(r.get("clustering_cost_usd") or 0.0 for r in manifest["rounds"])
    print(f"\nBroke the system: {total_ff}/{total_q} FAILED_FOUND")
    print(f"Generation: ${total_cost:.4f}"
          + (f" + ${total_clu:.4f} clustering" if total_clu else ""))
    v = manifest.get("verification")
    if v:
        print(f"Verified: {v['kept']}/{v['checked']} criteria upheld "
              f"({v['keep_rate']:.0%}, {_hms(v.get('elapsed_s') or 0)}) -> {v['kept_set']}")
        print(f"Verification: ${v['cost_usd']:.4f}")
    # The grand total is the number that matters and the one --budget-usd caps; keep the
    # components above it so no single line can be mistaken for the whole bill.
    print(f"GRAND TOTAL: ${manifest['total_cost_usd']:.4f}"
          + (f"  ({total_cost + total_clu:.2f} generation + "
             f"{v['cost_usd']:.2f} verification)" if v else "")
          + (f"  [cap ${manifest['budget_usd']:.2f}]" if manifest.get("budget_usd") else ""))
    print(f"Wall clock: {_hms(manifest['elapsed_s'])}"
          + (f"  ({_hms(manifest['elapsed_s'] / total_q)} per seed)" if total_q else ""))
    print(f"Manifest: {out_dir / 'loop.json'}")


if __name__ == "__main__":
    main()
