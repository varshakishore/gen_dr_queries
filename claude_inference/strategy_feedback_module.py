"""Strategy feedback: cluster generated questions against known strategies, pick what to target next.

This module, when called (`build_feedback`), does the following:

  1. CLUSTER   each input question's `strategy` text against a set of KNOWN (seed)
               strategies (the ones the generator is prompted with) or into a NEW cluster
               if it matches no seed. Two interchangeable methods:
                 * "llm"       — LLM reads each strategy and picks seed-vs-new, naming
                                 novel clusters; near-duplicate new clusters are then merged
                                 into broad themes. Defaults to claude-opus-4-1. This
                                 needs ANTHROPIC_API_KEY only.
                 * "embedding" — Not currently used but keeping as an option. Computes
                                 cosine nearest-seed with a distance threshold; anything
                                 farther than the threshold from every seed spawns a new
                                 cluster (new clusters keep a running-mean centroid, so
                                 several off-menu questions converge instead of each
                                 spawning its own). Defaults to OpenAI text-embedding-3-small
                                 as the original EvalTree pipeline does. Needs OPENAI_API_KEY.
  2. SCORE     every cluster: how many of its questions the answering agent FAILED.
               FAILURE IS GOOD HERE: a failed question is one the generator successfully
               made hard. Each cluster is also broken down per `source_run`, i.e. per
               generation prompt (original vs explore), so prompts can be compared strategy 
               by strategy.
  3. SELECT    the focus strategies to target next round (default ranking: works often or
               reasonably often but is currently RARE), and attach few-shot examples sampled 
               from the inputs. Default to selecting only questions that FAILED, up to 5 per
               strategy, so they demonstrate the strategy working.

--------------------------------------------------------------------------------
INPUTS
--------------------------------------------------------------------------------
`build_feedback(examples, seed_strategies=None, **options)`

examples : list[QuestionExample] | list[dict]
    One entry per generated question. Dicts are coerced via `QuestionExample.from_dict`
    (unknown keys are ignored, so DRChallenge `dataset.json` instances work as-is).
    Fields:
      updated_question       (str, required)  the generated/hardened question
      strategy               (str, required)  the strategy text used to make it harder;
                                              this is what gets clustered
      failed                 (bool, required) True if the answering agent FAILED it
                                              (dataset instances: drtulu_verdict=="FAILED")
      source_run             (str, "")        label of the generation prompt / run variant
                                              that produced it (e.g. "explore",
                                              "original"). Drives the per-prompt comparison.
      seed_question          (str, "")        the original question it was derived from
      verification_criterion (str, "")        what the grader checked
      round                  (int|None)       0 = seed as-is, 1..N = harder rewrites

    Data is loaded from output directories of the research pipeline:
      load_examples_from_runs(paths, ...)   <- raw research-pipeline run dirs (sample_*.json)

seed_strategies_file : str | None
    A file containing the KNOWN strategy menu to cluster against. Defaults to `SEED_STRATEGIES.txt`. 
    Every seed gets a cluster in the output even if empty (unused).
    `load_seed_strategies(path)` reads the file with one strategy per line (# comments ok).

Options (all keyword-only, shown with defaults):
    assign_method="llm"           "llm" | "embedding"  (see above)
    cluster_model="claude-opus-4-6"   [llm] model doing the cluster assignment
    merge_new_clusters=True       [llm] consolidate near-duplicate new clusters
    batch_size=25                 [llm] strategies per assignment call
    embedding_model="text-embedding-3-small"   [embedding] embedding model if using embeddings
    seed_max_distance=0.6         [embedding] cosine distance beyond which a strategy
                                  spawns a NEW cluster instead of joining a seed
    question_embeddings=None      [embedding] optional (N, D) array aligned with `examples`;
                                  supply cached vectors to skip the OpenAI call
    min_failure_rate=0.25         a cluster must stump the agent this often to be eligible
    max_share=0.5                 skip catch-all clusters covering more than this share
    min_cluster_size=0            ignore clusters with fewer questions than this; at 0 the
                                  UNUSED strategies (0 questions this round) are kept and fed
                                  back as untried, exempt from the failure_rate/share gates
    max_cluster_size=None         optionally ignore clusters larger than this
    rank_by="underrepresented"    "underrepresented" (failure_rate*(1-share)) | "diverse"
                                  (interleave failure-rate and volume rankings for a spread
                                  of cluster sizes) | "failure_rate" | "volume"
    always_include_new=True       additionally force-include every underrepresented NEW
                                  cluster that stumped the agent at least once (appended
                                  even if it missed the size/failure thresholds)
    examples_per_strategy=5       max few-shot failure examples attached to EACH focus
                                  strategy; deduped globally so no question repeats
    include_instances=True        include the full per-cluster instance list in
                                  cluster_comparison (set False for a compact output)
    anthropic_client=None         inject a client (tests / custom auth)

--------------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------------
A plain JSON-serializable dict (write it with `write_feedback`):

{
  "meta": {
    "generated_at": ISO-8601 UTC,
    "clustering": {cluster_mode:"seeded", assign_method, num_seeds,
                   num_seed_clusters_used, num_new_clusters,
                   + cluster_model/merged_new_clusters/new_cluster_labels   [llm]
                   + embedding_model/seed_max_distance/nearest_seed_distance_{min,mean,max}
                                                                            [embedding]},
    "source_runs": [...],            # the generation prompts present in the input
    "num_instances": int, "num_failed": int, "overall_failure_rate": float,
    "selection": {...echo of the selection options...},
    "notes": str                     # how to read failure_rate / share
  },

  "focus_strategies": [              # ranked; THE THING TO FEED BACK INTO GENERATION
    {
      "rank": 1,
      "cluster_id": "seed.3" | "new.1",
      "description": strategy text (seeds) or "[novel strategy not in seed menu] <name>",
      "num_questions", "num_failed", "failure_rate", "share", "score",
      "selected_for": "underrepresented" | "diverse" | "failure_rate" | "volume"
                      | "new-underrepresented" | "unused",
      "rationale": one-line explanation of why it was picked,
      "by_source_run": {prompt: {num_questions, num_failed, failure_rate}},
      "few_shot_failures": [         # sampled from the inputs, FAILED questions only
        {seed_question, updated_question, strategy, verification_criterion, source_run}
      ]
    }, ...
  ],

  "ineligible_strategies": [         # the inverse set: every cluster NOT fed back
    {cluster_id, description, num_questions, num_failed, failure_rate, share, score,
     excluded_for: [which selection thresholds it missed],
     rationale: one-line explanation, by_source_run: {prompt: {...}}}
  ],

  "cluster_comparison": [            # EVERY cluster (all seeds incl. empty ones + new ones)
    {cluster_id, description, num_questions, num_failed, failure_rate,
     by_source_run: {prompt: {...}},
     instances: [{index, seed_question, updated_question, strategy,
                  verification_criterion, failed, source_run, round}]}   # if include_instances
  ]
}

The output matches `build_strategy_feedback.py --cluster-mode seeded`, so the existing
`feedback_viewer.py` can also render this module's output.

--------------------------------------------------------------------------------
CLI
--------------------------------------------------------------------------------
    # from raw generation runs (several prompts folded in for comparison):
    python strategy_feedback_module.py --runs runs/sqa_50_100_explore runs/sqa_50_100_original \
        --out runs/round1/strategy_feedback.json

    # from an already-built DRChallenge dataset dir:
    python strategy_feedback_module.py --dataset-dir EvalTree/Datasets/DRChallenge \
        --model drtulu --out feedback.json

    # embedding-based assignment instead of the LLM:
    python strategy_feedback_module.py --runs runs/round1 --assign-method embedding \
        --seed-max-distance 0.45 --out feedback.json

Requires: ANTHROPIC_API_KEY (assign_method="llm") or OPENAI_API_KEY (assign_method=
"embedding", unless `question_embeddings` is supplied). numpy is imported only by the
embedding path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Known strategies (the menu the generator is prompted with)
# ---------------------------------------------------------------------------

# EvalTree's stage-2 prepends this before embedding each strategy; matched here so seeds and
# questions land in the same region of the embedding space (and so externally cached
# EvalTree embeddings can be passed in via `question_embeddings`).
EMBEDDING_PREFIX = "The model has the following capability: "

DEFAULT_ANTHROPIC_CLUSTER_MODEL = "claude-opus-4-6"
DEFAULT_OPENAI_CLUSTER_MODEL = "gpt-4o-mini"
DEFAULT_CLUSTER_MODEL = DEFAULT_ANTHROPIC_CLUSTER_MODEL

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _stage(msg: str) -> None:
    """Announce the pipeline stage in progress (stderr, so stdout stays the report)."""
    print(f"[stage] {msg}", file=sys.stderr, flush=True)


def load_seed_strategies(seeds_file: Path | str | None) -> list[str]:
    """The known-strategy menu: from a file (one per line, `#` comments ignored) or built-in."""
    if seeds_file is None:
        seeds_file = Path(__file__).parent / "SEED_STRATEGIES.txt"
    lines = [ln.strip() for ln in Path(seeds_file).read_text().splitlines()]
    seeds = [ln for ln in lines if ln and not ln.startswith("#")]
    if not seeds:
        raise ValueError(f"no seed strategies found in {seeds_file}")
    return seeds


# ---------------------------------------------------------------------------
# Input record
# ---------------------------------------------------------------------------


@dataclass
class QuestionExample:
    """One generated question: what was asked, how it was made hard, and whether it worked.

    `failed=True` means the answering agent FAILED the question — the generator succeeded.
    `source_run` labels the generation prompt/variant, and is what the per-prompt comparison
    in the output is keyed on.
    """

    updated_question: str
    strategy: str
    failed: bool
    source_run: str = ""
    seed_question: str = ""
    verification_criterion: str = ""
    round: int | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "QuestionExample":
        """Coerce a dict to a QuestionExample, tolerating DRChallenge `dataset.json` instances.

        Accepts either `failed` (bool/0/1) or a verdict string under `drtulu_verdict` /
        `verdict` ("FAILED" / "PASSED"). Accepts `question` as an alias for
        `updated_question`. Unknown keys are preserved in `extra`.
        """
        if isinstance(d, cls):
            return d
        known = {"updated_question", "question", "strategy", "failed", "source_run",
                 "seed_question", "verification_criterion", "round",
                 "drtulu_verdict", "verdict"}
        if "failed" in d:
            failed = bool(d["failed"])
        else:
            verdict = d.get("drtulu_verdict", d.get("verdict", ""))
            failed = str(verdict).upper() == "FAILED"
        return cls(
            updated_question=d.get("updated_question") or d.get("question") or "",
            strategy=d.get("strategy", "") or "",
            failed=failed,
            source_run=d.get("source_run") or "",
            seed_question=d.get("seed_question", "") or "",
            verification_criterion=d.get("verification_criterion", "") or "",
            round=d.get("round"),
            extra={k: v for k, v in d.items() if k not in known},
        )

    def as_instance(self, index: int) -> dict:
        """The flat view used inside `cluster_comparison.instances`."""
        return {
            "index": index,
            "seed_question": self.seed_question,
            "updated_question": self.updated_question,
            "strategy": self.strategy,
            "verification_criterion": self.verification_criterion,
            "failed": self.failed,
            "drtulu_verdict": "FAILED" if self.failed else "PASSED",
            "source_run": self.source_run,
            "round": self.round,
        }


def normalize_examples(examples: Iterable[QuestionExample | dict]) -> list[QuestionExample]:
    """Coerce the input list, dropping entries with no strategy text to cluster on."""
    out: list[QuestionExample] = []
    for e in examples:
        ex = e if isinstance(e, QuestionExample) else QuestionExample.from_dict(e)
        if not ex.strategy.strip():
            continue  # nothing to cluster; the strategy field IS the clustering signal
        out.append(ex)
    return out


# ---------------------------------------------------------------------------
# Loaders (raw pipeline runs / built DRChallenge dataset dirs)
# ---------------------------------------------------------------------------


def _strategy_label(status: str, chosen: str) -> str:
    """Human-readable strategy for the *deciding* attempt of a result."""
    if status == "ALREADY_HARD":
        return "ALREADY_HARD (seed already hard; no rewrite)"
    if status == "EXHAUSTED":
        return f"EXHAUSTED (no failing answer found; hardest attempt) | {chosen}"
    return chosen  # FAILED_FOUND and any other rewrite-based status


def _verdict_of(attempt: dict) -> str | None:
    """The grader's verdict for an attempt, or None if it wasn't (validly) graded."""
    judgment = attempt.get("judgment")
    if isinstance(judgment, dict):
        verdict = judgment.get("verdict")
        if verdict in ("PASSED", "FAILED"):
            return verdict
    return None


def _iter_sample_files(path: Path) -> Iterable[Path]:
    """Yield sample_*.json files under `path` (a file, a run dir, or a parent of run dirs)."""
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        print(f"[skip] not a file or directory: {path}", file=sys.stderr)
        return
    direct = sorted(path.glob("sample_*.json"))
    if direct:
        yield from direct
    else:
        yield from sorted(path.glob("*/sample_*.json"))


def load_examples_from_runs(
    run_paths: Sequence[Path | str],
    *,
    deciding_only: bool = False,
    keep_run_duplicates: bool = False,
    include_seed_round: bool = False,
) -> list[QuestionExample]:
    """Build the input list straight from research-pipeline run output.

    Each run dir holds `sample_NNN.json` files with a list of `results`; each result has a
    `seed` and a list of `attempts` carrying `harder.updated_question`, `harder.chosen_strategy`
    and `judgment.verdict`. `source_run` is set to the name of the dir holding the samples,
    so passing several run dirs (one per generation prompt) yields a comparable mix.

    A path may be a run dir, a parent of run dirs, or a single sample_*.json.

    deciding_only=True        keep only each result's deciding ("challenge") attempt.
    keep_run_duplicates=True  dedup per (run, question) instead of globally, so the same
                              question can appear once per run it came from.
    include_seed_round=True   also keep round-0 attempts (the seed tested as-is); by default
                              they are omitted since they are not generated questions.
    """
    deciding: list[tuple[str, QuestionExample]] = []
    other: list[tuple[str, QuestionExample]] = []

    for run_path in run_paths:
        for sample_path in _iter_sample_files(Path(run_path)):
            run_key = str(sample_path.parent)
            data = json.loads(sample_path.read_text())
            results = data if isinstance(data, list) else data.get("results", [])
            for result in results:
                if result.get("final_status") == "ERROR":
                    continue
                attempts = result.get("attempts", [])
                if not attempts:
                    continue
                last_i = len(attempts) - 1
                for i, attempt in enumerate(attempts):
                    round_idx = attempt.get("attempt", i)  # 0 = seed as-is
                    if round_idx == 0 and not include_seed_round:
                        continue
                    is_deciding = i == last_i
                    if deciding_only and not is_deciding:
                        continue
                    verdict = _verdict_of(attempt)
                    harder = attempt.get("harder", {}) or {}
                    updated = harder.get("updated_question", "")
                    if verdict is None or not updated:
                        continue
                    strategy = (
                        _strategy_label(result.get("final_status", ""), harder.get("chosen_strategy", ""))
                        if is_deciding else harder.get("chosen_strategy", "")
                    )
                    ex = QuestionExample(
                        updated_question=updated,
                        strategy=strategy or "",
                        failed=verdict == "FAILED",
                        source_run=Path(run_key).name,
                        seed_question=result.get("seed", ""),
                        verification_criterion=harder.get("verification_criterion", ""),
                        round=round_idx,
                    )
                    (deciding if is_deciding else other).append((run_key, ex))

    # Deciding/failing questions first, then the remaining attempts.
    seen: set = set()
    out: list[QuestionExample] = []
    for run_key, ex in deciding + other:
        key = (run_key, ex.updated_question) if keep_run_duplicates else ex.updated_question
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


# ---------------------------------------------------------------------------
# Clustering: assign each strategy to a known seed or a new cluster
# ---------------------------------------------------------------------------

_LLM_PROMPT = """You are clustering question-generation STRATEGIES used to make research \
questions harder. There is a fixed menu of SEED strategy clusters (numbered) as well as new \
clusters added by previous iterations. For EACH input strategy, assign it to the cluster that \
represents the same CORE IDEA. If none of the seed clusters or existing new clusters match, \
assign it to a NEW cluster and give that cluster a BROAD, general name that will also capture \
related strategies. Do NOT invent a new cluster if it fits an existing one. Do NOT assign a \
strategy to a seed cluster or existing new cluster if it does not actually match that cluster's \
core idea.

How to decide (accuracy first — do NOT distort assignments to hit a target count):
- Assign to a SEED cluster when the strategy matches that seed's core idea. A \
real thematic match, not a superficial keyword overlap — do NOT force a strategy into a \
seed it does not really fit just to avoid making a new cluster.
- If it fits NO seed well, consider assigning it to an EXISTING NEW cluster; only invent a \
brand-new cluster if no existing cluster fits.
- When you do invent a new cluster, give it a BROAD, general name that will also capture \
related strategies (e.g. "temporal reasoning traps", NOT "temporal verification trap"), and \
never a near-synonym of an existing new cluster.

Aim for a balance: capture distinct strategies as new clusters, but keep \
similar novel strategies together under one broad cluster rather than many tiny ones.

SEED CLUSTERS:
{seed_block}

EXISTING NEW CLUSTERS (reuse one of these before inventing another):
{new_so_far}

STRATEGIES TO ASSIGN (each prefixed with its index):
{strat_block}

Return ONLY a JSON array, one object per strategy, no prose:
[{{"i": <index>, "cluster": "seed:<n>"}}, {{"i": <index>, "cluster": "new:<broad name>"}}, ...]
Every input index must appear exactly once."""


_MERGE_PROMPT = """The following NEW strategy clusters were created because they did not \
fit a fixed seed menu. Some may be duplicates that should be a single cluster. Merge \
clusters that describe a similar underlying strategy into one group, using a \
broad canonical name. For example, all temporal reasoning variants should be together; \
all constraint-satisfaction variants should be together).

NEW CLUSTERS (name — example strategy):
{cluster_block}

Return ONLY a JSON array of groups, no prose. Every input name must appear in exactly one \
group's "members":
[{{"canonical": "<broad group name>", "members": ["<name>", "<name>", ...]}}, ...]"""


def _parse_json_array(text: str) -> list:
    a, b = text.find("["), text.rfind("]")
    if a == -1 or b == -1 or b < a:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")
    return json.loads(text[a:b + 1])


def _llm_text(client, provider: str, model: str, prompt: str, max_tokens: int) -> str:
    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content)

    if provider == "openai":
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        return resp.output_text

    raise ValueError(f"unknown LLM provider: {provider!r}")


def _norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def consolidate_new_clusters(new_labels: dict, examples: dict, provider: str,
                             model: str, client, max_tokens: int = 4000) -> dict:
    """Merge near-duplicate NEW clusters into broad themes.

    `new_labels` maps a new-cluster index to its name; `examples` maps the same index to a
    representative strategy (context for the merge). Returns
    {"remap": {old_index: canonical_index}, "canonical_labels": {canonical_index: name}},
    re-indexed contiguously from min(new_labels) so the caller can drop it straight back in.
    """
    idxs = sorted(new_labels)
    base = idxs[0]
    label_to_idx = {_norm(new_labels[i]): i for i in idxs}
    cluster_block = "\n".join(
        f"- {new_labels[i]} — {str(examples.get(i, '')).strip()[:140]}" for i in idxs
    )
    groups = _parse_json_array(
        _llm_text(
            client,
            provider,
            model,
            _MERGE_PROMPT.format(cluster_block=cluster_block),
            max_tokens,
        )
    )

    remap: dict[int, int] = {}
    canonical_labels: dict[int, str] = {}
    assigned: set[int] = set()
    for grp in (g for g in groups if isinstance(g, dict)):
        canon_idx = base + len(canonical_labels)
        canonical_labels[canon_idx] = str(grp.get("canonical", "")).strip() or new_labels[idxs[0]]
        for m in grp.get("members", []):
            old = label_to_idx.get(_norm(m))
            if old is not None and old not in remap:
                remap[old] = canon_idx
                assigned.add(old)
    # Any cluster the model dropped keeps its own group so nothing is lost.
    for i in idxs:
        if i not in assigned:
            canon_idx = base + len(canonical_labels)
            canonical_labels[canon_idx] = new_labels[i]
            remap[i] = canon_idx
    return {"remap": remap, "canonical_labels": canonical_labels}


def assign_llm(
    strategies: Sequence[str],
    seeds: Sequence[str],
    model: str | None = None,
    *,
    provider: str = "anthropic",
    batch_size: int = 25,
    max_tokens: int = 8000,
    client=None,
    merge: bool = True,
) -> dict:
    """Assign each strategy to a seed cluster or a NEW cluster using an LLM.

    `provider` specifies which LLM provider to use ("anthropic" or "openai").

    Processed in batches; new clusters discovered in earlier batches are offered to later
    batches so similar novel strategies merge online. With `merge` (default), a final
    consolidation pass merges the remaining near-duplicates into broad themes.

    Returns the assignment dict `build_nodes` consumes:
      {assignments: [cluster_index per strategy], n_seeds, n_clusters,
       is_seed: [bool per cluster], new_labels: {cluster_index: name}, method:"llm", model}
    """

    if model is None:
        if provider == "anthropic":
            model = DEFAULT_ANTHROPIC_CLUSTER_MODEL
        elif provider == "openai":
            model = DEFAULT_OPENAI_CLUSTER_MODEL
        else:
            raise ValueError(f"unknown LLM provider: {provider!r}")

    if client is None:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"unknown LLM provider: {provider!r}")
        
    n_seeds = len(seeds)
    seed_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(seeds))
    new_examples: dict[int, str] = {}  # a representative strategy per new cluster

    assignments: list[int | None] = [None] * len(strategies)
    new_order: list[str] = []          # normalized new-cluster names, creation order
    new_index: dict[str, int] = {}     # normalized name -> cluster index (>= n_seeds)
    new_labels: dict[int, str] = {}    # cluster index -> display label

    def register_new(name: str) -> int:
        key = _norm(name) or "unclassified"
        if key not in new_index:
            idx = n_seeds + len(new_order)
            new_index[key] = idx
            new_order.append(key)
            new_labels[idx] = name.strip() or "unclassified"
        return new_index[key]

    indexed = list(enumerate(strategies))
    n_batches = (len(indexed) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(indexed), batch_size), start=1):
        batch = indexed[start:start + batch_size]
        _stage(f"  assigning batch {bi}/{n_batches} ({len(batch)} strategies)")
        new_so_far = "; ".join(new_labels[new_index[k]] for k in new_order) or "none yet"
        strat_block = "\n".join(f"[{i}] {s}" for i, s in batch)
        prompt = _LLM_PROMPT.format(seed_block=seed_block, new_so_far=new_so_far,
                                    strat_block=strat_block)
        for obj in _parse_json_array(_llm_text(client, provider, model, prompt, max_tokens)):
            try:
                i = int(obj["i"])
                cl = str(obj["cluster"]).strip()
            except (KeyError, ValueError, TypeError):
                continue
            if not (0 <= i < len(strategies)):
                continue
            if cl.lower().startswith("seed:"):
                nums = re.findall(r"\d+", cl)
                n = (int(nums[0]) - 1) if nums else 0
                assignments[i] = min(max(n, 0), n_seeds - 1)
            else:
                name = cl.split(":", 1)[1].strip() if ":" in cl else cl
                idx = register_new(name)
                assignments[i] = idx
                new_examples.setdefault(idx, strategies[i])  # a representative for merging

    # Any strategy the model skipped -> an explicit "unclassified" cluster (never dropped).
    if any(a is None for a in assignments):
        unresolved = register_new("unclassified (model gave no assignment)")
        assignments = [unresolved if a is None else a for a in assignments]

    if merge and len(new_order) > 1:
        _stage(f"  merging {len(new_order)} new clusters")
        merged = consolidate_new_clusters(
            new_labels, new_examples, provider, model, client
        )
        remap = merged["remap"]
        assignments = [remap.get(a, a) if a >= n_seeds else a for a in assignments]
        new_labels = merged["canonical_labels"]

    return {
        "assignments": assignments,
        "n_seeds": n_seeds,
        "n_clusters": n_seeds + len(new_labels),
        "is_seed": [True] * n_seeds + [False] * len(new_labels),
        "new_labels": new_labels,
        "method": "llm",
        "provider": provider,
        "model": model,
    }


def embed_texts(texts: Sequence[str], embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    """Embed texts with the same model + prefix EvalTree's stage 2 used. Returns (N, D)."""
    import numpy as np
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(input=[EMBEDDING_PREFIX + t for t in texts],
                                    model=embedding_model)
    return np.stack([np.asarray(d.embedding, dtype=np.float64) for d in resp.data])  # order preserved


def _normalize_rows(mat):
    import numpy as np
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def assign_embedding(question_embs, seed_embs, max_distance: float = 0.5) -> dict:
    """Assign each strategy to the nearest seed, or a new cluster if every seed is too far.

    Cosine distance = 1 - cosine similarity. Seed centroids are fixed; new clusters keep a
    running-mean centroid so several off-menu strategies converge into one novel cluster
    rather than each spawning its own. Adds `nearest_seed_dist` (per question) so the
    threshold can be tuned.
    """
    import numpy as np

    q = _normalize_rows(np.asarray(question_embs, dtype=np.float64))
    n_seeds = len(seed_embs)
    centroids = list(_normalize_rows(np.asarray(seed_embs, dtype=np.float64)))
    is_seed = [True] * n_seeds
    new_sums: dict[int, Any] = {}

    assignments: list[int] = []
    nearest_seed_dist: list[float] = []
    for e in q:
        seed_d = 1.0 - (np.stack(centroids[:n_seeds]) @ e)
        nearest_seed_dist.append(float(seed_d.min()))
        dists = 1.0 - (np.stack(centroids) @ e)
        j = int(dists.argmin())
        if dists[j] <= max_distance:
            assignments.append(j)
            if not is_seed[j]:
                new_sums[j] = new_sums[j] + e
                centroids[j] = new_sums[j] / np.linalg.norm(new_sums[j])
        else:
            j = len(centroids)
            centroids.append(e.copy())
            is_seed.append(False)
            new_sums[j] = e.copy()
            assignments.append(j)
    return {
        "assignments": assignments,
        "n_seeds": n_seeds,
        "n_clusters": len(centroids),
        "is_seed": is_seed,
        "nearest_seed_dist": nearest_seed_dist,
        "method": "embedding",
    }


def build_nodes(examples: Sequence[QuestionExample], seeds: Sequence[str],
                assignment: dict) -> list[dict]:
    """One node per cluster: {node_id, description, num_questions, num_failed, failure_rate, leaves}.

    EVERY seed strategy gets a node even when no question landed on it (size 0), so the
    comparison always shows the full menu — an unused seed is itself a signal. New clusters
    (off-menu strategies) are appended as `new.<k>`, described by the LLM-given name when
    available, otherwise by a representative member.
    """
    n_seeds = assignment["n_seeds"]
    members: dict[int, list[int]] = {}
    for i, c in enumerate(assignment["assignments"]):
        members.setdefault(c, []).append(i)

    def make_node(node_id: str, description: str, leaves: list[int]) -> dict:
        size = len(leaves)
        num_failed = sum(1 for i in leaves if examples[i].failed)
        return {
            "node_id": node_id,
            "description": description,
            "num_questions": size,
            "num_failed": num_failed,
            "failure_rate": (num_failed / size) if size else 0.0,
            "leaves": leaves,
        }

    new_labels = assignment.get("new_labels", {})
    nodes = [make_node(f"seed.{c + 1}", seeds[c], members.get(c, [])) for c in range(n_seeds)]
    for new_ordinal, c in enumerate((c for c in sorted(members) if c >= n_seeds), start=1):
        leaves = members[c]
        label = new_labels.get(c)
        if label:
            description = f"[novel strategy not in seed menu] {label}"
        else:
            rep = examples[leaves[0]].strategy.strip().replace("\n", " ")
            description = f"[novel strategy not in seed menu] e.g. {rep[:160]}"
        nodes.append(make_node(f"new.{new_ordinal}", description, leaves))
    return nodes


# ---------------------------------------------------------------------------
# Selection: which strategies to target next
# ---------------------------------------------------------------------------


def select_focus(nodes: list[dict], total: int, *, min_failure_rate: float = 0.3,
                 max_share: float = 0.5, min_cluster_size: int = 0,
                 max_cluster_size: int | None = None,
                 rank_by: str = "underrepresented",
                 always_include_new: bool = True) -> tuple[list[dict], list[dict]]:
    """Pick the clusters that work (stump the agent) and are worth leaning into next round.

    A cluster is eligible when it is a real theme (>= min_cluster_size questions, and no
    larger than max_cluster_size if set), it stumps the agent often enough
    (failure_rate >= min_failure_rate), and it is not an over-represented catch-all
    (share <= max_share).

    `rank_by`:
      * underrepresented (default) — score = failure_rate * (1 - share): strategies that fail
        a lot yet are RARE this round, i.e. working approaches the generator under-uses.
      * diverse — interleave a failure_rate ranking (small, strongly-failing) with a volume
        ranking (bigger, medium-rate) for a spread of cluster sizes.
      * failure_rate / volume — pure failure_rate, or pure absolute number of failures.

    Each pick is tagged `selected_for` with the reason. When `always_include_new`, every NEW
    (off-menu) cluster that is underrepresented (share <= max_share) and stumped the agent at
    least once is ALWAYS included — even below the size/failure thresholds or the ranking
    cutoff — tagged 'new-underrepresented'.

    With min_cluster_size=0 (the default), UNUSED clusters — seeds the generator never tried
    this round — are eligible too. These are tagged 'unused'.

    Returns `(picked, ineligible)`. EVERY eligible cluster is returned, ranked. `ineligible` 
    is the inverse set, every cluster that did not make it into `picked`, each tagged with 
    the reason for exclusion (`excluded_for`), the list of thresholds it missed, for reporting.
    """
    def enrich(n: dict) -> dict:
        share = n["num_questions"] / total if total else 0.0
        return {**n, "share": share, "score": n["failure_rate"] * (1.0 - share)}

    def exclusion_reasons(n: dict) -> list[str]:
        """Every eligibility threshold this cluster misses ([] means it is eligible)."""
        size = n["num_questions"]
        reasons = []
        if size < min_cluster_size:
            reasons.append(f"too small ({size} < min_cluster_size={min_cluster_size})"
                           + (" — unused this round" if size == 0 else ""))
            return reasons
        if size == 0:
            return reasons  # unused but allowed in: the rate/share gates are moot at n=0
        if max_cluster_size is not None and size > max_cluster_size:
            reasons.append(f"too large ({size} > max_cluster_size={max_cluster_size})")
        if n["failure_rate"] < min_failure_rate:
            reasons.append(f"failure_rate {n['failure_rate']:.2f} < "
                           f"min_failure_rate={min_failure_rate}")
        share = size / total if total else 0.0
        if share > max_share:
            reasons.append(f"over-represented (share {share:.2f} > max_share={max_share})")
        return reasons

    # Unused (0-question) clusters are ranked separately: with no questions there is nothing to
    # score, so they are appended after the ranking.
    passing = [n for n in nodes if not exclusion_reasons(n)]
    eligible = [enrich(n) for n in passing if n["num_questions"] > 0]
    unused = [enrich(n) for n in passing if n["num_questions"] == 0]

    if rank_by == "diverse":
        rankings = [
            ("failure_rate", sorted(eligible, key=lambda n: (n["failure_rate"], n["num_failed"]), reverse=True)),
            ("volume", sorted(eligible, key=lambda n: (n["num_failed"], n["failure_rate"]), reverse=True)),
        ]
        picked: list[dict] = []
        picked_ids: set[str] = set()
        cursors, turn = [0, 0], 0
        while len(picked_ids) < len(eligible):
            tag, ranked = rankings[turn]
            i = cursors[turn]
            while i < len(ranked) and ranked[i]["node_id"] in picked_ids:
                i += 1
            cursors[turn] = i
            if i < len(ranked):
                picked.append({**ranked[i], "selected_for": tag})
                picked_ids.add(ranked[i]["node_id"])
            turn = 1 - turn
    else:
        if rank_by == "failure_rate":
            key = lambda n: (n["failure_rate"], -n["share"], n["num_failed"])
        elif rank_by == "volume":
            key = lambda n: (n["num_failed"], n["failure_rate"])
        else:  # underrepresented (default): high failure but low share first
            rank_by = "underrepresented"
            key = lambda n: (n["score"], n["failure_rate"], -n["share"])
        picked = sorted(eligible, key=key, reverse=True)

    if always_include_new:
        have = {p["node_id"] for p in picked}
        extra = [enrich(n) for n in nodes
                 if n["node_id"].startswith("new.")
                 and n["node_id"] not in have
                 and n["num_failed"] >= 1
                 and (n["num_questions"] / total if total else 0.0) <= max_share]
        extra.sort(key=lambda n: (n["score"], n["failure_rate"]), reverse=True)
        picked.extend({**n, "selected_for": "new-underrepresented"} for n in extra)

    # untried strategies last: nothing to score, but worth trying again
    unused.sort(key=lambda n: n["node_id"])
    picked.extend({**n, "selected_for": "unused"} for n in unused)

    # the inverse set: everything not picked, with why it was left out (for reporting)
    have = {p["node_id"] for p in picked}
    ineligible = [{**enrich(n), "excluded_for": exclusion_reasons(n)}
                  for n in nodes if n["node_id"] not in have]
    ineligible.sort(key=lambda n: (n["score"], n["failure_rate"], n["num_questions"]), reverse=True)

    return picked, ineligible


def source_run_breakdown(leaves: Sequence[int], examples: Sequence[QuestionExample]) -> dict:
    """Per-source_run {num_questions, num_failed, failure_rate} for a cluster's members.

    Lets a cluster be compared across generation prompts — e.g. whether 'original' or
    'explore' hit this strategy more, and whose questions stumped the agent more often.
    """
    agg: dict[str, dict] = defaultdict(lambda: {"num_questions": 0, "num_failed": 0})
    for i in leaves:
        run = examples[i].source_run or "unknown"
        agg[run]["num_questions"] += 1
        if examples[i].failed:
            agg[run]["num_failed"] += 1
    out: dict[str, dict] = {}
    for run in sorted(agg):
        n, f = agg[run]["num_questions"], agg[run]["num_failed"]
        out[run] = {"num_questions": n, "num_failed": f,
                    "failure_rate": round(f / n, 4) if n else 0.0}
    return out


def few_shot_failures(node: dict, examples: Sequence[QuestionExample], *,
                      per_strategy: int, seen: set[str]) -> list[dict]:
    """Up to `per_strategy` FAILED questions sampled from one cluster, as few-shot examples.

    Only failures — those are the questions the generator successfully made hard. Sampled in
    dataset order (deterministic) and deduped by question text against `seen`, which is shared
    across clusters so the same question is never shown twice.
    """
    out: list[dict] = []
    for i in node["leaves"]:
        if len(out) >= per_strategy:
            break
        ex = examples[i]
        if not ex.failed:
            continue
        q = ex.updated_question
        if not q or q in seen:
            continue
        seen.add(q)
        out.append({
            "seed_question": ex.seed_question,
            "updated_question": q,
            "strategy": ex.strategy,
            "verification_criterion": ex.verification_criterion,
            "source_run": ex.source_run,
            "drtulu_verdict": "FAILED",
        })
    return out


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def build_feedback(
    examples: Iterable[QuestionExample | dict],
    seed_strategies: Sequence[str] | None = None,
    *,
    # clustering
    assign_method: str = "llm",
    cluster_provider: str = "anthropic",
    cluster_model: str = DEFAULT_CLUSTER_MODEL,
    merge_new_clusters: bool = True,
    batch_size: int = 25,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    seed_max_distance: float = 0.5,
    question_embeddings=None,
    llm_client=None,
    # selection
    min_failure_rate: float = 0.3,
    max_share: float = 0.5,
    min_cluster_size: int = 0,
    max_cluster_size: int | None = None,
    rank_by: str = "underrepresented",
    always_include_new: bool = True,
    examples_per_strategy: int = 5,
    include_instances: bool = True,
) -> dict:
    """Cluster generated questions against known strategies and suggest what to target next.

    See the module docstring for the full input/output contract. Returns a JSON-serializable
    dict with `meta`, `focus_strategies` (each carrying `few_shot_failures`), and
    `cluster_comparison` (every cluster, broken down per generation prompt).
    """
    exs = normalize_examples(examples)
    if not exs:
        raise ValueError("no usable examples: every entry lacked a non-empty `strategy`")
    seeds = list(seed_strategies) if seed_strategies is not None else []
    strategies = [e.strategy for e in exs]

    # --- 1. cluster ---------------------------------------------------------
    _how = f"{cluster_provider}/{cluster_model or 'default'}" if assign_method == "llm" else embedding_model
    _stage(f"1/3 cluster: {len(exs)} strategies vs {len(seeds)} seeds [{assign_method}: {_how}]")
    if assign_method == "llm":
        assignment = assign_llm(
            strategies,
            seeds,
            cluster_model,
            provider=cluster_provider,
            batch_size=batch_size,
            client=llm_client,
            merge=merge_new_clusters,
        )
    elif assign_method == "embedding":
        q_embs = question_embeddings
        if q_embs is None:
            q_embs = embed_texts(strategies, embedding_model)
        elif len(q_embs) != len(exs):
            raise ValueError(f"question_embeddings has {len(q_embs)} rows but there are "
                             f"{len(exs)} examples")
        seed_embs = embed_texts(seeds, embedding_model)
        assignment = assign_embedding(q_embs, seed_embs, seed_max_distance)
    else:
        raise ValueError(f"unknown assign_method: {assign_method!r} (use 'llm' or 'embedding')")

    nodes = build_nodes(exs, seeds, assignment)

    # --- 2. score / select --------------------------------------------------
    _stage(f"2/3 score + select: {len(nodes)} clusters, rank_by={rank_by}")
    total = len(exs)
    num_failed = sum(1 for e in exs if e.failed)
    focus, ineligible = select_focus(
        nodes, total,
        min_failure_rate=min_failure_rate, max_share=max_share,
        min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
        rank_by=rank_by,
        always_include_new=always_include_new,
    )
    # No ancestor/descendant dedup needed: seeded clusters are disjoint by construction.

    source_runs = sorted({e.source_run or "unknown" for e in exs})
    n_new = assignment["n_clusters"] - assignment["n_seeds"]
    n_seed_used = sum(1 for n in nodes
                      if n["node_id"].startswith("seed.") and n["num_questions"] > 0)

    cluster_source = {
        "cluster_mode": "seeded",
        "assign_method": assign_method,
        "num_seeds": len(seeds),
        "num_seed_clusters_used": n_seed_used,
        "num_new_clusters": n_new,
    }
    if assign_method == "llm":
        cluster_source["cluster_model"] = cluster_model
        cluster_source["merged_new_clusters"] = merge_new_clusters
        cluster_source["new_cluster_labels"] = [
            assignment.get("new_labels", {}).get(assignment["n_seeds"] + k) for k in range(n_new)
        ]
    else:
        dists = assignment.get("nearest_seed_dist") or []
        cluster_source["embedding_model"] = embedding_model
        cluster_source["seed_max_distance"] = seed_max_distance
        cluster_source["nearest_seed_distance_min"] = round(min(dists), 4) if dists else None
        cluster_source["nearest_seed_distance_mean"] = round(sum(dists) / len(dists), 4) if dists else None
        cluster_source["nearest_seed_distance_max"] = round(max(dists), 4) if dists else None

    # --- 3. assemble --------------------------------------------------------
    _stage(f"3/3 assemble: {len(focus)} focus, {len(ineligible)} ineligible")

    def cluster_sort_key(n: dict):
        kind, _, num = n["node_id"].partition(".")
        order = {"seed": 0, "new": 1}.get(kind, 2)
        try:
            return (order, int(num))
        except ValueError:
            return (order, 0)

    cluster_comparison = []
    for n in sorted(nodes, key=cluster_sort_key):
        row = {
            "cluster_id": n["node_id"],
            "description": n["description"],
            "num_questions": n["num_questions"],
            "num_failed": n["num_failed"],
            "failure_rate": round(n["failure_rate"], 4),
            "by_source_run": source_run_breakdown(n["leaves"], exs),
        }
        if include_instances:
            row["instances"] = [exs[i].as_instance(i) for i in n["leaves"]]
        cluster_comparison.append(row)

    seen_examples: set[str] = set()
    focus_strategies = []
    for rank, n in enumerate(focus, start=1):
        focus_strategies.append({
            "rank": rank,
            "cluster_id": n["node_id"],
            "description": n["description"],
            "num_questions": n["num_questions"],
            "num_failed": n["num_failed"],
            "failure_rate": round(n["failure_rate"], 4),
            "share": round(n["share"], 4),
            "score": round(n["score"], 4),
            "selected_for": n.get("selected_for", ""),
            "rationale": (
                "Never used — the generator did not try this strategy at all this round. "
                "Untested rather than unproductive, so give it a go next round."
                if n["num_questions"] == 0 else
                f"Works ({n['num_failed']}/{n['num_questions']} stumped the agent, "
                f"failure rate {n['failure_rate']:.2f}) but under-used — only {n['share'] * 100:.0f}% "
                f"of this round's questions. Under-represented, so lean into it next round."
            ),
            "by_source_run": source_run_breakdown(n["leaves"], exs),
            "few_shot_failures": few_shot_failures(
                n, exs, per_strategy=examples_per_strategy, seen=seen_examples),
        })

    ineligible_strategies = [{
        "cluster_id": n["node_id"],
        "description": n["description"],
        "num_questions": n["num_questions"],
        "num_failed": n["num_failed"],
        "failure_rate": round(n["failure_rate"], 4),
        "share": round(n["share"], 4),
        "score": round(n["score"], 4),
        "excluded_for": n["excluded_for"],
        "rationale": "Not fed back next round: " + "; ".join(n["excluded_for"]) + ".",
        "by_source_run": source_run_breakdown(n["leaves"], exs),
    } for n in ineligible]

    return {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "clustering": cluster_source,
            "source_runs": source_runs,
            "num_instances": total,
            "num_failed": num_failed,
            "overall_failure_rate": round(num_failed / total, 4) if total else 0.0,
            "selection": {
                "rank_by": rank_by,
                "min_failure_rate": min_failure_rate,
                "max_share": max_share,
                "min_cluster_size": min_cluster_size,
                "max_cluster_size": max_cluster_size,
                "always_include_new": always_include_new,
                "examples_per_strategy": examples_per_strategy,
            },
            "notes": "failure_rate is the fraction of questions in the cluster that FAILED "
                     "verification; higher is better (the generator succeeded at making a hard "
                     "question). share is the cluster's fraction of this round's questions. "
                     f"focus_strategies are ranked by rank_by='{rank_by}'; the default "
                     "'underrepresented' favours strategies that work (high failure_rate) but are "
                     "RARE (low share) — working approaches the generator under-uses. each carries "
                     "up to examples_per_strategy few_shot_failures — questions that FAILED "
                     "verification — to demonstrate it. ineligible_strategies is the inverse set: "
                     "every cluster that missed a selection threshold, with excluded_for saying "
                     "which.",
        },
        "focus_strategies": focus_strategies,
        "ineligible_strategies": ineligible_strategies,
        "cluster_comparison": cluster_comparison,
    }


def write_feedback(feedback: dict, out_path: Path | str) -> Path:
    """Write the feedback dict to JSON, creating parent dirs. Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False))
    return out_path


def format_summary(feedback: dict) -> str:
    """A short human-readable report of a feedback dict (what the CLI prints)."""
    meta = feedback["meta"]
    cs = meta["clustering"]
    lines = [
        f"clustering: seeded[{cs['assign_method']}"
        + (f"={cs['cluster_model']}" if cs.get("cluster_model") else "")
        + (f"={cs['embedding_model']}, max_dist={cs['seed_max_distance']}"
           if cs.get("embedding_model") else "") + "]",
        f"  seeds: {cs['num_seed_clusters_used']}/{cs['num_seeds']} used, "
        f"{cs['num_new_clusters']} new cluster(s)",
    ]
    if cs.get("nearest_seed_distance_min") is not None:
        lines.append(f"  nearest-seed dist min/mean/max = {cs['nearest_seed_distance_min']}/"
                     f"{cs['nearest_seed_distance_mean']}/{cs['nearest_seed_distance_max']}")
    for lbl in (cs.get("new_cluster_labels") or []):
        if lbl:
            lines.append(f"    new cluster: {lbl}")
    lines.append(f"round:  {meta['num_instances']} questions ({meta['num_failed']} FAILED, "
                 f"overall failure rate {meta['overall_failure_rate']:.3f})")
    lines.append(f"source_runs: {', '.join(meta['source_runs'])}")

    rows = feedback["focus_strategies"]
    total_examples = sum(len(r["few_shot_failures"]) for r in rows)
    lines.append(f"selected {len(rows)} focus strategies, {total_examples} few-shot failure examples total")
    for r in rows:
        lines.append(f"  [{r['failure_rate']:.2f} fail | {r['num_questions']:>2} q | "
                     f"{r['share'] * 100:>4.0f}% | {len(r['few_shot_failures'])} ex | "
                     f"{r['selected_for']:>20}] {r['description'][:60]}")

    skipped = feedback.get("ineligible_strategies") or []
    if skipped:
        lines.append(f"skipped {len(skipped)} ineligible strategies")
        for r in skipped:
            lines.append(f"  [{r['failure_rate']:.2f} fail | {r['num_questions']:>2} q | "
                         f"{r['share'] * 100:>4.0f}% | {'; '.join(r['excluded_for'])}] "
                         f"{r['description'][:60]}")

    source_runs = meta["source_runs"]
    if len(source_runs) > 1:
        lines.append("\ncluster comparison (fail_rate n=num_questions, per source_run):")
        header = "  " + "cluster".ljust(9) + "overall".rjust(12)
        header += "".join(run[-18:].rjust(20) for run in source_runs)
        lines.append(header)
        for c in feedback["cluster_comparison"]:
            line = "  " + c["cluster_id"].ljust(9)
            line += f"{c['failure_rate']:.2f} n={c['num_questions']:<3}".rjust(12)
            for run in source_runs:
                b = c["by_source_run"].get(run)
                cell = f"{b['failure_rate']:.2f} n={b['num_questions']:<3}" if b else "-"
                line += cell.rjust(20)
            lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n---")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_argument_group("input (choose one)")
    src.add_argument("--runs", nargs="+", type=Path, default=None,
                     help="research-pipeline run dirs (or parents, or sample_*.json files); "
                          "each dir's name becomes the source_run label")
    src.add_argument("--model", default="drtulu",
                     help="[--dataset-dir] agent under eval_results/real/ (default: drtulu)")
    src.add_argument("--deciding-only", action="store_true",
                     help="[--runs] keep only each result's deciding (challenge) attempt")
    src.add_argument("--include-seed-round", action="store_true",
                     help="[--runs] also keep round-0 questions (the seed tested as-is)")
    src.add_argument("--keep-run-duplicates", action="store_true",
                     help="[--runs] dedup per (run, question) so cross-run repeats are kept")

    clu = ap.add_argument_group("clustering")
    clu.add_argument("--seeds-file", type=Path, default=None,
                     help="known strategies, one per line (default: built-in 8-item menu)")
    clu.add_argument("--assign-method", choices=("llm", "embedding"), default="llm",
                     help="how strategies are assigned to seed/new clusters (default: llm)")
    clu.add_argument("--cluster-provider", choices=("anthropic", "openai"), default="anthropic",
                     help="[llm] provider for strategy clustering (default: anthropic)")
    clu.add_argument("--cluster-model", default=None, 
                     help="[llm] model name; uses provider-specific default if omitted")
    clu.add_argument("--no-merge-new-clusters", action="store_true",
                     help="[llm] skip the pass that merges near-duplicate NEW clusters")
    clu.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                     help=f"[embedding] OpenAI embedding model (default: {DEFAULT_EMBEDDING_MODEL})")
    clu.add_argument("--seed-max-distance", type=float, default=0.6,
                     help="[embedding] cosine distance beyond which a strategy spawns a NEW cluster")

    sel = ap.add_argument_group("selection (failures are GOOD: higher failure_rate is better)")
    sel.add_argument("--min-failure-rate", type=float, default=0.25)
    sel.add_argument("--max-share", type=float, default=0.5)
    sel.add_argument("--min-cluster-size", type=int, default=0,
                     help="drop clusters with fewer questions than this (0 keeps unused ones)")
    sel.add_argument("--max-cluster-size", type=int, default=None)
    sel.add_argument("--rank-by", choices=("underrepresented", "diverse", "failure_rate", "volume"),
                     default="underrepresented")
    sel.add_argument("--no-always-include-new", action="store_true",
                     help="do NOT force-include underrepresented novel clusters that worked")
    sel.add_argument("--examples-per-strategy", type=int, default=5)
    sel.add_argument("--no-instances", action="store_true",
                     help="omit the per-cluster instance lists from cluster_comparison")

    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = ap.parse_args(argv)

    _stage(f"0/3 load: {len(args.runs or [])} run path(s)")
    examples = load_examples_from_runs(
        args.runs, deciding_only=args.deciding_only,
        keep_run_duplicates=args.keep_run_duplicates,
        include_seed_round=args.include_seed_round)
    _stage(f"  loaded {len(examples)} questions")

    feedback = build_feedback(
        examples,
        load_seed_strategies(args.seeds_file),
        assign_method=args.assign_method,
        cluster_provider=args.cluster_provider,
        cluster_model=args.cluster_model,
        merge_new_clusters=not args.no_merge_new_clusters,
        embedding_model=args.embedding_model,
        seed_max_distance=args.seed_max_distance,
        min_failure_rate=args.min_failure_rate,
        max_share=args.max_share,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        rank_by=args.rank_by,
        always_include_new=not args.no_always_include_new,
        examples_per_strategy=args.examples_per_strategy,
        include_instances=not args.no_instances,
    )
    out_path = write_feedback(feedback, args.out)
    print(format_summary(feedback))
    print(f"\nwrote -> {out_path}")


if __name__ == "__main__":
    main()
