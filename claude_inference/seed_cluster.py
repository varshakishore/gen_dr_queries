"""Seeded clustering of generated-question strategies around a fixed strategy menu.

Instead of clustering the `strategy` field from scratch (EvalTree stage-3 recursive
k-means), this assigns each generated question's strategy to one of the canonical
strategies the generator is prompted with (see research_pipeline.py's "EXAMPLE STRATEGIES
TO CONSIDER") or, if it fits none, a NEW cluster. Every seed cluster keeps its named,
interpretable meaning; novel/off-menu strategies group into new clusters.

Two assignment methods (both produce the same node shape build_strategy_feedback uses):
  * assign_llm (default) — prompt Claude (Opus) to pick, for each strategy, a seed cluster
    or a new named cluster. No embeddings; needs only ANTHROPIC_API_KEY. New clusters found
    in earlier batches are offered to later batches so similar novel strategies merge.
  * assign (embedding) — cosine nearest-seed with a distance threshold, reusing EvalTree's
    cached per-question strategy embeddings (stage2 .bin) and embedding only the ~8 seeds;
    a strategy farther than the threshold from every seed spawns a new cluster.

This module just produces the clusters; build_strategy_feedback.py turns them into the
focus-strategy + few-shot-failure JSON exactly as it does for the EvalTree tree.
"""

import os
from pathlib import Path

import numpy as np
import torch

# Mirrors research_pipeline.py's "EXAMPLE STRATEGIES TO CONSIDER" (items 1-8). Item 9
# ("something else you think of...") is intentionally omitted — that IS the "spawn a new
# cluster" case, handled by --seed-max-distance rather than by a centroid.
SEED_STRATEGIES = [
    "Require synthesis across 5+ sources or clearly disjoint domains (e.g., political science + economics).",
    "Require synthesis across differing viewpoints, stakeholder incentives, or theoretical frameworks.",
    "Require multi-step reasoning, structured argumentation, or hierarchical planning.",
    "Require handling conflicting, incomplete, or low-quality evidence.",
    'Require universal quantification ("for all X, is Y true?") or reasoning about edge cases and exceptions.',
    "Require correcting a hidden misconception or establishing key knowns before answering.",
    'Embed a specific context that changes the answer (e.g., "explain to a policymaker with no ML background").',
    "Make a question that is unanswerable by current research, no existing work is available.",
]

# EvalTree's stage-2 prepends this before embedding each strategy; match it so the seeds
# land in the same region of the space as the questions.
PREFIX = "The model has the following capability: "


def load_seed_strategies(seeds_file: Path | None) -> list[str]:
    """The seed strategy texts: from a file (one per line) or the built-in menu."""
    if seeds_file is None:
        return list(SEED_STRATEGIES)
    lines = [ln.strip() for ln in seeds_file.read_text().splitlines()]
    seeds = [ln for ln in lines if ln and not ln.startswith("#")]
    if not seeds:
        raise ValueError(f"no seed strategies found in {seeds_file}")
    return seeds


def load_question_embeddings(dataset_dir: Path, annotation: str, embedding_model: str) -> np.ndarray:
    """Cached per-question strategy embeddings (EvalTree stage-2), (N, D), dataset order."""
    bin_path = (dataset_dir / "EvalTree" / "stage2-CapabilityEmbedding"
                / f"[annotation={annotation}]_[embedding={embedding_model}].bin")
    if not bin_path.is_file():
        raise FileNotFoundError(
            f"no cached strategy embeddings at {bin_path}; run the EvalTree strategy "
            f"pipeline (stage 2) first so the questions are embedded")
    vecs = torch.load(bin_path)  # list of 1-D tensors, one per dataset instance
    return np.stack([np.asarray(v, dtype=np.float64) for v in vecs])


def embed_seeds(seeds: list[str], embedding_model: str) -> np.ndarray:
    """Embed the seed strategies with the same model+prefix EvalTree used, (K, D)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OpenAI_API_KEY"))
    resp = client.embeddings.create(input=[PREFIX + s for s in seeds], model=embedding_model)
    # `.data` preserves input order.
    return np.stack([np.asarray(d.embedding, dtype=np.float64) for d in resp.data])


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
- If it fits NO seed well, consier assigning it to an EXISTING NEW cluster; only invent a \
brand-new one if none fit.
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
    import json as _json
    a, b = text.find("["), text.rfind("]")
    if a == -1 or b == -1 or b < a:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")
    return _json.loads(text[a:b + 1])


def _llm_text(client, model: str, prompt: str, max_tokens: int) -> str:
    resp = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(getattr(b, "text", "") for b in resp.content)


def consolidate_new_clusters(new_labels: dict, examples: dict, model: str,
                             client, max_tokens: int = 4000) -> dict:
    """Ask the LLM to merge near-duplicate new clusters. Returns {old_index: canonical_index}.

    `new_labels` maps a new-cluster index to its name; `examples` maps the same index to a
    representative strategy (for context). Groups are re-indexed contiguously starting at
    min(new_labels) so the caller can drop them straight back into the assignment.
    """
    import re

    norm = lambda x: re.sub(r"\s+", " ", str(x).strip().lower())
    idxs = sorted(new_labels)
    base = idxs[0]
    label_to_idx = {norm(new_labels[i]): i for i in idxs}
    cluster_block = "\n".join(
        f"- {new_labels[i]} — {str(examples.get(i, '')).strip()[:140]}" for i in idxs
    )
    groups = _parse_json_array(_llm_text(client, model, _MERGE_PROMPT.format(cluster_block=cluster_block), max_tokens))

    remap: dict[int, int] = {}
    canonical_labels: dict[int, str] = {}
    assigned: set[int] = set()
    for g, grp in enumerate(g for g in groups if isinstance(g, dict)):
        canon_idx = base + len(canonical_labels)
        canonical_labels[canon_idx] = str(grp.get("canonical", "")).strip() or new_labels[idxs[0]]
        for m in grp.get("members", []):
            old = label_to_idx.get(norm(m))
            if old is not None and old not in remap:
                remap[old] = canon_idx
                assigned.add(old)
    # Any cluster the model dropped keeps its own (new) group so nothing is lost.
    for i in idxs:
        if i not in assigned:
            canon_idx = base + len(canonical_labels)
            canonical_labels[canon_idx] = new_labels[i]
            remap[i] = canon_idx
    return {"remap": remap, "canonical_labels": canonical_labels}


def assign_llm(strategies: list[str], seeds: list[str], model: str,
               batch_size: int = 25, max_tokens: int = 8000, client=None,
               merge: bool = True) -> dict:
    """Assign each strategy to a seed cluster or a NEW cluster using an LLM (Claude Opus).

    Replaces the embedding nearest-centroid assignment: instead of cosine distance, the
    model reads each strategy and the seed menu and decides seed-vs-new, naming novel
    clusters. The prompt strongly discourages new clusters (prefer a seed, then an existing
    new cluster). Processed in batches; new clusters discovered in earlier batches are
    offered to later batches so similar novel strategies merge (online). When `merge` is
    set (default), a final consolidation pass merges any remaining near-duplicate new
    clusters, so a handful of broad novel themes emerge rather than many tiny ones.

    Returns the same assignment shape build_nodes consumes, plus `new_labels` (cluster
    index -> LLM-given name) and `method`='llm'.
    """
    import os
    import re

    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    n_seeds = len(seeds)
    seed_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(seeds))
    norm = lambda x: re.sub(r"\s+", " ", x.strip().lower())
    new_examples: dict[int, str] = {}  # a representative strategy per new cluster

    assignments: list[int | None] = [None] * len(strategies)
    new_order: list[str] = []          # normalized new-cluster names, creation order
    new_index: dict[str, int] = {}     # normalized name -> cluster index (>= n_seeds)
    new_labels: dict[int, str] = {}    # cluster index -> display label

    def register_new(name: str) -> int:
        key = norm(name) or "unclassified"
        if key not in new_index:
            idx = n_seeds + len(new_order)
            new_index[key] = idx
            new_order.append(key)
            new_labels[idx] = name.strip() or "unclassified"
        return new_index[key]

    indexed = list(enumerate(strategies))
    for start in range(0, len(indexed), batch_size):
        batch = indexed[start:start + batch_size]
        new_so_far = "; ".join(new_labels[new_index[k]] for k in new_order) or "none yet"
        strat_block = "\n".join(f"[{i}] {s}" for i, s in batch)
        prompt = _LLM_PROMPT.format(seed_block=seed_block, new_so_far=new_so_far, strat_block=strat_block)
        for obj in _parse_json_array(_llm_text(client, model, prompt, max_tokens)):
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

    # Any strategy the model skipped -> an explicit "unclassified" new cluster (never dropped).
    if any(a is None for a in assignments):
        unresolved = register_new("unclassified (model gave no assignment)")
        assignments = [unresolved if a is None else a for a in assignments]

    # Consolidation pass: merge near-duplicate new clusters into a few broad themes.
    if merge and len(new_order) > 1:
        merged = consolidate_new_clusters(new_labels, new_examples, model, client)
        remap = merged["remap"]
        assignments = [remap.get(a, a) if a >= n_seeds else a for a in assignments]
        new_labels = merged["canonical_labels"]
        new_order = list(range(len(new_labels)))  # count only; indices are canonical now

    n_clusters = n_seeds + len(new_labels)
    return {
        "assignments": assignments,
        "n_seeds": n_seeds,
        "n_clusters": n_clusters,
        "is_seed": [True] * n_seeds + [False] * len(new_labels),
        "new_labels": new_labels,
        "method": "llm",
        "model": model,
    }


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def assign(question_embs: np.ndarray, seed_embs: np.ndarray, max_distance: float) -> dict:
    """Assign each question to the nearest seed, or a new cluster if all are too far.

    Cosine distance = 1 - cosine similarity. Seed centroids are fixed; new clusters keep a
    running-mean centroid so several off-menu questions can converge into one novel cluster
    rather than each spawning its own. Returns per-question assignment + per-question
    nearest-seed distance (for tuning the threshold).
    """
    q = _normalize(question_embs)
    n_seeds = len(seed_embs)
    centroids = list(_normalize(seed_embs))   # index 0..n_seeds-1 are the fixed seeds
    is_seed = [True] * n_seeds
    new_sums: dict[int, np.ndarray] = {}       # running vector sum for new clusters

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
    }


def build_nodes(dataset: list[dict], results: list[int], seeds: list[str],
                assignment: dict) -> list[dict]:
    """One node per cluster, in the shape build_strategy_feedback consumes.

    EVERY seed strategy gets a node even if no question landed on it (size 0), so the
    comparison always shows the full seed menu — an empty seed is itself a signal that the
    generator never used that strategy. New clusters (off-menu strategies) are appended and
    described by a representative member (tagged [novel]).
    """
    n_seeds = assignment["n_seeds"]
    members: dict[int, list[int]] = {}
    for i, c in enumerate(assignment["assignments"]):
        members.setdefault(c, []).append(i)

    def make_node(node_id: str, description: str, leaves: list[int]) -> dict:
        size = len(leaves)
        num_failed = sum(results[i] for i in leaves if 0 <= i < len(results))
        return {
            "node_id": node_id,
            "depth": 1,
            "description": description,
            "num_questions": size,
            "num_failed": num_failed,
            "failure_rate": (num_failed / size) if size else 0.0,
            "leaves": leaves,
        }

    new_labels = assignment.get("new_labels", {})

    nodes: list[dict] = []
    # All seed clusters, in menu order, including empties.
    for c in range(n_seeds):
        nodes.append(make_node(f"seed.{c + 1}", seeds[c], members.get(c, [])))
    # New clusters (always non-empty — a cluster only exists because a question spawned it).
    # Prefer the LLM-given cluster name; otherwise describe by a representative member.
    for new_ordinal, c in enumerate((c for c in sorted(members) if c >= n_seeds), start=1):
        leaves = members[c]
        label = new_labels.get(c)
        if label:
            description = f"[novel strategy not in seed menu] {label}"
        else:
            rep = dataset[leaves[0]].get("strategy", "").strip().replace("\n", " ")
            description = f"[novel strategy not in seed menu] e.g. {rep[:160]}"
        nodes.append(make_node(f"new.{new_ordinal}", description, leaves))
    return nodes
