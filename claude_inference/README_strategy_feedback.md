# Strategy Feedback Module

This module, when called (`build_feedback`), does the following:

  1. CLUSTER   each input question's `strategy` text against a set of KNOWN (seed)
               strategies (the ones the generator is prompted with) or into a NEW cluster
               if it matches no seed. Two methods:
                 * "llm"       — LLM reads each strategy and picks seed-vs-new, naming
                                 novel clusters; near-duplicate new clusters are then merged
                                 into broad themes. Defaults to claude-opus-4-6. This
                                 needs ANTHROPIC_API_KEY only.
                 * "embedding" — Not currently used but keeping as an option. Computes
                                 cosine nearest-seed with a distance threshold; anything
                                 farther than the threshold from every seed spawns a new
                                 cluster (new clusters keep a running-mean centroid, so
                                 several off-menu questions converge instead of each
                                 spawning its own).
  2. SCORE     every cluster: how many of its questions the answering agent FAILED.
               FAILURE IS GOOD HERE: a failed question is one the generator successfully
               made hard. Each cluster is also broken down per `source_run`, i.e. per
               generation prompt (exploit vs explore), so prompts can be compared strategy 
               by strategy.
  3. RANK      report EVERY cluster (no filters) each with its statistics, ordered by field
               in `rank_by`. Few-shot examples are attached to each cluster, only questions 
               that FAILED, up to 5 per strategy when available.

No selection thresholds are applied here: the module reports the numbers and the caller
can later decide what to do with each cluster.

## Output

`build_feedback` returns (and `--out` writes) a JSON dict with `meta`, `cluster_comparison`
(every cluster with its instances, broken down per `source_run`) and `strategy_clusters` —
the full ranked list, one entry per cluster:

| field | meaning |
| --- | --- |
| `rank` | position in the `rank_by` ordering (1 = first) |
| `cluster_id` | `seed.<n>` or `new.<n>` |
| `description` | the seed strategy text, or `[novel strategy not in seed menu] <name>` |
| `is_seed` | `true` = off the seed menu, `false` = discovered as novel |
| `num_questions` | cluster size; `0` = the generator never tried this strategy this round |
| `num_failed` | questions in the cluster the agent FAILED (higher is better) |
| `num_not_failed` | the rest of the cluster (`num_questions - num_failed`) |
| `failure_rate` | `num_failed / num_questions` |
| `share` | the cluster's fraction of this round's questions |
| `score` | `failure_rate * (1 - share)`, the `underrepresented` ranking score |
| `by_source_run` | the same counts per generation prompt |
| `few_shot_failures` | up to `--examples-per-strategy` FAILED questions from the cluster |

`--rank-by` only sets the ORDER of that list:

  * `underrepresented` (default) — by `score`: strategies that fail a lot yet are RARE this
    round, i.e. working approaches the generator under-uses.
  * `diverse` — interleaves a failure-rate ranking with a volume ranking, for a spread of
    cluster sizes.
  * `failure_rate` / `volume` — pure failure rate, or pure absolute number of failures.

Clusters with no questions this round have nothing to score, so they land at the bottom of
every ordering. They're still reported, since an untried strategy is untested rather than
unproductive.

## Example run

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"  # if using embedding clustering

python strategy_feedback_module.py 
  --runs runs/sqa_50_100_explore runs/sqa_50_100_original 
  --seeds-file SEED_STRATEGIES.txt     # default
  --assign-method llm                  # default
  --cluster-model claude-opus-4-6      # default
  --rank-by underrepresented           # default, sets the ORDER only
  --examples-per-strategy 5            # default
  --out round1_strategy_feedback.json
```

OR simpler, using defaults:

```bash
python strategy_feedback_module.py --runs runs/sqa_50_100_explore runs/sqa_50_100_original --seeds-file SEED_STRATEGIES.txt --out round1_strategy_feedback.json
```

## Feedback viewer

To view the clusters, run:

```bash 
python feedback_viewer.py round1_strategy_feedback.json  # or whatever the name is of the output json
```

This produces `round1_strategy_feedback.html`, which you can view in the browser.
