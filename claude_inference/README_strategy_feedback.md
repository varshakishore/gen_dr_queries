# Strategy Feedback Module

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
                                 as the original EvalTree pipeline does. Needs OpenAI_API_KEY.
  2. SCORE     every cluster: how many of its questions the answering agent FAILED.
               FAILURE IS GOOD HERE: a failed question is one the generator successfully
               made hard. Each cluster is also broken down per `source_run`, i.e. per
               generation prompt (exploit vs explore), so prompts can be compared strategy 
               by strategy.
  3. SELECT    the focus strategies to target next round (default ranking: works often or
               reasonably often but is currently RARE), and attach few-shot examples sampled 
               from the inputs. Default to selecting only questions that FAILED, up to 5 per
               strategy, so they demonstrate the strategy working.

## Example run

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_KEY="your-key-here"  # if using embedding clustering

python strategy_feedback_module.py 
  --runs runs/sqa_50_100_explore runs/sqa_50_100_original 
  --seeds-file SEED_STRATEGIES.txt     # default
  --assign-method llm                  # default
  --cluster-model claude-opus-4-1      # default
  --min-failure-rate 0.25              # default
  --max-share 0.5                      # default
  --min-cluster-size 0                 # default
  --rank-by underrepresented           # default
  --examples-per-strategy 5            # default
  --out round1_strategy_feedback.json
```

OR simpler, using defaults:

```bash
python strategy_feedback_module.py --runs runs/sqa_50_100_explore runs/sqa_50_100_original --seeds-file SEED_STRATEGIES.txt --out round1_strategy_feedback.json
```

OR using OpenAI model for clustering:

```bash
python strategy_feedback_module.py --runs runs/sqa_50_100_explore runs/sqa_50_100_original --cluster-provider openai --cluster-model gpt-5.6-luna --out round1_strategy_feedback.json
```

## Feedback viewer

To view the clusters, run:

```bash 
python feedback_viewer.py round1_strategy_feedback.json  # or whatever the name is of the output json
```

This produces `round1_strategy_feedback.html`, which you can view in the browser.
