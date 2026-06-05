# Research State — Hard Question Generation for DR-Tulu

## Goal
Automatically generate research questions that **break** a deep-research system
(DR-Tulu, an 8B model whose only tool is Semantic Scholar / "S2" search). For each
seed question we iteratively rewrite it to be harder until the system produces an
answer that fails a judge — surfacing the system's reasoning weaknesses.

## Components

### 1. `research_pipeline.py` — single-seed loop
For one seed, loops up to `--max-attempts` (default 5):
1. **Make harder** — Claude rewrites the seed into a harder question + a concrete
   `verification_criterion` (system prompt is cached; on rounds 2+ it gets prior
   attempts as feedback).
2. **Answer** — POSTs the question to the DR-Tulu server at `localhost:8007/ask`,
   which returns `{answer, trace}` (the trace holds S2 search calls + documents).
3. **Judge** — Claude judges the answer against the criterion → `PASSED`/`FAILED`.
4. **FAILED** → stop (`FAILED_FOUND`: we broke it). All **PASSED** → `EXHAUSTED`.
   Any exception → `ERROR`.

Tracks per-call token cost (Sonnet 4.5 default; prompt caching on the make-harder
system prompt). The full `answer` + `trace` are saved only in the results JSON; the
JSONL call-log redacts them (keeps latency/tokens/cost metadata).

### 2. `research_pipeline_parallel.py` — batch runner
Runs `research_pipeline.py` as one subprocess per seed, N at a time (`--concurrency`,
default 5). Seeds come from CLI args, a `.txt` file, or (default) the HF dataset
`allenai/asta-user-interactions` (`optin_queries`/train, filtered to `tool=sqa`),
capped at `--limit`. Per seed it writes into `--out-dir`:
- `sample_NNN.json` — the result (same schema as a single-run `--output`)
- `run-sample_NNN.jsonl` — that seed's per-call log
- `sample_NNN.console.txt` — **only if the run failed** (holds the traceback)
- `index.json` — manifest: index→seed→status, per-seed cost/attempts, run totals,
  and avg make-harder calls over FAILED_FOUND seeds

Skips seeds whose `sample_NNN.json` already exists (resumable; `--no-skip-existing`
to force).

### 3. Viewer — `summarize_run.py` + `view_answer.py` + `cite_utils.py`
- **`summarize_run.py <run_dir>`** — prints a console summary (status counts, cost,
  attempts-to-break histogram, error breakdown) and writes `report.html`: a
  dashboard with one foldable card per seed (final question, strategy, criterion,
  judge reasoning), each linking to its answer page. Also generates all answer pages.
- **`view_answer.py`** — renders a DR-Tulu answer as readable HTML: each attempt is
  foldable (last/decisive one open by default); inline `<cite id="...">` tags become
  numbered `[n]` links (hover shows paper title + abridged snippet), resolved against
  the trace into a linked **References** list (→ Semantic Scholar) plus a "Searches
  run" trail.
- **`cite_utils.py`** — shared citation-resolution logic used by both.

Citation ids are `<tool_call_id>-<doc_index>`, resolved via
`trace.tool_calls[*].documents` (+ `raw_output.data[*].paper` for authors/corpusId).

## Commands

```bash
# Prereqs: ANTHROPIC_API_KEY set; DR-Tulu server running on localhost:8007.

# Single seed (writes results + log to ./logs unless --output given)
python research_pipeline.py "external memory in LLMs" --output test.json

# Batch — 10 seeds from the HF dataset, 5 in parallel
python research_pipeline_parallel.py --out-dir runs/sqa10 --limit 10 --concurrency 5

# Batch — explicit seeds or a .txt file
python research_pipeline_parallel.py "seed one" "seed two" --out-dir runs/exp1
python research_pipeline_parallel.py --seeds-file seeds.txt --out-dir runs/exp1

# Summarize a run + build the HTML report and answer pages
python summarize_run.py runs/sqa10
open runs/sqa10/report.html

# (Optional) render answer pages directly, without the report
python view_answer.py runs/sqa10/sample_001.json     # one file
python view_answer.py runs/sqa10                      # whole run -> answers/
```

## Current status
- Pipeline + parallel runner + viewer all working.
- Latest run: `runs/test_sqa50` (50 SQA seeds, Sonnet 4.5, max 5 attempts).
  - **43 FAILED_FOUND (86%)**, 7 ERROR; total ~$2.75.
  - Avg ~1.77 make-harder calls per FAILED_FOUND seed; 20/43 broke on attempt 1.
  - Winning strategies cluster into: **false-premise/planted-error traps**,
    **reconcile-conflicting-evidence**, and **unanswerable/no-causal-isolation**
    (~84% of breaks). Softer prompt strategies (audience framing, broad synthesis)
    rarely win.

## Known issues
- **7 ERRORs are pipeline crashes, not interesting failures** — mostly
  `list index out of range` (Claude response with no text block at
  `resp.content[0].text`) and one JSON parse error in `extract_json`. Worth
  hardening with empty-response guards + a retry/repair on bad JSON.
- High first-attempt break rate warrants spot-checking that the judge isn't
  over-failing reasonable (hedged-but-correct) answers — easy to inspect in the
  answer pages.
- Prompt caching only helps the make-harder system prompt; the judge's bulk is the
  unique answer and can't be cached.

## Next steps — strategy diversity & finding new failure cases

### Problem
Generated questions collapse onto ~3 strategy families (false-premise traps,
reconcile-conflicting-evidence, unanswerable/no-causal-isolation = ~84% of breaks).
Root causes in `research_pipeline.py`:
1. The 3 few-shot examples (lines ~98-123) **are** those 3 dominant families —
   few-shot anchoring overpowers the "don't default to the examples" rule.
2. Stop-at-first-FAILED means ~half the seeds (20/43) only ever try ONE strategy —
   the loop stops exploring the moment it wins.
3. Each seed is independent — no run-wide anti-redundancy pressure.
4. The system profile ("bad at complex reasoning") nudges only toward reasoning traps.

### What does NOT help (and why)
- **Feeding citation/trace snippets to the judge** improves *measurement fidelity*
  (judge can verify attribution instead of guessing) but does **not** generate new
  generator strategies — it doesn't touch the generator's objective.
- **Targeting citation faithfulness** is a poor source of *new* failures: attribution
  is a standard, already-rewarded dimension DR-Tulu was trained on, so it's likely
  hardened. New failures live on axes **outside** the training/eval reward, not on
  ones already optimized.

### Levers to add diversity
- **Mandate a strategy family per attempt** (rotate a taxonomy) instead of letting
  the model free-pick. Biggest lever, low effort.
- **Rotate/expand the few-shot pool** (~12 examples across families, sample 2-3 per
  call from the mandated family) so the prompt stops anchoring to 3 clusters.
- **Explore past first failure** — for diversity studies, run K mandated-diverse
  attempts per seed regardless of pass/fail and record every failure, mapping the
  failure surface instead of stopping at the first hit.
- **Cross-seed coverage**: track family usage run-wide and steer toward under-used
  families (or round-robin); optionally novelty rejection-sampling on question
  embeddings.

### Candidate NEW failure axes (likely outside DR-Tulu's reward)
- **Cross-document numerical reasoning** — derive/aggregate a quantity from numbers
  spread across papers (citation rewards don't check arithmetic).
- **Absence/negation reasoning** — "what does *no* study find / which expected result
  is missing" (retrieval rewards optimize for presence, not gaps).
- **Multi-hop composition** — answer needs chaining 3+ facts, none citable alone
  (per-sentence citation rewards don't enforce the chain).
- **Self-consistency** — answer contradicts itself across sections, each part cited.
- **Retrieval-tool stress** — evidence exists but S2 keyword search can't surface it
  (obscure terminology, very recent, cross-lingual); tests the tool, not reasoning.

### Meta-method: bottom-up strategy discovery
Don't enumerate strategies top-down. After each batch, **cluster the judge's
`other_issues`**; any cluster that isn't already a strategy becomes a candidate new
family fed back as a mandated strategy. Makes discovery self-renewing.
- Data point: in `runs/test_sqa50`, `other_issues` (390 flags) are dominated by
  **over-claiming vs. evidence (~26%)** — presenting correlational/observational/
  modeling results as causal, conflating mismatched comparisons, quantitative
  overreach — plus **missing-gap acknowledgment (~15%)** and stylistic
  **verbosity (~20%)**. The over-claiming cluster is *not* yet a generation strategy
  and is a strong candidate next family.
