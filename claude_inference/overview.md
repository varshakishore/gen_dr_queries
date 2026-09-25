# Overview — hard-question generation & evaluation pipeline

Three scripts form a loop that **generates hard research questions, tests them against a
deep-research system, and reports on the results**:

| Script | Role |
| --- | --- |
| `research_pipeline.py` | The engine. Processes one (or a few) seed questions end-to-end. |
| `research_pipeline_parallel.py` | The driver. Fans the engine out over many seeds as subprocesses. |
| `research_loop.py` | The outer loop. Runs the driver in rounds, re-deriving both strategy menus from `strategy_feedback_module.py` after each round. |
| `summarize_run.py` | The reporter. Turns a run directory into console stats + an HTML dashboard. |
| `loop_report.py` | The loop reporter. One page across all rounds of a `research_loop.py` run: per-round yield, strategy-menu evolution, verification outcomes. Generates the per-round pages it links to — see §10. |
| `feedback_viewer.py` | Renders one round's `feedback.json` as a standalone strategy-cluster page. Called by `loop_report.py`; rarely run by hand. |
| `verify_questions.py` | The post-filter. Re-checks the harvest's verification criteria against retrieved S2 papers and drops the ones that don't hold up. |
| `retrieve_papers.py` | S2 retrieval (port of ScholarQA's RAG stage). Imported lazily by the engine; required only for `--verify-criterion`. |
| `view_answer.py` | Renders per-seed answer pages with citations resolved; called by the reporter. |
| `cite_utils.py` | Turns an answering system's trace into a reference list. Resolves DR-Tulu `<cite id>` tags; for systems with no inline citations it selects sources off the trace instead (§8, §9). Shared by the viewer, the comparison script, the annotation app, and the judge so all of them number citations identically. |
| `llm_client.py` | Provider layer. The one place that knows how Anthropic and OpenAI are called, so every prompt in the pipeline runs on either — see §0. |
| `test_providers.py` | Provider-parity checks against fake SDK clients. No keys, no network, no spend: `python test_providers.py`. |
| `annotate_answer.py` | Renders one generated question as a standalone human-annotation page (ratings, comments, rewrite; answers exported as JSON from the browser). |
| `filter_queries.py` | Seed prep. Labels raw `asta-user-interactions` queries against a usability rubric (English / clarity / research-question / request-type) and publishes the labels as an HF dataset. Stands outside the loop — see §5. |

Goal: find questions a deep-research system (Dr. Tulu, Tongyi, WebThinker, …) **fails** — the
`FAILED_FOUND` status is the success case.

---

## 0. `llm_client.py` — Claude or GPT

Every LLM call in the pipeline goes through this module, so the generator, the judge,
the meta-judge, the query decomposer, and the strategy clusterer each run on Claude or
on an OpenAI model. **Provider is inferred from the model id** — `gpt-*` / `o3-*` /
`chatgpt-*` route to OpenAI, everything else to Anthropic — so switching is just a
`--model`, with nothing new to thread through the three subprocess layers that already
forward it.

**The default is `gpt-5.6-terra`** (`research_pipeline.DEFAULT_MODEL`), so a run needs
`OPENAI_API_KEY` and nothing else. Every entry point reads that one constant — it used to
be a literal repeated in four places — and `CLAUDE_MODEL` survives as an alias because
`eval_other_model.py` and `test_providers.py` import it by name. `--model claude-sonnet-4-5`
moves the whole run back to Anthropic.

```bash
python research_pipeline_parallel.py --out-dir runs/gpt_test --limit 20 --model gpt-5.6-terra
python research_loop.py --out-dir runs/loop_gpt --total 50 --model gpt-5.6-terra
```

`--provider {auto,anthropic,openai}` overrides the inference for an id it cannot
classify (a self-hosted OpenAI-compatible endpoint). `--reasoning-effort
{minimal,low,medium,high}` sets `reasoning_effort` on OpenAI calls; unset leaves the
model's own default, and Anthropic models ignore it. Both flags exist on
`research_pipeline.py`, `research_pipeline_parallel.py`, `research_loop.py`,
`verify_questions.py`, and `eval_other_model.py`. The API-key check at startup asks for
whichever key the chosen models actually need, naming the model and what it is for.

**Models can be mixed.** `--decomposer-model` puts `retrieve_papers`' query
decomposition — a small structured-extraction step, ~4% of a verified run's spend — on a
cheap model independent of `--model`; `research_loop.py --cluster-model` /
`--cluster-provider` does the same for strategy clustering (`--cluster-provider auto`,
the default, follows `--cluster-model`'s id, or `--model`'s when no cluster model is set,
so the whole loop moves together). Each entry point requires the key for every provider
it will actually reach.

**One `--model` moves the whole run.** Every model a run selects by default follows
`--model`'s provider, so `--model gpt-5.6-terra` needs `OPENAI_API_KEY` and nothing else
— generation, judging, the meta-judge, **query decomposition**
(`retrieve_papers.DEFAULT_DECOMPOSER_MODELS`, `gpt-5.6-terra` on OpenAI), and strategy
clustering (`DEFAULT_OPENAI_CLUSTER_MODEL`) all land on the same side. The decomposer
defaults are **tier-matched** — `claude-sonnet-4-5` ($3/$15 per M) against
`gpt-5.6-terra` ($2/$12), both sonnet-class, per ScholarQA's original choice — so a
provider switch does not also drop a capability tier on the step that picks the papers.
`--decomposer-model gpt-5.6-luna` trades that down deliberately if you want it;
decomposition is only ~4% of a verified run's spend, so it buys little. Startup prints the
decomposer it resolved, and `retrieval.decomposer_model` records it per call.

Mixing is still available per step (`--decomposer-model`, `--cluster-model`); a
cross-provider override then requires both keys, and startup says which and why:

> `error: ANTHROPIC_API_KEY … is not set (needed for model 'claude-haiku-4-5').
> --decomposer-model 'claude-haiku-4-5' is on anthropic while --model 'gpt-5.6-terra' is
> on openai, so this run needs both keys — export the one above, or drop
> --decomposer-model to keep the run on openai alone.`

**Comparability caveat.** The decomposer chooses the search queries and filters, so it
decides which papers come back — which changes what the meta-judge sees and therefore
which criteria survive. Verified runs are comparable within a provider; across providers
the retrieval differs too, not just the judging. Pass `--decomposer-model` explicitly to
hold that fixed while changing `--model`.

Two provider differences are handled inside `llm_client` because getting either wrong
is silent:

- **Token budgets.** OpenAI reasoning models spend `max_completion_tokens` on hidden
  reasoning *as well as* visible output, so the pipeline's Claude-sized 800–3000
  budgets truncate to empty content. `budget_tokens` scales them (×4, floor 8000);
  output is billed on tokens produced, so the ceiling is free. A call that still burns
  its whole budget thinking raises an error naming the cause rather than "empty
  response".
- **What `input_tokens` means.** Anthropic reports cache reads *outside*
  `input_tokens`; OpenAI's `prompt_tokens` *includes* its `cached_tokens`.
  `normalize_usage` subtracts the cached count so one `price_call` formula bills both —
  without it, cached OpenAI input would be charged at the full *and* the discounted
  rate. Prompt caching itself needs no flag on either side: Claude gets an explicit
  `cache_control: ephemeral` on the static system prompt, OpenAI caches prefixes over
  ~1024 tokens on its own and reports the hits.

`MODEL_PRICING` is the single cost table for both providers (`filter_queries.py` imports
it rather than keeping its own). Unknown models still price at $0 with a one-time
stderr warning, so add an entry when adding a model. The JSONL log's `claude_call`
records — the `kind` string is unchanged, so old logs and the reporters still parse —
now carry a `provider` field, plus `reasoning_tokens` in `usage` for OpenAI calls;
`index.json` records the run's `provider` alongside its `model`.

---

## 1. `research_pipeline.py` — per-seed loop

For each seed question, rounds `0 … max_attempts` are run; the loop stops as soon as a
terminal condition is hit.

**Round 0 — is the seed already hard?**
Claude generates one atomic verification criterion for the seed as-is
(`PROMPT_FOR_SEED_CRITERION`; underspecified seeds get the "any non-empty answer is
acceptable" escape hatch). The unmodified seed is sent to the research server and judged.
If it already FAILS → status `ALREADY_HARD`, stop.

`--skip-seed-round` starts at attempt 1 instead. No seed can then come back `ALREADY_HARD`,
every seed yields a *generated* rewrite, and one research call plus one seed-criterion call
per seed are saved. That matters for the outer loop: round-0 attempts carry no strategy and
are excluded from clustering, so a round where most seeds come back `ALREADY_HARD` feeds the
feedback step nothing and the menus go un-rewritten. The cost is losing the "already hard"
signal — a seed the system already fails is hardened anyway and may overshoot into
unanswerable. Measured on 24 seeds: 67% `FAILED_FOUND` with the flag vs 42% without.

**Rounds 1..N — make it harder**

1. **Generate** (`harder_question_gen`) — Claude rewrites the seed into a harder question,
   returning `brainstorming`, `chosen_strategy`, `updated_question`, `why_harder`, and a
   `verification_criterion`. Every prior attempt is fed back as context so each round
   escalates rather than rephrases, and **an attempt can fail two ways**, which need
   opposite fixes:

   - the judge **PASSED** it — the question was too easy, so make it harder;
   - the meta-judge **rejected the criterion** — the question was never graded at all, so
     telling the model to escalate difficulty would push it away from the real problem.

   Each block names its own failure (`Why this attempt failed: …`) so a mixed history does
   not leave the model guessing which reason applies to which attempt. The rejected block
   carries `main_correctness_problem` (falling back to `reasoning`); the passed block
   carries the judge's verdict, summary and `other_issues`. Branching on
   `rec.judgment is None` is also what keeps the builder from dereferencing a `None`
   judgment on a rejected attempt.
2. **Verify the criterion** (optional, `--verify-criterion`) — before paying for a research
   call, S2 papers are retrieved for the harder question (`retrieve_papers`) and Claude acts
   as a *meta-judge* on whether the criterion is itself factually correct.

   **Criterion-aware retrieval** (`--verify-propose-queries`, off by default). Retrieval is
   driven by `retrieve_papers(question)` — the *question* — but what gets judged is the
   *criterion*, which routinely adds requirements the question never states: a named entity,
   a required distinction, a demanded study design ("metrics obtained through a controlled
   before/after on the same codebase"). Question-derived decomposition produces topic
   queries and will never search for study design, so those claims never reach the search
   engine. With the flag on, `PROMPT_TO_PROPOSE_VERIFY_QUERIES` asks for up to
   `--verify-max-extra-queries` (default 3) searches targeting exactly that, and they run
   alongside the decomposed query. `proposed_queries` records them per check: `null` means
   the step was off, `[]` that it ran and proposed nothing — which the prompt says should be
   common, since an empty list is the right answer whenever the question's own retrieval
   would already surface what the criterion asserts.

   The extra hits **widen the candidate pool without enlarging the result**: everything is
   reranked together against the original question and cut to `--verify-n-papers`, so extra
   queries change *which* papers the meta-judge sees, not how many. The verify prompt — 71%
   of pipeline spend at a ~19k-token median — does not grow; the only new cost is one small
   call, measured at roughly a tenth of a verify call. Queries that merely restate the
   question dedupe to zero new papers, and a query whose search fails is recorded with its
   error rather than sinking the check. A proposer failure degrades to question-only
   retrieval instead of dropping the seed.

   Two cautions. It **wants a reranker**: the design rests on reranking picking the best
   papers out of a larger pool, and reranking degrades silently to none without
   `--reranker-url` (see below), leaving the merge in arbitrary order. And it carries a
   **confirmation-bias risk** — retrieval that goes looking for the criterion's claims is no
   longer neutral the way question-driven retrieval is. The prompt guards against it in the
   instructions rather than the output schema: queries must *discriminate* between the
   criterion being right and wrong rather than supply background, absence/consensus/causality
   claims should be searched for contradictory evidence, and the criterion is explicitly not
   to be assumed correct. The schema itself is just `{"query": ...}`. Runs with and without
   the flag are not comparable: it changes the evidence, not just the judging.

   The check runs on rounds 1+ only; `--include-seed-round` extends it to round 0. Criteria
   that are just "Any non-empty answer is acceptable" are skipped either way — they state no
   fact to verify and `JUDGE_PROMPT_TEMPLATE` passes them unconditionally — which is not a
   rare case: 117 of 637 round-0 criteria on disk (18%) are the escape hatch. The check runs
   *before* the research call, so a rejected round-0 criterion costs the seed its
   `ALREADY_HARD` signal — the unmodified seed is never sent — and the run continues at
   attempt 1. `--include-seed-round` and `--skip-seed-round` are mutually exclusive.

   Labels:
   - `correct` → proceed as-is
   - `almost_correct` → proceed, swapping in the meta-judge's `rewrite` (the original is
     kept in `verification_criterion_original`)
   - `incorrect` / `insufficient_evidence` → **retry**: record the attempt, skip the
     research call, and regenerate both question and criterion with the objection fed back.
     The attempt then counts in strategy feedback exactly like a PASSED one (§4)

   **A rejected criterion no longer ends the seed.** It is treated exactly like a PASSED
   judgment — the attempt failed, so try again — because ending there threw away every
   remaining attempt and yielded nothing: 14 of 36 seeds in one measured three-system run.
   The two rejection labels are one failure case here; either way the criterion cannot be
   used, and the generator is told which of the two it was. A rejection still consumes an
   attempt and still costs a full verify call, so raise `--max-attempts` accordingly.
   `CRITERION_INVALID` is no longer emitted anywhere; a seed whose attempts all end in
   rejection finishes as `EXHAUSTED`.

   `almost_correct` was called `partly_correct` until recently. `normalize_criterion_label`
   maps the old name — and `partially_correct`, which a model writes when it paraphrases the
   enum — onto the canonical one, so the ~60 stored checks carrying the old spelling still
   read back and a paraphrase does not silently become an unrecognised label (which would
   stop the seed instead of applying the rewrite).

   **The check judges two things.** Factual validity — are the criterion's claims true? — and
   *gatekeeper necessity*: since a criterion is a requirement EVERY fully correct answer must
   satisfy, a criterion that a plausible correct answer could fail is too restrictive. The
   second is not evidence-grounded the way the first is; there is no literature that settles
   whether a requirement is fair, so it is the meta-judge's opinion riding in the same call.
   `unfair_requirements` records what it found (empty string when nothing), and it is parsed,
   stored, and shown in `report.html`, the answer pages, and `loop_report.py`.

   First measurement (8 questions, `gpt-5.6-terra`): 4 `correct`, 3 `almost_correct`,
   1 `incorrect`; `unfair_requirements` non-empty on exactly the 4 non-`correct` rows; $0.055
   per check. All four `correct` criteria use "must explicitly state …" wording, so the check
   is judging necessity rather than reacting to the phrasing — the worry that prompted it.
   What it flagged was criteria mandating a *specific enumeration or framing*: naming two
   named confounder families, requiring a Jane Jacobs contrast plus two named
   operationalizations, demanding an exhaustive confounder list.

   The meta-judge also emits `additional_queries` — searches it would want *after* seeing
   the evidence, each with a `targets_claim`. Nothing consumes it; it is now the **metric
   for whether `--verify-propose-queries` works**. Across the 275 checks on disk (all made
   with the pre-retrieval step off):

   | | count | share |
   | --- | --- | --- |
   | checks emitting `additional_queries` | 18 (41 queries) | 7% |
   | checks labelled `insufficient_evidence` | 16 | 6% |
   | **overlap** | **16 of 18** | — |

   Every `insufficient_evidence` check asked for more searches, and none of the 18 was
   labelled `correct`. So post-hoc emission is a clean read on retrieval falling short: if
   the pre-retrieval step is doing its job, emission should fall from 7% toward ~2% and the
   `insufficient_evidence` count should drop from 16. If it does not move, the proposed
   queries are not finding anything the question's own retrieval missed. Note the ceiling on
   the *rescue* side is only ~6% of checks — some of those topics are genuinely absent from
   the literature and no query will save them; the broader payoff is better-targeted context
   on the other 94%, which this metric does not capture. Emission is provider-independent
   (`claude-sonnet-4-5` and `gpt-5.6-terra` fire at the same rate), so it reflects the
   corpus and the criteria, not the model.

   **First measurement with the step on** (32 checks, `gpt-5.6-terra`): it does not rescue
   `insufficient_evidence`. Those checks received the *most* extra papers — 603 on average,
   70% of the candidate pool — and still could not settle their claims, which is the ceiling
   above showing up in practice. Two other observations: emission was 32/32 at close to the
   cap despite the prompt saying an empty list should be common (a blind proposer cannot know
   whether the papers will suffice, so it always spends its budget — lower
   `--verify-max-extra-queries` rather than reword), and extras ran 43–71% of the candidate
   pool on every check, so the evidence base is majority criterion-targeted rather than
   question-neutral. No paired A/B has been run yet, so whether the step *helps* the other
   94% is still unmeasured.
3. **Answer** — POST the harder question to the research server (`localhost:8007/ask` by
   default); response gives `answer`, `trace`, `model`, `usage`.
4. **Judge** (`JUDGE_PROMPT_TEMPLATE`) — Claude scores the answer against the criterion and
   flags other issues, returning `PASSED` / `FAILED`.
   - `FAILED` → status `FAILED_FOUND`, stop (we broke the system — the goal).
   - `PASSED` → loop back to step 1 with feedback.

   The judge does not see the raw answer: `format_answer_for_judge` first runs it through
   `cite_utils` (`build_doc_index` → `numbered_plaintext` → `references_block`), turning
   DR-Tulu's opaque `<cite id="<call_id>-<i>">` tags into inline `[n]` markers backed by a
   **References** section with each source's title, URL, authors, and *retrieved snippet* —
   the plain-text twin of what `answers/sample_NNN.html` shows. So the judge can check a
   claim against the source text rather than against DR-Tulu's paraphrase of it, and the
   prompt instructs it to file unsupported citations under `other_issues`. The results file
   still stores the raw answer; only the judge prompt is rewritten.

   **Two judge prompts, picked off the answer.** `format_answer_for_judge` returns
   `(text, inline_cites)`. Systems that emit `<cite>` tags get `JUDGE_PROMPT_TEMPLATE` and
   the References section above. Systems that cannot — WebThinker's refined article contains
   zero URLs and zero `[n]` markers, and Tongyi's carries none either — get
   `JUDGE_PROMPT_NO_INLINE_CITES` and a **SOURCES** block listing what the trace shows was
   consulted (§8, §9). That prompt states plainly that the list is provenance, not claim-level
   attribution, and tells the judge not to treat a claim as unsupported merely because no
   source is attached. Without it the judge marks every claim uncited and fails the answer on
   a formatting mismatch rather than on substance, which would make `FAILED_FOUND` a property
   of the output format instead of the question. Answers with no citations and no usable trace
   pass through unchanged.

   Measured on the first WebThinker run: the judge's `other_issues` were substantive —
   *"The reference to 'Mustafa (2025)' conflicts with the retrieved Mustafa article, which is
   dated 2017"* — with no verdict complaining that inline citations were missing.
   - **Cost**: resolved references roughly double the judge prompt (18k → 37k chars, ~9k
     input tokens, on a representative 10-reference sample). Still small next to
     `--verify-criterion`. `references_block(..., snippet_words=N)` abridges snippets if
     that needs trimming; the pipeline passes them whole.
   - **Gap**: `numbered_plaintext` silently drops a cite id that is absent from the trace,
     so a fabricated citation reaches the judge looking merely uncited rather than broken.
     `render_answer` does track these (it renders `[?]` and returns `missing`) — the
     viewer surfaces them, the judge does not.
   - **Comma-joined ids used to fall into that gap.** DR-Tulu writes several ids in one tag
     with either spaces or commas (`id="a-2 a-3"`, `id="a-2,a-3"`), and both call sites
     split on whitespace alone, so the comma form stayed one unresolvable token. Measured
     over 1,478 stored DR-Tulu answers: 917 of the 1,686 unresolved ids (54%) were
     comma-joined, and 84% of their parts resolve once split — 2,077 citations the judge was
     reading as uncited. `split_cite_ids` (`CITE_ID_SEP = [,\s]+`) now serves both
     `render_answer` and `numbered_plaintext`. It cannot break a real id: no doc-index key
     contains a comma (0 of 26,072) and no comma-joined token ever resolved whole.
     Re-deriving the old runs lifts total resolved references 14,210 → 14,503 and the
     unresolved rate 2.5% → 1.7%; 5.5% of answers gain references (median +2, max +17).
     References are re-derived on read, so this retroactively fixes the viewers and the
     annotation app — but **not** the judge verdicts already on disk, which were made
     against the thinner reference block.

All attempts exhausted without a failure → status `EXHAUSTED`, whatever mix of passed
answers and rejected criteria got there; the per-seed console line says how many attempts
were lost to a rejected criterion. Any exception → status `ERROR` with the message
recorded.

### Configuration knobs

- **Answering-system profiles** (`ANSWERING_SYSTEM_PROFILES`, `--profile`): `drtulu`,
  `tongyi`, `webthinker`. Injected into the make-harder prompt at the
  `{ANSWERING_SYSTEM_PROFILE}` sentinel to describe what the target system is good/bad at.
  **The `webthinker` text is currently a copy of `tongyi`'s** and tells the generator the
  system's only tool is academic-paper search and that difficulty must not require sources
  outside that corpus — WebThinker searches the open web via Serper. That constraint shapes
  every question generated against it.
- **Prompt variants** (`--prompt`): `explore` (default; forbids reusing the example
  strategies, pushing for novel ones) vs `exploit` (keeps an in-context strategy menu).
- **Strategy menus** (`STRATEGY_LISTS`, `--strategies`, `exploit` prompt only):
  `merged_v1` (18, **the default**), `default` (9 — mixed: multi-source synthesis,
  conflicting evidence, false premises, unanswerable questions…), or `jena_cog_biases`
  (13 cognitive-bias traps — survivorship, base-rate neglect, spurious causation,
  proxy/measurement, hindsight, …). `merged_v1` is the default because the per-seed draw
  takes `--max-example-strategies` (6) without replacement, and that only does real work
  when the cap sits well below the pool: 6 of 9 puts most strategies in most menus, 6 of 18
  genuinely varies (measured over 40 seeds: appearance rates 100% down to 15%).
- **Provider** (`--model`, `--provider`, `--reasoning-effort`, `--decomposer-model`):
  Claude or an OpenAI model, picked off the model id — see §0.
- Others: `--max-attempts`, `--server-url`, `--timeout` (raise for slow models),
  `--skip-seed-round`, `--include-seed-round`, `--verify-n-papers`,
  `--verify-max-chars-per-paper`, `--verify-propose-queries`,
  `--verify-max-extra-queries`, `--reranker[-url]`. The two `--verify-propose-*` flags exist
  on `research_pipeline.py`, `research_pipeline_parallel.py`, `research_loop.py` (forwarded
  to both generation and `--verify-after`), and `verify_questions.py`.
- **Reranking** is on by default (`--reranker auto`) but silently degrades to *no reranking*
  when neither `--reranker-url` nor `VLLM_RERANK_URL` is set — only a stderr warning marks
  it. Unreranked, keyword-search papers keep score 0.0 and sort last, and the meta-judge
  ends up retrieving *worse* than the system it is checking, which biases it toward
  accepting "the literature does not cover X" claims.

### Cost of `--verify-criterion`

The check dominates the bill: the retrieved-paper context is ~15× larger than any other
prompt in the pipeline, and it runs on every attempt. Measured over a 20-seed run at
`--verify-n-papers 50 --verify-max-chars-per-paper 8000` (median context ≈36k input
tokens, median latency ~74 s per check):

| | with `--verify-criterion` | without |
| --- | --- | --- |
| cost / seed | ~$0.33 | ~$0.05–0.08 |
| share of spend | 71% verify + 4% decomposition | — |

Lower `--verify-n-papers` to cut it — but note that in that run 34 of 36 checks cited at
least one paper ranked past 20, so the tail is thin rather than unused; ~25–30 papers is
the safer trim than 15.

### Logging & cost

- **JSONL run log** (`--log-dir` → `run-<run_id>.jsonl`): one append-only record per
  external call — `run_start`, `claude_call`, `research_call`, `criterion_check`,
  `verdict`, `run_end` — each with timestamps, latency, full request/response, token usage,
  and per-call cost. Bulky payloads (the research answer/trace, the retrieved paper
  context, and the judge prompt's answer + References block) are redacted from the log;
  they live in the results file.
- **Cost tracking**: `MODEL_PRICING` × usage, with 1.25× for cache writes and 0.10× for
  cache reads. Accumulated into a `CostBucket` per seed and a grand total. The static
  make-harder system prompt is sent with `cache_control: ephemeral` so repeat calls read it
  cheaply. Unknown models price at $0 with a one-time stderr warning.
  - **Gotcha**: `retrieve_papers` makes its own LLM call for query decomposition. Its
    usage is returned (`decompose_usage` / `decomposer_model`), priced, and folded into the
    `CostBucket` and the `criterion_check` record — but it is **not** a `claude_call` log
    record, so summing `claude_call` costs alone understates the total. The verify call
    itself passes `system=None`, so its ~36k-token prompt is billed uncached every time.
  - Cost is lost when the verify call throws: the decomposition bucket is merged only after
    `_call_llm` returns, so `ERROR` seeds report slightly low.
- **Results JSON** (`--output`): `run_id`, `model`, `grand_total_cost`, and per-seed
  `results` with every attempt — `attempt`, `harder`, `answer`, `answer_model`, `trace`,
  `judgment`, `criterion_check`. The answer and the trace are stored on **every** answered
  attempt, not just the deciding one, and identically for all three answering systems since
  they share `query_research_system` (verified: 51/51 DR-Tulu attempts, 8/8 WebThinker).
  Traces dominate the file — ~600 KB/seed for DR-Tulu, ~390 KB for WebThinker — so budget
  150-250 MB for a 90-seed three-system loop before pointing `sync_data.py` at it.
  - **References are not stored.** They are a pure function of `(answer, trace)` and are
    re-derived by `cite_utils` whenever the judge, the viewer or the annotation app needs
    them, so an adapter improvement retroactively upgrades old runs. The one persisted copy
    is `answers/sample_NNN.html`, written by `summarize_run.py` (`loop_report.py` generates
    those pages for any round dir missing them).
  - **The exact judge prompt is not recoverable.** The JSONL redacts the answer and the
    reference block — one judge call stored 3.7k chars with 91k redacted — so the precise
    SOURCES text the judge read is reproducible only by re-deriving it with the same
    `cite_utils` version. Fine for analysis; awkward for explaining one specific verdict.
  - **Server `usage` is dropped.** `query_research_system` returns it but `AttemptRecord`
    has no field for it, so WebThinker's `retrieved_pages` / `explorer_calls` and Tongyi's
    Serper/Jina counts reach the JSONL's `research_call` record but never `sample_NNN.json`
    — they are not joinable per attempt without parsing the log separately.

---

## 2. `research_pipeline_parallel.py` — parallel driver

Runs `research_pipeline.py` once per seed in its own subprocess, `--concurrency` at a time
(default 5) via a `ThreadPoolExecutor`.

**Seeds** come from CLI args, a `--seeds-file` `.txt`, or — if neither — the HF dataset
`varshak1/asta-user-interactions-filtered` (§5), keeping only `usable=true` rows, deduped
and sliced by `--start` / `--limit`. The pool is whatever `filter_queries.py` has labeled
so far (1078 usable of 2000 today), and its row order is not the raw dataset's, so
`--start`/`--limit` windows don't line up with pre-filter runs. Adding the request-type
criterion cut the pool from 1171 to 1078, so a window here covers different seeds than it
did before that relabel.

**Per seed `i`**, inside `--out-dir`:
- `sample_NNN.json` — the pipeline's result file
- `run-sample_NNN.jsonl` — that seed's call log
- `sample_NNN.console.txt` — subprocess stdout/stderr, **only if the run failed**
  (non-zero exit, or status outside `FAILED_FOUND` / `EXHAUSTED` / `CRITERION_INVALID` /
  `ALREADY_HARD`), so the traceback is kept without cluttering clean runs

**`index.json`** maps index → seed → status (the numbered filenames aren't
self-describing) plus per-seed cost/attempts and run totals: `total_cost_usd`,
`total_claude_calls`, `avg_make_harder_calls_failed_found` (averaged over `FAILED_FOUND`
seeds only, where the attempt count is exact), `num_failed_found`.

**Resumability**: seeds whose `sample_NNN.json` already exists are skipped, so re-running
the same command continues an interrupted run (`--no-skip-existing` forces a re-run).
Finished seeds cost nothing on a resume — no research call, no LLM call — and this holds at
loop level too, since a round re-runs the driver per cell and each cell skips what it has.
An `ERROR` seed counts as finished and is **not** retried; delete its file to re-run it.

`scan_existing` reads those files once, up front, before anything is spent:

- **The recovered row is the real one.** Skipping once emitted `status: "SKIPPED"` with no
  cost and no attempts, and `index.json` is rewritten wholesale every invocation — so a
  resume *overwrote* the run's own statuses. Every reporter reads status off the index row
  (`summarize_run.load_run`, `loop_report.side_counts`), so the harvest shrank on paper
  while the results sat intact on disk: `runs/tongyi_test` is really 6/6 `FAILED_FOUND` at
  $0.40 and its index claimed 3 and $0.22. Status, attempts, cost and call count are now
  read back from the result, the row is tagged `resumed: true`, and `SKIPPED` is retired.
  Re-running the driver over an old dir repairs an index damaged this way.
- **Seed/file pairing is checked.** The skip is by *filename* and the numbering is
  positional, so anything that shifts the seed list — `--start`, `--limit`,
  `--feedback-every`, `--prompt-mix`, a system dropped from `--systems-file` — silently
  pairs stale results with different questions, recording the new seed text beside the old
  result. Results carry their own seed, so a mismatch now aborts before the first
  subprocess, naming both questions. That is the one resume failure that corrupts data
  rather than just accounting.
- **A truncated result is visible.** The pipeline's `json.dump` is not atomic, so a kill
  mid-write leaves a half-file that looks done. It reads back as `BAD_OUTPUT` instead of
  being counted as a finished seed. (It is still not re-run — delete it.)

Because the recovered rows carry their original cost, a resumed run's totals — and the
loop's `--budget-usd` accounting — are cumulative over the whole directory rather than
per invocation.

Most pipeline flags are forwarded through: `--model`, `--prompt`, `--profile`,
`--strategies`, `--timeout`, `--max-attempts`, `--server-url`, and the whole
`--verify-criterion` group.

It runs the same startup server probe as the loop (§4) over `--server-url` and
`--reranker-url`, with the same `--skip-server-check` escape hatch — relevant when the
driver is run directly, since under `research_loop.py` the probe has already happened
and the flag is always forwarded.

---

## 3. `summarize_run.py` — reporting

`python summarize_run.py <run_dir>` reads `index.json` + each `sample_NNN.json` (and any
`sample_NNN.compare.json` produced by the Claude+web-search comparison) and emits:

**Console summary** — status counts with percentages, total cost and LLM calls, average
make-harder calls per `FAILED_FOUND` seed, a bar histogram of *attempts-to-break*, an
`ERROR` breakdown bucketed by leading error phrase, and — when comparisons exist —
`overall` / `criterion` win counts vs Claude+web-search.

**`<run_dir>/report.html`** — a dashboard with one expandable card per seed, ordered
`FAILED_FOUND` → `EXHAUSTED` → `ERROR` (color-coded green / amber / red). Each card shows
the seed, final question, chosen strategy, verification criterion (and the pre-rewrite
version if the meta-judge changed it), judge verdict + reasoning + other issues — or, when
the criterion was rejected, the criterion-check label, reasoning, and suggested rewrite.
Cards with more than one attempt include an "All attempts" table (round, verdict, question).

**`<run_dir>/answers/sample_NNN.html`** — a rich answer page per seed rendered via
`view_answer.render_sample`, with citations resolved to linked references; each report card
links out to it. Where a comparison exists, `sample_NNN.compare.html`
(`view_answer.render_compare`) shows the Claude answer plus the judge's reasoning.

---

## 4. `research_loop.py` — outer loop (M questions in rounds of N)

Splits M seeds into rounds of N and, after each round, rewrites the two strategy menus from
strategy feedback before running the next one.

Per round `<out-dir>/round_KK/`:

- the round's seeds are split between the two prompts (`--prompt-mix`, default 0.5 = an even
  split, with which seeds go where randomised — balanced rather than an independent per-seed
  coin flip, so no round leaves one prompt with no evidence), and the driver runs once per
  side into `round_KK/explore/` and
  `round_KK/exploit/` — separate dirs so the `source_run` labels the feedback's per-prompt
  comparison depends on survive (the feedback loader takes the shallowest matching depth —
  `*/sample_*.json`, or `*/*/sample_*.json` when several answering systems nest a level
  deeper — so passing the round dir alone covers every cell, and the `source_run` label
  picks up the system too: see the multi-system note below)
- `example_strategies.txt` / `banned_strategies.txt` — the menus that round actually ran
  with, passed to the pipeline via `--strategies-file` / `--banned-strategies-file`
- `cluster_seeds.txt` — what the *next* feedback call clusters against: every strategy in
  play, menu **and** bans. Deliberately not the example menu — see below
- `few_shots.exploit.json` / `few_shots.explore.json` — the worked examples each prompt was
  shown (`--few-shots-file`; round 0's are `DEFAULT_FEW_SHOTS`, written out like any other
  round so the rounds are diffable)
- `feedback.json` + `feedback.txt` — `build_feedback` over every round so far. Always
  cumulative; the per-round `--feedback-scope last` option is gone, because the ban quota
  below is a lifetime count and is meaningless against a single round's tally

The two menus are rewritten from complementary halves of the same feedback, all of it now
read off `strategy_clusters` (the module reports every cluster with its statistics and
applies no thresholds of its own — selection policy lives here):

| Menu | Source | Prompt it reaches |
| --- | --- | --- |
| `EXAMPLE STRATEGIES TO CONSIDER` | a weighted sample of `--max-example-strategies`, drawn **per seed** from every cluster **not** over the quota | `exploit` |
| `STRATEGIES TO NOT USE` | the base bans, plus every cluster with `num_failed > --ban-after-failures` | `explore` |
| `Here are a few examples:` | failures of the strategies on that prompt's own list — sampled menu for `exploit`, banned for `explore` | both, separately |

**The exploit menu is sampled per seed, not truncated.** Every exploit seed gets its own
draw, written to `round_KK/menus/sample_NNN.txt` and picked up by the driver's
`--strategies-dir` (falling back to the round-wide `example_strategies.txt` for any seed
without one — the explore side, round 0, or `--menu-sample round`). Draw weight is
`((num_failed + 1) / (num_questions + 2)) × (1 − share)`. The Beta(1,1) smoothing matters at
both ends: raw `failure_rate` makes a 1/1 cluster (1.00) outrank a 20/23 one (0.87), and it
scores a never-tried cluster 0.0 — an absorbing state, since weight 0 means it is never
sampled and so stays untried forever. Smoothed, those become 0.67, 0.84, and 0.50 (a real
"unknown" prior). The `(1 − share)` factor throttles ruts continuously — sampling a strategy
raises its share and lowers its own weight next round — which is why no separate
over-representation ban is needed. Sampling also stops mid-ranked clusters starving: under a
top-N cutoff a cluster ranked just past the cap is never tried again, never accumulates
evidence, and can never climb. Per-seed drawing is what makes the weighting do continuous work: across 35 seeds with a
pool of 11 and a cap of 6, every strategy reached some seed's menu, at appearance rates from
80% (weight 0.87) down to 20% (weight 0.19). A strategy missing from one seed's menu is
almost certainly in another's, so nothing sits out a whole round.

**Set the cap below the pool size or the draw is inert.** Taking `k` of `n` without
replacement compresses inclusion probabilities toward `k/n`, so at the old cap of 10 against
a pool of 11 the same 11 strategies appeared in 63–97% of menus — barely distinguishable.
Hence the default cap of 6. The loop prints a warning to stderr whenever the pool is at or
below the cap.

Sampling uses its own RNG stream (`--random-seed` + a fixed offset) so it does not perturb
the explore/exploit seed split. `loop.json` records the pool with each entry's weight, plus
`menu_draw_counts` — how many of the round's per-seed menus each strategy landed in.

**A rejected criterion counts like a PASSED answer.** Both spend an attempt and yield no
benchmark question, so `load_examples_from_runs` keeps a rejected attempt with
`failed=False` — in `num_questions`, out of `num_failed`. `failure_rate` therefore means
*harvested questions per attempt spent on this strategy*, not *questions the system failed
given that it was asked*: rejections sit in the denominator although the answering system
never saw them. That is the quantity the menu wants, because drawing a strategy costs an
attempt either way.

Dropping them — which is what happened before the pipeline started retrying rejections —
made a strategy that reliably generates unverifiable criteria score like one that had never
been tried: it kept the full Beta(1,1) prior of 0.5 and its weight was untouched, while
burning an attempt *and* a verify call on every seed. Two strategies that each harvest 3
questions, one in 10 attempts and one in 18 because 8 criteria were rejected, used to draw
at the same 0.300; they now draw at 0.300 and 0.164. Rejections raise `share` as well, so
`(1 − share)` throttles them too.

Three things deliberately unchanged: the **ban quota** keys on `num_failed`, so a rejection
never retires a strategy; **few-shot examples** are sampled from `failed=True` only, so a
question whose criterion was rejected is never demonstrated to the generator; and an attempt
with no judgment *and* no criterion rejection (a half-written result) is still skipped.
Recomputing feedback over an old run now yields more examples — `runs/loop_3sys` goes from
29 to 43, `num_failed` unchanged at 22 — so `failure_rate` is not comparable across the
change.

**The ban list is a harvest quota, rebuilt from scratch every round.** A strategy is retired
once it has yielded more than `--ban-after-failures` system-breaking questions — enough
questions of that type exist, so push the generator elsewhere. Rebuilding is safe precisely
because the quota re-derives itself: `num_failed` only grows, so a retired strategy is still
over the quota next round without anything being stored. That avoids persisting ban *strings*,
which would silently lapse — novel cluster descriptions are LLM-written and churn completely
between rounds (measured: 0 of 5 survive a round). Consequently `--max-banned-strategies`,
`--ban-min-evidence`, and the oldest-dropped-first FIFO cap are all gone.

Because the quota bans on volume rather than on failure rate, the retired set is by
construction the *best* performers. Two consequences worth watching: a low-yield cluster is
never retired (only throttled by its sampling weight), and `explore`'s few-shot examples —
which are drawn from the banned clusters as negative examples — now demonstrate high-quality
questions under a "do not use these" instruction, so watch for imitation rather than
avoidance.

The few-shot examples are resampled per prompt from the set that prompt needs: `exploit`'s
demonstrate the menu it is told to choose from, `explore`'s demonstrate what it is told to
avoid (its examples are negative examples). Round 0 uses the built-in `DEFAULT_FEW_SHOTS`;
short samples are padded with them; `--static-few-shots` pins them for every round. This
needs `brainstorming` / `why_harder`, which `strategy_feedback_module` now carries through
from the run into `few_shot_failures` and `cluster_comparison.instances`.

The two lists are independent — `exploit` draws from its menu (or a variation of an entry),
so a ban only ever speaks to `explore`, and a strategy may be sampled into one prompt while
banned from the other. The base bans (`--banned-strategies`, the strategies `explore`'s own
few-shot examples demonstrate) stay banned for `explore` every round, but are **not** withheld
from `exploit`: that reason is specific to `explore`, and `exploit` exists to work known-good
strategies. A base ban leaves the exploit pool only by crossing the quota like any other
cluster.

**The open-ended tail is never a cluster.** `merged_v1`'s menus end with an
"or something else you think of" line, and `sample_menu` appends it to every exploit menu as
a fixed line rather than a draw. It used to be added to `cluster_seeds` as well, which made
it a catch-all: a cluster whose description names no strategy, free to absorb the novel
strategies that are the whole point of tracking novel clusters. It could then cross the ban
quota like any other cluster and land on `explore`'s STRATEGIES TO NOT USE — telling the
prompt whose job is novelty not to think of something else, while `sample_menu` went on
offering the same line to `exploit`. Worse, `explore`'s few-shot examples are drawn from the
banned clusters, so a banned catch-all would supply the novel questions it had absorbed as
negative examples. `derive_menus` now drops the tail row at the source, so it reaches
neither the pool, the quota, nor the cluster seeds; round 0 already clustered without it, so
this makes every round consistent rather than adding it back from round 1. Measured before
the fix: in the only two stored feedback files where the tail was a cluster it had absorbed
0 questions of 11 and 0 of 15 — the mechanism was live but had not yet fired.

**Clustering seeds are decoupled from the menu.** `compute_feedback` clusters against
`cluster_seeds.txt` rather than `example_strategies.txt`, because a retired strategy is off the
menu but must stay a stable `seed.N` with a fixed description — otherwise its questions
re-cluster as a fresh `new.N` under a new name each round and the quota loses sight of them
entirely. This also gives the base bans real cluster identities: two of the three are not in
the example menu verbatim, so before this they had no `num_failed` at all and could never
cross the quota.

**Several answering systems in one loop** (`--systems-file`). A JSON list of systems splits
every round between them, equal slices unless an entry sets `"share"`:

```json
[{"name": "drtulu",     "profile": "drtulu",     "server_url": "http://a:8007/ask", "timeout": 900,  "concurrency": 10},
 {"name": "webthinker", "profile": "webthinker", "server_url": "http://b:8007/ask", "timeout": 5400, "concurrency": 2},
 {"name": "tongyi",     "profile": "tongyi",     "server_url": "http://c:8007/ask", "timeout": 7200, "concurrency": 4}]
```

Two nested splits — system first, then prompt within each system's share, so every system
gets its own balanced explore/exploit mix. Cells land in `round_KK/<system>/<prompt>/`, and
per-seed menus in `round_KK/menus/<system>/`: the driver numbers its seeds from 1 on every
invocation, so a shared menu dir would hand one cell the menu drawn for another. Per-system
`timeout` / `concurrency` matter — measured, a question takes ~32 min on Tongyi, ~19 min on
WebThinker and far less on DR-Tulu, and only WebThinker is unbounded server-side. Omit the
flag and nothing changes: one unnamed system, the pre-existing `round_KK/<prompt>/` layout,
and `by_system: null` in the manifest. The split has its own RNG stream, so adding systems
does not perturb the explore/exploit sequence of an earlier run.

Two consequences worth holding onto. **`--profile` reaches the make-harder prompt**, so
questions are generated *for* a system: three systems means three disjoint question sets,
not one set tried three ways (that is `eval_other_model.py`). And **feedback pools failures
across systems** — `num_failed` aggregates breaks against different targets, so the menu
optimises for "breaks something" and the ban quota retires a strategy once it has broken any
system often enough. That is a deliberate choice, on the assumption that failures are similar
across systems.

**Pooled for selection, split for reporting.** The assumption is checkable because
`source_run` carries the cell: `load_examples_from_runs` labels an example
`<system>/<prompt>` when the sample's grandparent is a round dir, so a three-system round
produces six labels rather than two. Cluster statistics are untouched by it — `num_questions`,
`num_failed`, the sampling weight and the ban quota all count a cluster's leaves whatever
their label — while `meta.source_runs`, each cluster's `by_source_run`, and the
cluster-comparison table in `feedback_viewer.py` break down six ways. So "does this strategy
break all three systems or only one?" is answerable straight off `feedback.json` without
changing what the menu does. Single-system runs keep the bare `explore` / `exploit` labels,
so old feedback files read back unchanged.

Finally, the round now divides six ways rather than two — at `--feedback-every 30` that is
5 seeds per cell, so raise it relative to a single-system run.

**Every endpoint is probed before anything is generated.** `check_server_reachable`
(in `research_pipeline.py`) runs at startup over every system's `server_url` plus
`--reranker-url`, and the loop refuses to start if any of them is unreachable, listing
each offender by system name. Without it a dead server is discovered only *after* each
seed has already paid a make-harder call and a criterion check — the two priciest steps,
both of which run before the research call — so a URL typo costs a full round of `ERROR`
seeds. With several systems it is worse than that: one dead endpoint fails its third of
the round while the others finish, leaving a report that looks complete.

The probe is deliberately loose. It `GET`s the URL's *origin* and treats **any HTTP
status as up, 404 included** — DR-Tulu's FastAPI serves only `/ask`, so demanding
`/health` would reject a live server. Only a transport failure (DNS, refused, timeout)
counts as down. A missing scheme is caught without touching the network, since
`requests` raises `InvalidSchema` on `host:8007/ask` — the likeliest typo and the one
that otherwise surfaces as an unhelpful mid-run exception.

`--skip-server-check` bypasses it, and the loop **always** passes that flag down to the
driver: the loop has already probed, so a second probe per cell is waste — and without
the forwarding, `--skip-server-check` on the loop would be silently ignored by the very
subprocesses it was meant to unblock. It is a point-in-time check: a server that dies
mid-run still produces `ERROR` seeds as before. What it catches is the launch-time typo
and the server nobody started.

**`--stop-after-rounds N` pauses the loop for inspection.** It counts rounds that did
*fresh* work in this invocation, not rounds in the run — a round replayed entirely off disk
(every seed recovered by the driver's resume pre-pass, §2) does not consume the budget. So
re-running the identical command walks the loop forward N rounds at a time, with no flag to
bump between invocations. The pause happens *after* the next round's feedback is computed
and written, so the rewritten menus are inspectable during the pause and are reused for free
on the restart. `--verify-after` is deferred while paused — it is meant to run once over a
finished harvest — and `loop.json` records `paused_after_round`. The last round never pauses;
there is nothing to inspect before.

```bash
python research_loop.py --out-dir runs/loop3 --total 90 --feedback-every 30 \
    --systems-file systems.json --stop-after-rounds 1     # run round 0, then stop
python loop_report.py runs/loop3                          # inspect
# re-run the first command verbatim -> round 0 replays off disk, round 1 runs, pause
```

Resumable: the driver skips seeds with an existing `sample_NNN.json`, and a round's feedback
is reused if `feedback.json` exists (`--refresh-feedback` recomputes). `<out-dir>/loop.json`
is the manifest — per round, the seed split, statuses, cost, the menus and cluster seeds used,
the sampled menu with each entry's weight, and the full quota-ban set with the `num_failed`
that triggered each. The ban set is recorded in full every round because it is recomputed: a
ban that flickers (its cluster dissolved and re-formed below the quota) is otherwise
invisible after the fact.

```bash
python research_loop.py --out-dir runs/loop1 --total 50 --feedback-every 10 --concurrency 10
```

---

## 5. `filter_queries.py` — seed filtering (OpenAI only)

Not part of the generate→test→report loop. It prepares *inputs* to it: the raw HF dataset
is full of non-English, one-word, and non-research queries, and this labels which ones are
worth spending a pipeline run on.

Unlike the rest of the pipeline this one has no provider switch — the rubric is an
OpenAI structured-output call — but it prices off the shared `llm_client.MODEL_PRICING`
table, so `--model` accepts any priced OpenAI id without a `--price-in`/`--price-out`.

`Request Type` classifies what the query asks the assistant to *do*, judged by the
instruction rather than the topic: `design` (draft the user's own study, protocol, proposal,
or paper section), `review` (critique, edit, or summarize text the user pasted in), or
`information seeking` (everything else). Only `information seeking` counts as `usable`.
The distinction that needs stating: asking for a survey of the *published literature* is
information seeking, so topic reviews survive — `review` is only for acting on the user's
own material. Of 2000 labeled today, 97 are `design` and 16 are `review`.

Queries over `--max-words` (default 300) are dropped *before* labeling rather than labeled
and thrown away, since a pasted manuscript costs thousands of input tokens to judge. The
window is over kept queries, so `--limit N` still yields N labels — but `--start`/`--limit`
only line up across runs with matching `--max-words`.

Output dataset (default `varshak1/asta-user-interactions-filtered`, split `train`):
`query`, `thread_id`, `english`, `clarity`, `research_question`, `request_type`, `usable`,
`filter_model`. Per-row token counts stay in the JSONL cache and are dropped at push.

**Gotcha**: the run resumes from the destination dataset, and `load_dataset` falls back to
`~/.cache/huggingface/datasets/` when the repo is gone from the Hub — so deleting the
dataset to relabel from scratch silently resurrects the old rows and pushes them back with
`None` in any new column. Clear that cache dir and the JSONL together.

```bash
export OPENAI_API_KEY=...
python filter_queries.py --limit 20 --no-push --max-cost 0.10   # smoke test
python filter_queries.py --limit 2000 --concurrency 16 --max-cost 5
```

---

## 6. `verify_questions.py` — post-hoc criterion filtering

The same check as `--verify-criterion`, run *after* generation instead of during it:

```bash
python verify_questions.py runs/loop1 --out runs/loop1/verified.json \
    --concurrency 8 --verify-n-papers 25 --reranker-url http://spark-9076:8017
```

Inputs are searched recursively, so a loop dir, a round dir, a run dir, or a bare
`sample_*.json` all work. By default it checks only the deciding attempt of each
`FAILED_FOUND` seed — the actual harvest — and `--all-questions` widens it to every graded
rewrite.

| Label | Outcome |
| --- | --- |
| `correct` | kept, criterion unchanged |
| `almost_correct` | kept, criterion replaced by the meta-judge's rewrite (`criterion_original` preserved) |
| `incorrect` / `insufficient_evidence` | dropped |

Writes `verified.json` (every question, with label, reasoning, retrieval meta, cost),
`verified.kept.json` (the filtered set, criteria already swapped in), and `verified.jsonl`
(per-call log). Resumable — a re-run skips questions already labelled and retries errors.

**Post-hoc is still cheaper, for a narrower reason than it used to be.** Inline once ended
the seed on a rejected criterion — 5 of 20 seeds in the first verified run, 14 of 36 in a
later three-system one. It no longer does (§1): the seed retries. What remains is that a
rejection inline still burns an attempt and a full verify call — the most expensive step in
the pipeline — on a question that is then discarded, and the check runs on *every* attempt
whether or not that question turns out to be worth keeping. Run afterwards, the per-question
cost is the same but it is paid only on the harvest, and thresholds can be re-run against a
finished run without regenerating anything. `research_loop.py --verify-after` runs it automatically once the last round
finishes, recording the totals under `verification` in `loop.json`.

---

## 7. `../annotation_app/` — human annotation study

A deployed web app that puts the harvest in front of Prolific workers. Built from
`allenai/skiff-template` (React UI as the root service, FastAPI sidecar) and running at
**dr-annotation.apps.allenai.org**; `annotation_app/README.md` has the deployment and
Prolific setup.

Questions come from `sync_data.py <run_dir>`, which copies `sample_NNN.json` files into
`api/data/` — they are baked into the image, so a new batch is a commit. Only the
**deciding (last) attempt** of each seed is served. Each worker signs in with their
Prolific ID, claims an unclaimed batch of 10 (create-if-absent write, so concurrent
claims cannot collide), and answers three required questions per item: is the
**question** benchmark-worthy, is the **criterion** correct and suitable, and does the
answer **fail** the criterion (yes / no / not applicable) — each with optional comments
and a rewrite box. The answer is shown with citations resolved; the app calls
`cite_utils` server-side rather than reimplementing it, so the app, the judge, and
`view_answer.py` number citations identically. The annotator's own verdict is collected
**before** the LLM judge's assessment is readable, so agreement is not anchored.

Answers autosave and survive leaving and returning. Everything lands in
`gs://ai2-skiff2-dr-annotation-data/`: `annotations/<pid>/<sample>__a<attempt>.json`
(submitted), `drafts/<pid>/…` (in progress), plus `assignments/` and `batches/`. Each
record carries the ratings plus `run_id`, `seed`, `question`, `criterion`,
`chosen_strategy`, and `judge_verdict`, so joining back to the run — and scoring
human-vs-judge agreement — is a groupby. **One batch per worker, no overlap**, so
inter-annotator agreement is not computable as configured.

---

## 8. WebThinker as an answering system

WebThinker (`WebThinker-QwQ-32B` + a Qwen2.5-32B aux model, behind a FastAPI `/ask`) answers
like the others but attributes nothing, so `cite_utils` has to reconstruct the reference list
from its trace.

**The trace is a list, not a dict.** `build_doc_index` dispatches on shape — dict → DR-Tulu,
list of `{search_query, Input, Output, Extracted_info}` → WebThinker, anything else → `{}`.
Tongyi's ReAct message list gets its own branch (§9); before the dispatch existed the viewers
handed it straight to the DR-Tulu walker and raised `AttributeError`. `None`, a bare string, an
empty list, a list of non-dicts and a malformed block are all tolerated now.

**Sources live inside a prompt string.** Each explorer call's `Input` embeds its search
results as numbered JSON blocks — `***Web Page 3:***` followed by `{"id", "title", "url",
"snippet", "page_info", …}`. One block is one search result is one URL. Measured on a real
run: 13 explorer calls, 125 blocks, 84 unique URLs (21 URLs returned by more than one search).

**Page lists are parsed with the separators the model actually writes.** The back-reference
regex took one separator token between numbers, so an Oxford comma ended the list:
`Web Pages 2, 7, and 8` resolved to {2, 7} and page 8 was not merely downgraded to its blurb
but **dropped from the references entirely**, since a call that back-references anything
selects only the pages it named. Separators now repeat (`(?:\s*(?:,|and|&|or))+`) and ranges
are expanded. Over the 15 stored traces that recovered 12 lost ids — 10 to the Oxford comma,
2 to `or` — of which 4 were pages actually present in their call, every one previously
excluded; selected references 815 → 819 and evidence 571,881 → 573,974 chars. The run of
numbers must start immediately after `Web Page(s)`, which is what keeps prose out: in
`Web Page 4 notes ... 3-5 years` the list ends at 4, because " notes" is not a separator.
No `Web Pages N-M` range occurs in the traces on disk, so that half is untested against real
output; a range wider than `WEBTHINKER_MAX_REF_RANGE` (20) contributes only its endpoints,
and `cited &= set(docs)` discards any id the call never had.

**Which sources count as used.** The answer has no citations, so selection comes from
`Extracted_info` — the only text an explorer call passes back to the main reasoner — which
references pages as "(Web Pages 3 and 5)". Calls that reference none get *all* their pages: a
superset, since the real source is in there but cannot be identified. On that run, 9 of 13
calls back-referenced (40 pages) and 4 did not (39 pages); after URL dedupe, 65 references,
36 of them back-referenced. Those 4 calls are not careless — they cite in prose ("according to
Jiang et al. (2020)") rather than by page number. Matching those names against page text was
tried and abandoned: it resolves ~1 call cleanly, matches 5 of 10 pages on another, and finds
nothing on the remaining two, for a ~15% reduction at a real false-positive rate.

**How much text each reference carries.** `snippet` is Serper's blurb (median 148 chars,
always present); `page_info` is the fetched page text (median 2,475 chars, occasionally
absent when a fetch fails). `page_info` is the true analogue of a DR-Tulu snippet — median
1,857 chars over 9,238 of them — while `snippet` is ~12× smaller and is provenance, not
evidence. So back-referenced pages contribute `page_info` truncated at `WEBTHINKER_PAGE_CHARS`
(2000), fallback pages contribute the blurb, and a back-referenced page with no `page_info`
degrades to its blurb.

The cap is truncation, not ranking — every selected page is kept. Uncapped, 36 back-referenced
pages run ~129k chars (~32k tokens), which is 1.7× the largest DR-Tulu judge prompt observed
(median 16.4k chars over 343 judge calls, p95 37k, max 77k). Capped, a whole judge prompt
lands at ~63k chars (~16k tokens), inside DR-Tulu's band. Set `WEBTHINKER_PAGE_CHARS = 0` to
disable truncation.

**Honest limits.** The list says what informed the report, never which source backs a given
sentence, and a 148-char blurb cannot support claim-checking at all. Runs judged this way are
not directly comparable to DR-Tulu runs: same token volume, different kind of evidence.

### Server-side gotchas (`serve/app.py`, outside this repo)

The `/ask` endpoint needed four fixes before it was usable, all silent failures:

- `AskResponse` never received `mode`, a required field → **every request 500'd** after the
  full inference run.
- The constructor passed `trace=` while the field was `TRACE`; pydantic drops the unknown
  kwarg, so the trace was always `[]`. Renamed to lowercase `trace`, which is what
  `query_research_system` reads.
- In `report` mode `item["Output"]` is the **reasoning chain**, not the report —
  `run_web_thinker_report.py` stores the finished article in `seq['article']` and writes it
  only as markdown (`outputs/**/markdown.<split>.*/article_1.md`), never into the output JSON.
- The output file was deleted *before* the response was built, so every failure destroyed its
  own evidence.

Two more worth carrying: the search layer returns `{}` on failure (including HTTP 400
"Not enough credits") and the caller **caches that empty result to disk**, so one credit lapse
poisons `cache/serper_search_cache.json` and keeps returning nothing long after the key is
fixed — 23 of 42 cached queries in practice, producing confident fully-hallucinated reports
that still exit 0. `app.py` now counts `***Web Page N:***` blocks and returns 502 when a run
retrieved fewer than `MIN_RETRIEVED_PAGES` (default 1), which surfaces as `ERROR` rather than
a fake `FAILED_FOUND`. `usage` carries `explorer_calls` and `retrieved_pages` per call.

Nothing bounds concurrency server-side: each `/ask` spawns a full WebThinker run against the
shared vLLM servers. A report-mode answer takes ~19 minutes, so `--timeout` must be raised
well above its 600s default.

---

## 9. Tongyi DeepResearch as an answering system

Tongyi (`Tongyi-DeepResearch-30B-A3B` behind a FastAPI `/ask`) is a ReAct agent: a message
list of `{role, content}` where assistant turns carry `<tool_call>` and user turns carry
`<tool_response>`. Like WebThinker it emits **no citations at all** — 0 URLs, 0 `[n]` markers,
0 markdown links across 11 answers — so it takes the same `JUDGE_PROMPT_NO_INLINE_CITES` path.

Four response shapes, measured over 127 tool responses:

| kind | format |
| --- | --- |
| `search` | `A Google search for '…' found N results:` → `1. [Title](url)` + snippet |
| `scholar` | `A Google scholar for '…' found N results:` → `[Title](pdfUrl: url)` + `publicationInfo` / `citedBy` |
| `visit` | `The useful information in <url> for user goal <goal> as follows:` + `Evidence in page:` + extracted text |
| error | `Error: Tool call is not a valid JSON…` |

**`visit` is the selection signal, and it is unambiguous.** WebThinker forced us to infer
which sources mattered from back-references, with a superset fallback for calls that made
none. Tongyi simply tells you: a `visit` is the model choosing to read a page. Visits pick
6–8% of the search pool (105 of 1278 across older traces; 18 of 278 on a live run), so there
is no back-reference parsing and no selection fallback — except the degenerate
never-visited-anything case, which falls back to the first `TONGYI_POOL_FALLBACK` (10) search
hits so the judge still sees something. No trace has hit that; the minimum was one visit.

Search and scholar hits build a URL-keyed candidate pool supplying title, blurb and — for
scholar only — authors parsed off `publicationInfo`. Only visited URLs become references.

**Three things the raw output required.**

- **A fifth of visits never reached the page.** 26 of 127 come back as a normal response
  containing a bot wall or an apology — mostly CAPTCHA interstitials on researchgate,
  openreview and pmc. Length cannot discriminate: the longest is **20,550 chars of reCAPTCHA
  page furniture**, an order of magnitude above the 2,182-char median of a real visit.
  `TONGYI_FETCH_FAIL_RE` matches against the opening `TONGYI_FAIL_WINDOW` (400) chars only,
  since a wall announces itself immediately while a page discussing CAPTCHAs would not.
- **Half the visited URLs never appeared in a search result**, so there is no title to
  inherit and PDFs carry no heading. Title resolution is four steps: search/scholar entry →
  first heading in the evidence → a label built from the URL's id (`arXiv:2604.01657`,
  `ACL Anthology 2020.findings-emnlp.309`, `doi:…`, `PMC…`) → the raw URL. That took raw-URL
  titles from 56% to 6%.
- **Every visit body opens with `Evidence in page:`** (127 of 127) — boilerplate that would
  otherwise be the first thing the judge reads in every reference. Stripped.

**Sizing.** Visit evidence is the reference text, capped at `TONGYI_PAGE_CHARS` (4000).

| | refs | reference block |
| --- | --- | --- |
| Tongyi, median over 11 traces | 8 (range 1–21) | **27,240** chars |
| DR-Tulu, over 343 judge calls | 9 | 16,428 median · 37,007 p95 · 76,974 max |

The cap is tail insurance, not a budget necessity: uncapped, Tongyi runs median 35k / max 58k,
already inside DR-Tulu's range — unlike WebThinker, which needed its cap to avoid 1.7× DR-Tulu's
maximum. And the cap is not free here. Visit text is *already condensed* — Tongyi keeps ~6% of
a fetched page, selected against its own goal — so every character survived a relevance pass and
truncation cuts chosen content rather than boilerplate. At 2000 chars a cap would truncate 53%
of references and discard 47% of the text; 4000 truncates 18% and keeps 75%. `0` disables it.

**Two caveats.** The evidence is **self-selected**: the judge sees what the system wanted from
each page, and contradicting material was discarded before anything downstream could see it —
weaker for claim-checking than DR-Tulu's retrieved passages, and weaker than WebThinker's raw
`page_info`. And Tongyi searches hard: **49 Serper calls for one question** on the live run
against WebThinker's 6–11, so it burns search credits roughly 5× faster per seed. One run
reported 19 Jina reads totalling 688k chars (~158k tokens by `cl100k_base`; the server's own
`jina_est_tokens` uses a flat chars÷4 and overestimates by ~8%), of which only **5.9%** reached
the model context.

---

## 10. `loop_report.py` — one page for a whole loop run

`summarize_run.py` covers a single run dir and `feedback_viewer.py` a single round's
clustering. What neither shows is the view *across* rounds, which is the only place the
loop's actual mechanism — menus rewritten from feedback — becomes visible.

```bash
python loop_report.py runs/loop_250
open runs/loop_250/loop_report.html
```

Three sections:

- **Rounds** — `FAILED_FOUND` per prompt side per round, plus seeds, cost, wall time, and
  links out to each side's `report.html` and the round's `feedback.html`. With several
  answering systems (§4) a side's figure is the total across them, broken out per system
  underneath with each system's own `report.html` linked — otherwise three systems would
  collapse into one number and hide the comparison they exist to make.
- **Strategy evolution** — each round's menu and ban list, diffed against the previous
  round: green added by the last round's feedback, red dropped, blue an off-menu strategy the
  clustering discovered and promoted (a lowercase first letter marks an LLM-named novel
  cluster). This is where you see whether the feedback loop is actually moving the menu or
  churning it.
- **Verification** — `verified.json` if present: keep rate, how many criteria were rewritten,
  cost, and a table of every question with its label. **Dropped questions sort first**, since
  those are the criteria the meta-judge found unsupported — the generator's own failure modes
  rather than the answering system's. Rows show `main_correctness_problem` and, when the
  criterion imposed one, `unfair_requirements` (§1).

It reads `loop.json` (required) and whatever else exists — per-round menus, ban lists,
few-shots, per-cell `index.json`, `verified.json`. **Missing sub-pages are generated on the
fly** by shelling out to `summarize_run.py` per run dir and `feedback_viewer.py` per round;
both are idempotent and API-free, so this costs nothing. `--no-subpages` skips that,
`--regenerate-subpages` rebuilds them all. A sub-page that fails is reported as a warning and
skipped rather than aborting the report, so a broken cell shows up as a missing link.

**Both round layouts are handled.** `cell_dirs` resolves a round's run dirs as
`round_KK/<prompt>/` or, when `--systems-file` is in play, `round_KK/<system>/<prompt>/`, and
the sub-page generation and the sample index each try both depths. All three of those sites
assumed the flat layout when multi-system support landed, and the failure was silent in the
worst way: the rounds table read 0 seeds and 0 `FAILED_FOUND` for every round, no per-cell
`report.html` or `answers/` was generated at all, and the verification rows had nothing to
link back to — an empty report at the end of a multi-hour run.

---

## Typical invocation

```bash
python research_pipeline_parallel.py \
    --out-dir runs/sqa_50_100_explore \
    --start 50 --limit 50 --concurrency 10 --prompt exploit \
  && python summarize_run.py runs/sqa_50_100_explore
```

```bash
python research_pipeline_parallel.py --out-dir runs/sqa_50_100_verify_vc --limit 20 --concurrency 10  --prompt exploit --verify-criterion --verify-n-papers 50 --verify-max-chars-per-paper 8000 --reranker-url http://spark-9076:8017
```

Against WebThinker (§8) — note the timeout, which must clear a ~19-minute answer, and the low
concurrency, since nothing bounds it server-side:

```bash
python research_loop.py --out-dir runs/wt_smoketest --total 10 --no-feedback \
    --concurrency 2 --max-attempts 3 --model gpt-5.6-terra \
    --profile webthinker --skip-seed-round --timeout 5400 \
    --verify-criterion --verify-n-papers 25 --verify-max-chars-per-paper 8000 \
    --reranker-url http://<reranker-host>:8017 \
    --server-url http://<webthinker-host>:8007/ask
```

`--no-feedback` is one round with no clustering and no menu rewrites — the loop's equivalent
of running the driver, but it keeps the round layout and the `--verify-after` option.

All three answering systems in one loop, a third of each round to each (§4):

```bash
python research_loop.py --out-dir runs/loop3 --total 90 --feedback-every 30 \
    --systems-file systems.json --model gpt-5.6-terra --skip-seed-round \
    --verify-criterion --verify-n-papers 25 --reranker-url http://<reranker-host>:8017
```

Environment: `pip install -r claude_inference/requirements.txt` (Python 3.10+, tested on
3.12.7). Three packages carry the whole loop — `anthropic`, `openai`, `requests` — plus
`datasets` only when seeds come from the HF dataset rather than `--seeds-file`; the repo-root
`requirements.txt` is the unrelated training stack.

Requires `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` for a `gpt-*` model — see §0);
`S2_API_KEY` additionally when `--verify-criterion` is on
(without it, S2 retrieval is heavily rate limited) and a research server listening on
`--server-url`. For reranking, pass `--reranker-url` (bare `host:port` — the `/v1/rerank`
path is appended automatically) or export `VLLM_RERANK_URL`. Both URLs are probed at
startup and the run refuses to begin if either is down (§4).

Note `--out-dir` must be fresh: seeds with an existing `sample_NNN.json` are skipped, so
re-running a completed run into the same directory does nothing.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| `FAILED_FOUND` | A harder question broke the system — the objective. |
| `EXHAUSTED` | All attempts used; the system passed every rewrite. |
| `ALREADY_HARD` | The unmodified seed already failed in round 0. |
| `CRITERION_INVALID` | **Retired.** The meta-judge rejecting a criterion used to end the seed here; it now retries with a regenerated question and criterion (§1), so this status is never emitted. Runs made before the change are not comparable on status counts: those seeds now land in `EXHAUSTED` or `FAILED_FOUND`. |
| `ERROR` | Pipeline exception (generation, retrieval, research server, or judging). |
| `BAD_OUTPUT` / `SUBPROCESS_FAILED` | Driver-level: the result file was unparseable (including truncated by a kill mid-write), or the subprocess died. |
| `SKIPPED` | **Retired.** A resumed seed once reported this instead of its real status, zeroing the run's harvest and cost in every report; the status is now read back off the result file (§2). Old `index.json` files still carry it. |
