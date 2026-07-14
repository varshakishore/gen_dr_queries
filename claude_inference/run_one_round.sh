#!/usr/bin/env bash
# run_one_round.sh — one full round of the difficulty pipeline:
#
#   1. GENERATE   hard questions from seeds        (research_pipeline_parallel.py)
#   2. CONVERT    the run into DRChallenge input   (to_eval_tree.py)
#   3. EVALTREE   cluster + failure-rate the tree  (EvalTree/run_pipeline.drchallenge.strategy.sh)
#   4. EXTRACT    promising strategy categories    (extract_promising_categories.py)
#
# Step 3 defaults to the STRATEGY variant of the pipeline (leaf labels = the dataset's
# own "strategy" field); pass --pipeline run_pipeline.drchallenge.sh for the capability
# variant. Either way step 4 reads back the exact tree the pipeline reports it built.
#
# Step 2 overwrites Datasets/DRChallenge/{dataset.json,eval_results,splits} with THIS
# round's questions (the previous dataset.json is backed up to dataset.prev.json).
#
# Prerequisites:
#   * A deep-research agent server reachable at --server-url (default localhost:8007).
#   * ANTHROPIC_API_KEY and OpenAI_API_KEY exported (EvalTree stages 1/2/4 need them).
#
# Usage:
#   ./run_one_round.sh --seeds-file data/test-lucy.seeds.txt
#   ./run_one_round.sh --limit 20 --prompt explore --run-name round1
#   ./run_one_round.sh --skip-generation --run-name round1   # reuse an existing run dir
#   ./run_one_round.sh --dry-run ...                         # print the commands only
set -euo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ET_DIR="$CI_DIR/EvalTree"
DATASET_DIR="$ET_DIR/Datasets/DRChallenge"
PY="${PYTHON:-python}"

# --- defaults (override via flags) ---
RUN_NAME="round_$(date +%Y%m%d_%H%M%S)"
SEEDS_FILE=""
LIMIT=50
START=0
MAX_ATTEMPTS=5
GEN_MODEL="claude-sonnet-4-5"
PROMPT="explore"
CONCURRENCY=5
SERVER_URL="http://localhost:8007/ask"
PIPELINE_SCRIPT="run_pipeline.drchallenge.strategy.sh"  # strategy variant by default
ANNOTATION="strategy"             # matches $PIPELINE_SCRIPT's leaf labels (for CSV naming/fallback)
MODEL_NAME="drtulu"               # eval_results/real/<name> the pipeline grades against
MIN_QUESTIONS=1
SKIP_GENERATION=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --run-name)        RUN_NAME="$2"; shift 2 ;;
        --seeds-file)      SEEDS_FILE="$2"; shift 2 ;;
        --limit)           LIMIT="$2"; shift 2 ;;
        --start)           START="$2"; shift 2 ;;
        --max-attempts)    MAX_ATTEMPTS="$2"; shift 2 ;;
        --model)           GEN_MODEL="$2"; shift 2 ;;
        --prompt)          PROMPT="$2"; shift 2 ;;
        --concurrency)     CONCURRENCY="$2"; shift 2 ;;
        --server-url)      SERVER_URL="$2"; shift 2 ;;
        --pipeline)        PIPELINE_SCRIPT="$2"; shift 2 ;;
        --annotation)      ANNOTATION="$2"; shift 2 ;;
        --model-name)      MODEL_NAME="$2"; shift 2 ;;
        --min-questions)   MIN_QUESTIONS="$2"; shift 2 ;;
        --skip-generation) SKIP_GENERATION=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

RUN_DIR="$CI_DIR/runs/$RUN_NAME"

# run CMD, or just print it under --dry-run
run() {
    echo "+ $*"
    [ "$DRY_RUN" = "1" ] || "$@"
}

echo "================ ROUND: $RUN_NAME ================"
echo "run dir:     $RUN_DIR"
echo "dataset dir: $DATASET_DIR"
echo

# --- 1. GENERATE -------------------------------------------------------------
echo "==> [1/4] Generating hard questions"
if [ "$SKIP_GENERATION" = "1" ]; then
    echo "    (skipped; reusing $RUN_DIR)"
    [ -d "$RUN_DIR" ] || { echo "    ERROR: $RUN_DIR does not exist" >&2; exit 1; }
else
    gen_args=(--out-dir "$RUN_DIR" --limit "$LIMIT" --start "$START"
              --max-attempts "$MAX_ATTEMPTS" --model "$GEN_MODEL" --prompt "$PROMPT"
              --concurrency "$CONCURRENCY" --server-url "$SERVER_URL")
    [ -n "$SEEDS_FILE" ] && gen_args+=(--seeds-file "$SEEDS_FILE")
    run "$PY" "$CI_DIR/research_pipeline_parallel.py" "${gen_args[@]}"
fi
echo

# --- 2. CONVERT --------------------------------------------------------------
echo "==> [2/4] Converting run -> DRChallenge dataset"
if [ "$DRY_RUN" != "1" ] && [ -f "$DATASET_DIR/dataset.json" ]; then
    cp "$DATASET_DIR/dataset.json" "$DATASET_DIR/dataset.prev.json"
    echo "    backed up existing dataset.json -> dataset.prev.json"
fi
run "$PY" "$CI_DIR/to_eval_tree.py" "$RUN_DIR" --out-dir "$DATASET_DIR" --model-name "$MODEL_NAME"
echo

# --- 3. EVALTREE -------------------------------------------------------------
echo "==> [3/4] Building EvalTree (cluster + per-node failure rate)"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY for EvalTree stages 1/4}"
: "${OpenAI_API_KEY:?set OpenAI_API_KEY for EvalTree stages 1/2}"
TREE_JSON=""   # exact stage-4 tree the pipeline built; captured from its own output
if [ "$DRY_RUN" = "1" ]; then
    echo "+ (cd $ET_DIR && bash EvalTree/$PIPELINE_SCRIPT)"
else
    PIPELINE_LOG="$RUN_DIR/evaltree.log"
    ( cd "$ET_DIR" && bash "EvalTree/$PIPELINE_SCRIPT" ) | tee "$PIPELINE_LOG"
    # The pipeline prints "<Strategy|Capability> tree: <path>" for whatever clustering
    # it ran; trust that over guessing the filename, so a clustering-algorithm change is
    # picked up automatically. Path is relative to the EvalTree dir the pipeline ran in.
    TREE_REL="$(grep -oE '(Strategy|Capability) tree: .*\.json' "$PIPELINE_LOG" | tail -1 | sed -E 's/^[A-Za-z]+ tree: *//')"
    [ -n "$TREE_REL" ] && TREE_JSON="$ET_DIR/$TREE_REL"
fi
echo

# --- 4. EXTRACT --------------------------------------------------------------
echo "==> [4/4] Extracting promising strategy categories"
CSV_OUT="$DATASET_DIR/promising_categories_${ANNOTATION}.csv"
extract_args=(--dataset-dir "$DATASET_DIR" --model "$MODEL_NAME"
              --min-questions "$MIN_QUESTIONS" --out "$CSV_OUT")
if [ -n "$TREE_JSON" ]; then
    extract_args+=(--tree "$TREE_JSON")                    # exact default output
else
    extract_args+=(--annotation "$ANNOTATION" --latest)    # dry-run / fallback
fi
run "$PY" "$CI_DIR/extract_promising_categories.py" "${extract_args[@]}"
echo
echo "================ DONE: $RUN_NAME ================"
echo "promising categories CSV: $CSV_OUT"
