#!/bin/bash
# One slice of the 1K sample. Same pattern as run_bundled.sh — spin up DR-Tulu,
# run the pipeline on the slice, tear down. Slice params come from CLI args.
#
# Usage: run_1k_slice.sh <START> <LIMIT>

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <SLICE_START> <SLICE_LIMIT>" >&2
    exit 1
fi

SLICE_START=$1
SLICE_LIMIT=$2

PROJECT=/weka/nora-default/jayd/query_generation/gen_dr_queries
OUT_DIR=$PROJECT/claude_inference/runs/bias22_1k_slice_${SLICE_START}_${SLICE_LIMIT}

pip install --quiet anthropic requests datasets openai

# auto_launch.py's downstream imports (e.g. cite_utils) resolve relative to the
# workflows dir, so it MUST be launched with that as cwd — running it via an
# absolute path from elsewhere breaks the DR-Tulu stack silently.
setsid bash -c 'cd /weka/nora-default/varshak/dr-tulu/agent/workflows && python auto_launch.py' \
    > /tmp/dr-tulu.log 2>&1 &
SERVER_PGID=$!

cleanup() {
    echo "[cleanup] tearing down DR-Tulu server (pgid=$SERVER_PGID)"
    kill -TERM -"$SERVER_PGID" 2>/dev/null || true
    sleep 5
    kill -KILL -"$SERVER_PGID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[wait] waiting for DR-Tulu /ask on :8007 ..."
for i in $(seq 1 180); do
    if curl -sf http://localhost:8007/docs > /dev/null 2>&1; then
        echo "[wait] DR-Tulu server ready after $((i * 10))s"
        break
    fi
    sleep 10
done
if ! curl -sf http://localhost:8007/docs > /dev/null 2>&1; then
    echo "[fatal] DR-Tulu server did not come up. Server log tail:"
    tail -80 /tmp/dr-tulu.log || true
    exit 1
fi

# /docs coming up only proves FastAPI booted — the vLLM + MCP + workflow stack
# behind /ask can still be broken (e.g. missing cite_utils). Exercise it once
# on a trivial query and abort if the answer is empty or malformed.
echo "[wait] running DR-Tulu answer healthcheck (1-2 min)..."
HEALTH=$(curl -s -X POST http://localhost:8007/ask \
    -H 'Content-Type: application/json' \
    -d '{"question": "What is deep learning?"}' \
    --max-time 600)
if ! echo "$HEALTH" | python -c "import sys, json; d = json.load(sys.stdin); sys.exit(0 if len(d.get('answer', '') or '') > 100 else 1)" 2>/dev/null; then
    echo "[fatal] DR-Tulu answer healthcheck failed. Response head:"
    echo "$HEALTH" | head -c 2000; echo
    echo "--- /tmp/dr-tulu.log tail ---"
    tail -100 /tmp/dr-tulu.log 2>&1 || true
    exit 1
fi
echo "[wait] DR-Tulu answer healthcheck OK"

cd "$PROJECT"

echo "=== 1K slice: --start $SLICE_START --limit $SLICE_LIMIT -> $OUT_DIR ==="
python claude_inference/research_pipeline_parallel.py \
    --out-dir "$OUT_DIR" \
    --start "$SLICE_START" --limit "$SLICE_LIMIT" \
    --concurrency 5 --max-attempts 5 \
    --prompt original --model claude-sonnet-4-5

python claude_inference/summarize_run.py "$OUT_DIR" || true
echo "=== slice complete ==="
