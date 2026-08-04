#!/bin/bash
# Bundled beaker task: launches the DR-Tulu server stack (vLLM + MCP + FastAPI /ask
# on :8007), waits for it, then runs the harden-and-judge pipeline against it.
#
# Two phases:
#   A. Option A — 100 seeds under --prompt original with the 22-item strategy menu
#      (natural strategy pickup, including the 13 new cognitive-bias-derived ones
#      added at positions 9-21).
#   B. Option B — 15 seeds per forced strategy for #9..#21, --max-attempts 2
#      (per-strategy break-rate signal).
#
# Output dirs are stable across re-runs; the parallel wrapper's --skip-existing
# default makes preemption + resume painless.
#
# Assumes: /weka/nora-default is mounted, ANTHROPIC_API_KEY and S2_API_KEY are set
# via beaker envVars, and the varshak/drtulu image is in use.

set -euo pipefail

PROJECT=/weka/nora-default/jayd/query_generation/gen_dr_queries
RUNS=$PROJECT/claude_inference/runs

# Install pipeline dependencies into the container's Python. Fast, and avoids
# the (non-portable) local .venv symlinks.
pip install --quiet anthropic requests datasets openai

# Start the DR-Tulu server stack in the background. `setsid` gives it its own
# process group so we can kill the whole tree on exit without racing with beaker.
# Must be launched with cwd = workflows/ so downstream imports (cite_utils etc.)
# resolve — running it via an absolute path from elsewhere breaks silently.
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

# Wait up to 30 min for :8007 to come up (vLLM cold-load can take a while).
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

# --------------------------------------------------------------------------
# Option A — 100 seeds, --prompt original, full 22-item menu.
# --------------------------------------------------------------------------
OUT_A=$RUNS/bias22_option_a_original
echo "=== Option A: 100 seeds --prompt original -> $OUT_A ==="
python claude_inference/research_pipeline_parallel.py \
    --out-dir "$OUT_A" \
    --limit 100 --concurrency 5 --max-attempts 5 \
    --prompt original --model claude-sonnet-4-5
python claude_inference/summarize_run.py "$OUT_A" || true

# --------------------------------------------------------------------------
# Option B — 15 seeds per forced strategy for positions 9..21 (the 13 new ones).
# --max-attempts 2 because we're testing strategy expressiveness, not iterative
# hardening.
# --------------------------------------------------------------------------
for N in 9 10 11 12 13 14 15 16 17 18 19 20 21; do
    OUT_B=$RUNS/bias22_forced_${N}
    echo "=== Option B: --force-strategy $N -> $OUT_B ==="
    python claude_inference/research_pipeline_parallel.py \
        --out-dir "$OUT_B" \
        --limit 15 --concurrency 5 --max-attempts 2 \
        --prompt original --force-strategy "$N" --model claude-sonnet-4-5
    python claude_inference/summarize_run.py "$OUT_B" || true
done

echo "=== all runs complete ==="
