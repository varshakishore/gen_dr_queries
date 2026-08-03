#!/bin/bash
# Submit 3 parallel Beaker experiments that together process 1000 unique SQA
# queries via the full harden-and-judge pipeline. Each slice runs its own
# DR-Tulu server on 1 GPU on ai2/saturn under nsf-uchicago-apto (normal, 8h).
#
# Slices start at 100 to avoid overlap with the still-running 100-seed
# experiment on seeds 0..99.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$HERE/launch_1k.yaml.template"

# start, limit — each slice is ~7h at concurrency 5, fits in the 8h timeout.
SLICES=(
    "100 334"    # seeds 100..433
    "434 333"    # seeds 434..766
    "767 333"    # seeds 767..1099
)

for pair in "${SLICES[@]}"; do
    read -r START LIMIT <<< "$pair"
    END=$((START + LIMIT - 1))
    TMP=$(mktemp --suffix=.yaml)
    sed -e "s|{SLICE_START}|$START|g" \
        -e "s|{SLICE_LIMIT}|$LIMIT|g" \
        -e "s|{SLICE_END}|$END|g" \
        "$TEMPLATE" > "$TMP"
    echo "submitting slice: start=$START limit=$LIMIT (seeds $START..$END) ..."
    beaker experiment create "$TMP" --workspace ai2/jayd-default 2>&1 \
        | grep -Ei "submitted|error" || true
    rm -f "$TMP"
done
