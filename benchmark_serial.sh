#!/bin/bash

set -euo pipefail

PIPELINE=./build/pipeline
OUT_DIR=data/benchmark_results
SINGLE_CSV=$OUT_DIR/serial_single.csv
FOLDER_CSV=$OUT_DIR/serial_folder.csv
RUNS=5
BASELINE=5100x7016

RESOLUTIONS=("1024x1409" "2048x2817" "4096x5635" "5100x7016")

mkdir -p "$OUT_DIR"

PERSIST_TIMINGS=1 TIMINGS=0 DEBUG=0 ./build_serial.sh

rm -f "$SINGLE_CSV"

for res in "${RESOLUTIONS[@]}"; do
    for imgpath in data/benchmark_counted_scaled/$res/*.jpg; do
        img=$(basename "$imgpath")
        pieces="${img%%_*}"

        echo "Benchmarking $res / $img ..."

        for run in $(seq 1 $RUNS); do
            PIPELINE_TIMINGS_CSV="$SINGLE_CSV" \
            PIPELINE_TIMING_PIECES="$pieces" \
            PIPELINE_TIMING_RUN="$run" \
            "$PIPELINE" "$imgpath" data/pipeline_output
        done
    done
done

echo "Done: $SINGLE_CSV"

rm -f "$FOLDER_CSV"

echo "Benchmarking folder mode: $BASELINE ..."

for run in $(seq 1 $RUNS); do
    PIPELINE_TIMINGS_CSV="$FOLDER_CSV" \
    PIPELINE_TIMING_RUN="$run" \
    "$PIPELINE" "data/benchmark_counted_scaled/$BASELINE" data/pipeline_output
done

echo "Done: $FOLDER_CSV"