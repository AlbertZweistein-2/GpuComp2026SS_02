#!/bin/bash

set -euo pipefail

PIPELINE=./build/pipeline
OUT_DIR=data/benchmark_results
CSV=$OUT_DIR/serial.csv
RUNS=5

IMAGES=("31_19_p2.jpg" "45_1_p2.jpg" "65_4_p1.jpg")
PIECES=(31 45 65)
RESOLUTIONS=("1024x1409" "2048x2817" "4096x5635" "5100x7016")

mkdir -p "$OUT_DIR"
mkdir -p data/pipeline_output

PERSIST_TIMINGS=1 TIMINGS=0 DEBUG=0 ./build_serial.sh

echo "resolution,image,pieces,run,total_ms,preprocessing_ms,connected_components_ms,boundary_extraction_ms,contour_extraction_ms,contour_smoothing_ms,enclosing_circle_ms,radial_signal_ms,signal_smoothing_ms,peak_detection_ms,edge_classification_ms,visualization_ms" > "$CSV"

for res in "${RESOLUTIONS[@]}"; do
    for i in "${!IMAGES[@]}"; do
        img="${IMAGES[$i]}"
        pieces="${PIECES[$i]}"
        imgpath="data/benchmark_counted_scaled/$res/$img"

        if [ ! -f "$imgpath" ]; then
            echo "SKIP: $imgpath not found"
            continue
        fi

        echo "Benchmarking $res / $img ..."

        for run in $(seq 1 $RUNS); do
            PIPELINE_TIMINGS_CSV="$CSV" \
            PIPELINE_TIMING_PIECES="$pieces" \
            PIPELINE_TIMING_RUN="$run" \
            "$PIPELINE" "$imgpath" data/pipeline_output
        done
    done
done

echo "Done: $CSV"
