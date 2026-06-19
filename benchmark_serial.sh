#!/bin/bash

PIPELINE=./build/pipeline
OUT_DIR=data/benchmark_results
CSV=$OUT_DIR/serial.csv
RUNS=5

IMAGES=("31_19_p2.jpg" "45_1_p2.jpg" "69_4_p1.jpg")
PIECES=(31 45 69)
RESOLUTIONS=("1024x1409" "2048x2817" "4096x5635" "5100x7016")

mkdir -p "$OUT_DIR"
mkdir -p data/pipeline_output

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
            out=$(TIMINGS=1 DEBUG=0 "$PIPELINE" "$imgpath" data/pipeline_output 2>&1)

            get() { echo "$out" | grep "$1" | awk '{print $(NF-1)}'; }

            total=$(get "Total wall time")
            pre=$(get "Preprocessing")
            cc=$(get "Connected components")
            boundary=$(get "Boundary extraction")
            contour=$(get "Contour extraction")
            csmooth=$(get "Contour smoothing")
            circle=$(get "Enclosing circle")
            radial=$(get "Radial signal")
            ssmooth=$(get "Signal smoothing")
            peaks=$(get "Peak detection")
            edges=$(get "Edge classification")
            viz=$(get "Visualization")

            echo "$res,$img,$pieces,$run,$total,$pre,$cc,$boundary,$contour,$csmooth,$circle,$radial,$ssmooth,$peaks,$edges,$viz" >> "$CSV"
        done
    done
done

echo "Done: $CSV"