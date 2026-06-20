#!/usr/bin/env bash
# Build the CUDA pipeline, profile one run with Nsight Systems, and print stats.
#
# Usage:
#   ./profile_cuda.sh [input_image] [output_dir] [report_base]
#
# Examples:
#   ./profile_cuda.sh
#   ARCH=sm_75 ./profile_cuda.sh data/1_p1.jpg data/pipeline_output build/pipeline_report

set -euo pipefail

INPUT=${1:-${INPUT:-data/single.JPG}}
OUTPUT_DIR=${2:-${OUTPUT_DIR:-test_output}}
REPORT=${3:-${REPORT:-build/pipeline_report_$(date +%Y%m%d_%H%M%S)}}

ARCH=${ARCH:-sm_75}
PIPELINE_BIN=${PIPELINE_BIN:-build/pipeline_cuda}
NSYS=${NSYS:-nsys}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
NSYS_SAMPLE=${NSYS_SAMPLE:-none}
NSYS_CPUCTXSW=${NSYS_CPUCTXSW:-none}
NSYS_SHOW_OUTPUT=${NSYS_SHOW_OUTPUT:-true}

TIMINGS=${TIMINGS:-1}
PERSIST_TIMINGS=${PERSIST_TIMINGS:-0}
DEBUG=${DEBUG:-0}
if [ "${DEBUG}" != "1" ]; then
    DEBUG=0
fi

REPORT_BASE=${REPORT%.nsys-rep}
REPORT_FILE="${REPORT_BASE}.nsys-rep"

if ! command -v "${NSYS}" >/dev/null 2>&1; then
    echo "Error: '${NSYS}' not found. Install Nsight Systems or set NSYS=/path/to/nsys." >&2
    exit 1
fi

mkdir -p "$(dirname "${PIPELINE_BIN}")" "$(dirname "${REPORT_BASE}")" "${OUTPUT_DIR}"

echo "== Build CUDA pipeline =="
ARCH="${ARCH}" \
TIMINGS="${TIMINGS}" \
PERSIST_TIMINGS="${PERSIST_TIMINGS}" \
DEBUG="${DEBUG}" \
NVTX=1 \
OUT="${PIPELINE_BIN}" \
./build_cuda.sh

echo
echo "== Profile run =="
echo "Input:      ${INPUT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Report:     ${REPORT_FILE}"

"${NSYS}" profile \
    --trace="${NSYS_TRACE}" \
    --sample="${NSYS_SAMPLE}" \
    --cpuctxsw="${NSYS_CPUCTXSW}" \
    --show-output="${NSYS_SHOW_OUTPUT}" \
    --force-overwrite=true \
    -o "${REPORT_BASE}" \
    "${PIPELINE_BIN}" "${INPUT}" "${OUTPUT_DIR}"

echo
echo "== Nsight Systems stats =="
"${NSYS}" stats \
    --force-export=true \
    --force-overwrite=true \
    --report nvtxsum,cudaapisum,gpukernsum,gpumemtimesum,gpumemsizesum \
    "${REPORT_FILE}"
