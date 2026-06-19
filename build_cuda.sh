#!/usr/bin/env bash
# Build the CUDA parallel pipeline without CMake.
#
# Override any variable via the environment, e.g.:
#   ARCH=sm_70 ./build_cuda.sh
#   DEBUG=0 OPT=-O2 OUT=bin/pipeline_cuda ./build_cuda.sh
#
# Variables:
#   NVCC    – CUDA compiler            (default: nvcc)
#   OPT     – optimisation flag        (default: -O3)
#   ARCH    – GPU architecture         (default: sm_75)
#               sm_70  → V100
#               sm_75  → T4
#               sm_80  → A100
#               sm_90  → H100
#   DEBUG   – 1 = pipeline log output  (default: 0)
#   TIMINGS – 1 = sub-stage timing measurement (default: 1)
#   PERSIST_TIMINGS – 1 = append timings to CSV (default: 0)
#   OUT     – output binary path       (default: build/pipeline_cuda)

set -euo pipefail

NVCC=${NVCC:-nvcc}
OPT=${OPT:--O3}
ARCH=${ARCH:-sm_75}
DEBUG=${DEBUG:-0}
TIMINGS=${TIMINGS:-1}
PERSIST_TIMINGS=${PERSIST_TIMINGS:-0}
OUT=${OUT:-build/pipeline_cuda}

# ── Compile definitions ───────────────────────────────────────────────────────
DEFS="-DCUDA_PIPELINE_BUILD_STANDALONE"
[ "${DEBUG}"   = "1" ] && DEFS="${DEFS} -DDEBUG_LEVEL=1"
[ "${TIMINGS}" = "1" ] && DEFS="${DEFS} -DSUB_TIMINGS=1"
[ "${PERSIST_TIMINGS}" = "1" ] && DEFS="${DEFS} -DPERSIST_TIMINGS=1"

# ── Source files ──────────────────────────────────────────────────────────────
# Only stb_image_impl.cpp is needed from serial: it provides STB_IMAGE_IMPLEMENTATION
# (stbi_load is called directly in pipeline.cu). Everything else is in the parallel sources.
SRCS=(
    src/stb_image_impl.cpp
    src/parallel/boundary_extraction.cu
    src/parallel/cleaning.cu
    src/parallel/component_labeling.cu
    src/parallel/contour_to_signal.cu
    src/parallel/preprocessing.cu
    src/parallel/signal_analysis.cu
    src/parallel/visualization.cpp
    src/parallel/pipeline.cu
)

# ── Build ─────────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "${OUT}")"

echo "Building CUDA pipeline"
echo "  NVCC=${NVCC}  ARCH=${ARCH}  OPT=${OPT}  DEBUG=${DEBUG}  TIMINGS=${TIMINGS}  PERSIST_TIMINGS=${PERSIST_TIMINGS}"
echo "  Output: ${OUT}"

${NVCC} \
    -std=c++17 \
    "${OPT}" \
    --use_fast_math \
    -arch="${ARCH}" \
    -Iinclude \
    ${DEFS} \
    -Xcompiler -fopenmp \
    -Xcompiler -march=native \
    "${SRCS[@]}" \
    -o "${OUT}" \
    -lgomp

echo "Done: ${OUT}"
