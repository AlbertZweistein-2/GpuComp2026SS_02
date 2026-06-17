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
#   ARCH    – GPU architecture         (default: sm_80)
#               sm_70  → V100
#               sm_80  → A100
#               sm_90  → H100
#   DEBUG   – 1 = pipeline log output  (default: 1)
#   TIMINGS – 1 = sub-stage timings    (default: 1, requires DEBUG=1)
#   OUT     – output binary path       (default: build/pipeline_cuda)

set -euo pipefail

NVCC=${NVCC:-nvcc}
OPT=${OPT:--O3}
ARCH=${ARCH:-sm_80}
DEBUG=${DEBUG:-1}
TIMINGS=${TIMINGS:-1}
OUT=${OUT:-build/pipeline_cuda}

# ── Compile definitions ───────────────────────────────────────────────────────
DEFS="-DCUDA_PIPELINE_BUILD_STANDALONE"
[ "${DEBUG}"   = "1" ]                            && DEFS="${DEFS} -DDEBUG_LEVEL=1"
[ "${DEBUG}"   = "1" ] && [ "${TIMINGS}" = "1" ] && DEFS="${DEFS} -DSUB_TIMINGS=1"

# ── Source files ──────────────────────────────────────────────────────────────
# Only stb_image_impl.cpp is needed from serial: it provides STB_IMAGE_IMPLEMENTATION
# (stbi_load is called directly in pipeline.cu). Everything else is in the parallel sources.
SERIAL_SRCS=(
    src/serial/stb_image_impl.cpp
)

# CUDA parallel modules + entry point
CUDA_SRCS=(
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
echo "  NVCC=${NVCC}  ARCH=${ARCH}  OPT=${OPT}  DEBUG=${DEBUG}  TIMINGS=${TIMINGS}"
echo "  Output: ${OUT}"

${NVCC} \
    -std=c++17 \
    "${OPT}" \
    -arch="${ARCH}" \
    -Iinclude \
    ${DEFS} \
    -Xcompiler -fopenmp \
    "${SERIAL_SRCS[@]}" \
    "${CUDA_SRCS[@]}" \
    -o "${OUT}" \
    -lgomp

echo "Done: ${OUT}"
