#!/usr/bin/env bash
# Build the serial pipeline without CMake.
#
# Override any variable via the environment, e.g.:
#   DEBUG=0 ./build_serial.sh
#   OPT=-O2 OUT=bin/pipeline ./build_serial.sh
#
# Variables:
#   CXX     – C++ compiler            (default: g++)
#   OPT     – optimisation flag       (default: -O3)
#   DEBUG   – 1 = pipeline log output (default: 1)
#   TIMINGS – 1 = sub-stage timings   (default: 1, requires DEBUG=1)
#   OUT     – output binary path      (default: build/pipeline)

set -euo pipefail

CXX=${CXX:-g++}
OPT=${OPT:--O3}
DEBUG=${DEBUG:-1}
TIMINGS=${TIMINGS:-1}
OUT=${OUT:-build/pipeline}

# ── Compile definitions ───────────────────────────────────────────────────────
DEFS="-DPIPELINE_BUILD_STANDALONE"
[ "${DEBUG}"   = "1" ]                            && DEFS="${DEFS} -DDEBUG_LEVEL=1"
[ "${DEBUG}"   = "1" ] && [ "${TIMINGS}" = "1" ] && DEFS="${DEFS} -DSUB_TIMINGS=1"

# ── Source files ──────────────────────────────────────────────────────────────
SRCS=(
    src/stb_image_impl.cpp
    src/serial/boundary_extraction.cpp
    src/serial/component_labeling.cpp
    src/serial/contour_to_signal.cpp
    src/serial/signal_analysis.cpp
    src/serial/visualization.cpp
    src/serial/preprocessing.cpp
    src/serial/cleaning.cpp
    src/serial/pipeline.cpp
)

# ── Build ─────────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "${OUT}")"

echo "Building serial pipeline"
echo "  CXX=${CXX}  OPT=${OPT}  DEBUG=${DEBUG}  TIMINGS=${TIMINGS}"
echo "  Output: ${OUT}"

${CXX} -std=c++17 "${OPT}" -Iinclude ${DEFS} "${SRCS[@]}" -o "${OUT}"

echo "Done: ${OUT}"
