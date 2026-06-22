# GPU-Accelerated Jigsaw Puzzle Piece Detection

Serial C++ and CUDA implementations of an image-processing pipeline for
detecting and classifying canonical jigsaw puzzle pieces.

The project was developed for GPU Architecture & Computing, SS2026. It focuses
on the full pipeline, not only isolated kernels: preprocessing, segmentation,
contour extraction, signal analysis, classification, visualization, timing, and
profiling.

| Input | Classified output |
|:-----:|:-----------------:|
| ![input](docs/single_input.jpg) | ![output](docs/single_output.png) |

## What The Pipeline Does

Given an image with non-overlapping puzzle pieces, the pipeline:

1. loads the RGB image,
2. converts it to grayscale and normalizes it,
3. blurs and thresholds it into a binary mask,
4. cleans the mask with morphological opening,
5. labels connected components and filters small regions,
6. extracts each piece boundary and ordered contour,
7. converts contours into radial signals,
8. detects corners and classifies the four edges,
9. draws contours, bounding boxes, and class labels into an overlay image.

Each piece receives a rotation-normalized edge class using:

- `L`: straight edge,
- `V`: knob,
- `C`: hole.

The serial implementation is the correctness and timing baseline. The CUDA
implementation keeps the same logical stages but changes the data flow: image
buffers stay on the device where possible, pieces are batched into flat buffers,
and small sequential stages remain on the CPU when a GPU launch would not pay
off.

## Repository Structure

```text
.
|-- include/
|   |-- serial/              # serial stage interfaces
|   |-- parallel/            # CUDA stage interfaces
|   |-- types.hpp            # shared image, region, contour, timing types
|   |-- stb_image*.h         # image load/write helpers
|   `-- stb_truetype.h       # text rendering for overlays
|-- src/
|   |-- serial/              # serial C++ pipeline and stages
|   |-- parallel/            # CUDA pipeline and stages
|   |-- serial/tests/        # small standalone serial tests
|   |-- stb_image_impl.cpp   # STB implementation unit
|   `-- analyze_benchmarks.py
|-- data/
|   |-- single.JPG           # default smoke-test image
|   |-- pipeline_output/     # default output directory
|   |-- benchmark_counted_scaled/
|   `-- benchmark_results/   # benchmark CSVs and generated plots
|-- docs/                    # README example images
|-- reference_py/            # original Python reference helpers
|-- build_serial.sh
|-- build_cuda.sh
|-- benchmark_serial.sh
|-- benchmark_cuda.sh
`-- profile_cuda.sh
```

## Requirements

For the serial pipeline:

- C++17 compiler, tested with `g++`
- OpenMP-capable standard toolchain is useful but not required for the serial
  binary

For the CUDA pipeline:

- NVIDIA GPU
- CUDA toolkit with `nvcc`
- OpenMP runtime for the CPU-side parallel parts of the CUDA pipeline

For benchmark plotting:

- Python 3
- `numpy`
- `pandas`
- `matplotlib`

For profiling:

- NVIDIA Nsight Systems (`nsys`)

## Build

The project intentionally uses small shell build scripts instead of CMake.
Both binaries are written to `build/` by default.

### Serial

```bash
./build_serial.sh
```

Output:

```text
build/pipeline
```

### CUDA

```bash
./build_cuda.sh
```

Output:

```text
build/pipeline_cuda
```

### Useful Build Variables

All build variables can be overridden through the environment.

| Variable | Serial default | CUDA default | Meaning |
|:---------|:---------------|:-------------|:--------|
| `CXX` | `g++` | - | C++ compiler for the serial build |
| `NVCC` | - | `nvcc` | CUDA compiler |
| `OPT` | `-O3` | `-O3` | optimization flag |
| `DEBUG` | `0` | `0` | enable pipeline summary/log output when set to `1` |
| `SUB_TIMINGS` | `1` | `1` | enable sub-stage timing measurement |
| `PERSIST_TIMINGS` | `0` | `0` | append CSV timing rows when set to `1` |
| `NVTX` | - | `0` | compile NVTX ranges for Nsight Systems when set to `1` |
| `ARCH` | - | `sm_75` | CUDA target architecture |
| `OUT` | `build/pipeline` | `build/pipeline_cuda` | output binary path |

Examples:

```bash
DEBUG=1 SUB_TIMINGS=1 ./build_serial.sh
ARCH=sm_80 SUB_TIMINGS=1 NVTX=1 ./build_cuda.sh
OUT=build/pipeline_cuda_a100 ARCH=sm_80 ./build_cuda.sh
```

Common CUDA architecture values:

| GPU | `ARCH` |
|:----|:-------|
| V100 | `sm_70` |
| T4 | `sm_75` |
| A100 | `sm_80` |
| H100 | `sm_90` |

## Run

Both standalone binaries accept the same arguments:

```bash
./build/pipeline [input_image_or_folder] [output_dir]
./build/pipeline_cuda [input_image_or_folder] [output_dir]
```

If no arguments are given, both use:

```text
input:  data/single.JPG
output: data/pipeline_output
```

### Quick Smoke Test

```bash
./build_serial.sh
./build/pipeline

./build_cuda.sh
./build/pipeline_cuda
```

### Run On One Image

```bash
./build/pipeline data/single.JPG data/pipeline_output
./build/pipeline_cuda data/single.JPG data/pipeline_output
```

Expected overlay outputs:

```text
data/pipeline_output/single_overlays.bmp
data/pipeline_output/single_cuda_overlays.bmp
```

### Run On A Folder

```bash
./build/pipeline data/benchmark_counted_scaled/5100x7016 data/pipeline_output
./build/pipeline_cuda data/benchmark_counted_scaled/5100x7016 data/pipeline_output
```

Folder mode processes all supported images in sorted order. The CUDA folder run
benefits from staying in one process with a warmed CUDA runtime, but the current
pipeline still recreates the main device buffers for each image.

## Output Files

For input `name.jpg`, the pipelines write:

| Pipeline | Overlay filename |
|:---------|:-----------------|
| serial | `name_overlays.bmp` |
| CUDA | `name_cuda_overlays.bmp` |

Image loading and final file writing are intentionally outside the measured
pipeline timer. The measured time includes preprocessing, segmentation,
contour/signal processing, classification, and in-memory overlay rendering.

If timing persistence is enabled, CSV files are written either to the output
directory or to `PIPELINE_TIMINGS_CSV`:

```bash
SUB_TIMINGS=1 PERSIST_TIMINGS=1 ./build_serial.sh
./build/pipeline data/single.JPG data/pipeline_output
# default CSV: data/pipeline_output/serial_timings.csv

SUB_TIMINGS=1 PERSIST_TIMINGS=1 ./build_cuda.sh
./build/pipeline_cuda data/single.JPG data/pipeline_output
# default CSV: data/pipeline_output/cuda_timings.csv
```

Benchmark scripts set `PIPELINE_TIMINGS_CSV` explicitly so that all runs are
collected in `data/benchmark_results/`.

## Benchmarking

The benchmark scripts rebuild the selected pipeline with persistent timing
output, run the scaled benchmark images, and write CSV files.

### Serial Benchmark

```bash
./benchmark_serial.sh
```

Outputs:

```text
data/benchmark_results/serial_single.csv
data/benchmark_results/serial_folder.csv
```

### CUDA Benchmark

```bash
./benchmark_cuda.sh
```

Outputs:

```text
data/benchmark_results/cuda_single.csv
data/benchmark_results/cuda_folder.csv
```

The scripts currently use:

- resolutions `1024x1409`, `2048x2817`, `4096x5635`, `5100x7016`,
- `10` runs per image,
- folder-mode benchmark at `5100x7016`.

### Generate Benchmark Plots

After both serial and CUDA CSVs exist:

```bash
python3 src/analyze_benchmarks.py
```

This writes report and presentation plot PDFs to `data/benchmark_results/`,
including:

- stage runtime plots by resolution and piece count,
- single-image vs folder-mode comparisons,
- serial/CUDA speedup plots,
- transfer-overhead plots.

## CUDA Profiling

`profile_cuda.sh` builds the CUDA binary with NVTX enabled, runs one profiled
execution with Nsight Systems, and prints common CUDA/NVTX summaries.

```bash
./profile_cuda.sh
```

Default input/output/report:

```text
input:  data/single.JPG
output: test_output
report: build/pipeline_report_<timestamp>.nsys-rep
```

Custom run:

```bash
ARCH=sm_75 ./profile_cuda.sh \
  data/single.JPG \
  data/pipeline_output \
  build/pipeline_report_single
```

The script runs:

- `nsys profile` with CUDA and NVTX tracing,
- `nsys stats` with `nvtxsum`, `cudaapisum`, `gpukernsum`,
  `gpumemtimesum`, and `gpumemsizesum`.

## Tuned Segmentation Parameters

The current serial and CUDA pipelines use matching preprocessing and filtering
parameters:

- Gaussian blur: `5 x 5`, `sigma = 1.4`
- Otsu threshold offset: `-15`
- morphological cleaning: one `5 x 5` opening
- minimum connected-component area: `45000` pixels at image height `5100`,
  scaled quadratically with image height

These values were tuned to reduce small boundary gaps and printed-mark failures
without switching to morphological closing, which caused worse topology errors
such as merged pieces and contour shortcuts.

## Serial Test Programs

Small standalone tests live in `src/serial/tests/`. They are not wired into the
build scripts, but can be compiled manually when needed. Example:

```bash
g++ -std=c++17 -O3 -Iinclude \
  src/serial/tests/test_contour_to_signal.cpp \
  src/serial/contour_to_signal.cpp \
  -o build/test_contour_to_signal

./build/test_contour_to_signal
```

## Notes On Generated Files

The following are generated artifacts and should usually stay out of version
control:

- `build/`
- `data/pipeline_output/`
- `data/benchmark_results/*.csv`
- generated benchmark plot PDFs, unless they are intentionally submitted
- Nsight reports such as `*.nsys-rep` and `*.sqlite`
- LaTeX report/presentation build artifacts and exported PDFs

## Group Members

| Name | GitHub username |
|:-----|:----------------|
| Jakob Olsacher | `jacky6134` |
| Ananthalakshmi Vinod Iyer | `ananthaiyer` |
| Philipp Sepin | `p0017` |
| Jan-Hill Arganda | `BeginnerCoderEurope` |
| Tobias Ponesch | `AlbertZweistein-2` |
