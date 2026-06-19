import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE, "data/benchmark_results/serial.csv")

STAGES = [
    "preprocessing_ms",
    "connected_components_ms",
    "boundary_extraction_ms",
    "contour_extraction_ms",
    "contour_smoothing_ms",
    "enclosing_circle_ms",
    "radial_signal_ms",
    "signal_smoothing_ms",
    "peak_detection_ms",
    "edge_classification_ms",
    "visualization_ms",
]

df = pd.read_csv(CSV_PATH)


def parse_pixels(res):
    w, h = res.split("x")
    return int(w) * int(h)


df["pixels"] = df["resolution"].apply(parse_pixels)

grouped = df.groupby(["pixels", "pieces"])

agg = grouped[["total_ms"] + STAGES].agg(["mean", "std"]).reset_index()
agg.columns = ["_".join(c).strip("_") for c in agg.columns]


def errorbars(ax, x, col, label=None, color=None):
    means = agg.groupby(x)[f"{col}_mean"].mean()
    stds = agg.groupby(x)[f"{col}_std"].mean()
    ax.errorbar(
        means.index,
        means.values,
        yerr=stds.values,
        fmt="o-",
        capsize=4,
        label=label,
        color=color,
    )


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Serial Pipeline Benchmarks")

ax = axes[0, 0]
errorbars(ax, "pixels", "total_ms")
ax.set_xlabel("Image size (pixels)")
ax.set_ylabel("Runtime (ms)")
ax.set_title("Total runtime vs image size")

ax = axes[0, 1]
for i, stage in enumerate(STAGES):
    errorbars(
        ax,
        "pixels",
        stage,
        label=stage.replace("_ms", ""),
        color=colors[i % len(colors)],
    )
ax.set_xlabel("Image size (pixels)")
ax.set_ylabel("Runtime (ms)")
ax.set_title("Stage runtimes vs image size")
ax.legend(fontsize=7)

ax = axes[1, 0]
errorbars(ax, "pieces", "total_ms")
ax.set_xlabel("Number of pieces")
ax.set_ylabel("Runtime (ms)")
ax.set_title("Total runtime vs number of pieces")

ax = axes[1, 1]
for i, stage in enumerate(STAGES):
    errorbars(
        ax,
        "pieces",
        stage,
        label=stage.replace("_ms", ""),
        color=colors[i % len(colors)],
    )
ax.set_xlabel("Number of pieces")
ax.set_ylabel("Runtime (ms)")
ax.set_title("Stage runtimes vs number of pieces")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(BASE, "data/benchmark_results/serial_plots.png"), dpi=150)
