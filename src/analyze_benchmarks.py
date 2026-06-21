import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINGLE_CSV = os.path.join(BASE, "data/benchmark_results/serial_single.csv")
FOLDER_CSV = os.path.join(BASE, "data/benchmark_results/serial_folder.csv")
OUT_DIR = os.path.join(BASE, "data/benchmark_results")

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

STAGE_LABELS = [s.replace("_ms", "") for s in STAGES]
BASELINE_RES = "5100x7016"


def parse_pixels(res):
    w, h = res.split("x")
    return int(w) * int(h)


df_single = pd.read_csv(SINGLE_CSV)
df_single["pixels"] = df_single["resolution"].apply(parse_pixels)

df_folder = pd.read_csv(FOLDER_CSV)
df_folder["pixels"] = df_folder["resolution"].apply(parse_pixels)

per_image = (
    df_single.groupby(["pixels", "resolution", "image", "pieces"])[
        ["total_ms"] + STAGES
    ]
    .mean()
    .reset_index()
)

baseline = per_image[per_image["resolution"] == BASELINE_RES].copy()

per_res = (
    per_image.groupby(["pixels", "resolution"])[["total_ms"] + STAGES]
    .mean()
    .reset_index()
    .sort_values("pixels")
)

baseline_folder = (
    df_folder[df_folder["resolution"] == BASELINE_RES]
    .groupby("image")[["total_ms"] + STAGES]
    .mean()
    .reset_index()
)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
stage_color = {stage: colors[i % len(colors)] for i, stage in enumerate(STAGES)}


def plot_stage_bars(ax):
    means = baseline[STAGES].mean()
    stds = baseline[STAGES].std()
    y = np.arange(len(STAGES))
    ax.barh(
        y,
        means[STAGES],
        xerr=stds[STAGES],
        color=[stage_color[s] for s in STAGES],
        capsize=4,
        alpha=0.8,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(STAGE_LABELS)
    ax.set_xlabel("Runtime (ms)")
    ax.invert_yaxis()


def plot_stages_vs_res(ax):
    x = per_res["pixels"].values
    cumsum = np.zeros(len(x))
    for stage in STAGES:
        y = per_res[stage].values
        new_cumsum = cumsum + y
        ax.fill_between(
            x,
            cumsum,
            new_cumsum,
            alpha=0.15,
            color=stage_color[stage],
            label=stage.replace("_ms", ""),
        )
        ax.plot(x, new_cumsum, alpha=0.5, color=stage_color[stage])
        ax.scatter(x, new_cumsum, color=stage_color[stage], s=20, zorder=3)
        cumsum = new_cumsum
    ax.set_xlabel("Image size")
    ax.set_ylabel("Runtime (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(per_res["resolution"], rotation=15, ha="right")
    ax.legend(fontsize=7, ncols=2)


def plot_stages_vs_pieces(ax):
    sub = baseline.sort_values("pieces")
    x = sub["pieces"].values
    cumsum = np.zeros(len(x))
    for stage in STAGES:
        y = sub[stage].values
        new_cumsum = cumsum + y
        ax.fill_between(
            x,
            cumsum,
            new_cumsum,
            alpha=0.15,
            color=stage_color[stage],
            label=stage.replace("_ms", ""),
        )
        ax.plot(x, new_cumsum, alpha=0.5, color=stage_color[stage])
        ax.scatter(x, new_cumsum, color=stage_color[stage], s=20, zorder=3)
        cumsum = new_cumsum
    ax.set_xlabel("Number of pieces")
    ax.set_ylabel("Runtime (ms)")
    ax.legend(fontsize=7, ncols=2)


def plot_mode_comparison(ax):
    means_single = baseline[STAGES].mean()
    stds_single = baseline[STAGES].std()
    means_folder = baseline_folder[STAGES].mean()
    stds_folder = baseline_folder[STAGES].std()
    y = np.arange(len(STAGES))
    bar_h = 0.35
    ax.barh(
        y - bar_h / 2,
        means_single[STAGES],
        height=bar_h,
        xerr=stds_single[STAGES],
        color=[stage_color[s] for s in STAGES],
        alpha=0.8,
        capsize=3,
        label="single",
    )
    ax.barh(
        y + bar_h / 2,
        means_folder[STAGES],
        height=bar_h,
        xerr=stds_folder[STAGES],
        color=[stage_color[s] for s in STAGES],
        alpha=0.4,
        capsize=3,
        label="folder",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(STAGE_LABELS)
    ax.set_xlabel("Runtime (ms)")
    ax.invert_yaxis()
    ax.legend(fontsize=7)


def plot_mode_comparison_stacked(ax):
    means_single = baseline[STAGES].mean()
    means_folder = baseline_folder[STAGES].mean()
    x = ["single", "folder"]
    cumsum = np.zeros(2)
    for stage in STAGES:
        vals = np.array([means_single[stage], means_folder[stage]])
        ax.bar(
            x,
            vals,
            bottom=cumsum,
            color=stage_color[stage],
            alpha=0.8,
            label=stage.replace("_ms", ""),
        )
        cumsum += vals
    ax.set_ylabel("Runtime (ms)")
    ax.legend(fontsize=7, ncols=2)


REPEAT_IMAGE = "65_7_p1.jpg"


def plot_runs_stacked(ax):
    sub = df_single[
        (df_single["resolution"] == BASELINE_RES) & (df_single["image"] == REPEAT_IMAGE)
    ].sort_values("run")
    x = [f"run {r}" for r in sub["run"].values]
    cumsum = np.zeros(len(sub))
    for stage in STAGES:
        vals = sub[stage].values
        ax.bar(
            x,
            vals,
            bottom=cumsum,
            color=stage_color[stage],
            alpha=0.8,
            label=stage.replace("_ms", ""),
        )
        cumsum += vals
    ax.set_ylabel("Runtime (ms)")
    ax.legend(fontsize=7, ncols=2)


plots = [
    ("serial_stages_vs_res", plot_stages_vs_res),
    ("serial_stages_vs_pieces", plot_stages_vs_pieces),
    ("serial_mode_comparison", plot_mode_comparison),
    ("serial_mode_comparison_stacked", plot_mode_comparison_stacked),
    ("serial_runs_stacked", plot_runs_stacked),
]

sizes = [("_report", (6, 4)), ("_presentation", (10, 6))]

HBAR_PLOTS = {"serial_mode_comparison"}
VBAR_PLOTS = {"serial_mode_comparison_stacked", "serial_runs_stacked"}

for name, plot_fn in plots:
    for suffix, figsize in sizes:
        fig, ax = plt.subplots(figsize=figsize)
        plot_fn(ax)
        if name in HBAR_PLOTS:
            ax.set_xscale("log")
            ax.grid(axis="x", linewidth=0.4, color="gray", alpha=0.4)
        elif name in VBAR_PLOTS:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(500))
            ax.grid(axis="y", linewidth=0.4, color="gray", alpha=0.4)
        else:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(500))
            ax.grid(axis="y", linewidth=0.4, color="gray", alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, name + suffix + ".pdf"))
        plt.close()

print("Done.")
