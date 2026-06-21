import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data/benchmark_results")

SERIAL_STAGES = [
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

CUDA_STAGES = [
    "preprocessing_ms",
    "cuda_setup_ms",
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

BASELINE_RES = "5100x7016"
REPEAT_IMAGE = "65_7_p1.jpg"


def parse_pixels(res):
    w, h = res.split("x")
    return int(w) * int(h)


def load_data(single_csv, folder_csv, stages):
    df_single = pd.read_csv(single_csv)
    df_single["pixels"] = df_single["resolution"].apply(parse_pixels)
    df_folder = pd.read_csv(folder_csv)
    df_folder["pixels"] = df_folder["resolution"].apply(parse_pixels)

    per_image = (
        df_single.groupby(["pixels", "resolution", "image", "pieces"])[
            ["total_ms"] + stages
        ]
        .mean()
        .reset_index()
    )
    baseline = per_image[per_image["resolution"] == BASELINE_RES].copy()
    per_res = (
        per_image.groupby(["pixels", "resolution"])[["total_ms"] + stages]
        .mean()
        .reset_index()
        .sort_values("pixels")
    )
    baseline_folder = (
        df_folder[df_folder["resolution"] == BASELINE_RES]
        .groupby("image")[["total_ms"] + stages]
        .mean()
        .reset_index()
    )
    return df_single, baseline, per_res, baseline_folder


def make_colors(stages):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {s: colors[i % len(colors)] for i, s in enumerate(stages)}


def plot_stages_vs_res(ax, per_res, stages, stage_color):
    x = per_res["pixels"].values
    cumsum = np.zeros(len(x))
    for stage in stages:
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


def plot_stages_vs_pieces(ax, baseline, stages, stage_color):
    sub = baseline.sort_values("pieces")
    x = sub["pieces"].values
    cumsum = np.zeros(len(x))
    for stage in stages:
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


def plot_mode_comparison(ax, baseline, baseline_folder, stages, stage_color):
    means_single = baseline[stages].mean()
    stds_single = baseline[stages].std()
    means_folder = baseline_folder[stages].mean()
    stds_folder = baseline_folder[stages].std()
    y = np.arange(len(stages))
    bar_h = 0.35
    ax.barh(
        y - bar_h / 2,
        means_single[stages],
        height=bar_h,
        xerr=stds_single[stages],
        color=[stage_color[s] for s in stages],
        alpha=0.8,
        capsize=3,
        label="single",
    )
    ax.barh(
        y + bar_h / 2,
        means_folder[stages],
        height=bar_h,
        xerr=stds_folder[stages],
        color=[stage_color[s] for s in stages],
        alpha=0.4,
        capsize=3,
        label="folder",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_ms", "") for s in stages])
    ax.set_xlabel("Runtime (ms)")
    ax.invert_yaxis()
    ax.legend(fontsize=7)


def plot_mode_comparison_stacked(ax, baseline, baseline_folder, stages, stage_color):
    means_single = baseline[stages].mean()
    means_folder = baseline_folder[stages].mean()
    x = ["single", "folder"]
    cumsum = np.zeros(2)
    for stage in stages:
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


def plot_runs_stacked(ax, df_single, stages, stage_color):
    sub = df_single[
        (df_single["resolution"] == BASELINE_RES) & (df_single["image"] == REPEAT_IMAGE)
    ].sort_values("run")
    x = [f"run {r}" for r in sub["run"].values]
    cumsum = np.zeros(len(sub))
    for stage in stages:
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


def save_plots(prefix, df_single, baseline, per_res, baseline_folder, stages):
    sc = make_colors(stages)
    HBAR = {f"{prefix}_mode_comparison"}
    VBAR = {f"{prefix}_mode_comparison_stacked", f"{prefix}_runs_stacked"}
    plots = [
        (
            f"{prefix}_stages_vs_res",
            lambda ax: plot_stages_vs_res(ax, per_res, stages, sc),
        ),
        (
            f"{prefix}_stages_vs_pieces",
            lambda ax: plot_stages_vs_pieces(ax, baseline, stages, sc),
        ),
        (
            f"{prefix}_mode_comparison",
            lambda ax: plot_mode_comparison(ax, baseline, baseline_folder, stages, sc),
        ),
        (
            f"{prefix}_mode_comparison_stacked",
            lambda ax: plot_mode_comparison_stacked(
                ax, baseline, baseline_folder, stages, sc
            ),
        ),
        (
            f"{prefix}_runs_stacked",
            lambda ax: plot_runs_stacked(ax, df_single, stages, sc),
        ),
    ]
    sizes = [("_report", (6, 4)), ("_presentation", (10, 6))]
    for name, plot_fn in plots:
        for suffix, figsize in sizes:
            fig, ax = plt.subplots(figsize=figsize)
            plot_fn(ax)
            if name in HBAR:
                ax.set_xscale("log")
                ax.grid(axis="x", linewidth=0.4, color="gray", alpha=0.4)
            elif name in VBAR:
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


df_s, baseline_s, per_res_s, baseline_folder_s = load_data(
    os.path.join(BASE, "data/benchmark_results/serial_single.csv"),
    os.path.join(BASE, "data/benchmark_results/serial_folder.csv"),
    SERIAL_STAGES,
)
save_plots("serial", df_s, baseline_s, per_res_s, baseline_folder_s, SERIAL_STAGES)

df_c, baseline_c, per_res_c, baseline_folder_c = load_data(
    os.path.join(BASE, "data/benchmark_results/cuda_single.csv"),
    os.path.join(BASE, "data/benchmark_results/cuda_folder.csv"),
    CUDA_STAGES,
)
save_plots("cuda", df_c, baseline_c, per_res_c, baseline_folder_c, CUDA_STAGES)

print("Done.")
