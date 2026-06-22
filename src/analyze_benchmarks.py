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
    "cuda_setup_ms",
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

BASELINE_RES = "5100x7016"


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
        .median()
        .reset_index()
    )
    baseline = per_image[per_image["resolution"] == BASELINE_RES].copy()
    per_res = (
        per_image.groupby(["pixels", "resolution"])[["total_ms"] + stages]
        .median()
        .reset_index()
        .sort_values("pixels")
    )
    baseline_folder = (
        df_folder[df_folder["resolution"] == BASELINE_RES]
        .groupby("image")[["total_ms"] + stages]
        .median()
        .reset_index()
    )
    return baseline, per_res, baseline_folder, per_image


def make_colors(stages, overrides=None):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    base = {s: colors[i % len(colors)] for i, s in enumerate(SERIAL_STAGES)}
    result = {}
    for stage in stages:
        if overrides and stage in overrides:
            result[stage] = overrides[stage]
        elif stage in base:
            result[stage] = base[stage]
        else:
            result[stage] = colors[len(result) % len(colors)]
    return result


def plot_stages_vs_res(ax, per_res, stages, stage_color):
    x = per_res["resolution"].values
    cumsum = np.zeros(len(x))
    for stage in stages:
        y = per_res[stage].values
        ax.bar(
            x,
            y,
            bottom=cumsum,
            color=stage_color[stage],
            alpha=0.8,
            label=stage.replace("_ms", ""),
        )
        cumsum += y
    ax.set_xlabel("Image size")
    ax.set_ylabel("Runtime (ms)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=7, ncols=2)


def plot_stages_vs_pieces(ax, baseline, stages, stage_color):
    sub = baseline.sort_values("pieces")
    x = sub["pieces"].astype(str).values
    cumsum = np.zeros(len(sub))
    for stage in stages:
        y = sub[stage].values
        ax.bar(
            x,
            y,
            bottom=cumsum,
            color=stage_color[stage],
            alpha=0.8,
            label=stage.replace("_ms", ""),
        )
        cumsum += y
    ax.set_xlabel("Number of pieces")
    ax.set_ylabel("Runtime (ms)")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.legend(fontsize=7, ncols=2)


def plot_mode_comparison(ax, baseline, baseline_folder, stages, stage_color):
    def iqr_err(df, stages):
        med = df[stages].median()
        q1 = df[stages].quantile(0.25)
        q3 = df[stages].quantile(0.75)
        return med, [med - q1, q3 - med]

    med_s, xerr_s = iqr_err(baseline, stages)
    med_f, xerr_f = iqr_err(baseline_folder, stages)
    y = np.arange(len(stages))
    bar_h = 0.35
    ax.barh(
        y - bar_h / 2,
        med_s[stages],
        height=bar_h,
        xerr=np.array([[xerr_s[0][s] for s in stages], [xerr_s[1][s] for s in stages]]),
        color=[stage_color[s] for s in stages],
        alpha=0.8,
        capsize=3,
        label="single",
    )
    ax.barh(
        y + bar_h / 2,
        med_f[stages],
        height=bar_h,
        xerr=np.array([[xerr_f[0][s] for s in stages], [xerr_f[1][s] for s in stages]]),
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
    med_single = baseline[stages].median()
    med_folder = baseline_folder[stages].median()
    x = ["single", "folder"]
    cumsum = np.zeros(2)
    for stage in stages:
        vals = np.array([med_single[stage], med_folder[stage]])
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


def plot_runs_stacked(ax, baseline_folder, stages, stage_color):
    sub = baseline_folder.head(5)
    x = [f"image {i+1}" for i in range(len(sub))]
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


def save_plots(
    prefix, baseline, per_res, baseline_folder, stages, color_overrides=None
):
    sc = make_colors(stages, overrides=color_overrides)
    HBAR = {f"{prefix}_mode_comparison"}
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
            lambda ax: plot_runs_stacked(ax, baseline_folder, stages, sc),
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
            else:
                ax.set_ylim(bottom=0)
                ymax = ax.get_ylim()[1]
                step = max(1, round(ymax / 6 / 50) * 50) if ymax > 0 else 500
                ax.yaxis.set_major_locator(plt.MultipleLocator(step))
                ax.grid(axis="y", linewidth=0.4, color="gray", alpha=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, name + suffix + ".pdf"))
            plt.close()


baseline_s, per_res_s, baseline_folder_s, per_image_s = load_data(
    os.path.join(BASE, "data/benchmark_results/serial_single.csv"),
    os.path.join(BASE, "data/benchmark_results/serial_folder.csv"),
    SERIAL_STAGES,
)
save_plots("serial", baseline_s, per_res_s, baseline_folder_s, SERIAL_STAGES)

baseline_c, per_res_c, baseline_folder_c, per_image_c = load_data(
    os.path.join(BASE, "data/benchmark_results/cuda_single.csv"),
    os.path.join(BASE, "data/benchmark_results/cuda_folder.csv"),
    CUDA_STAGES,
)
save_plots(
    "cuda",
    baseline_c,
    per_res_c,
    baseline_folder_c,
    CUDA_STAGES,
    color_overrides={"cuda_setup_ms": "#39ff14"},
)


def plot_combined_mode_comparison(
    ax, baseline_s, baseline_folder_s, baseline_c, baseline_folder_c
):
    def get_medians(df, stages):
        m = df[stages].median()
        return {s: m[s] for s in stages}

    ms = get_medians(baseline_s, SERIAL_STAGES)
    mfs = get_medians(baseline_folder_s, SERIAL_STAGES)
    mc = get_medians(baseline_c, CUDA_STAGES)
    mfc = get_medians(baseline_folder_c, CUDA_STAGES)

    bar_h = 0.18
    y = np.arange(len(CUDA_STAGES))
    labels = [s.replace("_ms", "") for s in CUDA_STAGES]

    colors4 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]

    def val(d, stage):
        return d.get(stage, 0.0)

    ax.barh(
        y - 1.5 * bar_h,
        [val(ms, s) for s in CUDA_STAGES],
        height=bar_h,
        color=colors4[0],
        label="serial single",
    )
    ax.barh(
        y - 0.5 * bar_h,
        [val(mfs, s) for s in CUDA_STAGES],
        height=bar_h,
        color=colors4[1],
        label="serial folder",
    )
    ax.barh(
        y + 0.5 * bar_h,
        [val(mc, s) for s in CUDA_STAGES],
        height=bar_h,
        color=colors4[2],
        label="cuda single",
    )
    ax.barh(
        y + 1.5 * bar_h,
        [val(mfc, s) for s in CUDA_STAGES],
        height=bar_h,
        color=colors4[3],
        label="cuda folder",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Runtime (ms)")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(axis="x", linewidth=0.4, color="gray", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7)


def plot_combined_vs_res(ax, per_res_s, per_res_c):
    resolutions = per_res_s["resolution"].values
    x = np.arange(len(resolutions))
    bar_w = 0.35
    ax.bar(
        x - bar_w / 2,
        per_res_s["total_ms"].values,
        width=bar_w,
        color="#1f77b4",
        label="serial",
    )
    ax.bar(
        x + bar_w / 2,
        per_res_c["total_ms"].values,
        width=bar_w,
        color="#ff7f0e",
        label="cuda",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions, rotation=15, ha="right")
    ax.set_xlabel("Image size")
    ax.set_ylabel("Runtime (ms)")
    ax.legend(fontsize=7)


for suffix, figsize in [("_report", (6, 4)), ("_presentation", (10, 6))]:
    fig, ax = plt.subplots(figsize=figsize)
    plot_combined_vs_res(ax, per_res_s, per_res_c)
    ax.set_yscale("log")
    ax.grid(axis="y", linewidth=0.4, color="gray", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"combined_vs_res{suffix}.pdf"))
    plt.close()


def plot_combined_vs_pieces(ax, baseline_s, baseline_c):
    sub_s = baseline_s.sort_values("pieces")
    sub_c = baseline_c.sort_values("pieces")
    pieces = sub_s["pieces"].astype(str).values
    x = np.arange(len(pieces))
    bar_w = 0.35
    ax.bar(
        x - bar_w / 2,
        sub_s["total_ms"].values,
        width=bar_w,
        color="#1f77b4",
        label="serial",
    )
    ax.bar(
        x + bar_w / 2,
        sub_c["total_ms"].values,
        width=bar_w,
        color="#ff7f0e",
        label="cuda",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pieces, rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("Number of pieces")
    ax.set_ylabel("Runtime (ms)")
    ax.legend(fontsize=7)


for suffix, figsize in [("_report", (6, 4)), ("_presentation", (10, 6))]:
    fig, ax = plt.subplots(figsize=figsize)
    plot_combined_vs_pieces(ax, baseline_s, baseline_c)
    ax.set_yscale("log")
    ax.grid(axis="y", linewidth=0.4, color="gray", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"combined_vs_pieces{suffix}.pdf"))
    plt.close()

for suffix, figsize in [("_report", (6, 4)), ("_presentation", (10, 6))]:
    fig, ax = plt.subplots(figsize=figsize)
    plot_combined_mode_comparison(
        ax, baseline_s, baseline_folder_s, baseline_c, baseline_folder_c
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"combined_mode_comparison{suffix}.pdf"))
    plt.close()


def plot_speedup(ax, baseline_s, baseline_folder_s, baseline_c, baseline_folder_c):
    def get_medians(df, stages):
        m = df[stages].median()
        return {s: m[s] for s in stages}

    ms = get_medians(baseline_s, SERIAL_STAGES)
    mfs = get_medians(baseline_folder_s, SERIAL_STAGES)
    mc = get_medians(baseline_c, CUDA_STAGES)
    mfc = get_medians(baseline_folder_c, CUDA_STAGES)

    def speedup(serial, cuda, stage):
        s = serial.get(stage, 0.0)
        c = cuda.get(stage, 0.0)
        return s / c if c > 0 and s > 0 else 0.0

    bar_h = 0.3
    y = np.arange(len(CUDA_STAGES))
    labels = [s.replace("_ms", "") for s in CUDA_STAGES]

    su_single = [speedup(ms, mc, s) for s in CUDA_STAGES]
    su_folder = [speedup(mfs, mfc, s) for s in CUDA_STAGES]

    ax.barh(y - bar_h / 2, su_single, height=bar_h, color="#1f77b4", label="single")
    ax.barh(y + bar_h / 2, su_folder, height=bar_h, color="#aec7e8", label="folder")

    for i, (vs, vf) in enumerate(zip(su_single, su_folder)):
        if vs > 0:
            ax.text(vs, i - bar_h / 2, f" {vs:.1f}", va="center", fontsize=7)
        if vf > 0:
            ax.text(vf, i + bar_h / 2, f" {vf:.1f}", va="center", fontsize=7)
    ax.axvline(x=1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Speedup (serial / cuda)")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(axis="x", linewidth=0.4, color="gray", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7)


for suffix, figsize in [("_report", (6, 4)), ("_presentation", (10, 6))]:
    fig, ax = plt.subplots(figsize=figsize)
    plot_speedup(ax, baseline_s, baseline_folder_s, baseline_c, baseline_folder_c)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"speedup{suffix}.pdf"))
    plt.close()


def plot_transfer_overhead(ax):
    stages = [
        "CUDA setup",
        "Preprocessing",
        "Connected components",
        "Boundary extraction",
        "Contour extraction",
        "Contour smoothing",
        "Enclosing circle",
        "Radial signal",
        "Signal smoothing",
        "Peak detection",
        "Edge classification",
        "Visualization",
    ]
    compute = [
        8.54,
        21.83,
        46.71,
        1.68,
        0.0,
        0.010,
        0.007,
        0.006,
        0.009,
        0.006,
        0.0,
        0.0,
    ]
    htod = [11.868, 0.001, 0.0, 0.0, 0.0, 0.042, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dtoh = [0.0, 0.002, 0.005, 2.083, 0.0, 0.0, 0.0, 0.031, 0.031, 0.021, 0.0, 0.0]

    eps = 1e-4
    y = np.arange(len(stages))
    bar_h = 0.25

    ax.barh(
        y - bar_h,
        [max(v, eps) for v in compute],
        height=bar_h,
        color="#1f77b4",
        label="compute",
    )
    ax.barh(y, [max(v, eps) for v in htod], height=bar_h, color="#ff7f0e", label="HtoD")
    ax.barh(
        y + bar_h,
        [max(v, eps) for v in dtoh],
        height=bar_h,
        color="#2ca02c",
        label="DtoH",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(stages)
    ax.set_xlabel("Runtime (ms)")
    ax.set_xscale("log")
    ax.set_xlim(left=1e-4)
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    ax.grid(axis="x", linewidth=0.4, color="gray", alpha=0.4)
    ax.set_axisbelow(True)


for suffix, figsize in [("_report", (6, 4)), ("_presentation", (10, 6))]:
    fig, ax = plt.subplots(figsize=figsize)
    plot_transfer_overhead(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"transfer_overhead{suffix}.pdf"))
    plt.close()

print("Done.")
