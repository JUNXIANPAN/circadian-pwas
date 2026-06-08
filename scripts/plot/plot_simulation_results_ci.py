
"""
plot_simulation_results_ci.py

Improved plotting script for the 6-scenario circadian-informed PWAS simulation study.

Main updates relative to the original script
--------------------------------------------
1) Error bars use 95% confidence intervals instead of standard deviation.
2) Main figures emphasize paired within-replicate improvements:
       delta = weighted - baseline
3) Informative / noninformative / null scenarios are visually distinguished.
4) A strip of per-replicate points is added for delta plots to show variability.
5) Tail-error plot uses log scale for better visibility.

Expected directory structure
----------------------------
Each run folder from circadian_simulation.py should contain:
- simulation_summary_<scenario>.tsv
- simulation_results_<scenario>.tsv
- simulation_config_<scenario>.json

Example
-------
python plot_simulation_results_ci.py \
  --base_dir . \
  --a sim_A_strong_informative \
  --b sim_B_moderate_informative \
  --c sim_C_weak_informative \
  --d sim_D_moderate_noninformative \
  --e sim_E_weak_noninformative \
  --f sim_F_global_null \
  --outdir sim_plots_ci
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANNOTATION_COLORS = {
    "informative": "#4C78A8",
    "noninformative": "#9AA0A6",
    "global_null": "#E45756",
}


def load_one_run(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    summary_files = list(run_dir.glob("simulation_summary_*.tsv"))
    result_files = list(run_dir.glob("simulation_results_*.tsv"))
    config_files = list(run_dir.glob("simulation_config_*.json"))

    if len(summary_files) != 1:
        raise FileNotFoundError(f"Expected exactly 1 summary TSV in {run_dir}, found {len(summary_files)}")
    if len(result_files) != 1:
        raise FileNotFoundError(f"Expected exactly 1 results TSV in {run_dir}, found {len(result_files)}")
    if len(config_files) != 1:
        raise FileNotFoundError(f"Expected exactly 1 config JSON in {run_dir}, found {len(config_files)}")

    summary_df = pd.read_csv(summary_files[0], sep="\t")
    result_df = pd.read_csv(result_files[0], sep="\t")
    config = json.loads(config_files[0].read_text(encoding="utf-8"))
    return summary_df, result_df, config


def metric_from_summary(summary_df: pd.DataFrame, metric: str, stat: str = "mean") -> float:
    row = summary_df.loc[summary_df["metric"] == metric]
    if row.empty:
        return np.nan
    value = row.iloc[0][stat]
    return float(value) if pd.notna(value) else np.nan


def mean_se_ci(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, 0.0, 0.0
    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(len(x))
    ci = 1.96 * se
    return mean, se, ci


def classify_scenario(config: Dict) -> str:
    return str(config.get("scenario", "unknown"))


def pretty_label_map() -> Dict[str, str]:
    return {
        "A": "A\nStrong\nInformative",
        "B": "B\nModerate\nInformative",
        "C": "C\nWeak\nInformative",
        "D": "D\nModerate\nNoninformative",
        "E": "E\nWeak\nNoninformative",
        "F": "F\nGlobal\nNull",
    }


def build_master_table(run_map: Dict[str, Path]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    result_tables = {}
    labels = pretty_label_map()

    for key, run_dir in run_map.items():
        summary_df, result_df, config = load_one_run(run_dir)
        result_tables[key] = result_df.copy()

        scenario_type = classify_scenario(config)

        row = {
            "scenario_id": key,
            "label": labels.get(key, key),
            "run_dir": str(run_dir),
            "scenario": scenario_type,
            "mu_causal": config.get("mu_causal"),
            "gamma_prior": config.get("gamma_prior"),
            "n_reps": config.get("n_reps"),
            "color": ANNOTATION_COLORS.get(scenario_type, "#777777"),
            "baseline_auc_mean": metric_from_summary(summary_df, "baseline_auc", "mean"),
            "weighted_auc_mean": metric_from_summary(summary_df, "weighted_auc", "mean"),
            "baseline_ap_mean": metric_from_summary(summary_df, "baseline_ap", "mean"),
            "weighted_ap_mean": metric_from_summary(summary_df, "weighted_ap", "mean"),
            "baseline_topk_mean": metric_from_summary(summary_df, "baseline_topk_hits", "mean"),
            "weighted_topk_mean": metric_from_summary(summary_df, "weighted_topk_hits", "mean"),
            "baseline_null_fpr_mean": metric_from_summary(summary_df, "baseline_null_fpr_absz_ge_5", "mean"),
            "weighted_null_fpr_mean": metric_from_summary(summary_df, "weighted_null_fpr_absz_ge_5", "mean"),
        }

        # Pairwise deltas with 95% CI
        if {"baseline_auc", "weighted_auc"}.issubset(result_df.columns):
            delta_auc = (result_df["weighted_auc"] - result_df["baseline_auc"]).to_numpy(dtype=float)
            row["delta_auc_mean"], row["delta_auc_se"], row["delta_auc_ci"] = mean_se_ci(delta_auc)
        else:
            row["delta_auc_mean"], row["delta_auc_se"], row["delta_auc_ci"] = np.nan, np.nan, np.nan

        if {"baseline_ap", "weighted_ap"}.issubset(result_df.columns):
            delta_ap = (result_df["weighted_ap"] - result_df["baseline_ap"]).to_numpy(dtype=float)
            row["delta_ap_mean"], row["delta_ap_se"], row["delta_ap_ci"] = mean_se_ci(delta_ap)
        else:
            row["delta_ap_mean"], row["delta_ap_se"], row["delta_ap_ci"] = np.nan, np.nan, np.nan

        if {"baseline_topk_hits", "weighted_topk_hits"}.issubset(result_df.columns):
            delta_topk = (result_df["weighted_topk_hits"] - result_df["baseline_topk_hits"]).to_numpy(dtype=float)
            row["delta_topk_mean"], row["delta_topk_se"], row["delta_topk_ci"] = mean_se_ci(delta_topk)
        else:
            row["delta_topk_mean"], row["delta_topk_se"], row["delta_topk_ci"] = np.nan, np.nan, np.nan

        if {"baseline_null_fpr_absz_ge_5", "weighted_null_fpr_absz_ge_5"}.issubset(result_df.columns):
            delta_null = (
                result_df["weighted_null_fpr_absz_ge_5"] - result_df["baseline_null_fpr_absz_ge_5"]
            ).to_numpy(dtype=float)
            row["delta_null_fpr_mean"], row["delta_null_fpr_se"], row["delta_null_fpr_ci"] = mean_se_ci(delta_null)
        else:
            row["delta_null_fpr_mean"], row["delta_null_fpr_se"], row["delta_null_fpr_ci"] = np.nan, np.nan, np.nan

        rows.append(row)

    master = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
    return master, result_tables


def save_master_table(master: pd.DataFrame, outdir: Path) -> None:
    master.to_csv(outdir / "combined_simulation_metrics_ci.tsv", sep="\t", index=False)


def add_group_separator(ax) -> None:
    ax.axvline(2.5, linestyle="--", linewidth=1)
    ax.axvline(4.5, linestyle="--", linewidth=1)


def add_group_labels(ax, ypad: float = 0.02) -> None:
    ymin, ymax = ax.get_ylim()
    y = ymax + (ymax - ymin) * ypad
    ax.text(1.0, y, "Informative", ha="center", va="bottom", fontsize=10)
    ax.text(3.5, y, "Noninformative", ha="center", va="bottom", fontsize=10)
    ax.text(5.0, y, "Global null", ha="center", va="bottom", fontsize=10)


def jittered_scatter(ax, x0: float, values: np.ndarray, color: str, alpha: float = 0.22, scale: float = 0.10) -> None:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return
    rng = np.random.default_rng(12345 + int(round(x0 * 100)))
    xs = x0 + rng.normal(0, scale, size=len(values))
    ax.scatter(xs, values, s=10, alpha=alpha, color=color, edgecolors="none")


def plot_delta_metric(master: pd.DataFrame, result_tables: Dict[str, pd.DataFrame], outdir: Path,
                      metric_name: str, outname: str, ylabel: str, title: str) -> None:
    metric_map = {
        "auc": ("baseline_auc", "weighted_auc", "delta_auc_mean", "delta_auc_ci"),
        "ap": ("baseline_ap", "weighted_ap", "delta_ap_mean", "delta_ap_ci"),
        "topk": ("baseline_topk_hits", "weighted_topk_hits", "delta_topk_mean", "delta_topk_ci"),
    }
    base_col, wt_col, mean_col, ci_col = metric_map[metric_name]

    df = master.loc[master["scenario_id"].isin(["A", "B", "C", "D", "E"])].copy()
    x = np.arange(len(df))

    plt.figure(figsize=(10.5, 5.4))

    for i, row in enumerate(df.itertuples(index=False)):
        color = row.color
        key = row.scenario_id
        rep = result_tables[key]
        vals = (rep[wt_col] - rep[base_col]).to_numpy(dtype=float)
        jittered_scatter(plt.gca(), x[i], vals, color=color)

    plt.bar(
        x,
        df[mean_col],
        yerr=df[ci_col],
        capsize=4,
        color=df["color"],
        edgecolor="black",
        linewidth=0.6,
    )
    plt.axhline(0, linewidth=1)
    add_group_separator(plt.gca())
    plt.xticks(x, df["label"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outdir / outname, dpi=300)
    plt.close()


def plot_baseline_vs_weighted_auc_ci(master: pd.DataFrame, result_tables: Dict[str, pd.DataFrame], outdir: Path) -> None:
    df = master.loc[master["scenario_id"].isin(["A", "B", "C", "D", "E"])].copy()
    x = np.arange(len(df))
    width = 0.38

    baseline_means, baseline_cis = [], []
    weighted_means, weighted_cis = [], []

    for row in df.itertuples(index=False):
        rep = result_tables[row.scenario_id]
        bmean, _, bci = mean_se_ci(rep["baseline_auc"].to_numpy(dtype=float))
        wmean, _, wci = mean_se_ci(rep["weighted_auc"].to_numpy(dtype=float))
        baseline_means.append(bmean)
        baseline_cis.append(bci)
        weighted_means.append(wmean)
        weighted_cis.append(wci)

    plt.figure(figsize=(10.5, 5.4))
    plt.bar(x - width/2, baseline_means, width=width, yerr=baseline_cis, capsize=4, label="Baseline", edgecolor="black", linewidth=0.6)
    plt.bar(x + width/2, weighted_means, width=width, yerr=weighted_cis, capsize=4, label="Weighted", edgecolor="black", linewidth=0.6)
    add_group_separator(plt.gca())
    plt.xticks(x, df["label"])
    plt.ylabel("AUC")
    ymax = max(np.nanmax(baseline_means), np.nanmax(weighted_means))
    ymin = min(np.nanmin(baseline_means), np.nanmin(weighted_means))
    plt.ylim(max(0.0, ymin - 0.03), min(1.0, ymax + 0.05))
    plt.title("Baseline vs weighted AUC (95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figure3_baseline_vs_weighted_auc_ci.png", dpi=300)
    plt.close()


def plot_null_fpr_log(master: pd.DataFrame, result_tables: Dict[str, pd.DataFrame], outdir: Path) -> None:
    df = master.copy()
    x = np.arange(len(df))
    width = 0.38

    baseline_means, baseline_cis = [], []
    weighted_means, weighted_cis = [], []

    for row in df.itertuples(index=False):
        rep = result_tables[row.scenario_id]
        b = rep["baseline_null_fpr_absz_ge_5"].to_numpy(dtype=float)
        w = rep["weighted_null_fpr_absz_ge_5"].to_numpy(dtype=float)
        bmean, _, bci = mean_se_ci(b)
        wmean, _, wci = mean_se_ci(w)

        # avoid log(0); keep a tiny floor just for visualization
        floor = 1e-7
        baseline_means.append(max(bmean, floor))
        baseline_cis.append(max(bci, 0.0))
        weighted_means.append(max(wmean, floor))
        weighted_cis.append(max(wci, 0.0))

    plt.figure(figsize=(10.5, 5.4))
    plt.bar(x - width/2, baseline_means, width=width, label="Baseline", edgecolor="black", linewidth=0.6)
    plt.bar(x + width/2, weighted_means, width=width, label="Weighted", edgecolor="black", linewidth=0.6)

    add_group_separator(plt.gca())
    plt.xticks(x, df["label"])
    plt.yscale("log")
    plt.ylabel("P(|Z| ≥ 5)  (log scale)")
    plt.title("Extreme-tail error / calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figure4_null_fpr_log.png", dpi=300)
    plt.close()


def plot_main_paper_panel(master: pd.DataFrame, result_tables: Dict[str, pd.DataFrame], outdir: Path) -> None:
    df = master.loc[master["scenario_id"].isin(["A", "B", "C", "D", "E"])].copy()
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: ΔAUC
    for i, row in enumerate(df.itertuples(index=False)):
        rep = result_tables[row.scenario_id]
        vals = (rep["weighted_auc"] - rep["baseline_auc"]).to_numpy(dtype=float)
        jittered_scatter(axes[0], x[i], vals, row.color)
    axes[0].bar(x, df["delta_auc_mean"], yerr=df["delta_auc_ci"], capsize=4,
                color=df["color"], edgecolor="black", linewidth=0.6)
    axes[0].axhline(0, linewidth=1)
    add_group_separator(axes[0])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["label"])
    axes[0].set_ylabel("Weighted - Baseline")
    axes[0].set_title("ΔAUC (95% CI)")

    # Panel 2: ΔTop-k
    for i, row in enumerate(df.itertuples(index=False)):
        rep = result_tables[row.scenario_id]
        vals = (rep["weighted_topk_hits"] - rep["baseline_topk_hits"]).to_numpy(dtype=float)
        jittered_scatter(axes[1], x[i], vals, row.color)
    axes[1].bar(x, df["delta_topk_mean"], yerr=df["delta_topk_ci"], capsize=4,
                color=df["color"], edgecolor="black", linewidth=0.6)
    axes[1].axhline(0, linewidth=1)
    add_group_separator(axes[1])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["label"])
    axes[1].set_ylabel("Weighted - Baseline")
    axes[1].set_title("ΔTop-k hits (95% CI)")

    # Panel 3: tail error in all 6 runs
    x2 = np.arange(len(master))
    baseline_vals = np.maximum(master["baseline_null_fpr_mean"].to_numpy(dtype=float), 1e-7)
    weighted_vals = np.maximum(master["weighted_null_fpr_mean"].to_numpy(dtype=float), 1e-7)
    axes[2].bar(x2 - 0.19, baseline_vals, width=0.38, label="Baseline", edgecolor="black", linewidth=0.6)
    axes[2].bar(x2 + 0.19, weighted_vals, width=0.38, label="Weighted", edgecolor="black", linewidth=0.6)
    add_group_separator(axes[2])
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(master["label"])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("P(|Z| ≥ 5)")
    axes[2].set_title("Tail error / calibration")
    axes[2].legend()

    fig.suptitle("Circadian-informed PWAS simulation summary", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "figure0_main_panel_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot 6-scenario circadian simulation results with 95% CI.")
    p.add_argument("--base_dir", type=str, default=".", help="Base directory containing run folders.")
    p.add_argument("--a", type=str, required=True, help="Run folder for scenario A.")
    p.add_argument("--b", type=str, required=True, help="Run folder for scenario B.")
    p.add_argument("--c", type=str, required=True, help="Run folder for scenario C.")
    p.add_argument("--d", type=str, required=True, help="Run folder for scenario D.")
    p.add_argument("--e", type=str, required=True, help="Run folder for scenario E.")
    p.add_argument("--f", type=str, required=True, help="Run folder for scenario F.")
    p.add_argument("--outdir", type=str, default="sim_plots_ci", help="Output directory for figures.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(args.base_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_map = {
        "A": base_dir / args.a,
        "B": base_dir / args.b,
        "C": base_dir / args.c,
        "D": base_dir / args.d,
        "E": base_dir / args.e,
        "F": base_dir / args.f,
    }

    master, result_tables = build_master_table(run_map)
    save_master_table(master, outdir)

    plot_main_paper_panel(master, result_tables, outdir)
    plot_delta_metric(
        master, result_tables, outdir,
        metric_name="auc",
        outname="figure1_delta_auc_ci.png",
        ylabel="Weighted - Baseline AUC",
        title="AUC improvement across simulation scenarios (95% CI)",
    )
    plot_delta_metric(
        master, result_tables, outdir,
        metric_name="ap",
        outname="figure2_delta_ap_ci.png",
        ylabel="Weighted - Baseline AP",
        title="Average precision improvement across simulation scenarios (95% CI)",
    )
    plot_delta_metric(
        master, result_tables, outdir,
        metric_name="topk",
        outname="figure3_delta_topk_ci.png",
        ylabel="Weighted - Baseline top-k hits",
        title="Top-k recovery gain across simulation scenarios (95% CI)",
    )
    plot_baseline_vs_weighted_auc_ci(master, result_tables, outdir)
    plot_null_fpr_log(master, result_tables, outdir)

    print("\nSaved files:")
    for path in sorted(outdir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
