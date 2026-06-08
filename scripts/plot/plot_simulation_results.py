
"""
plot_simulation_results.py

Plotting script for the 6-scenario circadian-informed PWAS simulation study.

Expected directory structure
----------------------------
You should have six output folders from circadian_simulation.py, each containing:
- simulation_summary_<scenario>.tsv
- simulation_results_<scenario>.tsv
- simulation_config_<scenario>.json

Recommended folders:
- sim_A_strong_informative
- sim_B_moderate_informative
- sim_C_weak_informative
- sim_D_moderate_noninformative
- sim_E_weak_noninformative
- sim_F_global_null

Example
-------
python plot_simulation_results.py \
  --base_dir . \
  --a sim_A_strong_informative \
  --b sim_B_moderate_informative \
  --c sim_C_weak_informative \
  --d sim_D_moderate_noninformative \
  --e sim_E_weak_noninformative \
  --f sim_F_global_null \
  --outdir sim_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        raise KeyError(f"Metric '{metric}' not found in summary table.")
    return float(row.iloc[0][stat])


def build_master_table(run_map: Dict[str, Path]) -> pd.DataFrame:
    rows = []

    pretty_label_map = {
        "A": "A\nStrong\nInformative",
        "B": "B\nModerate\nInformative",
        "C": "C\nWeak\nInformative",
        "D": "D\nModerate\nNoninformative",
        "E": "E\nWeak\nNoninformative",
        "F": "F\nGlobal\nNull",
    }

    for key, run_dir in run_map.items():
        summary_df, result_df, config = load_one_run(run_dir)

        row = {
            "scenario_id": key,
            "label": pretty_label_map.get(key, key),
            "run_dir": str(run_dir),
            "scenario": config.get("scenario"),
            "mu_causal": config.get("mu_causal"),
            "gamma_prior": config.get("gamma_prior"),
            "n_reps": config.get("n_reps"),
            "baseline_auc_mean": metric_from_summary(summary_df, "baseline_auc", "mean"),
            "baseline_auc_std": metric_from_summary(summary_df, "baseline_auc", "std"),
            "weighted_auc_mean": metric_from_summary(summary_df, "weighted_auc", "mean"),
            "weighted_auc_std": metric_from_summary(summary_df, "weighted_auc", "std"),
            "baseline_ap_mean": metric_from_summary(summary_df, "baseline_ap", "mean"),
            "baseline_ap_std": metric_from_summary(summary_df, "baseline_ap", "std"),
            "weighted_ap_mean": metric_from_summary(summary_df, "weighted_ap", "mean"),
            "weighted_ap_std": metric_from_summary(summary_df, "weighted_ap", "std"),
            "baseline_topk_mean": metric_from_summary(summary_df, "baseline_topk_hits", "mean"),
            "baseline_topk_std": metric_from_summary(summary_df, "baseline_topk_hits", "std"),
            "weighted_topk_mean": metric_from_summary(summary_df, "weighted_topk_hits", "mean"),
            "weighted_topk_std": metric_from_summary(summary_df, "weighted_topk_hits", "std"),
            "baseline_null_fpr_mean": metric_from_summary(summary_df, "baseline_null_fpr_absz_ge_5", "mean"),
            "baseline_null_fpr_std": metric_from_summary(summary_df, "baseline_null_fpr_absz_ge_5", "std"),
            "weighted_null_fpr_mean": metric_from_summary(summary_df, "weighted_null_fpr_absz_ge_5", "mean"),
            "weighted_null_fpr_std": metric_from_summary(summary_df, "weighted_null_fpr_absz_ge_5", "std"),
        }

        # Per-rep deltas for error bars of improvement plots
        if {"baseline_auc", "weighted_auc"}.issubset(result_df.columns):
            row["delta_auc_mean"] = float((result_df["weighted_auc"] - result_df["baseline_auc"]).mean())
            row["delta_auc_std"] = float((result_df["weighted_auc"] - result_df["baseline_auc"]).std())
        else:
            row["delta_auc_mean"] = np.nan
            row["delta_auc_std"] = np.nan

        if {"baseline_ap", "weighted_ap"}.issubset(result_df.columns):
            row["delta_ap_mean"] = float((result_df["weighted_ap"] - result_df["baseline_ap"]).mean())
            row["delta_ap_std"] = float((result_df["weighted_ap"] - result_df["baseline_ap"]).std())
        else:
            row["delta_ap_mean"] = np.nan
            row["delta_ap_std"] = np.nan

        if {"baseline_topk_hits", "weighted_topk_hits"}.issubset(result_df.columns):
            row["delta_topk_mean"] = float((result_df["weighted_topk_hits"] - result_df["baseline_topk_hits"]).mean())
            row["delta_topk_std"] = float((result_df["weighted_topk_hits"] - result_df["baseline_topk_hits"]).std())
        else:
            row["delta_topk_mean"] = np.nan
            row["delta_topk_std"] = np.nan

        rows.append(row)

    master = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
    return master


def save_master_table(master: pd.DataFrame, outdir: Path) -> None:
    master.to_csv(outdir / "combined_simulation_metrics.tsv", sep="\t", index=False)


def plot_auc_ap_improvement(master: pd.DataFrame, outdir: Path) -> None:
    informative_mask = master["scenario_id"].isin(["A", "B", "C", "D", "E"])
    df = master.loc[informative_mask].copy()

    x = np.arange(len(df))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, df["delta_auc_mean"], width=width, yerr=df["delta_auc_std"], capsize=4, label="ΔAUC")
    plt.bar(x + width/2, df["delta_ap_mean"], width=width, yerr=df["delta_ap_std"], capsize=4, label="ΔAP")
    plt.axhline(0, linewidth=1)
    plt.xticks(x, df["label"])
    plt.ylabel("Weighted - Baseline")
    plt.title("Improvement in discrimination metrics across simulation scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figure1_auc_ap_improvement.png", dpi=300)
    plt.close()


def plot_topk_improvement(master: pd.DataFrame, outdir: Path) -> None:
    informative_mask = master["scenario_id"].isin(["A", "B", "C", "D", "E"])
    df = master.loc[informative_mask].copy()

    x = np.arange(len(df))

    plt.figure(figsize=(9, 5))
    plt.bar(x, df["delta_topk_mean"], yerr=df["delta_topk_std"], capsize=4)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, df["label"])
    plt.ylabel("Weighted - Baseline top-k hits")
    plt.title("Gain in top-k causal protein recovery")
    plt.tight_layout()
    plt.savefig(outdir / "figure2_topk_improvement.png", dpi=300)
    plt.close()


def plot_baseline_vs_weighted_auc(master: pd.DataFrame, outdir: Path) -> None:
    informative_mask = master["scenario_id"].isin(["A", "B", "C", "D", "E"])
    df = master.loc[informative_mask].copy()

    x = np.arange(len(df))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, df["baseline_auc_mean"], width=width, yerr=df["baseline_auc_std"], capsize=4, label="Baseline")
    plt.bar(x + width/2, df["weighted_auc_mean"], width=width, yerr=df["weighted_auc_std"], capsize=4, label="Weighted")
    plt.xticks(x, df["label"])
    plt.ylabel("AUC")
    plt.ylim(0, min(1.0, max(df["weighted_auc_mean"].max(), df["baseline_auc_mean"].max()) + 0.05))
    plt.title("Baseline vs weighted AUC across simulation scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figure3_baseline_vs_weighted_auc.png", dpi=300)
    plt.close()


def plot_null_fpr(master: pd.DataFrame, outdir: Path) -> None:
    df = master.copy()
    x = np.arange(len(df))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, df["baseline_null_fpr_mean"], width=width, yerr=df["baseline_null_fpr_std"], capsize=4, label="Baseline")
    plt.bar(x + width/2, df["weighted_null_fpr_mean"], width=width, yerr=df["weighted_null_fpr_std"], capsize=4, label="Weighted")
    plt.xticks(x, df["label"])
    plt.ylabel("P(|Z| >= 5)")
    plt.title("Extreme-tail null/error behavior across simulation scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figure4_null_fpr.png", dpi=300)
    plt.close()


def plot_main_paper_panel(master: pd.DataFrame, outdir: Path) -> None:
    """
    One compact multi-panel figure, useful for slides/manuscript draft.
    """
    informative_mask = master["scenario_id"].isin(["A", "B", "C", "D", "E"])
    df = master.loc[informative_mask].copy()
    x = np.arange(len(df))
    width = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: ΔAUC
    axes[0].bar(x, df["delta_auc_mean"], yerr=df["delta_auc_std"], capsize=4)
    axes[0].axhline(0, linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["label"])
    axes[0].set_ylabel("Weighted - Baseline")
    axes[0].set_title("ΔAUC")

    # Panel 2: ΔTop-k
    axes[1].bar(x, df["delta_topk_mean"], yerr=df["delta_topk_std"], capsize=4)
    axes[1].axhline(0, linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["label"])
    axes[1].set_ylabel("Weighted - Baseline")
    axes[1].set_title("ΔTop-k hits")

    # Panel 3: null FPR in all 6 runs
    x2 = np.arange(len(master))
    axes[2].bar(x2 - width/2, master["baseline_null_fpr_mean"], width=width, label="Baseline")
    axes[2].bar(x2 + width/2, master["weighted_null_fpr_mean"], width=width, label="Weighted")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(master["label"])
    axes[2].set_ylabel("P(|Z| >= 5)")
    axes[2].set_title("Tail error / calibration")
    axes[2].legend()

    fig.suptitle("Circadian-informed PWAS simulation summary", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "figure0_main_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot 6-scenario circadian simulation results.")
    p.add_argument("--base_dir", type=str, default=".", help="Base directory containing run folders.")
    p.add_argument("--a", type=str, required=True, help="Run folder for scenario A.")
    p.add_argument("--b", type=str, required=True, help="Run folder for scenario B.")
    p.add_argument("--c", type=str, required=True, help="Run folder for scenario C.")
    p.add_argument("--d", type=str, required=True, help="Run folder for scenario D.")
    p.add_argument("--e", type=str, required=True, help="Run folder for scenario E.")
    p.add_argument("--f", type=str, required=True, help="Run folder for scenario F.")
    p.add_argument("--outdir", type=str, default="sim_plots", help="Output directory for figures.")
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

    master = build_master_table(run_map)
    save_master_table(master, outdir)

    plot_main_paper_panel(master, outdir)
    plot_auc_ap_improvement(master, outdir)
    plot_topk_improvement(master, outdir)
    plot_baseline_vs_weighted_auc(master, outdir)
    plot_null_fpr(master, outdir)

    print("\nSaved files:")
    for path in sorted(outdir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
