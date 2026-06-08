
"""
plot_circadian_simulation_results.py

Make clean presentation-ready figures from circadian_simulation.py outputs.

Expected inputs
---------------
Each simulation output directory should contain:
- simulation_summary_<scenario>.tsv

Example:
python plot_circadian_simulation_results.py \
  --inputs \
    "sim_A_strong_informative/simulation_summary_informative.tsv" \
    "sim_B_moderate_informative/simulation_summary_informative.tsv" \
    "sim_C_weak_informative/simulation_summary_informative.tsv" \
    "sim_D_moderate_noninformative/simulation_summary_noninformative.tsv" \
    "sim_F_global_null/simulation_summary_global_null.tsv" \
  --labels \
    "Strong informative" \
    "Moderate informative" \
    "Weak informative" \
    "Noninformative" \
    "Global null" \
  --outdir simulation_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_summary(path: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path, sep="\t")
    required = {"metric", "mean", "std"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} does not have columns: {required}")
    out = {}
    for _, row in df.iterrows():
        out[str(row["metric"])] = {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }
    return out


def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def metric_delta(summary: Dict[str, Dict[str, float]], base_metric: str, weighted_metric: str):
    base_mean = summary[base_metric]["mean"]
    weighted_mean = summary[weighted_metric]["mean"]
    base_std = summary[base_metric]["std"]
    weighted_std = summary[weighted_metric]["std"]
    delta_mean = weighted_mean - base_mean
    delta_err = np.sqrt(base_std**2 + weighted_std**2)
    return delta_mean, delta_err


def plot_delta_bars(
    labels: List[str],
    deltas: List[float],
    errs: List[float],
    title: str,
    ylabel: str,
    outfile_png: Path,
    outfile_pdf: Path,
):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, yerr=errs, capsize=4, width=0.68)

    ax.axhline(0, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin = min(min(deltas), 0)
    ymax = max(max(deltas), 0)
    yrange = ymax - ymin
    if yrange == 0:
        yrange = 0.02
    ax.set_ylim(ymin - 0.15 * yrange - 0.002, ymax + 0.2 * yrange + 0.002)

    for rect, value in zip(bars, deltas):
        y = rect.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y + offset if y >= 0 else y - offset,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(outfile_png, dpi=300, bbox_inches="tight")
    fig.savefig(outfile_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_baseline_weighted(
    labels: List[str],
    baseline_vals: List[float],
    weighted_vals: List[float],
    title: str,
    ylabel: str,
    outfile_png: Path,
    outfile_pdf: Path,
):
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    x = np.arange(len(labels))
    width = 0.36

    b1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline")
    b2 = ax.bar(x + width / 2, weighted_vals, width, label="Weighted")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin = min(baseline_vals + weighted_vals)
    ymax = max(baseline_vals + weighted_vals)
    yrange = ymax - ymin
    if yrange == 0:
        yrange = 0.02
    ax.set_ylim(max(0, ymin - 0.12 * yrange), ymax + 0.18 * yrange)

    for bars in [b1, b2]:
        for rect in bars:
            value = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                value + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"{value:.3f}" if value < 1 else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(outfile_png, dpi=300, bbox_inches="tight")
    fig.savefig(outfile_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_fpr_panel(
    labels: List[str],
    baseline_vals: List[float],
    weighted_vals: List[float],
    outfile_png: Path,
    outfile_pdf: Path,
):
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    x = np.arange(len(labels))
    width = 0.36

    b1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline")
    b2 = ax.bar(x + width / 2, weighted_vals, width, label="Weighted")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Extreme-value rate: |Z| ≥ 5")
    ax.set_title("Null calibration proxy across simulation settings", pad=14)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = max(baseline_vals + weighted_vals + [1e-5])
    ax.set_ylim(0, ymax * 1.35)

    for bars in [b1, b2]:
        for rect in bars:
            value = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                value + 0.02 * ax.get_ylim()[1],
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(outfile_png, dpi=300, bbox_inches="tight")
    fig.savefig(outfile_pdf, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(labels: List[str], summaries: List[Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for label, s in zip(labels, summaries):
        auc_delta, _ = metric_delta(s, "baseline_auc", "weighted_auc")
        ap_delta, _ = metric_delta(s, "baseline_ap", "weighted_ap")
        topk_delta, _ = metric_delta(s, "baseline_topk_hits", "weighted_topk_hits")
        rows.append(
            {
                "Scenario": label,
                "Baseline AUC": s["baseline_auc"]["mean"],
                "Weighted AUC": s["weighted_auc"]["mean"],
                "Delta AUC": auc_delta,
                "Baseline AP": s["baseline_ap"]["mean"],
                "Weighted AP": s["weighted_ap"]["mean"],
                "Delta AP": ap_delta,
                "Baseline Top-k": s["baseline_topk_hits"]["mean"],
                "Weighted Top-k": s["weighted_topk_hits"]["mean"],
                "Delta Top-k": topk_delta,
                "Baseline |Z|>=5": s["baseline_null_fpr_absz_ge_5"]["mean"],
                "Weighted |Z|>=5": s["weighted_null_fpr_absz_ge_5"]["mean"],
            }
        )
    return pd.DataFrame(rows)


def build_parser():
    p = argparse.ArgumentParser(description="Plot circadian simulation summaries.")
    p.add_argument("--inputs", nargs="+", required=True, help="Summary TSV files.")
    p.add_argument("--labels", nargs="+", required=True, help="Display labels for each summary.")
    p.add_argument("--outdir", type=str, default="simulation_figures")
    return p


def main():
    args = build_parser().parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length.")

    outdir = ensure_outdir(args.outdir)
    summaries = [load_summary(p) for p in args.inputs]
    labels = args.labels

    auc_deltas, auc_errs = zip(*[
        metric_delta(s, "baseline_auc", "weighted_auc") for s in summaries
    ])
    plot_delta_bars(
        labels=labels,
        deltas=list(auc_deltas),
        errs=list(auc_errs),
        title="Improvement in AUC after circadian-informed reweighting",
        ylabel="Delta AUC (Weighted - Baseline)",
        outfile_png=outdir / "delta_auc.png",
        outfile_pdf=outdir / "delta_auc.pdf",
    )

    ap_deltas, ap_errs = zip(*[
        metric_delta(s, "baseline_ap", "weighted_ap") for s in summaries
    ])
    plot_delta_bars(
        labels=labels,
        deltas=list(ap_deltas),
        errs=list(ap_errs),
        title="Improvement in average precision after circadian-informed reweighting",
        ylabel="Delta AP (Weighted - Baseline)",
        outfile_png=outdir / "delta_ap.png",
        outfile_pdf=outdir / "delta_ap.pdf",
    )

    topk_deltas, topk_errs = zip(*[
        metric_delta(s, "baseline_topk_hits", "weighted_topk_hits") for s in summaries
    ])
    plot_delta_bars(
        labels=labels,
        deltas=list(topk_deltas),
        errs=list(topk_errs),
        title="Recovery gain among top-ranked proteins",
        ylabel="Delta Top-k hits (Weighted - Baseline)",
        outfile_png=outdir / "delta_topk.png",
        outfile_pdf=outdir / "delta_topk.pdf",
    )

    plot_grouped_baseline_weighted(
        labels=labels,
        baseline_vals=[s["baseline_ap"]["mean"] for s in summaries],
        weighted_vals=[s["weighted_ap"]["mean"] for s in summaries],
        title="Baseline vs weighted average precision across scenarios",
        ylabel="Average precision",
        outfile_png=outdir / "baseline_vs_weighted_ap.png",
        outfile_pdf=outdir / "baseline_vs_weighted_ap.pdf",
    )

    plot_fpr_panel(
        labels=labels,
        baseline_vals=[s["baseline_null_fpr_absz_ge_5"]["mean"] for s in summaries],
        weighted_vals=[s["weighted_null_fpr_absz_ge_5"]["mean"] for s in summaries],
        outfile_png=outdir / "null_calibration_proxy.png",
        outfile_pdf=outdir / "null_calibration_proxy.pdf",
    )

    summary_table = build_summary_table(labels, summaries)
    summary_table.to_csv(outdir / "simulation_summary_table.tsv", sep="\t", index=False)

    print(f"Saved figures and summary table to: {outdir}")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
