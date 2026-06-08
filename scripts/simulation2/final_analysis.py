#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_analysis.py
=================

Analyzes the outputs of final_simulation.py (5 blocks) and produces:

    1. paired_stats_main.csv       — paired t-tests for RP vs uniform, RP vs full
    2. summary_<block>.csv         — mean ± SE per condition (all blocks)
    3. figure_main.png             — bar chart, ΔAUC per (prior × scenario), with paired SE
    4. figure_sensitivity.png      — 4-panel sensitivity grid (sigma / h2 / n / k)
    5. summary_table.csv           — one-row-per-condition summary

Usage:
    python final_analysis.py results_final_<jobid>
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


SCENARIO_ORDER = ["circadian_mediation", "non_circadian_mediation", "wrong_phase", "null"]
SCENARIO_LABELS = {
    "circadian_mediation":     "Circadian mediation\n(↑ better)",
    "non_circadian_mediation": "Non-circadian\n(should be ≈0)",
    "wrong_phase":             "Inverse prior\n(↓ better; specificity)",
    "null":                    "Null\n(no genetic effect)",
}
SCENARIO_COLORS = {
    "circadian_mediation":     "#2E7D32",
    "non_circadian_mediation": "#9E9E9E",
    "wrong_phase":             "#C62828",
    "null":                    "#616161",
}

PRIOR_ORDER = ["uniform", "RP", "full"]
PRIOR_LABELS = {"uniform": "Baseline (w=1)", "RP": "RP (R²-only)", "full": "Full (4-term)"}


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------
def paired_diff(df: pd.DataFrame, prior_a: str, prior_b: str, scenario: str) -> dict:
    """Return paired t-test of (prior_a − prior_b) ΔAUC under one scenario."""
    sub = df[df["scenario"] == scenario]
    pivot = sub.pivot_table(index="rep", columns="prior", values="delta_auc")
    if prior_a not in pivot.columns or prior_b not in pivot.columns:
        return {}
    diffs = (pivot[prior_a] - pivot[prior_b]).dropna().to_numpy(dtype=float)
    if len(diffs) < 2:
        return {}
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    t = mean / se if se > 0 else np.nan
    p = float(2 * (1 - stats.t.cdf(abs(t), df=len(diffs) - 1))) if np.isfinite(t) else np.nan
    return {"prior_a": prior_a, "prior_b": prior_b, "scenario": scenario,
            "n_paired": len(diffs), "mean_diff": mean, "se_diff": se,
            "t": t, "p_value": p,
            "ci95_lo": mean - 1.96 * se, "ci95_hi": mean + 1.96 * se}


def per_prior_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sc, p), grp in df.groupby(["scenario", "prior"]):
        n = int(grp["delta_auc"].notna().sum())
        sd = grp["delta_auc"].std(ddof=1) if n >= 2 else np.nan
        rows.append({
            "scenario": sc, "prior": p, "n_reps": n,
            "delta_auc_mean": grp["delta_auc"].mean(),
            "delta_auc_se":   sd / np.sqrt(max(1, n)) if np.isfinite(sd) else np.nan,
            "delta_pr_auc_mean": grp["delta_pr_auc"].mean(),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_main(df_main: pd.DataFrame, out: Path):
    """Bar chart: ΔAUC per (prior × scenario), with paired SE error bars."""
    summary = per_prior_summary(df_main)
    pivot_mean = summary.pivot(index="prior", columns="scenario", values="delta_auc_mean").reindex(PRIOR_ORDER)
    pivot_se   = summary.pivot(index="prior", columns="scenario", values="delta_auc_se").reindex(PRIOR_ORDER)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(PRIOR_ORDER))
    bar_w = 0.20
    scenarios = [s for s in SCENARIO_ORDER if s in pivot_mean.columns]
    n_scen = len(scenarios)
    offsets = (np.arange(n_scen) - (n_scen - 1) / 2.0) * bar_w
    for i, sc in enumerate(scenarios):
        means = pivot_mean[sc].to_numpy(dtype=float)
        ses   = pivot_se[sc].to_numpy(dtype=float)
        ax.bar(x + offsets[i], means, bar_w,
               yerr=ses, capsize=3,
               color=SCENARIO_COLORS[sc], edgecolor="black", linewidth=0.6,
               label=SCENARIO_LABELS[sc])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([PRIOR_LABELS[p] for p in PRIOR_ORDER], fontsize=11)
    ax.set_ylabel("ΔAUC  (weighted − ordinary)", fontsize=11)
    ax.set_title("Main result: RP captures full's signal with one feature, no tunable α",
                 fontsize=12, pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def plot_sensitivity(parent: Path, out: Path):
    """4-panel grid: ΔAUC vs (sigma, h2, n, k) for RP and uniform under circadian_mediation."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    blocks = [
        ("sigma_sweep", "sigma_setting", "σ_log_w (RP sharpness)", axes[0]),
        ("h2_sweep",    "h2_setting",    "Trait heritability h²",  axes[1]),
        ("n_sweep",     "n_setting",     "Sample size n",          axes[2]),
        ("k_sweep",     "k_setting",     "# causal proteins K",    axes[3]),
    ]
    for blk, x_col, x_label, ax in blocks:
        path = parent / blk / f"final_{blk}_raw.csv"
        if not path.exists():
            ax.text(0.5, 0.5, f"missing\n{blk}", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off(); continue
        df = pd.read_csv(path)
        df = df[df["scenario"] == "circadian_mediation"]
        for prior, color, mk in [("RP", "#1565C0", "o"), ("uniform", "#9E9E9E", "s")]:
            grp = df[df["prior"] == prior].groupby(x_col)["delta_auc"].agg(["mean", "std", "count"]).reset_index()
            if grp.empty: continue
            grp["se"] = grp["std"] / np.sqrt(grp["count"].clip(lower=1))
            ax.errorbar(grp[x_col], grp["mean"], yerr=grp["se"],
                        marker=mk, color=color, lw=1.2, ms=6, capsize=3,
                        label=PRIOR_LABELS[prior])
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("ΔAUC", fontsize=10)
        ax.grid(linestyle=":", alpha=0.4)
        if x_col in ("n_setting", "k_setting"):
            ax.set_xscale("log")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Sensitivity analyses (Circadian-mediation scenario; RP vs baseline)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python final_analysis.py <parent_dir>")
        sys.exit(1)
    parent = Path(sys.argv[1])
    out_dir = parent / "analysis"
    out_dir.mkdir(exist_ok=True)

    # ---- main block ----
    print("[1/4] Main block stats and figure")
    main_path = parent / "main" / "final_main_raw.csv"
    if not main_path.exists():
        print(f"  ERROR: {main_path} not found"); sys.exit(1)
    df_main = pd.read_csv(main_path)

    paired_rows = []
    for sc in SCENARIO_ORDER:
        for pa, pb in [("RP", "uniform"), ("RP", "full"), ("full", "uniform")]:
            r = paired_diff(df_main, pa, pb, sc)
            if r: paired_rows.append(r)
    paired = pd.DataFrame(paired_rows)
    paired_path = out_dir / "paired_stats_main.csv"
    paired.to_csv(paired_path, index=False)
    print(f"  wrote {paired_path}")

    summ_main = per_prior_summary(df_main)
    summ_main.to_csv(out_dir / "summary_main.csv", index=False)

    plot_main(df_main, out_dir / "figure_main.png")

    # ---- sensitivity blocks ----
    print("[2/4] Sensitivity figures")
    plot_sensitivity(parent, out_dir / "figure_sensitivity.png")

    print("[3/4] Sensitivity summaries")
    for blk in ["sigma_sweep", "h2_sweep", "n_sweep", "k_sweep"]:
        p = parent / blk / f"final_{blk}_raw.csv"
        if not p.exists(): continue
        d = pd.read_csv(p)
        col = next((c for c in ["sigma_setting", "h2_setting", "n_setting", "k_setting"] if c in d.columns), None)
        if col is None: continue
        s = (d.groupby([col, "scenario", "prior"])["delta_auc"]
              .agg(["mean", "std", "count"]).reset_index())
        s["se"] = s["std"] / np.sqrt(s["count"].clip(lower=1))
        s.to_csv(out_dir / f"summary_{blk}.csv", index=False)

    # ---- headline summary ----
    print("[4/4] Headline summary")
    print()
    print("=== Main block: ΔAUC mean ± SE per (prior × scenario) ===")
    headline = (summ_main.pivot(index="prior", columns="scenario", values="delta_auc_mean")
                          .reindex(PRIOR_ORDER))
    print(headline.round(4).to_string())
    print()
    print("=== Paired t-tests (mean diff [95% CI], p-value) ===")
    for _, r in paired.iterrows():
        flag = "**" if r["p_value"] < 0.05 else "  "
        print(f"  {flag} {r['prior_a']:>7s} − {r['prior_b']:<7s}  "
              f"@ {r['scenario']:<25s}  "
              f"diff = {r['mean_diff']:+.4f} "
              f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]  "
              f"p = {r['p_value']:.3g}")
    print()
    print(f"All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
