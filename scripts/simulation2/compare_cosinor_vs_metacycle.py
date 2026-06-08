#!/usr/bin/env python3
"""
compare_cosinor_vs_metacycle.py
================================

Sanity check + supplementary figure:
  - read the cosinor annotation (R²) and the meta2d annotation (-log10 p)
  - report correlation and rank agreement
  - scatter plot
  - if both simulation main blocks exist, compare ΔAUC side by side

This is the figure you'll cite to demonstrate "the result is robust to
rhythm-detection algorithm choice" — a question every reviewer asks.

Usage:
  python compare_cosinor_vs_metacycle.py \\
      --cosinor-ann results_final_<jobid>/main/circadian_annotation_real.csv \\
      --meta2d-ann  raw_data/circadian_info_meta2d/circadian_annotation_meta2d.csv \\
      --cosinor-main results_final_<jobid>/main/final_main_summary.csv \\
      --meta2d-main  results_meta2d_<jobid>/main/final_main_summary.csv \\
      --out figure_metacycle_robustness.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosinor-ann", required=True, type=Path)
    ap.add_argument("--meta2d-ann",  required=True, type=Path)
    ap.add_argument("--cosinor-main", type=Path,
                    help="Optional: ΔAUC summary from cosinor run")
    ap.add_argument("--meta2d-main",  type=Path,
                    help="Optional: ΔAUC summary from meta2d run")
    ap.add_argument("--out", default="figure_metacycle_robustness.png", type=Path)
    args = ap.parse_args()

    # --- Load annotations ---
    cos = pd.read_csv(args.cosinor_ann)
    m2d = pd.read_csv(args.meta2d_ann)

    # Align by protein_id
    cos_score = "rhythmicity" if "rhythmicity" in cos.columns else "R2"
    m2d_score = "neglog10p"   if "neglog10p"   in m2d.columns else "rhythmicity"
    merged = cos[["protein_id", cos_score]].merge(
        m2d[["protein_id", m2d_score]], on="protein_id", how="inner",
        suffixes=("_cos", "_m2d"))
    print(f"[1/3] Merged annotations: {len(merged)} proteins")

    x = merged[cos_score].to_numpy(dtype=float)
    y = merged[m2d_score].to_numpy(dtype=float)
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    print(f"      Pearson  r = {r_p:.3f}  (p = {p_p:.2e})")
    print(f"      Spearman r = {r_s:.3f}  (p = {p_s:.2e})")

    # Decide layout
    have_main = (args.cosinor_main and args.cosinor_main.exists()
                 and args.meta2d_main and args.meta2d_main.exists())
    if have_main:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(figsize=(6.5, 5.2))
        ax2 = None

    # --- Panel A: scatter ---
    ax1.scatter(x, y, alpha=0.35, s=14, edgecolor="none", color="#1565C0")
    ax1.set_xlabel(f"Cosinor R²  (annotation: {cos_score})", fontsize=11)
    ax1.set_ylabel(f"MetaCycle −log10(p)  ({m2d_score})", fontsize=11)
    ax1.set_title(f"A. Annotation agreement\nPearson r = {r_p:.3f},  "
                   f"Spearman ρ = {r_s:.3f}  (n = {len(merged)})",
                   fontsize=11.5, pad=8)
    ax1.grid(linestyle=":", alpha=0.4); ax1.set_axisbelow(True)
    # Add diagonal-ish trend line via rank
    xs = np.argsort(x)
    if len(xs) > 50:
        # bin into quartiles and connect medians
        bins = np.array_split(xs, 20)
        bx = [np.median(x[b]) for b in bins]
        by = [np.median(y[b]) for b in bins]
        ax1.plot(bx, by, color="#C62828", lw=1.6, label="binned medians")
        ax1.legend(fontsize=9)

    # --- Panel B: ΔAUC comparison (if data available) ---
    if ax2 is not None:
        cs = pd.read_csv(args.cosinor_main)
        ms = pd.read_csv(args.meta2d_main)
        scenarios = ["circadian_mediation", "non_circadian_mediation", "wrong_phase"]
        priors_to_show = ["RP"]
        rows = []
        for src, df in [("cosinor", cs), ("MetaCycle", ms)]:
            for sc in scenarios:
                for p in priors_to_show:
                    sub = df[(df["scenario"] == sc) & (df["prior"] == p)]
                    if not sub.empty:
                        r = sub.iloc[0]
                        rows.append({
                            "source": src, "scenario": sc, "prior": p,
                            "mean": r["mean_delta_auc"], "se": r["se_delta_auc"]
                        })
        cmp_df = pd.DataFrame(rows)
        if cmp_df.empty:
            ax2.text(0.5, 0.5, "no comparable\nsimulation outputs",
                     transform=ax2.transAxes, ha="center", va="center",
                     fontsize=11, color="grey")
            ax2.set_axis_off()
        else:
            x_pos = np.arange(len(scenarios))
            bw = 0.32
            for i, src in enumerate(["cosinor", "MetaCycle"]):
                vals = [cmp_df[(cmp_df["source"]==src) & (cmp_df["scenario"]==sc)]["mean"].values[0]
                        if not cmp_df[(cmp_df["source"]==src) & (cmp_df["scenario"]==sc)].empty else 0
                        for sc in scenarios]
                ses  = [cmp_df[(cmp_df["source"]==src) & (cmp_df["scenario"]==sc)]["se"].values[0]
                        if not cmp_df[(cmp_df["source"]==src) & (cmp_df["scenario"]==sc)].empty else 0
                        for sc in scenarios]
                color = "#1565C0" if src == "cosinor" else "#EF6C00"
                ax2.bar(x_pos + (i - 0.5) * bw, vals, bw, yerr=ses, capsize=4,
                        color=color, edgecolor="black", lw=0.5, label=src)
            ax2.axhline(0, color="black", lw=0.6)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(["Circadian\nmediation", "Non-\ncircadian",
                                 "Inverse\nprior"], fontsize=10)
            ax2.set_ylabel("ΔAUC  (RP prior)", fontsize=11)
            ax2.set_title("B. RP ΔAUC: cosinor vs MetaCycle annotation",
                          fontsize=11.5, pad=8)
            ax2.legend(loc="best", fontsize=10)
            ax2.grid(axis="y", linestyle=":", alpha=0.4); ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[2/3] Wrote: {args.out}")
    print(f"[3/3] Done.")


if __name__ == "__main__":
    main()
