#!/usr/bin/env python3
"""
Merge per-formula ablation outputs from an array job into a single
combined CSV + summary + pivot table.

Usage:
    python merge_ablation_results.py ablation_arr_20260507_<jobid>
"""
from __future__ import annotations
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_ablation_results.py <PARENT_DIR>")
        sys.exit(1)
    parent = Path(sys.argv[1])
    if not parent.is_dir():
        print(f"Not a directory: {parent}")
        sys.exit(1)

    raw_files = sorted(parent.glob("*/ablation_raw.csv"))
    if not raw_files:
        print(f"No ablation_raw.csv found under {parent}/*/")
        sys.exit(1)

    print(f"[1/3] Found {len(raw_files)} per-formula files:")
    for f in raw_files:
        print(f"  {f}")

    raw = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
    out_raw = parent / "ablation_raw_combined.csv"
    raw.to_csv(out_raw, index=False)
    print(f"[2/3] Wrote combined raw: {out_raw}  ({len(raw)} rows)")

    rows = []
    for (af, tf, sc), grp in raw.groupby(["analysis_formula", "truth_formula", "scenario"]):
        n = int(grp["delta_auc"].notna().sum())
        sd_da = grp["delta_auc"].std(ddof=1) if n >= 2 else np.nan
        sd_dp = grp["delta_pr_auc"].std(ddof=1) if n >= 2 else np.nan
        rows.append({
            "analysis_formula": af, "truth_formula": tf, "scenario": sc, "n_reps": n,
            "ordinary_auc_mean": grp["ordinary_auc"].mean(),
            "weighted_auc_mean": grp["weighted_auc"].mean(),
            "delta_auc_mean": grp["delta_auc"].mean(),
            "delta_auc_se": (sd_da / math.sqrt(max(1, n))) if np.isfinite(sd_da) else np.nan,
            "delta_pr_auc_mean": grp["delta_pr_auc"].mean(),
            "delta_pr_auc_se": (sd_dp / math.sqrt(max(1, n))) if np.isfinite(sd_dp) else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values(["scenario", "analysis_formula"])
    out_sum = parent / "ablation_summary_combined.csv"
    summary.to_csv(out_sum, index=False)
    print(f"      Wrote summary: {out_sum}")

    # Pivot for the headline figure
    formula_order = ["baseline", "r_only", "a_only", "r_a", "r_vmf", "full"]
    present = [f for f in formula_order if f in summary["analysis_formula"].unique()]
    pivot = (summary.pivot_table(index="analysis_formula",
                                 columns="scenario",
                                 values="delta_auc_mean")
                    .reindex(present))
    out_pivot = parent / "ablation_delta_auc_pivot_combined.csv"
    pivot.to_csv(out_pivot)
    print(f"[3/3] Wrote pivot: {out_pivot}")
    print()
    print("=== ΔAUC (mean) — rows: analysis_formula, cols: scenario ===")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
