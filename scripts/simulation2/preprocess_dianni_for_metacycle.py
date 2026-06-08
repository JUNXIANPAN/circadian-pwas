#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_dianni_for_metacycle.py
==================================

Parses a DIA-NN pg_matrix.tsv where sample columns are full mzML paths of the
form .../<SubjectID>-Dag<Day>-t<Hour>.mzML and emits a clean matrix ready for
MetaCycle (or any rhythm-detection tool that takes one-row-per-protein,
one-column-per-timepoint).

Output schema (TSV):
    protein_id   t9   t12   t15   t18   t21   t33   t36   t39   t42   t45
    P00450       ...  ...   ...   ...   ...   ...   ...   ...   ...   ...
    P02768       ...  ...   ...   ...   ...   ...   ...   ...   ...   ...

Columns are sorted by (day - 1) * 24 + timepoint. Subjects can be either
averaged (default; one value per timepoint) or kept as replicates (multiple
columns at the same timepoint — MetaCycle handles this).

Usage:
    python preprocess_dianni_for_metacycle.py \\
        --input  raw_data/circadian_info/report.pg_matrix.tsv \\
        --output raw_data/circadian_info/pg_matrix_clean.tsv \\
        --mode   average                          # or 'replicate'

After run, it prints the timepoints string to pass to MetaCycle, e.g.:
    Pass to R:  --timepoints "9,12,15,18,21,33,36,39,42,45"
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_PATTERN = re.compile(
    r"(?P<subject>\d+)\s*-\s*Dag(?P<day>\d+)\s*-\s*t(?P<hour>\d+)",
    re.IGNORECASE
)


def parse_sample_column(colname: str):
    """Extract (subject_id, day, hour) from a sample column name like
    .../Plate1_10582-Dag1-t9.mzML  →  ('10582', 1, 9)
    Returns None if pattern doesn't match (so we can skip metadata columns).
    """
    m = SAMPLE_PATTERN.search(colname)
    if m is None:
        return None
    return m.group("subject"), int(m.group("day")), int(m.group("hour"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to DIA-NN report.pg_matrix.tsv")
    ap.add_argument("--output", required=True, type=Path,
                    help="Path for the cleaned matrix")
    ap.add_argument("--id-col", default="Protein.Group",
                    help="Protein ID column (default: Protein.Group)")
    ap.add_argument("--mode", default="average",
                    choices=["average", "replicate"],
                    help="average: mean per timepoint  |  replicate: keep subjects as separate columns")
    ap.add_argument("--log-transform", action="store_true",
                    help="Apply log2(x+1) before averaging (recommended for MS intensities)")
    ap.add_argument("--strict-na", action="store_true",
                    help="Drop proteins with ANY missing values (required for MetaCycle LS/JTK)")
    ap.add_argument("--min-var", type=float, default=1e-6,
                    help="Drop proteins with timepoint variance <= this value (default 1e-6)")
    args = ap.parse_args()

    print(f"[1/4] Reading {args.input}")
    df = pd.read_csv(args.input, sep="\t", low_memory=False)
    print(f"      rows = {len(df)}, cols = {df.shape[1]}")

    if args.id_col not in df.columns:
        raise SystemExit(f"ID col '{args.id_col}' not found. Available: {list(df.columns[:8])}")

    # Identify sample columns (those matching the SAMPLE_PATTERN)
    sample_info = {}
    for col in df.columns:
        parsed = parse_sample_column(col)
        if parsed is not None:
            subj, day, hr = parsed
            total_t = (day - 1) * 24 + hr
            sample_info[col] = {"subject": subj, "day": day, "hour": hr, "total_t": total_t}

    if not sample_info:
        raise SystemExit("No sample columns matched pattern <SubjectID>-Dag<Day>-t<Hour>. "
                         "Inspect column names and adjust the regex if needed.")
    print(f"[2/4] Parsed {len(sample_info)} sample columns")

    info_df = pd.DataFrame(sample_info).T
    n_subj = info_df["subject"].nunique()
    n_days = info_df["day"].nunique()
    timepoints_per_day = sorted(info_df["hour"].unique().tolist())
    total_times = sorted(info_df["total_t"].unique().tolist())
    print(f"      subjects = {n_subj}, days = {n_days}, hours-per-day = {timepoints_per_day}")
    print(f"      unique total times = {total_times}")

    # Build wide matrix
    sample_cols = list(sample_info.keys())
    mat = df[[args.id_col] + sample_cols].copy()
    mat.rename(columns={args.id_col: "protein_id"}, inplace=True)

    # Cast samples to float; non-numeric → NaN
    for c in sample_cols:
        mat[c] = pd.to_numeric(mat[c], errors="coerce")

    if args.log_transform:
        print(f"      applying log2(x+1) transform")
        mat[sample_cols] = np.log2(mat[sample_cols].clip(lower=0) + 1.0)

    print(f"[3/4] Building output in '{args.mode}' mode")
    if args.mode == "average":
        # Group sample columns by total_t and average across subjects
        out = pd.DataFrame({"protein_id": mat["protein_id"]})
        for t in total_times:
            cols_at_t = [c for c, info in sample_info.items() if info["total_t"] == t]
            colname = f"t{t}"
            out[colname] = mat[cols_at_t].mean(axis=1, skipna=True)
        timepoint_string = ",".join(str(t) for t in total_times)
        n_out_cols = len(total_times)
    else:  # replicate
        # Keep all subject columns, but rename to t<total_t>_<subject> and sort by time
        sorted_cols = sorted(sample_cols,
                             key=lambda c: (sample_info[c]["total_t"], sample_info[c]["subject"]))
        new_names = {c: f"t{sample_info[c]['total_t']}_s{sample_info[c]['subject']}"
                     for c in sorted_cols}
        out = mat[["protein_id"] + sorted_cols].rename(columns=new_names)
        # Timepoint string has duplicates — one entry per sample column
        timepoint_string = ",".join(str(sample_info[c]["total_t"]) for c in sorted_cols)
        n_out_cols = len(sorted_cols)

    # Filter proteins based on data quality
    n_before = len(out)
    data_cols = [c for c in out.columns if c != "protein_id"]

    # Drop all-NaN proteins
    out = out.dropna(subset=data_cols, how="all")
    n_after_all_na = len(out)

    if args.strict_na:
        # Drop proteins with ANY NaN (required for MetaCycle LS/JTK methods)
        out = out.dropna(subset=data_cols, how="any")
        n_after_any_na = len(out)
        print(f"      strict NA filter: {n_after_all_na} → {n_after_any_na} (dropped any-NaN)")

    # Drop near-zero variance proteins (LOESS inside MetaCycle LS fails on them)
    if args.min_var > 0:
        var_per_protein = out[data_cols].var(axis=1, skipna=True).to_numpy()
        keep = var_per_protein > args.min_var
        n_dropped_lowvar = (~keep).sum()
        out = out[keep]
        print(f"      low-variance filter: dropped {n_dropped_lowvar} (var ≤ {args.min_var})")

    print(f"      proteins kept: {len(out)} / {n_before} ({len(out)/n_before*100:.1f}%)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    print(f"[4/4] Wrote {args.output}  ({len(out)} rows × {n_out_cols} sample cols)")

    print()
    print("=" * 70)
    print("Pass this exact string to MetaCycle:")
    print(f'  --timepoints "{timepoint_string}"')
    print("=" * 70)


if __name__ == "__main__":
    main()
