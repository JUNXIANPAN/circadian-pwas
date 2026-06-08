#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_simulation_v2.py
======================

UPDATED VERSION integrating all revisions from the project trajectory discussion.

KEY CHANGES VS final_simulation.py:
------------------------------------
1. max_ld_snps default: 500 → 1500
   Reduces artificial protein-protein correlation from SNP reuse.
   At 1500 proteins × 5 top SNPs / 1500 SNPs, average reuse ≈ 5x (vs previous 15x).

2. NEW SCENARIO: `lipid_gwas_truth`
   Anti-circularity level 2 — truth from EXTERNAL lipid GWAS (GLGC/Graham 2021),
   completely independent of rhythmicity. Addresses the "Mona Lisa concern"
   that current anti-circularity (log-noise on shared latent_w) is only level-1.

3. NEW BLOCK: `lipid_truth`
   Runs lipid_gwas_truth scenario on the REAL annotation pool (no bootstrap),
   since lipid GWAS gene matching requires real UniProt/gene names.

UNCHANGED FROM v1:
------------------
- RP = exp(τ·R²) with auto-calibrated τ = σ_log_w / std(R), σ=0.5
- Paired-replicate design (same seed across priors per rep)
- Truth track uses FULL latent_w (rich biology assumption)
- Three priors: uniform / RP / full

USAGE:
------
    # Original 4 scenarios with new SNP count
    python final_simulation_v2.py --block main --n-reps 200 \\
        --outdir results/main_v2 \\
        --pg-matrix raw_data/circadian_info/report.pg_matrix.tsv \\
        --pqtl work/pqtl_topk.csv \\
        --ld-dir /data/CommonData/ukbb-ld/

    # NEW: lipid GWAS-truth scenario (strict anti-circularity)
    python final_simulation_v2.py --block lipid_truth --n-reps 200 \\
        --outdir results/lipid_truth \\
        --pg-matrix raw_data/circadian_info/report.pg_matrix.tsv \\
        --pqtl work/pqtl_topk.csv \\
        --ld-dir /data/CommonData/ukbb-ld/
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from circadian_pwas_simulation_phase_causal import (
    WeightParams, SimParams,
    load_or_fit_circadian_annotation, load_pqtl_topk, list_ld_files,
    bootstrap_synthetic_universe, simulate_genotype_from_ld, load_ld_matrix,
    construct_protein_matrix, select_causal_proteins,
    simulate_phenotype_from_proteins, compute_pwas_z, ranking_metrics,
    compute_observed_weight, compute_strict_latent_weight,
    normalize_01, zscore_safe, mkdir, log,
)


# =============================================================================
# LIPID GWAS TRUTH SET (anti-circularity level 2)
# =============================================================================
# Source: GLGC / Graham 2021 (Nature) and lipid GWAS Nat Commun.
# These are plasma-detectable proteins whose CAUSAL relation to lipid traits
# was established by GWAS — completely independent of rhythmicity annotation.
#
# Pre-validated against the project's pg_matrix: 20 of these match
# (APOA1/2/4, APOB, APOC1/2/3, APOE, APOM, APOH, CETP, LCAT, PLTP, LPA,
#  PCSK9, ANGPTL3, PON1, SAA1, SAA2, TOMM40).
LIPID_GWAS_GENES = {
    # Apolipoproteins (plasma high-abundance, lipid GWAS core hits)
    "APOA1", "APOA2", "APOA4", "APOA5", "APOB", "APOC1", "APOC2", "APOC3",
    "APOE", "APOM", "APOH",
    # Lipid transport / remodeling enzymes (partially secreted)
    "CETP", "LCAT", "PLTP", "LPL", "LIPC", "LIPG", "LPA",
    # Secreted regulators (drug targets)
    "PCSK9", "ANGPTL3", "ANGPTL4", "PON1", "SAA1", "SAA2",
    # Linked loci frequently appearing in lipid GWAS
    "TOMM40", "ZPR1", "GCKR",
}


# =============================================================================
# THE FINAL FORMULA (RP) — unchanged from v1
# =============================================================================
def rhythmicity_prior(R2: np.ndarray, sigma_log_w: float = 0.5) -> Tuple[np.ndarray, float]:
    """RP weights: w_j = exp(τ·R²) / mean, with auto-calibrated τ = σ/std(R)."""
    R = normalize_01(np.asarray(R2, dtype=float))
    s = float(np.std(R))
    if s < 1e-9:
        return np.ones_like(R), 0.0
    tau = sigma_log_w / s
    w_star = np.exp(tau * R)
    return w_star / np.mean(w_star), tau


def uniform_prior(N: int) -> Tuple[np.ndarray, float]:
    return np.ones(N), 0.0


def full_prior(ann: pd.DataFrame, params: WeightParams) -> Tuple[np.ndarray, float]:
    """4-term S formula — used both as truth source and as upper-bound ablation."""
    out = compute_strict_latent_weight(ann, params)
    return out["latent_w"].to_numpy(dtype=float), float(params.tau)


# =============================================================================
# NEW: lipid GWAS truth set lookup
# =============================================================================
def find_lipid_gwas_causal_idx(ann: pd.DataFrame) -> np.ndarray:
    """Return positional indices of lipid GWAS genes within `ann`.

    Looks for a gene name column (one of 'gene', 'Genes', 'gene_symbol').
    Splits on ';' and ',' to handle protein groups mapping to multiple genes.
    """
    # Find gene column (be flexible about column naming)
    gene_col = None
    for c in ("gene", "Genes", "gene_symbol", "GeneName"):
        if c in ann.columns:
            gene_col = c
            break
    if gene_col is None:
        raise ValueError(
            "Cannot find gene name column in ann. "
            "Expected one of: gene, Genes, gene_symbol, GeneName."
        )

    causal_pos = []
    for i, raw in enumerate(ann[gene_col].astype(str).tolist()):
        genes = [g.strip().upper() for g in raw.replace(",", ";").split(";")]
        if any(g in LIPID_GWAS_GENES for g in genes):
            causal_pos.append(i)
    return np.array(causal_pos, dtype=int)


# =============================================================================
# Per-replicate engine — adds lipid_gwas_truth scenario
# =============================================================================
def run_paired_replicate(
    rep: int,
    ld_file: Path,
    ann: pd.DataFrame,
    pqtl: pd.DataFrame,
    weight_params: WeightParams,
    sim_params: SimParams,
    sigma_log_w: float,
    seed: int,
    priors_to_run: Sequence[str],
    scenarios: Sequence[str],
    extra_meta: Dict,
    lipid_causal_idx: np.ndarray = None,
) -> List[dict]:
    """One paired replicate across all priors and all scenarios.

    Truth track uses FULL latent_w (level-1 anti-circularity unchanged).
    NEW: if scenario == 'lipid_gwas_truth', causal set is fixed to lipid_causal_idx
    (truth from external lipid GWAS, anti-circularity level 2).
    """
    rng = np.random.default_rng(seed + rep * 100003)
    ld = load_ld_matrix(ld_file, sim_params.max_ld_snps)
    X = simulate_genotype_from_ld(ld, sim_params.n_individuals, rng)
    proteins = ann["protein_id"].tolist()
    Gmat = construct_protein_matrix(X, pqtl, proteins, rng, sim_params.protein_noise_sd)

    # ---- truth track: latent_w from FULL ----
    ann_truth = compute_strict_latent_weight(ann, weight_params)
    latent_w_truth = ann_truth["latent_w"].to_numpy(dtype=float)
    R2 = ann["rhythmicity"].to_numpy(dtype=float)

    # ---- precompute every analysis prior ONCE ----
    weights: Dict[str, np.ndarray] = {}
    taus: Dict[str, float] = {}
    if "uniform" in priors_to_run:
        w, t = uniform_prior(len(R2))
        weights["uniform"], taus["uniform"] = w, t
    if "RP" in priors_to_run:
        w, t = rhythmicity_prior(R2, sigma_log_w=sigma_log_w)
        w_obs = compute_observed_weight(w, weight_params, rng)
        weights["RP"], taus["RP"] = w_obs, t
    if "full" in priors_to_run:
        w, t = full_prior(ann, weight_params)
        w_obs = compute_observed_weight(w, weight_params, rng)
        weights["full"], taus["full"] = w_obs, t

    inverse = {p: (1.0 / np.maximum(w, 1e-9)) for p, w in weights.items()}
    for p in inverse:
        inverse[p] = inverse[p] / np.mean(inverse[p])

    # ---- run scenarios ----
    rows = []
    for scenario in scenarios:
        if scenario == "circadian_mediation":
            causal_idx = select_causal_proteins(
                latent_w_truth, sim_params.n_causal,
                sim_params.causal_enrichment_gamma, rng, "circadian")
            y, gamma = simulate_phenotype_from_proteins(
                Gmat, causal_idx, sim_params.trait_h2, rng)
            use_inverse = False

        elif scenario == "non_circadian_mediation":
            causal_idx = select_causal_proteins(
                latent_w_truth, sim_params.n_causal,
                sim_params.causal_enrichment_gamma, rng, "uniform")
            y, gamma = simulate_phenotype_from_proteins(
                Gmat, causal_idx, sim_params.trait_h2, rng)
            use_inverse = False

        elif scenario == "wrong_phase":
            causal_idx = select_causal_proteins(
                latent_w_truth, sim_params.n_causal,
                sim_params.causal_enrichment_gamma, rng, "circadian")
            y, gamma = simulate_phenotype_from_proteins(
                Gmat, causal_idx, sim_params.trait_h2, rng)
            use_inverse = True

        elif scenario == "null":
            # NULL: y 纯噪声，但保留 truth 标签用于 AUC 计算
            # 测的是: 当性状和蛋白无关时, 加权是否会产生 false positive inflation
            causal_idx = rng.choice(len(proteins), size=sim_params.n_causal, replace=False)
            gamma = np.zeros(len(proteins))
            gamma[causal_idx] = 1.0   # 仅用于标记 truth, 不影响 y
            y = zscore_safe(rng.normal(size=sim_params.n_individuals))
            use_inverse = False

        elif scenario == "lipid_gwas_truth":
            # NEW: truth from external lipid GWAS, NOT from rhythmicity
            if lipid_causal_idx is None or len(lipid_causal_idx) == 0:
                log(f"     [skip] lipid_gwas_truth: no overlap in protein pool")
                continue
            causal_idx = lipid_causal_idx       # FIXED across reps for this scenario
            y, gamma = simulate_phenotype_from_proteins(
                Gmat, causal_idx, sim_params.trait_h2, rng)
            use_inverse = False

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        z = compute_pwas_z(Gmat, y)
        ord_score = np.abs(z)
        truth = (gamma != 0).astype(int)
        om = ranking_metrics(truth, ord_score, sim_params.top_ks)

        for prior_name in priors_to_run:
            w_use = inverse[prior_name] if use_inverse else weights[prior_name]
            wtd_score = ord_score * np.sqrt(np.maximum(w_use, 0))
            wm = ranking_metrics(truth, wtd_score, sim_params.top_ks)
            row = {
                "rep": rep, "ld_file": ld_file.name,
                "scenario": scenario, "prior": prior_name,
                "n_causal": int(truth.sum()),
                "sigma_log_w": sigma_log_w if prior_name == "RP" else np.nan,
                "tau": taus[prior_name],
                "ordinary_auc": om["auc"], "weighted_auc": wm["auc"],
                "delta_auc": (wm["auc"] - om["auc"])
                    if np.isfinite(om["auc"]) and np.isfinite(wm["auc"]) else np.nan,
                "ordinary_pr_auc": om["pr_auc"], "weighted_pr_auc": wm["pr_auc"],
                "delta_pr_auc": (wm["pr_auc"] - om["pr_auc"])
                    if np.isfinite(om["pr_auc"]) and np.isfinite(wm["pr_auc"]) else np.nan,
            }
            row.update(extra_meta)
            rows.append(row)
    return rows


# =============================================================================
# Experiment blocks
# =============================================================================
def run_block(args, sim_params_override: Dict = None, extra_meta: Dict = None,
              priors=("uniform", "RP", "full"),
              scenarios=("circadian_mediation", "non_circadian_mediation",
                         "wrong_phase", "null"),
              sigma_log_w: float = None,
              n_reps: int = None, suffix: str = "",
              use_real_only: bool = False) -> pd.DataFrame:
    """Generic block runner.

    use_real_only=True: skip bootstrap, use real annotation pool only.
        Required for lipid_gwas_truth scenario (gene names needed).
    """
    log(f"-- block {args.block}{suffix}: priors={priors}, scenarios={scenarios}, "
        f"sigma={sigma_log_w}, n_reps={n_reps or args.n_reps}, "
        f"use_real_only={use_real_only}, override={sim_params_override}")

    rng_seed = args.seed
    rng = np.random.default_rng(rng_seed)
    ann_real = load_or_fit_circadian_annotation(args.pg_matrix, args.outdir)
    pqtl_real = load_pqtl_topk(args.pqtl, args.outdir)
    ld_files = list_ld_files(args.ld_dir)

    if use_real_only:
        ann_used = ann_real
        # Filter pQTL to proteins present in real annotation
        pids = set(ann_real["protein_id"].tolist())
        pqtl_used = pqtl_real[pqtl_real["protein_id"].isin(pids)].copy()
        n_proteins_used = len(ann_used)
        
        if "gene" not in ann_used.columns and "Genes" not in ann_used.columns:
                log(f"     gene column missing from ann, re-loading from {args.pg_matrix}")
                gene_df = pd.read_csv(args.pg_matrix, usecols=["protein_id", "gene"])
                gene_df["protein_id"] = gene_df["protein_id"].astype(str)
                ann_used = ann_used.copy()
                ann_used["protein_id"] = ann_used["protein_id"].astype(str)
                ann_used = ann_used.merge(gene_df, on="protein_id", how="left")
                log(f"     merged: {ann_used['gene'].notna().sum()} / {len(ann_used)} proteins got gene names")

        print(f"DEBUG ann_used columns: {ann_used.columns.tolist()}")
        lipid_causal_idx = find_lipid_gwas_causal_idx(ann_used)
        log(f"     real-only pool: {n_proteins_used} proteins, "
            f"{len(lipid_causal_idx)} lipid-GWAS matches")
    else:
        ann_used, pqtl_used = bootstrap_synthetic_universe(
            ann_real, pqtl_real, args.n_proteins, rng)
        n_proteins_used = args.n_proteins
        lipid_causal_idx = None

    weight_params = WeightParams(
        args.alpha_r, args.alpha_a, args.alpha_p, args.alpha_c, args.tau,
        args.target_phase_hour, args.main_noise_sigma, args.main_shrink_to_one,
    )
    sp = SimParams(
        n_individuals=args.n_individuals, n_proteins=n_proteins_used,
        n_causal=args.n_causal, trait_h2=args.trait_h2,
        causal_enrichment_gamma=args.main_gamma,
        protein_noise_sd=args.protein_noise_sd, top_ks=(10, 20, 50),
        max_ld_snps=args.max_ld_snps,         # NEW DEFAULT: 1500
        causal_prior_mode="full",
    )
    if sim_params_override:
        sp = replace(sp, **sim_params_override)

    nreps = n_reps or args.n_reps
    sigma = sigma_log_w if sigma_log_w is not None else args.sigma_log_w
    extra = extra_meta or {}

    all_rows: List[dict] = []
    for r in range(1, nreps + 1):
        ld_file = ld_files[(r - 1) % len(ld_files)]
        rows = run_paired_replicate(
            r, ld_file, ann_used, pqtl_used, weight_params, sp, sigma,
            rng_seed, priors, scenarios, extra,
            lipid_causal_idx=lipid_causal_idx,
        )
        all_rows.extend(rows)
        if r == 1 or r % max(1, nreps // 5) == 0 or r == nreps:
            log(f"     rep {r}/{nreps}")

    return pd.DataFrame(all_rows)


def block_main(args) -> pd.DataFrame:
    """Main: 3 priors × 4 scenarios × N reps, paired (bootstrap pool)."""
    return run_block(args)


def block_lipid_truth(args) -> pd.DataFrame:
    """NEW: anti-circularity level 2 — lipid GWAS-truth scenario on REAL pool.

    Truth is fixed by lipid GWAS gene membership (independent of rhythmicity).
    No bootstrap (gene names needed for matching).
    """
    return run_block(
        args,
        scenarios=("lipid_gwas_truth",),
        priors=("uniform", "RP", "full"),
        use_real_only=True,
        suffix="[lipid_gwas_truth]",
    )


def block_sigma_sweep(args) -> pd.DataFrame:
    out = []
    for sigma in [0.1, 0.25, 0.5, 1.0, 2.0]:
        df = run_block(args, priors=("uniform", "RP"), sigma_log_w=sigma,
                       n_reps=args.sensitivity_reps,
                       extra_meta={"sigma_setting": sigma},
                       suffix=f"[sigma={sigma}]")
        out.append(df)
    return pd.concat(out, ignore_index=True)


def block_h2_sweep(args) -> pd.DataFrame:
    out = []
    for h2 in [0.02, 0.05, 0.10, 0.20]:
        df = run_block(args, priors=("uniform", "RP"),
                       sim_params_override={"trait_h2": h2},
                       n_reps=args.sensitivity_reps,
                       extra_meta={"h2_setting": h2},
                       suffix=f"[h2={h2}]")
        out.append(df)
    return pd.concat(out, ignore_index=True)


def block_n_sweep(args) -> pd.DataFrame:
    out = []
    for n in [1000, 2000, 5000]:
        df = run_block(args, priors=("uniform", "RP"),
                       sim_params_override={"n_individuals": n},
                       n_reps=args.sensitivity_reps,
                       extra_meta={"n_setting": n},
                       suffix=f"[n={n}]")
        out.append(df)
    return pd.concat(out, ignore_index=True)


def block_k_sweep(args) -> pd.DataFrame:
    out = []
    for k in [10, 30, 100, 300]:
        df = run_block(args, priors=("uniform", "RP"),
                       sim_params_override={"n_causal": k},
                       n_reps=args.sensitivity_reps,
                       extra_meta={"k_setting": k},
                       suffix=f"[k={k}]")
        out.append(df)
    return pd.concat(out, ignore_index=True)


BLOCKS: Dict[str, Callable] = {
    "main":         block_main,
    "lipid_truth":  block_lipid_truth,   # NEW
    "sigma_sweep":  block_sigma_sweep,
    "h2_sweep":     block_h2_sweep,
    "n_sweep":      block_n_sweep,
    "k_sweep":      block_k_sweep,
}


def main():
    ap = argparse.ArgumentParser(
        description="Final circadian-informed PWAS simulation v2 "
                    "(adds lipid GWAS-truth + raises SNP count).")
    ap.add_argument("--block", required=True, choices=list(BLOCKS.keys()))
    ap.add_argument("--pg-matrix", required=True, type=Path)
    ap.add_argument("--pqtl", required=True, type=Path)
    ap.add_argument("--ld-dir", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260507)
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--sensitivity-reps", type=int, default=30)
    ap.add_argument("--sigma-log-w", type=float, default=0.5)
    ap.add_argument("--n-individuals", type=int, default=2000)
    ap.add_argument("--n-proteins", type=int, default=1500)
    ap.add_argument("--n-causal", type=int, default=30)
    ap.add_argument("--trait-h2", type=float, default=0.05)
    ap.add_argument("--max-ld-snps", type=int, default=1500,           # <<<<< NEW default
                    help="Raised from 500 to reduce SNP reuse / artificial protein correlation.")
    ap.add_argument("--protein-noise-sd", type=float, default=0.2)
    ap.add_argument("--alpha-r", type=float, default=0.20)
    ap.add_argument("--alpha-a", type=float, default=0.15)
    ap.add_argument("--alpha-p", type=float, default=0.25)
    ap.add_argument("--alpha-c", type=float, default=0.40)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--target-phase-hour", type=float, default=9.0)
    ap.add_argument("--main-noise-sigma", type=float, default=0.1)
    ap.add_argument("--main-shrink-to-one", type=float, default=0.1)
    ap.add_argument("--main-gamma", type=float, default=3.0)

    args = ap.parse_args()
    mkdir(args.outdir)
    log(f"=== Block: {args.block} === outdir: {args.outdir}")
    log(f"=== max_ld_snps={args.max_ld_snps} (v2: raised from 500)")
    df = BLOCKS[args.block](args)

    raw_path = args.outdir / f"final_v2_{args.block}_raw.csv"
    df.to_csv(raw_path, index=False)
    log(f"Wrote {raw_path}  ({len(df)} rows)")

    group_cols = [c for c in ["sigma_setting", "h2_setting", "n_setting", "k_setting",
                              "scenario", "prior"] if c in df.columns]
    summary = (df.groupby(group_cols)
                 .agg(n=("delta_auc", "count"),
                      mean_delta_auc=("delta_auc", "mean"),
                      sd_delta_auc=("delta_auc", "std"))
                 .reset_index())
    summary["se_delta_auc"] = summary["sd_delta_auc"] / np.sqrt(summary["n"].clip(lower=1))
    sum_path = args.outdir / f"final_v2_{args.block}_summary.csv"
    summary.to_csv(sum_path, index=False)
    log(f"Wrote {sum_path}")
    log("Done.")


if __name__ == "__main__":
    main()
