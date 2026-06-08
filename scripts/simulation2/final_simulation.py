#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_simulation.py
===================

Final circadian-informed PWAS simulation, built on the conclusion of the ablation:
the entire usable prior signal is captured by a single feature, R^2 (rhythmicity).

THE FORMULA (Rhythmicity Prior, RP):
    w_j = exp(tau * R²_j) / mean(exp(tau * R²))
    tau = sigma_log_w / std(R²)             # auto-calibrated; default sigma=0.5

PWAS score:
    score_j = |Z_j| * sqrt(w_j)

DESIGN:
    - PAIRED-REPLICATE framework: same RNG seed across all priors per rep
      → enables paired t-tests with full statistical power.
    - Three priors compared in every replicate:
        * uniform  (baseline, w=1)
        * RP       (the final, recommended formula)
        * full     (the original 4-term reference, included as upper-bound check)
    - Five experiment blocks selectable via --block:
        main         : RP vs uniform vs full × 4 scenarios, default params, n_reps=200
        sigma_sweep  : RP under sigma_log_w ∈ {0.1, 0.25, 0.5, 1.0, 2.0}
        h2_sweep     : RP vs uniform under h² ∈ {0.02, 0.05, 0.10, 0.20}
        n_sweep      : RP vs uniform under n ∈ {1000, 2000, 5000}
        k_sweep      : RP vs uniform under n_causal ∈ {10, 30, 100, 300}

Place this file beside circadian_pwas_simulation_phase_causal.py.

Usage:
    python final_simulation.py --block main --n-reps 200 --outdir results/main \\
        --pg-matrix raw_data/circadian_info/report.pg_matrix.tsv \\
        --pqtl       work/pqtl_topk.csv \\
        --ld-dir     /data/CommonData/ukbb-ld/
"""
from __future__ import annotations
import argparse
import math
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
    normalize_01, phase_hour_to_rad, circular_distance_rad,
    zscore_safe, mkdir, log,
)


# =============================================================================
# THE FINAL FORMULA
# =============================================================================
def rhythmicity_prior(R2: np.ndarray, sigma_log_w: float = 0.5) -> Tuple[np.ndarray, float]:
    """Compute the Rhythmicity Prior (RP) weights.

    Parameters
    ----------
    R2 : array of shape (N,)
        Per-protein 24h-cosinor R² values (unnormalized; will be 0-1 normalized inside).
    sigma_log_w : float
        Target std of log-weights; the only tunable. Default 0.5
        (gives weight ratio ≈ e between an average-vs-strong rhythm).

    Returns
    -------
    w : array of shape (N,), mean ≈ 1
    tau : float, the calibrated effective tau
    """
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
    """The original 4-term S formula. Kept here as an upper-bound reference."""
    out = compute_strict_latent_weight(ann, params)
    return out["latent_w"].to_numpy(dtype=float), float(params.tau)


# =============================================================================
# Per-replicate engine
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
    extra_meta: Dict,
) -> List[dict]:
    """Run a single replicate; evaluate ALL priors on the SAME truth & data.

    The truth track always uses the FULL formula's latent_w (the "rich biology"
    assumption) so we measure each prior as an analyst's tool.
    Within one rep, every prior sees the same y, same X, same Gmat, same causal
    set in each scenario. → cleanest paired comparison.
    """
    rng = np.random.default_rng(seed + rep * 100003)
    ld = load_ld_matrix(ld_file, sim_params.max_ld_snps)
    X = simulate_genotype_from_ld(ld, sim_params.n_individuals, rng)
    proteins = ann["protein_id"].tolist()
    Gmat = construct_protein_matrix(X, pqtl, proteins, rng, sim_params.protein_noise_sd)

    # ---- truth track: latent_w from FULL (anti-circularity unchanged) ----
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
        # add same observation noise as full, for fair comparison
        w_obs = compute_observed_weight(w, weight_params, rng)
        weights["RP"], taus["RP"] = w_obs, t
    if "full" in priors_to_run:
        w, t = full_prior(ann, weight_params)
        w_obs = compute_observed_weight(w, weight_params, rng)
        weights["full"], taus["full"] = w_obs, t

    # inverse weight per prior (for wrong_phase specificity)
    inverse: Dict[str, np.ndarray] = {}
    for p, w in weights.items():
        inv = 1.0 / np.maximum(w, 1e-9)
        inverse[p] = inv / np.mean(inv)

    # ---- four scenarios ----
    rows = []
    for scenario in ("circadian_mediation", "non_circadian_mediation", "wrong_phase", "null"):
        if scenario == "circadian_mediation":
            causal_idx = select_causal_proteins(latent_w_truth, sim_params.n_causal,
                                                sim_params.causal_enrichment_gamma, rng, "circadian")
            y, gamma = simulate_phenotype_from_proteins(Gmat, causal_idx, sim_params.trait_h2, rng)
            use_inverse = False
        elif scenario == "non_circadian_mediation":
            causal_idx = select_causal_proteins(latent_w_truth, sim_params.n_causal,
                                                sim_params.causal_enrichment_gamma, rng, "uniform")
            y, gamma = simulate_phenotype_from_proteins(Gmat, causal_idx, sim_params.trait_h2, rng)
            use_inverse = False
        elif scenario == "wrong_phase":
            causal_idx = select_causal_proteins(latent_w_truth, sim_params.n_causal,
                                                sim_params.causal_enrichment_gamma, rng, "circadian")
            y, gamma = simulate_phenotype_from_proteins(Gmat, causal_idx, sim_params.trait_h2, rng)
            use_inverse = True
        elif scenario == "null":
            gamma = np.zeros(len(proteins))
            y = zscore_safe(rng.normal(size=sim_params.n_individuals))
            use_inverse = False

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
                "sigma_log_w": sigma_log_w if prior_name == "RP" else np.nan,
                "tau": taus[prior_name],
                "ordinary_auc": om["auc"], "weighted_auc": wm["auc"],
                "delta_auc": (wm["auc"] - om["auc"]) if np.isfinite(om["auc"]) and np.isfinite(wm["auc"]) else np.nan,
                "ordinary_pr_auc": om["pr_auc"], "weighted_pr_auc": wm["pr_auc"],
                "delta_pr_auc": (wm["pr_auc"] - om["pr_auc"]) if np.isfinite(om["pr_auc"]) and np.isfinite(wm["pr_auc"]) else np.nan,
            }
            row.update(extra_meta)
            rows.append(row)
    return rows


# =============================================================================
# Experiment blocks
# =============================================================================
def run_block(args, sim_params_override: Dict = None, extra_meta: Dict = None,
              priors=("uniform", "RP", "full"), sigma_log_w: float = None,
              n_reps: int = None, suffix: str = "") -> pd.DataFrame:
    log(f"-- block {args.block}{suffix}: priors={priors}, sigma={sigma_log_w}, "
        f"n_reps={n_reps or args.n_reps}, override={sim_params_override}")
    rng_seed = args.seed
    rng = np.random.default_rng(rng_seed)
    ann_real = load_or_fit_circadian_annotation(args.pg_matrix, args.outdir)
    pqtl_real = load_pqtl_topk(args.pqtl, args.outdir)
    ld_files = list_ld_files(args.ld_dir)
    ann_syn, pqtl_syn = bootstrap_synthetic_universe(ann_real, pqtl_real, args.n_proteins, rng)

    weight_params = WeightParams(
        args.alpha_r, args.alpha_a, args.alpha_p, args.alpha_c, args.tau,
        args.target_phase_hour, args.main_noise_sigma, args.main_shrink_to_one,
    )
    sp = SimParams(
        n_individuals=args.n_individuals, n_proteins=args.n_proteins,
        n_causal=args.n_causal, trait_h2=args.trait_h2,
        causal_enrichment_gamma=args.main_gamma,
        protein_noise_sd=args.protein_noise_sd, top_ks=(10, 20, 50),
        max_ld_snps=args.max_ld_snps, causal_prior_mode="full",
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
            r, ld_file, ann_syn, pqtl_syn, weight_params, sp, sigma,
            rng_seed, priors, extra,
        )
        all_rows.extend(rows)
        if r == 1 or r % max(1, nreps // 5) == 0 or r == nreps:
            log(f"     rep {r}/{nreps}")

    return pd.DataFrame(all_rows)


def block_main(args) -> pd.DataFrame:
    """Main: 3 priors × 4 scenarios × N reps, paired."""
    return run_block(args)


def block_sigma_sweep(args) -> pd.DataFrame:
    """Sensitivity to sigma_log_w (RP's only hyperparameter)."""
    out = []
    for sigma in [0.1, 0.25, 0.5, 1.0, 2.0]:
        df = run_block(args, priors=("uniform", "RP"), sigma_log_w=sigma,
                       n_reps=args.sensitivity_reps,
                       extra_meta={"sigma_setting": sigma},
                       suffix=f"[sigma={sigma}]")
        out.append(df)
    return pd.concat(out, ignore_index=True)


def block_h2_sweep(args) -> pd.DataFrame:
    """Sensitivity to heritability."""
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
    """Sensitivity to sample size."""
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
    """Sensitivity to number of causal proteins."""
    out = []
    for k in [10, 30, 100, 300]:
        df = run_block(args, priors=("uniform", "RP"),
                       sim_params_override={"n_causal": k},
                       n_reps=args.sensitivity_reps,
                       extra_meta={"k_setting": k},
                       suffix=f"[k={k}]")
        out.append(df)
    return pd.concat(out, ignore_index=True)


# =============================================================================
# Main
# =============================================================================
BLOCKS: Dict[str, Callable] = {
    "main":         block_main,
    "sigma_sweep":  block_sigma_sweep,
    "h2_sweep":     block_h2_sweep,
    "n_sweep":      block_n_sweep,
    "k_sweep":      block_k_sweep,
}


def main():
    ap = argparse.ArgumentParser(description="Final circadian-informed PWAS simulation (RP-only).")
    ap.add_argument("--block", required=True, choices=list(BLOCKS.keys()))
    ap.add_argument("--pg-matrix", required=True, type=Path)
    ap.add_argument("--pqtl", required=True, type=Path)
    ap.add_argument("--ld-dir", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260507)
    ap.add_argument("--n-reps", type=int, default=200,
                    help="Reps for the main block.")
    ap.add_argument("--sensitivity-reps", type=int, default=30,
                    help="Reps per level in sensitivity sweeps (smaller for speed).")

    # The only RP hyperparameter
    ap.add_argument("--sigma-log-w", type=float, default=0.5,
                    help="Target std(log w) for RP. Default 0.5.")

    # Defaults from sim2 (kept the same to keep results comparable)
    ap.add_argument("--n-individuals", type=int, default=2000)
    ap.add_argument("--n-proteins", type=int, default=1500)
    ap.add_argument("--n-causal", type=int, default=30)
    ap.add_argument("--trait-h2", type=float, default=0.05)
    ap.add_argument("--max-ld-snps", type=int, default=500)
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

    log(f"=== Block: {args.block} ===")
    log(f"Output dir: {args.outdir}")
    df = BLOCKS[args.block](args)

    raw_path = args.outdir / f"final_{args.block}_raw.csv"
    df.to_csv(raw_path, index=False)
    log(f"Wrote {raw_path}  ({len(df)} rows)")

    # Quick summary
    summary = (df.groupby(list(filter(lambda c: c in df.columns,
                                      ["sigma_setting", "h2_setting", "n_setting", "k_setting",
                                       "scenario", "prior"])))
                 .agg(n=("delta_auc", "count"),
                      mean_delta_auc=("delta_auc", "mean"),
                      sd_delta_auc=("delta_auc", "std"))
                 .reset_index())
    summary["se_delta_auc"] = summary["sd_delta_auc"] / np.sqrt(summary["n"].clip(lower=1))
    sum_path = args.outdir / f"final_{args.block}_summary.csv"
    summary.to_csv(sum_path, index=False)
    log(f"Wrote {sum_path}")
    log("Done.")


if __name__ == "__main__":
    main()
