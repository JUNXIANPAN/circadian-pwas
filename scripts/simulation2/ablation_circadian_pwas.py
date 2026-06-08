#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation extension for circadian_pwas_simulation_phase_causal.py.

Tests several alternative weight formulas against the current full formula:

    F0 baseline:    w = 1                          # no prior
    F1 r_only:      S = R                          # rhythmicity only
    F2 a_only:      S = A                          # amplitude only
    F3 r_a:         S = 0.5*R + 0.5*A              # rhythmicity + amplitude
    F4 r_vmf:       S = R + kappa*cos(phi - phi*)  # rhythmicity + circular phase (vMF)
    F5 full:        S = aR*R + aA*A + aP*R*P + aC*R*A*P  # current 4-term

Two ablation modes:
  --ablation-mode analysis_only:
      Truth always uses 'full' formula. Analysis varies.
      Question: "Given biology is rich, which features in the analysis prior
      actually drive ΔAUC?"
  --ablation-mode joint:
      Both truth and analysis use the same simplified formula.
      Question: "If biology is simpler, does a simpler prior suffice / hurt?"

Tau is standardized across formulas so that std(tau * S) matches the FULL
formula at the user's chosen tau. This separates 'feature-set choice' from
'weight sharpness'.

Wrong-phase scenario uses literal 1/observed_w as the inverse prior — works
uniformly across all formulas (including the phase-free ones).

Place this file next to circadian_pwas_simulation_phase_causal.py.

Example:
    python ablation_circadian_pwas.py \\
        --pg-matrix report.pg_matrix.tsv \\
        --pqtl pqtl_topk.csv \\
        --ld-dir ./ld \\
        --outdir ablation_out \\
        --n-reps 50 \\
        --ablation-mode analysis_only
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse the original simulation infrastructure.
from circadian_pwas_simulation_phase_causal import (
    WeightParams, SimParams,
    load_or_fit_circadian_annotation, load_pqtl_topk, list_ld_files,
    bootstrap_synthetic_universe, simulate_genotype_from_ld, load_ld_matrix,
    construct_protein_matrix, select_causal_proteins,
    simulate_phenotype_from_proteins, compute_pwas_z, ranking_metrics,
    compute_observed_weight,
    normalize_01, phase_hour_to_rad, circular_distance_rad,
    zscore_safe, mkdir, log,
)

FORMULAS = ["baseline", "r_only", "a_only", "r_a", "r_vmf", "full"]


# -----------------------------------------------------------------------------
# Feature extraction (one place; reused for every formula)
# -----------------------------------------------------------------------------
def compute_features(ann: pd.DataFrame, target_phase_hour: float):
    R = normalize_01(ann["rhythmicity"].to_numpy(dtype=float))
    A = normalize_01(ann["amplitude"].to_numpy(dtype=float))
    phase_rad = phase_hour_to_rad(ann["phase_hour"].to_numpy(dtype=float) % 24)
    target_rad = float(phase_hour_to_rad(target_phase_hour))
    d = circular_distance_rad(phase_rad, target_rad)
    P_align = np.clip(1.0 - d / np.pi, 0.0, 1.0)
    return R, A, P_align, phase_rad, target_rad


# -----------------------------------------------------------------------------
# The S(formula) function: ONE place where ablation lives.
# -----------------------------------------------------------------------------
def compute_S(R, A, P_align, phase_rad, target_rad,
              params: WeightParams, formula: str, kappa: float = 1.0) -> np.ndarray:
    if formula == "baseline":
        return np.zeros_like(R)
    if formula == "r_only":
        return R.copy()
    if formula == "a_only":
        return A.copy()
    if formula == "r_a":
        return 0.5 * R + 0.5 * A
    if formula == "r_vmf":
        # log-vMF kernel: kappa controls phase concentration.
        # The combined S = R + kappa*cos(phi - phi*) makes weight = exp(eff_tau*R) * exp(eff_tau*kappa*cos(.)).
        return R + kappa * np.cos(phase_rad - target_rad)
    if formula == "full":
        RP = R * P_align
        RAP = R * A * P_align
        return (params.alpha_r * R + params.alpha_a * A
                + params.alpha_p * RP + params.alpha_c * RAP)
    raise ValueError(f"Unknown formula {formula!r}")


def standardize_eff_tau(S: np.ndarray, target_log_std: float) -> float:
    """Pick eff_tau so that std(eff_tau * S) ≈ target_log_std.
    For baseline (S all zeros), return 0 (will produce uniform weights)."""
    s = float(np.std(S))
    if s < 1e-9:
        return 0.0
    return target_log_std / s


def compute_latent_weight(ann: pd.DataFrame, params: WeightParams,
                          formula: str, target_log_std: float,
                          kappa: float = 1.0) -> Tuple[np.ndarray, float]:
    """Return latent_w (mean=1) and the eff_tau used."""
    R, A, P_align, phase_rad, target_rad = compute_features(ann, params.target_phase_hour)
    S = compute_S(R, A, P_align, phase_rad, target_rad, params, formula, kappa)
    eff_tau = standardize_eff_tau(S, target_log_std)
    if eff_tau <= 0:
        return np.ones(len(R)), 0.0
    w_star = np.exp(eff_tau * S)
    return w_star / np.mean(w_star), eff_tau


def calibrate_target_log_std(ann: pd.DataFrame, params: WeightParams) -> float:
    """Compute target_log_std from the FULL formula at user's chosen tau.
    All other formulas standardize to this same dynamic range."""
    R, A, P_align, phase_rad, target_rad = compute_features(ann, params.target_phase_hour)
    S = compute_S(R, A, P_align, phase_rad, target_rad, params, "full")
    return float(params.tau * np.std(S))


# -----------------------------------------------------------------------------
# Single-replicate runner. Decouples truth-formula and analysis-formula.
# -----------------------------------------------------------------------------
def run_replicate(rep: int, ld_file: Path, ann: pd.DataFrame, pqtl: pd.DataFrame,
                  weight_params: WeightParams, sim_params: SimParams,
                  scenarios: Sequence[str], seed: int,
                  analysis_formula: str, truth_formula: str,
                  target_log_std: float, kappa: float, max_ld_snps: int) -> List[dict]:
    rng = np.random.default_rng(seed + rep * 100003)
    ld = load_ld_matrix(ld_file, max_ld_snps)
    X = simulate_genotype_from_ld(ld, sim_params.n_individuals, rng)
    proteins = ann["protein_id"].tolist()
    Gmat = construct_protein_matrix(X, pqtl, proteins, rng, sim_params.protein_noise_sd)

    # -- Truth track --
    latent_w_truth, tau_truth = compute_latent_weight(ann, weight_params, truth_formula, target_log_std, kappa)
    # -- Analysis track --
    latent_w_analysis, tau_analysis = compute_latent_weight(ann, weight_params, analysis_formula, target_log_std, kappa)
    observed_w = compute_observed_weight(latent_w_analysis, weight_params, rng)
    # Inverse prior (literal): the strictest specificity test.
    inv = 1.0 / np.maximum(observed_w, 1e-9)
    inverse_w = inv / np.mean(inv)

    rows = []
    for scenario in scenarios:
        if scenario == "circadian_mediation":
            causal_idx = select_causal_proteins(latent_w_truth, sim_params.n_causal,
                                                sim_params.causal_enrichment_gamma, rng, "circadian")
            y, gamma = simulate_phenotype_from_proteins(Gmat, causal_idx, sim_params.trait_h2, rng)
            w_use = observed_w
        elif scenario == "non_circadian_mediation":
            causal_idx = select_causal_proteins(latent_w_truth, sim_params.n_causal,
                                                sim_params.causal_enrichment_gamma, rng, "uniform")
            y, gamma = simulate_phenotype_from_proteins(Gmat, causal_idx, sim_params.trait_h2, rng)
            w_use = observed_w
        elif scenario == "wrong_phase":
            # Specificity: causal selected by truth, but weighted with INVERSE prior.
            causal_idx = select_causal_proteins(latent_w_truth, sim_params.n_causal,
                                                sim_params.causal_enrichment_gamma, rng, "circadian")
            y, gamma = simulate_phenotype_from_proteins(Gmat, causal_idx, sim_params.trait_h2, rng)
            w_use = inverse_w
        elif scenario == "null":
            gamma = np.zeros(len(proteins))
            y = zscore_safe(rng.normal(size=sim_params.n_individuals))
            w_use = observed_w
        else:
            continue

        z = compute_pwas_z(Gmat, y)
        ord_score = np.abs(z)
        wtd_score = np.abs(z) * np.sqrt(np.maximum(w_use, 0))
        truth = (gamma != 0).astype(int)
        om = ranking_metrics(truth, ord_score, sim_params.top_ks)
        wm = ranking_metrics(truth, wtd_score, sim_params.top_ks)

        rows.append({
            "rep": rep, "ld_file": ld_file.name, "scenario": scenario,
            "analysis_formula": analysis_formula, "truth_formula": truth_formula,
            "target_log_std": target_log_std,
            "eff_tau_analysis": tau_analysis, "eff_tau_truth": tau_truth,
            "kappa": kappa,
            "ordinary_auc": om["auc"], "weighted_auc": wm["auc"],
            "delta_auc": (wm["auc"] - om["auc"]) if np.isfinite(wm["auc"]) and np.isfinite(om["auc"]) else np.nan,
            "ordinary_pr_auc": om["pr_auc"], "weighted_pr_auc": wm["pr_auc"],
            "delta_pr_auc": (wm["pr_auc"] - om["pr_auc"]) if np.isfinite(wm["pr_auc"]) and np.isfinite(om["pr_auc"]) else np.nan,
            "ordinary_hits_at_20": om.get("hits_at_20", np.nan),
            "weighted_hits_at_20": wm.get("hits_at_20", np.nan),
            "ordinary_hits_at_50": om.get("hits_at_50", np.nan),
            "weighted_hits_at_50": wm.get("hits_at_50", np.nan),
        })
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ablation for circadian-informed PWAS prior formulas.")
    parser.add_argument("--pg-matrix", required=True, type=Path)
    parser.add_argument("--pqtl", required=True, type=Path)
    parser.add_argument("--ld-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--ablation-mode", default="analysis_only", choices=["analysis_only", "joint"])
    parser.add_argument("--formulas", default=",".join(FORMULAS),
                        help=f"Comma-separated subset of {FORMULAS}")
    parser.add_argument("--kappa", type=float, default=1.0,
                        help="vMF concentration for r_vmf formula. ~1 gives R and phase comparable weight.")
    # Reuse defaults from the parent script
    parser.add_argument("--n-individuals", type=int, default=2000)
    parser.add_argument("--n-proteins", type=int, default=1500)
    parser.add_argument("--n-causal", type=int, default=30)
    parser.add_argument("--trait-h2", type=float, default=0.05)
    parser.add_argument("--max-ld-snps", type=int, default=500)
    parser.add_argument("--protein-noise-sd", type=float, default=0.2)
    parser.add_argument("--alpha-r", type=float, default=0.20)
    parser.add_argument("--alpha-a", type=float, default=0.15)
    parser.add_argument("--alpha-p", type=float, default=0.25)
    parser.add_argument("--alpha-c", type=float, default=0.40)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--target-phase-hour", type=float, default=9.0)
    parser.add_argument("--main-noise-sigma", type=float, default=0.1)
    parser.add_argument("--main-shrink-to-one", type=float, default=0.1)
    parser.add_argument("--main-gamma", type=float, default=3.0)
    args = parser.parse_args()

    mkdir(args.outdir)

    requested = [f.strip() for f in args.formulas.split(",") if f.strip()]
    formulas = [f for f in requested if f in FORMULAS]
    unknown = set(requested) - set(formulas)
    if unknown:
        log(f"WARNING: unknown formulas ignored: {unknown}")
    log(f"Ablation mode = {args.ablation_mode}")
    log(f"Formulas      = {formulas}")

    log("[1/4] Loading annotation and pQTL...")
    ann_real = load_or_fit_circadian_annotation(args.pg_matrix, args.outdir)
    pqtl_real = load_pqtl_topk(args.pqtl, args.outdir)
    ld_files = list_ld_files(args.ld_dir)
    log(f"  ann proteins={len(ann_real)}, pqtl proteins={pqtl_real['protein_id'].nunique()}, LD files={len(ld_files)}")

    log("[2/4] Bootstrapping synthetic universe...")
    rng = np.random.default_rng(args.seed)
    ann_syn, pqtl_syn = bootstrap_synthetic_universe(ann_real, pqtl_real, args.n_proteins, rng)

    weight_params = WeightParams(
        args.alpha_r, args.alpha_a, args.alpha_p, args.alpha_c, args.tau,
        args.target_phase_hour, args.main_noise_sigma, args.main_shrink_to_one,
    )
    sim_params = SimParams(
        n_individuals=args.n_individuals, n_proteins=args.n_proteins,
        n_causal=args.n_causal, trait_h2=args.trait_h2,
        causal_enrichment_gamma=args.main_gamma,
        protein_noise_sd=args.protein_noise_sd, top_ks=(10, 20, 50),
        max_ld_snps=args.max_ld_snps, causal_prior_mode="full",  # not used here
    )

    log("[3/4] Calibrating target_log_std from FULL formula at user's tau...")
    target_log_std = calibrate_target_log_std(ann_syn, weight_params)
    log(f"  target_log_std = {target_log_std:.4f} (anchors dynamic range across formulas)")

    scenarios = ["circadian_mediation", "non_circadian_mediation", "wrong_phase", "null"]
    all_rows: List[dict] = []
    log("[4/4] Running ablation...")
    for analysis_formula in formulas:
        truth_formula = "full" if args.ablation_mode == "analysis_only" else analysis_formula
        log(f"  >> analysis={analysis_formula:10s}  truth={truth_formula:10s}  reps={args.n_reps}")
        formula_seed = args.seed + abs(hash(analysis_formula)) % 100000
        for r in range(1, args.n_reps + 1):
            ld_file = ld_files[(r - 1) % len(ld_files)]
            rows = run_replicate(
                r, ld_file, ann_syn, pqtl_syn, weight_params, sim_params,
                scenarios, formula_seed,
                analysis_formula, truth_formula, target_log_std, args.kappa,
                args.max_ld_snps,
            )
            all_rows.extend(rows)
            if r == 1 or r % max(1, args.n_reps // 5) == 0 or r == args.n_reps:
                log(f"     rep {r}/{args.n_reps}")

    # -- Save raw + summary --
    raw = pd.DataFrame(all_rows)
    raw.to_csv(args.outdir / "ablation_raw.csv", index=False)

    summary_rows = []
    for (af, tf, sc), grp in raw.groupby(["analysis_formula", "truth_formula", "scenario"]):
        n = int(grp["delta_auc"].notna().sum())
        sd_da = grp["delta_auc"].std(ddof=1) if n >= 2 else np.nan
        sd_dp = grp["delta_pr_auc"].std(ddof=1) if n >= 2 else np.nan
        summary_rows.append({
            "analysis_formula": af, "truth_formula": tf, "scenario": sc, "n_reps": n,
            "ordinary_auc_mean": grp["ordinary_auc"].mean(),
            "weighted_auc_mean": grp["weighted_auc"].mean(),
            "delta_auc_mean": grp["delta_auc"].mean(),
            "delta_auc_se": (sd_da / math.sqrt(max(1, n))) if np.isfinite(sd_da) else np.nan,
            "delta_pr_auc_mean": grp["delta_pr_auc"].mean(),
            "delta_pr_auc_se": (sd_dp / math.sqrt(max(1, n))) if np.isfinite(sd_dp) else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["scenario", "analysis_formula"])
    summary.to_csv(args.outdir / "ablation_summary.csv", index=False)

    # Pivot table for the main figure: scenario × formula → ΔAUC
    pivot = summary.pivot_table(index="analysis_formula", columns="scenario",
                                values="delta_auc_mean").reindex(formulas)
    pivot.to_csv(args.outdir / "ablation_delta_auc_pivot.csv")
    log("\n=== ΔAUC (mean) — rows: analysis_formula, cols: scenario ===")
    log(pivot.round(4).to_string())

    log(f"\nWrote:")
    log(f"  {args.outdir / 'ablation_raw.csv'}")
    log(f"  {args.outdir / 'ablation_summary.csv'}")
    log(f"  {args.outdir / 'ablation_delta_auc_pivot.csv'}")
    log("Done.")


if __name__ == "__main__":
    main()
