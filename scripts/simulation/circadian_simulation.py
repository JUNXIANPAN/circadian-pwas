
"""
circadian_simulation.py

Semi-synthetic simulation for circadian-informed PWAS reweighting.

Design principles
-----------------
1) Use the real proteomics matrix only to learn empirical design features:
   - number of proteins
   - sampling times
   - missingness pattern
   - baseline expression / amplitude / noise scales
2) In simulation, rhythmic truth and causal truth are explicitly known.
3) Temporal priors are re-estimated from simulated expression, then used to
   reweight externally simulated baseline PWAS Z statistics:
       Z_weighted = Z_PWAS * sqrt(w)
4) Weights are normalized to mean 1, matching the framework slides and the
   calibration logic used by p-value/Z reweighting methods.

Run example
-----------
python scripts/simulation/circadian_simulation.py \
  --input_tsv "report.pg_matrix.tsv" \
  --outdir "sim_results" \
  --n_reps 100 \
  --scenario informative
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score


ANNOTATION_COLS = [
    "Protein.Group",
    "Protein.Names",
    "Genes",
    "First.Protein.Description",
]


# ----------------------------
# Utilities
# ----------------------------
def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def minmax_scale(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    if valid.sum() == 0:
        return pd.Series(np.nan, index=s.index)
    s_min = s[valid].min()
    s_max = s[valid].max()
    if s_max == s_min:
        out = pd.Series(0.5, index=s.index, dtype=float)
        out[~valid] = np.nan
        return out
    out = (s - s_min) / (s_max - s_min)
    return out.clip(0, 1)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


# ----------------------------
# Parse the real matrix design
# ----------------------------
def parse_sample_info(colname: str) -> Optional[Dict]:
    pattern = r'(\d+)-Dag(\d+)-t(\d+)\.mzML$'
    m = re.search(pattern, colname)
    if m is None:
        return None
    sample_id = m.group(1)
    day = int(m.group(2))
    hour = int(m.group(3))
    absolute_time = (day - 1) * 24 + hour
    return {
        "column_name": colname,
        "sample_id": sample_id,
        "day": day,
        "hour": hour,
        "absolute_time": absolute_time,
    }


def load_real_design(input_tsv: str) -> Dict:
    raw = pd.read_csv(input_tsv, sep="\t")

    expr_cols = [c for c in raw.columns if c not in ANNOTATION_COLS]
    sample_info = [parse_sample_info(c) for c in expr_cols]
    sample_info = [x for x in sample_info if x is not None]
    sample_info_df = pd.DataFrame(sample_info).sort_values(
        ["sample_id", "absolute_time"]
    )

    # Expression matrix
    expr = raw[sample_info_df["column_name"].tolist()].apply(
        pd.to_numeric, errors="coerce"
    )

    # Per-protein rough empirical summaries
    protein_mean = expr.mean(axis=1, skipna=True)
    protein_sd = expr.std(axis=1, skipna=True)

    # Rough amplitude estimate based on within-protein dynamic range
    q95 = expr.quantile(0.95, axis=1, interpolation="linear")
    q05 = expr.quantile(0.05, axis=1, interpolation="linear")
    rough_amplitude = (q95 - q05) / 2.0

    # Missingness templates across proteins
    obs_mask = ~expr.isna()
    mask_templates = obs_mask.to_numpy(dtype=bool)

    meta = raw[ANNOTATION_COLS].copy()
    meta["protein_index"] = np.arange(len(meta))

    design = {
        "n_proteins_real": raw.shape[0],
        "sample_info_df": sample_info_df.reset_index(drop=True),
        "times_unique": np.sort(sample_info_df["absolute_time"].unique()),
        "column_names": sample_info_df["column_name"].tolist(),
        "protein_mean": protein_mean.to_numpy(dtype=float),
        "protein_sd": protein_sd.fillna(protein_sd.median()).to_numpy(dtype=float),
        "rough_amplitude": rough_amplitude.fillna(rough_amplitude.median()).to_numpy(dtype=float),
        "mask_templates": mask_templates,
        "meta": meta,
    }
    return design


# ----------------------------
# Generate simulation truths
# ----------------------------
@dataclass
class SimTruth:
    rhythmic_true: np.ndarray
    amplitude_true: np.ndarray
    phase_true: np.ndarray
    baseline_true: np.ndarray
    sigma_true: np.ndarray
    phase_alignment_true: np.ndarray
    prior_score_true: np.ndarray
    causal_true: np.ndarray


def sample_truths(
    design: Dict,
    rng: np.random.Generator,
    target_time: float,
    rhythmic_frac: float,
    alpha_R: float,
    alpha_A: float,
    alpha_P: float,
    alpha_C: float,
    scenario: str,
    intercept_causal: float,
    gamma_prior: float,
) -> SimTruth:
    n = design["n_proteins_real"]

    # 1) rhythmic truth
    rhythmic_true = rng.binomial(1, rhythmic_frac, size=n).astype(int)

    # 2) baseline / noise from empirical distributions
    baseline_source = design["protein_mean"]
    sigma_source = np.clip(design["protein_sd"], 0.05, None)
    amp_source = np.clip(design["rough_amplitude"], 0.05, None)

    baseline_true = rng.choice(baseline_source, size=n, replace=True)
    sigma_true = rng.choice(sigma_source, size=n, replace=True)
    amp_emp = rng.choice(amp_source, size=n, replace=True)

    # Rhythmic proteins get substantial amplitude; arrhythmic get 0
    amplitude_true = np.where(
        rhythmic_true == 1,
        np.clip(amp_emp, np.quantile(amp_source, 0.20), np.quantile(amp_source, 0.95)),
        0.0,
    )

    # Random phase across 24h
    phase_true = rng.uniform(0, 24, size=n)

    # Ground-truth prior ingredients scaled to [0,1]
    R_true = rhythmic_true.astype(float)
    A_true = minmax_scale(pd.Series(amplitude_true)).fillna(0.0).to_numpy()

    diff = np.abs(phase_true - target_time)
    circ_dist = np.minimum(diff, 24 - diff)
    phase_alignment_true = np.clip(1 - circ_dist / 12.0, 0, 1)

    prior_score_true = (
        alpha_R * R_true
        + alpha_A * A_true
        + alpha_P * phase_alignment_true
        + alpha_C * (R_true * A_true)
    )

    # 3) causal truth generation
    if scenario == "informative":
        p_causal = logistic(intercept_causal + gamma_prior * prior_score_true)
    elif scenario == "noninformative":
        base_p = logistic(np.array([intercept_causal]))[0]
        p_causal = np.repeat(base_p, n)
    elif scenario == "global_null":
        p_causal = np.zeros(n)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    causal_true = rng.binomial(1, p_causal, size=n).astype(int)

    return SimTruth(
        rhythmic_true=rhythmic_true,
        amplitude_true=amplitude_true,
        phase_true=phase_true,
        baseline_true=baseline_true,
        sigma_true=sigma_true,
        phase_alignment_true=phase_alignment_true,
        prior_score_true=prior_score_true,
        causal_true=causal_true,
    )


# ----------------------------
# Simulate expression matrix
# ----------------------------
def simulate_expression(
    design: Dict,
    truth: SimTruth,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sample_info_df = design["sample_info_df"]
    times = sample_info_df["absolute_time"].to_numpy(dtype=float)
    n_proteins = design["n_proteins_real"]
    n_cols = len(times)

    y = np.empty((n_proteins, n_cols), dtype=float)
    omega = 2 * np.pi / 24.0

    for g in range(n_proteins):
        mu = truth.baseline_true[g]
        A = truth.amplitude_true[g]
        phi = truth.phase_true[g]
        sigma = truth.sigma_true[g]

        signal = mu + A * np.cos(omega * (times - phi))
        noise = rng.normal(0, sigma, size=n_cols)
        y[g, :] = signal + noise

    # Apply empirical missingness pattern by resampling row templates
    template_idx = rng.integers(0, design["mask_templates"].shape[0], size=n_proteins)
    obs_mask = design["mask_templates"][template_idx]
    y[~obs_mask] = np.nan

    expr_df = pd.DataFrame(y, columns=design["column_names"])
    out = pd.concat([design["meta"][ANNOTATION_COLS].copy(), expr_df], axis=1)
    return out


# ----------------------------
# Prior estimation from simulated expression
# ----------------------------
def melt_expression(expr_df: pd.DataFrame, sample_info_df: pd.DataFrame) -> pd.DataFrame:
    long_df = expr_df.melt(
        id_vars=ANNOTATION_COLS,
        value_vars=sample_info_df["column_name"].tolist(),
        var_name="column_name",
        value_name="expression",
    )
    long_df = long_df.merge(sample_info_df, on="column_name", how="left")
    long_df["expression"] = pd.to_numeric(long_df["expression"], errors="coerce")
    long_df = long_df.dropna(subset=["expression"]).copy()
    return long_df


def fit_cosinor_one_protein(df_protein: pd.DataFrame, target_time: float) -> pd.Series:
    dfp = df_protein.dropna(subset=["expression", "absolute_time"]).copy()
    if dfp.shape[0] < 6:
        return pd.Series(
            {
                "n_obs": dfp.shape[0],
                "rhythmicity_p": np.nan,
                "rhythmicity_score": np.nan,
                "amplitude_est": np.nan,
                "phase_hour_est": np.nan,
                "phase_alignment_est": np.nan,
            }
        )

    t = dfp["absolute_time"].to_numpy(dtype=float)
    y = dfp["expression"].to_numpy(dtype=float)
    omega = 2 * np.pi / 24.0

    X_full = pd.DataFrame(
        {"intercept": 1.0, "cos": np.cos(omega * t), "sin": np.sin(omega * t)}
    )
    X_null = pd.DataFrame({"intercept": np.ones_like(y)})

    full_model = sm.OLS(y, X_full).fit()
    null_model = sm.OLS(y, X_null).fit()

    lr_stat = 2 * (full_model.llf - null_model.llf)
    p_value = 1 - chi2.cdf(lr_stat, df=2)

    b1 = full_model.params["cos"]
    b2 = full_model.params["sin"]

    amplitude_est = float(np.sqrt(b1**2 + b2**2))
    phi = float(np.arctan2(b2, b1))
    phase_hour_est = (phi / omega) % 24

    diff = abs(phase_hour_est - target_time)
    circ_dist = min(diff, 24 - diff)
    phase_alignment_est = max(0.0, 1 - circ_dist / 12.0)

    return pd.Series(
        {
            "n_obs": dfp.shape[0],
            "rhythmicity_p": p_value,
            "rhythmicity_score": -np.log10(max(p_value, 1e-300)),
            "amplitude_est": amplitude_est,
            "phase_hour_est": phase_hour_est,
            "phase_alignment_est": phase_alignment_est,
        }
    )


def estimate_priors(
    expr_df: pd.DataFrame,
    sample_info_df: pd.DataFrame,
    target_time: float,
    alpha_R: float,
    alpha_A: float,
    alpha_P: float,
    alpha_C: float,
) -> pd.DataFrame:
    # 先保留完整蛋白列表，避免后面因为缺失把蛋白丢掉
    all_proteins = expr_df[ANNOTATION_COLS].copy()

    long_df = melt_expression(expr_df, sample_info_df)

    feature_table = (
        long_df.groupby(ANNOTATION_COLS, dropna=False)
        .apply(lambda x: fit_cosinor_one_protein(x, target_time=target_time))
        .reset_index()
    )

    # 合并回完整蛋白表，保证所有蛋白都在
    feature_table = all_proteins.merge(
        feature_table,
        on=ANNOTATION_COLS,
        how="left"
    )

    feature_table["R_g"] = minmax_scale(feature_table["rhythmicity_score"])
    feature_table["A_g"] = minmax_scale(feature_table["amplitude_est"])
    feature_table["P_g"] = pd.to_numeric(
        feature_table["phase_alignment_est"], errors="coerce"
    ).clip(0, 1)

    # 没法估计 prior 的蛋白，先设成 0
    feature_table["R_g"] = feature_table["R_g"].fillna(0.0)
    feature_table["A_g"] = feature_table["A_g"].fillna(0.0)
    feature_table["P_g"] = feature_table["P_g"].fillna(0.0)

    feature_table["interaction_RA"] = feature_table["R_g"] * feature_table["A_g"]
    feature_table["S_g"] = (
        alpha_R * feature_table["R_g"]
        + alpha_A * feature_table["A_g"]
        + alpha_P * feature_table["P_g"]
        + alpha_C * feature_table["interaction_RA"]
    )
    feature_table["w_star"] = np.exp(feature_table["S_g"])
    feature_table["w_g"] = feature_table["w_star"] / feature_table["w_star"].mean()

    return feature_table


# ----------------------------
# Simulate baseline PWAS and reweight
# ----------------------------
def simulate_baseline_pwas_z(
    truth: SimTruth,
    rng: np.random.Generator,
    mu_causal: float,
    sigma_z: float = 1.0,
) -> np.ndarray:
    n = len(truth.causal_true)
    z = rng.normal(0, sigma_z, size=n)
    causal_idx = truth.causal_true == 1
    z[causal_idx] = rng.normal(mu_causal, sigma_z, size=causal_idx.sum())
    return z


def apply_weighting(z_pwas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return z_pwas * np.sqrt(weights)


# ----------------------------
# Evaluation
# ----------------------------
def count_topk_hits(y_true: np.ndarray, scores: np.ndarray, k: int) -> int:
    k = min(k, len(scores))
    idx = np.argsort(-np.abs(scores))[:k]
    return int(y_true[idx].sum())


def approx_null_fpr(scores: np.ndarray, threshold: float = 5.0) -> float:
    return float(np.mean(np.abs(scores) >= threshold))


def evaluate_one_rep(
    truth: SimTruth,
    z_baseline: np.ndarray,
    z_weighted: np.ndarray,
    topk: int,
) -> Dict:
    y = truth.causal_true.astype(int)

    return {
        "n_causal": int(y.sum()),
        "baseline_auc": safe_auc(y, np.abs(z_baseline)),
        "weighted_auc": safe_auc(y, np.abs(z_weighted)),
        "baseline_ap": safe_ap(y, np.abs(z_baseline)),
        "weighted_ap": safe_ap(y, np.abs(z_weighted)),
        "baseline_topk_hits": count_topk_hits(y, z_baseline, topk),
        "weighted_topk_hits": count_topk_hits(y, z_weighted, topk),
        "baseline_null_fpr_absz_ge_5": approx_null_fpr(z_baseline, threshold=5.0),
        "weighted_null_fpr_absz_ge_5": approx_null_fpr(z_weighted, threshold=5.0),
    }


# ----------------------------
# Main simulation loop
# ----------------------------
def run_simulation(
    input_tsv: str,
    outdir: str,
    seed: int,
    n_reps: int,
    scenario: str,
    rhythmic_frac: float,
    target_time: float,
    alpha_R: float,
    alpha_A: float,
    alpha_P: float,
    alpha_C: float,
    intercept_causal: float,
    gamma_prior: float,
    mu_causal: float,
    topk: int,
) -> Dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    design = load_real_design(input_tsv)
    rng = np.random.default_rng(seed)

    rep_rows = []
    prior_example = None
    expr_example = None
    truth_example = None

    for rep in range(1, n_reps + 1):
        truth = sample_truths(
            design=design,
            rng=rng,
            target_time=target_time,
            rhythmic_frac=rhythmic_frac,
            alpha_R=alpha_R,
            alpha_A=alpha_A,
            alpha_P=alpha_P,
            alpha_C=alpha_C,
            scenario=scenario,
            intercept_causal=intercept_causal,
            gamma_prior=gamma_prior,
        )

        expr_df = simulate_expression(design, truth, rng)
        prior_df = estimate_priors(
            expr_df=expr_df,
            sample_info_df=design["sample_info_df"],
            target_time=target_time,
            alpha_R=alpha_R,
            alpha_A=alpha_A,
            alpha_P=alpha_P,
            alpha_C=alpha_C,
        )

        z_baseline = simulate_baseline_pwas_z(
            truth=truth,
            rng=rng,
            mu_causal=mu_causal,
            sigma_z=1.0,
        )
        
        if len(prior_df) != len(z_baseline):
           raise ValueError(
         f"Length mismatch: len(z_baseline)={len(z_baseline)}, len(prior_df)={len(prior_df)}"
        )

        z_weighted = apply_weighting(z_baseline, prior_df["w_g"].to_numpy(dtype=float))

        rep_eval = evaluate_one_rep(
            truth=truth,
            z_baseline=z_baseline,
            z_weighted=z_weighted,
            topk=topk,
        )
        rep_eval["rep"] = rep
        rep_rows.append(rep_eval)

        if rep == 1:
            joined = prior_df.copy()
            joined["rhythmic_true"] = truth.rhythmic_true
            joined["amplitude_true"] = truth.amplitude_true
            joined["phase_true"] = truth.phase_true
            joined["prior_score_true"] = truth.prior_score_true
            joined["causal_true"] = truth.causal_true
            joined["z_pwas"] = z_baseline
            joined["z_weighted"] = z_weighted
            prior_example = joined
            expr_example = expr_df
            truth_example = truth

    rep_df = pd.DataFrame(rep_rows)

    summary = rep_df.agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]

    rep_df.to_csv(outdir / f"simulation_results_{scenario}.tsv", sep="\t", index=False)
    summary.to_csv(outdir / f"simulation_summary_{scenario}.tsv", sep="\t", index=False)

    if prior_example is not None:
        prior_example.to_csv(outdir / f"prior_and_truth_example_{scenario}.tsv", sep="\t", index=False)
    if expr_example is not None:
        expr_example.to_csv(outdir / f"simulated_expression_example_{scenario}.tsv", sep="\t", index=False)

    config = {
        "input_tsv": input_tsv,
        "seed": seed,
        "n_reps": n_reps,
        "scenario": scenario,
        "rhythmic_frac": rhythmic_frac,
        "target_time": target_time,
        "alpha_R": alpha_R,
        "alpha_A": alpha_A,
        "alpha_P": alpha_P,
        "alpha_C": alpha_C,
        "intercept_causal": intercept_causal,
        "gamma_prior": gamma_prior,
        "mu_causal": mu_causal,
        "topk": topk,
        "n_proteins_real": int(design["n_proteins_real"]),
        "n_time_columns": int(len(design["column_names"])),
    }
    (outdir / f"simulation_config_{scenario}.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    return {
        "rep_df": rep_df,
        "summary": summary,
        "config": config,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semi-synthetic simulation for circadian-informed PWAS.")
    p.add_argument("--input_tsv", type=str, required=True, help="Path to the real proteomics TSV matrix.")
    p.add_argument("--outdir", type=str, default="sim_results", help="Output directory.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_reps", type=int, default=100)
    p.add_argument("--scenario", type=str, default="informative",
                   choices=["informative", "noninformative", "global_null"])
    p.add_argument("--rhythmic_frac", type=float, default=0.30)
    p.add_argument("--target_time", type=float, default=6.0)
    p.add_argument("--alpha_R", type=float, default=0.30)
    p.add_argument("--alpha_A", type=float, default=0.30)
    p.add_argument("--alpha_P", type=float, default=0.20)
    p.add_argument("--alpha_C", type=float, default=0.20)
    p.add_argument("--intercept_causal", type=float, default=-3.0,
                   help="Controls baseline causal rate on logit scale.")
    p.add_argument("--gamma_prior", type=float, default=2.5,
                   help="Strength of prior informativeness in the informative scenario.")
    p.add_argument("--mu_causal", type=float, default=2.5,
                   help="Mean baseline PWAS Z for causal proteins.")
    p.add_argument("--topk", type=int, default=50)
    return p


def main() -> None:
    args = build_parser().parse_args()

    alpha_sum = args.alpha_R + args.alpha_A + args.alpha_P + args.alpha_C
    if not np.isclose(alpha_sum, 1.0):
        raise ValueError(
            f"alpha_R + alpha_A + alpha_P + alpha_C must sum to 1. Currently: {alpha_sum}"
        )

    result = run_simulation(
        input_tsv=args.input_tsv,
        outdir=args.outdir,
        seed=args.seed,
        n_reps=args.n_reps,
        scenario=args.scenario,
        rhythmic_frac=args.rhythmic_frac,
        target_time=args.target_time,
        alpha_R=args.alpha_R,
        alpha_A=args.alpha_A,
        alpha_P=args.alpha_P,
        alpha_C=args.alpha_C,
        intercept_causal=args.intercept_causal,
        gamma_prior=args.gamma_prior,
        mu_causal=args.mu_causal,
        topk=args.topk,
    )

    print("\nSimulation finished.")
    print("\nConfig:")
    print(json.dumps(result["config"], indent=2))
    print("\nSummary:")
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
