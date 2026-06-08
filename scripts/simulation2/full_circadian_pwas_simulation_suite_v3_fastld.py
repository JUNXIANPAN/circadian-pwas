#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full semi-simulation pipeline for circadian-informed PWAS Z-score prioritization.

Run this single file to produce a complete simulation result directory.

Strict formula implemented:
  S_g(T) = alpha_R R_g + alpha_A A_g + alpha_P P_g(T) + alpha_C (R_g * A_g)
  w*_g(T) = exp(S_g(T))
  w_g(T) = w*_g(T) / mean_g(w*_g(T))
  Z_cw,g(T) = Z_PWAS,g * sqrt(w_g(T))

Outputs:
  all_simulation_raw.csv
  all_simulation_summary.csv
  all_simulation_summary_compact.csv
"""
from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=float)
    if finite.sum() == 0:
        return out
    xmin = np.nanmin(x[finite])
    xmax = np.nanmax(x[finite])
    if xmax <= xmin:
        out[finite] = 0.5
        return out
    out[finite] = (x[finite] - xmin) / (xmax - xmin)
    out[~finite] = np.nanmedian(out[finite]) if finite.sum() else 0.5
    return out


def zscore_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def circular_distance_rad(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def phase_hour_to_rad(hour):
    return np.asarray(hour) / 24.0 * 2.0 * np.pi


def find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    low_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in low_map:
            return low_map[c.lower()]
    return None


def clean_protein_id(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).strip())


def read_table_auto(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in [".tsv", ".txt"] else ","
    return pd.read_csv(path, sep=sep)


def parse_time_from_col(col: str) -> Optional[float]:
    s = str(col)
    patterns = [
        r"(?:ZT|CT|T|time|hour|h)[_\- ]?(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)h$",
        r"^(\d+(?:\.\d+)?)$",
    ]
    for p in patterns:
        m = re.search(p, s, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 48:
                return val % 24
    return None


def fit_simple_24h_cosinor(y: np.ndarray, times_h: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    times_h = np.asarray(times_h, dtype=float)
    ok = np.isfinite(y) & np.isfinite(times_h)
    y = y[ok]
    t = times_h[ok]
    if len(y) < 4 or np.nanstd(y) <= 1e-12:
        return 0.0, 0.0, 0.0
    omega = 2.0 * np.pi / 24.0
    X = np.column_stack([np.ones_like(t), np.cos(omega * t), np.sin(omega * t)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return 0.0, 0.0, 0.0
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-12))
    b_cos, b_sin = beta[1], beta[2]
    amplitude = float(np.sqrt(b_cos ** 2 + b_sin ** 2))
    phi = math.atan2(b_sin, b_cos)
    phase_h = (phi / omega) % 24.0
    return float(r2), amplitude, float(phase_h)


def load_or_fit_circadian_annotation(pg_matrix: Path, outdir: Path) -> pd.DataFrame:
    df = read_table_auto(pg_matrix)
    protein_col = find_first_existing_column(df, ["protein_id", "protein", "gene", "gene_name", "symbol", "uniprot", "UniProt", "analyte", "analyte_id"])
    rhythmicity_col = find_first_existing_column(df, ["rhythmicity", "rhythmicity_r2", "r2", "R2", "circadian_r2", "minus_log10_p", "neglog10p"])
    amp_col = find_first_existing_column(df, ["amplitude", "amp", "A"])
    phase_col = find_first_existing_column(df, ["phase_hour", "phase", "peak_time", "acrophase", "phase_h"])

    if protein_col and rhythmicity_col and amp_col and phase_col:
        ann = pd.DataFrame({
            "protein_id": df[protein_col].map(clean_protein_id),
            "rhythmicity": pd.to_numeric(df[rhythmicity_col], errors="coerce"),
            "amplitude": pd.to_numeric(df[amp_col], errors="coerce"),
            "phase_hour": pd.to_numeric(df[phase_col], errors="coerce") % 24,
        }).dropna()
        ann = ann[ann["protein_id"] != ""].drop_duplicates("protein_id")
        ann.to_csv(outdir / "circadian_annotation_real.csv", index=False)
        return ann

    if protein_col is None:
        protein_col = df.columns[0]
    time_cols, times = [], []
    for c in df.columns:
        if c == protein_col:
            continue
        t = parse_time_from_col(str(c))
        if t is not None:
            time_cols.append(c)
            times.append(t)
    if len(time_cols) < 4:
        numeric_cols = []
        for c in df.columns:
            if c == protein_col:
                continue
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().mean() > 0.5:
                numeric_cols.append(c)
        time_cols = numeric_cols
        if len(time_cols) < 4:
            raise ValueError("Could not identify enough time-series columns in pg_matrix.")
        times = np.linspace(0, 24, len(time_cols), endpoint=False).tolist()

    rows = []
    times_arr = np.asarray(times, dtype=float)
    for _, r in df.iterrows():
        pid = clean_protein_id(r[protein_col])
        if not pid:
            continue
        y = pd.to_numeric(r[time_cols], errors="coerce").values.astype(float)
        r2, amp, phase_h = fit_simple_24h_cosinor(y, times_arr)
        rows.append((pid, r2, amp, phase_h))
    ann = pd.DataFrame(rows, columns=["protein_id", "rhythmicity", "amplitude", "phase_hour"]).drop_duplicates("protein_id")
    ann.to_csv(outdir / "circadian_annotation_real.csv", index=False)
    return ann


def load_pqtl_topk(pqtl_path: Path, outdir: Path, top_k: int = 5) -> pd.DataFrame:
    pq = read_table_auto(pqtl_path)
    protein_col = find_first_existing_column(pq, ["protein_id", "protein", "gene", "gene_name", "symbol", "uniprot", "UniProt", "analyte", "analyte_id"])
    snp_col = find_first_existing_column(pq, ["snp", "SNP", "rsid", "variant", "variant_id", "ID"])
    beta_col = find_first_existing_column(pq, ["beta", "effect", "b", "BETA", "slope"])
    p_col = find_first_existing_column(pq, ["p", "pval", "pvalue", "P", "P_VALUE"])
    if protein_col is None or snp_col is None or beta_col is None:
        raise ValueError(f"pQTL file must contain protein, SNP, and beta columns. Found: {list(pq.columns)}")
    out = pd.DataFrame({
        "protein_id": pq[protein_col].map(clean_protein_id),
        "snp": pq[snp_col].astype(str),
        "beta": pd.to_numeric(pq[beta_col], errors="coerce"),
    })
    out["p"] = pd.to_numeric(pq[p_col], errors="coerce") if p_col else np.nan
    out = out.dropna(subset=["protein_id", "snp", "beta"])
    out = out[out["protein_id"] != ""]
    if p_col:
        out = out.sort_values(["protein_id", "p"])
    else:
        out = out.assign(abs_beta=lambda d: np.abs(d["beta"])).sort_values(["protein_id", "abs_beta"], ascending=[True, False])
    out = out.groupby("protein_id", as_index=False).head(top_k).reset_index(drop=True)
    out[["protein_id", "snp", "beta"]].to_csv(outdir / "pqtl_topk_real.csv", index=False)
    return out[["protein_id", "snp", "beta"]]


@dataclass
class WeightParams:
    alpha_r: float = 0.25
    alpha_a: float = 0.25
    alpha_p: float = 0.25
    alpha_c: float = 0.25
    target_phase_hour: float = 9.0
    observed_noise_sigma: float = 1.75
    observed_shrink_to_one: float = 0.5


@dataclass
class SimParams:
    n_individuals: int = 2000
    n_proteins: int = 1500
    n_causal: int = 30
    trait_h2: float = 0.05
    causal_enrichment_gamma: float = 2.0
    protein_noise_sd: float = 0.2
    top_ks: Tuple[int, ...] = (10, 20, 50)
    max_ld_snps: int = 500


def compute_strict_latent_weight(ann: pd.DataFrame, params: WeightParams) -> pd.DataFrame:
    s = params.alpha_r + params.alpha_a + params.alpha_p + params.alpha_c
    if abs(s - 1.0) > 1e-6:
        warnings.warn(f"Alpha coefficients sum to {s:.4f}, not 1.")
    out = ann.copy()
    R = normalize_01(out["rhythmicity"].to_numpy(dtype=float))
    A = normalize_01(out["amplitude"].to_numpy(dtype=float))
    phase_rad = phase_hour_to_rad(out["phase_hour"].to_numpy(dtype=float) % 24)
    target_rad = float(phase_hour_to_rad(params.target_phase_hour))
    d = circular_distance_rad(phase_rad, target_rad)
    P = np.clip(1.0 - d / np.pi, 0.0, 1.0)
    S = params.alpha_r * R + params.alpha_a * A + params.alpha_p * P + params.alpha_c * (R * A)
    w_star = np.exp(S)
    out["R_01"] = R
    out["A_01"] = A
    out["P_align"] = P
    out["RA_interaction"] = R * A
    out["S_score"] = S
    out["latent_w"] = w_star / np.mean(w_star)
    return out


def compute_observed_weight(latent_w: np.ndarray, params: WeightParams, rng: np.random.Generator) -> np.ndarray:
    log_w = np.log(np.asarray(latent_w, dtype=float) + 1e-12)
    eps = rng.normal(0.0, params.observed_noise_sigma, size=len(log_w))
    w_noisy = np.exp(log_w + eps)
    observed = (1.0 - params.observed_shrink_to_one) * w_noisy + params.observed_shrink_to_one
    return observed / np.mean(observed)


def make_wrong_phase_observed_weight(ann: pd.DataFrame, params: WeightParams, rng: np.random.Generator) -> np.ndarray:
    wrong = WeightParams(0.0, 0.0, 1.0, 0.0, (params.target_phase_hour + 12.0) % 24.0, params.observed_noise_sigma, params.observed_shrink_to_one)
    tmp = compute_strict_latent_weight(ann, wrong)
    return compute_observed_weight(tmp["latent_w"].to_numpy(), wrong, rng)


def bootstrap_synthetic_universe(ann_real: pd.DataFrame, pqtl_real: pd.DataFrame, n_proteins: int, rng: np.random.Generator, keep_real_overlap: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    circ_ids = ann_real["protein_id"].unique()
    pqtl_ids = pqtl_real["protein_id"].unique()
    overlap = sorted(set(circ_ids).intersection(set(pqtl_ids)))
    rows_ann, rows_pq = [], []
    if keep_real_overlap:
        for pid in overlap[:n_proteins]:
            new_id = f"REAL_{pid}"
            a = ann_real.loc[ann_real["protein_id"] == pid].iloc[0].copy(); a["protein_id"] = new_id; rows_ann.append(a)
            p = pqtl_real.loc[pqtl_real["protein_id"] == pid].copy(); p["protein_id"] = new_id; rows_pq.append(p)
    ann_pool = ann_real.reset_index(drop=True)
    pqtl_groups = {pid: g.copy() for pid, g in pqtl_real.groupby("protein_id")}
    pqtl_pool_ids = np.array(list(pqtl_groups.keys()), dtype=object)
    for i in range(len(rows_ann), n_proteins):
        new_id = f"SYNP_{i+1:05d}"
        a = ann_pool.iloc[int(rng.integers(0, len(ann_pool)))].copy(); a["protein_id"] = new_id; rows_ann.append(a)
        template_pid = str(rng.choice(pqtl_pool_ids))
        p = pqtl_groups[template_pid].copy(); p["protein_id"] = new_id; rows_pq.append(p)
    return pd.DataFrame(rows_ann).reset_index(drop=True), pd.concat(rows_pq, ignore_index=True)


def list_ld_files(ld_dir: Path) -> List[Path]:
    files = sorted(ld_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz LD files found in {ld_dir}")
    return files


def load_ld_matrix(ld_file: Path, max_snps: Optional[int] = None) -> np.ndarray:
    """
    Fast LD loader.

    Important optimization:
    For sparse LD npz files, subset BEFORE converting to dense.
    Otherwise a large sparse LD block may be converted to a huge dense matrix
    and the first replicate can take minutes or run out of memory.
    """
    data = np.load(ld_file, allow_pickle=True)
    keys = list(data.keys())
    mat = None

    def _subset_dense(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if max_snps is not None and a.shape[0] > max_snps:
            return a[:max_snps, :max_snps]
        return a

    # Dense matrix keys
    for k in ["ld", "LD", "corr", "Corr", "R", "matrix", "ld_matrix", "arr_0"]:
        if k in data:
            arr = data[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                mat = _subset_dense(arr)
                break

    # scipy sparse CSR .npz: subset before dense conversion
    if mat is None and {"data", "indices", "indptr", "shape"}.issubset(set(keys)):
        try:
            from scipy import sparse
            shape = tuple(data["shape"].tolist())
            csr = sparse.csr_matrix((data["data"], data["indices"], data["indptr"]), shape=shape)
            if max_snps is not None and csr.shape[0] > max_snps:
                csr = csr[:max_snps, :max_snps]
            mat = csr.toarray()
        except Exception as e:
            raise ValueError(
                f"LD file looks like sparse CSR but reconstruction failed: {ld_file}; "
                f"keys={keys}; error={e}"
            )

    # COO-like sparse format: build full sparse but subset before dense
    if mat is None and {"data", "row", "col", "shape"}.issubset(set(keys)):
        try:
            from scipy import sparse
            shape = tuple(data["shape"].tolist())
            coo = sparse.coo_matrix((data["data"], (data["row"], data["col"])), shape=shape).tocsr()
            if max_snps is not None and coo.shape[0] > max_snps:
                coo = coo[:max_snps, :max_snps]
            mat = coo.toarray()
        except Exception as e:
            raise ValueError(
                f"LD file looks like sparse COO but reconstruction failed: {ld_file}; "
                f"keys={keys}; error={e}"
            )

    # fallback first 2D numeric array
    if mat is None:
        for k in keys:
            arr = data[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                mat = _subset_dense(arr)
                break

    if mat is None:
        raise ValueError(
            f"Could not find or reconstruct LD matrix in {ld_file}. "
            f"Available keys={keys}."
        )

    mat = np.asarray(mat, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"LD matrix must be square. Got shape={mat.shape} from {ld_file}")

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 1.0)
    mat = np.clip(mat, -1.0, 1.0)
    mat = mat + np.eye(mat.shape[0]) * 1e-4
    return mat

def simulate_genotype_from_ld(ld: np.ndarray, n_individuals: int, rng: np.random.Generator) -> np.ndarray:
    try:
        L = np.linalg.cholesky(ld)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(ld)
        L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-6, None)))
    X = rng.normal(size=(n_individuals, ld.shape[0])) @ L.T
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


def construct_protein_matrix(X: np.ndarray, pqtl: pd.DataFrame, proteins: Sequence[str], rng: np.random.Generator, protein_noise_sd: float = 0.2) -> np.ndarray:
    n, m = X.shape
    p_index = {p: i for i, p in enumerate(proteins)}
    G = np.zeros((n, len(proteins)), dtype=float)
    for pid, grp in pqtl.groupby("protein_id"):
        if pid not in p_index:
            continue
        j = p_index[pid]
        k = len(grp)
        cols = rng.choice(m, size=k, replace=(k > m))
        betas = grp["beta"].to_numpy(dtype=float)
        betas = betas / (np.sqrt(np.sum(betas ** 2)) + 1e-8)
        G[:, j] = X[:, cols] @ betas
    G = G + rng.normal(0.0, protein_noise_sd, size=G.shape)
    return (G - G.mean(axis=0, keepdims=True)) / (G.std(axis=0, keepdims=True) + 1e-8)


def select_causal_proteins(latent_w: np.ndarray, n_causal: int, gamma_enrichment: float, rng: np.random.Generator, mode: str) -> np.ndarray:
    G = len(latent_w)
    n_causal = min(n_causal, G)
    if mode == "circadian":
        probs = np.asarray(latent_w, dtype=float) ** gamma_enrichment
        probs = probs / probs.sum()
    elif mode == "uniform":
        probs = np.ones(G) / G
    else:
        raise ValueError(mode)
    return rng.choice(G, size=n_causal, replace=False, p=probs)


def simulate_phenotype_from_proteins(Gmat: np.ndarray, causal_idx: np.ndarray, trait_h2: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n, p = Gmat.shape
    gamma = np.zeros(p)
    if len(causal_idx) == 0 or trait_h2 <= 0:
        return zscore_safe(rng.normal(size=n)), gamma
    effects = rng.normal(size=len(causal_idx))
    effects = effects / (np.sqrt(np.sum(effects ** 2)) + 1e-8)
    gamma[causal_idx] = effects
    genetic = zscore_safe(Gmat @ gamma)
    noise = zscore_safe(rng.normal(size=n))
    y = math.sqrt(trait_h2) * genetic + math.sqrt(max(0.0, 1.0 - trait_h2)) * noise
    return zscore_safe(y), gamma


def simulate_phenotype_ld_confounding(X: np.ndarray, trait_h2: float, rng: np.random.Generator, n_causal_snps: int = 20) -> np.ndarray:
    n, m = X.shape
    k = min(n_causal_snps, m)
    cols = rng.choice(m, size=k, replace=False)
    alpha = rng.normal(size=k); alpha = alpha / (np.sqrt(np.sum(alpha ** 2)) + 1e-8)
    genetic = zscore_safe(X[:, cols] @ alpha)
    noise = zscore_safe(rng.normal(size=n))
    return zscore_safe(math.sqrt(trait_h2) * genetic + math.sqrt(max(0.0, 1.0 - trait_h2)) * noise)


def compute_pwas_z(Gmat: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = zscore_safe(y)
    Gs = (Gmat - Gmat.mean(axis=0, keepdims=True)) / (Gmat.std(axis=0, keepdims=True) + 1e-8)
    n = Gmat.shape[0]
    r = (Gs.T @ y) / max(1, n - 1)
    r = np.clip(r, -0.999999, 0.999999)
    return r * np.sqrt(max(1, n - 2)) / np.sqrt(1.0 - r ** 2)


def auc_roc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int); scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_score_simple(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int); scores = np.asarray(scores, dtype=float)
    n_pos = y_true.sum()
    if n_pos == 0:
        return np.nan
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / n_pos)


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray, top_ks: Sequence[int]) -> Dict[str, float]:
    out = {"auc": auc_roc_score(y_true, scores), "pr_auc": average_precision_score_simple(y_true, scores)}
    order = np.argsort(-scores)
    n_pos = int(np.sum(y_true))
    for k in top_ks:
        kk = min(k, len(scores))
        hits = int(np.sum(y_true[order[:kk]]))
        out[f"hits_at_{k}"] = hits
        out[f"precision_at_{k}"] = hits / kk if kk > 0 else np.nan
        out[f"recall_at_{k}"] = hits / n_pos if n_pos > 0 else np.nan
    return out


def run_one_replicate(rep: int, ld_file: Path, ann: pd.DataFrame, pqtl: pd.DataFrame, weight_params: WeightParams, sim_params: SimParams, scenarios: Sequence[str], seed: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed + rep * 100003)
    X = simulate_genotype_from_ld(load_ld_matrix(ld_file, sim_params.max_ld_snps), sim_params.n_individuals, rng)
    proteins = ann["protein_id"].tolist()
    Gmat = construct_protein_matrix(X, pqtl, proteins, rng, sim_params.protein_noise_sd)
    ann_w = compute_strict_latent_weight(ann, weight_params)
    if rep == 1:
        print("\n[DEBUG] latent_w summary")
        print(ann_w["latent_w"].describe())

        print("[DEBUG] latent_w quantiles")
        print(np.quantile(ann_w["latent_w"], [0.01, 0.05, 0.5, 0.95, 0.99]))

        q = np.quantile(ann_w["latent_w"], [0.05, 0.95])
        print("[DEBUG] latent_w 95/5 ratio:", q[1] / q[0])

    latent_w = ann_w["latent_w"].to_numpy(dtype=float)
    observed_w = compute_observed_weight(latent_w, weight_params, rng)
    wrong_observed_w = make_wrong_phase_observed_weight(ann, weight_params, rng)
    corr_log = float(np.corrcoef(np.log(latent_w + 1e-12), np.log(observed_w + 1e-12))[0, 1])
    results = []
    for scenario in scenarios:
        if scenario == "circadian_mediation":
            causal_idx = select_causal_proteins(latent_w, sim_params.n_causal, sim_params.causal_enrichment_gamma, rng, "circadian")
            y, gamma = simulate_phenotype_from_proteins(Gmat, causal_idx, sim_params.trait_h2, rng)
            use_weight = observed_w
        elif scenario == "non_circadian_mediation":
            causal_idx = select_causal_proteins(latent_w, sim_params.n_causal, sim_params.causal_enrichment_gamma, rng, "uniform")
            y, gamma = simulate_phenotype_from_proteins(Gmat, causal_idx, sim_params.trait_h2, rng)
            use_weight = observed_w
        elif scenario == "wrong_phase":
            causal_idx = select_causal_proteins(latent_w, sim_params.n_causal, sim_params.causal_enrichment_gamma, rng, "circadian")
            y, gamma = simulate_phenotype_from_proteins(Gmat, causal_idx, sim_params.trait_h2, rng)
            use_weight = wrong_observed_w
        elif scenario == "ld_confounding":
            gamma = np.zeros(len(proteins)); y = simulate_phenotype_ld_confounding(X, sim_params.trait_h2, rng); use_weight = observed_w
        elif scenario == "null":
            gamma = np.zeros(len(proteins)); y = zscore_safe(rng.normal(size=sim_params.n_individuals)); use_weight = observed_w
        else:
            raise ValueError(scenario)
        z = compute_pwas_z(Gmat, y)
        ordinary_score = np.abs(z)
        weighted_score = np.abs(z * np.sqrt(use_weight))
        truth = (gamma != 0).astype(int)
        om = ranking_metrics(truth, ordinary_score, sim_params.top_ks)
        wm = ranking_metrics(truth, weighted_score, sim_params.top_ks)
        row = {
            "rep": rep, "ld_file": ld_file.name, "scenario": scenario, "n_proteins": len(proteins),
            "n_causal": int(truth.sum()), "trait_h2": sim_params.trait_h2,
            "causal_enrichment_gamma": sim_params.causal_enrichment_gamma,
            "observed_weight_noise_sigma": weight_params.observed_noise_sigma,
            "observed_weight_shrink_to_one": weight_params.observed_shrink_to_one,
            "weight_corr_log_latent_observed": corr_log,
            "alpha_r": weight_params.alpha_r, "alpha_a": weight_params.alpha_a, "alpha_p": weight_params.alpha_p, "alpha_c": weight_params.alpha_c,
            "target_phase_hour": weight_params.target_phase_hour,
        }
        for k, v in om.items(): row[f"ordinary_{k}"] = v
        for k, v in wm.items(): row[f"weighted_{k}"] = v
        row["delta_auc"] = row["weighted_auc"] - row["ordinary_auc"] if np.isfinite(row["ordinary_auc"]) and np.isfinite(row["weighted_auc"]) else np.nan
        row["delta_pr_auc"] = row["weighted_pr_auc"] - row["ordinary_pr_auc"] if np.isfinite(row["ordinary_pr_auc"]) and np.isfinite(row["weighted_pr_auc"]) else np.nan
        results.append(row)
    return results


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "scenario", "n_proteins", "n_causal", "trait_h2", "causal_enrichment_gamma", "observed_weight_noise_sigma", "observed_weight_shrink_to_one", "alpha_r", "alpha_a", "alpha_p", "alpha_c"]
    metric_cols = [c for c in raw.columns if c.startswith("ordinary_") or c.startswith("weighted_") or c.startswith("delta_") or c == "weight_corr_log_latent_observed"]
    rows = []
    for keys, grp in raw.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys)); row["n_reps"] = grp["rep"].nunique()
        for c in metric_cols:
            row[c] = grp[c].mean(skipna=True)
            row[c + "_se"] = grp[c].std(skipna=True) / math.sqrt(max(1, grp[c].notna().sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(name: str, ann_base: pd.DataFrame, pqtl_base: pd.DataFrame, ld_files: List[Path], outdir: Path, weight_params: WeightParams, sim_params: SimParams, scenarios: Sequence[str], n_reps: int, seed: int) -> pd.DataFrame:
    exp_dir = outdir / name; mkdir(exp_dir)
    rng = np.random.default_rng(seed + abs(hash(name)) % 100000)
    ann_syn, pq_syn = bootstrap_synthetic_universe(ann_base, pqtl_base, sim_params.n_proteins, rng)
    ann_syn.to_csv(exp_dir / "circadian_annotation_used.csv", index=False)
    pq_syn.to_csv(exp_dir / "pqtl_used.csv", index=False)
    log(f"\n=== Experiment: {name} ===")
    log(f"proteins={len(ann_syn)}, pQTL links={len(pq_syn)}, reps={n_reps}, scenarios={','.join(scenarios)}")
    all_rows = []
    for r in range(1, n_reps + 1):
        ld_file = ld_files[(r - 1) % len(ld_files)]
        if r == 1 or r % max(1, n_reps // 10) == 0 or r == n_reps:
            log(f"  rep={r}/{n_reps}; LD={ld_file.name}")
        rows = run_one_replicate(r, ld_file, ann_syn, pq_syn, weight_params, sim_params, scenarios, seed + abs(hash(name)) % 1000000)
        for row in rows: row["experiment"] = name
        all_rows.extend(rows)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(exp_dir / "simulation_raw.csv", index=False)
    summarize_results(raw).to_csv(exp_dir / "simulation_summary.csv", index=False)
    return raw


def parse_top_ks(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in str(s).split(",") if str(x).strip())


def main():
    parser = argparse.ArgumentParser(description="Full circadian-informed PWAS semi-simulation suite with strict formula implementation.")
    parser.add_argument("--pg-matrix", required=True, type=Path)
    parser.add_argument("--analyte-info", default=None, type=Path)  # accepted for compatibility, not used
    parser.add_argument("--pqtl", required=True, type=Path)
    parser.add_argument("--ld-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--n-individuals", type=int, default=2000)
    parser.add_argument("--n-proteins", type=int, default=1500)
    parser.add_argument("--n-reps-main", type=int, default=200)
    parser.add_argument("--n-reps-grid", type=int, default=100)
    parser.add_argument("--n-reps-negative", type=int, default=300)
    parser.add_argument("--n-causal", type=int, default=30)
    parser.add_argument("--trait-h2", type=float, default=0.05)
    parser.add_argument("--top-ks", type=str, default="10,20,50")
    parser.add_argument("--max-ld-snps", type=int, default=500)
    parser.add_argument("--protein-noise-sd", type=float, default=0.2)
    parser.add_argument("--alpha-r", type=float, default=0.15)
    parser.add_argument("--alpha-a", type=float, default=0.15)
    parser.add_argument("--alpha-p", type=float, default=0.35)
    parser.add_argument("--alpha-c", type=float, default=0.35)
    parser.add_argument("--target-phase-hour", type=float, default=9.0)
    parser.add_argument("--main-noise-sigma", type=float, default=0.1)
    parser.add_argument("--main-shrink-to-one", type=float, default=0.1)
    parser.add_argument("--main-gamma", type=float, default=3.0)
    parser.add_argument("--blocks", type=str, default="main,prior,gamma,h2,ncausal,nproteins,negative,alpha", help="Comma-separated: main,prior,gamma,h2,ncausal,nproteins,negative,alpha")
    args = parser.parse_args()
    mkdir(args.outdir)

    log("[1/5] Loading/fitting circadian annotation...")
    ann_real = load_or_fit_circadian_annotation(args.pg_matrix, args.outdir)
    log(f"  circadian proteins: {len(ann_real)}")

    log("[2/5] Loading pQTL top-k...")
    pqtl_real = load_pqtl_topk(args.pqtl, args.outdir)
    log(f"  pQTL proteins: {pqtl_real['protein_id'].nunique()}, links: {len(pqtl_real)}")
    log(f"  real overlap proteins: {len(set(ann_real['protein_id']).intersection(set(pqtl_real['protein_id'])))}")

    log("[3/5] Listing LD files...")
    ld_files = list_ld_files(args.ld_dir)
    log(f"  found {len(ld_files)} LD files")

    blocks = set(x.strip().lower() for x in args.blocks.split(",") if x.strip())
    all_raw = []
    base_wp = WeightParams(args.alpha_r, args.alpha_a, args.alpha_p, args.alpha_c, args.target_phase_hour, args.main_noise_sigma, args.main_shrink_to_one)
    base_sp = SimParams(args.n_individuals, args.n_proteins, args.n_causal, args.trait_h2, args.main_gamma, args.protein_noise_sd, parse_top_ks(args.top_ks), args.max_ld_snps)

    log("[4/5] Running experiments...")
    if "main" in blocks:
        all_raw.append(run_experiment("A_main_realistic", ann_real, pqtl_real, ld_files, args.outdir, base_wp, base_sp, ["circadian_mediation", "non_circadian_mediation", "wrong_phase", "ld_confounding", "null"], args.n_reps_main, args.seed))
    if "prior" in blocks:
        for noise, shrink in [(0.75, 0.4), (1.00, 0.4), (1.25, 0.4), (1.50, 0.5), (1.75, 0.5), (2.00, 0.55), (2.25, 0.6), (2.50, 0.65)]:
            wp = WeightParams(args.alpha_r, args.alpha_a, args.alpha_p, args.alpha_c, args.target_phase_hour, noise, shrink)
            all_raw.append(run_experiment(f"B_prior_noise_{noise}_shrink_{shrink}", ann_real, pqtl_real, ld_files, args.outdir, wp, base_sp, ["circadian_mediation", "non_circadian_mediation", "wrong_phase"], args.n_reps_grid, args.seed))
    if "gamma" in blocks:
        for gamma in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            sp = SimParams(**{**base_sp.__dict__, "causal_enrichment_gamma": gamma})
            all_raw.append(run_experiment(f"C_enrichment_gamma_{gamma}", ann_real, pqtl_real, ld_files, args.outdir, base_wp, sp, ["circadian_mediation"], args.n_reps_grid, args.seed))
    if "h2" in blocks:
        for h2 in [0.01, 0.02, 0.05, 0.10, 0.20]:
            sp = SimParams(**{**base_sp.__dict__, "trait_h2": h2})
            all_raw.append(run_experiment(f"D_h2_{h2}", ann_real, pqtl_real, ld_files, args.outdir, base_wp, sp, ["circadian_mediation", "non_circadian_mediation", "wrong_phase", "ld_confounding"], args.n_reps_grid, args.seed))
    if "ncausal" in blocks:
        for nc in [5, 10, 20, 30, 50, 100]:
            sp = SimParams(**{**base_sp.__dict__, "n_causal": nc})
            all_raw.append(run_experiment(f"E_ncausal_{nc}", ann_real, pqtl_real, ld_files, args.outdir, base_wp, sp, ["circadian_mediation", "non_circadian_mediation", "wrong_phase"], args.n_reps_grid, args.seed))
    if "nproteins" in blocks:
        for np_ in [500, 1000, 1500, 3000, 5000]:
            sp = SimParams(**{**base_sp.__dict__, "n_proteins": np_})
            all_raw.append(run_experiment(f"F_nproteins_{np_}", ann_real, pqtl_real, ld_files, args.outdir, base_wp, sp, ["circadian_mediation", "non_circadian_mediation", "wrong_phase", "ld_confounding"], args.n_reps_grid, args.seed))
    if "negative" in blocks:
        all_raw.append(run_experiment("H_negative_controls", ann_real, pqtl_real, ld_files, args.outdir, base_wp, base_sp, ["null", "ld_confounding", "non_circadian_mediation", "wrong_phase"], args.n_reps_negative, args.seed))
    if "alpha" in blocks:
        for ar in [0.10, 0.25, 0.40]:
            for aa in [0.10, 0.25, 0.40]:
                for ap in [0.10, 0.25, 0.40]:
                    ac = 1.0 - ar - aa - ap
                    if ac < 0: continue
                    wp = WeightParams(ar, aa, ap, ac, args.target_phase_hour, args.main_noise_sigma, args.main_shrink_to_one)
                    all_raw.append(run_experiment(f"J_alpha_r{ar}_a{aa}_p{ap}_c{round(ac, 3)}", ann_real, pqtl_real, ld_files, args.outdir, wp, base_sp, ["circadian_mediation", "wrong_phase", "non_circadian_mediation"], max(50, args.n_reps_grid // 2), args.seed))

    log("[5/5] Writing combined outputs...")
    if all_raw:
        combined = pd.concat(all_raw, ignore_index=True)
        combined.to_csv(args.outdir / "all_simulation_raw.csv", index=False)
        summary = summarize_results(combined)
        summary.to_csv(args.outdir / "all_simulation_summary.csv", index=False)
        keep = ["experiment", "scenario", "n_reps", "n_proteins", "n_causal", "trait_h2", "causal_enrichment_gamma", "observed_weight_noise_sigma", "observed_weight_shrink_to_one", "weight_corr_log_latent_observed", "ordinary_auc", "weighted_auc", "delta_auc", "ordinary_pr_auc", "weighted_pr_auc", "delta_pr_auc"]
        keep = [c for c in keep if c in summary.columns]
        summary[keep].to_csv(args.outdir / "all_simulation_summary_compact.csv", index=False)
        log(f"  wrote {args.outdir / 'all_simulation_raw.csv'}")
        log(f"  wrote {args.outdir / 'all_simulation_summary.csv'}")
        log(f"  wrote {args.outdir / 'all_simulation_summary_compact.csv'}")
    log("Done.")


if __name__ == "__main__":
    main()
