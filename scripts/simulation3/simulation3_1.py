#!/usr/bin/env python3
# =============================================================================
# circadian_pwas_realdata.py
#
# Circadian-informed PWAS evaluation driven by YOUR real data:
#   * UKB LD matrix            -> realistic genotype correlation structure
#   * UKB allele frequencies   -> realistic minor-allele frequencies
#   * UKB-PPP pQTL (beta)       -> realistic SNP -> protein effects
#   * circadian protein-time   -> REAL rhythmicity prior (via MetaCycle)
#
# What is real vs synthetic (read this before interpreting anything):
#   REAL      : LD, MAF, pQTL beta, protein rhythmicity (cosinor R^2).
#   SYNTHETIC : individual genotypes (drawn from real LD+MAF, because UKB does
#               not release individual genotypes), the protein levels they imply
#               (P = G*beta + e), and the ground-truth causal proteins + trait
#               (the labels we need to *score* the method against).
#   => The method is tested on real genetic + rhythm structure, with synthetic
#      ground truth.  This is a power/specificity check, NOT a biological result.
#
# Pipeline (unchanged from the spec):
#   1. G  = Z * chol(LD)^T      (+ optional HWE discretization via MAF)
#   2. P  = G * beta + e        (per-protein heritability scaled to h2_pqtl)
#   3. Z  = pwas(P, y)          (time-agnostic PWAS on protein level)
#   4. w  = exp(tau * z(R2)) , mean(w)=1 ;  score = |Z| * sqrt(w)
#   core indicator: Delta AUC = AUC(weighted) - AUC(vanilla)
#
# RUN:
#   python3 circadian_pwas_realdata.py \
#       --protein-time report_pg_matrix.tsv \
#       --ld ukb_ld.npy --ld-snps ukb_ld_snps.txt \
#       --maf ukb_maf.tsv \
#       --pqtl ukb_ppp_pqtl.tsv \
#       --out results_real
#
# Each loader documents the exact file format it expects. Adapt the column
# names in DataSpec to match your files; nothing else needs to change.
# =============================================================================

import os
import re
import argparse
import subprocess
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RSCRIPT = os.path.join(HERE, "run_metacycle.R")


# =============================================================================
# DataSpec -- EDIT the column names here to match your files
# =============================================================================
class DataSpec:
    # ---- protein-time matrix (default = report_pg_matrix.tsv layout) ----
    pt_sep            = "\t"
    pt_gene_col       = "Genes"          # protein/gene identifier column
    # measurement columns are auto-detected by containing a "t<hour>" token;
    # the circadian hour is parsed from that token and replicates are averaged.
    pt_time_regex     = r"t(\d+)"        # captures the circadian hour
    pt_period         = 24.0

    # ---- LD ----
    # LD file: a square correlation matrix (M x M). Supported: .npy, .npz
    # (key 'ld' or first array), or .csv/.tsv (no index). LD must correspond,
    # row/col-for-row/col, to the variant ids in --ld-snps (one id per line).

    # ---- MAF ----
    maf_sep           = "\t"
    maf_variant_col   = "variant"        # variant id, must match LD snp ids
    maf_value_col     = "maf"            # minor allele frequency in [0,0.5]

    # ---- pQTL (UKB-PPP) ----
    # A long table: one row per (protein, variant) pQTL with an effect size.
    pqtl_sep          = "\t"
    pqtl_protein_col  = "gene"           # protein id, must match pt_gene_col vals
    pqtl_variant_col  = "variant"        # variant id, must match LD snp ids
    pqtl_beta_col     = "beta"           # effect size (per-allele)


# =============================================================================
# Config -- simulation / evaluation knobs (NOT data)
# =============================================================================
class Config:
    N                 = 2000     # synthetic individuals drawn from real LD+MAF
    n_causal          = 30       # ground-truth causal proteins
    h2_pqtl           = 0.10     # per-protein heritability of the genetic value
    h2_trait          = 0.30     # trait heritability through proteins
    discretize_geno   = True     # HWE 0/1/2 via MAF (False = standardized dosage)

    tau               = 1.0      # weight sharpness; mean(w)=1 always
    prior_lognoise_sd = 0.15     # analysis-track noise on the prior (anti-circularity)
    select_lambda     = 3.0      # causal enrichment strength on real rhythmicity

    n_contrast_pairs  = 40       # scenario-B matched (mean-only vs more-rhythmic) pairs
    contrast_amp_pctl = (25, 90) # pick P1 from low-R2, P2 from high-R2 proteins

    n_reps            = 200
    seed              = 12345
    use_metacycle     = True


# =============================================================================
# Section A -- DATA LOADERS  (PLUG YOUR DATA HERE)
# =============================================================================
def load_protein_time(path, spec=DataSpec):
    """Read the circadian protein-time matrix and collapse to one population
    profile per protein over the unique circadian time points.

    Expected (report_pg_matrix.tsv) layout:
      - a gene/protein id column (spec.pt_gene_col)
      - many measurement columns whose header contains a 't<hour>' token, one
        per (subject, day, timepoint).  Replicates sharing the same hour are
        averaged (across subjects and days) into the population profile.

    Returns
    -------
    gene_ids   : (P,) array of protein/gene ids
    profiles   : (P, T) population-mean expression per unique circadian hour
    timepoints : (T,) sorted unique circadian hours
    """
    df = pd.read_csv(path, sep=spec.pt_sep)
    assert spec.pt_gene_col in df.columns, \
        f"gene column '{spec.pt_gene_col}' not found; columns={list(df.columns)[:6]}..."
    gene_ids = df[spec.pt_gene_col].astype(str).to_numpy()

    # find measurement columns and their circadian hour
    rgx = re.compile(spec.pt_time_regex)
    col_hour = {}
    for c in df.columns:
        m = rgx.search(str(c))
        if m and c != spec.pt_gene_col:
            # only treat as measurement if the cell values are numeric
            if pd.api.types.is_numeric_dtype(df[c]) or \
               pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
                col_hour[c] = int(m.group(1)) % int(spec.pt_period)
    assert col_hour, "no measurement columns with a t<hour> token were detected"

    hours = sorted(set(col_hour.values()))
    M = df[list(col_hour.keys())].apply(pd.to_numeric, errors="coerce")
    # log-transform intensities (proteomics is multiplicative); guard non-positive
    M = np.log2(M.clip(lower=1))
    prof = np.zeros((len(df), len(hours)))
    for j, h in enumerate(hours):
        cols = [c for c, hh in col_hour.items() if hh == h]
        prof[:, j] = M[cols].mean(axis=1).to_numpy()
    # drop proteins that are all-NaN
    ok = np.isfinite(prof).all(1)
    return gene_ids[ok], prof[ok], np.array(hours, float)


def load_ld(path, snps_path):
    """Load a square LD correlation matrix and its variant id list.

    LD file : .npy / .npz / .csv / .tsv -- an (M x M) matrix.
    snps    : text file, one variant id per line, in LD row/col order.
    """
    if path.endswith(".npy"):
        ld = np.load(path)
    elif path.endswith(".npz"):
        z = np.load(path); ld = z["ld"] if "ld" in z else z[list(z.keys())[0]]
    else:
        ld = pd.read_csv(path, sep=None, engine="python", header=None).to_numpy()
    with open(snps_path) as f:
        snps = np.array([ln.strip() for ln in f if ln.strip()])
    assert ld.shape[0] == ld.shape[1] == len(snps), \
        f"LD shape {ld.shape} inconsistent with {len(snps)} snp ids"
    return ld.astype(float), snps


def load_maf(path, spec=DataSpec):
    df = pd.read_csv(path, sep=spec.maf_sep)
    maf = df.set_index(df[spec.maf_variant_col].astype(str))[spec.maf_value_col]
    maf = maf.astype(float).clip(1e-4, 0.5)
    return maf  # pandas Series indexed by variant id


def load_pqtl(path, spec=DataSpec):
    df = pd.read_csv(path, sep=spec.pqtl_sep)
    for col in (spec.pqtl_protein_col, spec.pqtl_variant_col, spec.pqtl_beta_col):
        assert col in df.columns, f"pQTL column '{col}' missing"
    out = df[[spec.pqtl_protein_col, spec.pqtl_variant_col, spec.pqtl_beta_col]].copy()
    out.columns = ["protein", "variant", "beta"]
    out["protein"] = out["protein"].astype(str)
    out["variant"] = out["variant"].astype(str)
    out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
    return out.dropna()


# =============================================================================
# Section B -- HARMONIZE the three sources onto a common protein/SNP space
# =============================================================================
def harmonize(gene_ids, profiles, ld, ld_snps, maf, pqtl, verbose=True):
    """Intersect proteins (protein-time ∩ pQTL) and SNPs (LD ∩ pQTL ∩ MAF),
    then build aligned objects: LD (M'xM'), MAF (M'), beta (M'xP'), rhythm
    profiles (P'xT)."""
    # proteins present in BOTH the rhythm data and the pQTL table
    pt_set = set(gene_ids)
    pq_prot = set(pqtl["protein"]) & pt_set
    proteins = np.array(sorted(pq_prot))
    assert len(proteins) > 0, "no proteins shared between protein-time and pQTL"
    pidx = {g: i for i, g in enumerate(proteins)}

    # SNPs present in LD AND MAF AND used by the retained proteins' pQTL
    ld_idx = {s: i for i, s in enumerate(ld_snps)}
    maf_set = set(maf.index)
    pqtl_f = pqtl[pqtl["protein"].isin(pq_prot)]
    snp_set = (set(pqtl_f["variant"]) & set(ld_snps) & maf_set)
    snps = np.array([s for s in ld_snps if s in snp_set])  # keep LD order
    assert len(snps) > 0, "no SNPs shared between LD, MAF and pQTL"
    sidx = {s: i for i, s in enumerate(snps)}

    # subset LD and MAF
    keep = np.array([ld_idx[s] for s in snps])
    LD = ld[np.ix_(keep, keep)]
    MAF = maf.loc[snps].to_numpy()

    # build sparse beta (M' x P')
    beta = np.zeros((len(snps), len(proteins)))
    pqtl_use = pqtl_f[pqtl_f["variant"].isin(snp_set)]
    for prot, var, b in pqtl_use[["protein", "variant", "beta"]].itertuples(index=False):
        beta[sidx[var], pidx[prot]] += b

    # align rhythm profiles to the protein order
    g2row = {g: i for i, g in enumerate(gene_ids)}
    prof = np.vstack([profiles[g2row[g]] for g in proteins])

    if verbose:
        print(f"  harmonized: {len(proteins)} proteins, {len(snps)} SNPs")
        nz = (beta != 0).sum(0)
        print(f"  pQTL per protein: median {int(np.median(nz))}, "
              f"max {int(nz.max())}, proteins with >=1 pQTL: {(nz>0).sum()}")
    return dict(proteins=proteins, snps=snps, LD=LD, MAF=MAF, beta=beta, profiles=prof)


# =============================================================================
# Section C -- genotypes from REAL LD + MAF
# =============================================================================
def nearest_psd(A, eps=1e-6):
    """Project a symmetric matrix to the nearest PSD correlation-ish matrix."""
    A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    B = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(B), eps, None))
    return B / np.outer(d, d)


def genotypes_from_ld(cfg, LD, MAF, rng):
    """G = Z * chol(LD)^T  (latent), then optional HWE 0/1/2 via MAF.
    Returns standardized genotype dosages (N x M)."""
    M = LD.shape[0]
    LDp = nearest_psd(LD)
    L = np.linalg.cholesky(LDp + 1e-8 * np.eye(M))
    U = rng.standard_normal((cfg.N, M)) @ L.T            # correlated latent normals
    if cfg.discretize_geno:
        from scipy.stats import norm
        q0 = (1 - MAF) ** 2
        q1 = q0 + 2 * MAF * (1 - MAF)
        c1 = norm.ppf(np.clip(q0, 1e-6, 1 - 1e-6))
        c2 = norm.ppf(np.clip(q1, 1e-6, 1 - 1e-6))
        G = (U > c1).astype(float) + (U > c2).astype(float)  # 0/1/2, HWE marginals
    else:
        G = U
    G = (G - G.mean(0)) / (G.std(0) + 1e-9)
    return G


# =============================================================================
# Section D -- protein levels from REAL beta  (+ scenario-B matched pairs)
# =============================================================================
def protein_levels(cfg, G, beta, rng):
    """P = G*beta + e, each protein scaled to h2_pqtl. Returns standardized P
    and the genetic-value matrix gv (used to inject scenario-B pairs)."""
    gv = G @ beta                                          # N x P
    sd = gv.std(0)
    has_signal = sd > 1e-8
    gv[:, has_signal] /= sd[has_signal]                    # var(gv)=1 where defined
    # scale to target heritability and add environmental residual
    e = rng.standard_normal(gv.shape)
    P = np.sqrt(cfg.h2_pqtl) * gv + np.sqrt(1 - cfg.h2_pqtl) * e
    P = (P - P.mean(0)) / (P.std(0) + 1e-9)
    return P, gv, has_signal


def inject_contrast_pairs(cfg, P, gv, has_signal, r2, rng):
    """Create matched (P1, P2) protein pairs for scenario B:
       P1 = low-rhythmicity protein, P2 = high-rhythmicity protein, and we OVERWRITE
       both protein-level columns with the SAME genetic value so their PWAS Z ties
       exactly. The prior (real R^2) then separates them. Returns list of (p1,p2)."""
    Pn = P.shape[1]
    lo = np.percentile(r2, cfg.contrast_amp_pctl[0])
    hi = np.percentile(r2, cfg.contrast_amp_pctl[1])
    low_pool = [p for p in range(Pn) if r2[p] <= lo and has_signal[p]]
    high_pool = [p for p in range(Pn) if r2[p] >= hi and has_signal[p]]
    rng.shuffle(low_pool); rng.shuffle(high_pool)
    n = min(cfg.n_contrast_pairs, len(low_pool), len(high_pool))
    pairs = []
    for k in range(n):
        p1, p2 = low_pool[k], high_pool[k]
        shared = gv[:, p2].copy()                          # share a genetic value
        col = (np.sqrt(cfg.h2_pqtl) * shared +
               np.sqrt(1 - cfg.h2_pqtl) * rng.standard_normal(P.shape[0]))
        col = (col - col.mean()) / (col.std() + 1e-9)
        P[:, p1] = col
        P[:, p2] = col                                     # identical -> exact tie
        pairs.append((p1, p2))
    return pairs


# =============================================================================
# Section E -- REAL rhythmicity prior (MetaCycle on the real protein-time data)
# =============================================================================
_METACYCLE_OK = None
def _have_metacycle():
    global _METACYCLE_OK
    if _METACYCLE_OK is None:
        try:
            r = subprocess.run(["Rscript", "-e",
                                "suppressMessages(library(MetaCycle))"],
                               capture_output=True, timeout=60)
            _METACYCLE_OK = (r.returncode == 0)
        except Exception:
            _METACYCLE_OK = False
    return _METACYCLE_OK


def rhythmicity(cfg, profiles, timepoints, tag="real"):
    """Return cosinor R^2 per protein from the REAL profiles. Uses MetaCycle if
    available, else a least-squares cosinor fallback."""
    if cfg.use_metacycle and os.path.exists(RSCRIPT) and _have_metacycle():
        try:
            return _metacycle(profiles, timepoints, tag)
        except Exception as e:
            print(f"  [MetaCycle failed -> python cosinor fallback: {e}]")
    return _cosinor_r2(profiles, timepoints, cfg)


def _metacycle(profiles, timepoints, tag):
    tpstr = ",".join(str(int(x)) for x in timepoints)
    with tempfile.TemporaryDirectory() as d:
        infile = os.path.join(d, f"in_{tag}.csv")
        outfile = os.path.join(d, f"out_{tag}.csv")
        ids = [f"P{i}" for i in range(profiles.shape[0])]
        cols = [f"t{int(x)}" for x in timepoints]
        pd.DataFrame(profiles, index=ids, columns=cols).reset_index().rename(
            columns={"index": "id"}).to_csv(infile, index=False)
        subprocess.run(["Rscript", RSCRIPT, infile, outfile, tpstr],
                       capture_output=True, check=True, timeout=3600)
        out = pd.read_csv(outfile)
    order = {f"P{i}": i for i in range(profiles.shape[0])}
    out = out.assign(_o=out["id"].map(order)).sort_values("_o")
    return out["R2"].to_numpy()


def _cosinor_r2(profiles, timepoints, cfg):
    t = np.asarray(timepoints, float); omega = 2 * np.pi / DataSpec.pt_period
    X = np.column_stack([np.ones_like(t), np.cos(omega * t), np.sin(omega * t)])
    H = X @ np.linalg.pinv(X)
    r2 = np.empty(profiles.shape[0])
    for i, y in enumerate(profiles):
        fit = H @ y
        sst = np.sum((y - y.mean()) ** 2)
        r2[i] = 0.0 if sst <= 0 else max(0.0, 1 - np.sum((y - fit) ** 2) / sst)
    return r2


# =============================================================================
# Section F -- method, scenarios, Delta AUC, scenario-B contrast
# =============================================================================
def make_weight(r2, tau, invert=False):
    z = (r2 - r2.mean()) / (r2.std() + 1e-9)
    w = np.exp(tau * (-z if invert else z))
    return w / w.mean()


def pwas_z(P, y):
    N = P.shape[0]
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-9)
    yz = (y - y.mean()) / (y.std() + 1e-9)
    r = np.clip((Pz * yz[:, None]).mean(0), -0.999, 0.999)
    return r * np.sqrt(N - 2) / np.sqrt(1 - r ** 2)


def select_causal(cfg, P, r2, scenario, rng, has_signal):
    idx = np.where(has_signal)[0]
    if scenario in ("circadian_mediation", "wrong_phase"):
        z = (r2[idx] - r2[idx].mean()) / (r2[idx].std() + 1e-9)
        prob = np.exp(cfg.select_lambda * z); prob /= prob.sum()
        return rng.choice(idx, size=min(cfg.n_causal, len(idx)), replace=False, p=prob)
    return rng.choice(idx, size=min(cfg.n_causal, len(idx)), replace=False)


def make_phenotype(cfg, P, r2, scenario, rng, has_signal):
    causal = select_causal(cfg, P, r2, scenario, rng, has_signal)
    if scenario == "null":
        return rng.standard_normal(P.shape[0]), causal
    gamma = rng.normal(0, 1, len(causal))
    sig = (P[:, causal] * gamma).sum(1)
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    y = np.sqrt(cfg.h2_trait) * sig + np.sqrt(1 - cfg.h2_trait) * rng.standard_normal(P.shape[0])
    return y, causal


def delta_auc(Z, obs_r2, causal, P, invert=False, tau=1.0):
    labels = np.zeros(P); labels[causal] = 1
    w = make_weight(obs_r2, tau, invert=invert)
    auc_v = roc_auc_score(labels, np.abs(Z))
    auc_w = roc_auc_score(labels, np.abs(Z) * np.sqrt(w))
    return auc_w - auc_v, auc_v, auc_w


def scenario_b(cfg, Z, obs_r2, pairs):
    w = make_weight(obs_r2, cfg.tau)
    van = np.abs(Z); wtd = np.abs(Z) * np.sqrt(w)
    ties = sum(abs(van[a] - van[b]) < 1e-6 for a, b in pairs)
    p2hi = sum(wtd[b] > wtd[a] for a, b in pairs)
    n = max(1, len(pairs))
    return dict(n_pairs=len(pairs), vanilla_tie_rate=ties / n,
                weighted_p2_over_p1_rate=p2hi / n,
                mean_weighted_gap=float(np.mean([wtd[b] - wtd[a] for a, b in pairs])))


# =============================================================================
# Driver
# =============================================================================
def run(cfg, data, scenarios=("circadian_mediation", "non_circadian_mediation",
                              "wrong_phase", "null"), verbose=True):
    rng0 = np.random.default_rng(cfg.seed)
    LD, MAF, beta, profiles = data["LD"], data["MAF"], data["beta"], data["profiles"]

    if verbose: print("Generating genotypes from real LD + MAF ...")
    G = genotypes_from_ld(cfg, LD, MAF, rng0)
    if verbose: print("Building protein levels from real pQTL beta ...")
    P, gv, has_signal = protein_levels(cfg, G, beta, rng0)

    if verbose: print("Computing REAL rhythmicity prior (MetaCycle) ...")
    r2 = rhythmicity(cfg, profiles, data["timepoints"])

    pairs = inject_contrast_pairs(cfg, P, gv, has_signal, r2, rng0)
    if verbose:
        print(f"  scenario-B pairs injected: {len(pairs)} "
              f"(P1 R2~{np.percentile(r2,cfg.contrast_amp_pctl[0]):.2f}, "
              f"P2 R2~{np.percentile(r2,cfg.contrast_amp_pctl[1]):.2f})")

    rows, cacc = [], []
    Pn = P.shape[1]
    for sc in scenarios:
        for rep in range(cfg.n_reps):
            rng = np.random.default_rng(cfg.seed + 1000 * rep + hash(sc) % 997)
            y, causal = make_phenotype(cfg, P, r2, sc, rng, has_signal)
            Z = pwas_z(P, y)
            obs = np.exp(np.log(r2 + 1e-3) +
                         rng.normal(0, cfg.prior_lognoise_sd, Pn)) - 1e-3
            obs = np.clip(obs, 0, None)
            d, av, aw = delta_auc(Z, obs, causal, Pn,
                                  invert=(sc == "wrong_phase"), tau=cfg.tau)
            rows.append((sc, rep, d, av, aw))
            if sc == "circadian_mediation":
                cacc.append(scenario_b(cfg, Z, obs, pairs))

    df = pd.DataFrame(rows, columns=["scenario", "rep", "delta_auc",
                                     "auc_vanilla", "auc_weighted"])
    summ = (df.groupby("scenario")["delta_auc"]
              .agg(n="count", mean_delta_auc="mean", sd="std").reset_index())
    summ["se"] = summ["sd"] / np.sqrt(summ["n"])
    contrast = {k: float(np.mean([c[k] for c in cacc]))
                for k in ("vanilla_tie_rate", "weighted_p2_over_p1_rate",
                          "mean_weighted_gap")} if cacc else {}
    return dict(df=df, summary=summ, contrast=contrast, r2=r2, pairs=pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protein-time", required=True, help="protein-time matrix (tsv)")
    ap.add_argument("--ld", required=True, help="LD matrix (.npy/.npz/.csv)")
    ap.add_argument("--ld-snps", required=True, help="variant ids, one per line, LD order")
    ap.add_argument("--maf", required=True, help="MAF table (variant, maf)")
    ap.add_argument("--pqtl", required=True, help="pQTL table (gene, variant, beta)")
    ap.add_argument("--out", default=os.path.join(HERE, "results_real"))
    ap.add_argument("--no-metacycle", action="store_true")
    ap.add_argument("--no-discretize", action="store_true",
                    help="use standardized dosages instead of HWE 0/1/2")
    args = ap.parse_args()

    cfg = Config()
    if args.no_metacycle:  cfg.use_metacycle = False
    if args.no_discretize: cfg.discretize_geno = False
    os.makedirs(args.out, exist_ok=True)

    print("Loading data ...")
    gene_ids, profiles, tps = load_protein_time(args.protein_time)
    print(f"  protein-time: {len(gene_ids)} proteins, timepoints {list(tps.astype(int))}")
    ld, ld_snps = load_ld(args.ld, args.ld_snps)
    print(f"  LD: {ld.shape}")
    maf = load_maf(args.maf)
    pqtl = load_pqtl(args.pqtl)
    print(f"  pQTL rows: {len(pqtl)}")

    data = harmonize(gene_ids, profiles, ld, ld_snps, maf, pqtl)
    data["timepoints"] = tps

    res = run(cfg, data)

    print("\n================ Delta AUC by scenario ================")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(res["summary"].to_string(index=False))
    if res["contrast"]:
        c = res["contrast"]
        print("\n========== Scenario-B controlled contrast ==========")
        print(f"  vanilla PWAS tie rate            : {c['vanilla_tie_rate']:.3f}")
        print(f"  weighted ranks P2 (rhythm) > P1  : {c['weighted_p2_over_p1_rate']:.3f}")
        print(f"  mean weighted-score gap          : {c['mean_weighted_gap']:+.4f}")

    res["df"].to_csv(os.path.join(args.out, "delta_auc_replicates.csv"), index=False)
    res["summary"].to_csv(os.path.join(args.out, "scenario_summary.csv"), index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
