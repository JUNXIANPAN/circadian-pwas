#!/usr/bin/env python3
"""
simulation3_2.py

Pure-simulation demonstration that circadian-informed PWAS outperforms
traditional PWAS when causal proteins are rhythmic.

Core idea
---------
Traditional PWAS ranks proteins by |Z| (correlation with phenotype).
The method here re-ranks by |Z| * sqrt(w), where w ∝ exp(R²) and R² is
the MetaCycle rhythmicity score.

This simulation proves the method works by construction:

  1. Protein PROFILES  (for MetaCycle) are simulated with controlled 24-hour
     cosine amplitude.  The MESOR (mean level) is drawn independently from
     amplitude, so two proteins can share the same average level while differing
     completely in rhythmicity.  This is the key biological assumption: circadian
     variation does not necessarily change the population mean.

  2. Protein LEVELS  (for PWAS) are simulated via a simple genetic model
     (P = G*beta + noise) and are INDEPENDENT of the rhythmic profiles.
     This represents measuring proteins at a single/random time point.

  3. Matched pairs  are injected: two proteins are forced to have IDENTICAL
     protein-level columns (→ identical vanilla Z-scores) but very different R².
     The weighted method has no choice but to use R² to break the tie.

Scenarios
---------
  circadian : causal proteins are enriched for high R²
              → weighting improves AUC (the method works)
  random    : causal proteins are randomly selected
              → Delta AUC ≈ 0  (weighting is neutral, no harm)
  null      : trait is pure noise
              → both methods at AUC ≈ 0.5  (sanity check)

Run
---
  python3 simulation3_2.py [--out results_sim] [--reps 200]
  python3 simulation3_2.py --no-metacycle   # use Python cosinor fallback
"""

import os
import argparse
import subprocess
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE    = os.path.dirname(os.path.abspath(__file__))
RSCRIPT = os.path.join(HERE, "run_metacycle.R")


# =============================================================================
# Config
# =============================================================================
class Config:
    # population / proteins / SNPs
    N               = 500    # simulated individuals (for PWAS)
    P_total         = 300    # total proteins
    M               = 200    # SNPs
    k_pqtl          = 5      # pQTL SNPs randomly assigned per protein

    # genetic model
    h2_pqtl         = 0.15   # heritability of protein level
    h2_trait        = 0.30   # heritability of trait through causal proteins
    n_causal        = 20     # number of ground-truth causal proteins

    # circadian profile simulation
    timepoints      = list(range(0, 24, 2))  # 0,2,4,...,22  (12 points)
    amp_max         = 2.0    # maximum cosine amplitude
    mesor_mean      = 10.0   # mean baseline expression
    mesor_sd        = 0.5    # protein-to-protein baseline variation (independent of amp)
    profile_noise   = 0.3    # measurement noise on the simulated profile

    # weighting
    tau             = 1.0    # weight sharpness; higher → more extreme weights
    prior_noise_sd  = 0.10   # small jitter on R² before using as prior (anti-circularity)

    # matched pairs (scenario B)
    n_pairs         = 40
    low_r2_pct      = 20     # percentile ceiling for "low rhythmicity" pool
    high_r2_pct     = 80     # percentile floor for "high rhythmicity" pool

    # simulation
    n_reps          = 200
    seed            = 42
    select_lambda   = 3.0    # enrichment strength: how much more likely are
                             # rhythmic proteins to be causal (circadian scenario)
    use_metacycle   = True


# =============================================================================
# Step 1 — Simulate protein circadian profiles
# =============================================================================
def simulate_profiles(cfg, rng):
    """
    Each protein gets a random amplitude (how rhythmic it is) and a random phase
    (peak time).  The MESOR (baseline mean) is drawn INDEPENDENTLY of amplitude
    so that rhythmicity and mean level are uncorrelated — this prevents vanilla
    PWAS from accidentally using the mean to detect rhythm.
    """
    T   = len(cfg.timepoints)
    t   = np.array(cfg.timepoints, float)
    P   = cfg.P_total

    amplitudes = rng.uniform(0.0, cfg.amp_max, P)          # rhythm strength
    phases     = rng.uniform(0.0, 24.0, P)                 # peak hour
    mesors     = rng.normal(cfg.mesor_mean, cfg.mesor_sd, P)  # mean level (independent)

    profiles = np.zeros((P, T))
    for i in range(P):
        profiles[i] = (mesors[i]
                       + amplitudes[i] * np.cos(2 * np.pi * (t - phases[i]) / 24.0))
    profiles += rng.normal(0.0, cfg.profile_noise, profiles.shape)

    return profiles, amplitudes, phases, mesors


# =============================================================================
# Step 2 — Rhythmicity scoring via MetaCycle (or Python cosinor fallback)
# =============================================================================
def _have_metacycle():
    try:
        r = subprocess.run(
            ["Rscript", "-e", "suppressMessages(library(MetaCycle))"],
            capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def _run_metacycle(profiles, timepoints):
    tpstr = ",".join(str(int(x)) for x in timepoints)
    with tempfile.TemporaryDirectory() as d:
        infile  = os.path.join(d, "profiles.csv")
        outfile = os.path.join(d, "rhythm.csv")
        ids  = [f"P{i}" for i in range(profiles.shape[0])]
        cols = [f"t{int(x)}" for x in timepoints]
        pd.DataFrame(profiles, index=ids, columns=cols) \
          .reset_index().rename(columns={"index": "id"}) \
          .to_csv(infile, index=False)
        subprocess.run(
            ["Rscript", RSCRIPT, infile, outfile, tpstr],
            capture_output=True, check=True, timeout=3600)
        out = pd.read_csv(outfile)
    order = {f"P{i}": i for i in range(profiles.shape[0])}
    out = out.assign(_o=out["id"].map(order)).sort_values("_o")
    return out["R2"].to_numpy()


def _cosinor_r2(profiles, timepoints):
    """Least-squares 24h cosinor R² — fallback when MetaCycle is unavailable."""
    t     = np.array(timepoints, float)
    omega = 2 * np.pi / 24.0
    X     = np.column_stack([np.ones_like(t),
                              np.cos(omega * t),
                              np.sin(omega * t)])
    H  = X @ np.linalg.pinv(X)
    r2 = np.empty(profiles.shape[0])
    for i, y in enumerate(profiles):
        fit = H @ y
        sst = np.sum((y - y.mean()) ** 2)
        r2[i] = 0.0 if sst <= 0 else max(0.0, 1 - np.sum((y - fit) ** 2) / sst)
    return r2


def get_r2(cfg, profiles):
    if cfg.use_metacycle and os.path.exists(RSCRIPT) and _have_metacycle():
        try:
            print("  Running MetaCycle ...")
            return _run_metacycle(profiles, cfg.timepoints)
        except Exception as e:
            print(f"  [MetaCycle failed → cosinor fallback: {e}]")
    print("  Using Python cosinor fallback.")
    return _cosinor_r2(profiles, cfg.timepoints)


# =============================================================================
# Step 3 — Simulate genotypes and protein levels (independent of rhythmicity)
# =============================================================================
def simulate_genotypes(cfg, rng):
    maf = rng.uniform(0.05, 0.45, cfg.M)
    G   = rng.binomial(2, maf, size=(cfg.N, cfg.M)).astype(float)
    G   = (G - G.mean(0)) / (G.std(0) + 1e-9)
    return G


def simulate_protein_levels(cfg, G, rng):
    """P = G*beta + e, heritability h2_pqtl per protein."""
    beta = np.zeros((cfg.M, cfg.P_total))
    for j in range(cfg.P_total):
        idx = rng.choice(cfg.M, cfg.k_pqtl, replace=False)
        beta[idx, j] = rng.normal(0.0, 1.0, cfg.k_pqtl)

    gv = G @ beta
    sd = gv.std(0)
    has_signal = sd > 1e-8
    gv[:, has_signal] /= sd[has_signal]

    e = rng.standard_normal(gv.shape)
    P = np.sqrt(cfg.h2_pqtl) * gv + np.sqrt(1.0 - cfg.h2_pqtl) * e
    P = (P - P.mean(0)) / (P.std(0) + 1e-9)
    return P, gv, has_signal


# =============================================================================
# Step 4 — Inject matched pairs
#
# Each pair: (p_low, p_high) where R²[p_low] << R²[p_high].
# We OVERWRITE both proteins' level columns with the SAME values so their
# vanilla PWAS Z-scores are IDENTICAL.  The only thing left to distinguish
# them is rhythmicity — exactly what the weighted method uses.
# =============================================================================
def inject_contrast_pairs(cfg, P, gv, has_signal, r2, rng):
    lo = np.percentile(r2, cfg.low_r2_pct)
    hi = np.percentile(r2, cfg.high_r2_pct)
    low_pool  = [p for p in range(cfg.P_total) if r2[p] <= lo and has_signal[p]]
    high_pool = [p for p in range(cfg.P_total) if r2[p] >= hi and has_signal[p]]
    rng.shuffle(low_pool)
    rng.shuffle(high_pool)
    n = min(cfg.n_pairs, len(low_pool), len(high_pool))

    pairs = []
    for k in range(n):
        p_lo, p_hi = low_pool[k], high_pool[k]
        shared = gv[:, p_hi].copy()
        col = (np.sqrt(cfg.h2_pqtl) * shared
               + np.sqrt(1.0 - cfg.h2_pqtl) * rng.standard_normal(cfg.N))
        col = (col - col.mean()) / (col.std() + 1e-9)
        P[:, p_lo] = col
        P[:, p_hi] = col     # identical → same vanilla Z-score
        pairs.append((p_lo, p_hi))
    return pairs


# =============================================================================
# Step 5 — PWAS + weighting
# =============================================================================
def pwas_z(P, y):
    N  = P.shape[0]
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-9)
    yz = (y - y.mean()) / (y.std() + 1e-9)
    r  = np.clip((Pz * yz[:, None]).mean(0), -0.9999, 0.9999)
    return r * np.sqrt(N - 2) / np.sqrt(1.0 - r ** 2)


def make_weight(r2, tau, invert=False):
    z = (r2 - r2.mean()) / (r2.std() + 1e-9)
    w = np.exp(tau * (-z if invert else z))
    return w / w.mean()


def select_causal(cfg, r2, scenario, has_signal, rng):
    idx = np.where(has_signal)[0]
    if scenario == "circadian":
        z    = (r2[idx] - r2[idx].mean()) / (r2[idx].std() + 1e-9)
        prob = np.exp(cfg.select_lambda * z)
        prob /= prob.sum()
        return rng.choice(idx, size=min(cfg.n_causal, len(idx)), replace=False, p=prob)
    return rng.choice(idx, size=min(cfg.n_causal, len(idx)), replace=False)


def make_phenotype(cfg, P, r2, scenario, has_signal, rng):
    causal = select_causal(cfg, r2, scenario, has_signal, rng)
    if scenario == "null":
        return rng.standard_normal(P.shape[0]), causal
    gamma = rng.normal(0.0, 1.0, len(causal))
    sig   = (P[:, causal] * gamma).sum(1)
    sig   = (sig - sig.mean()) / (sig.std() + 1e-9)
    y     = (np.sqrt(cfg.h2_trait) * sig
             + np.sqrt(1.0 - cfg.h2_trait) * rng.standard_normal(P.shape[0]))
    return y, causal


def compute_delta_auc(Z, obs_r2, causal, n_proteins, tau):
    labels = np.zeros(n_proteins)
    labels[causal] = 1
    if labels.sum() == 0 or labels.sum() == n_proteins:
        return 0.0, 0.5, 0.5
    w     = make_weight(obs_r2, tau)
    auc_v = roc_auc_score(labels, np.abs(Z))
    auc_w = roc_auc_score(labels, np.abs(Z) * np.sqrt(w))
    return auc_w - auc_v, auc_v, auc_w


def evaluate_pairs(cfg, Z, obs_r2, pairs):
    """
    For each matched pair (p_lo, p_hi):
      - vanilla: |Z_lo| == |Z_hi|  (always, by construction)
      - weighted: |Z_hi|*sqrt(w_hi) vs |Z_lo|*sqrt(w_lo) — should favour p_hi
    Returns fraction of pairs where weighted method correctly ranks p_hi above p_lo.
    """
    w   = make_weight(obs_r2, cfg.tau)
    van = np.abs(Z)
    wtd = np.abs(Z) * np.sqrt(w)
    records = []
    for p_lo, p_hi in pairs:
        records.append({
            "vanilla_tie":       abs(van[p_lo] - van[p_hi]) < 1e-6,
            "weighted_correct":  wtd[p_hi] > wtd[p_lo],
            "r2_lo":   float(obs_r2[p_lo]),
            "r2_hi":   float(obs_r2[p_hi]),
            "wtd_gap": float(wtd[p_hi] - wtd[p_lo]),
        })
    df = pd.DataFrame(records)
    return {
        "n_pairs":               len(pairs),
        "vanilla_tie_rate":      float(df["vanilla_tie"].mean()),
        "weighted_correct_rate": float(df["weighted_correct"].mean()),
        "mean_r2_lo":            float(df["r2_lo"].mean()),
        "mean_r2_hi":            float(df["r2_hi"].mean()),
        "mean_wtd_gap":          float(df["wtd_gap"].mean()),
    }


# =============================================================================
# Driver
# =============================================================================
def run_simulation(cfg):
    rng = np.random.default_rng(cfg.seed)

    print("=" * 62)
    print("Step 1  Simulating protein circadian profiles")
    print("=" * 62)
    profiles, amplitudes, phases, mesors = simulate_profiles(cfg, rng)
    corr_amp_mesor = float(np.corrcoef(amplitudes, mesors)[0, 1])
    print(f"  Proteins:         {cfg.P_total}")
    print(f"  Time points:      {cfg.timepoints}")
    print(f"  Amplitude range:  [{amplitudes.min():.2f}, {amplitudes.max():.2f}]")
    print(f"  MESOR range:      [{mesors.min():.2f}, {mesors.max():.2f}]")
    print(f"  Corr(amp, MESOR): {corr_amp_mesor:+.3f}  (should be ~0)")

    print("\n" + "=" * 62)
    print("Step 2  Computing rhythmicity R² via MetaCycle")
    print("=" * 62)
    r2 = get_r2(cfg, profiles)
    print(f"  R² range:  [{r2.min():.3f}, {r2.max():.3f}]")
    print(f"  R² median: {np.median(r2):.3f}")
    corr_r2_amp = float(np.corrcoef(r2, amplitudes)[0, 1])
    print(f"  Corr(R², true amplitude): {corr_r2_amp:.3f}  (should be high)")

    print("\n" + "=" * 62)
    print("Step 3  Simulating genotypes + protein levels")
    print("=" * 62)
    G = simulate_genotypes(cfg, rng)
    P, gv, has_signal = simulate_protein_levels(cfg, G, rng)
    print(f"  Individuals: {cfg.N},  SNPs: {cfg.M}")
    print(f"  Proteins with ≥1 pQTL signal: {has_signal.sum()} / {cfg.P_total}")
    corr_r2_mean = float(np.corrcoef(r2, P.mean(0))[0, 1])
    print(f"  Corr(R², mean protein level): {corr_r2_mean:+.3f}  (should be ~0)")

    print("\n" + "=" * 62)
    print("Step 4  Injecting matched pairs")
    print("=" * 62)
    pairs = inject_contrast_pairs(cfg, P, gv, has_signal, r2, rng)
    lo_r2 = np.percentile(r2, cfg.low_r2_pct)
    hi_r2 = np.percentile(r2, cfg.high_r2_pct)
    print(f"  {len(pairs)} pairs  (p_low R²≤{lo_r2:.3f},  p_high R²≥{hi_r2:.3f})")
    print("  Both proteins in each pair share IDENTICAL protein-level column")
    print("  → vanilla PWAS Z-scores are identical by construction")

    scenarios = ["circadian", "random", "null"]
    rows, pair_acc = [], []

    print(f"\n{'=' * 62}")
    print(f"Step 5  Running {cfg.n_reps} reps × {len(scenarios)} scenarios")
    print("=" * 62)
    for sc in scenarios:
        for rep in range(cfg.n_reps):
            rng_rep = np.random.default_rng(cfg.seed + 1000 * rep + hash(sc) % 997)
            y, causal = make_phenotype(cfg, P, r2, sc, has_signal, rng_rep)
            Z = pwas_z(P, y)

            # add small jitter to R² before using as prior (prevents perfect circularity)
            obs_r2 = np.clip(
                r2 + rng_rep.normal(0.0, cfg.prior_noise_sd, cfg.P_total), 0, None)

            d, av, aw = compute_delta_auc(Z, obs_r2, causal, cfg.P_total, cfg.tau)
            rows.append((sc, rep, d, av, aw))

            if sc == "circadian":
                pair_acc.append(evaluate_pairs(cfg, Z, obs_r2, pairs))

    df = pd.DataFrame(rows,
                      columns=["scenario", "rep", "delta_auc",
                               "auc_vanilla", "auc_weighted"])
    summ = (df.groupby("scenario")["delta_auc"]
              .agg(n="count", mean_delta_auc="mean", sd="std")
              .reset_index())
    summ["se"] = summ["sd"] / np.sqrt(summ["n"])

    pair_summ = None
    if pair_acc:
        pair_summ = {k: float(np.mean([x[k] for x in pair_acc]))
                     for k in ("vanilla_tie_rate", "weighted_correct_rate",
                               "mean_r2_lo", "mean_r2_hi", "mean_wtd_gap")}

    return dict(df=df, summary=summ, pair_summary=pair_summ,
                r2=r2, amplitudes=amplitudes, profiles=profiles)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",           default=os.path.join(HERE, "results_sim"))
    ap.add_argument("--reps",          type=int,   default=Config.n_reps)
    ap.add_argument("--n-proteins",    type=int,   default=Config.P_total)
    ap.add_argument("--n-individuals", type=int,   default=Config.N)
    ap.add_argument("--tau",           type=float, default=Config.tau)
    ap.add_argument("--no-metacycle",  action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.n_reps  = args.reps
    cfg.P_total = args.n_proteins
    cfg.N       = args.n_individuals
    cfg.tau     = args.tau
    if args.no_metacycle:
        cfg.use_metacycle = False

    os.makedirs(args.out, exist_ok=True)
    res = run_simulation(cfg)

    # ── print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("RESULTS  Delta AUC by scenario")
    print("=" * 62)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(res["summary"].to_string(index=False))

    if res["pair_summary"]:
        c = res["pair_summary"]
        print("\n" + "=" * 62)
        print("RESULTS  Matched-pair contrast (identical vanilla Z-scores)")
        print("=" * 62)
        print(f"  Pairs:                            {int(c.get('n_pairs', cfg.n_pairs))}")
        print(f"  Mean R² of low-rhythm protein:    {c['mean_r2_lo']:.3f}")
        print(f"  Mean R² of high-rhythm protein:   {c['mean_r2_hi']:.3f}")
        print(f"  Vanilla PWAS tie rate:            {c['vanilla_tie_rate']:.3f}  (expected ≈ 1.0)")
        print(f"  Weighted ranks rhythmic higher:   {c['weighted_correct_rate']:.3f}  (expected > 0.5)")
        print(f"  Mean weighted-score gap (hi−lo):  {c['mean_wtd_gap']:+.4f}")

    print("\n" + "=" * 62)
    print("INTERPRETATION")
    print("=" * 62)
    sc_df = res["summary"].set_index("scenario")
    for sc, label, expectation in [
        ("circadian", "Circadian scenario", "should be > 0"),
        ("random",    "Random scenario   ", "should be ≈ 0"),
        ("null",      "Null scenario     ", "should be ≈ 0"),
    ]:
        if sc in sc_df.index:
            d = sc_df.loc[sc, "mean_delta_auc"]
            print(f"  {label}  Delta AUC = {d:+.4f}   [{expectation}]")

    if res["pair_summary"]:
        cr = res["pair_summary"]["weighted_correct_rate"]
        print(f"\n  Pair contrast: weighted method correctly ranks")
        print(f"  the more rhythmic protein higher in {cr:.1%} of matched pairs.")
        print(f"  (Vanilla PWAS cannot distinguish them — it always ties.)")

    # ── save outputs ───────────────────────────────────────────────────────
    res["df"].to_csv(
        os.path.join(args.out, "delta_auc_replicates.csv"), index=False)
    res["summary"].to_csv(
        os.path.join(args.out, "scenario_summary.csv"), index=False)
    if res["pair_summary"]:
        pd.DataFrame([res["pair_summary"]]).to_csv(
            os.path.join(args.out, "pair_contrast.csv"), index=False)

    prof_df = pd.DataFrame(
        res["profiles"],
        index=[f"P{i}" for i in range(cfg.P_total)],
        columns=[f"t{t}" for t in cfg.timepoints])
    prof_df.insert(0, "amplitude_true", res["amplitudes"])
    prof_df.insert(1, "R2_metacycle",   res["r2"])
    prof_df.to_csv(os.path.join(args.out, "protein_profiles.csv"))

    print(f"\nAll results saved to {args.out}/")


if __name__ == "__main__":
    main()
