#!/usr/bin/env python3
"""
simulation3_3.py

Pipeline (from memo):
  genotype matrix → protein-time matrix → average board pQTL → w_pQTL

Step-by-step:
  1. G  (genotype matrix)  – simulated from real 1000G MAF
  2. protein-time matrix   – P_i(t, j) = G_j @ beta_i
                                        + A_i * cos(2π(t - φ_i)/24)
                                        + noise
     Amplitude A_i and phase φ_i are drawn from distributions
     learned from report.pg_matrix.tsv (not used directly).
  3. average board pQTL    – average across time points:
                             P_avg_ij = mean_t P_i(t,j)
                                      ≈ G_j @ beta_i + noise
                             (circadian cos averages to zero)
  4. w_pQTL               – run PWAS on P_avg → Z_i
                             weight: score_i = |Z_i| * sqrt(w(R²_i))

R² is computed by running MetaCycle on the population-mean profile:
  pop_mean_i(t) = mean_j P_i(t,j) ≈ A_i * cos(2π(t-φ_i)/24) + small_noise

Why this is non-trivial:
  • Causal proteins are selected by true amplitude A (not R²).
  • R² is MetaCycle's estimate of A from noisy pop_mean – a noisy proxy.
  • Matched pairs share the same pQTL beta (same genetic effect) but
    different A → similar but NOT identical vanilla Z-scores.
    The 100%-correct trivial outcome is gone; the method now has to
    beat noise to correctly rank the rhythmic protein higher.

Usage:
  python3 simulation3_3.py [--reps 200] [--out results_sim3]
  python3 simulation3_3.py --no-metacycle   # Python cosinor fallback
  python3 simulation3_3.py --no-ld          # skip .bed reading
"""

import os, re, argparse, subprocess, tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

HERE    = os.path.dirname(os.path.abspath(__file__))
RSCRIPT = os.path.join(HERE, "run_metacycle.R")

D_PROFILES  = "raw_data/circadian_info/report.pg_matrix.tsv"
D_PQTL      = "work/pqtl_topk.csv"
D_LDREF     = "raw_data/pQTL/ldref/LDREF"
D_UKBB_LD   = "/data/CommonData/ukbb-ld"
D_CHAIN     = "reference/chain/hg38ToHg19.over.chain.gz"
D_ENSEMBL   = "raw_data/reference/ensembl_maf.csv"   # 永久 MAF 文件


# =============================================================================
# Config
# =============================================================================
class Config:
    N              = 2000   # synthetic individuals
    h2_pqtl        = 0.10   # per-protein pQTL heritability
    h2_trait       = 0.30   # trait heritability through causal proteins
    n_causal       = 20     # ground-truth causal proteins per rep

    timepoints     = list(range(0, 24, 2))  # 0,2,4,...,22
    n_tp_noise     = 30     # effective n per time point for pop-mean noise

    tau            = 1.0    # weight sharpness
    prior_noise_sd = 0.10   # noise added to R² before using as prior

    n_pairs        = 40
    low_amp_pct    = 20     # percentile for "low amplitude" pool
    high_amp_pct   = 80     # percentile for "high amplitude" pool

    n_reps         = 200
    seed           = 42
    select_lambda  = 3.0    # causal enrichment strength on amplitude A

    use_metacycle  = True
    use_real_ld    = True


# =============================================================================
# Section A  ── Learn circadian parameter distributions from real data
# =============================================================================
def _parse_real_profiles(path):
    df      = pd.read_csv(path, sep="\t")
    tp_re   = re.compile(r"-t(\d+)\.mzML$", re.IGNORECASE)
    col_hour = {}
    for c in df.columns:
        m = tp_re.search(c)
        if m:
            col_hour[c] = int(m.group(1)) % 24
    assert col_hour
    hours = sorted(set(col_hour.values()))
    meas  = df[list(col_hour.keys())].apply(pd.to_numeric, errors="coerce")
    meas  = np.log2(meas.clip(lower=1).to_numpy())
    profiles = np.zeros((len(df), len(hours)))
    for j, h in enumerate(hours):
        idx = [i for i, c in enumerate(col_hour.keys()) if col_hour[c] == h]
        if not idx:
            continue
        with np.errstate(all="ignore"):
            profiles[:, j] = np.nanmean(meas[:, idx], axis=1)
    ok = np.isfinite(profiles).all(axis=1)
    return profiles[ok], np.array(hours, float)


def _fit_cosinor(profiles, timepoints):
    t     = np.array(timepoints, float)
    omega = 2 * np.pi / 24.0
    X     = np.column_stack([np.ones_like(t),
                              np.cos(omega * t), np.sin(omega * t)])
    coef  = np.linalg.lstsq(X, profiles.T, rcond=None)[0]
    amp   = np.sqrt(coef[1] ** 2 + coef[2] ** 2)
    phase = (np.arctan2(coef[2], coef[1]) / omega) % 24.0
    fitted    = (X @ coef).T
    noise_sd  = (profiles - fitted).std(axis=1)
    mesor     = coef[0]
    return amp, phase, noise_sd, mesor


def learn_rhythm_params(path):
    """
    Fit cosinor to each protein in report.pg_matrix.tsv.
    Returns log-normal parameters for amplitude and noise,
    and normal parameters for MESOR.
    These are used to draw synthetic circadian parameters — the raw
    measurement values are never used again.
    """
    profiles, timepoints = _parse_real_profiles(path)
    amp, phase, noise_sd, mesor = _fit_cosinor(profiles, timepoints)

    ok = amp > 0.01
    log_amp   = np.log(amp[ok])
    log_noise = np.log(noise_sd[ok] + 1e-6)

    return dict(
        mu_log_amp   = float(log_amp.mean()),
        sd_log_amp   = float(log_amp.std()),
        mu_log_noise = float(log_noise.mean()),
        sd_log_noise = float(log_noise.std()),
        mu_mesor     = float(mesor.mean()),
        sd_mesor     = float(mesor.std()),
        timepoints   = timepoints,
    )


# =============================================================================
# Section B  ── Load pQTL table
# =============================================================================
def load_pqtl(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"EntrezGeneSymbol": "gene",
                             "SNP": "snp_id", "BETA": "beta"})
    df["gene"]   = df["gene"].astype(str)
    df["snp_id"] = df["snp_id"].astype(str)
    df["beta"]   = pd.to_numeric(df["beta"], errors="coerce")
    return df[["gene", "snp_id", "CHR", "POS", "beta"]].dropna()


# =============================================================================
# Section C  ── Read MAF + LD from 1000G .bed files
# =============================================================================
def _decode_bed_snp(raw, n_samples):
    raw    = np.frombuffer(raw, dtype=np.uint8)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    codes  = ((raw[:, None] >> shifts) & 0x03).ravel()[:n_samples]
    return np.where(codes == 0, 0.0,
           np.where(codes == 1, np.nan,
           np.where(codes == 2, 1.0, 2.0)))


def _load_chr_genotypes(bed_dir, chrom, snp_positions, n_ref=489):
    bim_p = os.path.join(bed_dir, f"1000G.EUR.{chrom}.bim")
    bed_p = os.path.join(bed_dir, f"1000G.EUR.{chrom}.bed")
    fam_p = os.path.join(bed_dir, f"1000G.EUR.{chrom}.fam")
    if not all(os.path.exists(p) for p in [bim_p, bed_p, fam_p]):
        return None, None, []

    bim = pd.read_csv(bim_p, sep="\t", header=None,
                      names=["chr", "rsid", "cm", "bp", "a1", "a2"])
    n_samples      = n_ref
    bytes_per_snp  = (n_samples + 3) // 4
    pos2row        = {int(bp): i for i, bp in enumerate(bim["bp"])}

    matched = [(pos2row[int(pos)], sid)
               for pos, sid in snp_positions if int(pos) in pos2row]
    if not matched:
        return None, None, []

    rows, ids = zip(*matched)
    G = np.full((n_samples, len(rows)), np.nan)
    with open(bed_p, "rb") as f:
        if f.read(3) != b"\x6c\x1b\x01":
            return None, None, []
        for out_i, row_i in enumerate(rows):
            f.seek(3 + row_i * bytes_per_snp)
            G[:, out_i] = _decode_bed_snp(f.read(bytes_per_snp), n_samples)

    maf = np.nanmean(G, axis=0) / 2.0
    maf = np.where(maf > 0.5, 1.0 - maf, maf)
    return G, np.clip(maf, 1e-4, 0.5), list(ids)


def liftover_hg38_to_hg19(pqtl_df, chain_file, cache_path=None):
    """
    用 CrossMap 把 pQTL SNP 从 hg38 坐标转换到 hg19。
    返回 dict: hg38_snp_id -> (hg19_chr, hg19_pos)
    转换失败的 SNP 不出现在返回字典里。
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached liftover from {cache_path}")
        df = pd.read_csv(cache_path)
        return dict(zip(df["snp_id"], zip(df["hg19_chr"], df["hg19_pos"])))

    print(f"  Running CrossMap hg38 → hg19 ...")

    # 构建 BED 格式输入（0-based start, 1-based end）
    lines = []
    snp_meta = {}
    for _, row in pqtl_df.drop_duplicates("snp_id").iterrows():
        try:
            chrom = int(row["CHR"])
            pos   = int(row["POS"])
            sid   = str(row["snp_id"])
            lines.append(f"chr{chrom}\t{pos-1}\t{pos}\t{sid}\n")
            snp_meta[sid] = chrom
        except (ValueError, TypeError):
            continue

    import tempfile, subprocess
    with tempfile.TemporaryDirectory() as d:
        in_bed  = os.path.join(d, "hg38.bed")
        out_bed = os.path.join(d, "hg19.bed")

        with open(in_bed, "w") as f:
            f.writelines(lines)

        ret = subprocess.run(
            ["CrossMap", "bed", chain_file, in_bed, out_bed],
            capture_output=True, text=True)

        if not os.path.exists(out_bed):
            print(f"  CrossMap failed: {ret.stderr[:200]}")
            return {}

        # 解析输出：hg19 chr, start, end, snp_id
        mapping = {}
        with open(out_bed) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                hg19_chr_str = parts[0].replace("chr", "")
                hg19_pos     = int(parts[2])   # 1-based end = SNP position
                sid          = parts[3]
                try:
                    mapping[sid] = (int(hg19_chr_str), hg19_pos)
                except ValueError:
                    continue

    n_total = pqtl_df["snp_id"].nunique()
    print(f"  Lifted over: {len(mapping)} / {n_total} SNPs "
          f"({len(mapping)/max(n_total,1):.1%})")

    if cache_path:
        rows = [{"snp_id": k, "hg19_chr": v[0], "hg19_pos": v[1]}
                for k, v in mapping.items()]
        pd.DataFrame(rows).to_csv(cache_path, index=False)

    return mapping


def _match_ukbb_ld(hg19_map, pqtl_df, ukbb_dir):
    """
    hg19 位置 → UKBB-LD .gz 里查 rsID。
    返回 dict: snp_id -> rsid
    """
    import gzip, re
    from collections import defaultdict

    # 建 block 索引
    block_idx = defaultdict(list)
    for f in os.listdir(ukbb_dir):
        m = re.match(r"chr(\d+)_(\d+)_(\d+)\.gz$", f)
        if m:
            c, s, e = int(m.group(1)), int(m.group(2)), int(m.group(3))
            block_idx[c].append((s, e, os.path.join(ukbb_dir, f)))

    # 按染色体分组 pQTL SNP 的 hg19 位置
    snps_by_chr = defaultdict(dict)   # chr -> {hg19_pos: snp_id}
    for snp_id, (chrom, pos) in hg19_map.items():
        snps_by_chr[chrom][pos] = snp_id

    rsid_map = {}   # snp_id -> rsid
    for chrom, pos_map in snps_by_chr.items():
        seen_pos = set()
        for start, end, fpath in block_idx.get(chrom, []):
            remaining = {p: sid for p, sid in pos_map.items()
                         if p not in seen_pos}
            if not remaining:
                break
            with gzip.open(fpath, "rt") as f:
                next(f)
                for line in f:
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    try:
                        pos = int(parts[2])
                    except ValueError:
                        continue
                    if pos in remaining:
                        rsid = parts[0].strip()
                        sid  = remaining[pos]
                        rsid_map[sid] = rsid
                        seen_pos.add(pos)

    return rsid_map


def _fetch_maf_ensembl(rsids, pop="gnomADg:nfe", batch_size=200,
                       cache_path=None, permanent_path=None):
    """
    Ensembl REST API 批量查询 rsID → EUR MAF。
    pop 默认用 gnomADg:nfe（gnomAD 基因组非芬兰欧洲人，最接近 UK Biobank）。
    返回 dict: rsid -> maf (float)
    """
    import urllib.request, urllib.error, json, time

    # 优先读永久文件，其次读缓存
    for src in [permanent_path, cache_path]:
        if src and os.path.exists(src):
            print(f"  Loading MAF from {src}")
            df = pd.read_csv(src)
            return dict(zip(df["rsid"], df["maf"]))

    url     = "https://rest.ensembl.org/variation/homo_sapiens"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    maf_res = {}
    rsid_list = list(set(rsids))
    n_batches = (len(rsid_list) + batch_size - 1) // batch_size

    print(f"  Querying Ensembl for {len(rsid_list)} rsIDs "
          f"({n_batches} batches of ≤{batch_size}) ...")

    for i in range(0, len(rsid_list), batch_size):
        batch = rsid_list[i: i + batch_size]
        body  = json.dumps({"ids": batch}).encode()
        req   = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  WARNING: Ensembl request failed ({e})")
            print(f"  → 计算节点可能无外网。请在登录节点预先运行 fetch_maf.py。")
            break   # 放弃剩余批次，fallback 到随机 MAF

        for rsid, info in data.items():
            if not isinstance(info, dict):
                continue
            best = None
            for p in info.get("populations", []):
                if p.get("population") == pop:
                    f = float(p.get("frequency", 0))
                    if best is None or f < best:
                        best = f     # take the minor allele (smaller freq)
            if best is not None and 0 < best <= 0.5:
                maf_res[rsid] = best
            elif info.get("MAF") is not None:
                maf_res[rsid] = float(info["MAF"])   # fallback to global MAF

        bn = (i // batch_size) + 1
        print(f"  Batch {bn}/{n_batches}: {len(maf_res)} MAF values so far")
        time.sleep(0.5)   # respect Ensembl rate limit (~15 req/s)

    if cache_path:
        rows = [{"rsid": k, "maf": v} for k, v in maf_res.items()]
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        print(f"  Cached Ensembl MAF → {cache_path}")

    return maf_res


def load_maf_ld(pqtl_df, bed_dir, chain_file=None, ukbb_dir=None, cache_dir=None):
    """
    Pipeline:
      1. CrossMap liftover hg38 → hg19
      2. 匹配 UKBB-LD .gz → 拿 rsID（~98.8%）
      3. Ensembl API 批量查 EUR MAF
      4. fallback 到随机 MAF 覆盖剩余 SNP
    """
    maf_cache = os.path.join(cache_dir, "maf.csv")      if cache_dir else None
    ld_cache  = os.path.join(cache_dir, "ld_blocks.npz") if cache_dir else None

    if (maf_cache and os.path.exists(maf_cache) and
            ld_cache  and os.path.exists(ld_cache)):
        print(f"  Loading cached MAF from {maf_cache}")
        maf_df    = pd.read_csv(maf_cache, index_col=0)["maf"]
        ld_blocks = np.load(ld_cache, allow_pickle=True)["ld_blocks"].item()
        return maf_df, ld_blocks

    all_snps = pqtl_df["snp_id"].unique().tolist()
    maf_dict = {}

    # ── Step 1: liftover hg38 → hg19 ─────────────────────────────────────────
    hg19_map = {}
    if chain_file and os.path.exists(chain_file):
        lo_cache = os.path.join(cache_dir, "liftover_hg19.csv") if cache_dir else None
        hg19_map = liftover_hg38_to_hg19(pqtl_df, chain_file, lo_cache)
        print(f"  Liftover: {len(hg19_map)} / {len(all_snps)} SNPs converted")

    # ── Step 2: 匹配 UKBB-LD → rsID ──────────────────────────────────────────
    rsid_map = {}   # snp_id -> rsid
    _ukbb = ukbb_dir or bed_dir   # 优先用专用 ukbb_dir
    if hg19_map and os.path.isdir(_ukbb):
        rs_cache = os.path.join(cache_dir, "rsid_map.csv") if cache_dir else None
        if rs_cache and os.path.exists(rs_cache):
            print(f"  Loading cached rsID map from {rs_cache}")
            rs_df   = pd.read_csv(rs_cache)
            rsid_map = dict(zip(rs_df["snp_id"], rs_df["rsid"]))
        else:
            print(f"  Matching {len(hg19_map)} SNPs to UKBB-LD → rsID ...")
            rsid_map = _match_ukbb_ld(hg19_map, pqtl_df, _ukbb)
            print(f"  rsID found: {len(rsid_map)} / {len(hg19_map)}")
            if rs_cache:
                pd.DataFrame([{"snp_id": k, "rsid": v}
                               for k, v in rsid_map.items()]
                              ).to_csv(rs_cache, index=False)

    # ── Step 3: Ensembl API → EUR MAF ─────────────────────────────────────────
    if rsid_map:
        ens_cache = os.path.join(cache_dir, "ensembl_maf.csv") if cache_dir else None
        rsid_to_maf = _fetch_maf_ensembl(
            list(rsid_map.values()),
            cache_path=ens_cache,
            permanent_path=D_ENSEMBL)
        for snp_id, rsid in rsid_map.items():
            if rsid in rsid_to_maf:
                maf_dict[snp_id] = rsid_to_maf[rsid]

    # ── Step 4: fallback 随机 MAF for 剩余 SNP ───────────────────────────────
    missing = [s for s in all_snps if s not in maf_dict]
    if missing:
        rng_fb = np.random.default_rng(999)
        for s in missing:
            maf_dict[s] = float(rng_fb.uniform(0.05, 0.45))

    n_real   = len(all_snps) - len(missing)
    n_random = len(missing)
    print(f"  MAF final: {n_real} from gnomAD-EUR, "
          f"{n_random} random fallback ({n_random/len(all_snps):.1%})")

    # LD blocks（仅在有真实基因型时可建，目前 UKBB-LD 不提供个人基因型，留空）
    ld_blocks = {}

    maf_series = pd.Series(maf_dict)
    if maf_cache:
        maf_series.rename("maf").reset_index()\
            .rename(columns={"index": "snp_id"}).to_csv(maf_cache, index=False)
        np.savez(ld_cache, ld_blocks=np.array(ld_blocks, dtype=object))
    return maf_series, ld_blocks


# =============================================================================
# Section D  ── Harmonize proteins across pQTL + MAF
# =============================================================================
def harmonize(pqtl_df, maf_series, verbose=True):
    maf_snps   = set(maf_series.index)
    pqtl_genes = set(pqtl_df["gene"])
    w_maf      = set(pqtl_df[pqtl_df["snp_id"].isin(maf_snps)]["gene"])
    proteins   = sorted(pqtl_genes & w_maf)
    assert proteins, "No proteins with both pQTL and MAF data"

    pqtl_h = pqtl_df[pqtl_df["gene"].isin(proteins) &
                      pqtl_df["snp_id"].isin(maf_snps)]
    snps   = pqtl_h["snp_id"].unique().tolist()
    maf_h  = maf_series.reindex(snps).fillna(0.05).clip(1e-4, 0.5).to_numpy()

    s2i = {s: i for i, s in enumerate(snps)}
    p2i = {p: i for i, p in enumerate(proteins)}
    beta = np.zeros((len(snps), len(proteins)))
    for _, row in pqtl_h.iterrows():
        beta[s2i[row["snp_id"]], p2i[row["gene"]]] += float(row["beta"])

    if verbose:
        nz = (beta != 0).sum(0)
        print(f"  {len(proteins)} proteins, {len(snps)} SNPs, "
              f"median {int(np.median(nz))} pQTL/protein")
    return dict(proteins=proteins, snps=snps, beta=beta, maf=maf_h)


# =============================================================================
# Section E  ── Load LD matrix from UKBB-LD + simulate genotypes (LD + MAF → G)
# =============================================================================
def _nearest_psd(A, eps=1e-6):
    """Project symmetric matrix to nearest positive semi-definite correlation matrix."""
    A    = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    w    = np.clip(w, eps, None)
    B    = (V * w) @ V.T
    d    = np.sqrt(np.clip(np.diag(B), eps, None))
    return B / np.outer(d, d)


def load_ukbb_ld(data, ukbb_dir, cache_dir=None):
    """
    Build LD matrix for pQTL SNPs using UKBB-LD .npz files.

    SNP pairs within the same UKBB-LD block get real LD values (r).
    All other pairs (different blocks or chromosomes) are set to 0.
    Result is stored in data["LD"] as a dense (M × M) numpy array.
    """
    import gzip

    snps = data["snps"]
    M    = len(snps)
    s2i  = {s: i for i, s in enumerate(snps)}

    ld_cache = os.path.join(cache_dir, "pqtl_ld.npy") if cache_dir else None
    if ld_cache and os.path.exists(ld_cache):
        print(f"  Loading cached LD matrix ({ld_cache})")
        data["LD"] = np.load(ld_cache)
        return

    # Load hg19 positions and rsIDs from existing cache files
    lo_path = os.path.join(cache_dir, "liftover_hg19.csv") if cache_dir else None
    rs_path = os.path.join(cache_dir, "rsid_map.csv")      if cache_dir else None
    if not (lo_path and os.path.exists(lo_path) and
            rs_path and os.path.exists(rs_path)):
        print("  WARNING: liftover/rsid cache missing → using identity LD")
        data["LD"] = None
        return

    lo_df    = pd.read_csv(lo_path)
    hg19_map = dict(zip(lo_df["snp_id"],
                        zip(lo_df["hg19_chr"].astype(int),
                            lo_df["hg19_pos"].astype(int))))
    rs_df    = pd.read_csv(rs_path)
    rsid_map = dict(zip(rs_df["snp_id"], rs_df["rsid"]))

    print(f"  Building LD matrix ({M}×{M}) from UKBB-LD ...")
    LD = np.eye(M, dtype=np.float32)

    # Index UKBB-LD blocks
    block_idx = defaultdict(list)
    for f in os.listdir(ukbb_dir):
        m = re.match(r"chr(\d+)_(\d+)_(\d+)\.gz$", f)
        if m:
            c, s, e = int(m.group(1)), int(m.group(2)), int(m.group(3))
            npz_p = os.path.join(ukbb_dir, f.replace(".gz", ".npz"))
            gz_p  = os.path.join(ukbb_dir, f)
            if os.path.exists(npz_p):
                block_idx[c].append((s, e, gz_p, npz_p))

    # Group pQTL SNPs by chromosome
    snps_by_chr = defaultdict(list)
    for sid in snps:
        if sid in hg19_map:
            chrom, pos = hg19_map[sid]
            snps_by_chr[int(chrom)].append((int(pos), sid))

    n_ld_pairs = 0
    seen_blocks = set()

    for chrom in sorted(snps_by_chr):
        for block_start, block_end, gz_p, npz_p in block_idx.get(chrom, []):
            in_block = [(pos, sid) for pos, sid in snps_by_chr[chrom]
                        if block_start <= pos <= block_end]
            if len(in_block) < 2 or npz_p in seen_blocks:
                continue
            seen_blocks.add(npz_p)

            # Read .gz → rsid order in this block
            with gzip.open(gz_p, "rt") as f:
                next(f)
                block_rsids = [line.split("\t")[0].strip() for line in f]
            rsid2bidx = {r: i for i, r in enumerate(block_rsids)}

            # Map pQTL SNPs to their row/col indices inside this block
            entries = []
            for _, sid in in_block:
                rsid = rsid_map.get(sid)
                if rsid and rsid in rsid2bidx:
                    entries.append((sid, rsid2bidx[rsid]))
            if len(entries) < 2:
                continue

            # 高效读取 LD：只过滤我们需要的行列，不加载全矩阵
            bidxs    = np.array([bi for _, bi in entries])
            sids     = [si for si, _ in entries]
            bidx_set = set(bidxs.tolist())
            k        = len(bidxs)
            local_i  = {int(bi): i for i, bi in enumerate(bidxs)}

            d         = np.load(npz_p, mmap_mode="r")
            rows_all  = d["row"]
            cols_all  = d["col"]
            # 只保留两端都在我们目标 SNP 集合里的 LD 值
            mask = np.isin(rows_all, bidxs) & np.isin(cols_all, bidxs)
            sub  = np.eye(k)
            if mask.any():
                for ri, ci, v in zip(rows_all[mask],
                                     cols_all[mask],
                                     d["data"][mask]):
                    sub[local_i[int(ri)], local_i[int(ci)]] = float(v)
            sub = (sub + sub.T) / 2
            np.fill_diagonal(sub, 1.0)

            # Fill into global LD
            for i1, sid1 in enumerate(sids):
                for i2, sid2 in enumerate(sids):
                    if i1 != i2 and sid1 in s2i and sid2 in s2i:
                        LD[s2i[sid1], s2i[sid2]] = float(sub[i1, i2])
                        n_ld_pairs += 1

        n_chr = sum(1 for _, sid in snps_by_chr[chrom] if sid in s2i)
        print(f"    chr{chrom}: {n_chr} SNPs")

    print(f"  LD pairs filled from UKBB-LD: {n_ld_pairs}  "
          f"(remaining off-diagonal = 0 / independent)")

    if ld_cache:
        np.save(ld_cache, LD)
        sz = os.path.getsize(ld_cache) / 1e6
        print(f"  Cached → {ld_cache}  ({sz:.1f} MB)")

    data["LD"] = LD.astype(np.float64)


def simulate_genotypes(cfg, data, rng):
    """
    Simulate N individuals from MAF + LD using the Cholesky method.

    If data["LD"] is available:
        1. Z ~ N(0, I)               independent standard normals
        2. Z_corr = Z @ chol(LD)^T   introduce LD correlations
        3. discretize via HWE MAF thresholds → G ∈ {0,1,2}

    Fallback (no LD): independent HWE Binomial draws.
    """
    from scipy.stats import norm as sp_norm

    maf = data["maf"]
    M   = len(maf)
    LD  = data.get("LD", None)

    if LD is not None:
        LD_psd = _nearest_psd(LD)
        L      = np.linalg.cholesky(LD_psd + 1e-8 * np.eye(M))
        Z      = rng.standard_normal((cfg.N, M)) @ L.T   # (N, M), correlated

        if cfg.discretize_geno:
            q0 = (1 - maf) ** 2
            q1 = q0 + 2 * maf * (1 - maf)
            c0 = sp_norm.ppf(np.clip(q0, 1e-6, 1 - 1e-6))
            c1 = sp_norm.ppf(np.clip(q1, 1e-6, 1 - 1e-6))
            G  = (Z > c0).astype(float) + (Z > c1).astype(float)
        else:
            G = Z
    else:
        G = (rng.binomial(2, maf, size=(cfg.N, M)).astype(float)
             if cfg.discretize_geno
             else rng.normal(0.0, 1.0, size=(cfg.N, M)))

    return (G - G.mean(0)) / (G.std(0) + 1e-9)


# =============================================================================
# Section F  ── Protein-time matrix  (THE CORE: memo pipeline step 2)
#
#  P_i(t, j) = G_j @ beta_i  +  A_i * cos(2π(t-φ_i)/24)  +  noise
#
#  P_avg[:,i] = mean_t P_i(t,j)  ≈  G_j @ beta_i + noise
#               (circadian part averages to zero)
#               → used for PWAS ("average board pQTL")
#
#  pop_mean[i,t] = mean_j P_i(t,j)  ≈  A_i * cos(...) + small_noise
#                  → fed to MetaCycle → R²
# =============================================================================
def simulate_protein_time_matrix(cfg, G, data, rp, rng):
    """
    rp: dict from learn_rhythm_params()
    Returns
    -------
    P_avg      (N × P)  standardized average protein levels  → PWAS input
    pop_mean   (P × T)  population-mean time series          → MetaCycle input
    A          (P,)     true amplitudes  (causal selection uses this)
    phi        (P,)     true phases
    P_genetic  (N × P)  genetic component before adding noise
    has_signal (P,)     bool – proteins with non-trivial genetic signal
    """
    beta  = data["beta"]          # (M, P)
    P     = beta.shape[1]
    # Use the same time points as the real data so pop_mean is consistent
    # with what compute_r2 expects
    t     = np.array(rp["timepoints"], float)
    T     = len(t)
    omega = 2 * np.pi / 24.0

    # ── Draw synthetic circadian parameters for each protein ─────────────────
    A       = np.exp(rng.normal(rp["mu_log_amp"],   rp["sd_log_amp"],   P))
    phi     = rng.uniform(0.0, 24.0, P)
    nsd     = np.exp(rng.normal(rp["mu_log_noise"], rp["sd_log_noise"], P))
    mesor   = rng.normal(rp["mu_mesor"], rp["sd_mesor"], P)

    # ── Genetic component: (N × P) ───────────────────────────────────────────
    gv = G @ beta
    sd = gv.std(0)
    has_signal = sd > 1e-8
    gv[:, has_signal] /= sd[has_signal]   # std = 1 where signal exists

    # ── Average protein level (circadian component cancels) ──────────────────
    # P_avg = sqrt(h2) * gv + sqrt(1-h2) * noise   (exactly as before)
    e     = rng.standard_normal(gv.shape)
    P_avg = np.sqrt(cfg.h2_pqtl) * gv + np.sqrt(1.0 - cfg.h2_pqtl) * e
    P_avg = (P_avg - P_avg.mean(0)) / (P_avg.std(0) + 1e-9)

    # ── Population-mean circadian profile ────────────────────────────────────
    # mean_j(G_j @ beta_i) ≈ 0  (G is zero-mean standardized)
    # mean_j(noise_ij(t))  ≈ 0  (large N)
    # → pop_mean[i,t]  ≈  A_i * cos(2π(t-φ_i)/24) + measurement_noise
    pop_mean = np.zeros((P, T))
    for i in range(P):
        circ         = A[i] * np.cos(omega * (t - phi[i]))
        meas_noise   = rng.normal(0.0, nsd[i], T)
        pop_mean[i]  = mesor[i] + circ + meas_noise

    return P_avg, pop_mean, A, phi, gv, has_signal


# =============================================================================
# Section G  ── MetaCycle / cosinor on pop_mean → R²
# =============================================================================
def _have_metacycle():
    try:
        r = subprocess.run(["Rscript", "-e",
                            "suppressMessages(library(MetaCycle))"],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def _metacycle(profiles, timepoints):
    tpstr = ",".join(str(int(x)) for x in timepoints)
    with tempfile.TemporaryDirectory() as d:
        inf  = os.path.join(d, "in.csv")
        outf = os.path.join(d, "out.csv")
        ids  = [f"P{i}" for i in range(profiles.shape[0])]
        cols = [f"t{int(x)}" for x in timepoints]
        pd.DataFrame(profiles, index=ids, columns=cols)\
          .reset_index().rename(columns={"index": "id"})\
          .to_csv(inf, index=False)
        subprocess.run(["Rscript", RSCRIPT, inf, outf, tpstr],
                       capture_output=True, check=True, timeout=3600)
        out = pd.read_csv(outf)
    order = {f"P{i}": i for i in range(profiles.shape[0])}
    return out.assign(_o=out["id"].map(order)).sort_values("_o")["R2"].to_numpy()


def _cosinor_r2(profiles, timepoints):
    t     = np.array(timepoints, float)
    omega = 2 * np.pi / 24.0
    X     = np.column_stack([np.ones_like(t),
                              np.cos(omega * t), np.sin(omega * t)])
    H  = X @ np.linalg.pinv(X)
    r2 = np.empty(profiles.shape[0])
    for i, y in enumerate(profiles):
        fit   = H @ y
        sst   = np.sum((y - y.mean()) ** 2)
        r2[i] = 0.0 if sst <= 0 else max(0.0, 1 - np.sum((y-fit)**2) / sst)
    return r2


def compute_r2(cfg, pop_mean, timepoints, cache_path=None):
    """Score pop_mean profiles with MetaCycle (or cosinor)."""
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached R² from {cache_path}")
        return pd.read_csv(cache_path)["R2"].to_numpy()

    if cfg.use_metacycle and os.path.exists(RSCRIPT) and _have_metacycle():
        print("  Running MetaCycle on population-mean profiles ...")
        r2 = _metacycle(pop_mean, timepoints)
    else:
        print("  Using Python cosinor fallback ...")
        r2 = _cosinor_r2(pop_mean, timepoints)

    if cache_path:
        pd.DataFrame({"R2": r2}).to_csv(cache_path, index=False)
    return r2


# =============================================================================
# Section H  ── Matched pairs  (non-trivial version)
#
#  Two proteins share the SAME pQTL beta vector → same expected Z-score.
#  But they have DIFFERENT amplitude A → different R².
#  Independent noise means Z_lo ≠ Z_hi exactly (non-trivial).
#  Correct rate < 100% in general → meaningful result.
# =============================================================================
def inject_contrast_pairs(cfg, P_avg, gv, has_signal, A, rng):
    lo = np.percentile(A, cfg.low_amp_pct)
    hi = np.percentile(A, cfg.high_amp_pct)
    lo_pool = [p for p in range(len(A)) if A[p] <= lo and has_signal[p]]
    hi_pool = [p for p in range(len(A)) if A[p] >= hi and has_signal[p]]
    rng.shuffle(lo_pool); rng.shuffle(hi_pool)
    n = min(cfg.n_pairs, len(lo_pool), len(hi_pool))

    pairs = []
    for k in range(n):
        p_lo, p_hi = lo_pool[k], hi_pool[k]
        # Give p_lo the SAME genetic component as p_hi
        shared_gv = gv[:, p_hi].copy()
        shared_gv = shared_gv / (shared_gv.std() + 1e-9)
        # Independent noise → Z scores similar but NOT identical
        noise_lo = rng.standard_normal(cfg.N)
        col = (np.sqrt(cfg.h2_pqtl) * shared_gv
               + np.sqrt(1.0 - cfg.h2_pqtl) * noise_lo)
        P_avg[:, p_lo] = (col - col.mean()) / (col.std() + 1e-9)
        pairs.append((p_lo, p_hi))
    return pairs


# =============================================================================
# Section I  ── PWAS + weighting
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


def select_causal(cfg, A, scenario, has_signal, rng):
    """Causal selection based on TRUE amplitude A, NOT R². Breaks circularity."""
    idx = np.where(has_signal)[0]
    n   = min(cfg.n_causal, len(idx) - 1)
    if n <= 0:
        return np.array([], dtype=int)
    if scenario == "circadian":
        z    = (A[idx] - A[idx].mean()) / (A[idx].std() + 1e-9)
        prob = np.exp(cfg.select_lambda * z); prob /= prob.sum()
        return rng.choice(idx, size=n, replace=False, p=prob)
    return rng.choice(idx, size=n, replace=False)


def make_phenotype(cfg, P_avg, A, scenario, has_signal, rng):
    causal = select_causal(cfg, A, scenario, has_signal, rng)
    if scenario == "null" or len(causal) == 0:
        return rng.standard_normal(P_avg.shape[0]), causal
    gamma = rng.normal(0.0, 1.0, len(causal))
    sig   = (P_avg[:, causal] * gamma).sum(1)
    sig   = (sig - sig.mean()) / (sig.std() + 1e-9)
    y     = (np.sqrt(cfg.h2_trait) * sig
             + np.sqrt(1.0 - cfg.h2_trait) * rng.standard_normal(cfg.N))
    return y, causal


def compute_delta_auc(Z, obs_r2, causal, n_proteins, tau):
    labels = np.zeros(n_proteins); labels[causal] = 1
    if labels.sum() == 0 or labels.sum() == n_proteins:
        return 0.0, 0.5, 0.5
    w     = make_weight(obs_r2, tau)
    auc_v = roc_auc_score(labels, np.abs(Z))
    auc_w = roc_auc_score(labels, np.abs(Z) * np.sqrt(w))
    return auc_w - auc_v, auc_v, auc_w


def evaluate_pairs(cfg, Z, obs_r2, pairs, A, return_detail=False):
    n_proteins = len(Z)
    van_rank   = pd.Series(np.abs(Z)).rank(ascending=False).to_numpy()
    w          = make_weight(obs_r2, cfg.tau)
    wtd_score  = np.abs(Z) * np.sqrt(w)
    wtd_rank   = pd.Series(wtd_score).rank(ascending=False).to_numpy()

    records = []
    for pid, (a, b) in enumerate(pairs):
        z_diff = abs(float(np.abs(Z)[a]) - float(np.abs(Z)[b]))
        records.append({
            "pair":              pid,
            "A_lo":              float(A[a]),
            "A_hi":              float(A[b]),
            "r2_lo":             float(obs_r2[a]),
            "r2_hi":             float(obs_r2[b]),
            "vanilla_score_lo":  float(np.abs(Z)[a]),
            "vanilla_score_hi":  float(np.abs(Z)[b]),
            "vanilla_z_diff":    z_diff,       # NOT zero – non-trivial
            "vanilla_rank_lo":   int(van_rank[a]),
            "vanilla_rank_hi":   int(van_rank[b]),
            "weighted_rank_lo":  int(wtd_rank[a]),
            "weighted_rank_hi":  int(wtd_rank[b]),
            "weighted_correct":  wtd_score[b] > wtd_score[a],
            "rank_improvement":  int(wtd_rank[a]) - int(wtd_rank[b]),
        })
    detail = pd.DataFrame(records)
    summary = {
        "n_pairs":               len(pairs),
        "mean_vanilla_z_diff":   float(detail["vanilla_z_diff"].mean()),
        "weighted_correct_rate": float(detail["weighted_correct"].mean()),
        "mean_A_lo":             float(detail["A_lo"].mean()),
        "mean_A_hi":             float(detail["A_hi"].mean()),
        "mean_r2_lo":            float(detail["r2_lo"].mean()),
        "mean_r2_hi":            float(detail["r2_hi"].mean()),
        "mean_rank_improvement": float(detail["rank_improvement"].mean()),
    }
    if return_detail:
        return summary, detail
    return summary


# =============================================================================
# Driver
# =============================================================================
def run_simulation(cfg, data, rp):
    rng   = np.random.default_rng(cfg.seed)
    Pn    = len(data["proteins"])

    print("Simulating genotypes ...")
    G = simulate_genotypes(cfg, data, rng)

    print("Building protein-time matrix (genetic + circadian) ...")
    P_avg, pop_mean, A, phi, gv, has_signal = simulate_protein_time_matrix(
        cfg, G, data, rp, rng)
    print(f"  P_avg shape: {P_avg.shape}  "
          f"(circadian component averaged out → input to PWAS)")
    print(f"  Amplitude range: [{A.min():.3f}, {A.max():.3f}], "
          f"median {np.median(A):.3f}")

    print("Running MetaCycle on population-mean profiles → R² ...")
    r2 = compute_r2(cfg, pop_mean, rp["timepoints"])
    corr_r2_A = float(np.corrcoef(r2, A)[0, 1])
    print(f"  R² range: [{r2.min():.3f}, {r2.max():.3f}], "
          f"median {np.median(r2):.3f}")
    print(f"  Corr(R², true amplitude A): {corr_r2_A:.3f}  "
          f"← R² is a NOISY proxy of A (not identical)")

    print("Injecting matched pairs (same genetic effect, different amplitude) ...")
    pairs = inject_contrast_pairs(cfg, P_avg, gv, has_signal, A, rng)
    print(f"  {len(pairs)} pairs  "
          f"(A_lo ≤ {np.percentile(A, cfg.low_amp_pct):.3f}, "
          f" A_hi ≥ {np.percentile(A, cfg.high_amp_pct):.3f})")
    print(f"  Vanilla Z-scores will NOT be exactly equal (independent noise)")

    scenarios = ["circadian", "random", "null"]
    rows, pair_results = [], []
    snapshot = snapshot_pairs = None

    print(f"Running {cfg.n_reps} reps × {len(scenarios)} scenarios ...")
    for sc in scenarios:
        for rep in range(cfg.n_reps):
            rng_rep   = np.random.default_rng(cfg.seed + 1000 * rep + hash(sc) % 997)
            y, causal = make_phenotype(cfg, P_avg, A, sc, has_signal, rng_rep)
            Z         = pwas_z(P_avg, y)
            obs_r2    = np.clip(r2 + rng_rep.normal(0, cfg.prior_noise_sd, Pn), 0, None)
            d, av, aw = compute_delta_auc(Z, obs_r2, causal, Pn, cfg.tau)
            rows.append((sc, rep, d, av, aw))

            if sc == "circadian":
                if rep == 0:
                    sp, detail_p = evaluate_pairs(
                        cfg, Z, obs_r2, pairs, A, return_detail=True)
                    pair_results.append(sp)
                    snapshot_pairs = detail_p
                    w_snap = make_weight(obs_r2, cfg.tau)
                    snapshot = pd.DataFrame({
                        "protein":        data["proteins"],
                        "amplitude_A":    A,
                        "r2":             obs_r2,
                        "vanilla_score":  np.abs(Z),
                        "vanilla_rank":   pd.Series(np.abs(Z)).rank(
                                              ascending=False).astype(int).to_numpy(),
                        "weighted_score": np.abs(Z) * np.sqrt(w_snap),
                        "weighted_rank":  pd.Series(np.abs(Z) * np.sqrt(w_snap)).rank(
                                              ascending=False).astype(int).to_numpy(),
                        "is_causal":      np.isin(np.arange(Pn), causal).astype(int),
                    }).sort_values("weighted_rank")
                else:
                    pair_results.append(
                        evaluate_pairs(cfg, Z, obs_r2, pairs, A))

    df   = pd.DataFrame(rows, columns=["scenario", "rep", "delta_auc",
                                       "auc_vanilla", "auc_weighted"])
    summ = (df.groupby("scenario")["delta_auc"]
              .agg(n="count", mean_delta_auc="mean", sd="std").reset_index())
    summ["se"] = summ["sd"] / np.sqrt(summ["n"])

    keys = ("mean_vanilla_z_diff", "weighted_correct_rate",
            "mean_A_lo", "mean_A_hi", "mean_r2_lo", "mean_r2_hi",
            "mean_rank_improvement")
    pair_summ = {k: float(np.mean([x[k] for x in pair_results]))
                 for k in keys} if pair_results else {}

    return dict(df=df, summary=summ, pair_summary=pair_summ,
                snapshot=snapshot, snapshot_pairs=snapshot_pairs,
                r2=r2, A=A)


def print_results(res, cfg):
    print(f"\n{'='*62}")
    print("RESULTS  Delta AUC by scenario")
    print(f"{'='*62}")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(res["summary"].to_string(index=False))

    if res["pair_summary"]:
        c = res["pair_summary"]
        print(f"\n{'='*62}")
        print("RESULTS  Matched-pair contrast")
        print(f"{'='*62}")
        print(f"  Pairs: {int(c.get('n_pairs', cfg.n_pairs))}")
        print(f"  Mean true amplitude  A_lo / A_hi: "
              f"{c['mean_A_lo']:.3f} / {c['mean_A_hi']:.3f}")
        print(f"  Mean MetaCycle R²   lo  / hi:     "
              f"{c['mean_r2_lo']:.3f} / {c['mean_r2_hi']:.3f}")
        print(f"  Mean vanilla |Z| difference:      "
              f"{c['mean_vanilla_z_diff']:.4f}  "
              f"← NOT zero (independent noise, non-trivial)")
        print(f"  Weighted ranks hi-A higher:       "
              f"{c['weighted_correct_rate']:.3f}  "
              f"(random chance = 0.5)")
        print(f"  Mean rank improvement (hi-A):     "
              f"+{c['mean_rank_improvement']:.1f} places")

    print(f"\n{'='*62}")
    print("INTERPRETATION")
    print(f"{'='*62}")
    sc_df = res["summary"].set_index("scenario")
    for sc, label, expect in [
        ("circadian", "Circadian scenario", "> 0"),
        ("random",    "Random scenario   ", "≈ 0"),
        ("null",      "Null scenario     ", "≈ 0"),
    ]:
        if sc in sc_df.index:
            d  = sc_df.loc[sc, "mean_delta_auc"]
            se = sc_df.loc[sc, "se"]
            ok = ("✓" if (sc == "circadian" and d > 2 * se)
                  else "✓" if (sc != "circadian" and abs(d) < 2 * se)
                  else "!")
            print(f"  {label}  ΔAuC = {d:+.4f} ± {se:.4f}  "
                  f"[expected {expect}]  {ok}")

    if res.get("snapshot_pairs") is not None:
        sp = res["snapshot_pairs"]
        print(f"\n{'='*62}")
        print("CONCRETE EXAMPLE  (rep 0, circadian scenario)")
        print("Same genetic effect → similar (not identical) vanilla Z")
        print(f"{'='*62}")
        cols = ["pair", "A_lo", "A_hi",
                "vanilla_score_lo", "vanilla_score_hi", "vanilla_z_diff",
                "vanilla_rank_lo", "vanilla_rank_hi",
                "weighted_rank_lo", "weighted_rank_hi", "weighted_correct"]
        show = sp[cols].head(10).copy()
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}",
                               "display.width", 140):
            print(show.to_string(index=False))
        print()
        print("  vanilla_z_diff > 0    → scores differ (NOT forced equal)")
        print("  weighted_rank_hi < weighted_rank_lo → rhythmic protein ranks higher")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles",      default=D_PROFILES)
    ap.add_argument("--pqtl",          default=D_PQTL)
    ap.add_argument("--ldref",         default=D_LDREF)
    ap.add_argument("--ukbb-ld",       default=D_UKBB_LD,
                    help="UKBB-LD directory (chr*_*_*.gz + .npz files)")
    ap.add_argument("--chain",         default=D_CHAIN,
                    help="hg38ToHg19 chain file for liftover (default: %(default)s)")
    ap.add_argument("--out",           default=os.path.join(HERE, "results_sim3"))
    ap.add_argument("--cache-dir",     default=None)
    ap.add_argument("--reps",          type=int,   default=Config.n_reps)
    ap.add_argument("--n-individuals", type=int,   default=Config.N)
    ap.add_argument("--tau",           type=float, default=Config.tau)
    ap.add_argument("--no-metacycle",  action="store_true")
    ap.add_argument("--no-ld",         action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.n_reps        = args.reps
    cfg.N             = args.n_individuals
    cfg.tau           = args.tau
    cfg.use_metacycle = not args.no_metacycle
    cfg.use_real_ld   = not args.no_ld

    os.makedirs(args.out, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.out, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print("Step 1  Learning circadian parameter distributions from real data")
    print(f"{'='*62}")
    rp = learn_rhythm_params(args.profiles)
    print(f"  Amplitude: log-normal  μ={rp['mu_log_amp']:.3f}  σ={rp['sd_log_amp']:.3f}")
    print(f"  Noise:     log-normal  μ={rp['mu_log_noise']:.3f}  σ={rp['sd_log_noise']:.3f}")
    print(f"  MESOR:     normal      μ={rp['mu_mesor']:.2f}  σ={rp['sd_mesor']:.2f}")
    print(f"  Time points from real data: {list(rp['timepoints'].astype(int))}")

    print(f"\n{'='*62}")
    print("Step 2  Loading pQTL data")
    print(f"{'='*62}")
    pqtl_df = load_pqtl(args.pqtl)
    print(f"  {pqtl_df['gene'].nunique()} proteins, "
          f"{pqtl_df['snp_id'].nunique()} unique SNPs")

    print(f"\n{'='*62}")
    print("Step 3  Loading MAF from 1000G EUR")
    print(f"{'='*62}")
    if cfg.use_real_ld:
        maf_series, _ = load_maf_ld(pqtl_df, args.ldref,
                                     chain_file=args.chain,
                                     ukbb_dir=args.ukbb_ld,
                                     cache_dir=cache_dir)
    else:
        snps = pqtl_df["snp_id"].unique()
        maf_series = pd.Series(
            np.random.default_rng(0).uniform(0.05, 0.45, len(snps)), index=snps)
        print(f"  --no-ld: random MAF for {len(snps)} SNPs")

    print(f"\n{'='*62}")
    print("Step 4  Harmonizing")
    print(f"{'='*62}")
    data = harmonize(pqtl_df, maf_series)

    print(f"\n{'='*62}")
    print("Step 4.5  Loading LD from UKBB-LD")
    print(f"{'='*62}")
    if cfg.use_real_ld and os.path.isdir(args.ukbb_ld):
        load_ukbb_ld(data, args.ukbb_ld, cache_dir=cache_dir)
        if data.get("LD") is not None:
            print(f"  LD matrix ready: {data['LD'].shape}  "
                  f"(Cholesky simulation enabled)")
        else:
            print("  LD unavailable → independent HWE fallback")
    else:
        data["LD"] = None
        print("  Skipping LD loading")

    res = run_simulation(cfg, data, rp)
    print_results(res, cfg)

    # ── Save ──────────────────────────────────────────────────────────────────
    res["df"].to_csv(
        os.path.join(args.out, "delta_auc_replicates.csv"), index=False)
    res["summary"].to_csv(
        os.path.join(args.out, "scenario_summary.csv"), index=False)
    if res["pair_summary"]:
        pd.DataFrame([res["pair_summary"]]).to_csv(
            os.path.join(args.out, "pair_contrast.csv"), index=False)
    if res.get("snapshot_pairs") is not None:
        res["snapshot_pairs"].to_csv(
            os.path.join(args.out, "pair_ranking_detail.csv"), index=False)
    if res.get("snapshot") is not None:
        res["snapshot"].to_csv(
            os.path.join(args.out, "full_ranking_snapshot.csv"), index=False)

    pd.DataFrame({"protein": data["proteins"],
                  "amplitude_A": res["A"],
                  "R2": res["r2"],
                  "n_pqtl": (data["beta"] != 0).sum(0)
                  }).to_csv(os.path.join(args.out, "protein_info.csv"), index=False)

    print(f"\nResults saved to {args.out}/")


if __name__ == "__main__":
    main()
