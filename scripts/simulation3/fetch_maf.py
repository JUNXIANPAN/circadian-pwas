#!/usr/bin/env python3
"""
fetch_maf.py  ──  在登录节点上运行，预取 gnomAD-NFE MAF 并缓存

用法（登录节点，不需要 sbatch）:
  cd /data2/pan/pwas
  python3 scripts/simulation3/fetch_maf.py

完成后生成:
  results_sim3/cache/rsid_map.csv      （pQTL SNP -> rsID）
  results_sim3/cache/ensembl_maf.csv   （rsID -> gnomAD-NFE MAF）

之后再 sbatch submit_simulation3_3.sh，
Slurm 计算节点直接读缓存，不需要联网。
"""

import os, sys, gzip, re, json, time
import urllib.request, urllib.error
from collections import defaultdict

import pandas as pd

CACHE_DIR = "results_sim3/cache"
UKBB_DIR  = "/data/CommonData/ukbb-ld"
LIFTOVER  = os.path.join(CACHE_DIR, "liftover_hg19.csv")
RSID_OUT  = os.path.join(CACHE_DIR, "rsid_map.csv")
MAF_OUT   = os.path.join(CACHE_DIR, "ensembl_maf.csv")

os.makedirs(CACHE_DIR, exist_ok=True)


# ── Step 1: 读取 liftover 缓存 ────────────────────────────────────────────────
if not os.path.exists(LIFTOVER):
    print(f"ERROR: {LIFTOVER} 不存在。请先运行一次 simulation3_3.py 生成 liftover 缓存。")
    sys.exit(1)

liftover = pd.read_csv(LIFTOVER)
print(f"liftover 缓存: {len(liftover)} 个 SNP")


# ── Step 2: 匹配 UKBB-LD .gz → rsID ──────────────────────────────────────────
if os.path.exists(RSID_OUT):
    print(f"读取已有 rsID 映射: {RSID_OUT}")
    rs_df    = pd.read_csv(RSID_OUT)
    rsid_map = dict(zip(rs_df["snp_id"], rs_df["rsid"]))
else:
    print(f"在 UKBB-LD 中查找 rsID ({UKBB_DIR}) ...")

    block_idx = defaultdict(list)
    for f in os.listdir(UKBB_DIR):
        m = re.match(r"chr(\d+)_(\d+)_(\d+)\.gz$", f)
        if m:
            c, s, e = int(m.group(1)), int(m.group(2)), int(m.group(3))
            block_idx[c].append((s, e, os.path.join(UKBB_DIR, f)))

    snps_by_chr = defaultdict(dict)
    for _, r in liftover.iterrows():
        snps_by_chr[int(r["hg19_chr"])][int(r["hg19_pos"])] = r["snp_id"]

    rsid_map = {}
    for chrom in sorted(snps_by_chr):
        pos_map  = snps_by_chr[chrom]
        seen_pos = set()
        for start, end, fpath in block_idx.get(chrom, []):
            remaining = {p: sid for p, sid in pos_map.items() if p not in seen_pos}
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
                        rsid_map[remaining[pos]] = parts[0].strip()
                        seen_pos.add(pos)
        n_found = sum(1 for p in pos_map if p in seen_pos)
        print(f"  chr{chrom}: {n_found}/{len(pos_map)}")

    print(f"rsID 匹配: {len(rsid_map)} / {len(liftover)}")
    rows = [{"snp_id": k, "rsid": v} for k, v in rsid_map.items()]
    pd.DataFrame(rows).to_csv(RSID_OUT, index=False)
    print(f"已保存 → {RSID_OUT}")


# ── Step 3: Ensembl API → gnomAD-NFE MAF ─────────────────────────────────────
if os.path.exists(MAF_OUT):
    print(f"读取已有 MAF 缓存: {MAF_OUT}")
else:
    rsids      = list(set(rsid_map.values()))
    batch_size = 200
    url        = "https://rest.ensembl.org/variation/homo_sapiens"
    headers    = {"Content-Type": "application/json", "Accept": "application/json"}
    maf_res    = {}
    n_batches  = (len(rsids) + batch_size - 1) // batch_size

    print(f"Ensembl API 批量查询: {len(rsids)} 个 rsID，共 {n_batches} 批 ...")
    for i in range(0, len(rsids), batch_size):
        batch = rsids[i: i + batch_size]
        body  = json.dumps({"ids": batch}).encode()
        req   = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  第 {i//batch_size+1} 批失败: {e}，等待 5 秒后跳过")
            time.sleep(5)
            continue

        for rsid, info in data.items():
            if not isinstance(info, dict):
                continue
            best = None
            for p in info.get("populations", []):
                if p.get("population") == "gnomADg:nfe":
                    f = float(p.get("frequency", 0))
                    if 0 < f <= 0.5:
                        best = f if best is None else min(best, f)
            if best is not None:
                maf_res[rsid] = best
            elif info.get("MAF") is not None:
                maf_res[rsid] = float(info["MAF"])

        print(f"  批次 {i//batch_size+1}/{n_batches}: 已获取 {len(maf_res)} 个 MAF")
        time.sleep(0.5)

    rows = [{"rsid": k, "maf": v} for k, v in maf_res.items()]
    pd.DataFrame(rows).to_csv(MAF_OUT, index=False)
    print(f"已保存 → {MAF_OUT}  ({len(maf_res)} 个 SNP)")

print("\n完成！现在可以提交 Slurm 任务：")
print("  sbatch submit_simulation3_3.sh")
