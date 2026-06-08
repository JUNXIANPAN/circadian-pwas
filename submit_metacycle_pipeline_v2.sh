#!/bin/bash
# ============================================================
# submit_metacycle_pipeline_v2.sh
#
# v2 升级 (相对于 v4):
#   1. max_ld_snps 默认 500 → 1500 (减少 SNP 重用导致的人为蛋白相关)
#   2. NEW block: lipid_truth (用血脂 GWAS 作为外部 truth, anti-circ level 2)
#   3. 调用 final_simulation_v2.py 而不是 final_simulation.py
#
# 数据参数 (沿用 v4):
#   - 9 timepoints: 9, 12, 15, 18, 21, 24, 27, 30, 33 (3h interval, 24h span)
#   - 759 effective proteins
#   - MetaCycle: LS only (period fixed at 24h)
#
# 重要前提:
#   - final_simulation_v2.py 必须和 circadian_pwas_simulation_phase_causal.py
#     在同一目录 (它 import 那个 helper)
#   - lipid_truth block 需要 ann 表里有 gene 名列;
#     如果 MetaCycle output 没有, 脚本会在 sanity check 阶段报错并退出
# ============================================================
#SBATCH -J cpwas_meta2d_v2
#SBATCH -p tosa,izumo,hiyama,nishio,yambaru,yame,guri,uji
#SBATCH --qos short
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 12:00:00                       # 增加到 12h: lipid_truth 是新增, 1500 SNPs 也变慢
#SBATCH -o logs/meta2d_v2_%j.out
#SBATCH -e logs/meta2d_v2_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

PG_CLEAN=raw_data/circadian_info/pg_matrix_clean.tsv
PG_RAW=raw_data/circadian_info/report.pg_matrix.tsv      # NEW: 需要原始 pg_matrix 拿 gene 列
TIMEPOINTS="9,12,15,18,21,24,27,30,33"
META2D_OUT=raw_data/circadian_info_meta2d/circadian_annotation_meta2d.csv
META2D_OUT_WITH_GENES=raw_data/circadian_info_meta2d/circadian_annotation_meta2d_with_genes.csv

# ---------- 0. Sanity ----------
if [[ ! -f "${PG_CLEAN}" ]]; then
    echo "ERROR: ${PG_CLEAN} 不存在. 请先跑预处理."
    exit 1
fi

if [[ ! -f "${PG_RAW}" ]]; then
    echo "ERROR: ${PG_RAW} 不存在 (lipid_truth block 需要从这里拿 gene 名)."
    exit 1
fi

# ---------- 1. MetaCycle ----------
if [[ ! -f "${META2D_OUT}" ]]; then
    echo "==== [1/4] Running MetaCycle (meta2d_env) ===="
    conda activate meta2d_env
    mkdir -p "$(dirname ${META2D_OUT})"
    Rscript scripts/simulation2/run_metacycle.R \
        --input       "${PG_CLEAN}" \
        --timepoints  "${TIMEPOINTS}" \
        --output      "${META2D_OUT}" \
        --id-col      protein_id \
        --methods     "LS" \
        --minper      24 \
        --maxper      24 \
        --workdir     /tmp/meta2d_work_${SLURM_JOB_ID}
    conda deactivate
else
    echo "==== [1/4] MetaCycle output 已存在, 跳过 ===="
fi

# ---------- 2. NEW: 把 gene 名加到 MetaCycle output ----------
# lipid_truth block 需要 gene 列来匹配血脂 GWAS 基因列表.
# 这一步从原 pg_matrix 抽 Protein.Group + Genes, 然后 left-join 到 MetaCycle 输出.
if [[ ! -f "${META2D_OUT_WITH_GENES}" ]]; then
    echo "==== [2/4] Augmenting MetaCycle output with gene names ===="
    conda activate pwas_env
    python - <<PYEOF
import pandas as pd
import sys

raw = pd.read_csv("${PG_RAW}", sep="\t", usecols=["Protein.Group", "Genes"])
raw = raw.rename(columns={"Protein.Group": "protein_id", "Genes": "gene"})
raw["protein_id"] = raw["protein_id"].astype(str)
raw["gene"] = raw["gene"].astype(str)

meta = pd.read_csv("${META2D_OUT}")
meta["protein_id"] = meta["protein_id"].astype(str)

merged = meta.merge(raw, on="protein_id", how="left")
n_with_gene = merged["gene"].notna().sum()
print(f"  merged: {len(merged)} rows total, {n_with_gene} with gene names")

if n_with_gene < 0.5 * len(merged):
    print(f"  WARNING: less than half of proteins got gene names. "
          f"Check protein_id key matches between files.", file=sys.stderr)

merged.to_csv("${META2D_OUT_WITH_GENES}", index=False)
print(f"  wrote {len(merged)} rows to ${META2D_OUT_WITH_GENES}")
PYEOF
    conda deactivate
else
    echo "==== [2/4] gene-augmented file 已存在, 跳过 ===="
fi

# ---------- 3. Prep ----------
RUN_TAG=meta2d_v2_${SLURM_JOB_ID}
OUTDIR=results_${RUN_TAG}
mkdir -p "${OUTDIR}/main" "${OUTDIR}/lipid_truth" "${OUTDIR}/sigma_sweep"

# ---------- 4. Simulation v2 ----------
echo "==== [3/4] Running final_simulation_v2.py ===="
conda activate pwas_env

# ---- 4a. Main block (1500 SNPs new default) ----
echo "---- main block ----"
# python scripts/simulation2/final_simulation_v2.py \
#     --block       main \
#     --pg-matrix   "${META2D_OUT_WITH_GENES}" \
#     --pqtl        work/pqtl_topk.csv \
#     --ld-dir      /data/CommonData/ukbb-ld/ \
#     --outdir      "${OUTDIR}/main" \
#     --n-reps      200 \
#     --sigma-log-w 0.5 \
#     --seed        20260507
    # NOTE: --max-ld-snps default 1500 (v2 new). 若想对比旧设置, 加 --max-ld-snps 500.

# ---- 4b. NEW: lipid_truth block (anti-circularity level 2) ----
echo "---- lipid_truth block (NEW: external lipid GWAS truth) ----"
python scripts/simulation2/final_simulation_v2.py \
    --block       lipid_truth \
    --pg-matrix   "${META2D_OUT_WITH_GENES}" \
    --pqtl        work/pqtl_topk.csv \
    --ld-dir      /data/CommonData/ukbb-ld/ \
    --outdir      "${OUTDIR}/lipid_truth" \
    --n-reps      20 \
    --sigma-log-w 1.0 \
    --seed        20260507

# ---- 4c. Sigma sweep (sensitivity) ----
echo "---- sigma sweep ----"
# python scripts/simulation2/final_simulation_v2.py \
#     --block       sigma_sweep \
#     --pg-matrix   "${META2D_OUT_WITH_GENES}" \
#     --pqtl        work/pqtl_topk.csv \
#     --ld-dir      /data/CommonData/ukbb-ld/ \
#     --outdir      "${OUTDIR}/sigma_sweep" \
#     --sensitivity-reps 30 \
#     --seed        20260507

# ---------- 5. Analysis ----------
echo "==== [4/4] Analysis ===="
python scripts/simulation2/final_analysis.py "${OUTDIR}" || true

echo ""
echo "==== Done. Output: ${OUTDIR} ===="
echo "Key files to check:"
echo "  - ${OUTDIR}/main/final_v2_main_summary.csv         (主结果, 4 scenarios)"
echo "  - ${OUTDIR}/lipid_truth/final_v2_lipid_truth_summary.csv  (NEW: anti-circ L2)"
echo "  - ${OUTDIR}/sigma_sweep/final_v2_sigma_sweep_summary.csv"
echo ""
seff "${SLURM_JOB_ID}" || true
