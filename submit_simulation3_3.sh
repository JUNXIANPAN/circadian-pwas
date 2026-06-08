#!/bin/bash
# ============================================================
# submit_simulation3_3.sh
# 跑 simulation3_3.py 的 SLURM 提交脚本（10次重复试跑版）
# ============================================================
#SBATCH -J sim3_3_test
#SBATCH -p tosa,izumo,hiyama,nishio,yambaru,yame,guri,uji
#SBATCH --qos short
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/sim3_3_%j.out
#SBATCH -e logs/sim3_3_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

echo "==== simulation3_3: synthetic circadian PWAS (test run, 10 reps) ===="
conda activate pwas_env

# 清理合并后的MAF缓存（ensembl_maf 和 pqtl_ld 永久保存，不删）
rm -f results_sim3/cache/maf.csv \
      results_sim3/cache/ld_blocks.npz
# 注：results_sim3/cache/pqtl_ld.npy 首次生成后保留，无需重建

python scripts/simulation3/simulation3_3.py \
    --profiles  raw_data/circadian_info/report.pg_matrix.tsv \
    --pqtl      work/pqtl_topk.csv \
    --ukbb-ld   /data/CommonData/ukbb-ld \
    --chain     reference/chain/hg38ToHg19.over.chain.gz \
    --reps      200 \
    --out       results_sim3_night

echo ""
echo "==== Done. Output: results_sim3 ===="
echo "Key files:"
echo "  results_sim3/scenario_summary.csv"
echo "  results_sim3/pair_ranking_detail.csv"
echo "  results_sim3/full_ranking_snapshot.csv"
seff "${SLURM_JOB_ID}" || true
