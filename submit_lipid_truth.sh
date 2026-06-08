#!/bin/bash
# ============================================================
# submit_lipid_truth.sh
# 单独跑 lipid_truth block, 用 per-timepoint mean 版的 cosinor annotation
# ============================================================
#SBATCH -J cpwas_lipid_v3
#SBATCH -p tosa,izumo,hiyama,nishio,yambaru,yame,guri,uji
#SBATCH --qos short
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o logs/lipid_v3_%j.out
#SBATCH -e logs/lipid_v3_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

# ---------- Sanity ----------
ANN_FILE=raw_data/circadian_info_meta2d/circadian_annotation_per_tp_mean.csv
if [[ ! -f "${ANN_FILE}" ]]; then
    echo "ERROR: ${ANN_FILE} 不存在. 请先跑 per-timepoint mean cosinor 脚本生成."
    exit 1
fi

# ---------- Run ----------
echo "==== lipid_truth block (per-timepoint mean cosinor) ===="
conda activate pwas_env

python scripts/simulation2/final_simulation_v2.py \
    --block       lipid_truth \
    --pg-matrix   "${ANN_FILE}" \
    --pqtl        work/pqtl_topk.csv \
    --ld-dir      /data/CommonData/ukbb-ld/ \
    --outdir      results_lipid_truth_v3 \
    --n-reps      200 \
    --sigma-log-w 0.5 \
    --seed        20260507

echo ""
echo "==== Done. Output: results_lipid_truth_v3 ===="
echo "Key file: results_lipid_truth_v3/final_v2_lipid_truth_summary.csv"
seff "${SLURM_JOB_ID}" || true
