#!/bin/bash
# ============================================================
# submit_simulation3_4.sh
# simulation3_4: non-mediated pathway + tagging scenario
# ============================================================
#SBATCH -J sim3_4
#SBATCH -p tosa,izumo,hiyama,nishio,yambaru,yame,guri,uji
#SBATCH --qos short
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/sim3_4_%j.out
#SBATCH -e logs/sim3_4_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

echo "==== simulation3_4: non-mediated + tagging (100 reps) ===="
conda activate pwas_env

python scripts/simulation3/simulation3_4.py \
    --profiles  raw_data/circadian_info/report.pg_matrix.tsv \
    --pqtl      work/pqtl_topk.csv \
    --ukbb-ld   /data/CommonData/ukbb-ld \
    --chain     reference/chain/hg38ToHg19.over.chain.gz \
    --reps      200 \
    --out       results_sim4_night

echo ""
echo "==== Done. Output: results_sim4 ===="
echo "Key files:"
echo "  results_sim4/scenario_summary.csv"
echo "  results_sim4/tagging_summary.csv"
echo "  results_sim4/tagging_detail.csv"
seff "${SLURM_JOB_ID}" || true
