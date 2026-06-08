#!/usr/bin/env bash
set -euo pipefail

PQTL_DIR="raw_data/pqtl/brain/brain_cis_pqtl"
RUN_ONE="scripts/analysis/run_one_protein.sh"
OUT="results/all_proteins_otters.tsv"
LOGDIR="work/logs_all"

mkdir -p "${LOGDIR}"
mkdir -p "results"

echo "=============================="
echo "[INFO] 开始批量运行所有蛋白"
echo "[INFO] pQTL目录: ${PQTL_DIR}"
echo "=============================="

# 找到所有 protein ID
mapfile -t PROTS < <(
  find "${PQTL_DIR}" -maxdepth 1 -type f -name 'reducepqtls.*.cis.glm.linear.gz' \
  | sed 's#.*/reducepqtls\.\(.*\)\.cis\.glm\.linear\.gz#\1#' \
  | sort
)

echo "[INFO] 检测到蛋白数量: ${#PROTS[@]}"

if [[ ${#PROTS[@]} -eq 0 ]]; then
  echo "[ERROR] 没有找到任何 pQTL 文件"
  exit 1
fi

# 清空旧汇总文件
echo -e "protein\tZ\tP\tn_snp" > "${OUT}"

# 并行数，可自己改
NJOB=4

printf "%s\n" "${PROTS[@]}" | xargs -n 1 -P "${NJOB}" -I {} bash -c '
  prot="{}"
  echo "[START] ${prot}"

  if bash "'"${RUN_ONE}"'" "${prot}" > "'"${LOGDIR}"'/${prot}.log" 2>&1; then
    result_file="work/test_single_protein/${prot}/results/${prot}_otters_mini.tsv"
    if [[ -f "${result_file}" ]]; then
      awk "NR==2{print \$1\"\t\"\$4\"\t\"\$5\"\t\"\$3}" "${result_file}" >> "'"${OUT}"'"
      echo "[DONE] ${prot}"
    else
      echo "[FAIL] ${prot} 结果文件不存在"
    fi
  else
    echo "[FAIL] ${prot} 运行失败"
  fi
'

echo "=============================="
echo "[INFO] 批量运行结束"
echo "[INFO] 汇总结果: ${OUT}"
echo "=============================="