#!/usr/bin/env bash
set -euo pipefail

# 用法:
# bash scripts/analysis/run_one_protein.sh X3220.40
#
# 第1个参数: protein ID, 例如 X3220.40
#
# 说明:
# 这个脚本假设你当前在项目根目录 pwas/ 下运行
# 它会自动从 pQTL 文件第一列 ID 中识别染色体号

if [[ $# -lt 1 ]]; then
  echo "用法: bash scripts/analysis/run_one_protein.sh <PROTEIN_ID>"
  echo "示例: bash scripts/analysis/run_one_protein.sh X3220.40"
  exit 1
fi

PROT="$1"

# ===== 路径配置 =====
PQTL_GZ="raw_data/pqtl/brain/brain_cis_pqtl/reducepqtls.${PROT}.cis.glm.linear.gz"
GWAS="raw_data/gwas/chrono_gwas_for_otters.txt"
LDREF_MAP="raw_data/pqtl/ldref/LDREF/LDREF_maps/ldref_harmonized.tsv"
CHAIN="reference/chain/hg38ToHg19.over.chain.gz"

HARMONIZE_SCRIPT="scripts/preprocessing/harmonize_and_overlap.py"
COMPUTE_SCRIPT="scripts/analysis/compute_otters_mini.py"

WORKDIR="work/test_single_protein/${PROT}"
INPUTDIR="${WORKDIR}/input"
OVERLAPDIR="${WORKDIR}/overlap"
LDDIR="${WORKDIR}/ld"
RESULTDIR="${WORKDIR}/results"
LOGDIR="${WORKDIR}/logs"

mkdir -p "${INPUTDIR}" "${OVERLAPDIR}" "${LDDIR}" "${RESULTDIR}" "${LOGDIR}"

LOGFILE="${LOGDIR}/run_one_protein.log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "=============================="
echo "[INFO] 开始运行单蛋白流程"
echo "[INFO] Protein: ${PROT}"
echo "=============================="

# ===== 0. 检查基础文件 =====
for f in "${PQTL_GZ}" "${GWAS}" "${LDREF_MAP}" "${CHAIN}" "${HARMONIZE_SCRIPT}" "${COMPUTE_SCRIPT}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] 文件不存在: $f"
    exit 1
  fi
done

if ! command -v CrossMap >/dev/null 2>&1; then
  echo "[ERROR] 未找到 CrossMap，请先安装 CrossMap"
  exit 1
fi

if ! command -v plink >/dev/null 2>&1; then
  echo "[ERROR] 未找到 plink，请先安装 plink"
  exit 1
fi

# ===== 1. 解压 pQTL =====
PQTL_TXT="${INPUTDIR}/reducepqtls.${PROT}.cis.glm.linear.txt"

echo "[INFO] Step 1: 解压 pQTL"
zcat "${PQTL_GZ}" > "${PQTL_TXT}"

# ===== 2. 自动识别染色体 =====
echo "[INFO] Step 2: 自动识别染色体"

CHR_RAW=$(awk 'NR==2{
  split($1,a,":");
  gsub(/^chr/, "", a[1]);
  print a[1]
  exit
}' "${PQTL_TXT}")

if [[ -z "${CHR_RAW}" ]]; then
  echo "[ERROR] 无法从 pQTL 文件识别染色体"
  exit 1
fi

# 统一处理 chrX / chrY / 数字染色体
case "${CHR_RAW}" in
  X|x) CHR="X" ;;
  Y|y) CHR="Y" ;;
  [0-9]*)
    CHR="${CHR_RAW}"
    ;;
  *)
    echo "[ERROR] 无法识别的染色体: ${CHR_RAW}"
    exit 1
    ;;
esac

LD_BFILE="raw_data/pqtl/ldref/LDREF/1000G.EUR.${CHR}"

for f in "${LD_BFILE}.bed" "${LD_BFILE}.bim" "${LD_BFILE}.fam"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] 自动识别到 chr${CHR}，但找不到 LDREF 文件: $f"
    exit 1
  fi
done

echo "[INFO] 自动识别到染色体: chr${CHR}"

# ===== 3. pQTL: hg38 -> hg19 =====
PQTL_BED_HG38="${INPUTDIR}/pqtl.hg38.bed"
PQTL_BED_HG19="${INPUTDIR}/pqtl.hg19.bed"
ID_MAP="${INPUTDIR}/id_map.tsv"
PQTL_HG19_TXT="${INPUTDIR}/pqtl.hg19.txt"

echo "[INFO] Step 3: 生成 hg38 BED"
awk 'BEGIN{OFS="\t"}
NR>1{
  split($1,a,":");
  chr=a[1];
  pos=a[2];
  print chr, pos-1, pos, $1
}' "${PQTL_TXT}" > "${PQTL_BED_HG38}"

echo "[INFO] Step 4: liftover 到 hg19"
CrossMap bed \
  "${CHAIN}" \
  "${PQTL_BED_HG38}" \
  "${PQTL_BED_HG19}"

echo "[INFO] Step 5: 构建 ID 映射"
awk 'BEGIN{OFS="\t"}
{
  split($4,a,":");
  ref=a[3];
  alt=a[4];
  new_id=$1 ":" $3 ":" ref ":" alt;
  print $4, new_id
}' "${PQTL_BED_HG19}" > "${ID_MAP}"

echo "[INFO] Step 6: 替换 pQTL ID"
awk 'BEGIN{FS=OFS="\t"}
FNR==NR{
  map[$1]=$2;
  next
}
NR==1{
  print;
  next
}
{
  if ($1 in map) $1=map[$1];
  print
}' "${ID_MAP}" "${PQTL_TXT}" > "${PQTL_HG19_TXT}"

# ===== 4. harmonization =====
HARMONIZED="${OVERLAPDIR}/harmonized_snps.tsv"

echo "[INFO] Step 7: harmonization"
python "${HARMONIZE_SCRIPT}" \
  "${PQTL_HG19_TXT}" \
  "${GWAS}" \
  "${LDREF_MAP}"

if [[ ! -f "harmonized_snps.tsv" ]]; then
  echo "[ERROR] harmonized_snps.tsv 未生成"
  exit 1
fi

mv -f "harmonized_snps.tsv" "${HARMONIZED}"

# ===== 5. 生成 rsid list =====
RSIDLIST="${OVERLAPDIR}/rsidlist.txt"

echo "[INFO] Step 8: 生成 rsid 列表"
python - <<PY
import pandas as pd

df = pd.read_csv("${HARMONIZED}", sep="\t")
if "rsid" not in df.columns:
    raise SystemExit("harmonized_snps.tsv 中没有 rsid 列")

rs = df["rsid"].dropna().drop_duplicates()
rs.to_csv("${RSIDLIST}", index=False, header=False)

print("n_rsid =", rs.nunique())
PY

# ===== 6. 提取 LD 子集 =====
LD_SUB_PREFIX="${LDDIR}/${PROT}_ldsub"
LD_PREFIX="${LDDIR}/${PROT}_ld"

echo "[INFO] Step 9: 提取 LD 子集"
plink \
  --bfile "${LD_BFILE}" \
  --extract "${RSIDLIST}" \
  --make-bed \
  --out "${LD_SUB_PREFIX}"

echo "[INFO] Step 10: 计算 LD 矩阵"
plink \
  --bfile "${LD_SUB_PREFIX}" \
  --r square \
  --out "${LD_PREFIX}"

# ===== 7. 计算蛋白统计量 =====
RESULT_TSV="${RESULTDIR}/${PROT}_otters_mini.tsv"

echo "[INFO] Step 11: 计算蛋白统计量"
python "${COMPUTE_SCRIPT}" \
  "${HARMONIZED}" \
  "${LD_SUB_PREFIX}.bim" \
  "${LD_PREFIX}.ld" \
  "${RESULT_TSV}"

echo "=============================="
echo "[INFO] 运行完成"
echo "[INFO] Protein: ${PROT}"
echo "[INFO] Chr: ${CHR}"
echo "[INFO] 最终结果: ${RESULT_TSV}"
echo "[INFO] 计算明细: ${RESULTDIR}/${PROT}_otters_mini.ordered_snps.tsv"
echo "[INFO] 丢失 SNP: ${RESULTDIR}/${PROT}_otters_mini.dropped_ld_only_rsids.tsv"
echo "[INFO] 日志: ${LOGFILE}"
echo "=============================="