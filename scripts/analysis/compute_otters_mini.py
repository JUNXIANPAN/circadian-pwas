import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

harmonized_file = sys.argv[1]
bim_file = sys.argv[2]
ld_file = sys.argv[3]
out_file = sys.argv[4]

protein_name = "X3220.40"

# 1. 读入 harmonized SNP 表
df = pd.read_csv(harmonized_file, sep="\t")

required_cols = ["rsid", "beta_pqtl", "beta_gwas", "se_gwas"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"harmonized_snps.tsv 缺少列: {missing}")

# 保留用于计算的列，并去掉缺失
df = df.dropna(subset=required_cols).copy()

# 如果同一个 rsid 重复，只保留第一条
df = df.drop_duplicates(subset=["rsid"]).copy()

# 2. 读入 PLINK 子集 SNP 顺序
bim = pd.read_csv(
    bim_file,
    sep=r"\s+",
    header=None,
    names=["chr", "rsid", "cm", "pos", "a1", "a2"]
)

# 3. 读入 LD 矩阵
R = np.loadtxt(ld_file)

if R.ndim == 1:
    R = np.array([[float(R)]])

if R.shape[0] != R.shape[1]:
    raise ValueError(f"LD 矩阵不是方阵: {R.shape}")

if R.shape[0] != len(bim):
    raise ValueError(
        f"LD 矩阵维度与 .bim SNP 数量不一致: R={R.shape}, bim={len(bim)}"
    )

# 4. 找出 bim 中哪些 rsid 真正在 harmonized 表里
available_rsid = set(df["rsid"])
mask = bim["rsid"].isin(available_rsid).to_numpy()

n_bim = bim["rsid"].nunique()
n_keep = int(mask.sum())

print(f"SNP in LD subset (.bim): {n_bim}")
print(f"SNP matched in harmonized table: {n_keep}")
print(f"SNP dropped from LD matrix: {n_bim - n_keep}")

if n_keep == 0:
    raise ValueError("没有任何 SNP 成功对齐到 harmonized 表，无法计算。")

# 5. 裁剪 bim 和 LD 矩阵到共同 SNP
bim_keep = bim.loc[mask].reset_index(drop=True)
R_keep = R[np.ix_(mask, mask)]

# 6. 按 bim_keep 顺序重排 harmonized 表
merged = bim_keep[["rsid"]].merge(df, on="rsid", how="left")

# 再保险检查
if merged["beta_pqtl"].isna().any() or merged["beta_gwas"].isna().any() or merged["se_gwas"].isna().any():
    bad = merged[merged[["beta_pqtl", "beta_gwas", "se_gwas"]].isna().any(axis=1)]
    bad.to_csv(out_file.replace(".tsv", ".missing_after_merge.tsv"), sep="\t", index=False)
    raise ValueError("裁剪后仍有缺失值，已输出 missing_after_merge.tsv")

if R_keep.shape[0] != len(merged):
    raise ValueError(
        f"裁剪后 LD 矩阵维度仍与 SNP 数量不一致: R={R_keep.shape}, SNP={len(merged)}"
    )

# 7. 构造向量
w = merged["beta_pqtl"].to_numpy(dtype=float)
z = (merged["beta_gwas"] / merged["se_gwas"]).to_numpy(dtype=float)

# 8. 计算统计量
numerator = float(np.dot(w, z))
denom2 = float(np.dot(w, np.dot(R_keep, w)))

if denom2 <= 0:
    raise ValueError(f"w'Rw <= 0，无法开方。当前值: {denom2}")

Z = numerator / np.sqrt(denom2)
p = 2 * norm.sf(abs(Z))

# 9. 输出结果
result = pd.DataFrame([{
    "protein": protein_name,
    "n_snp_ld": n_bim,
    "n_snp_used": len(merged),
    "Z": Z,
    "P": p,
    "numerator_wz": numerator,
    "denominator_wRw": denom2
}])

result.to_csv(out_file, sep="\t", index=False)
merged.to_csv(out_file.replace(".tsv", ".ordered_snps.tsv"), sep="\t", index=False)

# 输出被丢掉的 rsid，方便检查
dropped = bim.loc[~mask, ["rsid"]].copy()
if len(dropped) > 0:
    dropped.to_csv(out_file.replace(".tsv", ".dropped_ld_only_rsids.tsv"), sep="\t", index=False)

print(result.to_string(index=False))
print(f"Saved result to: {out_file}")
