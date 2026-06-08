import pandas as pd
import sys

pqtl_file = sys.argv[1]
gwas_file = sys.argv[2]
ldref_file = sys.argv[3]

print("Loading data...")

pqtl = pd.read_csv(pqtl_file, sep="\t")
gwas = pd.read_csv(gwas_file, sep=r"\s+")
ld = pd.read_csv(ldref_file, sep="\t")

# 统一列名
pqtl = pqtl.rename(columns={
    "ID": "SNP",
    "BETA": "beta_pqtl",
    "SE": "se_pqtl",
    "P": "p_pqtl"
})

gwas = gwas.rename(columns={
    "ID": "SNP",
    "BETA": "beta_gwas",
    "SE": "se_gwas",
    "P": "p_gwas"
})

# 去掉 LDREF 中 palindromic
ld = ld[ld["palindromic"] == 0].copy()

print("pQTL SNP count:", len(pqtl))
print("GWAS SNP count:", len(gwas))
print("LDREF non-palindromic count:", len(ld))

# 先做 pQTL 和 GWAS 的交集
pqtl_gwas = pqtl.merge(gwas, on="SNP")
print("pQTL ∩ GWAS:", len(pqtl_gwas))

# exact match
df_exact = pqtl_gwas.merge(ld, left_on="SNP", right_on="snp_a1a2")
df_exact["match_type"] = "exact"

# flip match
df_flip = pqtl_gwas.merge(ld, left_on="SNP", right_on="snp_a2a1")
df_flip["match_type"] = "flip"

# 合并
df = pd.concat([df_exact, df_flip], ignore_index=True)

# 去重，防止极少数情况下重复
df = df.drop_duplicates(subset=["SNP", "match_type"])

print("Exact match:", len(df_exact))
print("Flip match:", len(df_flip))
print("Total usable SNP:", len(df))

# flip 时把 GWAS 和 pQTL 的效应都翻转到 LDREF 的 a1:a2 方向
df.loc[df["match_type"] == "flip", "beta_gwas"] *= -1
df.loc[df["match_type"] == "flip", "beta_pqtl"] *= -1

# 构造 z 分数
df["z_gwas"] = df["beta_gwas"] / df["se_gwas"]
df["z_pqtl"] = df["beta_pqtl"] / df["se_pqtl"]

# 保存
df.to_csv("harmonized_snps.tsv", sep="\t", index=False)
print("Saved: harmonized_snps.tsv")
