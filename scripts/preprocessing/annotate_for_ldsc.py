import pandas as pd
import numpy as np

BIM_MAP = "raw_data/pqtl/ldref/LDREF/1000G_EUR_allbim.tsv"

def load_bim_map():
    bim = pd.read_csv(
        BIM_MAP,
        sep=r"\s+",
        header=None,
        names=["CHR", "BP", "SNP", "BIM_A1", "BIM_A2"],
        dtype={"CHR": str, "BP": int, "SNP": str, "BIM_A1": str, "BIM_A2": str}
    )
    bim = bim.drop_duplicates(subset=["CHR", "BP"], keep=False)
    return bim

def prepare_sumstats(infile, outfile):
    df = pd.read_csv(infile, sep="\t", dtype={"chr": str})

    required = ["chr", "pos", "ref", "alt", "beta_meta", "neglog10_pval_meta"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{infile} 缺少列: {missing}")

    df = df[required].copy()
    df = df.rename(columns={
        "chr": "CHR",
        "pos": "BP",
        "ref": "A2",
        "alt": "A1",
        "beta_meta": "BETA",
        "neglog10_pval_meta": "LOG10P"
    })

    df["CHR"] = df["CHR"].astype(str)
    df["BP"] = pd.to_numeric(df["BP"], errors="coerce")
    df["BETA"] = pd.to_numeric(df["BETA"], errors="coerce")
    df["LOG10P"] = pd.to_numeric(df["LOG10P"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["CHR", "BP", "A1", "A2", "BETA", "LOG10P"])

    df["BP"] = df["BP"].astype(int)
    df["P"] = np.power(10.0, -df["LOG10P"])

    bim = load_bim_map()
    merged = df.merge(bim, on=["CHR", "BP"], how="inner")

    allele_match = (
        ((merged["A1"] == merged["BIM_A1"]) & (merged["A2"] == merged["BIM_A2"])) |
        ((merged["A1"] == merged["BIM_A2"]) & (merged["A2"] == merged["BIM_A1"]))
    )
    merged = merged.loc[allele_match].copy()

    out = merged[["SNP", "A1", "A2", "BETA", "P"]].copy()
    out.to_csv(outfile, sep="\t", index=False)

    print(f"\n=== {infile} ===")
    print(f"原始行数: {len(df)}")
    print(f"成功映射并通过等位基因检查: {len(merged)}")
    print(f"输出: {outfile}")

#prepare_sumstats("raw_data/gwas/daytime_sleeping.tsv", "raw_data/gwas/annotated/daytime_sleeping_annotated.tsv")
#prepare_sumstats("raw_data/gwas/getting_up_in_morning.tsv", "raw_data/gwas/annotated/getting_up_in_morning_annotated.tsv")
#prepare_sumstats("raw_data/gwas/nap_during_day.tsv", "raw_data/gwas/annotated/nap_during_day_annotated.tsv")
prepare_sumstats("raw_data/gwas/sleeplessness.tsv", "raw_data/gwas/annotated/sleeplessness_annotated.tsv")