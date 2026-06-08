#!/usr/bin/env bash

OUT="results/all_proteins_otters.tsv"

echo -e "protein\tZ\tP\tn_snp" > $OUT

for f in work/test_single_protein/*/results/*_otters_mini.tsv; do
    awk 'NR==2{
        print $1"\t"$4"\t"$5"\t"$3
    }' $f >> $OUT
done

echo "[INFO] 汇总完成 -> $OUT"
