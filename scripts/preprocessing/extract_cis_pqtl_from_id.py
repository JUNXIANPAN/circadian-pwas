import argparse
import csv
import gzip
import os
import re
from pathlib import Path


def load_annotation(annotation_csv):
    id_to_gene = {}
    with open(annotation_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            soma_id = row["SOMAseqID"].strip()
            gene = row["EntrezGeneSymbol"].strip()
            if soma_id and gene:
                id_to_gene[soma_id] = gene
    return id_to_gene


def load_gene_coords(gene_coords_tsv):
    gene_to_coord = {}
    with open(gene_coords_tsv, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            arr = line.split("\t")
            if len(arr) < 4:
                continue
            gene, chrom, start, end = arr[:4]
            chrom = chrom.replace("chr", "")
            try:
                start = int(start)
                end = int(end)
            except ValueError:
                continue
            gene_to_coord[gene] = (chrom, start, end)
    return gene_to_coord


def parse_soma_id_from_filename(filename):
    m = re.match(r"reducepqtls\.(X[^.]+\.[^.]+)\.glm\.linear\.gz$", filename)
    if not m:
        return None
    return m.group(1)


def parse_variant_id(variant_id):
    parts = variant_id.split(":")
    if len(parts) < 2:
        return None, None
    chrom = parts[0].replace("chr", "")
    try:
        pos = int(parts[1])
    except ValueError:
        return None, None
    return chrom, pos


def extract_one_file(infile, outfile, target_chr, cis_start, cis_end):
    n_total = 0
    n_kept = 0

    with gzip.open(infile, "rt") as fin, gzip.open(outfile, "wt") as fout:
        header = fin.readline()
        if not header:
            return 0, 0
        fout.write(header)

        for line in fin:
            n_total += 1
            arr = line.rstrip("\n").split()
            if len(arr) == 0:
                continue

            variant_id = arr[0]
            chrom, pos = parse_variant_id(variant_id)
            if chrom is None or pos is None:
                continue

            if chrom == target_chr and cis_start <= pos <= cis_end:
                fout.write(line)
                n_kept += 1

    return n_total, n_kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, help="brain_analyte_info.csv")
    parser.add_argument("--gene-coords", required=True, help="gene_coords.tsv")
    parser.add_argument("--input-dir", required=True, help="directory of .glm.linear.gz files")
    parser.add_argument("--output-dir", required=True, help="directory for cis-pQTL files")
    parser.add_argument("--window", type=int, default=1000000, help="cis window size, default 1Mb")
    parser.add_argument("--limit", type=int, default=None, help="only process first N files")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    id_to_gene = load_annotation(args.annotation)
    gene_to_coord = load_gene_coords(args.gene_coords)

    files = sorted(
        [x for x in os.listdir(args.input_dir) if x.endswith(".glm.linear.gz")]
    )
    if args.limit is not None:
        files = files[:args.limit]

    summary_file = os.path.join(args.output_dir, "cis_extraction_summary.tsv")
    with open(summary_file, "w") as summary:
        summary.write(
            "file\tsoma_id\tgene\tchr\tgene_start\tgene_end\tcis_start\tcis_end\tn_total\tn_kept\tstatus\n"
        )

        for fn in files:
            soma_id = parse_soma_id_from_filename(fn)
            if soma_id is None:
                summary.write(f"{fn}\t\t\t\t\t\t\t\t\t\tbad_filename\n")
                continue

            gene = id_to_gene.get(soma_id)
            if gene is None:
                summary.write(f"{fn}\t{soma_id}\t\t\t\t\t\t\t\t\tno_gene_mapping\n")
                continue

            coord = gene_to_coord.get(gene)
            if coord is None:
                summary.write(f"{fn}\t{soma_id}\t{gene}\t\t\t\t\t\t\t\tno_gene_coord\n")
                continue

            chrom, gstart, gend = coord
            cis_start = max(1, gstart - args.window)
            cis_end = gend + args.window

            infile = os.path.join(args.input_dir, fn)
            outname = fn.replace(".glm.linear.gz", ".cis.glm.linear.gz")
            outfile = os.path.join(args.output_dir, outname)

            try:
                n_total, n_kept = extract_one_file(
                    infile, outfile, chrom, cis_start, cis_end
                )
                status = "ok" if n_kept > 0 else "no_cis_snps"
                summary.write(
                    f"{fn}\t{soma_id}\t{gene}\t{chrom}\t{gstart}\t{gend}\t{cis_start}\t{cis_end}\t{n_total}\t{n_kept}\t{status}\n"
                )
            except Exception as e:
                summary.write(
                    f"{fn}\t{soma_id}\t{gene}\t{chrom}\t{gstart}\t{gend}\t{cis_start}\t{cis_end}\t\t\tERROR:{str(e)}\n"
                )

    print(f"Done. Output directory: {args.output_dir}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
