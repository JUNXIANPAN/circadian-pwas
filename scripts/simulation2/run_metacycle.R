#!/usr/bin/env Rscript
# =============================================================================
# run_metacycle.R
# Run MetaCycle (meta2d combining JTK_CYCLE + Lomb-Scargle + ARSER) on a
# time-series proteomics matrix, output annotation TSV that drops into
# circadian_pwas_simulation_phase_causal.py without code changes.
#
# Output schema:
#   protein_id, rhythmicity (R² placeholder), neglog10p, amplitude,
#   phase_hour, period, meta2d_qvalue
# The Python pipeline picks up `neglog10p` automatically.
#
# Usage:
#   Rscript run_metacycle.R \
#     --input raw_data/circadian_info/report.pg_matrix.csv \
#     --timepoints "0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45" \
#     --output raw_data/circadian_info_meta2d/circadian_annotation_meta2d.csv \
#     --id-col protein_id \
#     --workdir /tmp/meta2d_work
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(MetaCycle)
  library(data.table)
})

# ---- args ----
option_list <- list(
  make_option(c("--input"),      type="character", help="Path to pg_matrix TSV"),
  make_option(c("--timepoints"), type="character",
              help="Comma-separated timepoints in hours, length = #sample columns"),
  make_option(c("--output"),     type="character", help="Path for the output annotation TSV"),
  make_option(c("--id-col"),     type="character", default="protein_id",
              help="Column name for protein ID (default: protein_id)"),
  make_option(c("--workdir"),    type="character", default="/tmp/meta2d_work",
              help="Temp workdir for meta2d intermediate files"),
  make_option(c("--minper"),     type="double",    default=20,
              help="Min period to test (default 20h)"),
  make_option(c("--maxper"),     type="double",    default=24,
              help="Max period to test (default 28h)"),
  make_option(c("--methods"),    type="character", default="JTK,LS,ARS",
              help="Comma-separated rhythm methods (default JTK,LS,ARS)"),
  make_option(c("--cap-neglog10p"), type="double", default=50,
              help="Cap -log10(p) at this value (default 50; protects against 0 p-values)")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$input) || is.null(opt$timepoints) || is.null(opt$output)) {
  stop("Required: --input, --timepoints, --output")
}

dir.create(opt$workdir, recursive=TRUE, showWarnings=FALSE)
dir.create(dirname(opt$output), recursive=TRUE, showWarnings=FALSE)

cat("[1/4] Reading pg_matrix:", opt$input, "\n")
mat <- fread(opt$input, sep="auto")
id_col <- opt$`id-col`
if (!(id_col %in% colnames(mat))) {
  stop(sprintf("ID column '%s' not in %s. Available columns: %s",
               id_col, opt$input, paste(head(colnames(mat),10), collapse=", ")))
}

# Reorder: id first, then numeric sample columns in their original order
sample_cols <- setdiff(colnames(mat), id_col)
keep_cols <- sapply(mat[, ..sample_cols], is.numeric)
sample_cols <- sample_cols[keep_cols]
cat("    proteins =", nrow(mat), ", numeric sample columns =", length(sample_cols), "\n")

timepoints <- as.numeric(strsplit(opt$timepoints, ",")[[1]])
if (length(timepoints) != length(sample_cols)) {
  stop(sprintf("Length mismatch: %d timepoints vs %d sample columns",
               length(timepoints), length(sample_cols)))
}

# Write meta2d input file (CSV is most robust)
m2d_in <- file.path(opt$workdir, "meta2d_input.csv")
fwrite(cbind(mat[, ..id_col], mat[, ..sample_cols]),
       m2d_in, sep=",")
cat("    wrote meta2d input:", m2d_in, "\n")

# ---- run meta2d ----
methods <- strsplit(opt$methods, ",")[[1]]
cat("[2/4] Running meta2d  methods =", paste(methods, collapse=","), "\n")

meta2d(
  infile         = m2d_in,
  filestyle      = "csv",
  outdir         = opt$workdir,
  timepoints     = timepoints,
  minper         = opt$minper,
  maxper         = opt$maxper,
  cycMethod      = methods,
  analysisStrategy = "auto",
  outputFile     = TRUE,
  outIntegration = "both",      # integrate p, amplitude, phase across methods
  adjustPhase    = "predictedPer",
  ARSmle         = "auto",
  parallelize    = FALSE
)

# meta2d writes "meta2d_<input>.csv" — find it
out_files <- list.files(opt$workdir, pattern="^meta2d_meta2d_input", full.names=TRUE)
if (length(out_files) == 0) stop("meta2d output not found in ", opt$workdir)
m2d_out <- out_files[1]
cat("    meta2d output:", m2d_out, "\n")

# ---- post-process ----
cat("[3/4] Post-processing\n")
res <- fread(m2d_out)

# Pick the integrated p-value and effect estimates (column names vary slightly
# across MetaCycle versions and depending on whether single or multiple methods
# are used; we use regex matching to be robust)
cat("    Available cols:", paste(colnames(res), collapse=", "), "\n")

find_col <- function(df, patterns) {
  # Try each pattern as a regex; return first matching column name (case-insensitive)
  for (pat in patterns) {
    hit <- grep(pat, colnames(df), ignore.case=TRUE, value=TRUE)
    if (length(hit) > 0) return(hit[1])
  }
  return(NA_character_)
}

# Each entry is a list of regex patterns to try in priority order.
# meta2d_ comes first because it's the integrated value; LS/JTK fallback.
pcol     <- find_col(res, c("^meta2d_(pvalue|p\\.value|p_value)$", "^LS_(pvalue|p\\.value|p_value)$", "^JTK_(pvalue|p\\.value|p_value)$"))
qcol     <- find_col(res, c("^meta2d_(BH\\.Q|qvalue|q_value|fdr)$", "^LS_(BH\\.Q|qvalue|q_value|fdr)$"))
ampcol   <- find_col(res, c("^meta2d_(AMP|amplitude|Amp)$", "^LS_(AMP|amplitude|Amp)$", "^JTK_(AMP|amplitude|Amp)$"))
phasecol <- find_col(res, c("^meta2d_(phase|adjphase|Phase)$", "^LS_(phase|adjphase|Phase)$", "^JTK_(phase|adjphase|Phase)$"))
percol   <- find_col(res, c("^meta2d_(period|Period)$", "^LS_(period|Period)$", "^JTK_(period|Period)$"))
idcol    <- find_col(res, c(paste0("^", id_col, "$"), "^CycID$", "^ID$"))

cat("    Resolved columns:\n")
cat("      pcol    =", pcol,    "\n")
cat("      qcol    =", qcol,    "\n")
cat("      ampcol  =", ampcol,  "\n")
cat("      phasecol=", phasecol,"\n")
cat("      percol  =", percol,  "\n")
cat("      idcol   =", idcol,   "\n")

if (any(is.na(c(pcol, ampcol, phasecol, percol, idcol)))) {
  stop("Could not resolve all required columns. See available list above.")
}

# Compute -log10(p), capped to avoid Inf when p=0
pvals <- res[[pcol]]
pvals[pvals < 1e-300] <- 1e-300
neglog10p <- pmin(-log10(pvals), opt$`cap-neglog10p`)

# Use raw -log10(p) as the rhythmicity score. The Python simulation will
# rescale via normalize_01 to [0, 1]. This preserves the heavy-tailed shape
# of MetaCycle evidence (analogous to cosinor R²'s natural distribution),
# making cross-method comparisons cleaner.

out <- data.table(
  protein_id   = res[[idcol]],
  rhythmicity  = neglog10p,                        # ★ raw -log10(p) as evidence
  neglog10p    = neglog10p,                        # explicit duplicate for clarity
  amplitude    = res[[ampcol]],
  phase_hour   = res[[phasecol]] %% 24,            # wrap to [0, 24)
  period       = res[[percol]],
  meta2d_pvalue = pvals,
  meta2d_qvalue = if (!is.na(qcol)) res[[qcol]] else NA_real_
)

cat("[4/4] Writing:", opt$output, "  (", nrow(out), "proteins )\n")
fwrite(out, opt$output, sep=",")

cat("\n=== Summary ===\n")
cat("    median neglog10p =", round(median(out$neglog10p, na.rm=TRUE), 3), "\n")
cat("    fraction with q < 0.05 =", round(mean(out$meta2d_qvalue < 0.05, na.rm=TRUE), 3), "\n")
cat("    proteins with valid output:", sum(!is.na(out$neglog10p)), "/", nrow(out), "\n")
cat("\nDone. Use this file as --pg-matrix in the simulation (or as a separate annotation feed).\n")
