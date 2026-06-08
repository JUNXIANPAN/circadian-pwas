#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# run_metacycle.R
# Rhythmicity engine for the circadian-informed PWAS simulation.
#
# Reads a CSV whose first column is a protein id and whose remaining columns
# are population-mean expression at each sampled circadian time point.
# Runs MetaCycle::meta2d (Wu et al., Bioinformatics 2016) integrating
# JTK_CYCLE + Lomb-Scargle, then writes a tidy CSV with, per protein:
#   pvalue  : integrated rhythmicity p-value (meta2d_pvalue)
#   period  : integrated period estimate
#   phase   : integrated peak phase (hours)
#   amp     : integrated amplitude
#   base    : integrated MESOR / baseline
#   R2      : coefficient of determination of the MetaCycle cosine fit
#             vs the observed profile  ->  this is the continuous prior signal
#
# Usage:
#   Rscript run_metacycle.R <infile.csv> <outfile.csv> <comma-separated-timepoints>
# Example:
#   Rscript run_metacycle.R profiles.csv rhythm.csv 0,3,6,9,12,15,18,21
# ---------------------------------------------------------------------------

suppressMessages(library(MetaCycle))

args      <- commandArgs(trailingOnly = TRUE)
infile    <- args[1]
outfile   <- args[2]
tp        <- as.numeric(strsplit(args[3], ",")[[1]])

dat <- read.csv(infile, check.names = FALSE)
ids <- dat[[1]]
mat <- as.matrix(dat[, -1, drop = FALSE])         # proteins x timepoints

# meta2d needs a file; write to a temp file in the same dir.
tmpdir  <- dirname(outfile)
tmpin   <- file.path(tmpdir, "._mc_in.csv")
write.csv(data.frame(id = ids, mat, check.names = FALSE),
          tmpin, row.names = FALSE)

invisible(capture.output(
  meta2d(infile = tmpin, filestyle = "csv", outdir = tmpdir,
         timepoints = tp,
         cycMethod = c("JTK", "LS"),          # robust for single-cycle, uneven-safe
         outIntegration = "onlyIntegration",
         outputFile = TRUE)
))

res <- read.csv(file.path(tmpdir, "meta2d_._mc_in.csv"), check.names = FALSE)

# Reconstruct the integrated cosine fit and compute R^2 against the data.
twopi <- 2 * pi
r2 <- numeric(nrow(res))
for (i in seq_len(nrow(res))) {
  per   <- res$meta2d_period[i]; if (is.na(per) || per <= 0) per <- 24
  amp   <- res$meta2d_AMP[i];    if (is.na(amp)) amp <- 0
  base  <- res$meta2d_Base[i];   if (is.na(base)) base <- mean(mat[i, ])
  phase <- res$meta2d_phase[i];  if (is.na(phase)) phase <- 0
  fit   <- base + amp * cos(twopi * (tp - phase) / per)
  obs   <- mat[i, ]
  sstot <- sum((obs - mean(obs))^2)
  ssres <- sum((obs - fit)^2)
  r2[i] <- if (sstot > 0) max(0, min(1, 1 - ssres / sstot)) else 0
}

out <- data.frame(
  id     = res$CycID,
  pvalue = res$meta2d_pvalue,
  period = res$meta2d_period,
  phase  = res$meta2d_phase,
  amp    = res$meta2d_AMP,
  base   = res$meta2d_Base,
  R2     = r2
)
write.csv(out, outfile, row.names = FALSE)
cat(sprintf("MetaCycle done: %d proteins -> %s\n", nrow(out), outfile))
