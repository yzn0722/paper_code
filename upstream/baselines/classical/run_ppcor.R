#!/usr/bin/env Rscript
# BEELINE-compatible PPCOR runner (Spearman partial correlation).
# Usage: Rscript run_ppcor.R <ExpressionData.csv> <outFile.txt> [pVal]
#
# ExpressionData: genes × cells, CSV with gene names as first column / rownames.
# Output raw table: Gene1, Gene2, corVal, pValue  (then Python parses like BEELINE).

suppressPackageStartupMessages({
  .libPaths(c("/home/yezhongni/R/library", .libPaths()))
  library(ppcor)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_ppcor.R <ExpressionData.csv> <outFile.txt> [pVal]")
}
inFile <- args[[1]]
outFile <- args[[2]]

inputExpr <- read.table(inFile, sep = ",", header = TRUE, row.names = 1, check.names = FALSE)
geneNames <- rownames(inputExpr)

# BEELINE: pcor on cells × genes with Spearman
pcorResults <- pcor(x = t(as.matrix(inputExpr)), method = "spearman")

DF <- data.frame(
  Gene1 = geneNames[c(row(pcorResults$estimate))],
  Gene2 = geneNames[c(col(pcorResults$estimate))],
  corVal = c(pcorResults$estimate),
  pValue = c(pcorResults$p.value),
  stringsAsFactors = FALSE
)
# drop self-loops
DF <- DF[DF$Gene1 != DF$Gene2, ]
outDF <- DF[order(DF$corVal, decreasing = TRUE), ]
dir.create(dirname(outFile), recursive = TRUE, showWarnings = FALSE)
write.table(outDF, outFile, sep = "\t", quote = FALSE, row.names = FALSE)
