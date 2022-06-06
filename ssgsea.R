## Rscript for calculating ssGSEA score.


library(GSVA)
library(limma)

args <- commandArgs(trailingOnly = TRUE)

###############################################################################
#                                                                             #
#                     Read Gene Expression data                               #
#                                                                             #
###############################################################################

# df <- read.table('~/Projects/DeepSurvial/Data/GBM/clean/expression_matrix.txt', 
#                  sep = '\t', 
#                  header=T, 
#                  row.names = 1)
df <- read.table(args[1], 
                 sep = '\t', 
                 header=T, 
                 row.names = 1)
df <- as.matrix(df)


###############################################################################
#                                                                             #
#                       Read Pathway data                                     #
#                                                                             #
###############################################################################

geneSets <- list()
pathnames <- c()
# fileName="~/Projects/DeepSurvial/Pathway/pathway_genes_list.txt"
fileName=args[2]
con=file(fileName,open="r")
line=readLines(con, 1)
while (length(line) > 0){
    ll <- strsplit2(line,split="\t")
    pathname <- ll[1]
    gene_sum <- length(ll)
    genes <- ll[2:gene_sum]
    geneSets <- c(geneSets, list(genes))
    pathnames <- c(pathnames, pathname)
    
    line <- readLines(con, n = 1)
}
close(con)

names(geneSets) <- pathnames

###############################################################################
#                                                                             #
#               Calculate Pathway Enrichment score                            #
#                                                                             #
###############################################################################

ssgsea_score = gsva(df, geneSets, method = "ssgsea", 
                    ssgsea.norm = F, verbose = TRUE)  # donot norm

# write.csv(ssgsea_score, file = "~/Projects/DeepSurvial/Data/GBM/clean/expression_matrix.ssgsea.csv",quote = F)
write.csv(ssgsea_score, file = args[3], quote = F)
