library(cicero)
library(monocle)

info<-read.csv("scChip_EXACT_old_uwot_trajectory_info.csv",header = T,row.names = 1)
head(info)

#filter Lineage
a<-which(info$slingPseudotime_1>0)
Pseudotime_1<-info[a,]
table(Pseudotime_1$clusterLabel)
a<-which(Pseudotime_1$clusterLabel %in% c("DE2","DE3","DE4","DE5","MES2","MES3","MES4","MES5"))
Pseudotime_1<-Pseudotime_1[-a,]
Pseudotime_1$clusterLabel<-as.character(as.matrix(Pseudotime_1$clusterLabel))
Pseudotime_1<-Pseudotime_1[order(Pseudotime_1$slingPseudotime_1,decreasing = F),]
Pseudotime_1<-Pseudotime_1[Pseudotime_1$slingPseudotime_1>5 & Pseudotime_1$slingPseudotime_1<11,]
head(Pseudotime_1)

library(Matrix)
indata <- readMM("K36me3_matrix.mtx")
barcode<-read.delim("barcodes.tsv",header = F)
colnames(indata)<-barcode$V1
features<-read.delim("K36me3_genes.tsv",header = F)
rownames(indata)<-features$V1
a<-match(rownames(Pseudotime_1),colnames(indata))
indata<-indata[,a]

cellinfo <- as.data.frame(colnames(indata))
row.names(cellinfo) <- cellinfo$`colnames(indata)`
names(cellinfo) <- "cells"

gene_anno <- rtracklayer::readGFF("Homo_sapiens.GRCh37.87.chr.gtf")
head(gene_anno)
gene_anno<-gene_anno[gene_anno$type=="gene",]
gene_anno<-gene_anno[gene_anno$gene_biotype=="protein_coding",]
a<-which(rownames(indata) %in% gene_anno$gene_name)
dim(indata)
indata<-indata[a,]
gene_annotation<-as.data.frame(rownames(indata))
rownames(gene_annotation)<-gene_annotation$`rownames(indata)`
colnames(gene_annotation)<-"gene_short_name"
head(gene_annotation)

pd <- new("AnnotatedDataFrame", data = cellinfo)
fd <- new("AnnotatedDataFrame", data = gene_annotation)
cds <- newCellDataSet(as.matrix(indata),phenoData = pd, featureData = fd,expressionFamily=negbinomial.size())
head(cds@assayData$exprs)[,1:5]
cds <- estimateSizeFactors(cds)
cds <- estimateDispersions(cds)
cds <- detectGenes(cds, min_expr = 0.1)
print(head(fData(cds)))
expressed_genes <- row.names(subset(fData(cds),num_cells_expressed >= 4))

pData(cds)$CellType<-Pseudotime_1$clusterLabel
pData(cds)$Pseudotime<-Pseudotime_1$slingPseudotime_1

#Finding Genes that Change as a Function of Pseudotime
diff_test_res_pse <- differentialGeneTest(cds,fullModelFormulaStr = "~sm.ns(Pseudotime)")
diff_test_res_pse<-diff_test_res_pse[,c("gene_short_name", "pval", "qval")]
head(diff_test_res_pse)
write.csv(diff_test_res_pse,"differentialGeneTest_along_Pseudotime.csv")














