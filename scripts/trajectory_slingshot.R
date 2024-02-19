library(slingshot)
library(Matrix)
library(tradeSeq)
counts <- read.delim("scChip_SparseMatrix.txt",header = F)
head(counts)
counts<-sparseMatrix(i=counts$V1,j=counts$V2,x=counts$V3)
dim(counts)
counts[1:5,1:5]
barcode<-read.delim("scChip_barcode.txt",header = F)
head(barcode)
peak<-read.table("feature.txt",header = T,sep = ",")
head(peak)
rownames(counts) <- peak$feature_name
colnames(counts) <- barcode$V1
sce <- SingleCellExperiment(assays = List(counts = counts))

celltype<-read.csv("scChip_celltype_info.csv",header = T)
emb<-read.csv("scChip_EXACT_old_uwot.csv",header = T)
colData(sce)$clusterLabel <- celltype$celltype
emb<-as.matrix(emb)
colnames(emb)<-c("UMAP1","UMAP2")
reducedDims(sce) <- SimpleList(UMAP = emb)
sce <- slingshot(sce, clusterLabels = "clusterLabel", reducedDim = "UMAP")
write.csv(colData(sce),"scChip_EXACT_old_uwot_trajectory_info.csv")
sce$clusterLabel<-factor(sce$clusterLabel)
lin1 <- getLineages(sce, clusterLabels = "clusterLabel", 
                    start.clus = 'H1',
                    reducedDim = "UMAP")
colours<-c("#fae7b5","#ffe4c4","#ffe135","#ffbf00","#f0ffff","#ace5ee","#1dacd6","#6495ed","#318ce7","#0070ff",
           "#8db600","#e0b0ff","#bf94e4","#f4bbff","#fbcce7")
pdf("scChip_slingshot_EXACT_UMAP_uwot_old.pdf")
plot(reducedDims(sce)$UMAP,pch=16,asp=1,cex=0.6,col = colours[sce$clusterLabel])
lines(SlingshotDataSet(lin1), lwd=4,col = 'black',type = 'lineages',show.constraints = TRUE,cex=1.5)
dev.off()






