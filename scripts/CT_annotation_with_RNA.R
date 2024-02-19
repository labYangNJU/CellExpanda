library(Seurat)
library(SeuratDisk)
library(ggplot2)
library(patchwork)

indata<-read.delim("pbmc3k_filter_SparseMatrix_ATACBinary.txt",header = F)
head(indata)
library(Matrix)
fullmatrix<-sparseMatrix(i=indata$V1,j=indata$V2,x=indata$V3)
dim(fullmatrix)
barcode<-read.delim("pbmc3k_filter_barcode.txt",header = F)
colnames(fullmatrix)<-barcode$V1
feature<-read.delim("pbmc3k_filter_features.txt",header = F)
head(feature)
rownames(fullmatrix)<-feature$V2
fullmatrix[1:5,1:5]
Gene<-fullmatrix[1:13263,]
dim(Gene)
Gene[1:5,1:5]
Peak<-fullmatrix[13264:89119,]
Peak[1:5,1:5]
rm(Peak)

pbmc <- CreateSeuratObject(counts = Gene, assay = "RNA", project = "pbmc")
pbmc <- SCTransform(pbmc, verbose = FALSE)

#load scRNA-seq data
reference <- LoadH5Seurat("../pbmc_multimodal.h5seurat")

#find anchors between reference and query
anchors <- FindTransferAnchors(
  reference = reference,
  query = pbmc,
  normalization.method = "SCT",
  reference.reduction = "spca",
  dims = 1:50
)
pbmc <- TransferData(
  anchorset = anchors, 
  reference = reference,
  query = pbmc,
  refdata = reference$celltype.l2
)
table(pbmc$predicted.id.score > 0.5)

cell_typeFromRNA_Full<-as.data.frame(pbmc$predicted.id)
head(cell_typeFromRNA_Full)
colnames(cell_typeFromRNA_Full)<-"celltype"
a<-which(cell_typeFromRNA_Full$celltype %in% c("B intermediate","B memory","B naive"))
cell_typeFromRNA_Full$celltype[a]<-"B cell"
head(cell_typeFromRNA_Full)
write.csv(cell_typeFromRNA_Full,"pbmc3k_cell_typeFromRNA_reference.csv")




