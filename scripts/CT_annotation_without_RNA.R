library(Seurat)
library(SeuratDisk)
library(ggplot2)
library(patchwork)

indata<-readRDS("GSM4732109_CD28_CD3_control_GeneScoreMatrix.rds")
dim(indata)
GSMatrix<-indata@assays@data$GeneScoreMatrix
GSMatrix[1:5,1:5]
head(indata@elementMetadata)
gene_info<-as.data.frame(indata@elementMetadata)
head(gene_info)
rownames(GSMatrix)<-gene_info$name

pbmc <- CreateSeuratObject(counts = GSMatrix, assay = "RNA")
pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc)
pbmc <- ScaleData(pbmc, features = rownames(pbmc))
pbmc <- RunPCA(pbmc)


#load scRNA-seq data
pbmc.rna<-LoadH5Seurat("pbmc_multimodal.h5seurat")
pbmc.rna[["RNA"]] <- CreateAssayObject(counts = pbmc.rna@assays$SCT@counts)
DefaultAssay(pbmc.rna) <- "RNA"
pbmc.rna <- NormalizeData(pbmc.rna)
pbmc.rna <- FindVariableFeatures(pbmc.rna)
pbmc.rna <- ScaleData(pbmc.rna)
pbmc.rna <- RunPCA(pbmc.rna)
pbmc.rna <- RunUMAP(pbmc.rna, dims = 1:30)

#find anchors between reference and query
anchors <- FindTransferAnchors(
  reference = pbmc.rna,
  query = pbmc,
  features = VariableFeatures(object = pbmc.rna),
  reference.assay = 'RNA',
  query.assay = 'RNA',
  reduction = 'cca'
)

pbmc <- TransferData(
  anchorset = anchors, 
  refdata = pbmc.rna$celltype.l2,
  weight.reduction = pbmc[['pca']],
  dims = 2:30
)

cell_typeFromRNA_Full<-as.data.frame(pbmc$predicted.id)
head(cell_typeFromRNA_Full)
dim(cell_typeFromRNA_Full)
rownames(cell_typeFromRNA_Full)<-colnames(GSMatrix)
colnames(cell_typeFromRNA_Full)<-"celltype"
a<-which(cell_typeFromRNA_Full$celltype %in% c("B intermediate","B memory","B naive"))
cell_typeFromRNA_Full$celltype[a]<-"B cell"
head(cell_typeFromRNA_Full)
write.csv(cell_typeFromRNA_Full,"ASAP_cell_typeFromRNA_Fullreference_integration.csv")


