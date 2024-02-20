### Load libraries ###
library(Signac)
library(Seurat)
library(Matrix)
library(future)
library(EnsDb.Hsapiens.v75)
library(BSgenome.Hsapiens.UCSC.hg19)
library(SeuratWrappers)
library(ggplot2)
library(reshape)
library(viridis)

### Set seed ###

set.seed(1)

### Load fragments ###

fragpath<-"Fig4_tsv_files/trilineage_differentiation.K27me3.tsv.gz"
ncounts <- CountFragments(fragments = fragpath)

fragpath2<-"Fig4_tsv_files/trilineage_differentiation.K4me1.tsv.gz"
ncounts2 <- CountFragments(fragments = fragpath2)
dim(ncounts2)

fragpath3<-"Fig4_tsv_files/trilineage_differentiation.K36me3.tsv.gz"
ncounts3 <- CountFragments(fragments = fragpath3)

### Refine cell barcode lists ###

ncounts<-ncounts[ncounts$CB %in% ncounts2$CB & ncounts$CB %in% ncounts3$CB,]
ncounts2<-ncounts2[ncounts2$CB %in% ncounts$CB & ncounts2$CB %in% ncounts3$CB,]
ncounts3<-ncounts3[ncounts3$CB %in% ncounts$CB & ncounts3$CB %in% ncounts2$CB,]
ncounts <- ncounts[order(ncounts$CB),]
ncounts2 <- ncounts2[order(ncounts2$CB),]
ncounts3 <- ncounts3[order(ncounts3$CB),]

cell.names <- ncounts[ncounts$frequency_count > 100 & ncounts2$frequency_count > 100 & ncounts3$frequency_count > 0, "CB"]
cell.names2 <- ncounts2[ncounts$frequency_count > 100 & ncounts2$frequency_count > 100 & ncounts3$frequency_count > 0, "CB"]
cell.names3 <- ncounts3[ncounts$frequency_count > 100 & ncounts2$frequency_count > 100 & ncounts3$frequency_count > 0, "CB"]

### Create K27 chromatin assay ###

plan("multiprocess", workers = 10)
frags <- CreateFragmentObject(path = fragpath, cells = cell.names)
agg_bins <- AggregateTiles(
  object = frags,
  genome = seqlengths(BSgenome.Hsapiens.UCSC.hg19)[1:24],
  min_counts = 5,
  cells = cell.names,
  binsize = 5000
)

annot <- GetGRangesFromEnsDb(EnsDb.Hsapiens.v75)
seqlevelsStyle(annot) <- "UCSC"
genome(annot) <- "hg19"

chrom_assay <- CreateChromatinAssay(
  counts = agg_bins,
  genome = 'hg19',
  annotation = annot,
  fragments = frags,
  min.features = -1
)

### Prepare inputs to K4m1 chromatin assay ###

plan("multiprocess", workers = 10)
frags2 <- CreateFragmentObject(path = fragpath2, cells = cell.names2)
agg_bins2 <- AggregateTiles(
  object = frags2,
  genome = seqlengths(BSgenome.Hsapiens.UCSC.hg19)[1:24],
  min_counts = 5,
  cells = cell.names2,
  binsize = 5000
)

annot2 <- GetGRangesFromEnsDb(EnsDb.Hsapiens.v75)
seqlevelsStyle(annot2) <- "UCSC"
genome(annot2) <- "hg19"

### Prepare inputs to K36 chromatin assay ###

plan("multiprocess", workers = 10)
frags3 <- CreateFragmentObject(path = fragpath3, cells = cell.names3)
agg_bins3 <- AggregateTiles(
  object = frags3,
  genome = seqlengths(BSgenome.Hsapiens.UCSC.hg19)[1:24],
  min_counts = 5,
  cells = cell.names3,
  binsize = 5000
)

annot3 <- GetGRangesFromEnsDb(EnsDb.Hsapiens.v75)
seqlevelsStyle(annot3) <- "UCSC"
genome(annot3) <- "hg19"

### Create SeuratObject with K27 chromatin assay ### 

object <- CreateSeuratObject(counts = chrom_assay, assay = "K27")
object[["K27"]] <- chrom_assay

### Add K4m1 chromatin assay ###
object[["K4m1"]]<-CreateChromatinAssay(counts = agg_bins2, genome = 'hg19', annotation = annot2, fragments = frags2, min.features = -1)

### Add K36 chromatin assay ###

object[["K36"]]<-CreateChromatinAssay(counts = agg_bins3, genome = 'hg19', annotation = annot3, fragments = frags3, min.features = -1)

### Subset SeuratObject ###

object <- subset(object, nCount_K27 < 5000 & nCount_K27 > 100 & nCount_K4m1 < 5000 & nCount_K4m1 > 100 & nCount_K36 < 5000 & nCount_K36 > 0)
object  #150766 features across 7727 samples within 3 assays

write.table(rownames(object@assays$K27@counts),"scChip_k27_peaks.txt",col.names = F,row.names = F,sep = "\t",quote = F)
write.table(rownames(object@assays$K4m1@counts),"scChip_k4m1_peaks.txt",col.names = F,row.names = F,sep = "\t",quote = F)
write.table(rownames(object@assays$K36@counts),"scChip_k36_peaks.txt",col.names = F,row.names = F,sep = "\t",quote = F)


### Assign celltypes to metadata ###
celltypes_temp<-read.table("Fig4_other_files/trilineage_differentiation.K27me3-K4me1-K36me3.celltypes.txt")
colnames(celltypes_temp)<-c("barcode", "celltype")
head(celltypes_temp)
celltypes_temp<-celltypes_temp[order(celltypes_temp$barcode),]
celltypes<-data.frame(celltype=celltypes_temp$celltype)
rownames(celltypes)<-celltypes_temp$barcode
celltypes<-celltypes[rownames(celltypes) %in% rownames(object@meta.data),]
object@meta.data<-object@meta.data[order(rownames(object@meta.data)),]
object<-AddMetaData(object, metadata=celltypes, col.name="celltype")
object@meta.data$barcode<-rownames(object@meta.data)
info<-object@meta.data[,c(16,8)]
head(info)
write.csv(info,"scChip_K27_K36_K4m1_celltype_info.csv",row.names = F)
write.table(info$barcode,"scChip_K27_K36_K4m1_barcode.txt",col.names = F,row.names = F,sep = "\t",quote = F)


K27_sm<-as.data.frame(summary(object@assays$K27@counts))
head(K27_sm)
max(K27_sm$i)
write.table(K27_sm,"scChip_K27_SparseMatrix.txt",col.names = F,row.names = F,sep = "\t",quote = F)
K4m1_sm<-as.data.frame(summary(object@assays$K4m1@counts))
head(K4m1_sm)
write.table(K4m1_sm,"scChip_K4m1_SparseMatrix.txt",col.names = F,row.names = F,sep = "\t",quote = F)
K36_sm<-as.data.frame(summary(object@assays$K36@counts))
head(K36_sm)
write.table(K36_sm,"scChip_K36_SparseMatrix.txt",col.names = F,row.names = F,sep = "\t",quote = F)


##K27_K36_K4m1
indata<-read.delim("scChip_K27_SparseMatrix.txt",header = F)
head(indata)
max(indata$V1)
indata$V3<-1

dat2<-read.delim("scChip_K36_SparseMatrix.txt",header = F)
head(dat2)
max(dat2$V1)
dat2$V3<-1
dat2$V1<-dat2$V1+59091

indata<-rbind(indata,dat2)
dat3<-read.delim("scChip_K4m1_SparseMatrix.txt",header = F)
head(dat3)
max(dat3$V1)
dat3$V3<-1
dat3$V1<-dat3$V1+74952
indata<-rbind(indata,dat3)
dat<-sparseMatrix(i=indata$V1,j=indata$V2,x=indata$V3)
rowsum<-rowSums(dat)
a<-which(rowsum==0)
dat<-dat[-a,]
class(dat)
indata<-as.data.frame(summary(dat))
head(indata)
write.table(indata,"scChip_K27_K36_K4m1_SparseMatrix.txt",row.names = F,col.names = F,sep = "\t",quote = F)


feature1<-read.delim("scChip_k27_peaks.txt",header = F)
head(feature1)
feature1$V1<-paste0("K27",feature1$V1)
feature2<-read.delim("scChip_K36_peaks.txt",header = F)
head(feature2)
feature2$V1<-paste0("K36",feature2$V1)
tmp<-rbind(feature1,feature2)
colnames(tmp)<-"feature_name"
tmp$d1<-0
feature3<-read.delim("scChip_K4m1_peaks.txt",header = F)
head(feature3)
feature3$V1<-paste0("K4",feature3$V1)
colnames(feature3)<-"feature_name"
feature3$d1<-0
tmp<-rbind(tmp,feature3)
tmp<-tmp[-a,]
write.table(tmp,"feature_discriminator.txt",col.names = T,row.names = F,sep = ",",quote = F)


