library(GenomicRanges)
human.hg38.genome<-read.delim("hg38.chrom.sizes",header = F)
head(human.hg38.genome)
library(monocle)
library(cicero)
peakinfo<-read.csv("pbmc3k_filter_interaction_peakinfo.csv",header = T,row.names = 1)
geneinfo<-read.csv("pbmc3k_filter_interaction_geneinfo.csv",header = T,row.names = 1)


indata <- read.table("recon.txt",header = F,sep = ",")
indata<-as(as.matrix(indata),"dgTMatrix")
indata<-t(indata)

gene_matrix<-indata[1:13263,]
indata<-indata[13264:89119,]
indata@x[indata@x > 0] <- 1
cellinfo <- read.delim("pbmc3k_filter_barcode.txt",header = F)
row.names(cellinfo) <- cellinfo$V1
names(cellinfo) <- "cells"
colnames(indata)<-cellinfo$cells
celltype<-read.csv("cluster_info_for_accuracy.csv",header = T)
Allgene<-read.delim("pbmc3k_filter_features.txt",header = F)
Allgene<-Allgene[1:13263,]
gene_anno_name<-read.csv("pbmc_gene_anno_name.csv",header = T)
a<-match(gene_anno_name$gene_name,Allgene$V2)
Allgene<-Allgene[a,]
gene_matrix<-gene_matrix[a,]
rownames(gene_matrix)<-geneinfo$site_name
colnames(gene_matrix)<-colnames(indata)
rownames(indata)<-peakinfo$site_name[1:75856]
dat<-rbind(indata,gene_matrix)
rownames(dat)<-peakinfo$site_name


fd <- methods::new("AnnotatedDataFrame", data = peakinfo)
pd <- methods::new("AnnotatedDataFrame", data = cellinfo)
input_cds <-  suppressWarnings(newCellDataSet(dat,
                                              phenoData = pd,
                                              featureData = fd,
                                              expressionFamily=VGAM::binomialff(),
                                              lowerDetectionLimit=0))
input_cds@expressionFamily@vfamily <- "binomialff"
input_cds <- monocle::detectGenes(input_cds)
input_cds <- input_cds[Matrix::rowSums(exprs(input_cds)) != 0,]
set.seed(2017)
input_cds <- detectGenes(input_cds)
input_cds <- estimateSizeFactors(input_cds)
input_cds <- reduceDimension(input_cds, max_components = 2, num_dim=6,
                             reduction_method = 'tSNE', norm_method = "none")
tsne_coords <- t(reducedDimA(input_cds))
row.names(tsne_coords) <- row.names(pData(input_cds))
cicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = tsne_coords,k=30)
##Call Cicero
conns <- run_cicero(cicero_cds, human.hg38.genome)
a<-which(conns$coaccess==0)
conns<-conns[-a,]
index<-which(is.na(conns))
conns1=conns[-index,]

#deduplicate
a<-match(conns1$Peak1,gene_anno_name$peakname)
conns1$Gene1<-gene_anno_name$gene_name[a]
a<-match(conns1$Peak2,gene_anno_name$peakname)
conns1$Gene2<-gene_anno_name$gene_name[a]
conns2<-conns1
conns2$name<-paste0(conns2$Peak1,":",conns2$Peak2)
conns3<-conns2
conns3$name<-paste0(conns3$Peak2,":",conns3$Peak1)
a<-match(conns3$name,conns2$name)
a<-as.data.frame(a)
a$index<-1:dim(a)[1]
b<-which(a$a>a$index)
conns2<-conns2[-(a$a[b]),]
#PG specific
a<-which(is.na(conns2$Gene1) & is.na(conns2$Gene2))
conns2<-conns2[-a,]
tmp1<-which(is.na(conns2$Gene1))
tmp2<-which(is.na(conns2$Gene2))
tmp<-unique(union(tmp1,tmp2))
conns2<-conns2[tmp,]
conns2<-conns2[order(abs(conns2$coaccess),decreasing=T),]
conns2<-conns2[which(conns2$coaccess>0.5),]
write.csv(conns2,paste0(i,"/pbmc3k_filter_Recon_cor0.5PGspecific_conns.csv"),row.names = F)





