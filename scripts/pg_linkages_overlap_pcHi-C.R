raw<-read.csv("pbmc3k_filter_Raw_cor0.5PGspecific_conns.csv",header = T)
recon<-read.csv("pbmc3k_filter_Recon_cor0.5PGspecific_conns.csv",header = T)

library(GenomicRanges)
all_pcHiC<-read.delim("PCHiC_peak_matrix_cutoff5.tsv",header = T)
dim(all_pcHiC)
head(all_pcHiC)
all_pcHiC$baitChr<-paste0("chr",all_pcHiC$baitChr)
all_pcHiC$oeChr<-paste0("chr",all_pcHiC$oeChr)
bait_Match<-read.delim("bait_MAPPED_COORDINATES_hg38.bed",header = F)
head(bait_Match)
dim(bait_Match)
oe_Match<-read.delim("oe_MAPPED_COORDINATES_hg38.bed",header = F)
head(oe_Match)
a<-which(bait_Match$V4 %in% oe_Match$V4)
bait_Match<-bait_Match[a,]
a<-which(oe_Match$V4 %in% bait_Match$V4)
oe_Match<-oe_Match[a,]
all_pcHiC_s<-all_pcHiC[oe_Match$V4,]
dim(all_pcHiC_s)
head(all_pcHiC_s)
oe_Match<-oe_Match[,1:3]
colnames(oe_Match)<-c("oeChr_hg38","oeStart_hg38","oeEnd_hg38")
all_pcHiC_s<-cbind(all_pcHiC_s,oe_Match)
bait_Match<-bait_Match[,1:3]
head(bait_Match)
colnames(bait_Match)<-c("baitChr_hg38","baitStart_hg38","baitEnd_hg38")
all_pcHiC_s<-cbind(all_pcHiC_s,bait_Match)
bait<-all_pcHiC_s[,c(34:36)]
head(bait)
colnames(bait)<-c("chr","bp1","bp2")
bait_bed <- with(bait, GRanges(chr, IRanges(bp1, bp2)))
oe<-all_pcHiC_s[,c(31:33)]
colnames(oe)<-c("chr","bp1","bp2")
oe_bed <- with(oe, GRanges(chr, IRanges(bp1, bp2)))

gene_anno_name<-read.csv("pbmc_gene_anno_name.csv",header = T)
head(gene_anno_name)
gene_anno_name<-gene_anno_name[,c(1,3:4,10)]
colnames(gene_anno_name)<-c("chr","bp1","bp2","site_name")
tail(gene_anno_name)
peak_info<-read.csv("pbmc3k_filter_interaction_peakinfo.csv",header = T)
tail(peak_info)
dim(peak_info)
gene_anno_name$site_name<-peak_info$site_name[75857:87074]
peak_info<-peak_info[1:75856,]
peak_info<-peak_info[,2:5]
peak_info<-rbind(peak_info,gene_anno_name)
overlap_in_cicero<-t(as.data.frame(c("celltype","number","percentage","type")))
overlap_in_Ref<-t(as.data.frame(c("celltype","number","percentage","type")))

index<-as.character(colnames(all_pcHiC_s)[12:28])
p=12
for(i in index){
  all_pcHiC_s1<-all_pcHiC_s[all_pcHiC_s[,p] > 5,]
  bait<-all_pcHiC_s1[,c(34:36)]
  colnames(bait)<-c("chr","bp1","bp2")
  bait_bed <- with(bait, GRanges(chr, IRanges(bp1, bp2)))
  oe<-all_pcHiC_s1[,c(31:33)]
  colnames(oe)<-c("chr","bp1","bp2")
  oe_bed <- with(oe, GRanges(chr, IRanges(bp1, bp2)))
  cicero_loop<-raw
  a<-match(cicero_loop$Peak1,peak_info$site_name)
  left<-peak_info[a,]
  left_bed <- with(left, GRanges(chr, IRanges(bp1, bp2)))
  left_overlap<-findOverlaps(left_bed,bait_bed)
  left_overlap_m<-as.data.frame(left_overlap)
  left_overlap_m$name<-paste0(left_overlap_m$queryHits,":",left_overlap_m$subjectHits)
  a<-match(cicero_loop$Peak2,peak_info$site_name)
  right<-peak_info[a,]
  right_bed <- with(right, GRanges(chr, IRanges(bp1, bp2)))
  right_overlap<-findOverlaps(right_bed,oe_bed)
  right_overlap_m<-as.data.frame(right_overlap)
  right_overlap_m$name<-paste0(right_overlap_m$queryHits,":",right_overlap_m$subjectHits)
  a<-which(left_overlap_m$name %in% right_overlap_m$name)
  out1<-left_overlap_m[a,]
  left_overlap<-findOverlaps(left_bed,oe_bed)
  left_overlap_m<-as.data.frame(left_overlap)
  left_overlap_m$name<-paste0(left_overlap_m$queryHits,":",left_overlap_m$subjectHits)
  right_overlap<-findOverlaps(right_bed,bait_bed)
  right_overlap_m<-as.data.frame(right_overlap)
  right_overlap_m$name<-paste0(right_overlap_m$queryHits,":",right_overlap_m$subjectHits)
  a<-which(left_overlap_m$name %in% right_overlap_m$name)
  out2<-left_overlap_m[a,]
  out<-rbind(out1,out2)
  index1<-unique(as.character(out$queryHits))
  index1<-as.numeric(index1)
  index2<-unique(as.character(out$subjectHits))
  index2<-as.numeric(index2)
  write.csv(cicero_loop[index1,],paste0("Raw/pbmc3k_filter_",i,"_Raw_in_Refcutoff5_Cicero_loops.csv"),row.names = F)
  write.csv(all_pcHiC_s[index2,],paste0("Raw/pbmc3k_filter_",i,"_Raw_in_Refcutoff5_Ref_loops.csv"),row.names = F)
  overlap_in_cicero<-rbind(overlap_in_cicero,c(i,length(index1),length(index1)/dim(cicero_loop)[1],"Raw"))
  overlap_in_Ref<-rbind(overlap_in_Ref,c(i,length(index2),length(index2)/dim(all_pcHiC_s)[1],"Raw"))
  p=p+1
}
write.csv(overlap_in_cicero,"./pbmc3kCor0.5PGspecific_overlap_RefCelltypeCutoff5_percentage_in_cicero.csv",row.names = F,quote = F)
write.csv(overlap_in_Ref,"./pbmc3kCor0.5PGspecific_overlap_RefCelltypeCutoff5_percentage_in_Ref.csv",row.names = F,quote = F)


