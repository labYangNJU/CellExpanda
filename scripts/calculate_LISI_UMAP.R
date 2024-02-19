library(lisi)

##ASAP
accuracy<-read.csv("cluster_info_for_accuracy_wnn.csv",header = T)
head(accuracy)
##LISI
emb<-read.csv("ASAP_WNN_umap.csv",header = T,row.names = 1)
lisi_res <- compute_lisi(emb, accuracy, "tlabel")
lisi_res$daoshu<-1/lisi_res$tlabel
write.csv(lisi_res,"ASAP_WNN_cLISI_UMAP.csv",row.names = F)

