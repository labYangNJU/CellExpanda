latent<-read.table("latent-512-24.txt",header = F,sep = " ")
dim(latent)
library(umap)
library(uwot)
umap<-uwot::umap(latent,n_neighbors =30,min_dist=0.3,metric = 'cosine')
head(umap)
colnames(umap)<-c("UMAP1","UMAP2")
write.csv(umap,"UMAP.csv",row.names = F)

