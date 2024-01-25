## calculate in python
import scipy
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

def overcorrection_score(emb, celltype, n_neighbors=100, n_pools=100, n_samples_per_pool=100, seed=124):
    n_neighbors = min(n_neighbors, len(emb) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(emb)
    kmatrix = nne.kneighbors_graph(emb) - scipy.sparse.identity(emb.shape[0])
    score = 0
    celltype_ = np.unique(celltype)
    celltype_dict = celltype.value_counts().to_dict()
    N_celltype = len(celltype_)
    for t in range(n_pools):
        indices = np.random.choice(np.arange(emb.shape[0]), size=n_samples_per_pool, replace=False)
        score += np.mean([np.mean(celltype[kmatrix[i].nonzero()[1]][:min(celltype_dict[celltype[i]], n_neighbors)] == celltype[i]) for i in indices])
    return 1-score / float(n_pools)
        
        
import pandas as pd
import numpy as np

##Cobolt
emb=pd.read_csv('ASAP_cobolt_UMAP.csv', delimiter=',',usecols=['0','1'])
celltype=pd.read_csv(cluster_info_for_accuracy_cobolt.csv', delimiter=',')
overcorrection_score(emb, celltype.tlabel)


##scAI
emb=pd.read_csv('ASAP_scAI_umap.csv', delimiter=',',usecols=['UMAP1','UMAP2'])
celltype=pd.read_csv('cluster_info_for_accuracy_scAI.csv', delimiter=',')
overcorrection_score(emb, celltype.tlabel)

##scDEC
emb=pd.read_csv('ASAP_UMAP_scDEC_old_uwot.csv', delimiter=',')
celltype=pd.read_csv('cluster_info_for_accuracy_scDEC.csv', delimiter=',')
overcorrection_score(emb, celltype.tlabel)


##scMM
emb=pd.read_csv('ASAP_UMAP_scMM.csv', delimiter=',',usecols=['V1','V2'])
celltype=pd.read_csv('cluster_info_for_accuracy_scMM.csv', delimiter=',')
overcorrection_score(emb, celltype.tlabel)


##MOFA+
emb=pd.read_csv('ASAP_MOFA_UMAP.csv', delimiter=',',usecols=['UMAP1','UMAP2'])
celltype=pd.read_csv('cluster_info_for_accuracy_MOFA.csv', delimiter=',')
overcorrection_score(emb, celltype.tlabel)


##MultiVI
emb=pd.read_csv('umap.csv', delimiter=',')
celltype=pd.read_csv('cluster_info_for_accuracy_MultiVI.csv', delimiter=',')
overcorrection_score(emb, celltype.tlabel)


##WNN
emb=pd.read_csv('ASAP_WNN_umap.csv', delimiter=',',usecols=['wnnUMAP_1','wnnUMAP_2'])
celltype=pd.read_csv('cluster_info_for_accuracy_WNN.csv', delimiter=',')
overcorrection_score(emb, celltype.tlabel)


##EXACT
emb=pd.read_csv('ASAP_UMAP_EXACT_old_uwot.csv', delimiter=',',usecols=['UMAP1','UMAP2'])
celltype=pd.read_csv('cluster_info_for_accuracy_EXACT.csv', delimiter=',')
overcorrection_score(emb, celltype.tlabel)



