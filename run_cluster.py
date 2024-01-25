
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from trainer import UnsupervisedTrainer


import torch
from dataset import SCDataset
from model import *
import pickle
import random
import torchsummary

from utils import *
from load import *
import argparse

save_path = 'models/'

plt.switch_backend("agg")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def main(args):
    set_seed(args.seed)
    use_batches = True if args.n_batch>0 else False
    
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(args.gpu)
        use_cuda = True if torch.cuda.is_available() else False
    else:
        use_cuda = False
    
    X, cells, peaks, labels, cell_types, tlabels, batch_ids, batches = load_dataset(args)   # downloaded
    
    if args.binary:
        X = np.where(X>0, 1, 0) 
     
    #X = tfidf(X)
    #X = read_tf().T
    #for i in range(X.shape[0]):
    #    X[i, :] = (X[i, :]- X[i, :].min())/(X[i, :].max()-X[i, :].min())
    #X = np.where(X>1.5, 1.5, X)
    #X = np.where(X<-1.5, -1.5, X)
    print('newly load data X', X.shape, X.max(), X.min())
    
    X, peaks, barcode, _, _ = filter_dataset(X, cells, low=args.min_cells, high=args.max_cells, min_peaks=args.min_peaks)
    labels = [labels[i] for i in barcode if labels is not None]
    tlabels = [tlabels[i] for i in barcode if tlabels is not None]
    cells = [cells[i] for i in barcode if cells is not None]
    batches = np.array([batches[i] for i in barcode if batches is not None])  # [0, 0, 1], [0, 1, 0], [1, 0, 0]
    batch_ids = np.array([batch_ids[i] for i in barcode if batch_ids is not None])  # 0, 1, 2, 
    
    index_sels = None
    if args.selective_weight > 0:
        index_sels  = selective_dataset(args)
        index_sels[0] = (np.array(index_sels[0])-1).tolist()
        
    print('filter data info', X.shape, X.max(), X.min(), len(peaks), len(barcode), len(batches), batches[0], args.n_batch, batch_ids[0])
    
    gene_dataset = SCDataset('models/', mat=X, ylabels=labels, tlabels=tlabels, cell_types=cell_types, batch_ids=batch_ids)  # out of memory 
    if args.n_batch > 0:
        gene_dataset.X = np.array(gene_dataset.X, dtype=np.float32)
        gene_dataset.X = batch_removal(gene_dataset.X, batch_ids)
    
    print('labels', len(labels), len(batches), args.n_batch)
    
    
    if use_batches:
        batch_indices = batches
    else:
        batch_indices = None

    #np.savetxt('result/%s/%d-%d-%d,txt'%(args.dataset, args.n_hidden, args.n_latent, args.n_louvain), latent)
    latent = np.loadtxt('result/%s/latent-%d-%d.txt'%(args.dataset, args.n_hidden, args.n_latent))
    clustering_scores(args, latent, labels, cells, args.dataset, '%d-%d-%d'%(args.n_hidden, args.n_latent, args.n_louvain), tlabels,
                   batch_indices=batch_indices)
    #np.savetxt('result/%s/%d-%d-%d-Accuracy.txt'%(args.dataset, args.n_hidden, args.n_latent, args.n_louvain), scores,fmt="%.4f",)
     
    
params = {
    'GSE99172': [512, 10, 50],
    'GSE96769': [1024, 20, 200],
    'GSE112091': [128, 40, 150],
    'forebrain-scale':[1024, 32, 50],
    'GM12878vsHEK': [128, 8, 150],
    'GM12878vsHL': [512, 24, 150],   # SCALE: 0.817, 0.866
    'Splenocyte':[512, 16, 50],
    'atac_pbmc_1k_merge': [128, 16, 50],
    'scChip-seq': [128, 16, 50],
    'scRNA_cortex': [64, 14, 50],
    'ZY_bin_cell_matrix': [128, 16, 50]
}
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAAE: Generative Adversarial ATAC-RNA-seq Analysis')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--n_hidden', type=int, help='hidden unit number', default=1024)
    parser.add_argument('--n_latent', type=int, help='latent size', default=10)
    parser.add_argument('--n_louvain', type=int, help='louvain number', default=10)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for repeat results')
    parser.add_argument('--gpu', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--min_peaks', type=float, default=100, help='Remove low quality cells with few peaks')
    parser.add_argument('--min_cells', type=float, default=0.05, help='Remove low quality peaks')
    parser.add_argument('--max_cells', type=float, default=0.95, help='Remove low quality peaks')
    parser.add_argument('--n_epochs', type=int, default=200, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout in the fc layer')
    parser.add_argument('--binary', type=int, default=1, help='binarize the data')
    parser.add_argument('--n_batch', type=int, default=0, help='the number of batch in the data')
    parser.add_argument('--prediction_algorithm', type=str, default='louvain', help='leiden, louvain, kmeans, hierarchical')
    parser.add_argument('--k_cluster', type=int, default=10, help='number of clusters in kmeans')
    parser.add_argument('--plot', type=str, default='tsne', help='tsne, umap')
    parser.add_argument('--labeled', type=int, default=1, help='has label data (cell type file)') # 1 stands for existing of celltype file
    parser.add_argument('--RNA_dim', type=int, default=-1, help='the dimension of RNA') # 1 stands for existing of celltype file
    parser.add_argument('--selective_weight', type=float, default=1, help='selective weight')
    parser.add_argument('--discriminator', type=int, default=1, help='discriminator_num')
    parser.add_argument('--reconstruction_loss', type=str, default="alpha-gan", help='reconstruction loss')
    
    args = parser.parse_args()
    main(args)
    
