
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from trainer import UnsupervisedTrainer


import torch
from dataset import SCDataset
from tensorboardX import SummaryWriter
from model import *


from utils import *
from load import *

save_path = 'models/'

plt.switch_backend("agg")
writer = SummaryWriter('result/logs')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def cluster(args):
    set_seed(args.seed)
    use_batches = True if args.n_batch>0 else False
    
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(args.gpu)
        use_cuda = True if torch.cuda.is_available() else False
    else:
        use_cuda = False
    
    X, cells, peaks, labels, cell_types, tlabels, batch_ids, batches = load_dataset(args)   # downloaded
#print('X[:,99999:100000]',X[:,99999:100000])
    
    if args.binary > 0:
        print("binarize data")
        X = np.where(X>0, 1, 0) 
    
    # filter dataset
    print('X shape',X.shape)
    X, peaks, barcode, _, _ = filter_dataset(X, cells, low=args.min_cells, high=args.max_cells, min_peaks=args.min_peaks)
    labels = [labels[i] for i in barcode if labels is not None]
    tlabels = [tlabels[i] for i in barcode if tlabels is not None]
    cells = [cells[i] for i in barcode if cells is not None]
    batches = np.array([batches[i] for i in barcode if batches is not None])  # [0, 0, 1], [0, 1, 0], [1, 0, 0]
    batch_ids = np.array([batch_ids[i] for i in barcode if batch_ids is not None])  # 0, 1, 2,
    print('X shape',X.shape)

    index_sels = None
    if args.selective_weight > 0:
        index_sels  = selective_dataset(args)
        index_sels[0] = (np.array(index_sels[0])-1).tolist()
    #print('index_sels[0]',index_sels[0])
 
    gene_dataset = SCDataset('models/', mat=X, ylabels=labels, tlabels=tlabels, cell_types=cell_types, batch_ids=batch_ids)  # out of memory 
    if args.n_batch > 0:
        gene_dataset.X = np.array(gene_dataset.X, dtype=np.float32)
        gene_dataset.X = batch_removal(gene_dataset.X, batch_ids)
    
    #print(gene_dataset.X[:, 0:75811].shape, gene_dataset.X[:, 0:75811].max(), gene_dataset.X[:, 0:75811].min(), gene_dataset.X[:, 0:75811].mean())
    #print(gene_dataset.X[:, 75811:91660].shape, gene_dataset.X[:, 75811:91660].max(), gene_dataset.X[:, 75811:91660].min(), gene_dataset.X[:, 75811:91660].mean())
    #print(gene_dataset.X[:, 91660:150751].shape, gene_dataset.X[:, 91660:150751].max(), gene_dataset.X[:, 91660:150751].min(), gene_dataset.X[:, 91660:150751].mean())
    model = GAATAC(args,len(peaks), index_sels=index_sels,n_batch=args.n_batch * use_batches, X=gene_dataset.X,
             n_hidden=args.n_hidden, n_latent=args.n_latent, dropout_rate=args.dropout, use_cuda=use_cuda, RNA_dim=args.RNA_dim,selective_weight=args.selective_weight,reconstruction_loss=args.reconstruction_loss)#, writer=writer)
    if use_cuda:
        model.cuda()
    
    trainer = UnsupervisedTrainer(
        args, model, gene_dataset,
        train_size=1.0, use_cuda=use_cuda, frequency=5,reconstruction_loss=model.reconstruction_loss, batch_size=args.batch_size
    )
    #train(gene_dataset, peaks, barcode, model, args)
    trainer.train(n_epochs=args.n_epochs, writer=writer)
    #latent = get_latent(gene_dataset, trainer.model, use_cuda)
    latent = get_latent(gene_dataset, model, use_cuda=use_cuda)
    
    torch.save(model.state_dict(), 'model/%s-%d-%d.pth'%(args.dataset, args.n_hidden, args.n_latent))
    np.savetxt('result/%s/latent-%d-%d.txt'%(args.dataset, args.n_hidden, args.n_latent), latent)
    
    if use_batches:
        batch_indices = batches
    else:
        batch_indices = None
    clustering_scores(args, latent, labels, cells, args.dataset, '%d-%d-%d'%(args.n_hidden, args.n_latent, args.n_louvain), tlabels, 
                   batch_indices=batch_indices)
    
    
    model.load_state_dict(torch.load('model/%s-%d-%d.pth'%(args.dataset, args.n_hidden, args.n_latent)))
    model.eval()
    #latent = np.loadtxt('result/forebrain/1024-32.txt')
    #print('latent', latent.shape, latent[0])
    clustering_scores(args, latent, labels, cells, args.dataset, '%d-%d-%d'%(args.n_hidden, args.n_latent, args.n_louvain), tlabels)
    #np.savetxt('result/%s/%d-%d-%d-Accuracy.txt'%(args.dataset, args.n_hidden, args.n_latent, args.n_louvain), scores)
     
    
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

# python3.6 main.py --dataset=pbmc3k --n_hidden=512 --n_latent=50 --min_peaks=10 --min_cells=0.01 --max_cells=0.95 --labeled=1 --gpu=0 --binary=1 --prediction_algorithm=kmeans --k_cluster=15
    
    
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
    
            
    cluster(args)
    
