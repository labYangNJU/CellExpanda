import torch

import community
import networkx as nx
import os, os.path

import numpy as np
from matplotlib import pyplot as plt
import leidenalg
import igraph as ig
import umap

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
#from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE

from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score, homogeneity_score

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (
    SequentialSampler,
    SubsetRandomSampler,
    RandomSampler,
)
import joblib 
import argparse
import pickle
import random
import torchsummary
from load import *

use_cuda = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor
LONG = torch.cuda.LongTensor

def batch_removal(X, batchids):
    for b in [0, 1, 2]:
        indices = [i for i in range(X.shape[0]) if int(batchids[i])==b]
        T = X[indices][:]
        
        mean, std = np.mean(X[indices]), np.std(X[indices])        
        X[indices] = (X[indices]-mean)/std
    return X


'''def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False
def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True
                
                

    
def collate_fn_builder(X, batch_indices=None, labels=None, corrupted=True):
    # sample_batch, local_l_mean, local_l_var, batch_index, _
    attributes_and_types = dict(
        [
            ("X", np.float32) if not corrupted else ("corrupted_X", np.float32),
            ("local_means", np.float32),
            ("local_vars", np.float32),
            ("batch_indices", np.int64),
            ("labels", np.int64),
        ]
    )
    
    local_means = np.zeros((X.shape[0], 1))  # [128, 1] 5.6499
    local_vars = np.zeros((X.shape[0], 1))   # [128, 1] 0.4496
    batch_indices = np.zeros((X.shape[0], 1))
    labels = np.zeros((X.shape[0], 1))
    
    sum_counts = X.sum(axis=1)
    masked_log_sum = np.ma.log(sum_counts)

    log_counts = masked_log_sum.filled(0)
    local_mean = (np.mean(log_counts).reshape(-1, 1)).astype(np.float32)
    local_var = (np.var(log_counts).reshape(-1, 1)).astype(np.float32)
    
    for i in range(len(local_means)):
        local_means[i], local_vars[i] = local_mean, local_var
    
    #batch_indices = np.asarray(batch_indices[:, 1]).reshape((-1, 1)) if batch_indices is not None else np.zeros((X.shape[0], 1)),
    #        categorical=True

    data_numpy = [X, local_means, local_vars, batch_indices, labels]
    
    data_torch = tuple(torch.cuda.FloatTensor(d).cuda() for d in data_numpy)
    return data_torch'''
                
           
class SequentialSubsetSampler(SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)
    

def get_latent(gene_dataset, model, use_cuda=True, batch_size=128):
    latent = []
    batch_indices = []
    labels = []
    
    
    sampler = SequentialSampler(gene_dataset)
    data_loader_kwargs = {"collate_fn": gene_dataset.collate_fn_builder(), "sampler": sampler, "batch_size": 128}
    data_loader = DataLoader(gene_dataset, **data_loader_kwargs)
    
    for tensors in data_loader:
        sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
        if use_cuda:
            
            latent += [model.sample_from_posterior_z(sample_batch.cuda(), give_mean=True).cpu()]
        else:
            latent += [model.sample_from_posterior_z(sample_batch, give_mean=True).cpu()]
    return np.array(torch.cat(latent).detach())


def batch_removal(X, batchids):
    for b in [0, 1, 2]:
        indices = [i for i in range(X.shape[0]) if int(batchids[i])==b]
        T = X[indices][:]
        
        mean, std = np.mean(X[indices]), np.std(X[indices])        
        X[indices] = (X[indices]-mean)/std
    return X
        

def read_selectives(args, filename):
    
    with open(os.path.dirname(__file__)+filename) as f:
        i = 0
        for line in f.readlines():
            values = line.strip('\n').split(',')
            #print('values',values)
            #print(len(values))
            #print(i)
            if i == 0:
                selective_peaks = [[] for i in range(len(values)-1)]
            else:
                for j, value in enumerate(values[1:]):     #enumerate遍历函数，遍历values中的值，j为index,从0开始，value为值
                    if int(value) == 1 :
                        selective_peaks[j].append(i)
                    
            i+=1

    return selective_peaks

# peak_indices [     4      7     15 ... 193206 193207 193208]
def selective_dataset(args):
    dirname = '/../data/%s/'%(args.dataset)
    if args.selective_weight>0:
        #X_sels = []
        selective_peaks = read_selectives(args, dirname+'feature_discriminator.txt')
        
        #print('selective', )
        return selective_peaks
        

def unsupervised_clustering_accuracy(y, y_pred):
    assert len(y_pred) == len(y)
    u = np.unique(np.concatenate((y, y_pred)))
    n_clusters = len(u)
    mapping = dict(zip(u, range(n_clusters)))
    reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for y_pred_, y_ in zip(y_pred, y):
        if y_ in mapping:
            reward_matrix[mapping[y_pred_], mapping[y_]] += 1
    cost_matrix = reward_matrix.max() - reward_matrix
    ind = linear_assignment(cost_matrix)
    return sum([reward_matrix[i, j] for i, j in ind]) * 1.0 / y_pred.size

def show_tsne(embedding, labels, filename, tlabels=None):
    n_components = len(np.unique(labels))
    
    vis_x = embedding[:, 0]
    vis_y = embedding[:, 1]
    colors = ['slategray', 'red', 'green', 'tan', 'purple', 'brown', 'pink', 'yellow', 'black', 'teal', 'plum', 'bisque', 'orange', 'beige', 'blue', 'OliveDrab', 'darkred', 'salmon', 'coral', 'olive', 'lightpink', 'lime', 'darkcyan', 'BlueViolet', 'CornflowerBlue', 'DarkKhaki', 'DarkTurquoise']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    for i, y in enumerate(range(n_components)):

        indexes = [j for j in range(len(labels)) if labels[j]==y]
        vis_x1 = embedding[indexes, 0]
        vis_y1 = embedding[indexes, 1]
        try: 
            c = colors[i]
        except IndexError:
            c = colors[0]

        if tlabels is None:
            sc = plt.scatter(vis_x1, vis_y1, c=c, marker='.', cmap='hsv', s=30, label=y)
        else:
            sc = plt.scatter(vis_x1, vis_y1, c=c, marker='.', cmap='hsv', s=30, label=tlabels[indexes[0]])
        
    ax.legend()
    plt.savefig(filename)
    plt.clf()
    plt.close()


def one_hot(index, n_cat):
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

def log_nb_positive(x, mu, theta, eps=1e-8):
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return res
    #return torch.sum(res, dim=-1)

def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)
    
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res
    #return torch.sum(res, dim=-1)

def clustering_scores(args, latent, labels, cells, dataset, suffix, tlabels, batch_indices=None):
    from scipy.spatial import distance
    vec = latent
    
    
    #alg = 'louvain'
    if args.prediction_algorithm == 'louvain':
        mat = kneighbors_graph(latent, args.n_louvain, mode='distance', include_self=True).todense()
        labels_pred = []
        G = nx.from_numpy_matrix(mat)
        partition = community.best_partition(G, random_state=args.seed)
        for i in range(mat.shape[0]):
            labels_pred.append(partition[i])
    elif args.prediction_algorithm == 'leiden':
        mat = kneighbors_graph(latent, args.n_louvain, mode='distance', include_self=True).todense()
        vcount = max(mat.shape)
        sources, targets = mat.nonzero()
        edgelist = zip(sources.tolist(), targets.tolist())
        g = ig.Graph(vcount, edgelist)
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        
        labels_pred = partition.membership
    elif args.prediction_algorithm == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=args.k_cluster, random_state=0).fit(latent)
        
        joblib.dump(kmeans, 'model/%s-%d-%d.kmeans.joblib'%(args.dataset, args.n_hidden, args.n_latent)) 
        
        labels_pred = kmeans.labels_
    elif args.prediction_algorithm == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=args.k_cluster).fit(latent)
        labels_pred = model.labels_

    labels_pred = np.array(labels_pred)

    if args.plot == 'tsne':
        embedding = TSNE(random_state=args.seed, perplexity=50).fit_transform(latent)  
    elif args.plot == 'umap':
        embedding = umap.UMAP(random_state=42, min_dist=0.01).fit_transform(latent)  
        
    dirname = 'result/%s'%(dataset)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        
    print('pred labels is', labels_pred.shape, np.max(labels_pred), vec[0,:5], embedding[:5], labels_pred[:100])
    print('labels is', np.array(labels).shape)
    show_tsne(embedding, labels_pred, 'result/%s/%s-GMVAE-%s-%s-pred.png'%(dataset, suffix, 'alpha-gan', args.plot))
    np.savetxt('result/%s/labels-%d-%d.txt'%(dataset, args.n_hidden, args.n_latent), labels_pred)

    #if labels is not None:   
    result_filename = 'result/%s-%d-%d-%d-%d-cluster_result.csv'%(dataset, args.n_hidden, args.n_latent, args.n_louvain,args.seed)
    if len(labels) == 0:
        with open(result_filename, 'w') as f:
            f.write('cell,predicted label,tsne-1,tsne-2\n')
            for cell, pred, t in zip(cells, labels_pred, embedding):
                f.write('%s,%d,%f,%f\n'%(cell, pred, t[0], t[1]))
        if batch_indices is not None:
            show_tsne(embedding, batch_indices, 'result/%s/%s-%s-batch.png'%(dataset, suffix, 'alpha-gan'), tlabels=batch_indices)
    else:
        show_tsne(embedding, labels, 'result/%s/%s-GMVAE-%s-%s-true.png'%(dataset, suffix, 'alpha-gan', args.plot), tlabels=tlabels)
        if batch_indices is None:
            with open(result_filename, 'w') as f:
                f.write('cell,tlabel id,label,predicted label,tsne-1,tsne-2\n')
                for cell, label, tlabel, pred, t in zip(cells, labels, tlabels, labels_pred, embedding):
                    f.write('%s,%d,%s,%d,%f,%f\n'%(cell, label, tlabel, pred, t[0], t[1]))
        else:
            with open(result_filename, 'w') as f:
                f.write('cell,tlabel id,label,predicted label,tsne-1,tsne-2,batch\n')
                for cell, label, tlabel, pred, t, batch in zip(cells, labels, tlabels, labels_pred, embedding, batch_indices):
                    f.write('%s,%d,%s,%d,%f,%f,%d\n'%(cell, label, tlabel, pred, t[0], t[1], args.n_batch))

        #print(labels, labels_pred, latent)
        #asw_score = silhouette_score(latent, labels)
        asw_score = 0
        nmi_score = NMI(labels, labels_pred)
        ari_score = ARI(labels, labels_pred)
        homo_score = homogeneity_score(labels, labels_pred) 
        uca_score = unsupervised_clustering_accuracy(labels, labels_pred)
        print("Clustering Scores:\nHOMO: %.4f\nNMI: %.4f\nARI: %.4f\nUCA: %.4f"%(homo_score, nmi_score, ari_score, uca_score))
        
        with open('result/%s/%s-%d-Accuracy.txt'%(dataset, suffix,args.seed), 'w') as f:
            f.write('\nHOMO: %.4f\nNMI: %.4f\nARI:%.4f\nUCA: %.4f' %(homo_score, nmi_score, ari_score, uca_score))
            
        if batch_indices is not None:
            #print('batch indices', labels, tlabels, np.argmax(batch_indices, axis=1))
            show_tsne(embedding, np.argmax(batch_indices, axis=1), 'result/%s/%s-%s-batch.png'%(dataset, suffix, 'alpha-gan'))
        return asw_score, nmi_score, ari_score, 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAAE: Generative Adversarial ATAC-RNA-seq Analysis')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--n_hidden', type=int, help='hidden unit number', default=1024)
    parser.add_argument('--n_latent', type=int, help='latent size', default=10)
    parser.add_argument('--n_louvain', type=int, help='louvain number', default=50)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for repeat results')
    parser.add_argument('--gpu', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--min_peaks', type=float, default=100, help='Remove low quality cells with few peaks')
    parser.add_argument('--min_cells', type=float, default=0.05, help='Remove low quality peaks')
    parser.add_argument('--max_cells', type=float, default=0.95, help='Remove low quality peaks')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout in the fc layer')
    parser.add_argument('--binary', type=int, default=1, help='binarize the data')
    parser.add_argument('--n_batch', type=int, default=0, help='the number of batch in the data')
    parser.add_argument('--prediction_algorithm', type=str, default='louvain', help='leiden, louvain, kmeans, hierarchical')
    parser.add_argument('--k_cluster', type=int, default=10, help='number of clusters in kmeans')
    parser.add_argument('--plot', type=str, default='tsne', help='tsne, umap')
    parser.add_argument('--labeled', type=int, default=1, help='has label data (cell type file)') # 1 stands for existing of celltype file
    

    args = parser.parse_args()
    
    
    latent = np.loadtxt('512-24-10.txt')
    print(latent.shape)
    
    X, cells, peaks, labels, cell_types, tlabels, batch_ids, batches = load_dataset(args)   # downloaded
    
    if args.binary:
        X = np.where(X>0, 1, 0) 
    
    X, peaks, barcode, _, _ = filter_dataset(X, cells, low=args.min_cells, high=args.max_cells, min_peaks=args.min_peaks)
    labels = [labels[i] for i in barcode if labels is not None]
    tlabels = [tlabels[i] for i in barcode if tlabels is not None]
    cells = [cells[i] for i in barcode if cells is not None]
    
    clustering_scores(args, latent, labels, cells, args.dataset, '%d-%d-%d'%(args.n_hidden, args.n_latent, args.n_louvain), tlabels, 
                   batch_indices=None)



