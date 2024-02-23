import torch

import community
import networkx as nx
import os, os.path

import numpy as np
from matplotlib import pyplot as plt
import leidenalg
import igraph as ig
import umap.umap_ as umap

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from scipy.optimize import linear_sum_assignment as linear_assignment
#from sklearn.utils.linear_assignment_ import linear_assignment
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
from tensorboardX import SummaryWriter

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = True

#FLOAT = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
#LONG = torch.cuda.LongTensor if use_cuda else torch.LongTensor

    

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
    # ind 2 (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([8, 1, 5, 0, 3, 2, 4, 6, 7, 9])) (10, 10) (2714,) (2714,)
    #print('ind', len(ind), ind, reward_matrix.shape, np.array(y).shape, np.array(y_pred).shape) 
    #return 0
    return sum([reward_matrix[ind[0][i], ind[1][i]] for i in range(len(ind[0]))]) * 1.0 / len(y_pred)

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

def clustering(config, latent):
    from scipy.spatial import distance
    vec = latent
    
    
    #alg = 'louvain'
    if config['clustering_params']['alg'] == 'louvain':
        mat = kneighbors_graph(latent, config['clustering_params']['n_louvain'], mode='distance', include_self=True).todense()
        labels_pred = []
        G = nx.from_numpy_matrix(mat)
        partition = community.best_partition(G, random_state=config['exp_params']['seed'])
        for i in range(mat.shape[0]):
            labels_pred.append(partition[i])
    elif config['clustering_params']['alg'] == 'leiden':
        mat = kneighbors_graph(latent, config['clustering_params']['n_louvain'], mode='distance', include_self=True).todense()
        vcount = max(mat.shape)
        sources, targets = mat.nonzero()
        edgelist = zip(sources.tolist(), targets.tolist())
        g = ig.Graph(vcount, edgelist)
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        
        labels_pred = partition.membership
    elif config['clustering_params']['alg'] == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=config['clustering_params']['n_louvain'], random_state=0).fit(latent)
        
        joblib.dump(kmeans, 'model/%s-%d-%d.kmeans.joblib'%(config['data_params']['name'], config['model_params']['z_encoder']['n_hidden'], config['clustering_params']['n_louvain'])) 
        
        labels_pred = kmeans.labels_
    elif config['clustering_params']['alg'] == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=config['clustering_params']['n_louvain']).fit(latent)
        labels_pred = model.labels_
        
    return labels_pred

def plot_tsne(config, dataset, latent, labels, labels_pred):
    labels_pred = np.array(labels_pred)

    dirname = 'result/%s'%(config['data_params']['name'])
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    
    if config['clustering_params']['plot'] == 'tsne':
        embedding = TSNE(random_state=config['exp_params']['seed'], perplexity=50).fit_transform(latent)  
    elif config['clustering_params']['plot'] == 'umap':
        embedding = umap.UMAP(n_neighbors=30, min_dist=0.3,metric = 'cosine').fit_transform(latent)
        

        
    print('pred labels is', labels_pred.shape, np.max(labels_pred), embedding[:5], labels_pred[:100])
    print('labels is', np.array(labels).shape)
    show_tsne(embedding, labels_pred, 'result/%s/CellExpanda-%s-%s-pred.png'%(config['data_params']['name'], config['data_params']['name'], config['clustering_params']['plot']))
    np.savetxt('result/%s/labels-%d-%d.txt'%(config['data_params']['name'], config['model_params']['z_encoder']['n_hidden'], config['clustering_params']['n_louvain']), labels_pred)

    #if labels is not None:   
    result_filename = 'result/%s-%d-%d-cluster_result.csv'%(config['data_params']['name'], config['model_params']['z_encoder']['n_hidden'], config['clustering_params']['n_louvain'])
    
    if config['data_params']['labeled']:
        show_tsne(embedding, labels, 'result/%s/CellExpanda-%s-%s-true.png'%(config['data_params']['name'], config['data_params']['name'], config['clustering_params']['plot']), tlabels=dataset.tlabels)

        with open(result_filename, 'w') as f:
            f.write('cell,tlabel id,label,predicted label,%s-1,%s-2\n'%(config['clustering_params']['plot'],config['clustering_params']['plot']))
            for cell, label, tlabel, pred, t in zip(dataset.cells, labels, dataset.tlabels, labels_pred, embedding):
                f.write('%s,%d,%s,%d,%f,%f\n'%(cell, label, tlabel, pred, t[0], t[1]))
    else:
        with open(result_filename, 'w') as f:
            f.write('cell,predicted label,%s-1,%s-2\n'%(config['clustering_params']['plot'],config['clustering_params']['plot']))
            for cell, pred, t in zip(dataset.cells, labels_pred, embedding):
                f.write('%s,%d,%f,%f\n'%(cell, pred, t[0], t[1]))
        

def clustering_score(labels, labels_pred, output=True):
    #print(labels, labels_pred, latent)
    #asw_score = silhouette_score(latent, labels)
    asw_score = 0
    #print(len(labels), labels[:10], len(labels_pred), labels_pred[:10])
    nmi_score = NMI(labels, labels_pred)
    ari_score = ARI(labels, labels_pred)
    homo_score = homogeneity_score(labels, labels_pred) 
    uca_score = unsupervised_clustering_accuracy(labels, labels_pred)
    if output:
        print("Clustering Scores:\nHOMO: %.4f\nNMI: %.4f\nARI: %.4f\nUCA: %.4f"%(homo_score, nmi_score, ari_score, uca_score))

    return homo_score, nmi_score, ari_score, uca_score


def read_selectives(args, filename):
    
    with open(filename) as f:
        i = 0
        for line in f.readlines():
            values = line.strip('\n').split(',')
            if i == 0:
                selective_peaks = [[] for i in range(len(values)-1)]
            else:
                for j, value in enumerate(values[1:]):     #enumerate遍历函数，遍历values中的值，j为index,从0开始，value为值
                    if int(value) == 1 :
                        selective_peaks[j].append(i)
                    
            i+=1

    return selective_peaks

# peak_indices [     4      7     15 ... 193206 193207 193208]
def selective_dataset(config):
    selective_peaks = read_selectives(config, os.path.join(config['data_params']['data_path'], 'feature_discriminator.txt'))

    return selective_peaks
