import os, sys
import torch
import torch.optim as optim

from abc import abstractmethod
from collections import defaultdict, OrderedDict
from itertools import cycle, chain

from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.optim import RMSprop, Adam, SGD
from tqdm import trange
from sklearn.cluster import KMeans
from pytorch_metric_learning import losses
from typing import Dict, Iterable, List, Tuple, Union, Optional, Callable
import scipy.sparse as sp_sparse
from sklearn.metrics import silhouette_samples, silhouette_score

from torch.utils.data import DataLoader

from utils import *

contrastive_loss = losses.ContrastiveLoss(pos_margin=0.5, neg_margin=2)
pdist = torch.nn.PairwiseDistance(p=2)




class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.02):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_loss = -np.inf

    def early_stop(self, validation_loss):
        print('early stop', validation_loss < self.max_validation_loss - self.min_delta, validation_loss, self.max_validation_loss - self.min_delta, self.counter)
        if validation_loss > self.max_validation_loss:
            self.max_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss < (self.max_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(config, model, dataset, device):
    writer = SummaryWriter('result/logs')
    
    optimizer_e = torch.optim.Adam(chain(model.z_encoder.parameters(), 
                                         model.l_encoder.parameters(),
                                    model.decoder.parameters()),
                                    lr=config['trainer_params']['lr_e'], eps=config['trainer_params']['eps'], weight_decay=config['trainer_params']['weight_decay'])
    
    optimizer_d = torch.optim.Adam(chain(*[d.parameters() for d in model.discriminators]), 
                                   lr=config['trainer_params']['lr_d'], eps=config['trainer_params']['eps'], weight_decay=config['trainer_params']['weight_decay'])
    
    
    global_step = 0
    max_validation_loss = -np.inf
    labels = None
    if config['trainer_params']['early_stop']:
        early_stopper = EarlyStopper(patience=config['trainer_params']['patience'], min_delta=config['trainer_params']['min_delta'])
    params = '%s-%d-%d'%(config['data_params']['name'], config['model_params']['z_encoder']['n_hidden'], config['clustering_params']['n_louvain'])
    
    
    random_state = np.random.RandomState(seed=0)
    permutation = random_state.permutation(dataset.X.shape[0])
    idx = permutation
    
    #data_loader_kwargs = {'batch_size': config['trainer_params']['batch_size'], 'pin_memory': True}
    torch.manual_seed(0)
    data_loader_kwargs = {'batch_size': config['trainer_params']['batch_size'], 'pin_memory': True, "sampler": SubsetRandomSampler(idx)}
    data_loader = DataLoader(dataset, **data_loader_kwargs)
    
    
    with trange(config['trainer_params']['n_epochs'], desc="training", file=sys.stdout, disable=False) as pbar:
        for epoch in pbar:
            pbar.update(1)
            
            
            if config['trainer_params']['early_stop'] and epoch % 50 == 0 and epoch>50:
                # kmeans clustering
                model.eval()
                latent = model.get_latent(dataset.X)

                mat = kneighbors_graph(latent.detach().cpu().numpy(), config['clustering_params']['n_louvain'], mode='distance', include_self=True).todense()
                labels = []
                G = nx.from_numpy_matrix(mat)
                partition = community.best_partition(G, random_state=config['exp_params']['seed'])
                for i in range(mat.shape[0]):
                    labels.append(partition[i])
                homo_score, nmi_score, ari_score, uca_score = clustering_score(dataset.labels, labels, output=False)
                silhouette_avg = silhouette_score(dataset.X, labels)
                writer.add_scalar('score/homo', homo_score, global_step=epoch)
                writer.add_scalar('score/nmi', nmi_score, global_step=epoch)
                writer.add_scalar('score/ari', ari_score, global_step=epoch)
                writer.add_scalar('score/silhouette', silhouette_avg, global_step=epoch)
                
                
                if silhouette_avg > max_validation_loss:
                    max_validation_loss = silhouette_avg
                    torch.save(model.state_dict(), 'model/%s-best.pth'%(params))
                    print('latent', latent.shape, latent.max(), latent.min())
                    print('current', epoch, silhouette_avg, 'max', max_validation_loss)
                    print('current scores', homo_score, nmi_score, ari_score)
                torch.save(model.state_dict(), 'model/%s-last.pth'%(params))
                
                if early_stopper.early_stop(silhouette_avg):             
                    break
            
            
            j = 0
            model.train()
            for indices in data_loader:
                X_batch = torch.FloatTensor(dataset.X[indices]).to(device)
                #print(global_step, 'X_batch', X_batch.mean())
                
                reconst_loss, kl_divergence, z_rec_loss, g_loss, d_loss = model.forward(X_batch, dataset.local_l_mean, dataset.local_l_var, writer=writer, global_step=global_step)
                
                kl_weight = min(1, epoch / 400)
                recon_weight = 0.1
                gan_weight = 1
                z_recover_weight = 1
                reconst_loss = torch.mean(reconst_loss*recon_weight + kl_weight * kl_divergence)
                
                vae_loss = reconst_loss + z_recover_weight*z_rec_loss + gan_weight*g_loss
                
                #break
                if epoch > 0:
                    model.z_encoder.requires_grad = True
                    model.decoder.requires_grad = True
                    for discriminator in model.discriminators:
                        discriminator.requires_grad = False
                    optimizer_e.zero_grad()
                    vae_loss.backward(retain_graph=True)   # decoder to reconstruct
                    optimizer_e.step()


                    # discriminator loss, freeze E and G, update D
                    model.z_encoder.requires_grad = False
                    model.decoder.requires_grad = False
                    for discriminator in model.discriminators:
                        discriminator.requires_grad = True
                    optimizer_d.zero_grad()
                    d_loss.backward()
                    optimizer_d.step()
                    #print(global_step, 'd loss', d_loss)
                
                
                if global_step % 100 == 0:
                    #writer.add_scalar('vae_loss/con_loss', con_loss, global_step=global_step)
                    writer.add_scalar('vae_loss/reconst_loss', torch.mean(reconst_loss), global_step=global_step)
                    #writer.add_scalar('vae_loss/inter_loss', inter_loss, global_step=global_step)
                    writer.add_scalar('vae_loss/all', vae_loss, global_step=global_step)
                    writer.add_scalar('vae_loss/kl_divergence', kl_divergence.mean(), global_step=global_step)
                    writer.add_scalar('vae_loss/g_loss', g_loss, global_step=global_step)
                    writer.add_scalar('d_loss/all', d_loss, global_step=global_step)
                
                
                global_step += 1
                j += 1
    model.eval()
    torch.save(model.state_dict(), 'model/%s-best.pth'%(params))
    return labels
        
        
            

            

        