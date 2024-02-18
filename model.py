

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from log_likelihood import log_zinb_positive, log_nb_positive
from modules import Encoder, Decoder, Discriminator, reparameterize_gaussian
from utils import one_hot
import math
import numpy as np
from sklearn.mixture import GaussianMixture

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from pytorch_metric_learning import losses

from utils import *

torch.backends.cudnn.benchmark = True

adversarial_loss = torch.nn.BCELoss()
l1loss = torch.nn.L1Loss()
nllloss = torch.nn.NLLLoss()
mseloss = torch.nn.MSELoss()

_eps = 1e-15



# VAE model
class GAMM(nn.Module):
    def __init__(self, dataset, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dataset = dataset
        
        self.px_r = torch.nn.Parameter(torch.randn(dataset.X.shape[1]))
        
        self.pi = nn.Parameter(torch.ones(12)/12)  # pc
        self.mu_c = nn.Parameter(torch.zeros(config['model_params']['z_encoder']['n_latent'], 12)) # mu
        self.var_c = nn.Parameter(torch.ones(config['model_params']['z_encoder']['n_latent'], 12)) # sigma^2
        
        self.z_encoder = Encoder(dataset.X.shape[1], 
            config['model_params']['z_encoder']['n_latent'],
            n_layers=config['model_params']['z_encoder']['n_layers'],
            n_hidden=config['model_params']['z_encoder']['n_hidden'],
            dropout_rate=config['model_params']['z_encoder']['dropout']
        ).to(device)
          
        self.l_encoder = Encoder(dataset.X.shape[1], 1, n_layers=1, n_hidden=config['model_params']['z_encoder']['n_hidden']).to(device)
        
        self.discriminators = []
        for dim in self.config['data_params']['dims']:
            discriminator = Discriminator(dim[1]-dim[0], config['model_params']['discriminator']['n_hidden']).to(device)
            self.discriminators.append(discriminator)
            
        self.decoder = Decoder(
                config['model_params']['z_encoder']['n_latent'],
                dataset.X.shape[1],
                n_layers=config['model_params']['decoder']['n_layers'],
                n_hidden=config['model_params']['decoder']['n_hidden'],
            ).to(device)
        
        if 'selective_weight' in config['data_params']:
            self.index_sels  = selective_dataset(config)
            self.index_sels[0] = (np.array(self.index_sels[0])-1).tolist()
            
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, px_alpha, global_step=0):
        reconst_loss = 0
        items = []
        for i, dim in enumerate(self.config['data_params']['dims']):
            if self.config['data_params']['reconstruction_loss'][i] == 'zinb':
                zinb = -log_zinb_positive(x[:, dim[0]:dim[1]], px_rate[:, dim[0]:dim[1]], px_r[dim[0]:dim[1]], px_dropout[:, dim[0]:dim[1]])
                items.append(zinb)
            elif self.config['data_params']['reconstruction_loss'][i] == 'nb':
                nb = -log_nb_positive(x[:, dim[0]:dim[1]], px_rate[:, dim[0]:dim[1]], px_r[dim[0]:dim[1]])
                items.append(nb)
        items.reverse()
        all_loss = torch.cat(items, 1)
        
        
        if 'selective_weight' in self.config['data_params']:
            all_loss[:, self.index_sels[0]] = all_loss[:, self.index_sels[0]] * self.config['data_params']['selective_weight']
        reconst_loss = torch.sum(all_loss, dim=-1)
        
        return reconst_loss
            
    def get_latent(self, x):
        batch_size = 256
        batch_num = (x.shape[0] + batch_size - 1) // batch_size
        
        latent = torch.Tensor().to(self.device)
        for i in range(batch_num):
            X_batch = torch.FloatTensor(x[i*batch_size:(i+1)*batch_size]).to(self.device)
            #self.z_encoder.cuda()
            qz_m, qz_logv, z = self.z_encoder(X_batch)
            
            latent = torch.cat((latent, qz_m), 0)
        return latent
 

    def forward(self, x, local_l_mean, local_l_var, writer=None, global_step=0):
        x_ = x        
        
        qz_m, qz_v, z = self.z_encoder(x_, global_step=global_step)
        
        ql_m, ql_v, library = self.l_encoder(x_)

        px_scale, _, px_rate, px_dropout, px_alpha = self.decoder(z, library, global_step=global_step)
        px_r = torch.exp(self.px_r)
        
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))).sum(dim=1)
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(torch.full(ql_m.shape, local_l_mean[0][0]).to(self.device), torch.full(ql_v.shape, np.sqrt(local_l_var[0][0])).to(self.device))).sum(dim=1)     
        
        reconst_loss = self.get_reconstruction_loss(x_, px_rate, px_r, px_dropout, px_alpha, global_step)

        valid = Variable(torch.FloatTensor(x.size(0), 1).to(self.device).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(x.size(0), 1).to(self.device).fill_(0.0), requires_grad=False)

        idx = torch.randperm(self.dataset.X.shape[0])
        x_real = torch.FloatTensor(self.dataset.X[idx[:x.size(0)]]).to(self.device)

        # generate sample from random priors
        z_prior = reparameterize_gaussian(torch.zeros(qz_m.shape).to(self.device), torch.ones(qz_v.shape).to(self.device))
        l_prior = reparameterize_gaussian(torch.zeros(ql_m.shape).to(self.device), torch.ones(ql_v.shape).to(self.device))
        _, _, x_fake_gen, x_dropout, _ = self.decoder(z_prior, l_prior)
        _, _, z_rec = self.z_encoder(x_fake_gen)
        x_fake = Normal(torch.zeros(x.shape), torch.ones(x.shape)).rsample().to(self.device)
        #_, _, x_fake, x_dropout, _ = self.decoder(z_prior, l_prior)

        z_rec_loss = l1loss(z_rec, z_prior)
        
        g_loss, d_loss = 0, 0
        for i, dim in enumerate(self.config['data_params']['dims']):
            discriminator = self.discriminators[i]
            d_gen_mid, d_gen = discriminator(px_rate[:, dim[0]:dim[1]])
            _, d_fake = discriminator(x_fake[:, dim[0]:dim[1]])
            d_real_mid, d_real = discriminator(x_[:, dim[0]:dim[1]])
            

            # G loss: geneate undistinguishables: px_rate and x_fake 
            g_loss += adversarial_loss(d_gen, valid) + adversarial_loss(d_fake, valid)

            # D loss: distinguish real (x)  and fakes (px_rate, x_fake)
            d_loss += adversarial_loss(d_real, valid) + adversarial_loss(d_gen.detach(), fake) + adversarial_loss(d_fake, fake)

            
            if global_step % 100 == 0:
                writer.add_scalar('g_loss/gen/%d'%(i), adversarial_loss(d_gen, valid), global_step=global_step)
                #writer.add_scalar('g_loss/fake_gen/%d'%(i), adversarial_loss(d_fake_gen, valid), global_step=global_step)
                #writer.add_scalar('g_loss/fake/%d'%(i), adversarial_loss(d_fake, valid), global_step=global_step)

                writer.add_scalar('d_loss/real/%d'%(i), adversarial_loss(d_real, valid), global_step=global_step)
                writer.add_scalar('d_loss/gen/%d'%(i), adversarial_loss(d_gen.detach(), fake), global_step=global_step)
                writer.add_scalar('d_loss/fake/%d'%(i), adversarial_loss(d_fake, fake), global_step=global_step)
                
                writer.add_scalar('reconst_loss/%d'%(i), reconst_loss.mean(), global_step=global_step)
        
        return reconst_loss, kl_divergence_z+kl_divergence_l, z_rec_loss, g_loss, d_loss
        