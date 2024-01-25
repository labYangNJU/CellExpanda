

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

#from log_likelihood import log_zinb_positive, log_nb_positive, log_bnb_positive
from modules import Encoder, Decoder, Discriminator, reparameterize_gaussian
from utils import one_hot
import math
import numpy as np
from sklearn.mixture import GaussianMixture

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

from utils import *
import scipy.sparse as sparse

torch.backends.cudnn.benchmark = True

adversarial_loss = torch.nn.BCELoss()
l1loss = torch.nn.L1Loss()
nllloss = torch.nn.NLLLoss()

_eps = 1e-15

# VAE model
class GAATAC(nn.Module):
    def __init__(
        self,
        args,
        n_input: int,
        index_sels = None,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 18,
        n_layers: int = 1,
        dropout_rate: float = 0.,
        dispersion: str = "gene",
        log_variational: bool = False,
        reconstruction_loss: str = "alpha-gan",
        n_centroids = 12,
        X = None,
        gan_loss = 'gan',
        reconst_ratio = 0.1,
        use_cuda: bool = True,
        RNA_dim: int = -1,
        selective_weight = 1
        #n_batch: int = 0
    ):
        super().__init__()
        self.args = args
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_centroids = n_centroids
        self.dropout_rate = dropout_rate
        self.X = X
        self.index_sels = index_sels
        self.gan_loss = gan_loss
        self.reconst_ratio = reconst_ratio
        self.use_cuda = use_cuda
        self.RNA_dim = RNA_dim
        self.selective_weight = selective_weight
        
        #self.n_batch = n_batch
        
        
        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":  # batch is different times of exp
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError("dispersion error")
            
        self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(n_latent, n_centroids)) # mu
        self.var_c = nn.Parameter(torch.ones(n_latent, n_centroids)) # sigma^2

        
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(n_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate)
        
        # discriminator to distinguish the generated/fake/real and batches
        if self.args.discriminator == 1:
            self.discriminator = Discriminator(n_input, n_hidden, self.n_batch, gan_loss)
        elif self.args.discriminator == 2 and self.args.RNA_dim > 0:
            self.discriminator = Discriminator(n_input-self.RNA_dim, n_hidden, self.n_batch, gan_loss)
            self.rna_discriminator = Discriminator(self.RNA_dim, n_hidden, self.n_batch, gan_loss)
        elif self.args.discriminator == 3:
            self.discriminator = Discriminator(75810, n_hidden, self.n_batch, gan_loss)
            self.rna_discriminator = Discriminator(59091, n_hidden, self.n_batch, gan_loss)
            self.chip_discriminator = Discriminator(15849, n_hidden, self.n_batch, gan_loss)
            
        
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = Decoder(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        
        

    def get_latents(self, x, y=None):
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def sample_from_posterior_l(self, x):        
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[
            "px_scale"
        ]

    def get_sample_rate(self, x, batch_index=None, y=None, n_samples=1):
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)["px_rate"]

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, px_alpha):
        # Reconstruction Loss
        # reconstruct torch.Size([128, 82727]) torch.Size([128, 82727]) torch.Size([82727]) torch.Size([128, 82727])
        # RNA 0-36600, ATAC: 36600-82727
        # print('reconstruct', x.shape, px_rate.shape, px_r.shape, px_dropout.shape)
        if self.reconstruction_loss == "alpha-gan" and self.RNA_dim > 0:
            #print('reconstruct', self.RNA_dim, x.shape, px_rate.shape, px_r.shape, px_dropout.shape)
            nb = -log_nb_positive(x[:, :self.RNA_dim], px_rate[:, :self.RNA_dim], px_r[:self.RNA_dim]) 
            zinb = -log_zinb_positive(x[:, self.RNA_dim:], px_rate[:, self.RNA_dim:], px_r[self.RNA_dim:], px_dropout[:, self.RNA_dim:])
            #print('nb', nb)
            #print('nb shape', nb.shape)
            #print('self.index_sels[0]', self.index_sels[0])
            #print('zinb', zinb)
            #print('zinb shape', zinb.shape)
            all_loss = torch.cat((nb,zinb),1)
            #print('all_loss', all_loss[0:1,108021:108022])
            #print('all_loss shape', all_loss.shape)
            #reconst_loss = all
            #self.index_sels[0] = self.index_sels[0] - 1
            #self.index_sels[0] = (np.array(self.index_sels[0])-1).tolist()
            #print('self.index_sels[0]',self.index_sels[0])
            #select_loss = all_loss[:,self.index_sels[0]] * self.selective_weight
            #print('select_loss shape', select_loss.shape)
            #print('self.index_sels[0]',self.index_sels[0])
            all_loss[:,self.index_sels[0]] = all_loss[:,self.index_sels[0]] * self.selective_weight
            
            reconst_loss=torch.sum(all_loss, dim=-1)

            #print('get reconstruction loss', zinb.mean(), nb.mean())
            #reconst_loss = zinb + nb
            #print('reconstruct', zinb, nb, reconst_loss)
            
        elif self.reconstruction_loss == "rna-adt" and self.RNA_dim > 0:
            #print('reconstruct loss', self.reconstruction_loss)
            # print('self.index_sels[0]',self.index_sels[0])
            nb1 = -log_nb_positive(x[:, :self.RNA_dim], px_rate[:, :self.RNA_dim], px_r[:self.RNA_dim])
            #print('zinb1 shape', zinb1.shape)
            nb2 = -log_nb_positive(x[:, self.RNA_dim:], px_rate[:, self.RNA_dim:], px_r[self.RNA_dim:])
            #print('zinb2 shape', zinb2.shape)
            all_loss = torch.cat((nb1,nb2),1)
            # print('all_loss shape', all_loss.shape)
            # select_loss = all_loss[:,self.index_sels[0]] * self.selective_weight
            # all_loss[:,self.index_sels[0]] = select_loss[:,:]
            all_loss[:, self.index_sels[0]] = all_loss[:, self.index_sels[0]] * self.selective_weight
                                       
            reconst_loss = torch.sum(all_loss, dim=-1)
        
        
        elif self.reconstruction_loss == "atac-adt" and self.RNA_dim > 0:
            zinb = -log_zinb_positive(x[:, :self.RNA_dim], px_rate[:, :self.RNA_dim], px_r[:self.RNA_dim], px_dropout[:, :self.RNA_dim])
            #print('zinb1 shape', zinb1.shape)
            nb = -log_nb_positive(x[:, self.RNA_dim:], px_rate[:, self.RNA_dim:], px_r[self.RNA_dim:])
            #print('zinb2 shape', zinb2.shape)
            all_loss = torch.cat((zinb,nb),1)
            # print('all_loss shape', all_loss.shape)
            # select_loss = all_loss[:,self.index_sels[0]] * self.selective_weight
            # all_loss[:,self.index_sels[0]] = select_loss[:,:]
            all_loss[:, self.index_sels[0]] = all_loss[:, self.index_sels[0]] * self.selective_weight
            
            reconst_loss = torch.sum(all_loss, dim=-1)
    
    
        elif self.reconstruction_loss == "zinb" or self.reconstruction_loss == "alpha-gan":
            #print('reconstruct loss', self.reconstruction_loss)
            #print('self.index_sels[0]',self.index_sels[0])
            all_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout)
            #print('all_loss shape', all_loss.shape)
            #select_loss = all_loss[:,self.index_sels[0]] * self.selective_weight
            #all_loss[:,self.index_sels[0]] = select_loss[:,:]
            all_loss[:,self.index_sels[0]] = all_loss[:,self.index_sels[0]] * self.selective_weight
            
            reconst_loss=torch.sum(all_loss, dim=-1)
        
        elif self.reconstruction_loss == "nb":
            all_loss = -log_nb_positive(x, px_rate, px_r)
            #select_loss = all_loss[:,self.index_sels[0]] * self.selective_weight
            #all_loss[:,self.index_sels[0]] = select_loss[:,:]
            #print('self.index_sels[0]',self.index_sels[0])
            all_loss[:,self.index_sels[0]] = all_loss[:,self.index_sels[0]] * self.selective_weight
            
            reconst_loss=torch.sum(all_loss, dim=-1)
        
        elif self.reconstruction_loss == "Module":
            all_loss = -log_nb_positive(x, px_rate, px_r)
            reconst_loss=torch.sum(all_loss, dim=-1)
        
        elif self.reconstruction_loss == "chip-chip" and self.RNA_dim > 0:
            #print('reconstruct loss', self.reconstruction_loss)
            # print('self.index_sels[0]',self.index_sels[0])
            zinb1 = -log_zinb_positive(x[:, :self.RNA_dim], px_rate[:, :self.RNA_dim], px_r[:self.RNA_dim],
                                     px_dropout[:, :self.RNA_dim])
            print('zinb1 shape', zinb1.shape)
            zinb2 = -log_zinb_positive(x[:, self.RNA_dim:], px_rate[:, self.RNA_dim:], px_r[self.RNA_dim:],
                                     px_dropout[:, self.RNA_dim:])
            print('zinb2 shape', zinb2.shape)
            all_loss = torch.cat((zinb1,zinb2),1)
            all_loss[:, self.index_sels[0]] = all_loss[:, self.index_sels[0]] * self.selective_weight
            
            reconst_loss = torch.sum(all_loss, dim=-1)
        
        elif self.reconstruction_loss == "chip":
            print('self.reconstruction_loss',self.reconstruction_loss)
            zinb1 = -log_zinb_positive(x[:, 0:59091], px_rate[:, 0:59091], px_r[0:59091], px_dropout[:, 0:59091])
            zinb2 = -log_zinb_positive(x[:, 59091:134901], px_rate[:, 59091:134901], px_r[59091:134901], px_dropout[:, 59091:134901])
            zinb3 = -log_zinb_positive(x[:, 134901:150750], px_rate[:, 134901:150750], px_r[134901:150750], px_dropout[:, 134901:150750])
            
            all_loss = torch.cat((zinb1,zinb2,zinb3),1)
            all_loss[:, self.index_sels[0]] = all_loss[:, self.index_sels[0]] * self.selective_weight
            
            reconst_loss = torch.sum(all_loss, dim=-1)
        
        return reconst_loss

    def scale_from_z(self, sample_batch, fixed_batch):
        if self.log_variational:
            sample_batch = torch.log(1 + sample_batch)
        qz_m, qz_v, z = self.z_encoder(sample_batch)
        batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
        library = 4.0 * torch.ones_like(sample_batch[:, [0]])
        px_scale, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)   

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            z = Normal(qz_m, qz_v.sqrt()).sample()
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = Normal(ql_m, ql_v.sqrt()).sample()

        px_scale, px_r, px_rate, px_dropout, px_alpha = self.decoder(
             self.dispersion, z, library, batch_index, y
        ) 
        
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            px_alpha = px_alpha,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    def get_reconstruction(self, x, batch_size=256):
        print(x.shape) # (7727, 134902)
        batch_num = (len(x) + batch_size - 1) // batch_size
        x_gen = np.zeros(x.shape)
        for i in range(batch_num):
            x_batch = x[i*batch_size:(i+1)*batch_size]
        
            outputs = self.inference(FLOAT(x_batch), None, None)
            px_rate = outputs["px_rate"]#/(torch.max(outputs["px_rate"])+1)
            px_r = outputs["px_r"]#/(torch.max(outputs["px_r"])+1)
            px_dropout = outputs["px_dropout"]
            px_alpha = outputs['px_alpha']

            #x_gen = px_rate
            x_gen[i*batch_size:(i+1)*batch_size] = px_rate.detach().cpu().numpy()
        #print('type(x_gen)',type(x_gen))
        #print('x_gen',x_gen)
        print('shape(x_gen)',x_gen.shape)
        return x_gen
        #row, col = np.nonzero(x_gen.detach().cpu().numpy())
        #values = x_gen.detach().cpu().numpy()[row, col]
        #print(type(row))
        #print(row[1:10])
        #print(col[1:10])
        #print(values[1:10])
        #print(len(row))
        #matrix=np.array(list(zip(row,col,values)))   #for 1 dim array   very slow
        #matrix=np.concatenate((row,col,values),axis=1)  #for 2 dim array
        #matrix=np.concatenate((row.reshape(len(row), 1),col.reshape(len(col), 1),values.reshape(len(values), 1)),axis=1)
        #print(matrix)
        #return matrix
        


    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        outputs = self.inference(x, None, y)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]    
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]
        px_alpha = outputs['px_alpha']
        z = outputs['z']
        

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
            
        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout, px_alpha)
        
        d_loss = 0
        g_loss = 0

        valid = Variable(FLOAT(x.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(FLOAT(x.size(0), 1).fill_(0.0), requires_grad=False)
        idx = torch.randperm(self.X.shape[0])
        x_real = FLOAT(self.X[idx[:x.size(0)]])
        batch_index = LONG(batch_index).reshape(-1)


        # generate sample from random priors
        z_prior = reparameterize_gaussian(torch.zeros(qz_m.shape).cuda(), torch.ones(qz_v.shape).cuda())
        l_prior = reparameterize_gaussian(torch.zeros(ql_m.shape).cuda(), torch.ones(ql_v.shape).cuda())
        _, _, x_fake, x_dropout, _ = self.decoder(self.dispersion, z_prior, l_prior, batch_index, y)
        _, _, z_rec = self.z_encoder(x_fake, y)
        x_fake = Normal(torch.zeros(x.shape), torch.ones(x.shape)).rsample().cuda()

        z_rec_loss = l1loss(z_rec, z_prior)
        
        
        if self.args.discriminator == 1:
            d_generated = self.discriminator(px_rate)
            d_fake = self.discriminator(x_fake)
            d_real = self.discriminator(x)
            # G loss: geneate undistinguishables: px_rate and x_fake
            g_loss = adversarial_loss(d_generated[0], valid) + adversarial_loss(d_fake[0], valid)
            # D loss: distinguish real (x)  and fakes (px_rate, x_fake)
            d_loss = adversarial_loss(d_real[0], valid) + adversarial_loss(d_generated[0].detach(), fake) + adversarial_loss(d_fake[0], fake)
        
        
        
        if self.args.discriminator == 2 and self.RNA_dim > 0:
            d_generated = self.discriminator(px_rate[:, self.RNA_dim:])
            d_fake = self.discriminator(x_fake[:, self.RNA_dim:])
            d_real = self.discriminator(x[:, self.RNA_dim:])
            # G loss: geneate undistinguishables: px_rate and x_fake
            g_loss_ATAC = adversarial_loss(d_generated[0], valid) + adversarial_loss(d_fake[0], valid)
            # D loss: distinguish real (x)  and fakes (px_rate, x_fake)
            d_loss_ATAC = adversarial_loss(d_real[0], valid) + adversarial_loss(d_generated[0].detach(), fake) + adversarial_loss(d_fake[0], fake)
            
            
            #x_real_RNA = FLOAT(self.X[idx[:X_batch.size(0)]][:, :self.args.RNA_dim])
            
            d_generated = self.rna_discriminator(px_rate[:, :self.RNA_dim])
            d_fake = self.rna_discriminator(x_fake[:, :self.RNA_dim])
            d_real = self.rna_discriminator(x[:, :self.RNA_dim])
            
            # G loss: geneate undistinguishables: px_rate and x_fake
            g_loss_RNA = adversarial_loss(d_generated[0], valid) + adversarial_loss(d_fake[0], valid)
            # D loss: distinguish real (x)  and fakes (px_rate, x_fake)
            d_loss_RNA = adversarial_loss(d_real[0], valid) + adversarial_loss(d_generated[0].detach(), fake) + adversarial_loss(d_fake[0], fake)
            
            g_loss = g_loss_ATAC + g_loss_RNA
            d_loss = d_loss_ATAC + d_loss_RNA
        
        if self.args.discriminator == 3 and self.RNA_dim > 0:
            d_generated = self.discriminator(px_rate[:, 59091:134901])
            d_fake = self.discriminator(x_fake[:, 59091:134901])
            d_real = self.discriminator(x[:, 59091:134901])
            g_loss_ATAC = adversarial_loss(d_generated[0], valid) + adversarial_loss(d_fake[0], valid)
            d_loss_ATAC = adversarial_loss(d_real[0], valid) + adversarial_loss(d_generated[0].detach(), fake) + adversarial_loss(d_fake[0], fake)
            
            
            
            d_generated = self.rna_discriminator(px_rate[:, 0:59091])
            d_fake = self.rna_discriminator(x_fake[:, 0:59091])
            d_real = self.rna_discriminator(x[:, 0:59091])
            g_loss_RNA = adversarial_loss(d_generated[0], valid) + adversarial_loss(d_fake[0], valid)
            d_loss_RNA = adversarial_loss(d_real[0], valid) + adversarial_loss(d_generated[0].detach(), fake) + adversarial_loss(d_fake[0], fake)
            
            
            d_generated = self.chip_discriminator(px_rate[:, 134901:150750])
            d_fake = self.chip_discriminator(x_fake[:, 134901:150750])
            d_real = self.chip_discriminator(x[:, 134901:150750])
            g_loss_chip = adversarial_loss(d_generated[0], valid) + adversarial_loss(d_fake[0], valid)
            d_loss_chip = adversarial_loss(d_real[0], valid) + adversarial_loss(d_generated[0].detach(), fake) + adversarial_loss(d_fake[0], fake)
            
            g_loss = g_loss_ATAC + g_loss_RNA + g_loss_chip
            d_loss = d_loss_ATAC + d_loss_RNA + d_loss_chip
    

        # Batch loss:  
        #print('batch', self.n_batch, batch_index.shape, batch_index[:10])
        if self.n_batch > 0:
            celoss = torch.nn.CrossEntropyLoss(ignore_index=self.n_batch)
            
            d_batch_loss = celoss(d_real[1], batch_index) + celoss(d_generated[1], batch_index)
            g_batch_loss = - celoss(d_generated[1], batch_index) - celoss(d_fake[1], batch_index)
            #print('batch loss', d_batch_loss, g_batch_loss)
        else:
            g_batch_loss, d_batch_loss = None, None


        
        return self.reconst_ratio*reconst_loss, kl_divergence_l+kl_divergence_z, g_loss, d_loss, z_rec_loss, g_batch_loss, d_batch_loss
        
