import collections
from typing import Iterable, List

import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList
import torch.nn.functional as F

from utils import *


    
def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


# Encoder
class Encoder(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.fc = nn.Linear(n_input, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001)
        self.relu = nn.ReLU()
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int, global_step=0):
        
        # Parameters for latent distribution
        q = self.fc(x)
        q1 = self.bn1(q)
        q1 = self.relu(q1)
        #print('encoder', x.shape, x.mean(), 'q', q.shape, q.mean())
        q_m = self.mean_encoder(q1)
        q_v = torch.exp(self.var_encoder(q1)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        
        if global_step == 133:
            print('x', x[6].detach().cpu().numpy(), x[6].sum())
            print('z', global_step, q.shape, q.dtype, q[6][0].detach().cpu().numpy()) 
        return q_m, q_v, latent


class Decoder(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.px_decoder = nn.Sequential(
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001),
            nn.ReLU()
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), 
            nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Sequential(
                    nn.Linear(n_hidden, n_output), 
                    nn.ReLU()
        )

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)
        
        # alpha for Beta
        self.px_alpha_decoder = nn.Sequential(
                    nn.Linear(n_hidden, n_output), 
                    nn.ReLU()
        )

    def forward(self, z, library, global_step=0):  
        # The decoder returns values for the parameters of the ZINB distribution
        x1 = self.fc1(z)
        px = self.px_decoder(x1)   # cat list includes batch index [128, 512]
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        
        
        
        px_alpha = self.px_alpha_decoder(px)+1
        library = torch.clamp(library, max=12)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  
        px_rate = torch.clamp(px_rate, max=12)
        
        return px_scale, None, px_rate, px_dropout, px_alpha
    


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_size)
        )
        self.final_model = nn.Sequential(
                nn.BatchNorm1d(512, momentum=0.01, eps=0.001),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256, momentum=0.01, eps=0.001),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Dropout(0.4),
                nn.Linear(256, 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        mid_out = self.model(x)
        final_out = self.final_model(mid_out)

        return mid_out, final_out