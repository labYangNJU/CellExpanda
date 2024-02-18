
import os, random
import yaml, argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#from trainer import UnsupervisedTrainer
import torch

from tensorboardX import SummaryWriter


from model import *
#from utils import *
from load import *
from train import train

if os.path.exists('models/'):
    message = 'OK'
else:
    os.makedirs("models")
save_path = 'models/'

plt.switch_backend("agg")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def main(config):
    device = torch.device(config['exp_params']['device'])
    set_seed(config['exp_params']['seed'])
    use_batches = True if config['data_params']['n_batch']>0 else False
    
    dataset = load_dataset(config)
    
    dataset, _, _ = filter_dataset(dataset, low=config['data_params']['min_cells'], high=config['data_params']['max_cells'], min_peaks=config['data_params']['min_peaks'])
    
    dims = config['data_params']['dims']
  
    for i, dim in enumerate(dims):
        if config['data_params']['normalize'][i]: 
            dataset.X[:, dim[0]:dim[1]] = (dataset.X[:, dim[0]:dim[1]] -  dataset.X[:, dim[0]:dim[1]].mean())/dataset.X[:, dim[0]:dim[1]].std()
        if config['data_params']['log_variational'][i]: 
            dataset.X[:, dim[0]:dim[1]] = np.log(dataset.X[:, dim[0]:dim[1]]+1)
        if config['data_params']['binary'][i]:
            dataset.X[:, dim[0]:dim[1]] = np.where(dataset.X[:, dim[0]:dim[1]]>0, 1, 0) 
            
        print(i, dim, dataset.X[:, dim[0]:dim[1]].shape, dataset.X[:, dim[0]:dim[1]].max(), dataset.X[:, dim[0]:dim[1]].min(), dataset.X[:, dim[0]:dim[1]].mean())

        
    print(dataset.X.shape, dataset.X.mean())
    model = GAMM(dataset, config, device).to(device)
    train(config, model, dataset, device)
    
    params = '%s-%d-%d'%(config['data_params']['name'], config['model_params']['z_encoder']['n_hidden'], config['clustering_params']['n_louvain'])
    model.load_state_dict(torch.load('model/%s-best.pth'%(params)))
    
    latent = model.get_latent(dataset.X).detach().cpu().numpy()
    
    labels_pred = clustering(config, latent)

    params = '%s-%d-%d'%(config['data_params']['name'], config['model_params']['z_encoder']['n_hidden'], config['clustering_params']['n_louvain'])
    

    if os.path.exists("result/%s"%(config['data_params']['name'])):
        message = 'OK'
    else:
        os.makedirs("result/%s"%(config['data_params']['name']))
    
    np.savetxt('result/%s/latent-%s.txt'%(config['data_params']['name'], params), latent)
    
    plot_tsne(config, dataset, latent, dataset.labels, labels_pred)
    
    if config['data_params']['labeled']:
        clustering_score(dataset.labels, labels_pred)
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAAE: Generative Adversarial ATAC-RNA-seq Analysis')
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()
    
    with open(os.path.join('configs', '%s.yaml'%(args.dataset)), 'r') as file:
        config = yaml.safe_load(file)
    
    print(config)
    main(config)
    
