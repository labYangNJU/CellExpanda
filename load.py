
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix
import gc
import os, os.path
import pysam

import time
import scipy

import torch
from torch.utils.data import Dataset
from typing import Dict, Iterable, List, Tuple, Union, Optional, Callable
import scipy.sparse as sp_sparse







def filter_dataset(dataset, low=0, high=0.9, min_peaks=0):
    peaks = np.array([i for i in range(dataset.X.shape[1])])
    print('peaks length',len(peaks))
    #dataset.cells = np.array([i for i in range(len(dataset.cells))])
        
    total_cells = dataset.X.shape[0]
    count = np.array((dataset.X >0).sum(0)).squeeze()
    #print('data[,100000]',data[,100000])
    #print('count[100000]',count[100000])
    peak_indices = np.where((count > low*total_cells) & (count < high*total_cells))[0]
    dataset.X = dataset.X[:, peak_indices]
    set_A = set(peak_indices)
    new_list = [item for item in peaks if item not in set_A]

    
    dataset.peaks = peaks[peak_indices]
    

    if min_peaks > 0:
        if min_peaks < 1:
            min_peaks = len(dataset.peaks)*min_peaks
        cell_indices = np.where(np.sum(data>0, 1)>=min_peaks)[0]
        dataset.X = dataset.X[cell_indices]
        dataset.cells = dataset.cells[cell_indices]
        
        
        dataset.labels = [dataset.labels[i] for i in dataset.cells if labels is not None]
        dataset.tlabels = [dataset.tlabels[i] for i in dataset.cells if tlabels is not None]
        #cells = [cells[i] for i in data['cells'] if cells is not None]
        dataset.batches = np.array([dataset.batches[i] for i in dataset.cells if batches is not None])  # [0, 0, 1], [0, 1, 0], [1, 0, 0]
        dataset.batch_ids = np.array([dataset.batch_ids[i] for i in dataset.cells if batch_ids is not None])  # 0, 1, 2,
    
        return dataset, cell_indices, peak_indices
    else:
        return dataset, None, peak_indices


def read_barcodes(filename='/GSE99172/GSE99172_barcode.txt'):
    cells, tlabels, y, cell_types = [], [], [], []
    
    with open(filename) as f:
        for line in f.readlines():
            cell = line.strip('\n').replace('"', '').split(',')[0]
            cells.append(cell)
    return cells


def read_peaks(filename='/GSE99172/peak.bed'):
    peaks = []
    i = 0
    with open(filename) as f:
        for line in f.readlines():
            values = line.strip('\n').split('\t')
            peaks.append(values)

    return peaks
    
    
def read_labels(filename, config, data, dataset):
    df = pd.read_csv(filename, delimiter=',')
    df = df[df.barcode.isin(dataset.cells)]
    df['cell_types'] = df['celltype'].apply(lambda x: x)
   
    
    
    data['cell_types'] = list(df.cell_types.value_counts().keys())
    data['labels'] = []
    for i, cell in enumerate(dataset.cells):
        cell_data = df[df.barcode==cell]
        if not pd.isna(cell_data.cell_types.values[0]):
            data['labels'].append(data['cell_types'].index(cell_data.cell_types.values[0]))
        
    data['tlabels'] = df.cell_types.values
    data['batches'] = np.zeros((dataset.X.shape[0], config['data_params']['n_batch']))
    data['batch_ids'] = np.ones(dataset.X.shape[0])*config['data_params']['n_batch']

    return data
    
    
def read_pos(filename):
    peaks = []
    
    with open(os.path.dirname(__file__) +filename) as f:
        chr_lines = f.readlines()
        for line in chr_lines:
            
            sp = ':' if 'txt' in filename else '\t'
            values = line.strip().split(sp)
            if len(values) > 3:
                key, start, end = '_'.join(values[:-2]), int(values[-2]), int(values[-1])
            elif len(values) == 2:
                offsets = values[1].split('-')
                key, start, end = values[0], int(offsets[0]), int(offsets[1])
            else:
                key, start, end = values[0], int(values[1]), int(values[2])
            
            peaks.append((key, start, end))
    return peaks

def load_data(filename, cell_num):
    #if '65361' in filename:
    #    return np.load(os.path.dirname(__file__)+filename)
    row, col, data = [], [], []
    #print(len(data), len(row), len(col))
    with open(filename) as f:
        for line in f.readlines():
            values = line.strip('\n').split()
            if len(values) != 3:
                continue
            try:
                if int(values[1])-1 < cell_num:
                    #print(values, len(data), len(row), len(col))
                    col.append(int(values[0]))
                    row.append(int(values[1])-1)
                    data.append(float(values[2]))
                    #print(values, len(data), len(row), len(col))
                    #break
            except ValueError:
                #print(line)
                pass
    X = coo_matrix((data, (row, col))).toarray()
    #print('X dim',X.shape)
    del data
    del row
    del col
    return np.float32(X)

def read_batches(filename):
    df = pd.read_csv(os.path.dirname(__file__)+filename, delimiter=',')
    df['batch'] = df['batch'].apply(lambda x: int(x[-1])-1)
    return df.batch.values
    
    
def compute_library_size(data: Union[sp_sparse.csr_matrix, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    sum_counts = data.sum(axis=1)
    masked_log_sum = np.ma.log(sum_counts)
    if np.ma.is_masked(masked_log_sum):
        logger.warning(
            "This dataset has some empty cells, this might fail scVI inference."
            "Data should be filtered with `my_dataset.filter_cells_by_count()"
        )
    log_counts = masked_log_sum.filled(0)
    local_mean = (np.mean(log_counts).reshape(-1, 1)).astype(np.float32)
    local_var = (np.var(log_counts).reshape(-1, 1)).astype(np.float32)
    # compute library (2714, 89119) [9.104647  9.661671  9.4187355 ... 9.034677  9.060563  9.492658 ] [ 8997. 15704. 12317. ...  8389.  8609. 13262.] (1, 1) [[9.163114]] (1, 1) [[0.19555624]]
    print('compute library', data.shape, log_counts, sum_counts, local_mean.shape, local_mean, local_var.shape, local_var)
    return local_mean, local_var
    
class CellDataset(Dataset):
    def __init__(self, X, cells=None, peaks=None, cell_types=None, labels=None, tlabels=None, batches=None, batch_ids=None):
        self.X = X
        self.local_l_mean, self.local_l_var = compute_library_size(X)
        self.cells = cells
        self.peaks = peaks
        self.cell_types = cell_types
        self.labels = labels
        self.tlabels = tlabels
        self.batches = batches
        self.batch_ids = batch_ids
        #print('cells', self.cells)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """Implements @abstractcmethod in ``torch.utils.data.dataset.Dataset`` ."""
        return idx
        
        

    


def load_dataset(config):
    data = {}
    
    #print(config['data_params'])
    data['cells'] = read_barcodes(os.path.join(config['data_params']['data_path'], '%s_barcode.txt'%(config['data_params']['name'])))
    
    #print('cells', data['cells'])
    dataset = CellDataset(load_data(os.path.join(config['data_params']['data_path'], '%s_SparseMatrix.txt'%(config['data_params']['name'])), len(data['cells'])),
                     data['cells'])
    try:
        dataset.peaks = read_pos(os.path.join(config['data_params']['data_path'], '%s_peak.bed'%(config['data_params']['name'])))
    except FileNotFoundError:
        dataset.peaks = range(dataset.X.shape[1])
    
    
        
    if config['data_params']['labeled']:
        data = read_labels(os.path.join(config['data_params']['data_path'], '%s_celltype_info.csv'%(config['data_params']['name'])), config, data, dataset)
        dataset.labels = data['labels']
        dataset.cell_types = data['cell_types']
        dataset.tlabels = data['tlabels']
        dataset.batch_ids = data['batch_ids']
        dataset.batches = data['batches']
    else:
        dataset.labels = None
        dataset.cell_types = None
        dataset.tlabels = None
        dataset.batch_ids = None
        dataset.batches = None
        
    for key in data:
        print(key, len(data[key]))
        
    return dataset



if __name__ == "__main__":
    #extract()
    #mat, cells, peaks, labels, cell_types, tlabels = extract_data()
    #load_data('simulated_data/GSE65360/simulated.true.nodups.sort.bam')
    
    #extract_simulated(dataset='For_specific_peak', suffix='')
    
    extract_simulated(dataset='pbmc_two_batch', suffix='', is_labeled=False, batch=True)
