
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

def filter_dataset(data, barcode, low=0, high=0.9, min_peaks=0):
    peaks = np.array([i for i in range(data.shape[1])])
    print('peaks length',len(peaks))
    barcode = np.array([i for i in range(len(barcode))])
        
    total_cells = data.shape[0]
    count = np.array((data >0).sum(0)).squeeze()
    #print('data[,100000]',data[,100000])
    #print('count[100000]',count[100000])
    peak_indices = np.where((count > low*total_cells) & (count < high*total_cells))[0]
    print('peak_indices length',len(peak_indices))
    data = data[:, peak_indices]
    set_A = set(peak_indices)
    new_list = [item for item in peaks if item not in set_A]
    print('new_list',new_list)
    print('new_list',count[new_list])
    peaks = peaks[peak_indices]
    

    if min_peaks > 0:
        if min_peaks < 1:
            min_peaks = len(peaks)*min_peaks
        cell_indices = np.where(np.sum(data>0, 1)>=min_peaks)[0]
        data = data[cell_indices]
        barcode = barcode[cell_indices]
        return data, peaks, barcode, cell_indices, peak_indices
    else:
        return data, peaks, barcode, None, peak_indices


def read_barcodes(filename='/GSE99172/GSE99172_barcode.txt'):
    cells, tlabels, y, cell_types = [], [], [], []
    
    with open(os.path.dirname(__file__) +filename) as f:
        for line in f.readlines():
            cell = line.strip('\n').replace('"', '').split(',')[0]
            cells.append(cell)
    return cells


def read_peaks(filename='/GSE99172/peak.bed'):
    peaks = []
    i = 0
    with open(os.path.dirname(__file__) +filename) as f:
        for line in f.readlines():
            values = line.strip('\n').split('\t')
            peaks.append(values)

    return peaks
    
    
def read_labels(filename, cells, dataset, X, n_batch):
    df = pd.read_csv(os.path.dirname(__file__) +filename, delimiter=',')
    df = df[df.barcode.isin(cells)]
    df['cell_types'] = df['celltype'].apply(lambda x: x)
   
    
    
    cell_types = list(df.cell_types.value_counts().keys())
    y = []
    for i, cell in enumerate(cells):
        data = df[df.barcode==cell]
        if not pd.isna(data.cell_types.values[0]):
            y.append(cell_types.index(data.cell_types.values[0]))
        
    tlabels = df.cell_types.values
    batches = np.zeros((X.shape[0], n_batch))
    batch_ids = np.ones(X.shape[0])*n_batch
    if X is not None:        
        
        if 'panc' in filename:
            for i, row in df.iterrows():
                if row['batch'] == 'celseq':
                    batches[i][0] = 1
                    batch_ids[i] = 0
                elif row['batch'] == 'celseq2':
                    batches[i][1] = 1
                    batch_ids[i] = 1
                elif row['batch'] == 'smartseq2':
                    batches[i][2] = 1
                    batch_ids[i] = 2
        print('batches', batches.shape, batch_ids.shape, batches[-10:], batch_ids[-10:])
    
    return y, cell_types, tlabels, batch_ids, batches
    
    
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
    with open(os.path.dirname(__file__)+filename) as f:
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
    print('load sparse matrix', len(data), len(row), len(col))
    #print('max row',max(row))
    #print('max col',max(col))
    X = coo_matrix((data, (row, col))).toarray()
    #print('X dim',X.shape)
    del data
    del row
    del col
    return X

def read_batches(filename):
    df = pd.read_csv(os.path.dirname(__file__)+filename, delimiter=',')
    df['batch'] = df['batch'].apply(lambda x: int(x[-1])-1)
    return df.batch.values
    
    


def load_dataset(args):
    dirname = '/../data/%s/'%(args.dataset) 
    
    
    cells = read_barcodes(dirname+'%s_barcode.txt'%(args.dataset))
    if args.dataset == 'pbmc_two_batch':
        X = load_data(dirname+'%s_ATAC_matrix.txt'%(args.dataset), len(cells))
        X1 = load_data(dirname+'%s_RNA_matrix.txt'%(args.dataset), len(cells))
        X1 = X1/7545.0
        X = np.concatenate((X, X1), axis=1)
    elif args.dataset == 'pbmc10k_selectMatrix_all':
        X = load_data(dirname+'%s_SparseMatrix.txt'%(args.dataset), len(cells))
        #print(X.shape)
        #print(X[:, :36600].max(), X[:, :36600].min()) # 929.0 0.0
        #print(X[:, 36600:].max(), X[:, 36600:].min()) # 2.0 0.0
        #return None
        if args.RNA_dim > 0:
            X[:, :36600] = np.log1p(X[:, :36600])
            X[:, 36600:] = np.where(X[:, 36600:]>0, 1, 0)
    else:
        X = load_data(dirname+'%s_SparseMatrix.txt'%(args.dataset), len(cells))
    #print('X dim',X.shape)
        #if args.dataset == 'pbmc3k':
        #X[:, :36572] = X[:, :36572]/7545.0
        #X[:, 36572:] = np.where(X[:, 36572:]>0, 1, 0)
    
    try:
        peaks = read_pos(dirname+'%s_peak.bed'%(args.dataset))
    except FileNotFoundError:
        peaks = range(X.shape[1])
    
    
        
    if args.labeled:
        labels, cell_types, tlabels, batchids, batches = read_labels(dirname+'%s_celltype_info.csv'%(args.dataset), cells, args.dataset, X, args.n_batch)
    else:
        labels, cell_types, tlabels, batchids, batches = None, None, None, None, None
#print('X dim',X.shape)
    return X, cells, peaks, labels, cell_types, tlabels, batchids, batches



if __name__ == "__main__":
    #extract()
    #mat, cells, peaks, labels, cell_types, tlabels = extract_data()
    #load_data('simulated_data/GSE65360/simulated.true.nodups.sort.bam')
    
    #extract_simulated(dataset='For_specific_peak', suffix='')
    
    extract_simulated(dataset='pbmc_two_batch', suffix='', is_labeled=False, batch=True)
