##calculate in python3.6
import pandas as pd
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import adjusted_rand_score as ARI
import sys

dataname = sys.argv[1]

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




df = pd.read_csv(dataname + "/cluster_info_for_accuracy.csv", delimiter=',')
t_label=[]
t_label=df.tlabelId
pre_label=[]
pre_label=df.predictedLabel

ari_score = ARI(t_label, pre_label)
acc_score = unsupervised_clustering_accuracy(t_label, pre_label)
ari_score
acc_score

