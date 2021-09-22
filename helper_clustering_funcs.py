from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
import numpy as np

def construct_linkage(data,method,metric):
    Z1 = linkage(data, method=method, metric=metric)
    return Z1

def measure_linkage(Z1,metric,data):
    c, coph_dists = cophenet(Z1, pdist(data,metric))
    return c, coph_dists

def choose_metric_method(data):
    results = {}
    metrics = ['euclidean','cityblock','cosine','hamming']
    methods = ['single','complete','average']
    for metric in metrics:
        for method in methods:
            Z = construct_linkage(data,method,metric)
            c, coph_dists = measure_linkage(Z,metric,data)
            key = metric + ' '+method
            print(metric , method)
            results[key] = c
    return results

def plot_hierarchical_clusters(data,method,metric):
    Z1 = construct_linkage(data,method,metric)
    fig = plt.figure(figsize=(14,6))
    ax1 = plt.subplot(1,2,1)    
    dendrogram(Z1)
    ax1.set_xticks([])
    plt.show()
    return Z1

def check_clusters(df,threshold,Z):
    max_d = threshold
    clusters = fcluster(Z, max_d, criterion='distance')
    # clusters
    print('Num clusters',np.unique(clusters))
    df['heirarchial_labels'] = clusters
    print(df['heirarchial_labels'].value_counts())