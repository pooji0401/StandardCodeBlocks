from scipy.cluster.hierarchy import dendrogram
from helper_clustering_funcs import plot_hierarchical_clusters
from helper_clustering_funcs import check_clusters,choose_metric_method
## gives cophonetic correlation how faithfully a dendrogram preserves the pairwise distances between the original unmodeled 
# data points
from helper_clustering_funcs import measure_linkage

# https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/
# https://en.wikipedia.org/wiki/Dunn_index

def check_dunn_index(df,threshold,Z,method='complete',distance_metric='euclidean'):
    check_clusters(df,threshold,Z)
    
    if method == 'complete':
    # Method 1. - Complete
    # Intra-cluster distance D(a) - Complete Diameter Linkage distance
    # Intercluster distance d(a,b) -  Complete linkage distance: Distance between two most remote objects belonging to a and b respectively

    ### Intracluster distances - Diameter - Distance between farthest points within each cluster OR MAX distance between the points within each cluster
    ### InterCluster Distance - Distance between the farthest points of 2 clusters OR MAX distance between the points from different clusters
    
        #### Intracluster distances
        cluster_diameters = {}
        for i in df['heirarchial_labels'].unique():
            temp_cluster_df = df[df['heirarchial_labels']==i]
            X = temp_cluster_df.drop('heirarchial_labels',axis=1)
            if len(X)>1:
                ## when there are atleast 2 points in the cluster - only then intra cluster distance will make sense
                diameters = pdist(X,metric= distance_metric)
                cluster_diameters[i] = max(diameters) ## farthest points
            else:
                cluster_diameters[i] = 0
        print('cluster_diameters ',cluster_diameters)
        max_cluster_diameter = max(list(cluster_diameters.values())) ## cluster with maximum cluster diameter

        #### Intercluster distances
        inter_cluster_distances = {}
        for i in range(len(df['heirarchial_labels'].unique())):
            for j in range(i+1,len(df['heirarchial_labels'].unique())):
                cluster_label_1 = df['heirarchial_labels'].unique()[i]
                cluster_label_2 = df['heirarchial_labels'].unique()[j]
                temp_cluster_df_1 = df[df['heirarchial_labels']==cluster_label_1]
                temp_cluster_df_2 = df[df['heirarchial_labels']==cluster_label_2]

                temp_cluster_df_1 = temp_cluster_df_1.drop('heirarchial_labels',axis=1)
                temp_cluster_df_2 = temp_cluster_df_2.drop('heirarchial_labels',axis=1)

                C = cdist(temp_cluster_df_1, temp_cluster_df_2, distance_metric)
                inter_cluster_distances[str(i)+' '+str(j)] = C.max()
        print('inter_cluster_distances ',inter_cluster_distances )
        min_inter_cluster_distance = min(inter_cluster_distances.values())  ## minimum inter cluster distance 

        dun_index = min_inter_cluster_distance/max_cluster_diameter
        print('DUN INDEX ', dun_index)
        
    
    elif method == 'average':
    # Method 2 - Average
    # Intracluster distance - Average diameter linkage distance
    # Intercluster distance - Centroid linkage distance

    ### Intracluster distance - diameter - mean distance between all pair of points within a cluster - max
    ### Intercluster distance - Distance between the centroids of pair of clusters - min
        
        #### Intracluster distances
        cluster_diameters = {}
        for i in df['heirarchial_labels'].unique():
            temp_cluster_df = df[df['heirarchial_labels']==i]
            X = temp_cluster_df.drop('heirarchial_labels',axis=1)
            cluster_diameters[i] = pdist(X,distance_metric).mean() ## farthest points
        print('cluster_diameters ',cluster_diameters)
        max_cluster_diameter = max(list(cluster_diameters.values())) ## cluster with maximum cluster diameter

        #### Intercluster distances 
            ## Calculating cluster centers
        mean_X = df.groupby('heirarchial_labels').mean()
        intercluster_distances_btwn_means = pdist(mean_X,distance_metric)
        print('inter_cluster_distances ',intercluster_distances_btwn_means)
        min_inter_cluster_distance = min(intercluster_distances_btwn_means)

        dun_index = min_inter_cluster_distance/max_cluster_diameter
        print('DUN INDEX ', dun_index)
        
    
    else : # centroid method
    # Method 3- Centroid
    # Intracluster distance - Centroid diameter linkage distance
    # Intercluster distance - Centroid linkage distance

    ### Intracluster distance - diameter - mean distance between centre and all other points within the cluster - max
    ### Intercluster distance - Distance between the centroids of pair of clusters - min

        #### Intracluster distances 
        mean_X = df_raw_basic_scores.groupby('heirarchial_labels').mean()
        cluster_diameters ={}
        for i in mean_X.index:
            current_mean = mean_X.loc[i]
            temp_cluster_df = df_raw_basic_scores[df_raw_basic_scores['heirarchial_labels']==i]
            C = cdist(np.array([current_mean]),temp_cluster_df.drop('heirarchial_labels',axis=1) , distance_metric)
            cluster_diameters[i] = C.mean() ## avg distance from mean to points

        print(cluster_diameters)
        max_cluster_diameter = max(list(cluster_diameters.values())) ## cluster with maximum cluster diameter
        max_cluster_diameter = 2*max_cluster_diameter

        #### Intercluster distances 
            ## Calculating cluster centers
        mean_X = df_raw_basic_scores.groupby('heirarchial_labels').mean()
        intercluster_distances_btwn_means = pdist(mean_X,distance_metric)
        print('inter_cluster_distances ',intercluster_distances_btwn_means)
        min_inter_cluster_distance = min(intercluster_distances_btwn_means)

        dun_index = min_inter_cluster_distance/max_cluster_diameter
        print('DUN INDEX ', dun_index) 
        
    plt.figure(figsize=(14,8))
    dendrogram(Z,p=100,truncate_mode='lastp')
    plt.axhline(y=threshold, c='k')
    plt.show()
    return dun_index

    

