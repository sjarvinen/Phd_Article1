import pandas as pd
from sklearn.decomposition import PCA
import gower
import hdbscan

def run_clustering(df, n_components=5, min_cluster_size=30):
    # PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(df)
    
    # Gower distance on original df
    gower_dist = gower.gower_matrix(df)
    
    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(gower_dist)
    
    return labels
