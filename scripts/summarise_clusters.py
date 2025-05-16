import pandas as pd

def summarise_clusters(df, cluster_labels):
    df['Cluster'] = cluster_labels
    summary = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Gender': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'Location': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'InternetSpending': 'mean',
        'Education': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'Satisfaction': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'Job_Search': lambda x: x.mode()[0] if not x.empty else 'Unknown',
    })
    return summary
