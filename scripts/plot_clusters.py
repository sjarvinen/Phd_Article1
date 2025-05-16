import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_cluster_distribution(df, cluster_col='Cluster'):
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=cluster_col)
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
