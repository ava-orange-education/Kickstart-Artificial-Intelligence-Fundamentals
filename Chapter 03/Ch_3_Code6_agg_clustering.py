
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Assuming 'scaled_features' and 'scaled_data_with_clusters' are derived from a dataset similar to previous examples
df = pd.read_csv('path_to_your_file/data5_Ch3.csv')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('LocationID', axis=1))

# Create a DataFrame for scaled data with cluster for visualization
scaled_data_with_clusters = pd.DataFrame(scaled_features, columns=df.columns[1:])
scaled_data_with_clusters['LocationID'] = df['LocationID']

# Perform Agglomerative Clustering with the 'ward' linkage method
# 'ward' minimizes the variance of clusters being merged.
agg_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
clusters_agg = agg_clustering.fit_predict(scaled_features)

# Add the cluster labels to the original data
scaled_data_with_clusters['AggCluster'] = clusters_agg

# Plotting the dendrogram
linked = linkage(scaled_features, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Scatter plot of Temperature vs. Rainfall colored by Agglomerative Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=scaled_data_with_clusters.columns[0], y=scaled_data_with_clusters.columns[1], hue='AggCluster', data=scaled_data_with_clusters, palette='viridis', alpha=0.6, s=100, edgecolor='k')
plt.title('Temperature vs. Rainfall - Clustered (Agglomerative Clustering)')
plt.xlabel('Temperature (°C) - Scaled')
plt.ylabel('Rainfall (mm) - Scaled')
plt.legend(title='Cluster')
plt.show()
