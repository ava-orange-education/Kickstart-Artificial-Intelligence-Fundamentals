
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('path_to_your_file/data5_Ch3.csv')

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('LocationID', axis=1))

# Elbow method to find the optimal number of clusters
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plotting the elbow plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Choose the number of clusters (e.g., 3) based on the elbow plot
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Adding cluster information to the DataFrame
df['Cluster'] = clusters

# Plotting the clusters
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red']
for i in range(3):
    plt.scatter(df[df['Cluster'] == i]['Temperature (°C)'], df[df['Cluster'] == i]['Rainfall (mm)'], label=f'Cluster {i}',  c=colors[i])
plt.title('Environmental Conditions Clustering')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()
