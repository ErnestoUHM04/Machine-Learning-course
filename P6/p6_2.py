# Práctica 6: Reducción de dimensionalidad y agrupación
# Hernández Martínez Ernesto Ulises

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# En cada practica se tiene que:                 IMPORTANTE
# 1. Mostrar la gráfica del dataset sin trabajar o procesar.
# 2. Mostrar las métricas: F1-score, accuracy, precision y recall. Y su matriz de confusión. (con gráficas)
# 3. Mostrar la gráfica del dataset ya trabajado o procesado.

# -------------- PARTE 2: Agrupación --------------
# Se usarán los datasets:
# - Iris
from sklearn.datasets import load_iris
iris = load_iris()
# - Mall Customers
mall_customers = pd.read_csv('Mall_Customers.csv')
# Encode species as numbers
le = LabelEncoder()
mall_customers['Gender'] = le.fit_transform(mall_customers['Gender'])
#print("Mall Customers dataset: \n", mall_customers.head(), "\n")
# - Make Blobs
from sklearn.datasets import make_blobs
blobs = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)
#print("Blobs dataset: \n", blobs[0][:5], "\n")
# - Make Moons
from sklearn.datasets import make_moons
moons = make_moons(n_samples=300, noise=0.05, random_state=0)
#print("Moons dataset: \n", moons[0][:5], "\n")

# Se usarán los siguientes métodos de agrupación:
# - KMeans
from sklearn.cluster import KMeans
# - Spectral Clustering
from sklearn.cluster import SpectralClustering
# - DBSCAN
from sklearn.cluster import DBSCAN
# - Birch Clustering
from sklearn.cluster import Birch
# - Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

# Herramienta para normalizar los datos:
# - StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

###########################################################################
# Lets fucking start
###########################################################################

# IRIS
iris_scaled = scaler.fit_transform(iris.data)
#print("Iris dataset: \n", iris.data[:5], "\n")
plt.figure()
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
plt.title("Iris dataset")
plt.show()

# Iris with Kmeans
kmeans_iris = KMeans(n_clusters=3)
kmeans_iris.fit(iris_scaled)
labels_kmeans_iris = kmeans_iris.labels_

plt.figure()
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels_kmeans_iris)
plt.title("Iris dataset with KMeans")
plt.show()

# Iris with Spectral Clustering
spectral_iris = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', assign_labels='kmeans', random_state=0)
spectral_iris.fit(iris_scaled)
labels_spectral_iris = spectral_iris.labels_

plt.figure()
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels_spectral_iris)
plt.title("Iris dataset with Spectral Clustering")
plt.show()

# Iris with DBSCAN
dbscan_iris = DBSCAN(eps = 0.5, min_samples = 5)
dbscan_iris.fit(iris_scaled)
labels_dbscan_iris = dbscan_iris.labels_

plt.figure()
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels_dbscan_iris)
plt.title("Iris dataset with DBSCAN")
plt.show()

# Iris with Birch
birch_iris = Birch(n_clusters=3)
birch_iris.fit(iris_scaled)
labels_birch_iris = birch_iris.labels_

plt.figure()
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels_birch_iris)
plt.title("Iris dataset with Birch")
plt.show()

# Iris with AgglomerativeClustering
agglo_iris = AgglomerativeClustering(n_clusters=3)
agglo_iris.fit(iris_scaled)
labels_agglo_iris = agglo_iris.labels_

plt.figure()
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels_agglo_iris)
plt.title("Iris dataset with Agglomerative Clustering")
plt.show()


# MALL CUSTOMERS
mall_scaled = scaler.fit_transform(mall_customers)
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(mall_scaled)
distances, indices = neighbors_fit.kneighbors(mall_scaled)
distances = np.sort(distances[:, 4])

"""
plt.plot(distances)
plt.title("k-distance Graph for DBSCAN")
plt.xlabel("Data points sorted by distance")
plt.ylabel("5th nearest neighbor distance")
plt.show()
"""
# It makes no sense printing the original dataset, it doesnt have classes
# Mall customers with Kmeans
kmeans_mall = KMeans(n_clusters=5)
kmeans_mall.fit(mall_scaled)
labels_kmeans_mall = kmeans_mall.labels_

plt.figure()
plt.scatter(
    mall_customers['Annual Income (k$)'],
    mall_customers['Spending Score (1-100)'],
    c=labels_kmeans_mall, s=50, cmap='viridis'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customers dataset with KMeans")
plt.show()

# Mall customers with Spectral Clustering
spectral_mall = SpectralClustering(n_clusters=5, affinity='rbf', assign_labels='kmeans', random_state=0)
spectral_mall.fit(mall_scaled)
labels_spectral_mall = spectral_mall.labels_

plt.figure()
plt.scatter(
    mall_customers['Annual Income (k$)'],
    mall_customers['Spending Score (1-100)'],
    c=labels_spectral_mall, s=50, cmap='viridis'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customers dataset with Spectral Clustering")
plt.show()

# Mall customers with DBSCAN
dbscan_mall = DBSCAN(eps = 0.75, min_samples = 5)
dbscan_mall.fit(mall_scaled)
labels_dbscan_mall = dbscan_mall.labels_

plt.figure()
plt.scatter(
    mall_customers['Annual Income (k$)'],
    mall_customers['Spending Score (1-100)'],
    c=labels_dbscan_mall, s=50, cmap='viridis'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customers dataset with DBSCAN")
plt.show()

# Mall customers with Birch
birch_mall = Birch(n_clusters=5)
birch_mall.fit(mall_scaled)
labels_birch_mall = birch_mall.labels_

plt.figure()
plt.scatter(
    mall_customers['Annual Income (k$)'],
    mall_customers['Spending Score (1-100)'],
    c=labels_birch_mall, s=50, cmap='viridis'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customers dataset with Birch")
plt.show()

# Mall customers with AgglomerativeClustering
agglo_mall = AgglomerativeClustering(n_clusters=5)
agglo_mall.fit(mall_scaled)
labels_agglo_mall = agglo_mall.labels_

plt.figure()
plt.scatter(
    mall_customers['Annual Income (k$)'],
    mall_customers['Spending Score (1-100)'],
    c=labels_agglo_mall, s=50, cmap='viridis'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customers dataset with Agglomerative Clustering")
plt.show()


# MAKE BLOBS
plt.figure()
plt.scatter(blobs[0][:, 0], blobs[0][:, 1], c=blobs[1], s=30, cmap='viridis')
plt.title("Blobs dataset (Original Labels)")
plt.show()
# Make blobs with Kmeans
kmeans_blobs = KMeans(n_clusters=5)
kmeans_blobs.fit(blobs[0])
labels_kmeans_blobs = kmeans_blobs.labels_

plt.figure()
plt.scatter(blobs[0][:, 0], blobs[0][:, 1], c=labels_kmeans_blobs, s=30, cmap='viridis')
plt.title("Blobs dataset with KMeans")
plt.show()

# Make blobs with Spectral Clustering
spectral_blobs = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_neighbors=10)
spectral_blobs.fit(blobs[0])
labels_spectral_blobs = spectral_blobs.labels_

plt.figure()
plt.scatter(blobs[0][:, 0], blobs[0][:, 1], c=labels_spectral_blobs, s=30, cmap='viridis')
plt.title("Blobs dataset with Spectral Clustering")
plt.show()

# Make blobs with DBSCAN
dbscan_blobs = DBSCAN(eps = 0.7, min_samples = 5)
dbscan_blobs.fit(blobs[0])
labels_dbscan_blobs = dbscan_blobs.labels_

plt.figure()
plt.scatter(blobs[0][:, 0], blobs[0][:, 1], c=labels_dbscan_blobs, s=30, cmap='viridis')
plt.title("Blobs dataset with DBSCAN")
plt.show()

# Make blobs with Birch
birch_blobs = Birch(n_clusters=5)
birch_blobs.fit(blobs[0])
labels_birch_blobs = birch_blobs.labels_

plt.figure()
plt.scatter(blobs[0][:, 0], blobs[0][:, 1], c=labels_birch_blobs, s=30, cmap='viridis')
plt.title("Blobs dataset with Birch")
plt.show()

# Make blobs with AgglomerativeClustering
agglo_blobs = AgglomerativeClustering(n_clusters=5)
agglo_blobs.fit(blobs[0])
labels_agglo_blobs = agglo_blobs.labels_

plt.figure()
plt.scatter(blobs[0][:, 0], blobs[0][:, 1], c=labels_agglo_blobs, s=30, cmap='viridis')
plt.title("Blobs dataset with Agglomerative Clustering")
plt.show()


# MAKE MOONS
plt.figure()
plt.scatter(moons[0][:, 0], moons[0][:, 1], c=moons[1], s=30, cmap='viridis')
plt.title("Moons dataset (Original Labels)")
plt.show()
# Make moons with Kmeans
kmeans_moons = KMeans(n_clusters=2)
kmeans_moons.fit(moons[0])
labels_moons = kmeans_moons.labels_

plt.figure()
plt.scatter(moons[0][:, 0], moons[0][:, 1], c=labels_moons, s=30, cmap='viridis')
plt.title("Moons dataset with KMeans")
plt.show()

# Make moons with Spectral Clustering
spectral_moons = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=10)
spectral_moons.fit(moons[0])
labels_spectral_moons = spectral_moons.labels_

plt.figure()
plt.scatter(moons[0][:, 0], moons[0][:, 1], c=labels_spectral_moons, s=30, cmap='viridis')
plt.title("Moons dataset with Spectral Clustering")
plt.show()

# Make moons with DBSCAN
dbscan_moons = DBSCAN(eps = 0.25, min_samples = 6)
dbscan_moons.fit(moons[0])
labels_dbscan_moons = dbscan_moons.labels_

plt.figure()
plt.scatter(moons[0][:, 0], moons[0][:, 1], c=labels_dbscan_moons, s=30, cmap='viridis')
plt.title("Moons dataset with DBSCAN")
plt.show()

# Make moons with Birch
birch_moons = Birch(n_clusters=2)
birch_moons.fit(moons[0])
labels_birch_moons = birch_moons.labels_

plt.figure()
plt.scatter(moons[0][:, 0], moons[0][:, 1], c=labels_birch_moons, s=30, cmap='viridis')
plt.title("Moons dataset with Birch")
plt.show()

# Make moons with AgglomerativeClustering
agglo_moons = AgglomerativeClustering(n_clusters=2)
agglo_moons.fit(moons[0])
labels_agglo_moons = agglo_moons.labels_

plt.figure()
plt.scatter(moons[0][:, 0], moons[0][:, 1], c=labels_agglo_moons, s=30, cmap='viridis')
plt.title("Moons dataset with Agglomerative Clustering")
plt.show()
