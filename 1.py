import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import metrics

# 讀取CSV數據集
data = pd.read_csv('banana.csv')

# 提取特徵和標籤
X = data.iloc[:, :-1].values  # 特徵
true_labels = data.iloc[:, -1].values  # 真實標籤

# K-means
print("K-means:")
start_time = time.time()
kmeans = KMeans(n_clusters=2, random_state=42)
pred_labels_kmeans = kmeans.fit_predict(X)
elapsed_time = time.time() - start_time

# 顯示分群效能指標
sse_kmeans = kmeans.inertia_
accuracy_kmeans = metrics.accuracy_score(true_labels, pred_labels_kmeans)
entropy_kmeans = metrics.cluster.normalized_mutual_info_score(true_labels, pred_labels_kmeans)

# 顯示分群效能指標
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"SSE: {sse_kmeans:.4f}")
print(f"Accuracy: {accuracy_kmeans:.4f}")
print(f"Entropy: {entropy_kmeans:.4f}")

# 繪製分群結果
plt.figure(figsize=(12, 4))

# K-means
plt.subplot(1, 3, 1)
plt.scatter(X[pred_labels_kmeans == 0, 0], X[pred_labels_kmeans == 0, 1], marker='o', s=25, edgecolor='none', label='Cluster 1')
plt.scatter(X[pred_labels_kmeans == 1, 0], X[pred_labels_kmeans == 1, 1], marker='o', s=25, edgecolor='none', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='+', s=100, label='Centroids')
plt.title("K-means Clustering")
plt.legend()

# Hierarchical Clustering
print("\nHierarchical Clustering:")
start_time = time.time()
agg_clustering = AgglomerativeClustering(n_clusters=2)
pred_labels_agg = agg_clustering.fit_predict(X)
elapsed_time = time.time() - start_time

# 顯示分群效能指標
pred_labels_agg = 1 - pred_labels_agg
accuracy_agg = metrics.accuracy_score(true_labels, pred_labels_agg)
entropy_agg = metrics.cluster.normalized_mutual_info_score(true_labels, pred_labels_agg)

# 顯示分群效能指標
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"Accuracy: {accuracy_agg:.4f}")
print(f"Entropy: {entropy_agg:.4f}")

# 繪製分群結果
plt.subplot(1, 3, 2)
plt.scatter(X[pred_labels_agg == 0, 0], X[pred_labels_agg == 0, 1], marker='o', s=25, edgecolor='none', label='Cluster 1')
plt.scatter(X[pred_labels_agg == 1, 0], X[pred_labels_agg == 1, 1], marker='o', s=25, edgecolor='none', label='Cluster 2')
plt.title("Hierarchical Clustering")
plt.legend()

# DBSCAN
print("\nDBSCAN:")
start_time = time.time()
dbscan = DBSCAN(eps=0.1, min_samples=5)
pred_labels_dbscan = dbscan.fit_predict(X)
elapsed_time = time.time() - start_time

# 顯示分群效能指標
accuracy_dbscan = metrics.accuracy_score(true_labels, pred_labels_dbscan)
entropy_dbscan = metrics.cluster.normalized_mutual_info_score(true_labels, pred_labels_dbscan)

# 顯示分群效能指標
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"Accuracy: {accuracy_dbscan:.4f}")
print(f"Entropy: {entropy_dbscan:.4f}")

# 繪製分群結果
plt.subplot(1, 3, 3)
plt.scatter(X[pred_labels_dbscan == 0, 0], X[pred_labels_dbscan == 0, 1], marker='o', s=25, edgecolor='none', label='Cluster 1')
plt.scatter(X[pred_labels_dbscan == 1, 0], X[pred_labels_dbscan == 1, 1], marker='o', s=25, edgecolor='none', label='Cluster 2')
plt.title("DBSCAN Clustering")
plt.legend()

plt.tight_layout()
plt.show()