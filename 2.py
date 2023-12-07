import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import metrics

# 讀取CSV數據集
data = pd.read_csv('sizes3.csv')

# 提取特徵和標籤
X = data.iloc[:, :-1].values  # 特徵
true_labels = data.iloc[:, -1].values  # 真實標籤

# K-means
print("K-means:")
start_time = time.time()
kmeans = KMeans(n_clusters=4, random_state=42)
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
for cluster_num in range(4):
    plt.scatter(X[pred_labels_kmeans == cluster_num, 0], X[pred_labels_kmeans == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Cluster {cluster_num + 1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='+', s=100, label='Centroids')
plt.title("K-means Clustering")
plt.legend()
plt.show()

# 階層式分群
print("\nHierarchical Clustering:")
start_time = time.time()
agg_clustering = AgglomerativeClustering(n_clusters=4)
pred_labels_agg = agg_clustering.fit_predict(X)
elapsed_time = time.time() - start_time

# 顯示分群效能指標
accuracy_agg = metrics.accuracy_score(true_labels, pred_labels_agg)
entropy_agg = metrics.cluster.normalized_mutual_info_score(true_labels, pred_labels_agg)

# 顯示分群效能指標
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"Accuracy: {accuracy_agg:.4f}")
print(f"Entropy: {entropy_agg:.4f}")

# 繪製分群結果
for cluster_num in range(4):
    plt.scatter(X[pred_labels_agg == cluster_num, 0], X[pred_labels_agg == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Cluster {cluster_num + 1}')

plt.title("Hierarchical Clustering")
plt.legend()
plt.show()

# DBSCAN
print("\nDBSCAN:")
start_time = time.time()
dbscan = DBSCAN(eps=1.0, min_samples=15)
pred_labels_dbscan = dbscan.fit_predict(X)
elapsed_time = time.time() - start_time

# 計算分群效能指標
accuracy_dbscan = metrics.accuracy_score(true_labels, pred_labels_dbscan)
entropy_dbscan = metrics.cluster.normalized_mutual_info_score(true_labels, pred_labels_dbscan)

# 顯示分群效能指標
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"Accuracy: {accuracy_dbscan:.4f}")
print(f"Entropy: {entropy_dbscan:.4f}")

# 繪製分群結果
unique_labels = np.unique(pred_labels_dbscan)
for cluster_num in unique_labels:
    if cluster_num == -1:
        plt.scatter(X[pred_labels_dbscan == cluster_num, 0], X[pred_labels_dbscan == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Noise')
    else:
        plt.scatter(X[pred_labels_dbscan == cluster_num, 0], X[pred_labels_dbscan == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Cluster {cluster_num + 1}')

plt.title("DBSCAN Clustering")
plt.legend()
plt.show()