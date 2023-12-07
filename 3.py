import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics

# 讀取CSV數據集
data = pd.read_csv('sizes3.csv')

# 提取特徵和標籤
X = data.iloc[:, :-1].values  # 特徵
true_labels = data.iloc[:, -1].values  # 真實標籤

# DBSCAN 1
print("\nDBSCAN (eps=1.0, min_samples=17):")
start_time = time.time()
dbscan1 = DBSCAN(eps=1.0, min_samples=17)
pred_labels_dbscan1 = dbscan1.fit_predict(X)
elapsed_time = time.time() - start_time

# 計算分群效能指標
accuracy_dbscan1 = metrics.accuracy_score(true_labels, pred_labels_dbscan1)
entropy_dbscan1 = metrics.cluster.normalized_mutual_info_score(true_labels, pred_labels_dbscan1)

# 顯示分群效能指標
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"Accuracy: {accuracy_dbscan1:.4f}")
print(f"Entropy: {entropy_dbscan1:.4f}")

# DBSCAN 2
print("\nDBSCAN (eps=1.0, min_samples=15):")
start_time = time.time()
dbscan2 = DBSCAN(eps=1.0, min_samples=15)
pred_labels_dbscan2 = dbscan2.fit_predict(X)
elapsed_time = time.time() - start_time

# 計算分群效能指標
accuracy_dbscan2 = metrics.accuracy_score(true_labels, pred_labels_dbscan2)
entropy_dbscan2 = metrics.cluster.normalized_mutual_info_score(true_labels, pred_labels_dbscan2)

# 顯示分群效能指標
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"Accuracy: {accuracy_dbscan2:.4f}")
print(f"Entropy: {entropy_dbscan2:.4f}")

# DBSCAN 3
print("\nDBSCAN (eps=1.5, min_samples=15):")
start_time = time.time()
dbscan3 = DBSCAN(eps=1.5, min_samples=15)
pred_labels_dbscan3 = dbscan3.fit_predict(X)
elapsed_time = time.time() - start_time

# 計算分群效能指標
accuracy_dbscan3 = metrics.accuracy_score(true_labels, pred_labels_dbscan3)
entropy_dbscan3 = metrics.cluster.normalized_mutual_info_score(true_labels, pred_labels_dbscan3)

# 顯示分群效能指標
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"Accuracy: {accuracy_dbscan3:.4f}")
print(f"Entropy: {entropy_dbscan3:.4f}")

# 合併三次DBSCAN的結果到一張圖
plt.figure(figsize=(15, 5))

# DBSCAN 1的圖
plt.subplot(1, 3, 1)
unique_labels1 = np.unique(pred_labels_dbscan1)
for cluster_num in unique_labels1:
    if cluster_num == -1:
        plt.scatter(X[pred_labels_dbscan1 == cluster_num, 0], X[pred_labels_dbscan1 == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Noise')
    else:
        plt.scatter(X[pred_labels_dbscan1 == cluster_num, 0], X[pred_labels_dbscan1 == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Cluster {cluster_num + 1}')

plt.title("DBSCAN Clustering (eps=1.0, min_samples=17)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# DBSCAN 2的圖
plt.subplot(1, 3, 2)
unique_labels2 = np.unique(pred_labels_dbscan2)
for cluster_num in unique_labels2:
    if cluster_num == -1:
        plt.scatter(X[pred_labels_dbscan2 == cluster_num, 0], X[pred_labels_dbscan2 == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Noise')
    else:
        plt.scatter(X[pred_labels_dbscan2 == cluster_num, 0], X[pred_labels_dbscan2 == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Cluster {cluster_num + 1}')

plt.title("DBSCAN Clustering (eps=1.0, min_samples=15)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# DBSCAN 3的圖
plt.subplot(1, 3, 3)
unique_labels3 = np.unique(pred_labels_dbscan3)
for cluster_num in unique_labels3:
    if cluster_num == -1:
        plt.scatter(X[pred_labels_dbscan3 == cluster_num, 0], X[pred_labels_dbscan3 == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Noise')
    else:
        plt.scatter(X[pred_labels_dbscan3 == cluster_num, 0], X[pred_labels_dbscan3 == cluster_num, 1], marker='o', s=25, edgecolor='none', label=f'Cluster {cluster_num + 1}')

plt.title("DBSCAN Clustering (eps=1.5, min_samples=15)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()