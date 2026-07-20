import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# 1. 准备数据
iris = load_iris()
X = iris.data
y = iris.target  # 真实标签（仅用于后续评估聚类效果）

# 2. 数据标准化 (StandardScaler)
# ⚠️ 关键步骤：PCA和K-Means都对特征尺度极其敏感，必须先标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA 降维
# 将 4 维原始特征降维到 2 维，方便后续可视化和聚类
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 打印解释方差比，看看这两个主成分保留了多少原始信息
print(f"两个主成分的累计解释方差比: {sum(pca.explained_variance_ratio_):.4f}")

# 4. K-Means 聚类
# 因为鸢尾花有3个品种，这里我们设定 K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# 5. 结果可视化
plt.figure(figsize=(10, 6))

# 绘制聚类结果
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, edgecolor='k')
plt.title('K-Means Clustering on PCA-reduced Iris Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter, label='Cluster Label')
plt.show()

# 6. 简单评估（对比聚类标签与真实标签）
# 注意：K-Means的标签(0,1,2)和真实标签(0,1,2)没有必然对应关系，
# 这里仅做简单的准确率演示，实际中常用轮廓系数等无监督指标
print(f"聚类标签分布: {np.bincount(clusters)}")