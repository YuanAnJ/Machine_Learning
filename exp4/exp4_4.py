from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据集
breast = load_breast_cancer()
df = pd.DataFrame(breast.data, columns=breast.feature_names)

# PCA降维
pca = PCA(n_components=3)  # 降为3维
df_pca = pca.fit_transform(df)
print(df_pca[:5])

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], color='red', marker='o')
ax.set_xlabel('dimension1')
ax.set_ylabel('dimension2')
ax.set_zlabel('dimension3')

plt.show()
