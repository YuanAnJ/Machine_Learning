from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入数据
df = pd.read_csv('ex1data2.txt', names=['房子大小', '房间数量', '房价'])
# print(df.head())

# 划分自变量和因变量
cols=df.shape[1]
X = df.iloc[:,0:cols-1]
y = df.iloc[:,cols-1:cols]
# print(X.head())
# print(y.head())

X = np.matrix(X.values)
y = np.matrix(y.values)
# print(X.shape)
# print(y.shape)

# 解决sklearn不支持矩阵结构错误
X1 = np.asarray(X)
y1 = np.asarray(y)
# print(X1)
# print(y1)

# 线性回归
model = linear_model.LinearRegression()
reg = model.fit(X1, y1)

# 预测房价
print("模型求解结果：", model.coef_[0,0],model.coef_[0,1])
new_data = np.array([[1300,2]])
print('预测一个两间房，1300平方尺的房子的房价：', model.predict(new_data)[0,0])

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1[:,0],X1[:,1],y1,color='red',marker='o')

x_surf = np.linspace(X1[:,0].min(),X1[:,0].max(),100)
y_surf = np.linspace(X1[:,1].min(),X1[:,1].max(),100)
x_surf,y_surf = np.meshgrid(x_surf,y_surf)
z_surf = model.intercept_ + model.coef_[0,0] * x_surf + model.coef_[0,1] * y_surf
ax.plot_surface(x_surf,y_surf,z_surf,alpha=0.3)

ax.set_xlabel('area')
ax.set_ylabel('rooms')
ax.set_zlabel('price')

plt.show()