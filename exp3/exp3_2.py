import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report


seed = 1
# 读取数据，并给数据每列命名
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv("./iris_data.csv", names=colnames)

# irisdata.info() 查看数据类型
X = irisdata.drop('Class', axis=1)  # 取出特征
y = irisdata['Class']   # 取出标签

# 随机分割训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 分别使用三种SVM方法训练数据
dct = DecisionTreeClassifier()
dct.fit(X_train,y_train)

# 分别使用训练好的模型进行预测
y_pred = dct.predict(X_test)

# 评估模型预测结果并输出
print('kernel = ''rbf'':')
print('混淆矩阵：\n', confusion_matrix(y_test, y_pred))
print('评估结果报告：\n', classification_report(y_test, y_pred))
