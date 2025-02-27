import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB

# 读取数据
df = pd.read_excel('watermelon3.xlsx', header=None,
                   names=['颜色', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜'])
# print(df.info)

# 划分特征向量和目标变量
cols = df.shape[1]
X_train = df.iloc[:, 0:cols - 1]
y_train = df['好瓜']
# print(X_train)
# print(Y_train)

# 建立测试集
X_test = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460],
          ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267],
          ['乌黑', '稍缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091],
          ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376],
          ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460],
          ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099]]
X_test = pd.DataFrame(X_test, columns=['颜色', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率'])
# print(X_test.head())

# 特征工程
encoder = ce.OneHotEncoder(cols=['颜色', '根蒂', '敲声', '纹理', '脐部', '触感'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.fit_transform(X_test)
# print(X_train.head())
# print(X_test.head())

# 特征缩放（归一化）
cols = X_train.columns
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
# print(X_train.head())
# print(X_test.head())

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
for var in y_pred:
    print("是否好瓜：", var)
