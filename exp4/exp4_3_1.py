import pandas as pd
from sklearn.tree import DecisionTreeClassifier     # 决策树模型
from sklearn.utils import Bunch     # Bunch方便存储读取数据
from sklearn.preprocessing import LabelEncoder


if __name__ == "__main__":
    # 加载数据集
    df = pd.read_csv('./xiguadata.csv')
    le = LabelEncoder()
    df_train = df
    d = {}
    cols_to_encode = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    for col in cols_to_encode:
        df_train[col] = le.fit_transform(df_train[col])
        d[col] = le.classes_
    # print(df_train)
    # print(d)

    data = df_train.to_numpy()
    # print(data)
    dataset = Bunch(
        data=data[:, :-1],
        target=data[:, -1],
        feature_names=['feature {}'.format(i) for i in range(1, 7)],
        target_names=['好瓜', '坏瓜']
    )

    print(dataset.data)

    X, y = dataset.data, dataset.target     # 令Bunch中的data, target分别为X, y

    # 创建模型并训练
    # max_depth=1 令树的最大深度为1，可自行修改，比较异同
    clf = DecisionTreeClassifier(criterion="gini", max_depth=10)
    clf.fit(X, y)

    # 使用模型预测
    test_example = [1,2,0,2,1,0]      # 测试样例
    pred = clf.predict([test_example])[0]       # 预测，并将结果存储于pred中
    print("test_example: {}".format(test_example))
    print("Prediction: " + dataset.target_names[pred])
