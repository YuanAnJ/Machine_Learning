import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris      # 数据集
from sklearn.tree import DecisionTreeClassifier     # 决策树模型
from sklearn.tree import export_graphviz    # 绘制树模型
from sklearn.utils import Bunch     # Bunch方便存储读取数据

# 选择需要运行的数据集
# datafile = "breast"
datafile = "iris"
# datafile = "wifi"


if __name__ == "__main__":
    # 加载数据集
    dataset = Bunch()       # 定义dataset数据类型
    if datafile == "breast":
        dataset = load_breast_cancer()
    elif datafile == "iris":
        dataset = load_iris()
    elif datafile == "wifi":
        df = pd.read_csv("./wifi_localization.txt", delimiter="\t")
        data = df.to_numpy()
        print(data)
        dataset = Bunch(
            data=data[:, :-1],      # 分开特征与标签
            target=data[:, -1] - 1,
            feature_names=["Wifi {}".format(i) for i in range(1, 8)],       # 特征名字
            target_names=["Room {}".format(i) for i in range(1, 5)],        # 标签名字
        )

    X, y = dataset.data, dataset.target     # 令Bunch中的data, target分别为X, y

    # 创建模型并训练
    # max_depth=1 令树的最大深度为1，可自行修改，比较异同
    clf = DecisionTreeClassifier(criterion="gini", max_depth=10)
    clf.fit(X, y)

    # 使用模型预测
    test_example = []      # 测试样例
    if datafile == "iris":
        test_example = [0, 0, 5.0, 1.5]
    elif datafile == "wifi":
        test_example = [-70, 0, 0, 0, -40, 0, 0]
    elif datafile == "breast":
        test_example = [np.random.rand() for _ in range(30)]
    pred = clf.predict([test_example])[0]       # 预测，并将结果存储于pred中
    print("test_example: {}".format(test_example))
    print("Prediction: " + dataset.target_names[pred])

    # export_graphviz() 可视化决策树
    export_graphviz(
        clf,
        out_file="tree.dot",
        feature_names=dataset.feature_names,
        class_names=dataset.target_names,
        rounded=True,
        filled=True,
    )
    print("Done. Open dot file with Pycharm to view.")