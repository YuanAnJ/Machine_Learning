import pandas as pd
import numpy as np


class DecisionTree:
    '''id3 tree'''

    def __init__(self):
        self.model = None

    def calEntropy(self, y):  # 计算熵
        valRate = y.value_counts().apply(lambda x: x / y.size)  # 频次汇总 得到各个特征对应的概率
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1
        return valEntropy

    def fit(self, xTrain, yTrain=pd.Series()):
        if yTrain.size == 0:  # 如果不传，自动选择最后一列作为分类标签
            yTrain = xTrain.iloc[:, -1]
            xTrain = xTrain.iloc[:, :len(xTrain.columns) - 1]
        self.model = self.buildDecisionTree(xTrain, yTrain)
        return self.model

    def buildDecisionTree(self, xTrain, yTrain):
        propNamesAll = xTrain.columns
        # print(propNamesAll)
        yTrainCounts = yTrain.value_counts()
        if yTrainCounts.size == 1:
            # print('only one class', yTrainCounts.index[0])
            return yTrainCounts.index[0]
        entropyD = self.calEntropy(yTrain)

        maxGain = None
        maxEntropyPropName = None
        for propName in propNamesAll:
            propDatas = xTrain[propName]
            propClassSummary = propDatas.value_counts().apply(lambda x: x / propDatas.size)  # 频次汇总 得到各个特征对应的概率

            sumEntropyByProp = 0
            for propClass, dvRate in propClassSummary.items():
                yDataByPropClass = yTrain[xTrain[propName] == propClass]
                entropyDv = self.calEntropy(yDataByPropClass)
                sumEntropyByProp += entropyDv * dvRate
            gainEach = entropyD - sumEntropyByProp
            if maxGain == None or gainEach > maxGain:
                maxGain = gainEach
                maxEntropyPropName = propName
        # print('select prop:', maxEntropyPropName, maxGain)
        propDatas = xTrain[maxEntropyPropName]
        propClassSummary = propDatas.value_counts().apply(lambda x: x / propDatas.size)  # 频次汇总 得到各个特征对应的概率

        retClassByProp = {}
        for propClass, dvRate in propClassSummary.items():
            whichIndex = xTrain[maxEntropyPropName] == propClass
            if whichIndex.size == 0:
                continue
            xDataByPropClass = xTrain[whichIndex]
            yDataByPropClass = yTrain[whichIndex]
            del xDataByPropClass[maxEntropyPropName]  # 删除已经选择的属性列

            # print(propClass)
            # print(pd.concat([xDataByPropClass, yDataByPropClass], axis=1))

            retClassByProp[propClass] = self.buildDecisionTree(xDataByPropClass, yDataByPropClass)

        return {'Node': maxEntropyPropName, 'Edge': retClassByProp}

    def predictBySeries(self, modelNode, data):
        if not isinstance(modelNode, dict):
            return modelNode
        nodePropName = modelNode['Node']
        prpVal = data.get(nodePropName)
        for edge, nextNode in modelNode['Edge'].items():
            if prpVal == edge:
                return self.predictBySeries(nextNode, data)
        return None

    def predict(self, data):
        if isinstance(data, pd.Series):
            return self.predictBySeries(self.model, data)
        return data.apply(lambda d: self.predictBySeries(self.model, d), axis=1)


if __name__ == "__main__":
    # 加载数据集
    data = pd.read_csv("./xiguadata.csv", encoding="utf-8")
    dataTrain = data.iloc[0:15,:]
    dataTest = data.iloc[15:,:]

    # 创建模型并训练
    decisionTree = DecisionTree()
    treeData = decisionTree.fit(dataTrain)

    # 测试
    print(pd.DataFrame({'Prediction': decisionTree.predict(dataTest), 'Value': dataTest.iloc[:, -1]}))

    # 输出决策树
    print(treeData)