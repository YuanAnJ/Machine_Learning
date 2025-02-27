import numpy as np


def load_data_set():
    posting_list = [
        # 颜色    根蒂    敲声    纹理    脐部    触感    密度    含糖率  好瓜
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.460', '是'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '0.774', '0.376', '是'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.634', '0.264', '是'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '0.608', '0.318', '是'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.556', '0.215', '是'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '0.403', '0.237', '是'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '0.481', '0.149', '是'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '0.437', '0.211', '是'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '0.666', '0.091', '否'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '0.243', '0.267', '否'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '0.245', '0.057', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '0.343', '0.099', '否'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '0.639', '0.161', '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '0.657', '0.198', '否'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '0.360', '0.370', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '0.593', '0.042', '否'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '0.719', '0.103', '否']]

    property_list = [
        '青绿', '乌黑', '浅白',
        '蜷缩', '稍缩', '硬挺',
        '浊响', '沉闷', '清脆',
        '清晰', '稍糊', '模糊',
        '凹陷', '稍陷', '稍凹',
        '平坦', '硬滑', '软粘']
    return posting_list, property_list

def getNums(posting_list,col,rows,nums):
    Nums = [0]*nums
    for n in range(0,nums):
        Nums[n] = float(posting_list[rows+n][col])
    return Nums

def train_naive_bayes(posting_list,property_list):
    trainNum = len(posting_list)
    pSampleNum = 0
    for sample in posting_list:
        if sample[-1] == '是':
            pSampleNum += 1
    # 类先验概率
    prioClass = pSampleNum / trainNum
    # 存储正样本类条件概率
    propertyConditionalProbabilityPositive = []
    propertyConditionalProbabilityNegative = []

    # 通过遍历各个属性的属性值集合列表来求其属于 正/负样本的类条件概率，property_list含所有的属性值（无含糖率和密度）
    for propertyy in property_list:
        # 拉普拉斯平滑，防止为0, 西瓜书p153
        # 触感属性只有两个值：硬滑和软粘；拉普拉斯平滑时，该属性正/负样本数目即分母加2，其他属性正/负样本数目即其分母加3
        if (propertyy == "硬滑") or (propertyy == "软粘"):
            pSampleNumLap = pSampleNum + 2
            nSampleNumLap = trainNum - pSampleNum + 2
        elif (propertyy == "凹陷") or (propertyy == "稍陷") or (propertyy == "稍凹") or (propertyy == "平坦"):
            pSampleNumLap = pSampleNum + 4
            nSampleNumLap = trainNum - pSampleNum + 4
        else:
            pSampleNumLap = pSampleNum + 3
            nSampleNumLap = trainNum - pSampleNum + 3

        # 拉普拉斯平滑，初始化为0.1， 此处为参数，请同学们调节不同参数，观察结果
        posNumPropertyPositive = 0.5
        negNumPropertyPositive = 0.5

        # 遍历数据集的每一个样本
        for rows in range(0, len(posting_list)):
            # 如果此时的属性值在样本中
            if propertyy in posting_list[rows]:
                # 如果该样本为正样本
                if posting_list[rows][-1] == '是':
                    # 计算此属性值的正样本数目
                    posNumPropertyPositive += 1

                else:
                    # 计算此属性值的负样本数目
                    negNumPropertyPositive += 1
        # 计算此属性值的正/负类条件概率
        propertyConditionalProbabilityPositive.append(posNumPropertyPositive / pSampleNumLap)
        propertyConditionalProbabilityNegative.append(negNumPropertyPositive / nSampleNumLap)

    # 最后计算正/负样本的 密度和含糖率 的均值和标准差，添加到类条件概率的后面。为了后续通过概率密度函数计算概率
    propertyConditionalProbabilityPositive.append(np.mean(getNums(posting_list, 6, 0, pSampleNum)))
    propertyConditionalProbabilityPositive.append(np.var(getNums(posting_list, 6, 0, pSampleNum)) ** (1 / 2))

    propertyConditionalProbabilityNegative.append(np.mean(getNums(posting_list, 6, pSampleNum, trainNum - pSampleNum)))
    propertyConditionalProbabilityNegative.append(
        np.var(getNums(posting_list, 6, pSampleNum, trainNum - pSampleNum)) ** (1 / 2))

    propertyConditionalProbabilityPositive.append(np.mean(getNums(posting_list, 7, 0, pSampleNum)))
    propertyConditionalProbabilityPositive.append(np.var(getNums(posting_list, 7, 0, pSampleNum)) ** (1 / 2))

    propertyConditionalProbabilityNegative.append(np.mean(getNums(posting_list, 7, pSampleNum, trainNum - pSampleNum)))
    propertyConditionalProbabilityNegative.append(
        np.var(getNums(posting_list, 7, pSampleNum, trainNum - pSampleNum)) ** (1 / 2))
    # 方差标准差装进类条件概率列表中，大致如下:
    # [青绿的正类条件概率,乌黑的正类条件概率,浅白的正类条件概率,蜷缩的正类条件概率,'稍缩的正类条件概率,硬挺的正类条件概率,
    #  浊响的正类条件概率,沉闷的正类条件概率,清脆的正类条件概率,清晰的正类条件概率,稍糊的正类条件概率,模糊的正类条件概率,
    #  凹陷的正类条件概率,稍陷的正类条件概率,稍凹的正类条件概率,硬滑的正类条件概率,软粘的正类条件概率,
    #  正类别的密度的均值,正类别的密度方差,
    #  正类别的含糖率均值，正类别的含糖率方差]
    return propertyConditionalProbabilityPositive, propertyConditionalProbabilityNegative

def classify_naive_bayes(data, propertyConditionalProbabilityPositive, property_list, propertyConditionalProbabilityNegative, posting_list):
    """
    classify_naive_bayes(data,propertyConditionalProbabilityPositive,property_list,propertyConditionalProbabilityNegative):
    功能：求正负类别的概率，返回1或者0， 1表示为正样本
    输入：data：想要测试的数据，格式见底部说明。propertyConditionalProbabilityPositive正类条件概率；propertyConditionalProbabilityNegative负类条件概率
    输出：返回1或者0；其中1代表正样本（好瓜），0代表负样本
    """
    # 初始化正负类条件概率的值
    probabilityPos = 0
    probabilityNeg = 0

    # 下面计算离散属性的后验概率
    # 先算先验概率

    # 总的样本数目
    trainNum = len(posting_list)
    # 正样本数目
    pSampleNum = 0
    for sample in posting_list:
        if sample[-1] == '是':
            pSampleNum += 1
    # 正类先验概率
    prioClassPos = pSampleNum / trainNum
    # 负类先验概率
    prioClassNeg = (trainNum - pSampleNum) / trainNum

    # 遍历测试数据的属性， 其密度和含糖率不在循环中计算
    for propertyData in data[:-1]:
        if propertyData in property_list:
            # 取该属性的下标(索引值)
            index = property_list.index(propertyData)
            # 取该属性值的正/负样本类条件概率值，并取对数，然后加起来来求正负样本各自的概率。 取对数为了防止下溢，将乘法转为加法计算。
            probabilityPos += np.log(propertyConditionalProbabilityPositive[index])+np.log(prioClassPos)  # 条件概率乘上正类先验概率
            probabilityNeg += np.log(propertyConditionalProbabilityNegative[index])+np.log(prioClassNeg)  # 条件概率乘上负类先验概率

    # 对于连续属性密度和含糖率，通过概率密度函数计算其属于正/负样本的概率。西瓜书p151
    # np.log默认底数为e，因此后面直接加上e的指数项而不用取log（因为log（exp ** n）= n）
    probabilityPos += np.log(((2 * np.pi) ** (1 / 2) * propertyConditionalProbabilityPositive[-3]) ** (-1)) + (-1/2)*((float(data[-2])- propertyConditionalProbabilityPositive[-4]) ** 2) / (propertyConditionalProbabilityPositive[-3] ** 2)
    probabilityPos += np.log(((2 * np.pi) ** (1 / 2) * propertyConditionalProbabilityPositive[-1]) ** (-1)) + (-1/2)*((float(data[-1])- propertyConditionalProbabilityPositive[-2]) ** 2) / (propertyConditionalProbabilityPositive[-1] ** 2)
    probabilityNeg += np.log(((2 * np.pi) ** (1 / 2) * propertyConditionalProbabilityNegative[-3]) ** (-1)) + (-1/2)*((float(data[-2])- propertyConditionalProbabilityNegative[-4]) ** 2) / (propertyConditionalProbabilityNegative[-3] ** 2)
    probabilityNeg += np.log(((2 * np.pi) ** (1 / 2) * propertyConditionalProbabilityNegative[-1]) ** (-1)) + (-1/2)*((float(data[-1])- propertyConditionalProbabilityNegative[-2]) ** 2) / (propertyConditionalProbabilityNegative[-1] ** 2)

    # 对算出来的正负概率进行比较，大的为正样本
    if probabilityPos > probabilityNeg:
        return 1
    else:
        return 0

if __name__ == "__main__":
    # 载入数据
    posting_list, property_list = load_data_set()
    # 预训练
    propertyConditionalProbabilityPositive, propertyConditionalProbabilityNegative = train_naive_bayes(posting_list,  property_list)

    # 朴素贝叶斯求类别

    # 输入数据集中前两个正样本例子
    data = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.460']
    result = classify_naive_bayes(data, propertyConditionalProbabilityPositive, property_list, propertyConditionalProbabilityNegative, posting_list)
    print('is it the good melon : {}'.format(result))

    data = ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '0.774', '0.376']
    result = classify_naive_bayes(data, propertyConditionalProbabilityPositive, property_list, propertyConditionalProbabilityNegative, posting_list)
    print('is it the good melon : {}'.format(result))

    # 输入数据集中前两个负样本例子
    data = ['乌黑', '稍缩', '沉闷', '稍糊', '稍凹', '硬滑', '0.666', '0.091']
    result = classify_naive_bayes(data, propertyConditionalProbabilityPositive, property_list, propertyConditionalProbabilityNegative, posting_list)
    print('is it the good melon : {}'.format(result))

    data = ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '0.243', '0.267']
    result = classify_naive_bayes(data, propertyConditionalProbabilityPositive, property_list, propertyConditionalProbabilityNegative, posting_list)
    print('is it the good melon : {}'.format(result))

    # 西瓜书p151 列子
    data = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.460']
    result = classify_naive_bayes(data, propertyConditionalProbabilityPositive, property_list, propertyConditionalProbabilityNegative, posting_list)
    print('is it the good melon : {}'.format(result))

    data = ['浅白','蜷缩','浊响','模糊','平坦','软粘','0.343','0.099']
    result = classify_naive_bayes(data,propertyConditionalProbabilityPositive, property_list, propertyConditionalProbabilityNegative, posting_list)
    print('it is the good melon : {}'.format(result))