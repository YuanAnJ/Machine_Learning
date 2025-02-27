import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

# 导入原始数据集
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# print(train.info())
# print(test.info())

# print(train[['Age','SibSp','Parch','Fare']].describe())
# print(train.describe(include=['O']))

# 画出Age特征的值分布
# sns.violinplot(y='Age',data=train)
# plt.show()

# 拼接训练集和测试集
df = train._append(test, ignore_index=True)
# print(df.shape)
# print(df.head())
# print(df.info())

# 查看类别型特征的缺失值
# categorical = [var for var in df.columns if df[var].dtype == 'O']
# for var in categorical:
#     print(df[var].value_counts())
# print(df[categorical].isnull().sum())
# print(df[categorical].isnull().sum() / float(len(df)))

# 查看数值型特征的缺失值
# numerical = [var for var in df.columns if df[var].dtype != 'O']
# print(df[numerical].isnull().sum())
# print(df[numerical].isnull().sum() / float(len(df)))

'''
数据总共有891行，12列
其中特征Age Cabin Embarked有缺失数据
1) Age里共有数据714，缺失了177，缺失率接近20%
2) Cabin里共有数据204，缺失了687，缺失率为77%
3) Embarked里共有数据889，缺失了2，缺失率为0.2%
'''

# 数据预处理
df['Age'] = df['Age'].fillna(df['Age'].mean())  # 用平均值填充缺失值
# print(df['Embarked'].value_counts())  # 查看众数
df['Embarked'] = df['Embarked'].fillna('S')  # 观察到S为众数，将缺失值填充为S
df['Cabin'] = df['Cabin'].fillna('U')  # 因为缺失值较多，将缺失值填充为U，表示未知

# 查看预处理后结果
print(df.shape)

'''
该数据集有数值类型和分类类型的特征
1. 数值型数据
PassengerId Age Fare Sibsp Parch

2.分类数据
Sex: male female
Embarked: S Q C
Pclass: 1 2 3
Name Cabin Ticket没有明显类别，但可能从中提取出特征
'''

# 将Sex特征的值映射为数值，male为1，female为0
sex_mapDict = {'male': 1, 'female': 0}
df['Sex'] = df['Sex'].map(sex_mapDict)
# print(df['Sex'].head())

# 对Embarked特征进行独热编码
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies(df['Embarked'], prefix='Embarked', dtype='int')
# print(embarkedDf.head())
df = pd.concat([df, embarkedDf], axis=1)
df = df.drop('Embarked', axis=1)
# print(df.head())

# 对Pclass特征进行独热编码
pclassDf = pd.DataFrame()
pclassDf = pd.get_dummies(df['Pclass'], prefix='Pclass', dtype='int')
# print(pclassDf.head())
df = pd.concat([df, pclassDf], axis=1)
df = df.drop('Pclass', axis=1)


# print(df.head())

# 从Name特征中提取有用的特征类别
# 定义函数提取头衔
def getTitle(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()
    return str3


titleDf = pd.DataFrame()
titleDf['Title'] = df['Name'].map(getTitle)
# print(titleDf.head())
'''
将头衔分为一下几类：
Officer 政府官员
Royalty 王室成员
Mr 已婚男士
Mrs 已婚女士
Miss 未婚女士
Master 有技能的人
'''
title_mapDict = {
    'Capt': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Sir': 'Royalty',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'the Countess': 'Royalty',
    'Dona': 'Royalty',
    'Mme': 'Mrs',
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Master': 'Master',
    'Lady': 'Royalty'
}
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
titleDf = pd.get_dummies(titleDf['Title'], dtype='int')
# print(titleDf.head())
df = pd.concat([df, titleDf], axis=1)
df = df.drop('Name', axis=1)
# print(df.head())

# 从Cabin特征中提取有用的特征类别
df['Cabin'] = df['Cabin'].map(lambda c: c[0])
cabinDf = pd.DataFrame()
cabinDf = pd.get_dummies(df['Cabin'], prefix='Cabin', dtype='int')
# print(cabinDf.head())
df = pd.concat([df, cabinDf], axis=1)
df = df.drop('Cabin', axis=1)
# print(df.head())

# 从SibSp和Parch特征中提取有用的特征类别
'''
家庭类别：
Family_Small：家庭人数=1
Family_Middle：2<=家庭人数<=4
Family_Large：家庭人数>=5
'''
familyDf = pd.DataFrame()
familyDf['FamilySize'] = df['Parch'] + df['SibSp'] + 1
familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda s: 1 if s == 1 else 0)
familyDf['Family_Middle'] = familyDf['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
familyDf['Family_Large'] = familyDf['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
# print(familyDf.head())
df = pd.concat([df, familyDf], axis=1)
# print(df.head())

print(df.shape)

# 特征选择
df = df.drop('Ticket', axis=1)
corrDf = df.corr()  # 相关性矩阵
# print(corrDf['Survived'].sort_values(ascending=False))
df_X = pd.concat([titleDf, pclassDf, familyDf, df['Fare'], cabinDf, embarkedDf, df['Sex']], axis=1)
# print(df_X.head())
source_X = df_X.loc[0:890, :]
source_y = df.loc[0:890, 'Survived']
pred_X = df_X.loc[891:, :]

X_train, X_test, y_train, y_test = train_test_split(source_X, source_y, test_size=0.3, random_state=0)
# print(X_train.shape, X_test.shape)
# print(X_train.head())

# 特征缩放（标准化）
# scaler = StandardScaler()
# X_train_s = scaler.fit_transform(X_train)
# X_test_s = scaler.fit_transform(X_test)
# print(X_train_s)

# 模型训练
lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                        intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                        penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                        verbose=0, warm_start=False)
lr.fit(X_train, y_train)
lr_pred_train = lr.predict(X_test)
lr_pred_test = lr.predict(pred_X)
lr_report = classification_report(y_test,lr_pred_train)

clf = DecisionTreeClassifier(criterion="gini", max_depth=13)
clf.fit(X_train, y_train)
clf_pred_train = lr.predict(X_test)
clf_pred_test = clf.predict(pred_X)
clf_report = classification_report(y_test,clf_pred_train)

lr_pred_df = pd.DataFrame(lr_pred_test,columns=['Survived_Pred'])
test_lr = pd.concat([test,lr_pred_df],axis=1)
clf_pred_df = pd.DataFrame(clf_pred_test,columns=['Survived_Pred'])
test_clf = pd.concat([test,clf_pred_df],axis=1)

print('逻辑回归模型预测结果：',lr_pred_test)  # 对测试集的预测结果
print('CART模型预测结果：',clf_pred_test)  # 对测试集的预测结果
print('逻辑回归分类报告：\n',lr_report)
print('CART分类报告：\n',clf_report)

test_lr.to_csv('test_lr.csv',index=False)
test_clf.to_csv('test_clf.csv',index=False)