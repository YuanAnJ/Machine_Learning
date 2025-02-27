from sklearn.linear_model import Lasso,LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

df = pd.read_csv('BJ_AIR_data.csv')
# print(df.head())
# print(df.info())
# print((df['pm2.5']==0).sum())
new_df = df.drop(df[df['pm2.5']==0].index) # 删除数据中的零值，方便对数化
# print(new_df.info())

# 正态检验
# sns.displot(y, kde=True)
# stats.probplot(df['pm2.5'],plot=plt)
# stats.probplot(np.log(new_df['pm2.5']),plot=plt)
# plt.show()

new_df['log_pm2.5'] = np.log(new_df['pm2.5'])
# print(new_df['log_pm2.5'])

features = ['month','hour','DEWP','TEMP','Iws','Is','Ir']
labels = 'log_pm2.5'
X_train,X_test,Y_train,Y_test = train_test_split(new_df[features],new_df[labels],test_size=0.2,random_state=10)

# 数据标准化
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 通过交叉验证找到最佳λ
lasso_cv = LassoCV(cv=10,max_iter=10000)
lasso_cv.fit(X_train,Y_train)
# print(lasso_cv.alpha_)

# Lasso回归
lasso = Lasso(alpha=lasso_cv.alpha_,max_iter=10000)
lasso.fit(X_train,Y_train)
lasso_pre = lasso.predict(X_test)

# print(mean_squared_error(Y_test,lasso_pre))
# print(r2_score(Y_test,lasso_pre))

print(pd.Series(index=['Intercept']+X_train.columns.tolist(),data=[lasso.intercept_]+lasso.coef_.tolist()))