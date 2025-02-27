import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ex1data1.txt', names=['人口', '利润'])
fig = plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
sns.set(font='SimHei',font_scale=1.5)
sns.lmplot(x='人口',y='利润',data=df,height=6,fit_reg=False)
plt.show()
def computeCost (x,y,theta):
    inner = np.power((x*theta.T)-y,2)
    return np.sum(inner)/(2*len(x))

cols = df.shape[1]
x = df.iloc[:,0:cols-1]
y = df.iloc[:,cols-1:cols]

x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

computeCost(x,y,theta)
def gradientDescent(x,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (x*theta.T)-y

        for j in range(parameters):
            term = np.multiply(error,x[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sun(term))

        theta = temp
        cost[i] = computeCost(x,y,theta)
    return theta,cost

alpha = 0.01
iters = 1500

g,cost = gradientDescent(x,y,theta,alpha,iters)

computeCost(x,y,g)

x = np.linspace(df.人口.min(),df.人口.max(),100)
f = g[0,0] + (g[0,1] * x)
