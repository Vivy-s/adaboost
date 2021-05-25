import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random
from numpy import *

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 加载数据集
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

#单层决策树分类函数
#对于该算法的弱分类器，我们采用单层决策树函数进行表示
#xMat 数据矩阵 Q 阈值  S 标志  re 分类结果
def Classify0(xMat,i,Q,S):
    re=np.ones((xMat.shape[0],1))
    if S=='lt':
        re[xMat[:,i]<=Q]=-1
    else:
        re[xMat[:,i]> Q]=-1
    return re

#构建决策树桩
#这里是为了找到最优的决策树桩
#我们可以找到特征中的最大最小值，并由此以及我们设定的步长来求出每循环一次之后阈值的变化，并返回给上面的单层决策树分类函数
#xMat 特征矩阵   yMat 标签矩阵   D 样本权重
#bestStump：最佳决策树信息  minError：最小误差  bestCla：最佳分类结果
def get_Stump(xMat,yMat,D):
    m,n=xMat.shape      #m为样本个数  n为特征数
    Steps=10     #初始化步数，决定要取的样本点
    bestStump={ }
    bestCla=np.mat(np.zeros((m,1)))
    minError=np.inf     #最小误差初始化为无穷大，用来更新最小误差
    for i in range(n):
        Min=xMat[:].min()
        Max=xMat[:].max()
        stepSize=(Max-Min)/Steps   #计算步长
        for j in range(-1,int(Steps)+1):
            for S in ['lt','gt']:      #无差别遍历
                Q=Min+j*stepSize    #计算阈值，用来判别
                re=Classify0(xMat,i,Q,S)   #计算分类结果
                err=np.mat(np.ones((m,1)))  #误差先全部初始化为1
                err[re==yMat]=0   #对与err内的对应分类正确的元素。就将其上的1换为0，并借此来计算我们的误差
                eca=D.T*err      #计算误差
            if eca<minError:    #遍历完所有特征，以找出最小的误差值
                minError=eca
                bestCla=re.copy()
                bestStump['特征列'] = i
                bestStump['阈值'] = Q
                bestStump['标志'] = S
    return bestStump,minError,bestCla          #返回最优决策树桩以及最小误差还有最好的分类结果

#adaboost训练过程
#这里是为了寻找弱分类器并且把他们封装在一个集合里面
#一开始，我们将所有元素的权重设置为一样，进行分类。分对的元素我们在下一次迭代过程中降低它的权重，分错的元素在下一次迭代中提高权重，以保证分类器不断优化
#利用符号函数的公式将我们最终分类器的结果与实际结果进行比较，以求得最终强分类器的准确率
#maxC:最大迭代次数    weakClass：弱分类器信息
def Ada_train(xMat,yMat,maxC=10):
    weakClass=[]
    m=xMat.shape[0]      #获取样本数据个数
    D=np.mat(np.ones((m,1))/m)    #所有加权取为相同值
    aggClass=np.mat(np.zeros((m,1)))
    for i in range(maxC):
        Stump,error,bestCla=get_Stump(xMat,yMat,D)    #构建单层决策树
        alpha=float(0.5*log((1-error)/max(error,1e-16)))    #计算弱分类器权重alpha
        Stump['alpha']=np.round(alpha,2)    #存储弱学习器权重
        weakClass.append(Stump)     #将得到的弱分类器（单层决策树）加入到我们的集合之中
        expon=np.multiply(-1*alpha*yMat,bestCla)
        D=np.multiply(D,exp(expon))
        D=D/D.sum()     #根据权重更新公式  更新样本权重
        aggClass+=alpha*bestCla    #更新累计类别估计值
        aggErr=np.multiply(np.sign(aggClass)==yMat,np.ones((m,1)))    #计算误差
        errRate=aggErr.sum()/m
        if errRate==0: break      #当误差为0时，结束循环
    return weakClass,aggClass

#ada分类函数
#这里是利用已经有的分类器求出分类结果，最后将分类结果进行合并
def AdaClassify(data,weakClass):
    dataMat=np.mat(data)
    m=dataMat.shape[0]
    aggClass=np.mat(np.zeros((m,1)))
    for i in range(len(weakClass)):
        classEst=Classify0(dataMat,
                           weakClass[i]['特征列'],
                           weakClass[i]['阈值'],
                           weakClass[i]['标志'])
        aggClass+=weakClass[i]['alpha']*classEst
    return np.sign(aggClass)

#adaboost的应用
#将我们的数据集导入并进行学习求出最终的准确率
def calAcc(maxC=10):
    xMat, yMat = create_data()
    xMat=np.mat(xMat)
    yMat=np.mat(yMat).transpose()
    m=xMat.shape[0]
    weakClass,aggClass=Ada_train(xMat,yMat,maxC=10)
    yhat=AdaClassify(xMat,weakClass)
    train_re=0
    for i in range(m):
        if (yhat[i]+yMat[i]<=2):
            train_re+=1
    train_acc=train_re/m
    print(f'训练集准确率为{train_acc}')
    return train_acc

calAcc(10)
xMat,yMat=create_data()
yMat=np.mat(yMat).T
weakClass, aggClass = Ada_train(xMat, yMat, 10)









