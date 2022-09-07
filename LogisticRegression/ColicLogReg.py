# -*- coding:UTF-8 -*-

import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

"""
sigmoid函数
Parameters:
Returns:
"""

def Sigmoid(inX):
    return 1/(1+np.exp(-inX))

"""
数据读取方法
"""

def dataLoader(filename):
    feature_list = [];class_list = []
    with open(filename,encoding='UTF-8',mode='r') as f:
        for line in f.readlines():
            lineArr = line.strip().split('\t')
            feature_list.append([float(item) for item in lineArr[:-1]])
            class_list.append(float(lineArr[-1]))
    
    return np.array(feature_list),np.array(class_list)

"""
逻辑回归
"""
def GradUpdate(X,Y,epochs = 150):
    m,n = X.shape
    W = np.ones(n)
    for j in range(epochs):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            A = Sigmoid(sum(X[randIndex]*W))
            W += alpha*(Y[randIndex]-A)
            del(dataIndex[randIndex])
    return W
"""
分类器
"""
def classifyVector(inX,W):
    prob = Sigmoid(sum(inX*W))
    if prob > 0.5: return 1.0
    else : return 0.0

if __name__ == "__main__":
    train_data_list,train_class_list = dataLoader('./horseColicTraining.txt')
    test_data_list,test_class_list = dataLoader('./horseColicTest.txt')
    scaler = MinMaxScaler()
    train_data_list = scaler.fit_transform(train_data_list)
    test_data_list = scaler.transform(test_data_list)
    classifier = LogisticRegression(solver='liblinear',max_iter=10)
    classifier.fit(train_data_list,train_class_list)
    test_acc = classifier.score(test_data_list,test_class_list)*100
    print("%.2f%%"%test_acc)