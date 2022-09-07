# 逻辑回归


from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
"""
加载数据集
Parameters:
    filename - 文件名称
Returns:
    feature_list - 数据特征集
    class_list - 数据类别集
"""
def dataLoader(filename):
    feature_list = []
    class_list = []
    with open(filename,'r',encoding='UTF-8') as f:
        content = f.readlines()
        for line in content:
            lineArr = line.strip().split('\t')
            feature_list.append([1.0,float(lineArr[0]),float(lineArr[1])])
            class_list.append(int(lineArr[-1]))
    return np.array(feature_list),np.array(class_list).reshape(-1,1)
"""
绘制数据集
Parameters:
    feature_list - 数据集特征
    class_list - 数据类别
Returns:

"""
def plotData(feature_list,class_list):
    class_color = []
    for item in class_list:
        if item[0] == 1:
            class_color.append('red')
        else:
            class_color.append('green')
            
    plt.scatter(feature_list[:,1],feature_list[:,2],color=class_color,s=20,alpha=.5)
    plt.title('DataSet')
    plt.xlabel('fea1')
    plt.ylabel('fea2')
    negative = mlines.Line2D([],[],color='green',marker='.',markersize=6,label='Negative Sample')
    positive = mlines.Line2D([],[],color='red',marker='.',markersize=6,label='Positive Sample')
    plt.legend(handles = [negative,positive])
    plt.show()
    
"""
绘制决策边界
Parameters:
    feature_list - 数据集特征
    class_list - 数据类别
Returns:

"""
def plotBestFit(feature_list,class_list,W_trained):
    class_color = []
    for item in class_list:
        if item == 1:
            class_color.append('red')
        else:
            class_color.append('green')
            
    plt.scatter(feature_list[:,1],feature_list[:,2],color=class_color,s=20,alpha=.5)
    x = np.arange(-3,3,0.1)
    y = (-W_trained[0][0]-W_trained[1][0]*x)/W_trained[2][0]
    plt.plot(x,y,color = 'orange')
    plt.title('Bestfit')
    plt.xlabel('fea1')
    plt.ylabel('fea2')
    negative = mlines.Line2D([],[],color='green',marker='.',markersize=6,label='Class0')
    positive = mlines.Line2D([],[],color='red',marker='.',markersize=6,label='Class1')
    boundary = mlines.Line2D([],[],color='orange',markersize=6,label='Decision Boundary')
    plt.legend(handles = [negative,positive,boundary])
    plt.show()
    
    
"""
逻辑回归
Parameters:
Returns:
""" 
def logisticRegression(train_data_list,test_data_list,train_class_list,test_class_list,epoch = 1000,alpha = 0.01):
    np.random.seed(1)
    m_train = train_data_list.shape[0] # 样本数量
    m_test = test_data_list.shape[0]
    n = train_data_list.shape[1] # 特征数量
    W = np.random.randn(n,1)
    A = np.zeros((m_train,1))
    for i in range(epoch):
        A = forward(train_data_list,W)
        W = gradUpdate(train_data_list,train_class_list,A,W,alpha)
    W_trained = W # 已经训练好的参数
    error_Count = 0
    train_predict_class = np.around(A)
    train_predict_class.astype(np.int32)
    for i in range(m_train):
        if train_class_list[i][0] != train_predict_class[i][0]:
            error_Count += 1
            
    
    error = error_Count*100/float(m_train)
    acc = 100-error
    print("训练集上错误率为:%.2f%%"%error)
    print("训练集上准确率为:%.2f%%"%acc)
    
    
    # 评估测试集
    error_Count = 0
    A_test = forward(X=test_data_list,W=W)
    test_predict_class = np.around(A_test)
    test_predict_class.astype(np.int32)
    for i in range(m_test):
        if test_class_list[i][0] != test_predict_class[i][0]:
            error_Count += 1
    error = error_Count*100/float(m_test)
    acc = 100-error
    print("测试集上错误率为:%.2f%%"%error)
    print("测试集上准确率为:%.2f%%"%acc)
    return W_trained
"""
前向传播
Parameters:
    X - 输入数据
    W - 权重参数
Returns:
    A - 估计值
"""
def forward(X,W):
    Z = np.dot(X,W)
    A = Sigmoid(Z)
    return A
"""
反向传播,梯度更新
Parameters:
Returns:
"""
def gradUpdate(X,Y,A,W,alpha):
    dW = 1/Y.shape[0]*np.sum(np.dot(X.T,Y-A))
    W += alpha*dW
    return W
"""
Sigmoid函数
Parameters:
Returns:
"""
def Sigmoid(inX):
    return 1/(1+np.exp(-inX))

"""
损失函数,求交叉熵
Parameters:
Returns:
"""
def Cost(Y,A):
    loss = np.multiply(Y,np.log(A)) + np.multiply(1-Y,np.log(1-A))
    cost = 1/loss.shape[0]*np.sum(loss)
    return cost


if __name__ == "__main__":
    filename = "./testSet.txt"
    feature_list,class_list = dataLoader(filename)
    # plotData(feature_list,class_list)
    train_data_list = feature_list[10:]
    test_data_list = feature_list[:10]
    train_class_list = class_list[10:]
    test_class_list = class_list[:10]
    W_trained = logisticRegression(train_data_list,test_data_list,train_class_list,test_class_list)
    plotBestFit(train_data_list,train_class_list,W_trained)