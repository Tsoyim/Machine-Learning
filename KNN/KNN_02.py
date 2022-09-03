# -*- utf-8 -*-

from fileinput import filename
from turtle import color, distance
from unittest import result
import numpy as np

def get_dataSet(file_dir):
    
    f = open(file_dir)
    # 获取全部内容
    f_content = f.readlines()
    # 返回文件行数
    row = len(f_content)
    # 用于保存输入特征
    features = np.zeros((row,3))
    # 用于保存类别向量
    Labels = []
    # 行索引值
    index = 0
    # 将数据输入特征向量和标签向量中
    for line in f_content:
        # 删除空白字符
        line = line.strip()
        # 切片分割
        ListFromLine = line.split('\t')
        # 特征提取
        features[index,:] = ListFromLine[:3]
        
        label = ListFromLine[-1]
        if label == 'didntLike':
            Labels.append(0)
        if label == 'smallDoses':
            Labels.append(1)
        if label == 'largeDoses':
            Labels.append(2)
        index += 1
    return features,Labels
'''
消除数据量纲，对数据标准化处理
'''
def dataProceesor(features):
    features_mean = np.mean(features,axis=0)
    print(features_mean)
    features_std = np.std(features,axis=0)
    print(features_std)
    features_standard = (features-features_mean)/features_std
    return features_mean,features_std,features_standard

'''
数据可视化
'''
def showdatas(features,Labels):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import matplotlib.lines as mlines
    font = FontProperties(fname=r"C:\\Windows\\Fonts\\simsunb.ttf",size = 12)
    # 创建画布
    fig , axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize = (13,8))
    
    numberOfLabels = len(Labels)
    LabelsColors = []
    # 黑色代表不喜欢，橙色代表一点，红色代表很
    for i in Labels:
        if i == 0:
            LabelsColors.append('black')
        if i == 1:
            LabelsColors.append('orange')
        if i == 2:
            LabelsColors.append('red')
    # 画散点图
    axs[0][0].scatter(x=features[:,0],y=features[:,1],color=LabelsColors,s=15,alpha=0.5)
    axs0_title_text = axs[0][0].set_title(u'每年飞行常客里程数与玩视频游戏时间',fontproperties='SimHei')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年飞行常客里程数',fontproperties='SimHei')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏时间',fontproperties='SimHei')
    plt.setp(axs0_title_text,size=9,weight='bold',color = 'red')
    plt.setp(axs0_xlabel_text,size=7,weight='bold',color = 'black')
    plt.setp(axs0_ylabel_text,size=7,weight='bold',color = 'black')
    
    axs[0][1].scatter(x=features[:,0],y=features[:,2],color=LabelsColors,s=15,alpha=0.5)
    axs1_title_text = axs[0][1].set_title(u'每年飞行常客里程数与每周消费冰淇淋公升数',fontproperties='SimHei')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年飞行常客里程数',fontproperties='SimHei')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费冰淇淋公升数',fontproperties='SimHei')
    plt.setp(axs1_title_text,size=9,weight='bold',color = 'red')
    plt.setp(axs1_xlabel_text,size=7,weight='bold',color = 'black')
    plt.setp(axs1_ylabel_text,size=7,weight='bold',color = 'black')
            
    axs[1][0].scatter(x=features[:,1],y=features[:,2],color=LabelsColors,s=15,alpha=0.5)
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏时间与每周消费冰淇淋公升数',fontproperties='SimHei')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏时间',fontproperties='SimHei')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费冰淇淋公升数',fontproperties='SimHei')
    plt.setp(axs2_title_text,size=9,weight='bold',color = 'red')
    plt.setp(axs2_xlabel_text,size=7,weight='bold',color = 'black')
    plt.setp(axs2_ylabel_text,size=7,weight='bold',color = 'black')
    
    # 设置图例
    didntLike = mlines.Line2D([],[],color='black',marker='.',markersize=6,label='didntLike')
    smallDoses = mlines.Line2D([],[],color='orange',marker='.',markersize=6,label='smallDoses')
    largeDoses = mlines.Line2D([],[],color='red',marker='.',markersize=6,label='largeDoses') 
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
      
    plt.show()

'''
KNN算法执行
Input:只能是单个特征
'''
def classify(test,features,Labels,k):
    test_trans = np.tile(test,(features.shape[0],1))
    sqdiff = (test_trans-features)**2
    distances = np.sqrt(np.sum(sqdiff,axis=1))
    sortedIndices = distances.argsort()
    classCount = np.zeros((3,1))
    for i in range(k):
        classes = Labels[sortedIndices[i]]
        classCount[classes][0] += 1
    return classCount.argmax()

'''
测试某个人你是否喜欢
'''
def classifyPerson(train_features,train_Labels,k):
    classes_val = ['讨厌','有些喜欢','很喜欢']
    flight = float(input("请输入这个人每年的飞行里程数："))
    games = float(input("请输入这个人每周玩游戏的时间："))
    ice_cream = float(input("请输入这个人每周冰淇淋消耗数："))
    person_fea = np.array([flight,games,ice_cream])     
    mean,std,standard_train_features = dataProceesor(train_features)
    person_fea = (person_fea-mean)/std
    result = classify(person_fea,standard_train_features,train_Labels,k)
    print(f"你可能{classes_val[result]}这个人")

'''
计算分类错误率
'''   
def datasetTest(features,Labels,k):
    train_features = features[:800]
    train_Labels = Labels[:800]
    test_features = features[800:]
    test_Labels = Labels[800:]
    mean,std,train_std_features = dataProceesor(train_features)
    test_std_features = (test_features-mean)/std
    correct_Count = 0
    i = 0
    for test_std_feature in test_std_features:
        result = classify(test_std_feature,train_std_features,train_Labels,k)
        if result == test_Labels[i]:
            correct_Count += 1
        i+=1
    acc = correct_Count*100/len(test_Labels)
    error = 1-acc
    print(f"准确率为{acc}%")
        
    
if __name__ == "__main__":
    features,Labels = get_dataSet(".\dataSet02.txt")
    datasetTest(features,Labels,3)
    
    