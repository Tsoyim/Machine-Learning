# 决策树模型
from pyexpat import features
import numpy as np

def generate_dataSet():
    # 年龄（0青年、1中年、2老年） 工作（0无、1有） 房子（0无、1有）信贷情况（0一般、1好、2非常好）
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no']]
    features = ['年龄', '有工作', '有自己的房子', '信贷情况']	
    return dataSet, features
'''
计算香农经验熵
'''
def calcEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVect in dataSet:
        currentLabel = featVect[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        
    Entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries