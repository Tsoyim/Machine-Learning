# 过滤垃圾邮件

# -*- coding:UTF-8 -*-
import os
import re
import numpy as np
import random

"""
接收字符串并解析为字符串列表
Parameters:
    bigString - 大字符串
Returns:
    字符串列表
"""

def textParse(bigString):
    listOfTokens = re.split(r'\W+',bigString) # 用正则表达式切分
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] # 除了单个字母，其余字母变成小写

"""
创建词条列表
Parameters:
    dataSet - 数据集
Returns:
    VocabSet - 返回词汇表
"""
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

"""
将词汇表转化为向量,构建词向量模型
Parameters:
    inputSet - 数据集
    VocabList - 词汇表
Returns:
    returnVec - 返回的向量
"""
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word %s is not in VocabList!" % word)
    return returnVec
"""
根据词汇表构建词袋模型
Parameters:
    vocabList - 词汇表
    inputSet - 数据集
Returns:
    returnVec - 词袋模型
"""
def bagOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

"""
创建朴素贝叶斯分类器
Parameters:
    trainMat - 训练数据矩阵
    trainClass - 训练数据类别
Returns:
    p0Vect - ham类的条件概率
    p1Vect - spam类的条件概率
    pSpam - 先验概率
"""
def trainNBayes(trainMat,trainClass):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    pSpam = sum(trainClass)/float(numTrainDocs) #先验概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords) 
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainClass[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    # 求条件概率（采用对数防止溢出）
    p0Vect = np.log(p0Num/p0Denom)
    p1Vect = np.log(p1Num/p1Denom)

    return p0Vect,p1Vect,pSpam
"""
贝叶斯分类器分类函数
Parameters:
    vec2Classify - 待分类标签
    p0Vect - ham类条件概率
    p1Vect - spam类条件概率
    pClass1 - spam类先验概率
Returns:
    0 - ham
    1 - spam
"""
def classifyNB(vec2Classify,p0Vect,p1Vect,pClass1):
    p0 = sum(vec2Classify*p0Vect) + np.log(1.0-pClass1)
    p1 = sum(vec2Classify*p1Vect) + np.log(pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
"""
测试朴素贝叶斯分类器
Parameters:
Returns:
"""
def spamTest():
    docList = []; classList = []
    
    dir_path = "./email/ham"
    for path in os.listdir(dir_path):
        wordList = textParse(open(os.path.join(dir_path,path),'r').read())
        docList.append(wordList)
        classList.append(0) # 标记为邮件，1表示垃圾邮件
    dir_path = "./email/spam"
    for path in os.listdir(dir_path):
        wordList = textParse(open(os.path.join(dir_path,path),'r').read())
        docList.append(wordList)
        classList.append(1) # 标记为邮件，1表示垃圾邮件
    # 创建词汇表
    vocabList = createVocabList(docList)
    # 存储训练集和测试机抽样的索引值
    trainingSet = list(range(50)); testSet = [] 
    # 随机选取索引值加入测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    trainMat = []; trainClass = [] # 生成模型训练数据
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    # 生成贝叶斯模型
    p0V,p1V,pSpam = trainNBayes(trainMat,trainClass)
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        predict_class = classifyNB(wordVector,p0V,p1V,pSpam)
        if predict_class != classList[docIndex]:
            errorCount += 1
    print(errorCount)
    print("错误率：%.2f%%" % (float(errorCount)/len(testSet) * 100))

if __name__ == "__main__":
    spamTest()