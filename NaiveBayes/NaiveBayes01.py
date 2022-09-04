# 朴素贝叶斯算法
from functools import reduce
import numpy as np
'''
创建实验样本
Parameters:
    None
Returns:
    postingList - 实验样本切分的词条
    classVec - 类别标签向量0代表不是,1代表侮辱性词汇
'''

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],				#切分的词条
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]   																#类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec																#返回实验样本切分的词条和类别标签向量
'''
根据词汇表将inputSet向量化,向量的每个元素为1或0
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 划分的词条列表
Returns:
    returnVec - 文档向量,词集模型
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary"%word)
    return returnVec

'''
创建词汇表

Parameters:
    dataSet - 样本数据集
Returns:
    vocabSet - 返回词汇表
'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
'''
利用朴素贝叶斯进行分类
Parameters:
Returns:
'''
def trainBayes(trainMat,classVec):
    numsDocs = len(trainMat)
    numWords = len(trainMat[0])
    # 先验概率
    pAbusive = classVec.count(1)/numsDocs
    p0Nums = np.zeros(numWords)
    p1Nums = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numsDocs):
        if classVec[i] == 1:
            p1Nums += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Nums += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vect = p1Nums/p1Denom
    p0Vect = p0Nums/p0Denom
    return p0Vect,p1Vect,pAbusive

'''
朴素贝叶斯分类函数
Parameters:
    vec2Classify - 待分类词条
    p0Vec - 侮辱类条件概率
    p1Vec - 非侮辱类条件概率
    pClass1 - 侮辱类先验概率
Returns:
    0 - 非侮辱类
    1 - 侮辱类
'''
def claaifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = reduce(lambda x,y:x**y,vec2Classify*p1Vec)*pClass1
    p0 = reduce(lambda x,y:x*y,vec2Classify*p0Vec)*(1.0-pClass1)
    print("p0=",p0)
    print("p1=",p1)
    if p1 > p0:
        return 1
    else:
        return 0
'''
测试贝叶斯分类器
'''
def testBayes():
    postingList,classVec = loadDataSet()
    print('postingList:\n',postingList)
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n',myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    print('trainMat:\n',trainMat)
    p0Vect,p1Vect,pAbusive = trainBayes(trainMat,classVec)
    testEntry = ['stupid']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    if claaifyNB(thisDoc,p0Vect,p1Vect,pAbusive):
        print(testEntry,"属于侮辱类")
    else:
        print(testEntry,"属于非侮辱类")
    
if __name__ == '__main__':
    testBayes()
    
