# 实现新闻文本分类
# -*- coding:UTF-8 -*-
import operator
import os
import random
import jieba
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
"""
新闻文本预处理
Parameters:
    folder_path - 文件路径
    test_size - 测试集比例
Returns:
    data_list - 数据集
    class_list - 类别

"""
def TextProcessing(folder_path,test_size):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []
    
    # 遍历每个子文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        j = 1
        # 遍历每个txt文件
        for file in files:
            if j > 100:
                break
            with open(os.path.join(new_folder_path,file),'r',encoding='utf-8') as f:
                raw = f.read()
            # 精简模式，返回一个可迭代的generator
            word_cut = jieba.cut(raw,cut_all=False)
            word_list = list(word_cut)
            
            data_list.append(word_list)
            class_list.append(folder)
            j += 1
    data_class_list = list(zip(data_list,class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size)
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list , train_class_list = zip(*train_list) # 训练集解压缩
    test_data_list , test_class_list = zip(*test_list) # 测试集解压缩
    
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(),key = lambda f:f[1],reverse = True)
    all_words_list , all_words_nums = zip(*all_words_tuple_list) # 解压缩
    all_words_list = list(all_words_list)
    return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list
    
"""
读取文件里的内容并去重
Parameters:
    words_file - 文件路径
Returns:
    words_set - 读取内容的集合
"""
def MakeWordSet(words_file):
    words_set = set()
    with open(words_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)
    return words_set

"""
文本特征提取
Parameters:
    all_word_list - 训练集所有文本列表
    deleteN - 删除词频最高的deleteN个词
    stopwords_set - 指定的结束语
Returns:
    feature_words - 特征集
"""
def words_dict(all_words_list,deleteN,stopwords_set = set()):
    features_words = []
    n = 1
    for t in range(deleteN,len(all_words_list),1):
        if n > 1000:
            break    
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            features_words.append(all_words_list[t])
        n += 1
    
    return features_words

"""
文本特征向量化
Parameters:
    train_data_list - 训练集
    test_data_list - 测试集
    feature_words - 特征集
Returns:
    train_feature_list - 训练集向量化列表
    test_feature_list - 测试集向量化列表
"""     
def TextFeatures(train_data_list,test_data_list,feature_words):
    def text_features(text,feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text,feature_words) for text in train_data_list]
    test_feature_list = [text_features(text,feature_words) for text in test_data_list]
    return train_feature_list,test_feature_list

"""
生成贝叶斯分类器
Parameters:
    train_feature_list - 训练向量
    test_feature_list - 测试向量
    train_class_list - 训练集分类标签
    test_class_list - 测试集分类标签
Returns:
"""
def textClassifier(train_feature_list,test_feature_list,train_class_list,test_class_list):
    classifier = MultinomialNB(alpha=1,fit_prior=True)
    classifier.fit(train_feature_list,train_class_list)
    estimate_class_list = classifier.predict(test_feature_list)
    error_Count = 0.0
    for i in range(len(test_class_list)):
        if test_class_list[i] != estimate_class_list[i]:
            error_Count += 1
    loss = error_Count/float(len(test_class_list))
    acc = 1 - loss
    loss *= 100
    acc *= 100
    print("在测试集上的错误率为:%.2f%%\n在测试集上的正确率为:%.2f%%" % (loss,acc))
    
if __name__ == "__main__":
    folder_path = "./SogouC/Sample"
    all_words_list,train_data_list,test_data_list,train_class_list,test_class_list = TextProcessing(folder_path,0.2)
    
    stopwords_file = "./stopwords_cn.txt"
    stopwords_set = MakeWordSet(stopwords_file)
    
    features_words = words_dict(all_words_list,100,stopwords_set)
    
    train_feature_list,test_feature_list = TextFeatures(train_data_list,test_data_list,features_words)
    print(len(train_feature_list),len(test_feature_list))
    textClassifier(train_feature_list,test_feature_list,train_class_list,test_class_list)
    
        