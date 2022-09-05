# 实现新闻文本分类
# -*- coding:UTF-8 -*-
import operator
import os
import random
import jieba

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
    print(data_list)
    print(class_list)
    data_class_list = list(zip(data_list,class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size) + 1
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
    
if __name__ == "__main__":
    folder_path = "./SogouC/Sample"
    TextProcessing(folder_path)
    
        