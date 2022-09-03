# 用KNN实现数字识别

import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier as KNN


def image2Vector(filename):
    f = open(file=filename)
    content = f.readlines()
    returnVector = np.zeros((1,32*32))
    for i in range(32):
        for j in range(32):
            returnVector[0,32*i+j] = content[i][j]
    return returnVector

def DataLoader(dir):
    labels = []
    # 返回指定目录下的文件名
    File_list = os.listdir(dir)
    # 取得文件个数
    File_nums = len(File_list)
    features = np.zeros((File_nums,1024))
    for i in range(File_nums):
        filename = File_list[i]
        features[i] = image2Vector(os.path.join(dir,filename))
        labels.append(int(filename.split('_')[0]))
    return features,labels

def handwritingClass():
    dir = "./trainingDigits"
    train_features,train_labels = DataLoader(dir)
    dir = "./testDigits"
    test_features,test_labels = DataLoader(dir)
    # 创建分类器
    estimator = KNN(n_neighbors=3,algorithm='auto')
    estimator.fit(train_features,train_labels)
    predict_labels = estimator.predict(test_features)
    for i in range(len(predict_labels)):
        print(f"分类结果:{predict_labels[i]},真实结果:{test_labels[i]}")
    print("准确率为:",estimator.score(test_features,test_labels))

if __name__ == "__main__":
    handwritingClass()
    