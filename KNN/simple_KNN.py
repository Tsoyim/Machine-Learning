# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
import numpy as np
import operator

def generate_dataset():
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    labels = ['Love','Love','Action','Action']
    return group,labels

def KNN_classfiy(test,train,labels,k): 
    sqdistance = np.power(train-test,2)
    distance = np.sqrt(np.sum(sqdistance,axis=1))
    classCount = {}
    sortedIndices = distance.argsort()
    for i in range(k):
        key = labels[sortedIndices[i]]
        classCount[key] = classCount.get(key,0) + 1
    classCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    
    return classCount[0][0] 
    
        
    

if __name__ == '__main__':
    group,labels = generate_dataset()
    print(group)
    print(group.shape)
    print(labels)
    trans_labels = []
    test = np.array([100,20])
    
    # 执行KNN
    result = KNN_classfiy(test,group,labels,3)
    print(result)