# 利用决策树预测隐形眼镜类型
from io import StringIO
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pydotplus

def generate_dataSet(filename):
    fr = open(filename)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    lenses_pd = pd.DataFrame(lenses_dict)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    return lenses_pd,lenses_target

def generate_estimator(filename):
    lenses_pd,lenses_target = generate_dataSet(filename)
    estimator = tree.DecisionTreeClassifier(max_depth=4)
    estimator.fit(lenses_pd.values.tolist(),lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(estimator,out_file=dot_data,feature_names=lenses_pd.keys(),
                         class_names=estimator.classes_,filled=True,rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')
    
if __name__ == '__main__':
    filename = './lenses.txt'
    generate_estimator(filename)