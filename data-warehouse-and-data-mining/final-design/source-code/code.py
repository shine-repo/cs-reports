import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
import pydotplus
import matplotlib.pyplot as plt

def data_preprocessing(data):
    # 数据清洗，采用用平均值填充空缺值
    for column in list(data.columns[data.isnull().sum() > 0]):
        mean_val = data[column].mean()
        data[column].fillna(mean_val, inplace=True)
    data = np.array(data.values)
    feature = data[:,0:3]
    label = data[:,3]
    for i in range(np.size(label)):
        if label[i]<60:
            label[i] = 3
        elif label[i]<80 and label[i]>59:
            label[i] = 2
        else:
            label[i] = 1
    return feature,label

def structural_model(feature,label):
    # 数据集划分，70%训练数据，30%测试数据
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3)
    # 选取最合适的深度
    max_depths = []
    for max_depth in range(10):
        clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=max_depth+1)
        clf.fit(feature_train, label_train) # 拟合
        score = clf.score(feature_test,label_test)
        max_depths.append(score)
    best_depth = max_depths.index(max(max_depths))+1
    # plt.figure(figsize=(20, 8), dpi=80)
    # plt.plot(range(1, 11), max_depths)
    # plt.xlabel('max depth')
    # plt.ylabel('evaluate score')
    # plt.show();
    # 选取最合适的最小叶子树
    min_samples = []
    for min_sample in range(30):
        clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=best_depth,min_samples_leaf=min_sample+5)
        clf.fit(feature_train, label_train)  # 拟合
        score = clf.score(feature_test, label_test)
        min_samples.append(score)
    best_min_samples_leaf = min_samples.index(max(min_samples))+5
    # plt.figure(figsize=(20, 8), dpi=80)
    # plt.plot(range(4, 34), min_samples)
    # plt.xlabel('min samples leaf')
    # plt.ylabel('evaluate score')
    # plt.show();
    # 根据最合适的参数构建模型
    mytree = tree.DecisionTreeClassifier(criterion='entropy',max_depth=best_depth,min_samples_leaf=best_min_samples_leaf)
    mytree.fit(feature_train,label_train)
    return mytree

def tree_visualise(mytree):
    # 可视化
    dot_data = tree.export_graphviz(mytree, out_file=None,\
                                    feature_names=["Attendance","Preview","Job"],\
                                    class_names=["excellent","good","poor"],\
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("dtree10.pdf")

if __name__ == '__main__':
    # 加载数据
    data = pd.read_csv('student_data.csv')
    # 数据预处理
    feature,label = data_preprocessing(data)
    # 构造模型
    mytree = structural_model(feature,label)
    # 可视化训练好的决策树
    tree_visualise(mytree)
    # 计算预测正确率
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3)
    rate = np.sum(mytree.predict(feature_test) == label_test) / mytree.predict(feature_test).size
    print('训练集数量：',label_train.size)
    print('测试集数量：',label_test.size)
    print('正确率：',rate)