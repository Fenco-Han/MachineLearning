# -×- encoding=utf-8 -*-
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree

# read the csv file
allElectronicsData = open('play.csv', 'rt')
reader = csv.reader(allElectronicsData)
header = next(reader)

print(header)

featureList = []
labelList = []

for row in reader:
    # Class_buys_computer字段数据保存至lableList
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        # 读取每一行数据
        rowDict[header[i]] = row[i]
    # 将读取数据保存至fearureList
    featureList.append(rowDict)

print(featureList)

# vectorize feature:特征向量化
vec = DictVectorizer()
# 将数据向量化成0，1
dumpyX = vec.fit_transform(featureList).toarray()

print("dunmpyX " + str(dumpyX))
# vec.get_feature_names_out() 可以看到各个字段取值有哪些['age=middle_aged' 'age=senior' 'age=youth' 'credit_rating=excellent' 'credit_rating=fair' 'income=high' 'income=low' 'income=medium''student=no' 'student=yes']
print("feature_name" + str(vec.get_feature_names_out()))
# labeList中为最终的结果取值，是否买电脑
print("labelList " + str(labelList))

# 标签二值化，将原来的yes,no 标签转换成1，0
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:" + str(dummyY))

# use the decision tree for classfication
# 使用决策树，entropy信息熵
clf = tree.DecisionTreeClassifier(criterion='entropy')
# 构造决策树
clf = clf.fit(dumpyX, dummyY)

# 打印构造决策树采用的参数
print("clf : " + str(clf))

# visilize the model

with open('play_hpf.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names_out(), out_file=f)
# dot -Tpdf in.dot -o out.pdf输出pdf文件

# 验证数据，【取第一行数据】，修改几个属性预测结果
oneRowX = dumpyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
# 将第一行数据修改 001修改为100，修改第一个和第三个值
newRowX[0] = 0
newRowX[2] = 0
newRowX[1] = 1
print("newRowX:" + str(newRowX))
# 开始进行预测
predictedY = clf.predict(np.array(newRowX).reshape(1, -1))
print("predictedY:" + str(predictedY))
