# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:27:58 2021

@author: shu
"""

import os
import pandas as pd
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# -----Load Data
iris=datasets.load_iris()

# ----Process Data
#產生訓練的特徵向量，從資料挖出特徵欄位
X=pd.DataFrame(iris.data,columns=iris.feature_names)
# 看是哪一類型的燕尾花
target=pd.DataFrame(iris.target,columns=["target"])
y=target["target"]
# 將資料分成training跟data，隨機取1/3當亂數，2/3當種子
XTrain,XTest,yTrain,yTest=train_test_split(X,y,test_size=0.33,random_state=1)

print("X info  ----->\n",X.info())
print("X describe --->\n",X.describe())
print("columns feature -->\n",X.columns,sep="")

# ----- Define and train model
# 決策樹最多看8層，這題最多用到4
# 做決策樹
dtree=tree.DecisionTreeClassifier(max_depth=8)
dtree.fit(XTrain,yTrain)

# ----Evaluation and Predict
print("準確率:",dtree.score(XTest,yTest))
# Predict 真的拿出去用
print("Predict -->\n",dtree.predict(XTest),sep="")
print("Target -->\n",yTest.values,sep="")

# 把模型(決策樹)存出去，要存成.dot的格式，export輸出成圖形可以視覺化的方式
with open("iris_DT.dot","w") as fptr:
    fptr = tree.export_graphviz(dtree,feature_names=iris.feature_names,out_file=fptr)

# Predict 特徵非類別型，較不適合當作cross table的分析
preds=dtree.predict_proba(X=XTest)
print(pd.crosstab(preds[:,0], columns=[XTest["sepal length (cm)"],
                                     XTest["sepal width (cm)"]]))

# df=pd.crosstab(preds[:,0], columns=[XTest["sepal length (cm)"],
#                                      XTest["sepal width (cm)"]])
# df.to_html("iris_dt_crosstab.html")

# ---Draw decision tree
import pydotplus
dotData=tree.export_graphviz(dtree)
graph=pydotplus.graphviz.graph_from_dot_data(dotData)
graph.write_jpg('iris_DT.jpg')
graph.write_png('iris_DT.png')
