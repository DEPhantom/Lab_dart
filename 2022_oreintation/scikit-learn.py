# import函式庫
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

# loading data ( iris )
iris_dataset = datasets.load_iris()
iris_data = iris_dataset.data
iris_label = iris_dataset.target

# data content
# 4 features
print(pd.DataFrame(data=iris_dataset['data'], columns=iris_dataset['feature_names']))

# spilt data
train_data, test_data, train_label, test_label = train_test_split(iris_data, iris_label, test_size=0.2, shuffle=True) # test data are random

# standardization
"""
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
"""

# ------------------------------KNN---------------------------------------
print( "--------------------KNN--------------------------" )
# 使用KNN演算法，k為5
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_data,train_label)

# 輸出預測結果及正確結果
predicted = classifier.predict(test_data)
print('predicted:', predicted)
print('true:     ', test_label)

# 顯示混淆矩陣
disp = metrics.ConfusionMatrixDisplay.from_predictions(test_label, predicted)
disp.figure_.suptitle("Confusion Matrix(KNN)")
plt.show()

# 顯示結果報表
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# f1-score= 2 * precision * recall/(recision + recall)，為precision和recall的harmonic mean調和平均數
# support為實際手寫數字的總數
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )
   
print( "--------------------compare--------------------------" )
 
classifier = KNeighborsClassifier(n_neighbors=3) # k=3
classifier.fit(train_data,train_label)
predicted = classifier.predict(test_data)
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )
   
classifier = KNeighborsClassifier(n_neighbors=7) # k=7
classifier.fit(train_data,train_label)
predicted = classifier.predict(test_data)
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )
   
classifier = KNeighborsClassifier(n_neighbors=18) # k=18
classifier.fit(train_data,train_label)
predicted = classifier.predict(test_data)
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )

classifier = KNeighborsClassifier(n_neighbors=11) # k=11
classifier.fit(train_data,train_label)
predicted = classifier.predict(test_data)
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )
   
# ------------------------------SVM---------------------------------------
print( "--------------------SVM--------------------------" )
# https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72
# https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-4%E8%AC%9B-%E6%94%AF%E6%8F%B4%E5%90%91%E9%87%8F%E6%A9%9F-support-vector-machine-%E4%BB%8B%E7%B4%B9-9c6c6925856b
# C for avoid misclassify, higher value may overfitting
# gamma for consider point distance, lower may consider noise
# gamma defualt value = 1 / n_features. so more features lower value
# 那如何找出適合的gamma與Ｃ值？其實就是暴力法地毯式搜索下圖就是利用accuracy做一個gammar與Ｃ的gridserch，通常大於0.92就很不錯了。 
# classifier = svm.SVC( gamma=0.001, decision_function_shape="ovo" ) # like Binary classification
# probability for calculate predict accuracy probability, lower mean the result may be error
classifier = svm.SVC( kernel="rbf", probability=False, gamma=0.1 )
classifier.fit(train_data, train_label)
# print( classifier.decision_function( [[1, 1, 1, 1]] ).shape[1] ) check the number of class
# 輸出預測結果及正確結果
predicted = classifier.predict(test_data)
print('predicted:', predicted)
print('true:     ', test_label)

# 顯示混淆矩陣
disp = metrics.ConfusionMatrixDisplay.from_predictions(test_label, predicted)
disp.figure_.suptitle("Confusion Matrix(SVM)")
plt.show()

# 顯示結果報表
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# f1-score= 2 * precision * recall/(recision + recall)，為precision和recall的harmonic mean調和平均數
# support為實際手寫數字的總數
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )


# ------------------------------Decision Trees---------------------------------------
print( "--------------------Decision Trees--------------------------" )
classifier = DecisionTreeClassifier()
# DecisionTreeClassifier(criterion = 'entropy', max_depth=6, random_state=42)
classifier.fit(train_data, train_label)

# 輸出預測結果及正確結果
predicted = classifier.predict(test_data)
print('predicted:', predicted)
print('true:     ', test_label)

# 顯示混淆矩陣
disp = metrics.ConfusionMatrixDisplay.from_predictions(test_label, predicted)
disp.figure_.suptitle("Confusion Matrix(Tree)")
plt.show()

# 顯示結果報表
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# f1-score= 2 * precision * recall/(recision + recall)，為precision和recall的harmonic mean調和平均數
# support為實際手寫數字的總數
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )

# ------------------------------Logistic regression---------------------------------------
# https://www.youtube.com/watch?v=yIYKR4sgzI8&ab_channel=StatQuestwithJoshStarmer
print( "--------------------Logistic regression--------------------------" )
classifier = LogisticRegression( max_iter=200 ) # converge slowly
classifier.fit(train_data, train_label) # here get wrong
# 輸出預測結果及正確結果
predicted = classifier.predict(test_data)
print('predicted:', predicted)
print('true:     ', test_label)

# 顯示混淆矩陣
disp = metrics.ConfusionMatrixDisplay.from_predictions(test_label, predicted)
disp.figure_.suptitle("Confusion Matrix(Logistic regression)")
plt.show()

# 顯示結果報表
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# f1-score= 2 * precision * recall/(recision + recall)，為precision和recall的harmonic mean調和平均數
# support為實際手寫數字的總數
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )