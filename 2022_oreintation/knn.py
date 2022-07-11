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
disp.figure_.suptitle("Confusion Matrix")
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
# C for avoid misclassify, higher value may overfitting
# gamma for consider point distance, lower may consider noise
# 那如何找出適合的gamma與Ｃ值？其實就是暴力法地毯式搜索下圖就是利用accuracy做一個gammar與Ｃ的gridserch，通常大於0.92就很不錯了。 
# classifier = svm.SVC( gamma=0.001 ) # like Binary classification( why? )
# probability for calculate predict accuracy probability, lower mean the result may be error
classifier = svm.SVC( kernel="rbf", probability=False, gamma=0.1 ) # Multiple classification
classifier.fit(train_data, train_label)

# 輸出預測結果及正確結果
predicted = classifier.predict(test_data)
print('predicted:', predicted)
print('true:     ', test_label)

# 顯示混淆矩陣
disp = metrics.ConfusionMatrixDisplay.from_predictions(test_label, predicted)
disp.figure_.suptitle("Confusion Matrix")
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
disp.figure_.suptitle("Confusion Matrix")
plt.show()

# 顯示結果報表
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# f1-score= 2 * precision * recall/(recision + recall)，為precision和recall的harmonic mean調和平均數
# support為實際手寫數字的總數
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )
# ------------------------------Logistic regression---------------------------------------
print( "--------------------Logistic regression--------------------------" )
classifier = LogisticRegression()
classifier.fit(train_data, train_label) # here get wrong
"""
D:\Program Files\Anaconda\lib\site-packages\sklearn\linear_model\_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
"""
# 輸出預測結果及正確結果
predicted = classifier.predict(test_data)
print('predicted:', predicted)
print('true:     ', test_label)

# 顯示混淆矩陣
disp = metrics.ConfusionMatrixDisplay.from_predictions(test_label, predicted)
disp.figure_.suptitle("Confusion Matrix")
plt.show()

# 顯示結果報表
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# f1-score= 2 * precision * recall/(recision + recall)，為precision和recall的harmonic mean調和平均數
# support為實際手寫數字的總數
print(f"Classification report for classifier {classifier}:\n"
   f"{metrics.classification_report(test_label, predicted)}\n"  )