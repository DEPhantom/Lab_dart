import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# 0) data import and preprocessing
from sklearn import datasets

iris = datasets.load_iris()

# use pandas as dataframe and merge features and targetsf
#我們的資料有三個種類，且有四種不同的參數特徵
#為了方便示範，就取兩個類別，且只留兩種特徵做判斷
feature = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=['target'])
iris_data = pd.concat([feature, target], axis=1)

# keep only sepal length in cm, sepal width in cm and target
iris_data = iris_data[['sepal length (cm)', 'sepal width (cm)', 'target']]

# keep only Iris-Setosa and Iris-Versicolour classes
iris_data = iris_data[iris_data.target <= 1]
iris_data.head(5)

feature = iris_data[['sepal length (cm)', 'sepal width (cm)']]
target = iris_data[['target']]

n_samples, n_features = feature.shape

# split training data and testing data
from sklearn.model_selection import train_test_split

feature_train, feature_test, target_train, target_test = train_test_split(
    feature, target, test_size=0.3, random_state=4
)

from sklearn.preprocessing import StandardScaler

#對資料做特徵的縮放
sc = StandardScaler()
feature_train = sc.fit_transform(feature_train)
feature_test = sc.fit_transform(feature_test)
target_train = np.array(target_train)
target_test = np.array(target_test)

# change data to torch
feature_train = torch.from_numpy(feature_train.astype(np.float32))
feature_test = torch.from_numpy(feature_test.astype(np.float32))
target_train = torch.from_numpy(target_train.astype(np.float32))
target_test = torch.from_numpy(target_test.astype(np.float32))

# 1) model build
# sigmoid function 交給了 torch.sigmoid
# Logistic Regression 跟 Linear Regression 並沒有太大的差別，主要差別就差在多了一個 sigmoid function，而 Pytorch 也有提供 sigmoid function
# forward() 裡面說明了我們實際上是先做一次 linear 計算，之後才做 sigmoid，這也代表著如果今天遇到一個未知網路，我們都可以裡用撰寫者寫的 forward() function 去理解這個神經網路是如何傳遞資料的，因此整個 Model 最重要的可以說是這個部份了
class DNN(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(DNN, self).__init__()
        # define first layer
        self.l1 = nn.Linear(input_size, hidden1_size)
        # activation function
        self.relu = nn.ReLU()
        # define second layer
        self.l2 = nn.Linear(hidden1_size, hidden2_size)
        
        self.relu = nn.ReLU()
        # define second layer
        self.l3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out


model = DNN(input_size, hidden1_size, hidden2_size, num_classes)

# 2) loss and optimizer
learning_rate = 0.01
# BCE stands for Binary Cross Entropy
criterion = nn.BCELoss()
# SGD stands for stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
epochs = 100
for epoch in range(epochs):
    # forward pass and loss
    y_predicted = model(feature_train)
    loss = criterion(y_predicted, target_train)

    # backward pass
    loss.backward()

    # optimizer
    optimizer.step()

    # init optimizer
    optimizer.zero_grad()

    if (True):
        print(f'epoch {epoch + 1}: loss = {loss:.8f}')

# checking testing accuracy
with torch.no_grad():
    y_predicted = model(feature_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(target_test).sum() / float(target_test.shape[0])
    print(f'accuracy = {acc: .4f}')