from sklearn import datasets
from sklearn import naive_bayes as NB
import numpy as np
iris = datasets.load_iris()
dataset = iris.data
label = iris.target
trainNum = range(len(dataset))
testNum = []
for i in range(50):
    randIndex = int(np.random.uniform(0, len(trainNum)))
    testNum.append(trainNum[randIndex])
    del(trainNum[randIndex])
X_train = []; Y_train = []
X_test = []; Y_test = []
for i in trainNum:
    X_train.append(dataset[i])
    Y_train.append(label[i])
for i in testNum:
    X_test.append(dataset[i])
    Y_test.append(label[i])

gnb = NB.GaussianNB()
y_predict = gnb.fit(X_train, Y_train).predict(X_test)
print(y_predict)
print(Y_test)
print ((y_predict != Y_test).sum())

