import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')

data = data[['G1','G2','G3','studytime','failures','absences',]]

predict = 'G3'


X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y ,test_size = 0.1)

best = 0
for _ in range(30):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y ,test_size = 0.1)

    linear = linear_model.LinearRegression()
    train = linear.fit(X_train, Y_train)
    acc = linear.score(X_test, Y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
best = linear.score(X_test, Y_test)


print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print (predictions[x], X_test[x], Y_test[x])

inpredict = linear.predict([[15,15,3,0,0]])
print('prediction = ')
print (inpredict)

#print(X_test)

print(best)