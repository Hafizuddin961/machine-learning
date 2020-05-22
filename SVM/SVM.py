import sklearn
from sklearn import datasets
from sklearn import svm

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
Y = cancer.target

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y ,test_size = 0.1)

print(X_train,Y_train)
print(len(cancer.feature_names))

svm.LinearSVR