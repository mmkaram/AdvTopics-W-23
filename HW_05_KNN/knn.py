# Mahdy Karam
# EPS TICS/AIML Class 13
#
# Sample program for k nearest neighbors with a iris dataset
#Dataset
from sklearn.datasets import load_iris
# things to bring in for classifying
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
# make pretty visualizations
import matplotlib as plt
import pandas as pd
import seaborn as sns
# load in my dataset
iris = load_iris()
# print info
print(iris.DESCR)
print(iris.data.shape)
print(iris.data[13])
print(iris.data[13])
# Visualize the data (skip)
# Split the dataset for training and testing
# X values are samples
# y values are targets (labels)
# random state which is a seed
# 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(iris.data,
iris.target,
random_state=11)
#Creating and training the model
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)
#Actually going to predict on the last 25%
predicted = knn.predict(X=X_test)
expected = y_test
# print out the predictions and actual answers for the first 20
print("predicted", predicted[:20])
print("expected", expected[:20])
# print out the score (formatted) for how well we did
print(f'{knn.score(X_test,y_test):.2%}')
# print the confusion matrix
# a value at row y column x indicates how many ys were classified as xs
confusion = confusion_matrix(y_true=expected, y_pred=predicted)
print(confusion)
