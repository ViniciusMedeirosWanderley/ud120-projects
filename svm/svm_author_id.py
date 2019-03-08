#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# try decrease training set
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
# your code goes here #
# test with rbf, linear kernel and with C = 10.0, 100.0, 1000.0, 10000.0
clf = SVC(C=10000.0, kernel='rbf')

# training
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time: {0:.3f}".format(time()-t0))

# testing
t1 = time()
pred = clf.predict(features_test)
print("Test Time: {0:.3f}".format(time()-t1))

# print(pred)
# print(pred[10])
# print(pred[26])
# print(pred[50])
# print(len(pred[pred == 1]))
print(accuracy_score(pred, labels_test))
#########################################################
