#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# number of features in data
print(len(features_train[0]))

#########################################################
### your code goes here ###

# create classifier
clt = tree.DecisionTreeClassifier(min_samples_split=40)

# train
t0 = time()
clt = clt.fit(features_train, labels_train)
print("Training Time: {0:.3f}".format(time()-t0))

# test
t1 = time()
pred = clt.predict(features_test)
print("Test Time: {0:.3f}".format(time()-t1))
print(pred)

# evaluate
acc = accuracy_score(labels_test, pred)
print(acc)

#########################################################
