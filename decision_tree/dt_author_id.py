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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
print 'Features count: %s' % len(features_train[0])

from sklearn import tree, metrics
from time import time

clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print 'time to train: %s seconds' % round(time()-t0, 3)

t1 = time()
predictions = clf.predict(features_test)
print 'time to predict: %s seconds' % round(time()-t1, 3)

print metrics.accuracy_score(labels_test, predictions)

#########################################################


