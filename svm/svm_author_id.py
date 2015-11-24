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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time

clf = SVC(kernel='linear')

t0 = time()
clf.fit(features_train, labels_train)
print 'time to train: %s seconds' % (round(time()-t0, 3))

t1 = time()
predictions = clf.predict(features_test)
print 'time to predict: %s seconds' % (round(time()-t1, 3))

print accuracy_score(labels_test, predictions)

print '\n'


t2 = time()
clf.fit(features_train[:len(features_train)/100], labels_train[:len(labels_train)/100])
print 'time to train smaller dataset: %s seconds' % (round(time()-t2, 3))


t3 = time()
predictions = clf.predict(features_test)
print 'time to predict with smaller train data: %s seconds' % (round(time()-t3, 3))

print accuracy_score(labels_test, predictions)

print '\n'

clf_1 = SVC(kernel='rbf', C=10)
clf_2 = SVC(kernel='rbf', C=100)
clf_3 = SVC(kernel='rbf', C=1000)
clf_4 = SVC(kernel='rbf', C=10000)

t4 = time()
clf_1.fit(features_train, labels_train)
clf_2.fit(features_train[:len(features_train)/100], labels_train[:len(labels_train)/100])
clf_3.fit(features_train[:len(features_train)/100], labels_train[:len(labels_train)/100])
clf_4.fit(features_train[:len(features_train)/100], labels_train[:len(labels_train)/100])
print 'time to train smaller dataset: %s seconds' % (round(time()-t4, 3))


t5 = time()
predictions_1 = clf_1.predict(features_test)
predictions_2 = clf_2.predict(features_test)
predictions_3 = clf_3.predict(features_test)
predictions_4 = clf_4.predict(features_test)
print 'time to predict with smaller train data: %s seconds' % (round(time()-t5, 3))

print accuracy_score(labels_test, predictions_1)
print accuracy_score(labels_test, predictions_2)
print accuracy_score(labels_test, predictions_3)
print accuracy_score(labels_test, predictions_4)
#########################################################


