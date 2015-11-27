#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics

clf_1 = KNeighborsClassifier(n_neighbors=5)
clf_1.fit(features_train, labels_train)
predictions_1 = clf_1.predict(features_test)

clf_2 = RandomForestClassifier()
clf_2.fit(features_train, labels_train)
predictions_2 = clf_2.predict(features_test)

clf_3 = AdaBoostClassifier()
clf_3.fit(features_train, labels_train)
predictions_3 = clf_3.predict(features_test)

clf_4 = SVC(kernel='rbf', C=100000)
clf_4.fit(features_train, labels_train)
predictions_4 = clf_4.predict(features_test)

print 'KNN score: %s' % metrics.accuracy_score(labels_test, predictions_1)
print 'Random Forest score: %s' % metrics.accuracy_score(labels_test, predictions_2)
print 'AdaBoost score: %s' % metrics.accuracy_score(labels_test, predictions_3)
print 'Support Vector Machine score: %s' % metrics.accuracy_score(labels_test, predictions_4)

clf = clf_4

### visualization code (prettyPicture) to show you the decision boundary








try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
