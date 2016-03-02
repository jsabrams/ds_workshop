#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import train_test_split
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print 'The accuracy score is ', accuracy_score(y_test, y_pred)
print 'The predicted number of predicted POIs is ', sum(y_pred)
print 'The number of people in the test set is', len(y_pred)
print 'The number of true positives is ', sum([x if pv == 1 and tv == 1 else 0 
	for (pv, tv) in zip(y_pred, y_test)])
print 'The precision score is ', precision_score(y_test, y_pred)
print 'The recall score is ', recall_score(y_test, y_pred)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print precision_score(true_labels, predictions)
print recall_score(true_labels, predictions)

