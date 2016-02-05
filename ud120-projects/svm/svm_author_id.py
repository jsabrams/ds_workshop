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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = SVC(kernel = 'rbf', C = 10000.0)			#Get the classifier
t0 = time()

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

clf.fit(features_train, labels_train)	#Fit the classifier
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)		#Make a prediction
print "prediction time:", round(time()-t0, 3), "s"

acc = accuracy_score(labels_test, pred)	#Get the accuracy
print "accuracy score: ", acc, "%"


#print "Element 10 is class: ", pred[10]
#print "Element 26 is class: ", pred[26]
#print "Element 50 is class: ", pred[50]

print "Number predicted as 1: ", sum(pred)
#########################################################


