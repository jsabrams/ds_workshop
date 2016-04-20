import pandas as pd
import numpy as np
import json
from sklearn.externals import joblib

def get_files(scaler_fn, feat_fn, clf_fn):
	"""
	Purpose: get files to do prediction
	Inputs:	scaler_fn: string filename for the sklearn scaler
			feat_fn: string filename for the pandas dataframe of reliable features
			clf_fn: string filename for the sklearn classifier
	Outputs: 	scaler: the scaler tranformer
				reliable_feat: the dataframe of reliable features
				clf: the sklearn classifier
				cont: a list of the continuous variable names
				cat: a list of the categorical variable names
	"""
	scaler = joblib.load(scaler_fn)
	reliable_feat = pd.read_pickle(feat_fn)
	clf = joblib.load(clf_fn)
	with open("continuous_feats.txt") as f: cont = json.load(f)
	with open("categorical_feats.txt") as f: cat = json.load(f)

	return scaler, reliable_feat, clf, cont, cat

def organize_data(X, cont, cat, scaler, reliable_feat):
	"""
	Purpose: scale and organize the data
	Inputs:	X: the dataframe containing the data for prediction
			cont: a list of the continuous variables
			cat: a list of the categorical variables
			reliable_feat: a list of the reliable features
	"""
	X_cont = X[cont]
	X_cat = X[cat]
	X_cont = scaler.transform(X_cont.values)
	dat = np.hstack((X_cont, X_cat.values))
	dat = pd.DataFrame(data = X, index = X.index, columns = X.columns)

	return dat[reliable_feat.index]

def predict_default_tree(X):
	"""
	Purpose: predict probability of loan default using decision tree
	Inputs: X: a pandas dataframe of loan data 
	Output: probability of default
	"""
	scaler, reliable_feat, clf, cont, cat = get_files('scaler.pkl', 'DT_reliable_feat', 'DT_clf.pkl')
	X = organize_data(X, cont, cat, scaler, reliable_feat)

	return clf.predict_proba(X)[:,1]

def predict_default_ensemble(X):
	"""
	Purpose: predict probability of loan default using a stacked ensemble
	Inputs: X: a pandas dataframe of loan data 
	Output: probability of default
	"""

	scaler, reliable_feat, clf_2, cont, cat = get_files('scaler.pkl', 'EN_reliable_feat', 'clf_stack_2.pkl')
	clfs = [joblib.load('clf0_stack_1.pkl'),  joblib.load('clf1_stack_1.pkl'), joblib.load('clf2_stack_1.pkl'),
		joblib.load('clf3_stack_1.pkl'), joblib.load('clf4_stack_1.pkl'), 
		joblib.load('clf5_stack_1.pkl'), joblib.load('clf6_stack_1.pkl')]
	X = organize_data(X, cont, cat, scaler, reliable_feat)

	test_prob_est = np.zeros((X.shape[0], len(clfs)))	#Empty array for probability estimates
	print 'Estimating probabilities for test set'
	for i, clf in enumerate(clfs): 						#For each stage-one classifier
		print 'Classifier', i
		test_prob_est[:, i] = clf.predict_proba(X)[:,1]	#Predict the class probability for the test data

	return clf_2.predict_proba(test_prob_est)[:,1]