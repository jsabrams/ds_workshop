import numpy as np
import pandas as pd
import model_fitting as mfit
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.externals import joblib

def train_ensemble():
	"""
	Purpose: Train the stacked ensemble for the lending club data set
	Inputs: none, loads all necessary files
	Outputs: saves all files for the classifier
	"""
	X_train = pd.read_pickle('X_train')
	y_train = pd.read_pickle('y_train')

	reliable_feat_no_amnt_intr = mfit.get_reliable_features(X_train.drop(['int_rate', 'loan_amnt'], axis = 1), y_train)
	cv = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=42)

	DT_entropy_clf = DecisionTreeClassifier(max_depth = 3, class_weight = {0: (1.0 - 0.1841), 1: 0.1841},
		criterion = 'entropy')
	DT_gini_clf = DecisionTreeClassifier(max_depth = 3, class_weight = {0: (1.0 - 0.1841), 1: 0.1841},
		criterion = 'gini')

	clfs = [RandomForestClassifier(n_estimators=100, 
									min_samples_leaf = 5, 
									class_weight = {0: (1.0 - 0.1841), 1: 0.1841},
									criterion = 'gini'),
		RandomForestClassifier(n_estimators=100,
								min_samples_leaf = 5,
								class_weight = {0: (1.0 - 0.1841), 1: 0.1841},
								criterion = 'entropy'),
		ExtraTreesClassifier(n_estimators=100,
								min_samples_leaf = 5,
								class_weight = {0: (1.0 - 0.1841), 1: 0.1841},
								n_jobs = -1,
								criterion = 'gini'),
		ExtraTreesClassifier(n_estimators=100,
								min_samples_leaf = 5,
								class_weight = {0: (1.0 - 0.1841), 1: 0.1841},
								criterion = 'entropy'),
		GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
		AdaBoostClassifier(base_estimator=DT_entropy_clf, n_estimators=100, random_state=42),
		AdaBoostClassifier(base_estimator=DT_gini_clf, n_estimators=100, random_state=42)]

	clf_2 = LogisticRegression(penalty='l2', C=1, class_weight = {0: (1.0 - 0.1841), 1: 0.1841})
	clfs, clf_2 = mfit.fit_stacked_clf(clfs, clf_2, cv, X_train[reliable_feat_no_amnt_intr[0:43].index], y_train)

	joblib.dump(clf_2, 'clf_stack_2.pkl')
	joblib.dump(clfs[0], 'clf0_stack_1.pkl')
	joblib.dump(clfs[1], 'clf1_stack_1.pkl')
	joblib.dump(clfs[2], 'clf2_stack_1.pkl')
	joblib.dump(clfs[3], 'clf3_stack_1.pkl')
	joblib.dump(clfs[4], 'clf4_stack_1.pkl')
	joblib.dump(clfs[5], 'clf5_stack_1.pkl')
	joblib.dump(clfs[6], 'clf6_stack_1.pkl')