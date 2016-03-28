import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp

def fit_plot_ROC(features, labels, classifier, cross_val):
	"""
	Purpose: fit a classifier to training data with k-fold cross validation
	Inputs:	features: a pandas dataframe of feature values
			labels: a pandas dataframe of class labels
			classifier: an sklearn classifier e.g., sklearn.linear_model.LogisticRegression()
			cross_val: an sklearn cross-validation object 
	Outputs:	mean_auc: the mean area under the ROC curve for the different folds
				classifier: the fit classifier
				also plots the ROC curves for each fold
	"""
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []
	for i, (train, test) in enumerate(cross_val):
		probas_ = classifier.fit(features.values[train], labels.values[train]).predict_proba(features.values[test])
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(labels.values[test], probas_[:, 1])
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Chance')
	mean_tpr /= len(cross_val)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, 'k--',
		label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	return mean_auc, classifier