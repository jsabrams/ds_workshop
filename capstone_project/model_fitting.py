import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from scipy import interp

def fit_plot_ROC(features, labels, classifier, cross_val):
	"""
	Purpose: fit a classifier to training data with k-fold cross validation, plot ROC curves for each fold
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

def fit_plot_PR(features, labels, classifier, cross_val):
	"""
	Purpose: fit a classifier to training data with k-fold cross validation, plot precision-recall curves for each fold
	Inputs:	features: a pandas dataframe of feature values
			labels: a pandas dataframe of class labels
			classifier: an sklearn classifier e.g., sklearn.linear_model.LogisticRegression()
			cross_val: an sklearn cross-validation object 
	Outputs:	classifier: the fit classifier
				also plots the PR curves for each fold
	"""
	mean_precision = 0.0
	mean_recall = np.linspace(0, 1, 100)
	for i, (train, test) in enumerate(cross_val):
		probas_ = classifier.fit(features.values[train], labels.values[train]).predict_proba(features.values[test])
		precision, recall, _ = precision_recall_curve(labels.values[test], probas_[:, 1])
		mean_precision += interp(mean_recall, recall[::-1], precision[::-1])
		mean_precision[0] = 1.0
		precision_score = average_precision_score(labels.values[test], probas_[:, 1], average='micro')
		plt.plot(recall, precision, lw=1, label='P-R fold %d (area = %0.2f)' % (i, precision_score))
	mean_precision /= len(cross_val)
	mean_auc = auc(mean_recall, mean_precision)
	plt.plot(mean_recall, mean_precision, 'k--', label='Mean P-R (area = %0.2f)' % mean_auc, lw=2)
	plt.legend(loc="upper right")
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	return mean_auc, classifier