import pandas as pd
import numpy as np

def recject_loans(p_default, loans, reject_p = 0.95):
	"""
	Purpose: "reject" items from loans
	Inputs:	p_default: a pandas series of default probabilities for
			each item in loans
			oans: pandas data frames containing the loan data
			reject_p: a percentile at which to reject loans
	"""
	loans['p_default'] = p_default
	quant = loans.p_default.quantile(q = reject_p)
	return loans[loans.p_default <= quant]

def random_reject(ROI, col, n_accept, lo_ci = 2.5, hi_ci = 97.5, reps = 1000):
	"""
	Purpose: reject random selection of loans to predict chance performance (median ROI)
	Inputs:	ROI: a pandas dataframe to randomly sample
			col: the name of the column to get the median values from
			n_accept: the number of items to accept from the dataframe
			lo_ci: a lower confidence interval bound for the bootstrapped distribution
			hi_ci: upper confidence interval bound
			reps: number of bootstrapping repetitions
	Outputs:	mean_boot: the mean of the bootstrapped distribution of median values of col
				prctile: the low and high values for the confidence interval
	"""
	boot = [ROI.sample(n = n_accept, random_state = 42)[col].median() for i in range(reps)]
	prctile = np.percentile(boot, [lo_ci, hi_ci])
	boot_mean = np.mean(boot)
	return boot_mean, prctile

def perfect_reject(ROI, rej_col, est_col, val):
	"""
	Purpose: median performance if only a specific values were accepted
	Inputs:	ROI: a pandas dataframe
			rej_col: the column name for the dataframe to accept/reject based on
			est_col: the column name for the median measurement
			val: the value to accept from rej_col
	Output:	median performance based on the accepted value
	"""
	return ROI[ROI[rej_col] == val][est_col].median()


def dollar_value(roi, base = 10000):
	"""
	Purpose: return the profit made off a base investment
	Inputs:	roi: return on investment
			base: the base value in dollars to consider returns from
	Output:	the profit (or loss) on a portfolio
	"""
	return roi * base

