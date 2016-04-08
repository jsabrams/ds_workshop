import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
	boot = [ROI.sample(n = n_accept)[col].median() for i in range(reps)]
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

def model_perf_dollars(ROI, p_default, p_rej, rej_col, est_col, val, b = 10000, do_plot = True):
	"""
	Purpose: calculate loan profitability based upon predicted default probability and rejection criteria
	Inputs:	ROI: a pandas dataframe
			p_default: a pandas series of default probabilities for
			each item in ROI
			p_rej: a list of percentiles at which to reject loans
			rej_col: the column name for the dataframe to accept/reject based on
			est_col: the column name for the median measurement
			val: the value to accept from rej_col
			b: the base value in dollars to consider returns from
			do_plot: whether or not to plot the results
	Output:	val_frame: a dataframe with loan profits based on model-based rejection, random rejection, and pefect rejection
	"""
	rej_value = []
	rnd_value = []
	hi_value = []
	lo_value = []
	perf_roi = perfect_reject(ROI, rej_col, est_col, val)	#Calculate roi for perfect rejection
	perf_value = dollar_value(perf_roi, base = b)			#Calculate profit for perfect rejection
	
	#Get model-based and random rejection profitability over the rejection percentiles
	for p in p_rej:
		rej_ROI = recject_loans(p_default, ROI, reject_p = 1-p)
		rnd_ROI, prctile = random_reject(ROI, est_col, len(rej_ROI))
		rej_value.append(dollar_value(rej_ROI.roi.median(), base = b))
		rnd_value.append(dollar_value(rnd_ROI, base = b))
		hi_value.append(dollar_value(prctile[1], base = b))
		lo_value.append(dollar_value(prctile[0], base = b))

	#Create dataframe to output
	rej_value = pd.Series(data = rej_value, index = p_rej)
	rnd_value = pd.Series(data = rnd_value, index = p_rej)
	hi_value =  pd.Series(data = hi_value, index = p_rej)
	lo_value =  pd.Series(data = lo_value, index = p_rej)
	val_frame = pd.DataFrame()
	val_frame['reject'] = rej_value
	val_frame['random'] = rnd_value
	val_frame['random_hi_ci'] = hi_value
	val_frame['random_lo_ci'] = lo_value
	val_frame['perfect'] = perf_value

	#Make a plot
	if do_plot:
		plt.figure()
		plt.errorbar(val_frame.index, val_frame.random.values, 
			yerr=[val_frame.random.values - val_frame.random_lo_ci.values, 
			val_frame.random_hi_ci.values - val_frame.random.values], label = 'Random Rejection')
		plt.plot(val_frame.index, val_frame.reject.values, label = 'Model-Based Rejection')
		plt.title("Profit on $10,000 investment")
		plt.xlabel("Rejection Rate (%)")
		plt.ylabel("Profit ($)")
		plt.legend(loc='upper right')
		plt.show()

	return val_frame