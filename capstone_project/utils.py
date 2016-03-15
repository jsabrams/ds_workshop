import numpy as np
import pandas as pd

def med_diff_bootstrap(a, b, hi_ci, lo_ci, reps = 10000):
	"""
	Purpose: return a confidence interval on the difference in the medians of a and b
	Inputs: a: pandas series a
			b: pandas series b
			hi_ci: the upper percentile of the difference distribution desired (0-100)
			lo_ci: the lower percentile of the difference distribution desired (0-100)
			reps: the number of bootstrap repetitions (default is 10,000)
	Output: a list with the lower and uper bounds specified
	"""
	a_boot = [a[a.index[np.random.randint(0,len(a),len(a))]].median() for i in range(reps)]
	b_boot = [b[b.index[np.random.randint(0,len(b),len(b))]].median() for i in range(reps)]
	diff = np.array(a_boot) - np.array(b_boot)
	return np.percentile(diff, [lo_ci, hi_ci])

def mean_diff_bootstrap(a, b, hi_ci, lo_ci, reps = 10000):
	"""
	Purpose: return a confidence interval on the difference in the means of a and b
	Inputs: a: pandas series a
			b: pandas series b
			hi_ci: the upper percentile of the difference distribution desired (0-100)
			lo_ci: the lower percentile of the difference distribution desired (0-100)
			reps: the number of bootstrap repetitions (default is 10,000)
	Output: a list with the lower and uper bounds specified
	"""
	a_boot = [a[a.index[np.random.randint(0,len(a),len(a))]].mean() for i in range(reps)]
	b_boot = [b[b.index[np.random.randint(0,len(b),len(b))]].mean() for i in range(reps)]
	diff = np.array(a_boot) - np.array(b_boot)
	return np.percentile(diff, [lo_ci, hi_ci])

def mean_bootstrap(a, hi_ci, lo_ci, reps = 10000):
	"""
	Purpose: return a confidence interval on the mean of a
	Inputs: a: pandas series a
			hi_ci: the upper percentile of the distribution desired (0-100)
			lo_ci: the lower percentile of the distribution desired (0-100)
			reps: the number of bootstrap repetitions (default is 10,000)
	Output: a list with the lower and uper bounds specified
	"""
	a_boot = [a[a.index[np.random.randint(0,len(a),len(a))]].mean() for i in range(reps)]
	return np.percentile(a_boot, [lo_ci, hi_ci])

def zipcode_processor(df, fields):
	"""
	Purpose: get aggregated zipcode info with the last two numbers anonymized
	Inputs:	df: original zipcode dataframe with full five digit codes
			fields: the fields in df to aggregate (become new columns)
	Output:	a new dataframe with the data aggregated for fields
	"""
	codes = np.floor(df.zipcd/100).unique()
	df_new = {}
	for field in fields:
		vals = [df[np.floor(df.zipcd/100) == x][field].sum() for x in codes]
		ser = pd.Series(data = vals, index = codes)
		df_new[field] = ser
	return pd.DataFrame(df_new)
