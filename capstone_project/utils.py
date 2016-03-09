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