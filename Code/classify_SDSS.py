'''
~~~
The wrapper for creating the LDA_out tables for the fully SDSS dataset
~~~
'''
import MergerMonger as MM
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from util_LDA import cross_term
import os

run = 'major_merger_late'
#run = 'minor_merger_postc_include_coal_1.0'
#run = 'major_merger_late'
verbose = True

if verbose:
	print(str(os.getcwd())+'../frames/')

LDA,RFR, df = MM.load_LDA_from_simulation(run,verbose=verbose)
#LDA, RFR, df = MM.load_LDA_from_simulation_sliding_time(0.5, run_parent, verbose=verbose)
#LDA, RFR, df = MM.load_LDA_from_simulation_sliding_time_include_coal(1.0, run_parent, verbose=verbose)

if verbose:
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Output from LDA~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print('inputs', LDA[2])
	print('coefficients', LDA[3])
	print('intercept', LDA[4])
	print('accuracy, precision, and recall for simulated galaxies [5-7]', LDA[5], LDA[6], LDA[7])

	print('Standardized means LDA[0]', LDA[0])
	print('standardized stds LDA[1]', LDA[1])
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print('~~~~~~~~~~~~~~~~~Output from RFR~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print(RFR)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


# The output of this is in the format:
# 0 = standardized means on all of the coefficients
# 1 = standardized stds
# 2 = inputs
# 3 = coefficients
# 4 = intercept
# 5 = Accuracy
# 6 = Precision
# 7 = Recall
# 8 = LDA values of all simulated galaxies
# 9 = myr
# 10 = myr_non
# 11 = covariance matrix
# 12 = means of all classes







'''
plt.clf()
plt.hist(LDA[8], bins=50, alpha=0.5)

plt.xlabel("LD1")
plt.axvline(x=0)

plt.title('LDA values from the simulation')
plt.show()
''' 
    

type_gal = 'predictors'
verbose='yes'

#LDA, p_merg, CDF = MM.classify_from_flagged_interpretive_table('../Tables/','../frames/', run, LDA, RFR, df, 100, verbose=True, all = True, cut_flagged = False)

LDA, p_merg, CDF = MM.classify_from_flagged('../Tables/','../frames/', run, LDA, RFR, df, 10000, 
	verbose=True, all = True, cut_flagged = False)

#LDA, p_merg, CDF = classify('../Tables/','../frames',type_gal, run, LDA, RFR, df, verbose=False)
    

# Now plot the LDA values from the classified galaxies
plt.clf()
plt.hist(LDA, bins=50, alpha=0.5)

plt.xlabel("LD1")
plt.axvline(x=0)

plt.title('LDA values from the classified galaxies')
plt.show()
