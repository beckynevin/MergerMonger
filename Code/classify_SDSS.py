'''
~~~
The wrapper for creating the LDA_out tables for the fully SDSS dataset
~~~
'''
from MergerMonger import load_LDA_from_simulation, classify
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from util_LDA import cross_term

run = 'major_merger'
LDA = load_LDA_from_simulation(run, verbose=False)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Output from LDA~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(LDA)
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

print('coefficients', LDA[3])
print('intercept', LDA[4])

print('len intercept', len(LDA[4]))






plt.clf()
plt.hist(LDA[8], bins=50, alpha=0.5)

plt.xlabel("LD1")
plt.axvline(x=0)

plt.title('LDA values from the simulation')
plt.show()
    
    

type_gal = 'predictors'
verbose='yes'


classify('../Tables/',type_gal, run, LDA)
    
