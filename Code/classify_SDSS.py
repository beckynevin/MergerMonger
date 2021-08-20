'''
~~~
Iteratively run MergerMonger on the full SDSS sample to make the figure: 'adjusting_prior_all.png'
The goal is to be able to adjust the prior without adjusting the alphas to determine how much the choice of input priors affects the measured merger fraction.
~~~
'''
from MergerMonger import load_LDA_from_simulation
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from util_LDA import cross_term

run = 'major_merger'
LDA = load_LDA_from_simulation(run)
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

input_singular = []
crossterms = []
ct_1 = []
ct_2 = []
for j in range(len(LDA[2])):
    if '*' in LDA[2][j]:
        crossterms.append(LDA[2][j])
        split = str.split(LDA[2][j],'*')
        ct_1.append(split[0])
        ct_2.append(split[1])
        
    else:
        input_singular.append(LDA[2][j])
        





plt.clf()
plt.hist(LDA[8], bins=50, alpha=0.5)

plt.xlabel("LD1")
plt.axvline(x=0)

plt.title('LDA values from the simulation')
plt.show()
    
    

type_gal = 'predictors'
verbose='yes'


classify(prefix,type_gal)
    
