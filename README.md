### MergerMonger: Identify different types of mergers in SDSS imaging
Based on the imaging classification presented in Nevin+2019 and Nevin+2022.

There are three main steps:
## 1) Creating the classification from simulations of merging galaxies:
<img src="images_for_github/panel_merger_timeline.png" alt="sims" width="700">

The simulations are fully described in Nevin+2019.

There are several variations of the same function within the MergerMonger.py code that load up the predictor tables from the simulated galaxies and run the LDA using various utilities in util_LDA.py. 
They return three things: 
1) The LDA results, which include information to standardize the predictors, the LDA coefficients, and the LD1 values of the simulated galaxies used as input, 
2) The terms selected by the random forest, and
3) The dataframe used to create the LDA

An example of how to run the classification is provided in classify_SDSS.py. Here's the relevant code:

```
from MergerMonger import load_LDA_from_simulation, classify, classify_from_flagged
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from util_LDA import cross_term
import os

run = 'major_merger'

print(str(os.getcwd())+'../frames/')

LDA,RFR, df = load_LDA_from_simulation(run,str(os.getcwd())+'../frames/', type_gal = 'predictors',name='img',verbose=False)



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

```

## 2) Measure predictor values from images (GalaxySmelter):
I also include some utilities for visualizing individual galaxies and their predictor values.
<img src="images_for_github/prob_panel_low.png" alt="probability panel" width="700">

## 3) Classify galaxies and obtain merger probability values:
<img src="images_for_github/p_merg_recipe.png" alt="walkthrough" width="700">

I also include some utilities for interpreting these probability values using the CDF of the full population.
