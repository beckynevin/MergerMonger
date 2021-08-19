# Combine the predictors from two different runs and save as the largest predictor table

# Also find out how many galaxies fail due to cutouts and how many fail due to galfit/statmorph (in the _fails.txt files)


import numpy as np
import math
import astropy.io.fits as fits
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import seaborn as sns

# path
dir = '/Users/beckynevin/CfA_Code/MergerMonger/Tables/'

type_gal = 'DR12_predictors_MPI'

# first, open everything that was already in the table
df_classified = pd.io.parsers.read_csv(filepath_or_buffer='../Tables/SDSS_'+str(type_gal)+'_all.txt',header=[0],sep='\t')

df_classified.columns = ['ID','Sep','Flux Ratio',  'Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)', 'Sersic AR', 'S/N', 'Sersic N Statmorph', 'A_S Statmorph']

print(df_classified.columns, 'len classified', len(df_classified))
#print(df_classified.head())



# Match to the total SQL table


df_SQL = pd.io.parsers.read_csv(dir+'five_sigma_detection_saturated_mode1_beckynevin.csv',header=[0],sep=',')

#photo_DR16_0_5sigma_detection_beckynevin.csv

print('len df SQL',len(df_SQL))



# The condition is where galaxies that you've already run are in the SQL list, so you'll want to keep only these
cond = df_classified['ID'].isin(df_SQL['objID'])


keep = df_classified[cond]
print('length that you are keeping', len(keep))

# Now bring in the new stuff :)
df_classified_2 = pd.io.parsers.read_csv(filepath_or_buffer='../Tables/SDSS_'+str(type_gal)+'_s2_0.txt',header=[0],sep='\t')

df_classified_2.columns = ['ID','Sep','Flux Ratio',  'Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)', 'Sersic AR', 'S/N', 'Sersic N Statmorph', 'A_S Statmorph']
print(len(df_classified_2))

df_all = pd.concat([keep, df_classified_2])
print('length of all', len(df_all))

df_all = df_all.drop_duplicates(subset=['ID'])
print('length after dropping dups', len(df_all))


df_all.to_csv('../Tables/SDSS_predictors_all.csv')


# Great, but how many are lost in the fails file and why do these fail?
