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


#df_all.to_csv('../Tables/SDSS_predictors_all.txt',sep='\t')


# Great, but how many are lost in the fails file and why do these fail?

# The fails are stored somewhere different, first get the fails from the original run, then get the selection_2 fails
prefix = '/Users/beckynevin/CfA_Code/Cannon/parallel_SDSS/'
type_gal = 'DR12_predictors_MPI'
df_fails_0 = pd.io.parsers.read_csv(prefix+'SDSS_'+str(type_gal)+'_fails_0.txt', sep='\t',dtype={'ID':str})
df_fails_0.columns = ['ID','Reason']

df_fails_1 = pd.io.parsers.read_csv(prefix+'SDSS_'+str(type_gal)+'_fails_1.txt', sep='\t',dtype={'ID':str})
df_fails_1.columns = ['ID','Reason']

df_fails_2 = pd.io.parsers.read_csv(prefix+'SDSS_'+str(type_gal)+'_fails_2.txt', sep='\t',dtype={'ID':str})
df_fails_2.columns = ['ID','Reason']

df_fails_3 = pd.io.parsers.read_csv(prefix+'SDSS_'+str(type_gal)+'_fails_3.txt', sep='\t',dtype={'ID':str})
df_fails_3.columns = ['ID','Reason']

df_fails_4 = pd.io.parsers.read_csv(prefix+'SDSS_'+str(type_gal)+'_fails_4.txt', sep='\t',dtype={'ID':str})
df_fails_4.columns = ['ID','Reason']

df_fails_5 = pd.io.parsers.read_csv(prefix+'SDSS_'+str(type_gal)+'_fails_5.txt', sep='\t',dtype={'ID':str})
df_fails_5.columns = ['ID','Reason']

df_fails_6 =  pd.io.parsers.read_csv('../Tables/SDSS_DR12_predictors_MPI_s2_fails_0.txt', sep='\t',dtype={'ID':str})
df_fails_6.columns = ['ID','Reason']




# Put these all together and delete repeats, then delete things that aren't in the overall photo...csv file
df_fails_01 = df_fails_0.append(df_fails_1)
df_fails_23 = df_fails_2.append(df_fails_3)
df_fails_45 = df_fails_4.append(df_fails_5)
df_fails_456 = df_fails_45.append(df_fails_6)

df_fails_0123 = df_fails_01.append(df_fails_23)

df_fails_all = df_fails_0123.append(df_fails_456)
  

# Now delete all duplicates in here
# Now drow all duplicates of the first column
df_fails_nodup = df_fails_all.drop_duplicates(subset=['ID'],keep='first')



# Drop all rows that have the header:
df_fails_all_nodup = df_fails_nodup[df_fails_nodup.ID != 'ID']
print('len fails after dropping duplicate IDs and headers', len(df_fails_all_nodup))



# Now drop anything that is not in the photo list
cond = df_fails_all_nodup['ID'].isin(df_SQL['objID'])

keep_fails = df_fails_all_nodup[cond]

print('length of fails that you are keeping', len(keep_fails))

# And now drop anything from fails that is already in df_all:
cond = keep_fails['ID'].isin(df_all['ID'])


double_keep_fails = keep_fails[~cond]
print('len of fails after eliminating sources already measured', len(double_keep_fails))

print('total length of SQL', len(df_SQL), 'length of fails and keeps', len(double_keep_fails),len(df_all),len(double_keep_fails)+len(df_all))
print(double_keep_fails['Reason'].value_counts())

double_keep_fails['Reason'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()


STOP
