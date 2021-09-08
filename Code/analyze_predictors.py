#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loads in predictor table and LDA table and SQL and 
# allows me to look more closely at individual galaxies
# with weird predictor values, even download them and
# rerun all of smelter to see why the asymmetry values 
# and other predictor values are strange.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import numpy as np
import math
import astropy.io.fits as fits
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from util_SDSS import SDSS_objid_to_values, download_galaxy
from util_smelter import get_predictors
import scipy.stats
# path
dir = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger/Tables/'

run = 'major_merger'
type_gal = 'predictors'

# zeroth, look at the predictor value mins and maxes from the simulations:
df_predictors_sim_major = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_merged_major_merger.txt', header=[0], sep='\t')

df_predictors_sim_minor = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_merged_minor_merger.txt', header=[0], sep='\t')

df_predictors_sim = pd.concat([df_predictors_sim_major, df_predictors_sim_minor])

print(df_predictors_sim['Asymmetry (A)'])
plt.hist(df_predictors_sim['Asymmetry (A)'].values, bins=100)
plt.show()

predictor_list = ['Gini','M20','Concentration (C)', 'Asymmetry (A)', 'Clumpiness (S)', 'Sersic N', 'Shape Asymmetry (A_S)']

predictor_dict = {'Gini':[], 'M20':[], 'Concentration (C)':[], 'Asymmetry (A)':[], 'Clumpiness (S)':[], 'Sersic N':[], 'Shape Asymmetry (A_S)':[]}

for p in predictor_list:
    print('Predictor', p)
    vals = df_predictors_sim[p].values
    print('min, max')
    print(np.min(vals), np.max(vals))

    predictor_dict[p].append(np.min(vals))
    predictor_dict[p].append(np.max(vals))
    '''
    print('What about defining this using standard deviation?')
    print('3 sigma')
    print(np.mean(vals) - 3*np.std(vals), np.mean(vals) + 3*np.std(vals))
    print('5 sigma')
    print(np.mean(vals) - 5*np.std(vals), np.mean(vals) + 5*np.std(vals))
    '''
# Okay, now the next step is to find out how many of the predictor values fall outside of this range



# first, open everything that was already in the table
df_predictors = pd.io.parsers.read_csv(filepath_or_buffer=dir+'SDSS_'+str(type_gal)+'_all.txt',header=[0],sep='\t')

#df_classified.columns = ['ID','Sep','Flux Ratio',  'Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)', 'Sersic AR', 'S/N', 'Sersic N Statmorph', 'A_S Statmorph']

# Match to the total SQL table
df_SQL = pd.io.parsers.read_csv(dir+'five_sigma_detection_saturated_mode1_beckynevin.csv',header=[0],sep=',')

# try to plot these, first merge on SQL:                                                                   
df_SQL_predictors = df_SQL.merge(df_predictors, left_on='objID', right_on='ID')
print('len SQL', len(df_SQL), 'len preds', len(df_predictors), 'len merged', len(df_SQL_predictors))


# Also load up that old LDA table and merge it also:                                                       
df_LDA = pd.io.parsers.read_csv(filepath_or_buffer='../Tables/LDA_out_all_SDSS_'+str(type_gal)+'_'+str(run)+'.txt',header=[0],sep='\t')
print(len(df_LDA))

df_all = df_SQL_predictors.merge(df_LDA, on='ID')
print('len all merged', len(df_all))


# Okay look at weird predictor values
#plt.clf()
#plt.hist(df_all['Asymmetry (A)'].values, bins=1000)
#plt.show()

print('length less than zero', len(df_all[df_all['Asymmetry (A)'] < 0]))

df_all_A_cut = df_all[df_all['Asymmetry (A)'] > -1000]
df_all_cut = df_all_A_cut[df_all_A_cut['S/N'] < 20]

y = 'S/N'
x = 'Asymmetry (A)'


'''
heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df_all_cut[y].values, df_all_cut[x].values, df_all_cut['LD1'].values,statistic='mean', bins=100)


xmin = xedges[0]
xmax = xedges[-1]

ymin = yedges[0]
ymax = yedges[-1]

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.flipud(heatmap), cmap='magma_r',extent=[ymin, ymax, xmin, xmax], norm=matplotlib.colors.LogNorm())
ax.set_aspect((ymax-ymin)/(xmax-xmin))
plt.colorbar(im, label='LD1')
ax.set_xlabel(x)
ax.set_ylabel(y)
plt.show()
'''

# Investigate the correlation:                                                                                        
# Pearson assumes linear relationship:
from scipy.stats import pearsonr


corr, _= pearsonr(df_all_cut[x].values, df_all_cut['LD1'].values)
print('Pearsons correlation for A and LD1: %.3f' % corr)
corr, _= pearsonr(df_all_cut[y].values, df_all_cut[x].values)
print('Pearsons correlation for S/N and A: %.3f' % corr)
print('\n')

# Now iteratively do this for different A cutoffs:
'''
cutoffs = [-10,-5,-1,-0.5,-0.1,0]

for c in cutoffs:
    print('all of this for when A is less than %.1f' % c)
    df_lt_0 = df_all_cut[df_all_cut['Asymmetry (A)'] < c]
    corr, _= pearsonr(df_lt_0[x].values, df_lt_0['LD1'].values)
    print('Pearsons correlation for A and LD1: %.3f' % corr)
    corr, _= pearsonr(df_lt_0[y].values, df_lt_0[x].values)
    print('Pearsons correlation for S/N and A: %.3f' % corr)
    print('\n')
    print('all of this for when A is greater than zero')
    df_gt_0 = df_all_cut[df_all_cut['Asymmetry (A)'] > c]
    corr, _= pearsonr(df_gt_0[x].values, df_gt_0['LD1'].values)
    print('Pearsons correlation for A and LD1: %.3f' % corr)
    corr, _= pearsonr(df_gt_0[y].values, df_gt_0[x].values)
    print('Pearsons correlation for S/N and A: %.3f' % corr)
    print('\n')


plt.clf()
plt.scatter(df_all_cut[y].values, df_all_cut[x].values, c= df_all_cut['LD1'].values, s=0.1, vmin=-100, vmax=100)
plt.xlabel('Average S/N')
plt.ylabel('Asymmetry')
plt.axvline(x=2.5, color='black')
#plt.axhline(y=-1, color='grey')
#plt.axhline(y=0, color='grey')
plt.colorbar(label='LD1')

plt.show()


STOP

plt.clf()
plt.scatter(df_all_cut[x].values, df_all_cut['LD1'].values, c= df_all_cut[y].values, s=0.1)
plt.xlabel('Asymmetry')
plt.ylabel('LD1')
plt.colorbar(label='S/N')
plt.show()



STOP

'''


#plt.scatter(df_all['Sersic N'].values, df_all['Asymmetry (A)'].values, c=df_all['LD1'].values)
#plt.colorbar()
#plt.show()


cutoff = 2.5

print('starting size', len(df_all))
df_SN_low = df_all[(df_all['S/N'] < cutoff) & (df_all['S/N'] > 0)]
df_SN_high = df_all[(df_all['S/N'] > cutoff) & (df_all['S/N'] > 0)]

df_SN_high = df_all
print('remaining after S/N cut', len(df_SN_high))

for p in predictor_list:
    # Go through and make cuts in df_SN_high using the simulation min and maxes:
    print('making cut in', p, 'len before', len(df_SN_high))
    df_SN_high = df_SN_high[df_SN_high[p] > predictor_dict[p][0]]
    df_SN_high = df_SN_high[df_SN_high[p] < predictor_dict[p][1]]
    print('len after cut', len(df_SN_high))
    print('\n')
print('final length after cuts', len(df_SN_high))



print('len below 2.5', len(df_SN_high[df_SN_high['S/N'] < 2.5]))

STOP


print('length retained', len(df_SN_high), 'length cut', len(df_SN_low))
plt.hist(df_SN_high['Asymmetry (A)'].values, bins=1000)
plt.show()



print(df_SN_low.columns)
print(df_SN_low)
# Look at the A distribution now that I've made these cuts:
#plt.hist(df_SN_low['Asymmetry (A)'].values, bins=100)
#plt.show()

# Find out how many galaxies in the high and low bin have weird A values:
A_cutoff = -1
df_weird = df_SN_high[df_SN_high['Asymmetry (A)']<A_cutoff]
df_weird_low = df_SN_low[df_SN_low['Asymmetry (A)']<A_cutoff]
print('When A cutoff is ', A_cutoff, 'S/N > ', cutoff, len(df_weird), len(df_SN_high), len(df_weird)/len(df_SN_high))
print('When A cutoff is ', A_cutoff, 'S/N < ', cutoff, len(df_weird_low), len(df_SN_low), len(df_weird_low)/len(df_SN_low))

'''
A_cutoff = -0.1
df_weird = df_SN_high[df_SN_high['Asymmetry (A)']<A_cutoff]
df_weird_low = df_SN_low[df_SN_low['Asymmetry (A)']<A_cutoff]
print('When A cutoff is ', A_cutoff, 'S/N > ', cutoff, len(df_weird), len(df_SN_high), len(df_weird)/len(df_SN_high))
print('When A cutoff is ', A_cutoff, 'S/N < ', cutoff, len(df_weird_low), len(df_SN_low), len(df_weird_low)/len(df_SN_low))
'''

heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df_all[y].values, df_all[x].values, df_all['LD1'].values,statistic='mean', bins=100)


xmin = xedges[0]
xmax = xedges[-1]

ymin = yedges[0]
ymax = yedges[-1]

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.flipud(heatmap), cmap='magma_r',extent=[ymin, ymax, xmin, xmax], norm=matplotlib.colors.LogNorm())
ax.set_aspect((ymax-ymin)/(xmax-xmin))
plt.colorbar(im, label='LD1')
ax.set_xlabel(x)
ax.set_ylabel(y)
plt.show()







df_weird = df_weird.reset_index(drop=True)

#df_weird_high = df_SN_high[df_SN_high['Asymmetry (A)'] < 0]
#df_superweird_high = df_SN_high[df_SN_high['Asymmetry (A)'] < -1]
#print('length of S/N > ', cutoff, 'with A < 0', len(df_weird_high), 'A < -1', len(df_superweird_high))
print('length of weird', len(df_weird))
#407k less than 0
# only 206 less than -1
# But should probably have a good reason for the cutoff value you end up using


for i in range(len(df_weird)):
    row = df_weird.loc[i].values
    predictors = row[8:17]

    id = row[0]
    ra = row[1]
    dec = row[2]
    LD1 = row[20]
    p_merg = row[21]

    size = 80
    img = download_galaxy(id, ra, dec, '../frames/', size)

    preds = get_predictors(id, img, '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger/', size)
    
    shape = np.shape(img)[0]
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(abs(img), norm=matplotlib.colors.LogNorm())
    ax.annotate('S/N = '+str(round(predictors[8],1))+'\nGini = '+str(round(predictors[0],2))+' M20 = '+str(round(predictors[1],2))+'\nC = '+str(round(predictors[2],2))+' A = '+str(round(predictors[3],2))+'\nS = '+str(round(predictors[4],2))+'\nn = '+str(round(predictors[5],2))+' A_s = '+str(round(predictors[6],2))+'\nLD1 = '+str(round(LD1,2))+' p = '+str(round(p_merg,2)), xy=(0.03, 0.7),  xycoords='axes fraction',                                                       
            xytext=(0.03, 0.7), textcoords='axes fraction',                                               
            bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')   
    ax.set_title('ObjID = '+str(id))
    try:
        ax.annotate('S/N = '+str(round(preds[9],1))+'\nGini = '+str(round(preds[1],2))+' M20 = '+str(round(preds[2],2))+'\nC = '+str(round(preds[3],2))+' A = '+str(round(preds[4],2))+'\nS = '\
+str(round(preds[5],2))+'\nn = '+str(round(preds[6],2))+' A_s = '+str(round(preds[7],2)), xy=(0.03, 0.2),  xycoords='axes fraction',       
            xytext=(0.03, 0.2), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')
    except TypeError:# meaning output is None
        ax.annotate('None',xy=(0.03,0.2), xycoords='axes fraction', xytext=(0.03, 0.2), textcoords='axes fraction', bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')
    ax.set_xticks([0, (shape - 1)/2, shape-1])
    ax.set_xticklabels([-size/2, 0, size/2])
    ax.set_yticks([0, (shape- 1)/2,shape-1])
    ax.set_yticklabels([-size/2, 0,size/2])
    ax.set_xlabel('Arcsec')
    plt.show()

    

STOP

print('S/N cutoff', cutoff, 'len cut', len(df_SN_low))

# Cross-match with SQL to get r-band magnitude
df_high = df_SQL.merge(df_SN_high, left_on='objID', right_on='ID')
df_low = df_SQL.merge(df_SN_low, left_on='objID', right_on='ID')

plt.scatter(df_high['S/N'].values, df_high['dered_petro_r'].values, c = df_high['Asymmetry (A)'].values, s=0.5, vmin=-10, vmax=1)
plt.scatter(df_low['S/N'].values, df_low['dered_petro_r'].values, c = df_low['Asymmetry (A\
)'].values, s=0.5, vmin=-10, vmax=1)
plt.xlabel('S/N')
plt.colorbar(label='Asymmetry (A)')
plt.ylabel('Extinction-corrected r-band magnitude')
plt.show()


plt.scatter(df_high['S/N'].values, df_high['dered_petro_r'].values, c = df_high['Sersic N'].values, s=0.5)
plt.scatter(df_low['S/N'].values, df_low['dered_petro_r'].values, c = df_low['Sersic N'].values, s=0.5)
plt.xlabel('S/N')
plt.colorbar(label='Sersic N')
plt.ylabel('Extinction-corrected r-band magnitude')
plt.show()

print('len of low S/N', len(df_SN_low))


plt.clf()
#plt.hist(df_all['Asymmetry (A)'].values, color='black', alpha=0.5, bins=100)
plt.hist(df_cut['Asymmetry (A)'].values, alpha=0.5, bins=100)
plt.show()

plt.clf()
#plt.hist(df_all['Asymmetry (A)'].values, color='black', alpha=0.5, bins=100)                 
plt.hist(df_cut['Sersic N'].values, alpha=0.5, bins=100)
plt.show()

STOP

#df_all.to_csv('../Tables/SDSS_predictors_all.txt',sep='\t')


# Great, but how many are lost in the fails file and why do these fail?

# The fails are stored somewhere different, first get the fails from the original run, then get the selection_2 fails
prefix = '/Users/rebeccanevin/Documents/Cannon_parallel/'
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
