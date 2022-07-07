#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is to test if changing the bins changes the result
# of the trends of f_merg with redshift and mass

# This is combo'ed with Mendel,
# so is restricted to the cross-over between these 
# two samples
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
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
import glob
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from scipy.stats import iqr


# path
dir = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'


# This is to load up the mass complete table:
mass = 'log_stellar_mass_from_color'
red = 'z'
spacing_z = 0.02
complete = False
completeness = 95
nbins_mass = 6
suffix = str(spacing_z)+'_'+str(red)+'_'+str(mass)+'_completeness_'+str(completeness)


# Check if this table ^ even exists:
if os.path.exists(dir+'all_mass_color_complete_'+str(suffix)+'.txt'):
    print('it exists! you can run the f_merg analysis')
else:
    print('missing mass table to run this analysis')


# Now import the stuff to get your LDA and predictor table
# so you can have various properties of each galaxy
type_marginalized = '_flags_cut_segmap'
type_gal = 'predictors'
run = 'major_merger'
# set this if you want to do a limited subset
num = None
savefigs = False
# set this if you want to save the result
save_df = False
JK_anyway = False


# The below are flags for doing additional S/N
# or B/T cuts
s_n_cut = False
B_T_cut = False
#BT = '0.5_to_1.0_'
if s_n_cut:
    low_s_n = 15
    high_s_n = 50000
    add_on_binned_table = 's_n_'+str(low_s_n)+'_to_'+str(high_s_n)
    
else:
    add_on_binned_table = ''


if complete:
    table_name = dir + 'f_merg_'+str(run)+'_'+str(suffix)+'_mass_bins_'+str(nbins_mass)+'_'+str(add_on_binned_table)+'.csv'
else:
    table_name = dir + 'f_merg_'+str(run)+'_'+str(suffix)+'_incomplete_mass_bins_'+str(nbins_mass)+'_'+str(add_on_binned_table)+'.csv'
    
if os.path.exists(table_name) and save_df:
    print('table already exists do you want to oversave?')

    




df_LDA = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run+'_flags.txt',header=[0],sep='\t')

# Because the df_LDA doesn't have the final flag, use the predictor table to instead clean via merging

# Run OLS with predictor values and z and stellar mass and p_merg:
df_predictors = pd.io.parsers.read_csv(filepath_or_buffer=dir+'SDSS_predictors_all_flags_plus_segmap.txt',header=[0],sep='\t')

if len(df_LDA) != len(df_predictors):
    print('these have different lengths cannot use one to flag')
    STOP

# First clean this so that there's no segmap flags
df_predictors_clean = df_predictors[(df_predictors['low S/N'] ==0) & (df_predictors['outlier predictor']==0) & (df_predictors['segmap']==0)]

clean_LDA = df_LDA[df_LDA['ID'].isin(df_predictors_clean['ID'].values)]

if complete:
    masstable = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_color_complete_'+str(suffix)+'.txt',header=[0],sep='\t')
else:
    masstable = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_'+str(suffix)+'.txt',header=[0],sep='\t')



if red == 'z_spec':
    masstable = masstable[masstable['logBD'] < 13]
    masstable = masstable[masstable['dBD'] < 1]
    masstable['B/T'] = 10**masstable['logMb']/(10**masstable['logMb']+10**masstable['logMd'])
    masstable = masstable[['objID',
    'z_x','z_spec',
    'logBD','log_stellar_mass_from_color','B/T']]
else:
    masstable = masstable[masstable['log_stellar_mass_from_color'] < 13]
    masstable = masstable[['objID',
    'z','log_stellar_mass_from_color']]



# Now merge this with LDA
#merged_1 = masstable_m.merge(clean_LDA, left_on='objID', right_on='ID')#[0:1000]# Now merging on dr8

merged_1 = masstable.merge(clean_LDA,left_on='objID', right_on='ID',# left_index=True, right_index=True,
                  suffixes=('', '_y'))
if red == 'z_spec':
    merged_1 = merged_1[['ID','z_x','z_spec','logBD','log_stellar_mass_from_color',
        'p_merg','B/T']]
else:
    merged_1 = merged_1[['ID','z','log_stellar_mass_from_color',
        'p_merg']]


#merged_1.drop(merged_1.filter(regex='_y$').columns, axis=1, inplace=True)

final_merged = merged_1.merge(df_predictors_clean, on='ID')
if red == 'z_spec':
    final_merged = final_merged[['ID','z_x','z_spec','logBD','log_stellar_mass_from_color',
        'p_merg','S/N','B/T']]
else:
    final_merged = final_merged[['ID','z','log_stellar_mass_from_color',
        'p_merg','S/N']]
print('len merged with mendel', len(final_merged))


if s_n_cut:
    # Then cut by S/N
    print('length before S/N cut', len(final_merged))

    '''
    # figure out where to bin
    cats_s_n, bins_s_n = pd.qcut(final_merged['S/N'], q=3, retbins=True, precision = 1)
    print('cats s/n', cats_s_n)
    print('bins s/n', bins_s_n)
    STOP
    '''


    final_merged = final_merged[(final_merged['S/N'] > low_s_n) & (final_merged['S/N'] < high_s_n)]

    print('length after S/N cut', len(final_merged))

    




# Load in the bins in z unless its the incomplete version (suffix == 'color'), 
# in that case define your own spacing
bins_z = np.load(dir+'all_mass_color_complete_'+str(suffix)+'_bins.npy')




# This is for dropping some stuff if you need it to run fast
if num:
    final_merged = final_merged.dropna()[0:num]
else:
    final_merged = final_merged.dropna()
print('length after dropping Nans', len(final_merged))









# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run OLS with predictor values and z and stellar mass and p_merg:
# Now merge these two and run an OLS:



#cats, bins



cats_mass, bins_mass = pd.qcut(final_merged[mass], q=nbins_mass, retbins=True, precision = 1)
#df['mass_bin'] = cats_mass

centers_z = [(bins_z[x+1] - bins_z[x])/2 + bins_z[x] for x in range(len(bins_z)-1)]
centers_mass = [(bins_mass[x+1] - bins_mass[x])/2 + bins_mass[x] for x in range(len(bins_mass)-1)]


# Now that you have the bins in both dimensions, make a figure of your different bins:
plt.clf()
'''
plt.scatter(df_pred_merged['z_x'].values, df_pred_merged['log_stellar_mass_from_color'].values, color='orange', s=0.2)
plt.annotate(str(len(df_pred_merged['z_x'].values)), 
    xy = (np.mean(df_pred_merged['z_x'].values), np.mean(df_pred_merged['log_stellar_mass_from_color'].values)), 
    xycoords='data', color='black')
'''
for i in range(len(bins_z)-1):
    bin_start_z = bins_z[i]
    bin_end_z = bins_z[i+1]
    for j in range(len(bins_mass)-1):
        bin_start_mass = bins_mass[j]
        bin_end_mass = bins_mass[j+1]
        df_select = final_merged[(final_merged[red] > bin_start_z) 
            & (final_merged[red] < bin_end_z) 
            & (final_merged[mass] > bin_start_mass) 
            & (final_merged[mass] < bin_end_mass)]
        plt.scatter(df_select[red].values, df_select[mass].values, 
            s=0.2)
        plt.annotate(str(len(df_select[red].values)), 
            xy = (np.mean(df_select[red].values) - 0.005, 
                np.mean(df_select[mass].values - 0.05)), 
            xycoords='data', color='black')
plt.xlabel(r'$z$')
plt.ylabel(mass)
if savefigs:
    plt.savefig('../Figures/MCMC_scatter_'+str(run)+'_'+suffix+'_color.png', dpi=1000)
else:
    plt.show()






if save_df:
    # first go through and load up all of prior files
    list_of_prior_files = glob.glob(dir + 'change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'*'+str(type_marginalized)+'.txt')
    print('length of prior files', len(list_of_prior_files))
    if len(list_of_prior_files) ==0:
        print('there are no priors prepared')
        name_single = dir + 'LDA_out_all_SDSS_predictors_'+str(run)+'_flags.txt'
        table = pd.io.parsers.read_csv(filepath_or_buffer=name_single,header=[0],sep='\t')[['ID','p_merg']]
    else:
        table_list = []
        for p in range(len(list_of_prior_files)):
            
            prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[p],header=[0],sep='\t')
            # cut it way down
            if p == 0:
                table = prior_file[['ID','p_merg']]

            else:
                table_p = prior_file[['p_merg']] # just take p_merg if not the last one
                table_p.columns = ['p_merg_'+str(p)]
                # Now stack all of these tables
                table = table.join(table_p)
        
        

    # Now that these are all joined together, identify which ones have IDs that match mergers
    
    count = {}
    S_N = {}
    if red == 'z_spec':
        B_T = {}
    f_merg = {}
    f_merg_avg = {}
    f_merg_std = {}
    for i in range(len(bins_z)-1):
        bin_start_z = bins_z[i]
        bin_end_z = bins_z[i+1]
        print('start z ', bin_start_z, 'stop z ', bin_end_z)
        for j in range(len(bins_mass)-1):
            bin_start_mass = bins_mass[j]
            bin_end_mass = bins_mass[j+1]
            print('start mass ', bin_start_mass, 'stop mass ', bin_end_mass)
            # build dataset
            df_select = final_merged[(final_merged[red] > bin_start_z) 
                    & (final_merged[red] < bin_end_z) 
                    & (final_merged[mass] > bin_start_mass) 
                    & (final_merged[mass] < bin_end_mass)]
            
            S_N[centers_z[i],centers_mass[j]] = np.mean(df_select['S/N'].values)
            if red == 'z_spec':
                B_T[centers_z[i],centers_mass[j]] = np.mean(df_select['B/T'].values)
               
            df_select = df_select[['ID']] # just take this because you don't need the other deets
        

            count[centers_z[i],centers_mass[j]] = len(df_select)
            

            merged = table.merge(df_select, on = 'ID')#left_on='ID', right_on='objID')
            # for each column of p_merg, calculate the the f_merg and then find the median
        

            gt = (merged > 0.5).apply(np.count_nonzero)
            
        

            #fmerg_here = len(np.where(merged['p_merg_x'] > 0.5)[0])/len(merged)
            
            #f_merg[centers_z[i]].append(fmerg_here)
            f_merg_avg[centers_z[i],centers_mass[j]] = np.median(gt.values[1:]/len(merged))
            f_merg_std[centers_z[i],centers_mass[j]] = np.std(gt.values[1:]/len(merged))



    # find a way to put this into df format
    mass_val = []
    z_val = []
    f_merg_val = []
    f_merg_e_val = []
    count_val = []
    s_n_val = []
    if red == 'z_spec':
        b_t_val = []
    for i in range(len(bins_z)-1):
        for j in range(len(bins_mass)-1):
            f_merg_val.append(f_merg_avg[centers_z[i],centers_mass[j]])
            f_merg_e_val.append(f_merg_std[centers_z[i],centers_mass[j]])
            z_val.append(centers_z[i])
            mass_val.append(centers_mass[j])
            count_val.append(count[centers_z[i],centers_mass[j]])
            s_n_val.append(S_N[centers_z[i],centers_mass[j]])
            if red == 'z_spec':
                b_t_val.append(B_T[centers_z[i],centers_mass[j]])
    # Now make a df out of these lists
    if red == 'z_spec':
        df_fmerg = pd.DataFrame(list(zip(mass_val, z_val, f_merg_val, f_merg_e_val, count_val, s_n_val, b_t_val)),
                             columns =['mass', 'z', 'fmerg', 'fmerg_std', 'count', 'S/N', 'B/T'])
    else:
        df_fmerg = pd.DataFrame(list(zip(mass_val, z_val, f_merg_val, f_merg_e_val, count_val, s_n_val)),columns =['mass', 'z', 'fmerg', 'fmerg_std', 'count', 'S/N'])
    #except:
    #		df_fmerg = pd.DataFrame(list(zip(mass_val, z_val, f_merg_val, f_merg_e_val, count_val, s_n_val, b_t_val)),
    #           columns =['mass', 'z_x', 'fmerg', 'fmerg_std', 'count', 'S/N', 'B/T'])
    df_fmerg.to_csv(table_name, sep='\t')
else:
    df_fmerg = pd.read_csv(table_name, sep = '\t')

    
df_fmerg = df_fmerg.dropna()
print(df_fmerg)

# First, regress z and S/N:
X = df_fmerg[['z']] 
y = df_fmerg['S/N']
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
print(est.summary())

# Do a 2D regression first with just mass and z:
X = df_fmerg[['S/N']] 

y = df_fmerg['fmerg']
## fit a OLS model with intercept on mass and z
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
print(est.summary())


# Do a 2D regression first with just mass and z:
if red == 'z_spec':
    X = df_fmerg[['mass', 'z', 'S/N','B/T']] 
else:
    X = df_fmerg[['mass', 'z']] 

y = df_fmerg['fmerg']
## fit a OLS model with intercept on mass and z
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
print(est.summary())

# Do a 2D regression:
if red == 'z_spec':
    X = df_fmerg[['mass', 'z', 'S/N','B/T']] 
else:
    X = df_fmerg[['mass', 'z', 'S/N']] 

y = df_fmerg['fmerg']
## fit a OLS model with intercept on mass and z
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
print(est.summary())

STOP


# same thing but standardized
if red == 'z_spec':
    X = df_fmerg[['mass', 'z', 'S/N','B/T']] 
else:
    X = df_fmerg[['mass', 'z', 'S/N']] 

from sklearn import preprocessing
# l1, l2, max don't really make much of a difference
# still getting super small #s when normalized
# Used to be Normalizer
normalizer = preprocessing.StandardScaler()#norm='l1')
normalized_train_X = normalizer.fit_transform(X)


X_std = normalizer.transform(X)

## fit a OLS model with intercept on mass and z
X = sm.add_constant(X_std)
est = sm.OLS(y, X).fit()
print(est.summary())

STOP





count = df_fmerg['count'].values

max_count = max(count)
print('also max?', max_count)



plt.clf()

# there should be 8 different redshifts
colors = ['#493843','#61988E','#A0B2A6','#CBBFBB','#EABDA8','#FF9000','#DE639A','#D33E43']
colors = ['#C5979D','#E7BBE3','#78C0E0','#449DD1','#3943B7','#150578','#0E0E52','black']
colors = ['#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black',
    '#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black',
    '#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black']

print('centers of mass', centers_mass)
print('centers of zs', centers_z)

for zcen in centers_z:
    
    # Grab everything with a z value from the df
    df_select = df_fmerg[df_fmerg['z'] == zcen].dropna().reset_index()
    if len(df_select)==0:
        continue
    # should already be sorted by mass
    count = df_select['count'].values
    masscen = df_select['mass'].values


    
    # to do a linear regression, I'm going to need to define X and y
    #masscen_masked = np.ma.masked_where(count < 1000, masscen)
    x = masscen.reshape(-1,1)
    x = x[count > 1000]
    #x = np.ma.masked_where(count < 1000, x)
    try:
        X = sm.add_constant(x)
    except ValueError:
        continue
    y = df_select['fmerg']

    # mask y where count is less than 1000
    
    y = y[count > 1000].reset_index(drop=True)
    #np.ma.masked_where(count < 1000, y)

    error = df_select['fmerg_std']

    # If there are no errors because this didn't have a bunch of different priors,
    # 
    if np.all(error) == 0.0 or JK_anyway:
        # This means you should jacknife because we don't have errors

        res = sm.OLS(y, X, missing='drop').fit()
        try:
            slope = res.params[1]
        except IndexError:
            continue
        st, data, ss2 = summary_table(res, alpha=0.05)
        fittedvalues = data[:,2]
        predict_mean_se  = data[:,3]
        predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
        predict_ci_low, predict_ci_upp = data[:,6:8].T

        print('this is X', X)
        print('this is y', y)


        plt.plot(x, fittedvalues, color='#3D3B8E')#, label='OLS')
        plt.scatter(x, y, color='#E072A4', label='data', zorder=100)

        plt.plot(x, predict_ci_low, color='#6883BA', lw=2)
        plt.plot(x, predict_ci_upp, color='#6883BA', lw=2)
        plt.plot(x, predict_mean_ci_low, color='#6883BA', lw=2)
        plt.plot(x, predict_mean_ci_upp, color='#6883BA', lw=2)

        slopes = []
        # What about bootstrapping, iteratively dropping each of the points and re-fitting the line?
        for i in range(len(y)):
            '''
            xcrop= np.delete(masscen,i).reshape(-1,1)#.pop(i)
            #print(np.shape(x))
            Xcrop = sm.add_constant(xcrop)
            '''

            xcrop = np.delete(x,i)
            Xcrop = sm.add_constant(xcrop)
            
            #X_crop = X.crop(i).reset_index()
            ycrop = y.drop(i).reset_index(drop=True)

            

        
            res = sm.OLS(ycrop, Xcrop, missing='drop').fit()
            try:
                slope = res.params[1]
            except IndexError:
                continue
            st, data, ss2 = summary_table(res, alpha=0.05)
            fittedvalues = data[:,2]
            plt.plot(xcrop, fittedvalues, color='#B0E298', alpha=0.5)#, label='OLS')
            slopes.append(slope)

        
        plt.xlabel('Mass bins')
        plt.ylabel('fmerg')
        plt.title('$z = $'+str(zcen))
        #plt.legend()
        plt.annotate(f'{round(slope,3)} JK {round(np.mean(slopes),3)} pm {round(np.std(slopes),3)}', 
            xy=(0.01,0.9), xycoords='axes fraction', color='#3D3B8E')
        plt.show()

        
    
    else:
        # This is if you want to MCMC

        res_bb = sm.OLS(y, X, missing = 'drop').fit()#, missing = 'drop'
        try:
            _, data_bb, _ = summary_table(res_bb, alpha=0.05)
        except TypeError:
            continue
        big_boy_fit = data_bb[:,2]


        error = error[count > 1000].reset_index(drop=True)

        print(f'y after, {y}, xs {x}, errors {error}')



        mu, sigma = 0, 1 # mean and standard deviation
        

        plt.clf()
        
        # iterate
        # save slope values
        slope_list = []
        for num in range(100):

            
            Y = [y[i]+ error[i] * np.random.normal(mu, sigma, 1)  for i in range(len(y))]
            
            
            if num == 0:
                print('y',y)
                print('error', error)
                print('big Y', Y)
            

            #scaler = StandardScaler()
            #scaler.fit(X)
            #X_standardized = scaler.transform(X)
            
            res = sm.OLS(Y, X).fit()#, missing = 'drop'
            
            

            
            #plt.scatter(x, Y, s=0.1, color=colors[color_count])
            try:
                slope_list.append(res.params[1])
            except:
                continue
                slope_list.append(999)
            
            

            st, data, ss2 = summary_table(res, alpha=0.05)
            fittedvalues = data[:,2]
            predict_mean_se  = data[:,3]
            predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
            predict_ci_low, predict_ci_upp = data[:,6:8].T

            #print(summary_table(res, alpha=0.05))
            

            

            plt.plot(x, fittedvalues, 'black', alpha=0.5)#, label='OLS')

        #plt.plot(x, predict_ci_low, 'b--')
        #plt.plot(x, predict_ci_upp, 'b--')
        #plt.plot(x, predict_mean_ci_low, 'g--')
        #plt.plot(x, predict_mean_ci_upp, 'g--')
        plt.scatter(x, y, color='black', label='data', zorder=100)
        plt.plot(x, big_boy_fit, color = 'black', zorder=100)

        plt.errorbar(x, y, yerr = error, linestyle='None', color='black', zorder=100)
        for (count, x, y) in zip(count, x, y):
            plt.annotate(str(count), xy = (x, y+0.07), xycoords='data', color = 'black')

        plt.xlabel('Mass bins')
        plt.ylabel('fmerg')
        plt.title('$z = $'+str(zcen))
        #plt.legend()
        try:
            plt.annotate(str(round(res_bb.params[1],2)), 
                xy=(0.01,0.95), xycoords='axes fraction', color='black')
        except IndexError:
            continue
        plt.annotate(str(round(np.mean(slope_list),2))+'+/-'+str(round(np.std(slope_list),2)), 
            xy=(0.01,0.9), xycoords='axes fraction', color='black')
        plt.show()


    

plt.clf()
# Make the same plot but for redshift on the x axis:
color_count = 0
for masscen in centers_mass:

    print(masscen)
    print(str(round(bins_mass[color_count],3))+'$ < M < $'+str(round(bins_mass[color_count+1],3)))
    #color_count+=1
    #continue
    # Grab everything with a z value from the df
    df_select = df_fmerg[df_fmerg['mass'] == masscen].dropna().reset_index()
    print(df_select)
    if len(df_select)==0:
        color_count+=1
        continue
    # should already be sorted by mass
    count = df_select['count'].values
    print('total count', np.sum(count))
    try:
        zcen = df_select['z'].values
    except KeyError:
        zcen = df_select['z_x'].values


    
    

    # to do a linear regression, I'm going to need to define X and y
    #masscen_masked = np.ma.masked_where(count < 1000, masscen)
    x = zcen.reshape(-1,1)
    x = x[count > 1000]
    #x = np.ma.masked_where(count < 1000, x)
    try:
        X = sm.add_constant(x)
    except ValueError:
        continue
    y = df_select['fmerg']

    # mask y where count is less than 1000
    print(f'y before, {y}')
    y = y[count > 1000].reset_index(drop=True)
    #np.ma.masked_where(count < 1000, y)

    error = df_select['fmerg_std']

    if np.all(error) == 0.0 or JK_anyway:
        res = sm.OLS(y, X, missing='drop').fit()
        try:
            slope = res.params[1]
        except IndexError:
            continue
        st, data, ss2 = summary_table(res, alpha=0.05)
        fittedvalues = data[:,2]
        predict_mean_se  = data[:,3]
        predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
        predict_ci_low, predict_ci_upp = data[:,6:8].T

        #print(summary_table(res, alpha=0.05))
        
        plt.plot(x, fittedvalues, color='#3D3B8E')#, label='OLS')
        plt.scatter(x, y, color='#E072A4', label='data', zorder=100)

        plt.plot(x, predict_ci_low, color='#6883BA', lw=2)
        plt.plot(x, predict_ci_upp, color='#6883BA', lw=2)
        plt.plot(x, predict_mean_ci_low, color='#6883BA', lw=2)
        plt.plot(x, predict_mean_ci_upp, color='#6883BA', lw=2)

        slopes = []
        # What about bootstrapping, iteratively dropping each of the points and re-fitting the line?
        for i in range(len(y)):
            xcrop = np.delete(x,i)
            Xcrop = sm.add_constant(xcrop)
            
            #X_crop = X.crop(i).reset_index()
            ycrop = y.drop(i).reset_index(drop=True)

            

        
            res = sm.OLS(ycrop, Xcrop, missing='drop').fit()
            try:
                slope = res.params[1]
            except IndexError:
                continue
            st, data, ss2 = summary_table(res, alpha=0.05)
            fittedvalues = data[:,2]
            plt.plot(xcrop, fittedvalues, color='#B0E298', alpha=0.5)#, label='OLS')
            slopes.append(slope)

        
        plt.xlabel('Redshift bins')
        plt.ylabel('fmerg')
        plt.title('M = '+str(masscen))
        #plt.legend()
        plt.annotate(f'{round(slope,3)} JK {round(np.mean(slopes),3)} pm {round(np.std(slopes),3)}', 
            xy=(0.01,0.9), xycoords='axes fraction', color='#3D3B8E')
        plt.show()
    else:

        res_bb = sm.OLS(y, X).fit()#, missing = 'drop'
        try:
            _, data_bb, _ = summary_table(res_bb, alpha=0.05)
        except TypeError:
            continue
        big_boy_fit = data_bb[:,2]



        error = df_select['fmerg_std']
        error = error[count > 1000].reset_index(drop=True)

        print(f'y after, {y}, xs {x}, errors {error}')

        
        plt.clf()
        
        # iterate
        # save slope values
        slope_list = []
        for num in range(100):
            Y = [y[i] + error[i] * np.random.normal(mu, sigma, 1) for i in range(len(y))]

            #scaler = StandardScaler()
            #scaler.fit(X)
            #X_standardized = scaler.transform(X)


            res = sm.OLS(Y, X).fit()
            
            #plt.scatter(x, Y, s=0.1, color=colors[color_count])
            try:
                slope_list.append(res.params[1])
            except:
                continue
                slope_list.append(999)
            
            

            st, data, ss2 = summary_table(res, alpha=0.05)
            fittedvalues = data[:,2]
            predict_mean_se  = data[:,3]
            predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
            predict_ci_low, predict_ci_upp = data[:,6:8].T

            #print(summary_table(res, alpha=0.05))
            

            

            plt.plot(x, fittedvalues, 'black', alpha=0.5)#, label='OLS')
        #plt.plot(x, predict_ci_low, 'b--')
        #plt.plot(x, predict_ci_upp, 'b--')
        #plt.plot(x, predict_mean_ci_low, 'g--')
        #plt.plot(x, predict_mean_ci_upp, 'g--')
        plt.scatter(x, y, color=colors[color_count], label='data', zorder=100)
        plt.plot(x, big_boy_fit, color = 'black', zorder=100)

        plt.errorbar(x, y, yerr = error, linestyle='None', color='black', zorder=100)
        for (count, x, y) in zip(count, x, y):
            plt.annotate(str(count), xy = (x, y+0.07), xycoords='data', color = 'black')

        plt.xlabel('Redshift bins')
        plt.ylabel('fmerg')
        plt.title('$M = $'+str(masscen))
        #plt.legend()
        try:
            plt.annotate(str(round(res_bb.params[1],2)), 
                xy=(0.01,0.95), xycoords='axes fraction', color='black')
        except IndexError:
            continue
        plt.annotate(str(round(np.mean(slope_list),2))+'+/-'+str(round(np.std(slope_list),2)), 
            xy=(0.01,0.9), xycoords='axes fraction', color='black')
        plt.show()

    color_count+=1

    