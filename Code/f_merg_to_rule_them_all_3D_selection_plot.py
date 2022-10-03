#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# remake figure 10 but with S/N as the color
# maybe a binned plot?
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from matplotlib.patches import Rectangle
import scipy
import matplotlib



# path
dir = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'


# This is to load up the mass complete table:
ack = False # option whether to use ackermann cross-matches
mass = 'log_stellar_mass_from_color'
red = 'z'
spacing_z = 0.02
complete = True
completeness = 95
nbins_mass = 7#15
nbins_S_N = 7
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
savefigs = True
# set this if you want to save the result
save_df = True
JK_anyway = False
S_N_cut = True
S_N_cut_val = 50


if complete:
    table_name = dir + 'f_merg_'+str(run)+'_'+str(suffix)+'_mass_bins_'+str(nbins_mass)+'_S_N_bins_'+str(nbins_S_N)+'_flags.csv'
else:
    table_name = dir + 'f_merg_'+str(run)+'_'+str(suffix)+'_incomplete_mass_bins_'+str(nbins_mass)+'_S_N_bins_'+str(nbins_S_N)+'_flags.csv'

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


if S_N_cut:
    final_merged = final_merged[final_merged['S/N'] < S_N_cut_val]


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

cats_S_N, bins_S_N = pd.qcut(final_merged['S/N'], q=nbins_S_N, retbins=True, precision = 1)
print('cats S/N', cats_S_N)
print('bins S/N', bins_S_N)


cats_mass, bins_mass = pd.qcut(final_merged[mass], q=nbins_mass, retbins=True, precision = 1)
#df['mass_bin'] = cats_mass

centers_z = [(bins_z[x+1] - bins_z[x])/2 + bins_z[x] for x in range(len(bins_z)-1)]
centers_mass = [(bins_mass[x+1] - bins_mass[x])/2 + bins_mass[x] for x in range(len(bins_mass)-1)]
centers_S_N = [(bins_S_N[x+1] - bins_S_N[x])/2 + bins_S_N[x] for x in range(len(bins_S_N)-1)]
 


# Before you do any analysis, it's necessary to drop certain parts of the table that are not
# complete
# Is this possible to do after you save?
# What kind of criteria do I want to use to do this?
# One way could be to drop bins that have a weird distribution of masses?
# Likewise, a weird distribution of redshifts
# Maybe the mean in one of these is significantly off





if save_df:
    # make a massive plot
    plt.clf()
    fig = plt.figure(figsize = (10,5))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    #currentAxis = plt.gca()
    ax0.scatter(final_merged[red], final_merged[mass], color='grey', s=0.1)
    ax0.set_ylabel(r'log stellar mass ($M_*$)')
    ax0.set_xlabel(r'$z$')
    
    ax1.scatter(final_merged[red], final_merged['S/N'], color='grey', s=0.1)
    ax1.set_ylabel('S/N')
    ax1.set_xlabel(r'$z$')
    
    
    
    
    # first go through and load up all of prior files
    list_of_prior_files = glob.glob(dir + 'change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'_0.*'+str(type_marginalized)+'.txt')
    print('length of prior files', len(list_of_prior_files))
    if len(list_of_prior_files) ==0:
        print('there are no priors prepared')
        name_single = dir + 'LDA_out_all_SDSS_predictors_'+str(run)+'_flags.txt'
        table = pd.io.parsers.read_csv(filepath_or_buffer=name_single,header=[0],sep='\t')[['ID','p_merg']]
    else:
        table_list = []
        for p in range(len(list_of_prior_files)):
            print('p prior', p)
            prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[p],header=[0],sep='\t')
            # cut it way down
            if p == 0:
                table = prior_file[['ID','p_merg']]

            else:
                table_p = prior_file[['p_merg']] # just take p_merg if not the last one
                table_p.columns = ['p_merg_'+str(p)]
                # Now stack all of these tables
                table = table.join(table_p)
        
    '''
    plt.clf()   
    plt.scatter(final_merged[red],final_merged[mass],c=final_merged['S/N'], s=0.1)
    plt.colorbar(label='S/N')
    plt.xlabel(r'$z$')
    plt.ylabel(r'log stellar mass (M$_{\odot}$)')
    plt.show()
    '''
    
    #histogram definition
    xyrange = [[0,0.37],[9.3,13.3]] # data range
    bins = [100,100] # number of bins
    thresh = 10  #density threshold

    #data definition
    
    xdat, ydat = final_merged[red],final_merged[mass]

    # histogram the data
    hh, locx, locy = np.histogram2d(xdat, ydat, range=xyrange, bins=bins)
    print('hh', hh)
    print('locx', locx)
    print('locy', locy)
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)

    #select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < thresh] # low density points
    ydat1 = ydat[ind][hhsub < thresh]
    hh[hh < thresh] = np.nan # fill the areas with low density by NaNs
    
    '''
    plt.clf()
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    plt.imshow(np.flipud(hh.T),cmap='jet',extent=np.array(xyrange).flatten(), 
                   interpolation='none', origin='upper')
    plt.colorbar()   
    plt.plot(xdat1, ydat1, '.',color='darkblue')#, size=0.1)
    plt.axis('square')
    #ax.set_aspect('equal')
    plt.show()
    '''
    nbins = 100
    
    x_bins = np.linspace(np.min(final_merged[red]),np.max(final_merged[red]),nbins)
    
    y_bins = np.linspace(np.min(final_merged[mass]),np.max(final_merged[mass]),nbins)

    ret = scipy.stats.binned_statistic_2d(final_merged[red], final_merged[mass], final_merged['S/N'], statistic=np.mean, bins=[x_bins, y_bins])

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    im1 = ax1.imshow(ret.statistic.T, origin='lower',  
                     norm=matplotlib.colors.LogNorm(vmin=2.5, vmax=40), interpolation=None)#, #norm=matplotlib.colors.LogNorm(vmin=2.5, vmax=50)
                     #3extent=(np.min(final_merged[red]), np.max(final_merged[red]), np.min(final_merged[mass]),np.min(final_merged[mass])))
    plt.colorbar(im1, ax = ax1, fraction = 0.046, label='mean S/N in each bin')
    
    xlabels = np.linspace(np.min(final_merged[red]),np.max(final_merged[red]),5)
    xlabels = [round(x,2) for x in xlabels]
    print('locations',np.arange(0,nbins-1,int(nbins/5)))
    print('labels', xlabels)
    print('trying to go from', np.min(final_merged[red]), np.max(final_merged[red]))
    ax1.set_xticks(np.arange(0,nbins-1,int(nbins/5)))
    ax1.set_xticklabels(xlabels)
    
    ylabels = np.linspace(np.min(final_merged[mass]),np.max(final_merged[mass]),5)
    ylabels = [round(x,2) for x in ylabels]
    
    print('locations',np.arange(0,nbins-1,int(nbins/5)))
    print('labels', ylabels)
    print('trying to go from', np.min(final_merged[mass]), np.max(final_merged[mass]))
    
    ax1.set_yticks(np.arange(0,nbins-1,int(nbins/5)))
    ax1.set_yticklabels(ylabels)
    
    ax1.set_ylabel(r'log stellar mass (M$_{\odot}$)')
    ax1.set_xlabel(r'$z$')
    
    plt.show()
    print('xaxis should go from ',np.min(final_merged[red]),' to ', np.max(final_merged[red]))
    
    
    print(hh.T)
 
    STOP

    # Now that these are all joined together, identify which ones have IDs that match mergers
    
    count = {}
    flag = {}
    S_N = {}
    if red == 'z_spec':
        B_T = {}
    f_merg = {}
    f_merg_avg = {}
    f_merg_std = {}
    for i in range(len(bins_z)-1):
        bin_start_z = bins_z[i]
        bin_end_z = bins_z[i+1]
        bin_center_z = (bin_end_z - bin_start_z)/2 + bin_start_z
        print('start z ', round(bin_start_z,2), 'stop z ', round(bin_end_z,2))
        for j in range(len(bins_mass)-1):
            bin_start_mass = bins_mass[j]
            bin_end_mass = bins_mass[j+1]
            bin_center_mass = (bin_end_mass - bin_start_mass)/2 + bin_start_mass
            print('start mass ', round(bin_start_mass,2), 'stop mass ', round(bin_end_mass,2))
            
            # Okay do the second completeness before you do S/N stuff:
            df_select = final_merged[(final_merged[red] > bin_start_z) 
                        & (final_merged[red] < bin_end_z) 
                        & (final_merged[mass] > bin_start_mass) 
                        & (final_merged[mass] < bin_end_mass)]
                
            # Now figure out where the means are 
            
            med_x = np.median(df_select[red].values)
            med_y = np.median(df_select[mass].values)
            std_x = np.std(df_select[red].values)
            std_y = np.std(df_select[mass].values)
            
            
            if (((med_x - std_x) > bin_center_z) & ((med_x + std_x) > bin_center_z)) | (((med_x - std_x) < bin_center_z) & ((med_x + std_x) < bin_center_z)) | (std_x > (bin_end_z - bin_start_z)/2):
                off_in_z = True
            else:
                off_in_z = False
                
            if (((med_y - std_y) > bin_center_mass) & ((med_y + std_y) > bin_center_mass)) | (((med_y - std_y) < bin_center_mass) & ((med_y + std_y) < bin_center_mass)) | (std_y > (bin_end_mass - bin_start_mass)/2):
                off_in_mass = True
            else:
                off_in_mass = False
                
            
            
            
            
            
            # so basically, if the medians of the distribution
            # are significantly different than the zcen and masscen,
            # then throw a flag because probably incomplete
            if off_in_mass or off_in_z:
                off = 1
                #flag[centers_z[i],centers_mass[j],:] = 1
            else:
                #flag[centers_z[i],centers_mass[j],:] = 0
                
                if len(df_select) > 1000:
                    # Rectangle is expanded from lower left corner
                    #
                   ax0.scatter(df_select[red], df_select[mass], color='#52DEE5', s=0.1)
                   ax0.add_patch(
                        Rectangle((bin_start_z, bin_start_mass), 
                                bin_end_z - bin_start_z, bin_end_mass - bin_start_mass, facecolor='None', 
                                edgecolor='black')
                        )
                   ax0.annotate(f'{round(len(df_select)/1000,1)}', 
                                xy = (bin_start_z+0.005, bin_start_mass+0.005), 
                                xycoords='data', size=5)
            
            for k in range(len(bins_S_N)-1):
                bin_start_S_N = bins_S_N[k]
                bin_end_S_N = bins_S_N[k+1]
                print('start S/N', round(bin_start_S_N,1), 'stop S/N', round(bin_end_S_N,1))
                # build dataset
                
                df_select = final_merged[(final_merged[red] > bin_start_z) 
                        & (final_merged[red] < bin_end_z) 
                        & (final_merged[mass] > bin_start_mass) 
                        & (final_merged[mass] < bin_end_mass)
                        & (final_merged['S/N'] > bin_start_S_N) 
                        & (final_merged['S/N'] < bin_end_S_N)]
                if off_in_mass or off_in_z:
                    flag[centers_z[i],centers_mass[j],centers_S_N[k]] = 1
                else:
                    flag[centers_z[i],centers_mass[j],centers_S_N[k]] = 0
                    if len(df_select) > 1000:
                        # Rectangle is expanded from lower left corner
                        #
                        ax1.scatter(df_select[red], df_select[mass], c=np.mean(df_select['S/N']), s=0.1)
                        ax1.add_patch(
                                Rectangle((bin_start_z, bin_start_S_N), 
                                        bin_end_z - bin_start_z, bin_end_S_N - bin_start_S_N, facecolor='None', 
                                        edgecolor='black')
                                )
                        ax1.annotate(f'{round(len(df_select)/1000,1)}', 
                                        xy = (bin_start_z+0.005, bin_start_S_N+0.005), 
                                        xycoords='data', size=5)
                   
            
                
                S_N[centers_z[i],centers_mass[j],centers_S_N[k]] = np.mean(df_select['S/N'].values)
                if red == 'z_spec':
                    B_T[centers_z[i],centers_mass[j],centers_S_N[k]] = np.mean(df_select['B/T'].values)
                
                
                count[centers_z[i],centers_mass[j],centers_S_N[k]] = len(df_select)
                
                '''
                # if the selection is > 0 then plot:
                plot_red = False
                if len(df_select) > 0 and plot_red:
                    plt.clf()
                    plt.scatter(final_merged[red].values, final_merged[mass].values, color='grey', s=0.1)
                    plt.scatter(df_select[red].values, df_select[mass].values, color='red', s=0.1)
                    plt.scatter(bin_center_z, bin_center_mass, color='black', s=1)
                    plt.scatter(np.median(df_select[red].values), np.median(df_select[mass].values), color='orange', s=1)
                
                    
                    plt.errorbar(med_x, med_y, 
                            xerr = std_x,
                            yerr = std_y,
                            color='orange')
                    
                    if off_in_mass:
                        plt.annotate('off in mass', xy=(0.03,0.93), xycoords='axes fraction')
                    if off_in_z:
                        plt.annotate('off in z', xy=(0.03,0.97), xycoords='axes fraction')
                    
                    
                    if off_in_mass or off_in_z:
                        plt.title('flagging')
                        
                    plt.show()
                '''
                df_select = df_select[['ID']] # just take this because you don't need the other deets
            

                merged = table.merge(df_select, on = 'ID')#left_on='ID', right_on='objID')
                # for each column of p_merg, calculate the the f_merg and then find the median
            

                gt = (merged > 0.5).apply(np.count_nonzero)
                
            

                #fmerg_here = len(np.where(merged['p_merg_x'] > 0.5)[0])/len(merged)
                
                #f_merg[centers_z[i]].append(fmerg_here)
                f_merg_avg[centers_z[i],centers_mass[j],centers_S_N[k]] = np.median(gt.values[1:]/len(merged))
                f_merg_std[centers_z[i],centers_mass[j],centers_S_N[k]] = np.std(gt.values[1:]/len(merged))
    plt.show()
    
    STOP
    
   

    # find a way to put this into df format
    mass_val = []
    z_val = []
    f_merg_val = []
    f_merg_e_val = []
    count_val = []
    s_n_val = []
    flag_val = []
    if red == 'z_spec':
        b_t_val = []
    for i in range(len(bins_z)-1):
        for j in range(len(bins_mass)-1):
            for k in range(len(bins_S_N)-1):
                f_merg_val.append(f_merg_avg[centers_z[i],centers_mass[j],centers_S_N[k]])
                f_merg_e_val.append(f_merg_std[centers_z[i],centers_mass[j],centers_S_N[k]])
                flag_val.append(flag[centers_z[i],centers_mass[j],centers_S_N[k]])
                z_val.append(centers_z[i])
                mass_val.append(centers_mass[j])
                count_val.append(count[centers_z[i],centers_mass[j],centers_S_N[k]])
                s_n_val.append(centers_S_N[k])
                
    # Now make a df out of these lists
    df_fmerg = pd.DataFrame(list(zip(flag_val, mass_val, z_val, f_merg_val, f_merg_e_val, count_val, s_n_val)),columns =['flag', 'mass', 'z', 'fmerg', 'fmerg_std', 'count', 'S/N'])
    #except:
    #		df_fmerg = pd.DataFrame(list(zip(mass_val, z_val, f_merg_val, f_merg_e_val, count_val, s_n_val, b_t_val)),
    #           columns =['mass', 'z_x', 'fmerg', 'fmerg_std', 'count', 'S/N', 'B/T'])
    df_fmerg.to_csv(table_name, sep='\t')
else:
    df_fmerg = pd.read_csv(table_name, sep = '\t')

print(df_fmerg)
print('sum of counts', np.sum(df_fmerg['count'].values))
print('table name', table_name)

df_fmerg = df_fmerg[df_fmerg['count'] > 100]

# total up the count in all of these columns

centers_mass = df_fmerg.mass.unique()
centers_z = df_fmerg.z.unique()
centers_S_N = df_fmerg['S/N'].unique()

print('centers of mass', centers_mass)
print('centers z', centers_z)
print('centers S/N', centers_S_N)

# Run some OLS
X = df_fmerg[['mass', 'z', 'S/N']] 
y = df_fmerg['fmerg']
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
print(est.summary())

# also do a standardized fit
from sklearn import preprocessing
# l1, l2, max don't really make much of a difference
# still getting super small #s when normalized
# Used to be Normalizer
normalizer = preprocessing.StandardScaler()#norm='l1')
X = df_fmerg[['mass', 'z', 'S/N']] 
normalized_train_X = normalizer.fit_transform(X)


X_std = normalizer.transform(X)

## fit a OLS model with intercept on mass and z
X = sm.add_constant(X_std)
est = sm.OLS(y, X).fit()
print(est.summary())
STOP

X = df_fmerg[['mass', 'z','S/N']] 
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF = pd.Series([variance_inflation_factor(X, i) 
               for i in range(X.shape[1])], 
              index=['mass','z','S/N'])
print(VIF)

