'''
~~~
Mergers all different classifications for all SDSS
Galaxies together into one massive table
And also get p_merg values and errors from marginalizing
across all priors :)
~~~
'''

# import modules
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import glob
import matplotlib.pyplot as plt



# prefix where all tables go to die
prefix = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'
type_marginalized = '_flags_cut_segmap'

# Grab all object IDs you've classified in SDSS:

df_predictors = pd.io.parsers.read_csv(prefix+'SDSS_predictors_all_flags_plus_segmap.txt', sep='\t')
df_predictors = df_predictors[['ID', 'Gini', 'M20', 'Concentration (C)',
       'Asymmetry (A)', 'Clumpiness (S)', 'Sersic N', 'Shape Asymmetry (A_S)',
       'S/N', 'low S/N', 'outlier predictor', 'segmap']]
df_just_id = df_predictors[['ID']]


# Import the probability values for all SDSS galaxies
merger_type_list = ['major_merger','major_merger_early','major_merger_late','major_merger_prec','major_merger_postc_include_coal_0.5','major_merger_postc_include_coal_1.0',
    'minor_merger','minor_merger_early','minor_merger_late','minor_merger_prec','minor_merger_postc_include_coal_0.5','minor_merger_postc_include_coal_1.0']
df_LDA_save_list = []
#'minor_merger_postc_include_coal_0.5'

for merger_type in merger_type_list:
    print('running this table', merger_type)
    
    # Instead of just getting one LDA value (from fiducial priors)
    # get and average (median) all of the p_merg values
    list_of_prior_files = glob.glob(prefix + 'change_prior/LDA_out_all_SDSS_predictors_'+str(merger_type)+'_0.*'+str(type_marginalized)+'.txt')
    print('length of prior files', len(list_of_prior_files))
    if len(list_of_prior_files) == 46:
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
    
    else:
        continue
    
    '''
    color_list = ['#CB48B7','#2E2D4D','#337357','#6D9F71','#E4E3D3']
    # plot the p_merg values and the associated median and std we end up pulling from them for x # of galaxies
    counter = 0
    for i in range(len(table)):
        if np.percentile(table.iloc[i,1:].values, 50) < 0.1:
            continue
        if counter > 4:
            break
        
        # Okay actually plot the distribution
        hist = np.histogram(table.iloc[i,1:].values, bins=10)
        #plt.hist(table.iloc[i,1:].values, bins=10)
        #plt.show()
        
        hist_dist = scipy.stats.rv_histogram(hist)
       
        
        # 68 is one standard deviation
        error_down = np.percentile(table.iloc[i,1:].values, 16)
        error_up = np.percentile(table.iloc[i,1:].values, 84)
        med_perc = np.percentile(table.iloc[i,1:].values, 50)
        
        print('minus one std',np.percentile(table.iloc[i,1:].values, 16))
        print('med',np.percentile(table.iloc[i,1:].values, 50))
        
        print('plus one std',np.percentile(table.iloc[i,1:].values, 84))

        
        
        plt.scatter(np.arange(0.05,0.51,0.01), table.iloc[i,1:].values, color=color_list[counter])
        med = table.iloc[i,1:].median()
        std = table.iloc[i,1:].std()
        print('med and std before', med, std)
        plt.axhline(y = med - std, color=color_list[counter])
        plt.axhline(y = med + std, color=color_list[counter])
        plt.axhline(y = error_down, color=color_list[counter], ls=':')
        plt.axhline(y = error_up, color=color_list[counter], ls=':')
        #print('shapes of error', len(0.))
        #print(len(table.iloc[:,1:].median(axis=1)))
        #plt.errorbar(0, table.iloc[:,1:].median(axis=1), yerr = table.iloc[:,1:].std(axis=1), 
        #             linestyle=None, color=color_list[i])
        counter+=1
    plt.xlabel(r'$\pi$')
    plt.ylabel('p_merg')
        
    plt.show()
    STOP
    '''
    
    table_just_p = table.iloc[:,1:]
    print(table_just_p)
    
    table['p_merg_stat_16'] = table_just_p.apply(lambda row : np.percentile(row,16), axis = 1)
    table['p_merg_stat_50'] = table_just_p.apply(lambda row : np.percentile(row,50), axis = 1)
    table['p_merg_stat_84'] = table_just_p.apply(lambda row : np.percentile(row,84), axis = 1)
    
    
    #table['p_merg_med'] = table.iloc[:,2:].median(axis=1)
    #table['p_merg_std'] = table.iloc[:,2:].std(axis=1)
    
    #df_LDA = table[['ID','p_merg_med','p_merg_std']]
    df_LDA = table[['ID','p_merg_stat_16','p_merg_stat_50','p_merg_stat_84']]
    df_LDA = df_LDA.astype({'ID': 'int64'})

    print('length of LDA', len(df_LDA))
    merged = df_just_id.merge(df_LDA, on='ID')
    print(merged)

    # do the cdf calculation
    p_vals = merged['p_merg_stat_50'].values

    spacing = 10000 # this will be the histogram binning but also how finely sampled the CDF is                                                     
    hist = np.histogram(p_vals, bins=spacing)
    # Put this in continuous distribution form in order to calculate the CDF                                                                            \
    hist_dist = scipy.stats.rv_histogram(hist)
    merged['cdf'] = hist_dist.cdf(merged['p_merg_stat_50'])

    df_LDA_save_list.append(merged)
    
    


    

# Now go through all of the LDAs and make one big df with all of the classifications
#df_LDAs_combo = pd.DataFrame()
counter = 0
for LDA in df_LDA_save_list:
    # select the cols you want out of each LDA df
    if counter == 0:
        df_LDAs_combo = LDA[['ID','p_merg_stat_16','p_merg_stat_50','p_merg_stat_84','cdf']]
        df_LDAs_combo.columns = ['ID', 
            'p_merg_stat_16_'+str(merger_type_list[counter]), 
            'p_merg_stat_50_'+str(merger_type_list[counter]),
            'p_merg_stat_84_'+str(merger_type_list[counter]),
            'cdf_'+str(merger_type_list[counter])]
    else:
        select = LDA[['p_merg_stat_16','p_merg_stat_50','p_merg_stat_84','cdf']]
        select.columns = [
            'p_merg_stat_16_'+str(merger_type_list[counter]), 
            'p_merg_stat_50_'+str(merger_type_list[counter]),
            'p_merg_stat_84_'+str(merger_type_list[counter]),
            'cdf_'+str(merger_type_list[counter])]
        
        # Now add this collection to the df
        df_LDAs_combo = df_LDAs_combo.join(select)
    counter+=1
    

# I decided that we don't actually need any of the predictors cross-matched
#df_merged = df_LDAs_combo.merge(df_predictors, on='ID')
df_merged = df_LDAs_combo
print(df_merged)

df_merged.to_csv(prefix+'LDA_out_all_SDSS_predictors_all_priors_percentile_all_classifications_flags.txt', sep='\t')

