'''
~~~
This code allows the user to input a single or list of galaxies by Object ID.
The output is the list of parameter values and the list of LDA values and classifications.
The code also optionally plots the galaxy with the probabilities and LDA values.
Requires: the LDA_out_* tables that have probability values and the SDSS_predictor* table which has predictor values for all SDSS photometric galaxies (DR16).
~~~
'''

# import modules
import pandas as pd
import numpy as np
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import coordinates as coords
import astropy.io.fits as fits
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from util_SDSS import SDSS_objid_to_values, download_galaxy

# This is code from compare_mpmerg_to_full_population_CDF.py
# But here for a list of p values
def calculate_cdf(p_vals, p_list, percent):
    # Define a histogram with spacing defined                                                                                                                
    spacing = 1000 # this will be the histogram binning but also how finely sampled the CDF is                                                               
    hist = np.histogram(p_vals, bins=spacing)

    # Put this in continuous distribution form in order to calculate the CDF                                                                                 
    hist_dist = scipy.stats.rv_histogram(hist)

    # Find individual cdf values corresponding to a p_merg value                                                                                             
    cdf_list = []
    for p in p_list:
        cdf_list.append(hist_dist.cdf(p))
        

    # Define the xs of this distribution                                                                                                                     
    X = np.linspace(0, 1.0, spacing)

    # Get all cdf values                                                                                                                                     
    cdf_val = [hist_dist.cdf(x) for x in X]
    # Find the x point at which the cdf value is 10% and 90% - 0.1 and 0.9 (can replace this with your own thresholds)                                       
    idx_non, val_non = find_nearest(np.array(cdf_val), percent)
    X_non = X[idx_non]

    idx_merg, val_merg = find_nearest(np.array(cdf_val), 1 - percent)
    X_merg =X[idx_merg]

    print('p_merg value is ', X_non, 'when ',val_non,' of the full population has a lower p_merg value')
    print('p_merg value is ', X_merg, 'when ',1-val_merg,' of the full population has a higher p_merg value')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]




# Step 1: import the SDSS galaxies

type_gal = 'predictors'
merger_type = 'major_merger'
plot = True
size = 80
 
prefix = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger/Tables/'

df_predictors = pd.io.parsers.read_csv(prefix+'SDSS_'+str(type_gal)+'_all_flags.txt', sep='\t')

#df_predictors.columns = ['ID','Sep','Flux Ratio',  'Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)', 'Sersic AR', 'S/N', 'Sersic N Statmorph', 'A_S Statmorph']

if len(df_predictors.columns) ==15: #then you have to delete the first column which is an empty index
    df_predictors = df_predictors.iloc[: , 1:]
  
# Change the type of the ID in predictors:
df_predictors = df_predictors.astype({'ID': 'int64'})#.dtypes                                                                                     


# Step 2: import the probability values for these galaxies

df_LDA = pd.io.parsers.read_csv(prefix+'LDA_out_all_SDSS_'+str(type_gal)+'_'+str(merger_type)+'_flags.txt', sep='\t')
#print(df_LDA.columns, df_predictors.columns)
df_LDA = df_LDA.astype({'ID': 'int64'})


# Step 3: match from the list of IDs:
ID_list = [1237661125071208589, 1237654654171938965, 1237651212287475940, 1237659325489742089, 1237651273490628809, 1237661852007596057, 1237657878077702182, 1237667912748957839, 1237662665888956491, 1237655504567926924, 0, 1237664673793245248, 1237653009194090657, 1237667212116492386, 1237660024524046409, 1237654949448646674, 1237656496169943280, 1237663784217804830, 1237673706113925406, 1237656567042539991, 1237653587947815102, 1237651191354687536, 1237661387069194275, 1237651226784760247, 1237658204508258428, 1237661957225119836, 1237653589018018166, 1237651251482525781, 1237658802034573341, 1237663457241268416, 1237663529718841406, 1237651272956641566, 1237667910601932987, 1237659326029365299, 1237661852538437650, 1237665549422952539, 1237659327099896118, 1237651212287672564, 1237666299480309878, 1237657856607649901, 1237654952670789707, 1237654949448450067, 1237660241386143913, 1237652899700998392, 1237664837002395706, 1237654626785821087, 1237654391639638190]

RA_list = [118.074345115, 119.617127519, 258.548475763, 241.150984777, 114.096383278, 189.213252539, 120.087417603, 205.690424787, 246.255977023, 251.335933471, 331.12290045, 205.753337262, 316.841308073, 127.170800449, 46.2941968423, 234.541843983, 319.193098655, 46.6649126989, 111.733682055, 322.213310988, 127.178093813, 123.820325773, 217.629970676, 262.399282723, 173.537567139, 215.017906947, 119.486337418, 123.330544326, 171.400653635, 321.00791167, 118.184152672, 119.182151794, 206.627529147, 247.159333462, 169.513447059, 206.007949798, 241.271446954, 258.82741019, 46.7170694138, 171.657261919, 255.029869751, 233.968342684, 47.1429889739, 61.4532623945, 131.725410562, 114.695391123, 118.855393765]

dec_list = [19.5950803976, 37.7866245591, 57.9760977732, 43.8797876113, 39.4382798053, 45.6511702678, 26.613527298, 24.5900537916, 24.2631556488, 42.757790046, 12.4426263785, 36.1656555546, 11.0664208661, 17.5814003238, -1.07547036487, 57.6036530229, 11.0437407875, 0.0619768826479, 41.0266910782, -1.07011823351, 45.7425550277, 46.0752533433, 52.7071590288, 54.4944236725, 49.2545623779, 47.1213302326, 39.9933651629, 46.1471565611, 54.3825744827, -0.366323610281, 45.949276388, 44.8567085266, 22.7060191923, 39.551266027, 45.1130288476, 25.9411986626, 45.4429921738, 57.6587701679, -0.896544995426, 51.5730412626, 37.8395021377, 57.9026365222, 0.55093690996, -6.32383706666, 25.3700887716, 29.8912832677, 39.1860944299]

# Testing if A, B, and C still exist in the catalog:
#id_list = [1237665179521187863, 1237661069252231265, 1237665329864114245]

index_save_LDA = []
index_save_predictors = []
cdf_list = []

# get all p_values from the sample:
p_vals = df_LDA['p_merg'].values

spacing = 10000 # this will be the histogram binning but also how finely sampled the CDF is                                                     
hist = np.histogram(p_vals, bins=spacing)
# Put this in continuous distribution form in order to calculate the CDF                                                                            \
hist_dist = scipy.stats.rv_histogram(hist)

# Define the xs of this distribution                                                                                                                     
X_lin = np.linspace(-100,100,10000)
X = [1/(1+np.exp(-x)) for x in X_lin]# transform into logit space

# Get all cdf values
cdf_val = [hist_dist.cdf(x) for x in X]

idx_non, val_non = find_nearest(np.array(cdf_val), 0.1)
X_non = X[idx_non]

idx_merg, val_merg = find_nearest(np.array(cdf_val),0.9)
X_merg =X[idx_merg]

print('p_merg value is ', X_non, 'when ',val_non,' of the full population has a lower p_merg value')
print('p_merg value is ', X_merg, 'when ',1-val_merg,' of the full population has a higher p_merg value')

indices_preds = df_predictors.index




for i in range(len(ID_list)):
    id = ID_list[i]

    print('trying to find a match for this', id, type(id))
    ra = RA_list[i]
    dec = dec_list[i]
    # Find each item in each data frame:

    
    where_LDA = np.where(np.array(df_LDA['ID'].values)==id)[0]
    
    #print('trying to find where preds', df_predictors.loc[df_predictors['ID']==id])
    
    condition = df_predictors["ID"] == id
    try:
        where_predictors = indices_preds[condition].values.tolist()[0]
    except:
        continue

    print('where preds', where_predictors)
    '''

    try:
        where_predictors = df_predictors.index(df_predictors.loc[df_predictors['ID']==id])
    
        print('where preds', where_predictors)
    except TypeError:
        continue
        STOP
    #np.where(df_predictors['ID'].values==id)[0]
    '''
    
    # get the corresponding cdf value:
    try:
        cdf = hist_dist.cdf(df_LDA.values[where_LDA][0][4])
    except IndexError:
        # This means there was no match
        continue
    cdf_list.append(cdf)

    if where_LDA.size != 0:

        # so IF it exists, then go ahead and download it:
        



        if plot:
            img = download_galaxy(id, ra, dec, prefix+'../frames/', size)

            shape = np.shape(img)[0]
            
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(abs(img), norm=matplotlib.colors.LogNorm())
            ax.annotate('LD1 = '+str(round(df_LDA.values[where_LDA][0][3],2))+'\n$p_{\mathrm{merg}}$ = '+str(round(df_LDA.values[where_LDA][0][4],4))+'\nCDF = '+str(round(cdf,4)), xy=(0.03, 0.85),  xycoords='axes fraction',
            xytext=(0.03, 0.85), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9"), color='black')
            if df_LDA.values[where_LDA][0][8] ==1 or df_LDA.values[where_LDA][0][9] ==1:
                ax.annotate('Photometric Flag', xy=(0.8, 0.85),  xycoords='axes fraction',
            xytext=(0.8, 0.85), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9"), color='black')
            ax.set_title('ObjID = '+str(id))
            ax.set_xticks([0, (shape - 1)/2, shape-1])
            ax.set_xticklabels([-size/2, 0, size/2])
            ax.set_yticks([0, (shape- 1)/2,shape-1])
            ax.set_yticklabels([-size/2, 0,size/2])
            ax.set_xlabel('Arcsec')
            plt.savefig(prefix+'../Figures/ind_galaxies_classify/'+str(id)+'_'+str(merger_type)+'.png', dpi=1000)
            
        index_save_LDA.append(where_LDA[0])
        index_save_predictors.append(where_predictors)

index_save_LDA = np.array(index_save_LDA)
index_save_predictors = np.array(index_save_predictors)

print(index_save_predictors)

print('length of input table', len(ID_list), 'length that we found in the table', len(index_save_LDA), 'length found in predictors', len(index_save_predictors))        


df_LDA_save = pd.DataFrame(df_LDA.values[index_save_LDA], columns = df_LDA.columns)
df_predictors_save = pd.DataFrame(df_predictors.iloc[index_save_predictors], columns = df_predictors.columns)

'''
df_i = df_predictors.loc[df_predictors['ID']==id]
print(df_predictors[df_predictors['ID']==id])
print(df_i)
print(df_predictors[df_i])
STOP
'''

print(df_LDA_save.dtypes['ID'], df_predictors_save.dtypes['ID'])
df_predictors_save = df_predictors_save.astype({'ID': 'int64'})#.dtypes

print(df_predictors_save)
print(df_LDA_save.dtypes['ID'], df_predictors_save.dtypes['ID'])

print(df_LDA_save)
print(df_predictors_save)

df_merged = df_LDA_save.merge(df_predictors_save, on='ID')

print(df_merged)

df_merged.to_csv(prefix+'classification_out_and_predictors_'+str(merger_type)+'.txt', sep='\t')
STOP        
        
	
		
'''
if plot:
    ra = RA_list[i]
    dec = dec_list[i]
    
    download_galaxy(id, ra, dec, prefix+'imaging/')'''

STOP

# Filter out the bad parameter values - high n values:
'''
df_bad = df2[df2['Sersic N'] > 10]
pd.set_option('display.max_columns', 20)
print('bad sersic', df_bad)

df_bad = df2[df2['Asymmetry (A)'] <-1]
print('bad A', len(df_bad))
print(df_bad)

'''

# First, delete all rows that have weird values of n:
print('len before crazy values', len(df2))
df_filtered = df2[df2['Sersic N'] < 10]

df_filtered_2 = df_filtered[df_filtered['Asymmetry (A)'] > -1]

df2 = df_filtered_2

# Delete duplicates:
print('len bf duplicate delete', len(df2))
df2_nodup = df2.duplicated()
df2 = df2[~df2_nodup]
print('len af duplicate delete', len(df2))

# make it way shorter
#df2 = df2[50000:57000]

input_singular = terms_RFR
#Okay so this next part actually needs to be adaptable to reproduce all possible cross-terms
crossterms = []
ct_1 = []
ct_2 = []
for j in range(len(input_singular)):
    for i in range(len(input_singular)):
        if j == i or i < j:
            continue
        #input_singular.append(input_singular[j]+'*'+input_singular[i])
        crossterms.append(input_singular[j]+'*'+input_singular[i])
        ct_1.append(input_singular[j])
        ct_2.append(input_singular[i])

inputs = input_singular + crossterms

# Now you have to construct a bunch of new rows to the df that include all of these cross-terms
for j in range(len(crossterms)):
    
    df2[crossterms[j]] = df2.apply(cross_term, axis=1, args=(ct_1[j], ct_2[j]))
    

X_gal = df2[inputs_all].values




X_std=[]
testing_C=[]
testing_A=[]
testing_Gini=[]
testing_M20=[]
testing_n=[]
testing_A_S=[]

testing_n_stat=[]
testing_A_S_stat=[]
testing_S_N=[]
testing_A_R = []

LD1_SDSS=[]
p_merg_list=[]
score_merg=[]
score_nonmerg=[]

if run[0:12]=='major_merger':
    prior_nonmerg = 0.9
    prior_merg = 0.1
else:
    if run[0:12]=='minor_merger':
        prior_nonmerg = 0.7
        prior_merg = 0.3
        
    else:
        STOP

