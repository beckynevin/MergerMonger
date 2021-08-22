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
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import coordinates as coords
import astropy.io.fits as fits
import os

def SDSS_objid_to_values(objid):

    # Determined from http://skyserver.sdss.org/dr7/en/help/docs/algorithm.asp?key=objID                                                                                                   

    bin_objid = bin(objid)
    bin_objid = bin_objid[2:len(bin_objid)]
    bin_objid = bin_objid.zfill(64)

    empty = int( '0b' + bin_objid[0], base=0)
    skyVersion = int( '0b' + bin_objid[1:4+1], base=0)
    rerun = int( '0b' + bin_objid[5:15+1], base=0)
    run = int( '0b' + bin_objid[16:31+1], base=0)
    camcol = int( '0b' + bin_objid[32:34+1], base=0)
    firstField = int( '0b' + bin_objid[35+1], base=0)
    field = int( '0b' + bin_objid[36:47+1], base=0)
    object_num = int( '0b' + bin_objid[48:63+1], base=0)

    return skyVersion, rerun, run, camcol, field, object_num

def download_galaxy(ID, RA, DEC, prefix_frames):
    decode=SDSS_objid_to_values(ID)
    if decode[2] < 1000:
        pref_run = '000'
    else:
        pref_run = '00'
        
    if decode[4] > 100:
        pref_field = '0'
    else:
        pref_field = '00'
    
    name = prefix_frames + 'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits'


    os.system('wget https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/'+str(decode[2])+'/'+str(decode[3])+'/frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2')

    print('this is the wget')
    print('wget https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/'+str(decode[2])+'/'+str(decode[\
3])+'/frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2')

    os.system('bunzip2 frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2')

    im=fits.open(prefix_frames + 'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits')

    obj_coords = SkyCoord(str(ra)+' '+str(dec),unit=(u.deg, u.deg), frame='icrs')

    size = u.Quantity((80,80), u.arcsec)#was 80,80                                             
    wcs_a = WCS(im[0].header)

    stamp_a = Cutout2D(im[0].data, obj_coords, size, wcs=wcs_a)#was image_a[0].data    

    camera_data=(np.fliplr(np.rot90(stamp_a.data))/0.005)

    im.close()

    return camera_data


# Step 1: import the SDSS galaxies

type_gal = 'predictors'
merger_type = 'major_merger'
plot = 'False'
 
prefix = '/Users/beckynevin/CfA_Code/MergerMonger/Tables/'

df_predictors = pd.io.parsers.read_csv(prefix+'SDSS_'+str(type_gal)+'_all.txt', sep='\t')

#df_predictors.columns = ['ID','Sep','Flux Ratio',  'Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)', 'Sersic AR', 'S/N', 'Sersic N Statmorph', 'A_S Statmorph']

if len(df_predictors.columns) ==15: #then you have to delete the first column which is an empty index
    df_predictors = df_predictors.iloc[: , 1:]
  


# Step 2: import the probability values for these galaxies

df_LDA = pd.io.parsers.read_csv(prefix+'LDA_out_all_SDSS_'+str(type_gal)+'_'+str(merger_type)+'.txt', sep='\t')
#print(df_LDA.columns, df_predictors.columns)

# Step 3: match from the list of IDs:
ID_list = [1237646379932320786,
		1237663783674314797,
		1237664668971958306,
		1237661387070308495,
		1237655742409539752,
		1237665429176713295,
		1237659153148608547]

RA_list = [197.614455635, 197.614455635]
#[02:06:15.990000, 02:06:15.990000]

dec_list = [18.438168849,18.438168849]
#[-00:17:29.20000, -00:17:29.20000]

for i in range(len(ID_list)):
	id = ID_list[i]

	
	# Find each item in each data frame:
	where_LDA = np.where(df_LDA['MaNGA_ID'].values==id)[0]
	if where_LDA.size != 0:
		print(df_LDA.values[where_LDA])

		where_predictors = np.where(df_predictors['ID'].values==id)[0]
		print(df_predictors.values[where_predictors])


		print('where 1', where_LDA, 'where 2', where_predictors)

		# It would be quite cool if there was an option to download the frame image
                '''
		if plot:
			ra = RA_list[i]
			dec = dec_list[i]
			
			download_galaxy(id, ra, dec, prefix+'imaging/')
                '''
	else:
		continue
		
	

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

