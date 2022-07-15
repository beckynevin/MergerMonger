'''
~~~
The purpose of this code is to start with an image or a list of images
and return a table of predictor values

The default is SDSS, so this code includes an example of how to use
wget to grab SDSS frame images and find galaxies using their coordinates.

This could conceivably by turned into a code that works with pre-downloaded
galaxy images
~~~
'''

# import modules
import os
import pandas as pd
import numpy as np
import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
from util_SDSS import download_galaxy
from util_smelter import get_predictors



# optional name of your sample, I usually put the name of the
# table if you want to load in a table
appellido = 'testing'

print('appellido is', appellido)

# This is if you want to load in the galaxy IDs from a table
load = False
# option to make and save a plot
plot = True
# diameter size of the cutout in arcsec
size = 80

# where you want to store everything:
# you should have a sub-folder in here called 'imaging' that will store all of the
# files this code creates for galfit and source extractor and statmorph
# as well as a sub-folder called 'Figures'
prefix = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/'
if os.path.isdir(prefix + 'imaging'):
    pass
else:
    print('missing imaging directory')
    

if os.path.isdir(prefix + 'Figures'):
    pass
else:
    print('missing Figures directory')
    
if os.path.isdir(prefix + 'Figures/'+str(appellido)):
    pass
else:
    print('missing folder to store images (appellido)')
    
if os.path.isdir(prefix + 'Tables'):
    pass
else:
    print('missing Table directory')
    



# to load up the IDs for the galaxies, I either use a list or I look them up from a table:
ID_list = [1237657775542698092,
 1237655129839108147, 1237659330848817218, 1237656895066931656,
 1237674288607461402, 1237658802572558360, 1237655130376241254,
 1237655473967136890, 1237661386538614980, 1237662336796065949,
 1237654391639376175, 1237660024523849823, 1237664835392241744,
 1237651753456107524]

ID_list = [1237654390032629943,1237661464918688114,1237664667887534206]



if load:
    try:
        print('loading in fits')
        # Get ID list instead from Dave's tables:
        fits_table = fits.open('../Tables/'+appellido+'.fits')
            #drpall-v3_1_1_objid.fits')

        

        ID_list = fits_table[1].data['objid']#[0:10]
        print('IDs', ID_list)
        print('length', len(ID_list))
    except:
        print('loading in CSV')
        data=pd.read_csv('../Tables/'+appellido+'.csv')#,  names=cols, skiprows=1, header=None)

        ID_list = data['objID'].values


# option for SDSS to look up RA and dec:
ra_dec_lookup = pd.read_csv(prefix + 'Tables/five_sigma_detection_saturated_mode1_beckynevin.csv')


# Do this in a better way than a double for loop:
RA_list = []
dec_list = []
for j in range(len(ID_list)):
    try:
        RA_list.append(ra_dec_lookup[ra_dec_lookup['objID']==ID_list[j]]['ra'].values[0])
        dec_list.append(ra_dec_lookup[ra_dec_lookup['objID']==ID_list[j]]['dec'].values[0])
    except IndexError:
        RA_list.append(0)
        dec_list.append(0)


    
# Now going to loop through the ID_list and the RA_list and dec_list
# 1) looks up these galaxies (I use download_galaxy from util_SDSS.py to do this)
# 2) runs get_predictors from util_smelter.py to grab predictor values

save_df = pd.DataFrame(columns=['ID','Gini','M20','C','A','S','n','A_s','S/N'])

for i in range(len(ID_list)):
    id = ID_list[i]

    print('trying to find a match for this', id, type(id), i)
    ra = RA_list[i]
    dec = dec_list[i]
    
    # so the below produces img, which is an 80x80 arcsec cutout
    # of a sdss frame image centered on the ra and dec
    img = download_galaxy(id, ra, dec, prefix+'frames/', size)
    try:
        shape = np.shape(img)[0]
    except IndexError:
        # meaning it returned with an error
        continue
            

    img = np.array(img)
    # Check if the file already exists:
    preds = get_predictors(id, img, prefix)
    # preds gets you all the imaging predictors:
    camera_data = preds[0]
    segmap = preds[1]
    gini = preds[2]
    m20 = preds[3]
    con = preds[4]
    asy = preds[5]
    clu = preds[6]
    ser = preds[7]
    A_s = preds[8]
    s_n = preds[10]
    
    # save these by modifying a row of the dataframe:
    
    save_df.loc[i,:] = [id,gini,m20,con,asy,clu,ser,A_s,s_n]
    #data.loc[0:2,['Num','NAME']] = [100,'Python']
    try:
        segmap = preds[1]
        shape = np.shape(segmap)[0]
    except TypeError:
        print('No segmap')
                
    if plot:
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(abs(img), norm=matplotlib.colors.LogNorm(vmax=10**5, vmin=10**(0.5)), 
            cmap='afmhot', interpolation='None')

        ax.contour(np.fliplr(np.rot90(preds[1])), levels=[0,1], colors='yellow')
       
        
        ax.annotate('Gini = '+str(round(preds[2],2))+
            r' M$_{20}$ = '+str(round(preds[3],2))+
            '\nC = '+str(round(preds[4],2))+
            ' A = '+str(round(preds[5],2))+
            ' S = '+str(round(preds[6],2))+
            '\nn = '+str(round(preds[7],2))+
            r' A$_S$ = '+str(round(preds[8],2))+
            '\n<S/N> = '+str(round(preds[10],2)), 
            xy=(0.03, 0.05),  xycoords='axes fraction',
        xytext=(0.03, 0.05), textcoords='axes fraction',
        bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')

        
        
        
        ax.set_title('ID = '+str(id))
        ax.set_xticks([0, (shape - 1)/2, shape-1])
        ax.set_xticklabels([-size/2, 0, size/2])
        ax.set_yticks([0, (shape- 1)/2,shape-1])
        ax.set_yticklabels([-size/2, 0,size/2])
        ax.set_xlabel('Arcsec')
        plt.savefig(prefix+'Figures/'+str(appellido)+'/image_'+str(id)+'.png', dpi=1000)
        

print(save_df)

# save it
save_df.to_csv(prefix + 'Tables/'+str(appellido)+'_predictor_values.txt', sep='\t')
