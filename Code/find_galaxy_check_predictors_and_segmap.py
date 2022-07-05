# I'm not entirely convinced find_galaxy doesn't do this all cleaner (it makes the statmorph check file)
'''
~~~
This code allows the user to input a single or list of galaxies by Object ID.
The output is the list of parameter values and the segmap used by statmorph.
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
from util_smelter import get_predictors
from util_LDA import calculate_cdf, find_nearest



appellido = 'dave_interp'

# Get ID list instead from Dave's tables:
fits_table = fits.open('../Tables/breakbrd_objids.fits')
    #drpall-v3_1_1_objid.fits')

ID_list = fits_table[1].data['objid']
print('IDs', ID_list)


# Step 1: import the SDSS galaxies

merger_type = 'major_merger'
plot = True
size = 80
 
prefix = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'

df_predictors = pd.io.parsers.read_csv(prefix+'SDSS_predictors_with_flags.txt', sep='\t')

#df_predictors.columns = ['ID','Sep','Flux Ratio',  'Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)', 'Sersic AR', 'S/N', 'Sersic N Statmorph', 'A_S Statmorph']

if len(df_predictors.columns) ==15: #then you have to delete the first column which is an empty index
    df_predictors = df_predictors.iloc[: , 1:]
  
# Change the type of the ID in predictors:
df_predictors = df_predictors.astype({'ID': 'int64'})#.dtypes                                                                                     


# Step 2: import the probability values for these galaxies

df_LDA = pd.io.parsers.read_csv(prefix+'LDA_out_all_SDSS_predictors_'+str(merger_type)+'_flags_leading_preds.txt', sep='\t')
#df_LDA = pd.io.parsers.read_csv(prefix+'LDA_out_all_SDSS_predictors_'+str(merger_type)+'_flags.txt', sep='\t')
#print(df_LDA.columns, df_predictors.columns)
df_LDA = df_LDA.astype({'ID': 'int64'})


# Step 3: match from the list of IDs:
# these are galaxyzoos that are classified as mergers
#ID_list = [1237655693557170541, 1237648673456455904, 1237668649315008771, 1237668649315074455, 1237668649315205346, 1237668568247370040, 1237668649852076495, 1237668649852207363, 1237668568784437849, 1237668568784437611, 1237668568784503051, 1237668568784503170, 1237668650389340761, 1237668650926342749, 1237668650926342753, 1237668650926408199, 1237668569858638234, 1237668650926539340, 1237668569858703829, 1237668569858703926]


# What about getting the IDs straight from the predictor table?:
# First get rid of everything that is flagged:


df_no_flags = df_predictors[(df_predictors['low S/N']==0) & (df_predictors['outlier predictor']==0) & (df_predictors['segmap']==0)]
print('length of no flags', len(df_no_flags))

#ID_list = df_no_flags['ID'].values[0:100]

lookup = pd.read_csv(prefix+'clean_df_run_segmap_edge.csv')

#ID_list = lookup['ID'].values[0:100]

#ID_list = [1237646380469453821]
#ID_list = [1237646381006389406,1237646379396433282,1237646379933106243,
#    1237645941833662760,1237646379933302933,1237646379395907879,1237646585565939033,1237646585566135403]




#ID_list = [1237665179521187863,1237661069252231265,1237665329864114245]
#appellido = 'ABC'

print('IDs',ID_list)
print('appellido', appellido)

# these are galaxyzoos that are not classified as mergers
#ID_list = [1237661871347138764,1237662264322097387,1237662264325767706,1237662264848351296,1237667782823903325]
#ID_list = [1237661125071208589, 1237654654171938965, 1237651212287475940, 1237659325489742089, 1237651273490628809, 1237661852007596057, 1237657878077702182, 1237667912748957839, 1237662665888956491, 1237655504567926924, 0, 1237664673793245248, 1237653009194090657, 1237667212116492386, 1237660024524046409, 1237654949448646674, 1237656496169943280, 1237663784217804830, 1237673706113925406, 1237656567042539991, 1237653587947815102, 1237651191354687536, 1237661387069194275, 1237651226784760247, 1237658204508258428, 1237661957225119836, 1237653589018018166, 1237651251482525781, 1237658802034573341, 1237663457241268416, 1237663529718841406, 1237651272956641566, 1237667910601932987, 1237659326029365299, 1237661852538437650, 1237665549422952539, 1237659327099896118, 1237651212287672564, 1237666299480309878, 1237657856607649901, 1237654952670789707, 1237654949448450067, 1237660241386143913, 1237652899700998392, 1237664837002395706, 1237654626785821087, 1237654391639638190]
#ID_list = [1237652899700998392]
# Look up RAs and Dec from some sort of table:
# Alternately, just use your own lists of RAs and decs like so:
#RA_list = [118.074345115, 119.617127519, 258.548475763, 241.150984777, 114.096383278, 189.213252539, 120.087417603, 205.690424787, 246.255977023, 251.335933471, 331.12290045, 205.753337262, 316.841308073, 127.170800449, 46.2941968423, 234.541843983, 319.193098655, 46.6649126989, 111.733682055, 322.213310988, 127.178093813, 123.820325773, 217.629970676, 262.399282723, 173.537567139, 215.017906947, 119.486337418, 123.330544326, 171.400653635, 321.00791167, 118.184152672, 119.182151794, 206.627529147, 247.159333462, 169.513447059, 206.007949798, 241.271446954, 258.82741019, 46.7170694138, 171.657261919, 255.029869751, 233.968342684, 47.1429889739, 61.4532623945, 131.725410562, 114.695391123, 118.855393765]
#dec_list = [19.5950803976, 37.7866245591, 57.9760977732, 43.8797876113, 39.4382798053, 45.6511702678, 26.613527298, 24.5900537916, 24.2631556488, 42.757790046, 12.4426263785, 36.1656555546, 11.0664208661, 17.5814003238, -1.07547036487, 57.6036530229, 11.0437407875, 0.0619768826479, 41.0266910782, -1.07011823351, 45.7425550277, 46.0752533433, 52.7071590288, 54.4944236725, 49.2545623779, 47.1213302326, 39.9933651629, 46.1471565611, 54.3825744827, -0.366323610281, 45.949276388, 44.8567085266, 22.7060191923, 39.551266027, 45.1130288476, 25.9411986626, 45.4429921738, 57.6587701679, -0.896544995426, 51.5730412626, 37.8395021377, 57.9026365222, 0.55093690996, -6.32383706666, 25.3700887716, 29.8912832677, 39.1860944299]

# Hector's galaxies
#ID_list = [1237661125071208589, 1237654654171938965, 1237651212287475940, 1237659325489742089, 1237651273490628809, 1237661852007596057, 1237657878077702182, 1237667912748957839, 1237662665888956491, 1237655504567926924, 0, 1237664673793245248, 1237653009194090657, 1237667212116492386, 1237660024524046409, 1237654949448646674, 1237656496169943280, 1237663784217804830, 1237673706113925406, 1237656567042539991, 1237653587947815102, 1237651191354687536, 1237661387069194275, 1237651226784760247, 1237658204508258428, 1237661957225119836, 1237653589018018166, 1237651251482525781, 1237658802034573341, 1237663457241268416, 1237663529718841406, 1237651272956641566, 1237667910601932987, 1237659326029365299, 1237661852538437650, 1237665549422952539, 1237659327099896118, 1237651212287672564, 1237666299480309878, 1237657856607649901, 1237654952670789707, 1237654949448450067, 1237660241386143913, 1237652899700998392, 1237664837002395706, 1237654626785821087, 1237654391639638190]
ID_list = [1237673706113925406,1237660024524046409]

print(len(ID_list))





ra_dec_lookup = pd.read_csv('../Tables/five_sigma_detection_saturated_mode1_beckynevin.csv')
RA_list = []
dec_list = []

for j in range(len(ID_list)):
    counter = 0
    for i in range(len(ra_dec_lookup)):
        if ID_list[j]==ra_dec_lookup['objID'].values[i]:
            RA_list.append(ra_dec_lookup['ra'].values[i])
            dec_list.append(ra_dec_lookup['dec'].values[i])
            counter+=1
    if counter==0:
        RA_list.append(0)
        dec_list.append(0)
print('got RAs and Decs')
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


flag_list = []


for i in range(len(ID_list)):
    id = ID_list[i]

    print('trying to find a match for this', id, type(id))
    ra = RA_list[i]
    dec = dec_list[i]

    print('ra and dec', ra, dec)
    # Find each item in each data frame:

    
    where_LDA = np.where(np.array(df_LDA['ID'].values)==id)[0]
    
    #print('trying to find where preds', df_predictors.loc[df_predictors['ID']==id])
    
    condition = df_predictors["ID"] == id
    try:
        where_predictors = indices_preds[condition].values.tolist()[0]
    except:
        continue

    
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
            print(os.getcwd())
            print('prefix to download galaxy', prefix+'../frames/')
            print('prefix to get predictors', prefix+'../')
            
            img = download_galaxy(id, ra, dec, prefix+'../frames/', size)

            print('got the image')
            
            
            # Now check out the predictors
            
            preds = get_predictors(id, img, prefix+'../', size)
            print('preds', preds)
            try:
                shape = np.shape(preds[0])[0]
            except TypeError:
                continue


            # Okay now only plot if weird segmap values:
            segmap = preds[1]


            center_val = segmap[int(np.shape(segmap)[0]/2),int(np.shape(segmap)[1]/2)]
            # Need to make a list of all points on the edge of the map
            edges=np.concatenate([segmap[0,:-1], segmap[:-1,-1], segmap[-1,::-1], segmap[-2:0:-1,0]])

            '''
            coord_list = []
            for p in range(np.shape(segmap)[0]):
                coord_list.append((0,p))
                coord_list.append((np.shape(segmap)[0]-1,p))
                coord_list.append((p,0))
                coord_list.append((p,np.shape(segmap)[0]-1))

            print('coordinate 0', coord_list[0], coord_list[0])
            print(segmap[0,0], segmap[(0,0)])
            print('play with this', segmap[0:10,0:10])
            print('shape of coord list', np.shape(coord_list))
            print('shape of thing youre doing .any to ', np.shape(segmap[coord_list]))
            '''
            #print(coord_list)
            print('getting ready for flags')

            flag = False
            flag_name = []
            if (df_predictors.values[where_predictors][11]==0) & (df_predictors.values[where_predictors][12]==0) & (df_predictors.values[where_predictors][13]==0):
                flag_name.append('No flag')
            else:
                if df_predictors.values[where_predictors][11]==1:
                    flag_name.append('low S/N')
                if df_predictors.values[where_predictors][12]==1:
                    flag_name.append('outlier predictor')
                if df_predictors.values[where_predictors][13]==1:
                    flag_name.append('segmap')
                flag = True



            '''
            if edges.any():
                flag = 2
                flag_name = 'On edge'
            if center_val == False:
                flag = 1
                flag_name = 'Not centered'
            if (center_val == False) & edges.any():
                flag = 3
                flag_name = 'Not centered + on edge'
            # 1 is not centered, 2 is on edge, 3 is both
            #if flag > 0: # this means its been flagged for something
            '''


            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            im1 = ax1.imshow(abs(np.fliplr(np.rot90(preds[0]))), norm=matplotlib.colors.LogNorm())#, origin = 'lower'
            ax1.set_title('Image, flag = \n'+str(flag_name))
            ax1.annotate('LD1 = '+str(round(df_LDA.values[where_LDA][0][3],2))+'\n$p_{\mathrm{merg}}$ = '+str(round(df_LDA.values[where_LDA][0][4],4))+
                '\nCDF = '+str(round(cdf,4))+
                '\n'+str(df_LDA.values[where_LDA][0][6])+'\n'+str(df_LDA.values[where_LDA][0][8])+'\n'+str(df_LDA.values[where_LDA][0][10]), 
                xy=(0.03, 0.7),  xycoords='axes fraction',
            xytext=(0.03, 0.7), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')

            if flag:
                ax1.annotate('Photometric Flag', xy=(0.5, 0.85),  xycoords='axes fraction',
            xytext=(0.5, 0.85), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9"), color='black')

            ax2 = fig.add_subplot(122)
            im2 = ax2.imshow(np.fliplr(np.rot90(preds[1])))
            ax2.scatter(np.shape(segmap)[0]/2,np.shape(segmap)[0]/2, color='red')
            ax2.set_title('Segmap, <S/N> = '+str(round(preds[10],2)))
            ax2.annotate('Gini = '+str(round(df_predictors.values[where_predictors][3],2))+
                ' M20 = '+str(round(df_predictors.values[where_predictors][4],2))+
                '\nC = '+str(round(df_predictors.values[where_predictors][5],2))+
                '\nA = '+str(round(df_predictors.values[where_predictors][6],2))+
                '\nS = '+str(round(df_predictors.values[where_predictors][7],2))+
                '\nn = '+str(round(df_predictors.values[where_predictors][8],2))+
                '\nA_S = '+str(round(df_predictors.values[where_predictors][9],2)), 
                xy=(0.03, 0.55),  xycoords='axes fraction',
            xytext=(0.03, 0.55), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')
            ax1.set_xticks([0, (shape - 1)/2, shape-1])
            ax1.set_xticklabels([-size/2, 0, size/2])
            ax1.set_yticks([0, (shape- 1)/2,shape-1])
            ax1.set_yticklabels([-size/2, 0,size/2])
            ax1.set_xlabel('Arcsec')
            plt.savefig(prefix+'../Figures/ind_galaxies_classify/statmorph_check_'+str(id)+'_'+str(merger_type)+'_'+appellido+'.png', dpi=1000)
            if flag:
                flag_list.append(1)
            else:
                flag_list.append(0)
print(np.sum(flag_list), 'out of this length', len(flag_list))
            