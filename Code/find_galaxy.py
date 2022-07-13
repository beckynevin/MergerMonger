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
import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from util_SDSS import SDSS_objid_to_values, download_galaxy
from util_smelter import get_predictors
from util_LDA import calculate_cdf, find_nearest


# First, define the list of object IDs you want to run through the classification
# These can either be from a list or imported from a table

# these are galaxyzoos that are not classified as mergers
#ID_list = [1237661871347138764,1237662264322097387,1237662264325767706,1237662264848351296,1237667782823903325]


#ID_list = [1237661125071208589, 1237654654171938965, 1237651212287475940, 1237659325489742089, 1237651273490628809, 1237661852007596057, 1237657878077702182, 1237667912748957839, 1237662665888956491, 1237655504567926924, 0, 1237664673793245248, 1237653009194090657, 1237667212116492386, 1237660024524046409, 1237654949448646674, 1237656496169943280, 1237663784217804830, 1237673706113925406, 1237656567042539991, 1237653587947815102, 1237651191354687536, 1237661387069194275, 1237651226784760247, 1237658204508258428, 1237661957225119836, 1237653589018018166, 1237651251482525781, 1237658802034573341, 1237663457241268416, 1237663529718841406, 1237651272956641566, 1237667910601932987, 1237659326029365299, 1237661852538437650, 1237665549422952539, 1237659327099896118, 1237651212287672564, 1237666299480309878, 1237657856607649901, 1237654952670789707, 1237654949448450067, 1237660241386143913, 1237652899700998392, 1237664837002395706, 1237654626785821087, 1237654391639638190]

# Look up RAs and Dec from some sort of table:
# Alternately, just use your own lists of RAs and decs like so:
#RA_list = [118.074345115, 119.617127519, 258.548475763, 241.150984777, 114.096383278, 189.213252539, 120.087417603, 205.690424787, 246.255977023, 251.335933471, 331.12290045, 205.753337262, 316.841308073, 127.170800449, 46.2941968423, 234.541843983, 319.193098655, 46.6649126989, 111.733682055, 322.213310988, 127.178093813, 123.820325773, 217.629970676, 262.399282723, 173.537567139, 215.017906947, 119.486337418, 123.330544326, 171.400653635, 321.00791167, 118.184152672, 119.182151794, 206.627529147, 247.159333462, 169.513447059, 206.007949798, 241.271446954, 258.82741019, 46.7170694138, 171.657261919, 255.029869751, 233.968342684, 47.1429889739, 61.4532623945, 131.725410562, 114.695391123, 118.855393765]
#dec_list = [19.5950803976, 37.7866245591, 57.9760977732, 43.8797876113, 39.4382798053, 45.6511702678, 26.613527298, 24.5900537916, 24.2631556488, 42.757790046, 12.4426263785, 36.1656555546, 11.0664208661, 17.5814003238, -1.07547036487, 57.6036530229, 11.0437407875, 0.0619768826479, 41.0266910782, -1.07011823351, 45.7425550277, 46.0752533433, 52.7071590288, 54.4944236725, 49.2545623779, 47.1213302326, 39.9933651629, 46.1471565611, 54.3825744827, -0.366323610281, 45.949276388, 44.8567085266, 22.7060191923, 39.551266027, 45.1130288476, 25.9411986626, 45.4429921738, 57.6587701679, -0.896544995426, 51.5730412626, 37.8395021377, 57.9026365222, 0.55093690996, -6.32383706666, 25.3700887716, 29.8912832677, 39.1860944299]


ID_list = [1237657775542698092,
 1237655129839108147, 1237659330848817218, 1237656895066931656,
 1237674288607461402, 1237658802572558360, 1237655130376241254,
 1237655473967136890, 1237661386538614980, 1237662336796065949,
 1237654391639376175, 1237660024523849823, 1237664835392241744,
 1237651753456107524]

ID_list = [1237654391642128470]

# GZ mergers as non-mergers:
ID_list = [1237648673993654418, 1237655550744527565, 1237674648853414113, 1237648702967513214, 1237671139870310556, 1237648703524176162, 1237648704041844890, 1237648704043089981, 1237648704067142300, 1237648704067142301, 1237648704067273097, 1237648704589987959, 1237648705133543652, 1237648705139572888, 1237674651535147100, 1237649918430675064, 1237656494580170993, 1237666091125309664, 1237653650771542160, 1237653651311952032]
# GZ mergers as mergers:
ID_list = [1237668649851945457, 1237668649851945458, 1237668569858375932, 1237674649386811408, 1237648703503138948, 1237648703503138949, 1237648703510610056, 1237648703523389861, 1237674649923223726, 1237674649929318470, 1237648704053444900, 1237648720718463286, 1237648704067338299, 1237648704067338300, 1237648704067535666, 1237648704068518352, 1237648704068649960, 1237648704068649961, 1237648704591167730, 1237648704594772379]
# GZ spirals as mergers:
ID_list = [1237668568247370051, 1237668568247370012, 1237668650389340599, 1237648673459667002, 1237655559866614390, 1237655549670654093, 1237668568784109875, 1237668568784110087, 1237668569321308424, 1237648704065765844, 1237648673994834641, 1237648673994965546, 1237648673994965546, 1237648673995096731, 1237648704066552032, 1237648673996473001, 1237648673996473042, 1237655554497315264, 1237648673997127724, 1237648674529804864]
# GZ spirals as non-mergers:
ID_list = [1237655693557170541, 1237648673456455904, 1237668649315008771, 1237668649315074455, 1237668649315205346, 1237668568247370040, 1237668649852076495, 1237668649852207363, 1237668568784437849, 1237668568784437611, 1237668568784503051, 1237668568784503170, 1237668650389340761, 1237668650926342749, 1237668650926342753, 1237668650926408199, 1237668569858638234, 1237668650926539340, 1237668569858703829, 1237668569858703926]
# GZ ellipticals as mergers:
#ID_list = [1237668648777941378, 1237668568784437728, 1237668568784437730, 1237668568784503189, 1237668650389340774, 1237668650926408139, 1237655549668229596, 1237648673459274392, 1237648673459405604, 1237655558793330808, 1237648721255334064, 1237648673993195770, 1237648673993392521, 1237668650926080045, 1237668569858507188, 1237668651463278988, 1237648673994637868, 1237648673994637589, 1237648673994637877, 1237648673994834323]
# GZ ellipticals as non-mergers:
#ID_list = [1237655693558153341, 1237655693558153619, 1237655693558153741, 1237668648777941094, 1237655549666525534, 1237655549666525341, 1237655549666525599, 1237655549666525282, 1237655549666525475, 1237655549666525565, 1237668649315008817, 1237648673456652659, 1237668649315140216, 1237668568247304508, 1237668568247370078, 1237668649852207403, 1237668568784306748, 1237668649852272973, 1237668568784372205, 1237668649852338537]

ID_list = [1237663782592971113,1237664854190194720,1237662336256639070,1237662336261161190,1237666299481817126,1237653665787478127,1237659326566039633]


ID_list = [1237664836996104704,1237664835384771072,1237667108496737024,
           1237657609109569792,1237660764300574976,1237655744016482560,
           1237661382772195584,1237661381698781440,1237658206122278912,
           1237667443511591168,1237659154219532544,1237655503499756032,
           1237667211583225856,1237660635461779456,1237661850933067776,
           1237657775543746816,1237657775006482688,1237664854713565440,
           1237661387607572480,1237678617939280384,1237663787417338112,
           1237663530794811648,1237657877003370752,1237667324334571776,
           1237667442974392320,1237664667890614528,1237657858220032000,
           1237673808654107136,1237664094509531392]
ID_list = [1237654390032629943,1237661464918688114,1237664667887534206]

ID_list = [1237659326029365299,1237663529718841406,1237654654171938965]


# This is what you're going to use as a suffix for the names of the saved images
appellido = 'breakbrd_objids'#'drpall-v3_1_1_objid'#'breakbrd_objids'#
#appellido = 'AGNs'
appellido = 'AGN2_fails'#'control'#'Control_Sample1_MaNGA_MPL7'#AGN1.1_MaNGA_MPL7
appellido = 'MaNGA_MPL11'
appellido = 'control2'

appellido = 'GZ_spirals_classified_as_nonmergers'
appellido = 'julie_close'
appellido = 'radio_AGN_hector'
print('appellido is', appellido)
load = False
find_anyway = False
recalc_preds = False

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

        
print('length of stuff to run', len(ID_list))

print('ID_list', ID_list)






# Define the sizes of the images and where your LDA and SDSS tables live:
merger_type = 'major_merger'
plot = True
size = 200
prefix = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'

# Now import the tables of predictor values and LDA values

df_predictors = pd.io.parsers.read_csv(prefix+'SDSS_predictors_all_flags_plus_segmap.txt', sep='\t')
#df_predictors.columns = ['ID','Sep','Flux Ratio',  'Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)', 'Sersic AR', 'S/N', 'Sersic N Statmorph', 'A_S Statmorph']

if len(df_predictors.columns) ==15: #then you have to delete the first column which is an empty index
    df_predictors = df_predictors.iloc[: , 1:]
  
# Change the type of the ID in predictors:
df_predictors = df_predictors.astype({'ID': 'int64'})#.dtypes    

print('length of predictors', len(df_predictors))

                                                                              


# Import the probability values for all SDSS galaxies

df_LDA = pd.io.parsers.read_csv(prefix+'LDA_out_all_SDSS_predictors_'+str(merger_type)+'_flags.txt', sep='\t')#_leading_preds
#print(df_LDA.columns, df_predictors.columns)
df_LDA = df_LDA.astype({'ID': 'int64'})
'''
print('length of LDA', len(df_LDA))
print(df_LDA.columns)
print(df_LDA['Leading_term'])

ID_list_df = pd.DataFrame(ID_list, columns=['ID'])
ID_list_df.astype({'ID': 'int64'})
merged = df_LDA.merge(ID_list_df, on='ID')
print(merged)

print(ID_list_df['ID'])
print(df_LDA['ID'])




# Okay maybe try to cross-reference with dr7:
# ^its not this
matching = pd.io.parsers.read_csv(prefix + 'classifications_by_objid/classifications_out_all_dr7_dr8_crossmatch_MaNGA_MPL11_objid.txt', sep = '\t')
print(matching)
print(matching.columns)

merged_dr7 = ID_list_df.merge(matching, left_on = 'ID', right_on = 'objID')

print()

print(merged_dr7)
STOP

'''




'''
# Merge the two tables together on ID

ID_list_df = pd.DataFrame(ID_list.byteswap().newbyteorder(), columns=['ID'])

print(ID_list_df)
print(len(df_LDA))
print(len(ID_list_df))
print('length of merged', len(df_LDA.merge(ID_list_df, on='ID')))

df_all = df_LDA.merge(ID_list_df.drop_duplicates(), on=['ID'], 
                   how='right', indicator=True)
print(df_all['_merge'].unique())
print(df_all[df_all['_merge']=='right_only']['ID'].values)
print('length of not in df_LDA', len(df_all[df_all['_merge']=='right_only']['ID'].values))

STOP
'''



ra_dec_lookup = pd.read_csv(prefix + 'five_sigma_detection_saturated_mode1_beckynevin.csv')

print('overall len ra/dec', len(ra_dec_lookup))
print('length where ra==0', len(ra_dec_lookup[ra_dec_lookup['ra']==0.0]))

print(ra_dec_lookup)


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


print('length of RA_list', len(RA_list))







        
'''
print(RA_list)
STOP

RA_list = []
dec_list = []
for j in range(len(ID_list)):
    counter = 0
    for i in range(len(ra_dec_lookup)):
        if ID_list[j]==ra_dec_lookup['objID'].values[i]:
            RA_list.append(ra_dec_lookup['ra'].values[i])
            dec_list.append(ra_dec_lookup['dec'].values[i])
            counter+=1
            print('found this RA', j)
    if counter==0:
        RA_list.append(0)
        dec_list.append(0)

print(RA_list)
print(dec_list)
'''


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

# Test what the cdf value is for various p_merg threshold value choices:
threshold = 0.75
cdf = hist_dist.cdf(threshold)
print('cdf value for ', threshold, cdf)







indices_preds = df_predictors.index


if appellido == 'ABC': # This is for making the ABC plot in the paper
    counter_ABC = 0
    ABC = ['A','B','C']
    gal_colors = ['#CE7DA5','#6DD3CE','#E59500']





for i in range(len(ID_list)):
    id = ID_list[i]

    print('trying to find a match for this', id, type(id), i)
    ra = RA_list[i]
    dec = dec_list[i]
    # Find each item in each data frame:

    
    where_LDA = np.where(np.array(df_LDA['ID'].values)==id)[0]

    print('where LDA', where_LDA, where_LDA.size)
    print(df_LDA.columns)
    
    print(df_LDA.values[where_LDA][0])
    
    


    #print('trying to find where preds', df_predictors.loc[df_predictors['ID']==id])
    
    condition = df_predictors["ID"] == id
    try:
        where_predictors = indices_preds[condition].values.tolist()[0]
    except:
        if find_anyway:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('no match but finding anyway')
        else:
            continue
    print(df_predictors.columns)
    print('where preds', df_predictors.values[where_predictors])
   
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
        if find_anyway:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('no match but finding anyway')
            cdf = 0
        else:
            continue
    cdf_list.append(cdf)

    if where_LDA.size != 0 or find_anyway == True:

        # so IF it exists, then go ahead and download it:
        



        if plot:
            try:
                img = download_galaxy(id, ra, dec, prefix+'../frames/', size)
            except FileNotFoundError:
                STOP
                continue

            shape = np.shape(img)[0]

            print('getting ready for flags')
            if find_anyway == False: # Then you have flag info already
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
            else:
                flag = True
                flag_name = ['not run']

            img = np.array(img)
            # Check if the file already exists:
            preds = get_predictors(id, img, prefix+'../', size)
            print('preds', preds)
            try:
                segmap = preds[1]
            except TypeError:
                print('No segmap')
                
            '''
            if os.path.exists(prefix+'../Figures/ind_galaxies_classify/'+appellido+'/'+str(merger_type)+'/classification_'+str(id)+'.png'):
                print('already have classification image')
            else:
            '''
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(abs(img), norm=matplotlib.colors.LogNorm(vmax=10**5, vmin=10**(0.5)), 
                cmap='afmhot', interpolation='None')

            if preds is None:
                print('no segmap')
            else:
                ax.contour(np.fliplr(np.rot90(preds[1])), levels=[0,1], colors='yellow')
            #im2 = ax2.imshow(np.fliplr(np.rot90(preds[1])))
            
            if appellido == 'ABC':

                ax.annotate(ABC[counter_ABC],xy=(0.7,0.05),xycoords='axes fraction',
                    color=gal_colors[counter_ABC], size=100)
                counter_ABC+=1
            if preds is None or recalc_preds == False:
                ax.annotate('Gini = '+str(round(df_predictors.values[where_predictors][3],2))+
                    r' M$_{20}$ = '+str(round(df_predictors.values[where_predictors][4],2))+
                    '\nC = '+str(round(df_predictors.values[where_predictors][5],2))+
                    ' A = '+str(round(df_predictors.values[where_predictors][6],2))+
                    ' S = '+str(round(df_predictors.values[where_predictors][7],2))+
                    '\nn = '+str(round(df_predictors.values[where_predictors][8],2))+
                    r' A$_S$ = '+str(round(df_predictors.values[where_predictors][9],2))+
                    '\n<S/N> = '+str(round(df_predictors.values[where_predictors][10],2)), 
                    xy=(0.03, 0.05),  xycoords='axes fraction',
                xytext=(0.03, 0.05), textcoords='axes fraction',
                bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')
            
                
                ax.annotate('LD1 = '+str(round(df_LDA.values[where_LDA][0][3],2))+
                    '\n$p_{\mathrm{merg}}$ = '+str(round(df_LDA.values[where_LDA][0][4],4))+
                    '\nCDF = '+str(round(cdf,4))+'\n'+str(df_LDA.values[where_LDA][0][6]), xy=(0.03, 0.8),  xycoords='axes fraction',
                xytext=(0.03, 0.8), textcoords='axes fraction',
                bbox=dict(boxstyle="round", fc="0.9"), color='black')
            else:
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

                ax.annotate('LD1 = '+str(round(df_LDA.values[where_LDA][0][3],2))+
                    '\n$p_{\mathrm{merg}}$ = '+str(round(df_LDA.values[where_LDA][0][4],4))+
                    '\nCDF = '+str(round(cdf,4))+'\n'+str(df_LDA.values[where_LDA][0][6]), xy=(0.03, 0.8),  xycoords='axes fraction',
                xytext=(0.03, 0.8), textcoords='axes fraction',
                bbox=dict(boxstyle="round", fc="0.9"), color='black')
            
                
                


            
            if flag:
                ax.annotate('Photometric Flag \n '+str(flag_name), xy=(0.8, 0.85),  xycoords='axes fraction',
            xytext=(0.8, 0.85), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9"), color='black')
            
            ax.set_title('ObjID = '+str(id))
            ax.set_xticks([0, (shape - 1)/2, shape-1])
            ax.set_xticklabels([-size/2, 0, size/2])
            ax.set_yticks([0, (shape- 1)/2,shape-1])
            ax.set_yticklabels([-size/2, 0,size/2])
            ax.set_xlabel('Arcsec')
            plt.savefig(prefix+'../Figures/ind_galaxies_classify/'+str(appellido)+'/'+str(merger_type)+'/classification_'+str(id)+'.png', dpi=1000)

            '''
            # Make a second plot that includes the segmentation map
            center_val = segmap[int(np.shape(segmap)[0]/2),int(np.shape(segmap)[1]/2)]
            # Need to make a list of all points on the edge of the map
            edges=np.concatenate([segmap[0,:-1], segmap[:-1,-1], segmap[-1,::-1], segmap[-2:0:-1,0]])
            '''

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

            '''
            continue
            # Check if the figure already exists:
            if os.path.exists(prefix+'../Figures/ind_galaxies_classify/'+appellido+'/'+str(merger_type)+'/statmorph_check_'+str(id)+'.png') or find_anyway==True:
                print('already made statmorph fig')
            
            else:
            '''
            '''
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            im1 = ax1.imshow(abs(np.fliplr(np.rot90(preds[0]))), norm=matplotlib.colors.LogNorm())#, origin = 'lower'
            ax1.set_title(str(merger_type)+'\nImage, flag = \n'+str(flag_name))
            ax1.annotate('LD1 = '+str(round(df_LDA.values[where_LDA][0][2],2))+
                '\n$p_{\mathrm{merg}}$ = '+str(round(df_LDA.values[where_LDA][0][3],4))+
                '\nCDF = '+str(round(cdf,4)), 
                xy=(0.03, 0.75),  xycoords='axes fraction',
            xytext=(0.03, 0.75), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')

            ax1.annotate(str(df_LDA.values[where_LDA][0][6])+' '+str(df_LDA.values[where_LDA][0][5])+
                '\n'+str(df_LDA.values[where_LDA][0][8])+' '+str(df_LDA.values[where_LDA][0][7])+
                '\n'+str(df_LDA.values[where_LDA][0][10])+' '+str(df_LDA.values[where_LDA][0][9]),
                xy=(0.03, 0.08),  xycoords='axes fraction',
            xytext=(0.03, 0.08), textcoords='axes fraction',
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
            ax2.set_xlabel('ObjID = '+str(id))
            plt.savefig(prefix+'../Figures/ind_galaxies_classify/'+appellido+'/'+str(merger_type)+'/statmorph_check_'+str(id)+'.png', dpi=1000)
            '''

            '''
            if flag:
                flag_list.append(1)
            else:
                flag_list.append(0)
            '''
        else:
            index_save_LDA.append(where_LDA[0])
            index_save_predictors.append(where_predictors)
            print('saving this because you didnt plot', where_LDA[0])

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

df_merged.to_csv(prefix+'classifications_by_objid/classification_out_and_predictors_'+str(merger_type)+'_'+appellido+'.txt', sep='\t')
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

