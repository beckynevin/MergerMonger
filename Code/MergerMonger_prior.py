'''
~~~
Iteratively run MergerMonger on the full SDSS sample to make the figure: 'adjusting_prior_all.png'
The goal is to be able to adjust the prior without adjusting the alphas.
~~~
'''
from MergerMonger import load_LDA_from_simulation
import numpy as np
import math
import matplotlib.pyplot as plt

run = 'major_merger'
LDA = load_LDA_from_simulation(run)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Output from LDA~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(LDA)
# The output of this is in the format:
# 0 = standardized means on all of the coefficients
# 1 = standardized stds
# 2 = inputs
# 3 = coefficients
# 4 = intercept
# 5 = Accuracy
# 6 = Precision
# 7 = Recall
# 8 = LDA values of all simulated galaxies
# 9 = myr
# 10 = myr_non
# 11 = covariance matrix
# 12 = means of all classes

print('coefficients', LDA[3])
print('intercept', LDA[4])

print('len intercept', len(LDA[4]))






plt.clf()
plt.hist(LDA[8], bins=50, alpha=0.5)

plt.xlabel("LD1")
plt.axvline(x=0)

# Move the line
# what the original contribution is:
prior_line = math.log(0.9/0.1)
prior_50 = math.log(1)
print('original prior contribution', prior_line)
print('50/50 contribution', prior_50)

nmerg = len(np.where(np.array(LDA[8]) > (prior_50 - prior_line))[0])

plt.axvline(x = 0)
plt.axvline(x = 0 + (prior_50 - prior_line))
plt.annotate(r'$f_{\mathrm{merg}}$ = '+str(nmerg/len(LDA[8])), xy=(0.02, 0.95), xycoords='axes fraction')

plt.show()

# Find out what fraction of galaxies are above the LD1 = 0 line

STOP
    
    

type_gal = 'DR12_predictors_all'
verbose='no'


# Okay but run iteratively for a bunch of different priors:

prior_list = [[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.5,0.5],
    [0.6,0.4],[0.7,0.3],[0.8,0.2],[0.9,0.1]]
list_fmerg = np.linspace(0,1,21)

prior_list = [[x, 1-x] for x in list_fmerg]
prior_list = prior_list[1:-1]
print(prior_list)

    

fmerg = []
skipped_LDA = []
for i in range(np.shape(prior_list)[0]):
    priors = prior_list[i]
    #was: LDA_prep_predictors_all_combined_major.txt
    df = pd.io.parsers.read_csv(filepath_or_buffer='../Tables/LDA_merged_'+str(run)+'.txt',header=[0],sep='\t')

    #Rename all of the kinematic columns (is this necessary?)
    df.rename(columns={'kurtvel':'$h_{4,V}$','kurtsig':'$h_{4,\sigma}$','lambda_r':'\lambdare',
             'epsilon':'$\epsilon$','Delta PA':'$\Delta$PA','A_2':'$A_2$',
              'varsig':'$\sigma_{\sigma}$',
             'meanvel':'$\mu_V$','abskewvel':'$|h_{3,V}|$',
             'abskewsig':'$|h_{3,\sigma}|$',
             'meansig':'$\mu_{\sigma}$',
             'varvel':'$\sigma_{V}$'},
    inplace=True)
    #was LDA_prep_predictors_all_combined.txt
    #LDA_img_ratio_statmorph_fg3_m12_A_S.txt'
    df.columns = [l for i,l in sorted(feature_dict.items())]

    df.dropna(how="all", inplace=True) # to drop the empty line at file-end
    df.dropna(inplace=True) # to drop the empty line at file-end



    myr=[]
    myr_non=[]
    for j in range(len(df)):
        if df[['class label']].values[j][0]==0.0:
            myr_non.append(df[['Myr']].values[j][0])
        else:
            myr.append(df[['Myr']].values[j][0])

    myr_non=sorted(list(set(myr_non)))
    myr=sorted(list(set(myr)))
            
            

    terms_RFR, reject_terms_RFR = run_RFR(df, features_list, run, verbose)
    try:
        output_LDA = run_LDA(run, df, priors,terms_RFR, myr, myr_non, 21,  verbose)
    except IndexError:
        skipped_LDA.append(i)
        fmerg.append(0)
        continue



    #~~~~~~~
    # Now bring in the SDSS galaxies!
    #~~~~~~~

    feature_dict2 = {i:label for i,label in zip(
                range(14),
                  ('Counter',
                  'ID',
                  'Merger?',
                  '# Bulges',
                   'Sep',
                   'Flux Ratio',
                  'Gini',
                  'M20',
                  'Concentration (C)',
                  'Asymmetry (A)',
                  'Clumpiness (S)',
                  'Sersic N',
                'Shape Asymmetry (A_S)', 'Sersic AR'))}

    df2 = pd.read_csv('../Tables/all_sdss_corrected_A_n_A_offcenter.txt', sep='\t')
    
    

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
        

    X_gal = df2[output_LDA[2]].values


    p_merg_list = []
    LD1_list = []

    for j in range(len(X_gal)):
        #print(X_gal[j])
        X_standardized = list((X_gal[j]-output_LDA[0])/output_LDA[1])
        # use the output from the simulation to assign LD1 value:
        LD1_gal = float(np.sum(X_standardized*output_LDA[3])+output_LDA[4])
        
        LD1_list.append(LD1_gal)
        
        p_merg = 1/(1 + np.exp(-LD1_gal))
        
        
        p_merg_list.append(p_merg)
        #STOP
    # find out what fraction of things are 0s or 1s in the classification
    
    #print('len of 1s', len(np.where(np.array(p_merg_list) >0.999)[0]))
    #print('len of 0s', len(np.where(np.array(p_merg_list) <0.001)[0]))
    #print('fraction of 0s and 1s', (len(np.where(np.array(p_merg_list) >0.999)[0]) + len(np.where(np.array(p_merg_list) <0.001)[0]))/len(p_merg_list))
    #print('overall length', len(p_merg_list))
    frac_0_1 = (len(np.where(np.array(p_merg_list) >0.999)[0]) + len(np.where(np.array(p_merg_list) <0.001)[0]))/len(p_merg_list)
    
    logLD1 = [math.log10(abs(x)) for x in LD1_list]
    plt.clf()
    plt.title('Fraction of 0s and 1s = '+str(round(frac_0_1,2)))
    plt.scatter(logLD1, p_merg_list, s=1)
    plt.xlabel('log LD1')
    plt.ylabel('p_merg')
    plt.savefig('../LDA_figures/LD1_and_pmerg_'+str(run)+'_'+str(i)+'.png')
    
    
    
    p_merg_merg = []
    p_merg_nonmerg = []

    for j in range(len(X_gal)):

        if p_merg_list[j] > 0.5:#merger
            p_merg_merg.append(p_merg_list[j])

        else:
            p_merg_nonmerg.append(p_merg_list[j])
        
    fraction = len(p_merg_merg)/(len(p_merg_merg)+len(p_merg_nonmerg))

    fmerg.append(fraction)
print('input prior', np.array(prior_list)[:,0])
print('fmerg out', fmerg)
print('skip these', skipped_LDA)

plt.clf()
plt.scatter(np.array(prior_list)[:,0], fmerg)
plt.xlabel(r'Prior, $f_{\mathrm{merg}}$')
plt.ylabel(r'Output, $f_{\mathrm{merg}}$')
plt.savefig('../LDA_figures/adjusting_prior_'+str(run)+'.png')
