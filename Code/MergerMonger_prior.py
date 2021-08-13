'''
~~~
Iteratively run MergerMonger on the full SDSS sample to make the figure: 'adjusting_prior_all.png'
The goal is to be able to adjust the prior without adjusting the alphas to determine how much the choice of input priors affects the measured merger fraction.
~~~
'''
from MergerMonger import load_LDA_from_simulation
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from util_LDA import cross_term

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

input_singular = []
crossterms = []
ct_1 = []
ct_2 = []
for j in range(len(LDA[2])):
    if '*' in LDA[2][j]:
        crossterms.append(LDA[2][j])
        split = str.split(LDA[2][j],'*')
        ct_1.append(split[0])
        ct_2.append(split[1])
        
    else:
        input_singular.append(LDA[2][j])
        





plt.clf()
plt.hist(LDA[8], bins=50, alpha=0.5)

plt.xlabel("LD1")
plt.axvline(x=0)

# Move the line
# what the original contribution is:
fmerg_prior = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
color = ['#16BAC5','#5FBFF9','#F26430','#171D1C',
    '#5863F8','#EF767A','#721817','#466060','#0E0F19']
for i in range(len(fmerg_prior)):



    prior_line = math.log(fmerg_prior[i]/(1-fmerg_prior[i]))
    print('this is the term added from the intercept', fmerg_prior[i], prior_line)
    nmerg = len(np.where(np.array(LDA[8]) > (prior_line))[0])

    #plt.axvline(x = 0)
    plt.axvline(x = prior_line, color=color[i])
    plt.annotate(r'$f_{\mathrm{in}}$ = '+str(fmerg_prior[i])+', $f_{\mathrm{merg}}$ = '+str(round(nmerg/len(LDA[8]),2)), xy=(-14, 55-3*i), xycoords='data', color=color[i])

plt.savefig('../Figures/adjusting_prior_simulation.png', dpi=1000)

# Find out what fraction of galaxies are above the LD1 = 0 line


    
    

type_gal = 'DR12_predictors_MPI'
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
    
    # Okay add in the additional term to correct for shifting the threshold around in f_in:
    starting_prior = math.log(0.1/(1-0.1))
    
    new_line = math.log(prior_list[i][0]/(1-prior_list[i][0]))
    
    shift = new_line - starting_prior
    
    


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

    df2 = pd.io.parsers.read_csv('../Tables/SDSS_'+str(type_gal)+'_all.txt', sep='\t')
    df2.columns = ['ID','Sep','Flux Ratio',
      'Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)', 'Sersic AR', 'S/N', 'Sersic N Statmorph', 'A_S Statmorph']
    
    df2 = df2[0:1000]
    #Okay so this next part actually needs to be adaptable to reproduce all possible cross-terms
    
    
    '''
    # Need to fix this section
    crossterms = []
    ct_1 = []
    ct_2 = []
    for j in range(len(input_singular)):
        for i in range(len(input_singular)):
            if j == i or i < j:
                continue
                
            print(input_singular[j], input_singular[i])
            #input_singular.append(input_singular[j]+'*'+input_singular[i])
            crossterms.append(input_singular[j]+'*'+input_singular[i])
            ct_1.append(input_singular[j])
            ct_2.append(input_singular[i])
    
    
    
    inputs = input_singular + crossterms
    '''

    # Now you have to construct a bunch of new rows to the df that include all of these cross-terms
    for j in range(len(crossterms)):
        
        df2[crossterms[j]] = df2.apply(cross_term, axis=1, args=(ct_1[j], ct_2[j]))
        
    print('did I add them?', df2, df2.columns)
    print('what you r trying to get values from', LDA[2])
    
    X_gal = df2[LDA[2]].values


    p_merg_list = []
    LD1_list = []

    for j in range(len(X_gal)):
        #print(X_gal[j])
        X_standardized = list((X_gal[j]-LDA[0])/LDA[1])
        # use the output from the simulation to assign LD1 value:
        LD1_gal = float(np.sum(X_standardized*LDA[3])+LDA[4]+shift)
        
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
    '''
    plt.clf()
    plt.title('Fraction of 0s and 1s = '+str(round(frac_0_1,2)))
    plt.scatter(logLD1, p_merg_list, s=1)
    plt.xlabel('log LD1')
    plt.ylabel('p_merg')
    plt.savefig('../Figures/LD1_and_pmerg_'+str(run)+'_'+str(i)+'.png')
    '''
    
    
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
ys = [x for x in np.array(prior_list)[:,0]]
plt.plot(np.array(prior_list)[:,0], ys, color='black')

plt.scatter(np.array(prior_list)[:,0], fmerg)

plt.xlabel(r'Prior, $f_{\mathrm{merg}}$')
plt.ylabel(r'Output, $f_{\mathrm{merg}}$')
plt.axvline(x=0.1)
plt.savefig('../Figures/adjusting_prior_'+str(run)+'.png')
