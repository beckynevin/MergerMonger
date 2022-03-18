# Imports multiple different tables and compares the p_merg values from both
# in histogram form (with an option for calculating the percentage above a threshold)
# and then also puts things into boxplot form

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

prefix = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'

merger_type = 'major_merger'
appellido1 = 'AGN1'# AGNs
appellido2 = 'control'#'breakbrd_objids'
thresh = True
clean = True



print('existing classifications', os.listdir(prefix+'classifications_by_objid/'))

pop1 = pd.read_csv(prefix+'classifications_by_objid/classification_out_and_predictors_'+str(merger_type)+'_'+appellido1+'.txt', sep='\t')
pop2 = pd.read_csv(prefix+'classifications_by_objid/classification_out_and_predictors_'+str(merger_type)+'_'+appellido2+'.txt', sep='\t')

# Option to clean using flags:
if clean:
	pop1 = pop1[(pop1['low S/N_y'] == 0) & (pop1['outlier predictor_y'] == 0) & (pop1['segmap'] ==0)]
	pop2 = pop2[(pop2['low S/N_y'] == 0) & (pop2['outlier predictor_y'] == 0) & (pop2['segmap'] ==0)]



colors = ['#D81E5B','#3A3335']
colors = ['#084C61','#DB504A']

if thresh:
	thresh_p = 0.75
	frac_gt_thresh_1 = len(pop1[pop1['p_merg'] > thresh_p])/len(pop1)
	frac_gt_thresh_2 = len(pop2[pop2['p_merg'] > thresh_p])/len(pop2)


plt.clf()

values, bins = np.histogram(pop1['p_merg'].values, bins=50)
if thresh:
	plt.hist(pop1['p_merg'].values, label=appellido1+' (# = '+str(len(pop1))+'), fraction over threshold = '+str(round(frac_gt_thresh_1,2)), 
		bins=bins, alpha=0.7, density=True, color = colors[0])
	plt.hist(pop2['p_merg'].values, label=appellido2+' (# = '+str(len(pop2))+'), fraction over threshold = '+str(round(frac_gt_thresh_2,2)), 
		bins=bins, alpha=0.7, density=True, color = colors[1])
		
else:
	plt.hist(pop1['p_merg'].values, label=appellido1+' (# = '+str(len(pop1))+')', 
		bins=bins, alpha=0.7, density=True, color = colors[0])
	plt.hist(pop2['p_merg'].values, label=appellido2+' (# = '+str(len(pop2))+')', 
		bins=bins, alpha=0.7, density=True, color = colors[1])
plt.axvline(x = np.mean(pop1['p_merg'].values), color = colors[0])
plt.axvline(x = np.mean(pop2['p_merg'].values), color = colors[1])

if thresh:
	plt.axvline(x = thresh_p, color='black')
plt.legend()
plt.xlabel('p_merg')
plt.show()

# Try a boxplot
data = [pop1['p_merg'].values,pop2['p_merg'].values]
plt.clf()
fig = plt.figure(figsize =(11, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data, patch_artist = True, vert = 0)#

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='black',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='black',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='black',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax.set_yticklabels([appellido1+' (# = '+str(len(pop1))+')', appellido2+' (# = '+str(len(pop2))+')'])
ax.set_xlabel('p_merg')
if thresh:
	plt.axvline(x = thresh_p, color='black')
plt.show()