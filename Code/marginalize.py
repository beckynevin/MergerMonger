# Read through all of the different prior classifications and see how the probability of merging changes
# as a function of input prior
# Also has the option to eliminate all flagged galaxies (photometrically)
# and calculates the overall merger fraction making a correction for completeness

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity
import os

# Figure out what exists:
files = os.listdir('../Tables/change_prior')#/LDA_out_all_SDSS_predictors_''))
type_gal = 'major_merger'
flag = True
plot = False

# Make a list of input fractions:
run_these = []
for file in files:
	#print(file)
	if str(type_gal+'_') in file:
		if flag:
			if 'flag' in str.split(str.split(file,type_gal+'_')[1],'.txt')[0]:
				run_these.append(float(str.split(str.split(str.split(file,type_gal+'_')[1],'.txt')[0],'_flag')[0]))
		else:
			if 'flag' in str.split(str.split(file,type_gal+'_')[1],'.txt')[0]:
				continue
			else:
				run_these.append(float(str.split(str.split(file,type_gal+'_')[1],'.txt')[0]))



'''
#start at 0.05 for minor
prior_mfrac = np.arange(0.01,0.51,0.01)# spacing is the last argument
print(prior_mfrac, len(prior_mfrac))
'''
prior_mfrac = np.sort(run_these)


dict_values = {}

f_merg_avg = []

cmap = sns.color_palette("mako")#s, as_cmap=True)
cmap=plt.get_cmap("magma")

for i in range(len(prior_mfrac)):
	if flag:
		probs = pd.io.parsers.read_csv('../Tables/change_prior/LDA_out_all_SDSS_predictors_'+str(type_gal)+'_'+str(prior_mfrac[i])+'_flag.txt', sep='\t', header=[0])
		# Go through and get rid of all of the flagged ones:
		
		probs_no_flags = probs[(probs['low S/N'] == 0) & (probs['outlier predictor'] == 0)]
		#probs = probs_no_flags
		
	else:
		probs = pd.io.parsers.read_csv('../Tables/change_prior/LDA_out_all_SDSS_predictors_'+str(type_gal)+'_'+str(prior_mfrac[i])+'.txt', sep='\t', header=[0])
	dict_values[prior_mfrac[i]] = probs['p_merg'].values#[0:100]
	#print (probs.columns)
	#if i%50:

	if plot:

		if i % 10 == 0:
			plt.hist(probs['p_merg'].values, bins=100, alpha=0.5, label=str(round(prior_mfrac[i],2)), color=cmap(i/len(prior_mfrac)))
			#probs['p_merg'].plot.kde(bw_method = 0.1, label=str(round(prior_mfrac[i],2)), disp = "filled.contour")
			#kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(probs['p_merg'].values)
			#log_dens = kde.score_samples(X_plot)

			#print(np.shape(X_plot[:, 0]), np.shape(log_dens))
			#plt.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
		#sns.kdeplot(probs['p_merg'],  alpha=0.5, linewidth=0)
	   
		#plt.scatter(prior_mfrac[i], probs['p_merg'].values[0])
	f_merg_avg.append(len(np.where(probs['p_merg'] > 0.5)[0])/len(probs['p_merg']))

if plot:
	plt.xlabel(r'$p_{\mathrm{merg}}$')
	plt.legend()
	plt.xlim([0,1])
	plt.show()

#print(dict_values)

# Now from this plot the first entry of all:
#plt.xlabel('prior on fmerg')
#plt.ylabel('p_merg')
#plt.show()

#print(sum(dict_values.values(), []))
from itertools import chain

all = list(chain(*dict_values.values()))


fmerg = len(np.where(np.array(all) > 0.5)[0])/len(all)

print('fmerg = ',fmerg)
print('comparing this to the average', np.mean(f_merg_avg))
print('comparing this to the average', np.median(f_merg_avg))
print(len(f_merg_avg))

print('f_merg_avg', f_merg_avg)

plt.hist(all, bins=100)
plt.axvline(x=fmerg)
plt.show()

# Make a completeness correction:
if type_gal=='major_merger':
	completeness_correction = [1.2801932367149758, 1.218390804597701, 1.2211981566820274, 1.1725663716814159, 1.1622807017543861, 1.1572052401746724, 1.1324786324786325, 1.1228813559322035, 1.111111111111111, 1.0838445807770962, 1.0860655737704918, 1.1064718162839247, 1.075050709939148, 1.083844580777096, 1.0728744939271255, 1.029126213592233, 1.0474308300395256, 1.017274472168906, 1.0251450676982592, 1.0453648915187377, 1.0192307692307692, 1.0192307692307694, 1.0211946050096339, 1.017274472168906, 1.0114503816793892, 0.9869646182495344, 1.0271317829457365, 1.0311284046692606, 0.9943714821763603, 0.9851301115241635, 0.974264705882353, 0.977859778597786, 0.9584086799276672, 0.974264705882353, 0.9833024118738405, 0.9584086799276672, 0.9618874773139747, 0.9618874773139746, 0.9618874773139746, 0.9413854351687388, 0.9566787003610109, 0.9549549549549551, 0.9566787003610109, 0.9906542056074766, 0.9925093632958801, 0.9869646182495344]
if type_gal=='minor_merger':
	completeness_correction = [3.0588235294117645, 2.9714285714285715, 2.5365853658536586, 2.6666666666666665, 2.4186046511627906, 2.390804597701149, 2.1894736842105265, 2.08, 1.980952380952381, 2.2127659574468086, 1.7333333333333332, 1.7627118644067796, 1.6377952755905512, 1.575757575757576, 1.71900826446281, 1.857142857142857, 1.4545454545454546, 1.464788732394366, 1.625, 1.4545454545454546, 1.3866666666666667, 1.4344827586206894, 1.424657534246575, 1.5522388059701493, 1.5294117647058822, 1.4751773049645391, 1.518248175182482, 1.2023121387283238, 1.2023121387283238, 1.1751412429378534, 1.1555555555555557, 1.149171270718232, 1.1751412429378534, 1.142857142857143, 1.1122994652406417, 1.3506493506493504, 1.2163742690058479, 1.1818181818181819, 1.089005235602094, 1.077720207253886, 1.0558375634517767, 1.0505050505050506, 1.106382978723404, 1.2235294117647058, 1.0947368421052632, 1.094736842105263]

plt.clf()
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot(311)
ax.scatter(prior_mfrac, f_merg_avg, color='#EC4E20')
ax.axhline(y = np.mean(f_merg_avg), color='#EC4E20')
ax.set_xlabel('Input prior')
ax.set_ylabel('Measured fmerg')

if flag:
	ax.set_title('Flagged are outsies')

ax1 = fig.add_subplot(312)
ax1.scatter(prior_mfrac, completeness_correction, color='#353531')
ax1.set_xlabel('Input prior')
ax1.set_ylabel('completeness correction')

ax2 = fig.add_subplot(313)
ax2.scatter(prior_mfrac, np.array(completeness_correction)*np.array(f_merg_avg), color='#FF9505')
ax2.axhline(y = np.mean(np.array(completeness_correction)*np.array(f_merg_avg)), color='#FF9505')
ax2.set_xlabel('Input prior')
ax2.set_ylabel('completeness corrected fmerg')

plt.tight_layout()
plt.show()

print('fmerg (completeness corrected) = ',np.mean(np.array(completeness_correction)*np.array(f_merg_avg)))

