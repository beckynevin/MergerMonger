#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import the LDA_flag tables and match with the 
# Mendel catalog
# or my own catalog of masses
# bin the galaxies as a function of mass, redshift, B/T, 
# or even multiple of these quantities,
# and compute statistical conclusions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import numpy as np
import math
import astropy.io.fits as fits
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from util_SDSS import SDSS_objid_to_values, download_galaxy
from util_smelter import get_predictors
import glob
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import iqr


# path
dir = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'

type_gal = 'predictors'
run = 'major_merger'
suffix = 'color_complete'
type_marginalized = '_flags_cut_segmap'
num = None
#choose_mass = 'mendel'


df_LDA = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run+'_flags.txt',header=[0],sep='\t')

# Because the df_LDA doesn't have the final flag, use the predictor table to instead clean via merging

# Run OLS with predictor values and z and stellar mass and p_merg:
df_predictors = pd.io.parsers.read_csv(filepath_or_buffer=dir+'SDSS_predictors_all_flags_plus_segmap.txt',header=[0],sep='\t')

if len(df_LDA) != len(df_predictors):
	print('these have different lengths cannot use one to flag')
	STOP

# First clean this so that there's no segmap flags
df_predictors_clean = df_predictors[(df_predictors['low S/N'] ==0) & (df_predictors['outlier predictor']==0)]#& (df_predictors['segmap']==0)]

clean_LDA = df_LDA[df_LDA['ID'].isin(df_predictors_clean['ID'].values)]

if suffix == 'color_complete':
	masstable = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_color_complete.txt',header=[0],sep='\t')
if suffix == 'color':
	masstable = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_color.txt',header=[0],sep='\t')
print(masstable.columns)

masstable = masstable[masstable['log_stellar_mass_from_color'] < 13]


# Now merge this with LDA
final_merged = masstable.merge(clean_LDA, left_on='objID', right_on='ID')#[0:1000]# Now merging on dr8
print('len final merged', final_merged)

print(final_merged.columns)

# mask nans
if num:
	final_merged = final_merged.dropna()[0:num]
else:
	final_merged = final_merged.dropna()
print('length after dropping Nans', len(final_merged))






'''
plt.clf()
plt.scatter(final_merged['B/T'].values, final_merged['p_merg'].values, s=0.1, color='black')
plt.xlabel('B/T ratio')
plt.ylabel('p_merg')
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('../Figures/B_T.png')
'''


final_merged = final_merged[['objID','p_merg_x','LD1','z','log_stellar_mass_from_color']]
df = final_merged


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run OLS with predictor values and z and stellar mass and p_merg:
# Now merge these two and run an OLS:

df_pred_merged = df_predictors_clean.merge(df, left_on='ID', right_on='objID')# Now merging on dr8

print(df_pred_merged.columns)
X = df_pred_merged[[ 'z','Concentration (C)','Gini','Asymmetry (A)']]
# standardize

scaler = StandardScaler()
scaler.fit(X)
X_standardized = scaler.transform(X)
y = df_pred_merged['log_stellar_mass_from_color']#LD1


regr = linear_model.LinearRegression()
print(regr.fit(X_standardized, y))
print(regr.coef_)


N = len(X)
p = len(X.columns) + 1  # plus one because LinearRegression adds an intercept term

X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = X_standardized#.values

ols = sm.OLS(y.values, X_with_intercept)
ols_result = ols.fit()
print(ols_result.summary())





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Do multi-linear regression without doing the following fancy marginalization as part of it
X = df[[ 'z']]
# standardize

scaler = StandardScaler()
scaler.fit(X)
X_standardized = scaler.transform(X)
y = df['LD1']


regr = linear_model.LinearRegression()
print(regr.fit(X_standardized, y))
print(regr.coef_)


N = len(X)
p = len(X.columns) + 1  # plus one because LinearRegression adds an intercept term

X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = X_standardized#.values

ols = sm.OLS(y.values, X_with_intercept)
ols_result = ols.fit()
print(ols_result.summary())


# Do a 2D binning ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
spacing_mass = 0.5# was 0.5
ranges_mass = np.arange(9, 13, spacing_mass)# was 8.5
centers_mass = [round(x + spacing_mass/2,2) for x in ranges_mass[:-1]]

#spacing = 0.5
#ranges = np.arange(8.5, 12.5, spacing)# did start at 8.5



spacing_z = 0.04
ranges_z = np.arange(0.0, 0.32, spacing_z)
centers_z= [round(x + spacing_z/2,2) for x in ranges_z[:-1]]

print('previous z centers', centers_z)
print('previous mass centers', centers_mass)
'''



#cats, bins

nbins = 6

cats_mass, bins_mass = pd.qcut(df['log_stellar_mass_from_color'], q=nbins, retbins=True, precision = 1)
#df['mass_bin'] = cats_mass
cats_z, bins_z = pd.qcut(df['z'], q=nbins, retbins=True, precision = 1)
#df['z_bin'] = cats_z
print('~~~~~~~~~~~~~~')
centers_z = [(bins_z[x+1] - bins_z[x])/2 + bins_z[x] for x in range(len(bins_z)-1)]
centers_mass = [(bins_mass[x+1] - bins_mass[x])/2 + bins_mass[x] for x in range(len(bins_mass)-1)]
print(bins_z)
print(centers_z)

print(bins_mass)
print(centers_mass)






count = {}
f_merg = {}
f_merg_avg = {}
f_merg_std = {}
for i in range(len(bins_z)-1):
	bin_start_z = bins_z[i]
	bin_end_z = bins_z[i+1]
	print('start z ', bin_start_z, 'stop z ', bin_end_z)
	for j in range(len(bins_mass)-1):
		bin_start_mass = bins_mass[j]
		bin_end_mass = bins_mass[j+1]
		print('start mass ', bin_start_mass, 'stop mass ', bin_end_mass)
		# build dataset
		df_select = final_merged[(final_merged['z'] > bin_start_z) & (final_merged['z'] < bin_end_z) & (final_merged['log_stellar_mass_from_color'] > bin_start_mass) & (final_merged['log_stellar_mass_from_color'] < bin_end_mass)]
		count[centers_z[i],centers_mass[j]] = len(df_select)

		# Now for each of these go through all of the LDA tables and match and pull a merger fraction from this :)

		
		list_of_prior_files = glob.glob('../Tables/change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'*'+str(type_marginalized)+'.txt')
		
		# Okay go through all of these, load them up, and match to the object IDs:
		f_merg[centers_z[i],centers_mass[j]] = []
		for p in range(len(list_of_prior_files)):
			if p > 10: 
				continue
			prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[p],header=[0],sep='\t')
			# cut it way down
			table = prior_file[['ID','p_merg']]
			#table = table[0:100]
			merged = table.merge(df_select, left_on='ID', right_on='objID')
			'''
			if len(merged) == 0:
				f_merg_here = 0
				continue
			print('length of merge', len(merged))
			'''
			
			# Count the fraction of things greater than 0.5:
			try:
				fmerg_here = len(np.where(merged['p_merg'] > 0.5)[0])/len(merged)
			except ZeroDivisionError:
				continue
			f_merg[centers_z[i],centers_mass[j]].append(fmerg_here)
		f_merg_avg[centers_z[i],centers_mass[j]] = np.median(f_merg[centers_z[i],centers_mass[j]])
		f_merg_std[centers_z[i],centers_mass[j]] = np.std(f_merg[centers_z[i],centers_mass[j]])



print('fmerg', f_merg)
print('count', count)
print('max count', count[max(count)])
max_count = count[max(count, key=count.get)]
print('also max?', max_count)

plt.clf()

# there should be 8 different redshifts
colors = ['#493843','#61988E','#A0B2A6','#CBBFBB','#EABDA8','#FF9000','#DE639A','#D33E43']
colors = ['#C5979D','#E7BBE3','#78C0E0','#449DD1','#3943B7','#150578','#0E0E52','black']
colors = ['#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black']

print('centers of mass', centers_mass)
print('centers of zs', centers_z)

color_count = 0
for zcen in centers_z:
	
	# Go through and for every mass make a plot
	avgs = []
	stds = []
	first_count = 0
	for masscen in centers_mass:
		#avgs.append(f_merg_avg[zcen,masscen])
		#stds.append(f_merg_std[zcen,masscen])
		print('mass and z center', masscen, zcen)
		adjusted_location = 0.03*(color_count - nbins/2 +0.5) + masscen#1.35*(zcen - 0.14) + masscen
		print('alpha', 'all', count[zcen,masscen]/max_count)
		print('alpha', 'numerator', count[zcen,masscen])
		print('alpha', 'denominator',max_count)
		if first_count == 0:
			plt.scatter(adjusted_location, f_merg_avg[zcen,masscen], 
				color=colors[color_count], label=str(round(bins_z[color_count],3))+'$ < z < $'+str(round(bins_z[color_count+1],3)), marker='s',
				alpha = count[zcen,masscen]/max_count)
		if count[zcen,masscen] < 0.5* max_count:
			plt.errorbar(adjusted_location, f_merg_avg[zcen,masscen], yerr = f_merg_std[zcen,masscen], 
				color=colors[color_count], linestyle='None', marker='s',
				alpha = count[zcen,masscen]/max_count)
		else:
			plt.errorbar(adjusted_location, f_merg_avg[zcen,masscen], yerr = f_merg_std[zcen,masscen], 
				color=colors[color_count], linestyle='None', marker='s')

		first_count+=1

	#adjusted_location = [1.3*(zcen - 0.14) + x for x in centers_mass]
	#plt.scatter(adjusted_location, avgs, color=colors[color_count], label='$z = $'+str(zcen), marker='s')
	#plt.errorbar(adjusted_location, avgs, yerr = stds, color=colors[color_count], linestyle='None', marker='s')

	color_count+=1
for boundary in bins_mass:
	plt.axvline(x = boundary, ls=':', color='black')

leg = plt.legend()
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xlabel(r'log stellar mass')
plt.ylabel(r'$f_{\mathrm{merg}}$')
plt.ylim([0,1])
plt.xlim([bins_mass[0]-0.1,bins_mass[-1]+0.1])
plt.savefig('../Figures/2D_hist_equal_bins_z_mass_'+str(run)+'_'+suffix+'.png', dpi=1000)


STOP

STOP


d1 = final_merged.assign(
	Mass_cut=pd.cut(df.log_stellar_mass_from_color, ranges_mass),
	z_cut=pd.cut(final_merged.z, ranges_z))
	
print(d1)

# Then combine the two categories:
d2 = d1.assign(cartesian=pd.Categorical(d1.filter(regex='_cut').apply(tuple, 1)))
print(d2)

print(set(d2['cartesian'].tolist()))





count = {}
f_merg = {}
f_merg_avg = {}
f_merg_std = {}
for i in range(len(ranges_z)-1):
	bin_start_z = ranges_z[i]
	bin_end_z = ranges_z[i+1]
	print('start z ', bin_start_z, 'stop z ', bin_end_z)
	for j in range(len(ranges_mass)-1):
		bin_start_mass = ranges_mass[j]
		bin_end_mass = ranges_mass[j+1]
		print('start mass ', bin_start_mass, 'stop mass ', bin_end_mass)
		# build dataset
		df = final_merged[(final_merged['z'] > bin_start_z) & (final_merged['z'] < bin_end_z) & (final_merged['log_stellar_mass_from_color'] > bin_start_mass) & (final_merged['log_stellar_mass_from_color'] < bin_end_mass)]
		count[round(bin_start_z+(bin_end_z-bin_start_z)/2,2),round(bin_start_mass+(bin_end_mass-bin_start_mass)/2,2)] = len(df)

		# Now for each of these go through all of the LDA tables and match and pull a merger fraction from this :)

		
		list_of_prior_files = glob.glob('../Tables/change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'*'+str(type_marginalized)+'.txt')
		# Okay go through all of these, load them up, and match to the object IDs:
		f_merg[round(bin_start_z+(bin_end_z-bin_start_z)/2,2),round(bin_start_mass+(bin_end_mass-bin_start_mass)/2,2)] = []
		for j in range(len(list_of_prior_files)):
			if j > 10: 
				continue
			prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[j],header=[0],sep='\t')
			# cut it way down
			table = prior_file[['ID','p_merg']]
			#table = table[0:100]
			merged = table.merge(df, left_on='ID', right_on='objID')
			'''
			if len(merged) == 0:
				f_merg_here = 0
				continue
			print('length of merge', len(merged))
			'''
			
			# Count the fraction of things greater than 0.5:
			try:
				fmerg_here = len(np.where(merged['p_merg'] > 0.5)[0])/len(merged)
			except ZeroDivisionError:
				continue
			f_merg[round(bin_start_z+(bin_end_z-bin_start_z)/2,2),round(bin_start_mass+(bin_end_mass-bin_start_mass)/2,2)].append(fmerg_here)
		f_merg_avg[round(bin_start_z+(bin_end_z-bin_start_z)/2,2),round(bin_start_mass+(bin_end_mass-bin_start_mass)/2,2)] = np.median(f_merg[round(bin_start_z+(bin_end_z-bin_start_z)/2,2),round(bin_start_mass+(bin_end_mass-bin_start_mass)/2,2)])
		f_merg_std[round(bin_start_z+(bin_end_z-bin_start_z)/2,2),round(bin_start_mass+(bin_end_mass-bin_start_mass)/2,2)] = np.std(f_merg[round(bin_start_z+(bin_end_z-bin_start_z)/2,2),round(bin_start_mass+(bin_end_mass-bin_start_mass)/2,2)])

labels, data = list(f_merg_avg.keys()), f_merg_avg.values()

print('labels', labels)

print(f_merg_avg)
print(f_merg_std)
print(count)

f_merg_2d = np.zeros((len(centers_z), len(centers_mass)))
count_array = np.zeros((len(centers_z), len(centers_mass)))
for x in range(len(centers_z)):
	for y in range(len(centers_mass)):
		for j in range(len(f_merg_avg)):
			if labels[j][0]==centers_z[x] and labels[j][1]==centers_mass[y]:
				f_merg_2d[y,x] = f_merg_avg[labels[j]]
				count_array[y,x] = count[labels[j]]


'''
plt.clf()
fig = plt.figure(figsize = (11,5))
ax0 = fig.add_subplot(121)
im0 = ax0.imshow(f_merg_2d)
ax0.set_title(r'$f_{\mathrm{merg}}$')
plt.colorbar(im0, fraction = 0.046)
ax0.set_xlabel(r'$z$')
ax0.set_ylabel(r'log stellar mass')
ax0.set_xticks([0,1,2,3,4,5,6])
ax0.set_yticks([0,1,2,3,4,5,6])
ax0.set_xticklabels(centers_z)
ax0.set_yticklabels(centers_mass)


ax1 = fig.add_subplot(122)
im1 = ax1.imshow(np.ma.masked_where(count_array==0, count_array))
plt.colorbar(im1, fraction = 0.046)
ax1.set_title(r'Count')
ax1.set_xlabel(r'$z$')
#ax1.set_ylabel(r'log stellar mass')
ax1.set_xticks([0,1,2,3,4,5,6])
ax1.set_yticks([0,1,2,3,4,5,6])
ax1.set_xticklabels(centers_z)
ax1.set_yticklabels(centers_mass)
plt.tight_layout()
plt.savefig('../Figures/2d_z_stellar_'+str(run)+'_'+str(suffix)+'.png', dpi=1000)
'''

# Also see if you can plot a boxplot instead:

print('fmerg', f_merg)
print('count', count)
print('max count', count[max(count)])
max_count = count[max(count, key=count.get)]
print('also max?', max_count)

plt.clf()

# there should be 8 different redshifts
colors = ['#493843','#61988E','#A0B2A6','#CBBFBB','#EABDA8','#FF9000','#DE639A','#D33E43']
colors = ['#C5979D','#E7BBE3','#78C0E0','#449DD1','#3943B7','#150578','#0E0E52','black']
colors = ['#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black']

print('centers of mass', centers_mass)
print('centers of zs', centers_z)

color_count = 0
for zcen in centers_z:
	
	# Go through and for every mass make a plot
	avgs = []
	stds = []
	first_count = 0
	for masscen in centers_mass:
		#avgs.append(f_merg_avg[zcen,masscen])
		#stds.append(f_merg_std[zcen,masscen])
		adjusted_location = 1.35*(zcen - 0.14) + masscen
		print('alpha', 'all', count[zcen,masscen]/max_count)
		print('alpha', 'numerator', count[zcen,masscen])
		print('alpha', 'denominator',max_count)
		if first_count == 0:
			plt.scatter(adjusted_location, f_merg_avg[zcen,masscen], 
				color=colors[color_count], label='$z = $'+str(zcen), marker='s',
				alpha = count[zcen,masscen]/max_count)
		if count[zcen,masscen] < 0.5* max_count:
			plt.errorbar(adjusted_location, f_merg_avg[zcen,masscen], yerr = f_merg_std[zcen,masscen], 
				color=colors[color_count], linestyle='None', marker='s',
				alpha = count[zcen,masscen]/max_count)
		else:
			plt.errorbar(adjusted_location, f_merg_avg[zcen,masscen], yerr = f_merg_std[zcen,masscen], 
				color=colors[color_count], linestyle='None', marker='s')

		first_count+=1

	#adjusted_location = [1.3*(zcen - 0.14) + x for x in centers_mass]
	#plt.scatter(adjusted_location, avgs, color=colors[color_count], label='$z = $'+str(zcen), marker='s')
	#plt.errorbar(adjusted_location, avgs, yerr = stds, color=colors[color_count], linestyle='None', marker='s')

	color_count+=1
for boundary in ranges_mass:
	plt.axvline(x = boundary, ls=':', color='black')

leg = plt.legend()
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xlabel(r'log stellar mass')
plt.ylabel(r'$f_{\mathrm{merg}}$')
plt.ylim([0,1])
plt.savefig('../Figures/2D_hist_z_mass_'+str(run)+'_'+suffix+'.png', dpi=1000)


STOP


plt.clf()
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)
ax.boxplot(data, showfliers=False)
ax.set_xticks(range(1, len(labels) + 1))


#ax.set_xticklabels([f'{label}\n{format(count,"5.0e")}' for label, count in zip(labels, counts)])


'''
labels = [str(spacing)] + [f'{r + str(spacing):.2f}' for r in ranges[1:-1]]#-{r + 0.015:.2f}
ax.set_xticklabels([f'{label}\n{count / sum(counts) * 100:.1f}%' for label, count in zip(labels, counts)])

'''

'''
ax.set_ylabel(r'$f_{\mathrm{merg}}$')
ax.set_xlabel(r'stellar mass')

x_bt = [0.5,7.5]
x_bt = [np.min(ranges),len(ranges)]
ax.fill_between(x_bt, y_whisker_top, y_top, facecolor='grey')
ax.fill_between(x_bt, y_whisker_bottom, y_bottom, facecolor='grey')
ax.fill_between(x_bt, y_Q3, y_whisker_top, facecolor='grey', alpha=0.5)
ax.fill_between(x_bt, y_Q1, y_whisker_bottom, facecolor='grey', alpha=0.5)

ax.axhline(y = median_all, color='orange', ls='-')
'''
#ax.axhline(y = bottom_all, color='black', ls='--')
#ax.axhline(y = top_all, color='black', ls='--')

plt.tight_layout()
#ax.set_xlim([0.5,10.5]) # x_bt is from [0,11], which is the length of range
# for B/T ranges = np.arange(0.0,1.1,0.1)
# here, range is ranges = np.arange(0.0, 0.32, spacing)
#ax.set_xlim([np.min(ranges)+0.5,len(ranges)-0.5])

#ax.set_ylim([0,top_all+0.3])

ax.set_xticks([0,1,2,3,4,5,6])
ax.set_yticks([0,1,2,3,4,5,6])
ax.set_xticklabels(centers_z)
plt.savefig('../Figures/boxplot_2D_z_mass_'+str(run)+'_'+suffix+'.png', dpi=1000)





'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the prior plot
# get the flagged ones
list_of_prior_files = glob.glob('../Tables/change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'*'+'flag.txt')


f_in = []
for j in range(len(list_of_prior_files)):
	split_1 = str.split(list_of_prior_files[j],'_merger_')[1]
	split_2 = str.split(split_1,'_')[0]
	f_in.append(float(str.split(split_2,'.txt')[0]))

print(f_in)

f_merg_out = []
for j in range(len(list_of_prior_files)):
	prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[j],header=[0],sep='\t')
	
	# cut it way down
	table = prior_file[['ID','p_merg','low S/N','outlier predictor']]
	
	p_merg_clean = table[(table['low S/N']==0) & (table['outlier predictor']==0)]['p_merg'].values 
	
	# Count how many of these are greater than 0.5:
	f_merg_out.append(len(np.where(p_merg_clean > 0.5)[0])/len(p_merg_clean))
	
print('f_in', f_in)
print('f_out', f_merg_out)


STOP
'''

'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Option for going through the master LDA table and looking at the p_merg values for individual galaxies
list_of_prior_files = glob.glob('../Tables/change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'*')


f_in = []
for j in range(len(list_of_prior_files)):
	split_1 = str.split(list_of_prior_files[j],'_merger_')[1]
	f_in.append(float(str.split(split_1,'.txt')[0]))

print(f_in)

p_merg_ind = {}
for i in range(1000):#len(final_merged)):
	ID = final_merged['objID'].values[i]
	p_merg_ind[ID] = []
	for j in range(len(list_of_prior_files)):
		prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[j],header=[0],sep='\t')
		
		# cut it way down
		table = prior_file[['ID','p_merg']]
		find = np.where(table['ID'].values==ID)[0]
		if find.size == 0:
			p_merg_ind[ID].append(-99)
			continue
		p_merg_ind[ID].append(table[table['ID'] == ID]['p_merg'].values[0])
		
		# select the right ID
print(p_merg_ind)

'''


'''
bad_value = -99
for key, value in p_merg_ind.items():
	if value.count(value[0]) == len(value):
		del p_merg_ind[key]
		break
''' 



ranges = np.arange(0.0, 1.1, 0.1)# was 0.1 for the last number
x_bt = [np.min(ranges),len(ranges)]



# Now what about pulling from the prior tables and getting the f_merg 
# distribution for each category

# So that involves pulling all of the IDs from each bin and then getting their f_merg from each of the tables
# Lets just do by category
f_merg = {}

# First, calculate the overall merger fraction for the sample:


# Now for each of these go through all of the LDA tables and match and pull a merger fraction from this :)

list_of_prior_files = glob.glob('../Tables/change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'*'+str(type_marginalized)+'.txt')


# Okay go through all of these, load them up, and match to the object IDs:
f_merg_all = []
for j in range(len(list_of_prior_files)):
	prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[j],header=[0],sep='\t')
	
	# cut it way down
	table = prior_file[['ID','p_merg']]
	merged = table.merge(df, left_on='ID', right_on='objID')
	
	# Count the fraction of things greater than 0.5:
	try:
		fmerg = len(np.where(merged['p_merg'] > 0.5)[0])/len(merged)
	except ZeroDivisionError:
		continue
	f_merg_all.append(fmerg)


print('iqr',iqr(f_merg_all))
print('np.quantile', np.quantile(f_merg_all, 0.25) - 1.5* iqr(f_merg_all), np.quantile(f_merg_all, 0.25),
	np.quantile(f_merg_all, 0.5),
	np.quantile(f_merg_all, 0.75), np.quantile(f_merg_all, 0.75) + 1.5 * iqr(f_merg_all))

print(np.mean(f_merg_all), np.std(f_merg_all))



bottom_all = np.quantile(f_merg_all, 0.25) - 1.5* iqr(f_merg_all)
median_all = np.quantile(f_merg_all, 0.5)
top_all = np.quantile(f_merg_all, 0.75) + 1.5* iqr(f_merg_all)
Q1_all = np.quantile(f_merg_all, 0.25)
Q3_all = np.quantile(f_merg_all, 0.75)


#x_bt = [0,11]
y_top = [1, 1]
y_bottom = [0,0]
y_whisker_bottom = [bottom_all, bottom_all]
y_whisker_top = [top_all, top_all]
y_Q1 = [Q1_all, Q1_all]
y_Q3 = [Q3_all, Q3_all]




f_merg = {}
for i in range(len(ranges)-1):
	bin_start = ranges[i]
	bin_end = ranges[i+1]
	print('start', bin_start, 'stop', bin_end)
	# build dataset
	df = final_merged[(final_merged['B/T'] > bin_start) & (final_merged['B/T'] < bin_end)]
	

	# Now for each of these go through all of the LDA tables and match and pull a merger fraction from this :)

	
	
	# Okay go through all of these, load them up, and match to the object IDs:
	f_merg[round(bin_start+(bin_end-bin_start)/2,2)] = []
	for j in range(len(list_of_prior_files)):
		prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[j],header=[0],sep='\t')
		# cut it way down
		table = prior_file[['ID','p_merg']]
		merged = table.merge(df, left_on='ID', right_on='objID')
		
		print('this is the number of galaxies we are considering in this bin', len(merged))
		
		# Count the fraction of things greater than 0.5:
		try:
			fmerg = len(np.where(merged['p_merg'] > 0.5)[0])/len(merged)
		except ZeroDivisionError:
			continue
		f_merg[round(bin_start+(bin_end-bin_start)/2,2)].append(fmerg)
		
	#df_LDA = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run+'_flags.txt',header=[0],sep='\t')

# Python 3.5+
labels, data = [*zip(*f_merg.items())]  # 'transpose' items to parallel key, value lists

# or backwards compatable    
labels, data = f_merg.keys(), f_merg.values()

plt.clf()
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111)
ax.boxplot(data, showfliers=False)
ax.set_xticks(range(1, len(labels) + 1))

final_merged['category'] = pd.cut(final_merged['B/T'], ranges)
counts = final_merged.groupby(['category'])['B/T'].count()

#ax.set_xticklabels(labels)
ax.set_ylabel(r'$f_{\mathrm{merg}}$')
ax.set_xlabel('B/T')

ax.set_xticklabels([f'{label}\n{format(count,"2.0e")}' for label, count in zip(labels, counts)])


ax.fill_between(x_bt, y_whisker_top, y_top, facecolor='grey')
ax.fill_between(x_bt, y_whisker_bottom, y_bottom, facecolor='grey')
ax.fill_between(x_bt, y_Q3, y_whisker_top, facecolor='grey', alpha=0.5)
ax.fill_between(x_bt, y_Q1, y_whisker_bottom, facecolor='grey', alpha=0.5)

ax.axhline(y = median_all, color='orange', ls='-')
#ax.axhline(y = bottom_all, color='black', ls='--')
#ax.axhline(y = top_all, color='black', ls='--')

plt.tight_layout()
ax.set_xlim([0.5,10.5]) # x_bt is from [0,11], which is the length of range
ax.set_ylim([0,top_all+0.1])
plt.savefig('../Figures/boxplot_B_T_fmerg_'+str(run)+'_'+suffix+'.png', dpi=1000)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~redshift~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# Same thing but for redshift:
# Okay but I'd actually like to bin up by B/T and then plot p_merg and merger fraction
# Also bin up the data

spacing = 0.04
ranges = np.arange(0.0, 0.32, spacing)


plt.clf()

if suffix == 'mendel':
	final_merged['category'] = pd.cut(final_merged['z_y'], ranges)
	counts = final_merged.groupby(['category'])['z_y'].count()
if suffix == 'color':
	final_merged['category'] = pd.cut(final_merged['z_x'], ranges)
	counts = final_merged.groupby(['category'])['z_x'].count()

ax = sns.violinplot(x='category', y='p_merg_x', data=final_merged, palette='Greens')
labels = [str(spacing)] + [f'{r + spacing:.2f}' for r in ranges[1:-1]]#-{r + 0.015:.2f}
ax.set_xticklabels([f'{label}\n{count}' for label, count in zip(labels, counts)])
ax.set_ylim([0,1])
#plt.tight_layout()


plt.savefig('../Figures/violin_z_'+str(run)+'_'+suffix+'.png', dpi=1000)


# Now what about pulling from the prior tables and getting the f_merg 
# distribution for each category

# So that involves pulling all of the IDs from each bin and then getting their f_merg from each of the tables
# Lets just do by category
f_merg = {}
for i in range(len(ranges)-1):
	bin_start = ranges[i]
	bin_end = ranges[i+1]
	print('start', bin_start, 'stop', bin_end)
	# build dataset
	if suffix == 'mendel':
		df = final_merged[(final_merged['z_y'] > bin_start) & (final_merged['z_y'] < bin_end)]
	if suffix == 'color':
		df = final_merged[(final_merged['z_x'] > bin_start) & (final_merged['z_x'] < bin_end)]
	

	# Now for each of these go through all of the LDA tables and match and pull a merger fraction from this :)

	
	# Okay go through all of these, load them up, and match to the object IDs:
	f_merg[round(bin_start+(bin_end-bin_start)/2,2)] = []
	for j in range(len(list_of_prior_files)):
		prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[j],header=[0],sep='\t')
		# cut it way down
		table = prior_file[['ID','p_merg']]
		merged = table.merge(df, left_on='ID', right_on='objID')
		
		# Count the fraction of things greater than 0.5:
		try:
			fmerg = len(np.where(merged['p_merg'] > 0.5)[0])/len(merged)
		except ZeroDivisionError:
			continue
		f_merg[round(bin_start+(bin_end-bin_start)/2,2)].append(fmerg)
		
	#df_LDA = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run+'_flags.txt',header=[0],sep='\t')

# Get the redshift dependencies from Lopez-Sanjuan:
print(ranges)
zs_parameterized = [0.001*(1+z)**(5.4) for z in ranges]

# Python 3.5+
labels, data = [*zip(*f_merg.items())]  # 'transpose' items to parallel key, value lists

# or backwards compatable    
labels, data = f_merg.keys(), f_merg.values()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# Now, fit a functional form exponential


def func(z, a, b):#, c):
    return a * (1 + z)**b #* (1 + M)**c

x = np.array(list(labels))
y_data = np.array([np.median(list(x)) for x in data])
y_spread = np.array([np.std(list(x)) for x in data])
print('x', x)
print('y', y_data)



popt, pcov = curve_fit(func, x, y_data,sigma = y_spread,absolute_sigma = True)
sigma_ab = np.sqrt(np.diagonal(pcov))



print('popt', popt)
print('covariance', pcov)

plt.clf()
plt.scatter(x, y_data, color='black')

# plotting the unaveraged data
flier_kwargs = dict(marker = 'o', markerfacecolor = 'silver',markersize = 3, alpha=0.7)
line_kwargs = dict(color = 'k', linewidth = 1)
#bp = plt.boxplot(y_data, positions = x, capprops = line_kwargs, boxprops = line_kwargs, whiskerprops = line_kwargs, 
#	medianprops = line_kwargs,flierprops = flier_kwargs,widths = 5,manage_ticks = False)


plt.plot(x, func(x,*popt), color='black')

bound_upper = func(x, *(popt + sigma_ab))
bound_lower = func(x, *(popt - sigma_ab))
# plotting the confidence intervals
plt.fill_between(x, bound_lower, bound_upper,color = 'black', alpha = 0.15)

from uncertainties import ufloat
a = ufloat(popt[0], sigma_ab[0])
b = ufloat(popt[1], sigma_ab[1])
text_res = "Best fit parameters:\n{} * (1 + z)^{}".format(a, b)

plt.text(0.02, 0.5*np.max(bound_upper), text_res)
plt.xlabel(r'$z$')
plt.ylabel(r'$f_{\mathrm{merg}}$')


plt.savefig('../Figures/exponential_z_'+str(run)+'_'+suffix+'.png', dpi=1000)




plt.clf()
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)
ax.boxplot(data, showfliers=False)
ax.set_xticks(range(1, len(labels) + 1))


ax.set_xticklabels([f'{label}\n{format(count,"5.0e")}' for label, count in zip(labels, counts)])


'''
labels = [str(spacing)] + [f'{r + str(spacing):.2f}' for r in ranges[1:-1]]#-{r + 0.015:.2f}
ax.set_xticklabels([f'{label}\n{count / sum(counts) * 100:.1f}%' for label, count in zip(labels, counts)])

'''
ax.set_ylabel(r'$f_{\mathrm{merg}}$')
ax.set_xlabel(r'$z$')

x_bt = [0.5,7.5]
x_bt = [np.min(ranges),len(ranges)]
ax.fill_between(x_bt, y_whisker_top, y_top, facecolor='grey')
ax.fill_between(x_bt, y_whisker_bottom, y_bottom, facecolor='grey')
ax.fill_between(x_bt, y_Q3, y_whisker_top, facecolor='grey', alpha=0.5)
ax.fill_between(x_bt, y_Q1, y_whisker_bottom, facecolor='grey', alpha=0.5)

ax.axhline(y = median_all, color='orange', ls='-')
#ax.axhline(y = bottom_all, color='black', ls='--')
#ax.axhline(y = top_all, color='black', ls='--')

ax.plot(ranges, zs_parameterized, color='black')

plt.tight_layout()
#ax.set_xlim([0.5,10.5]) # x_bt is from [0,11], which is the length of range
# for B/T ranges = np.arange(0.0,1.1,0.1)
# here, range is ranges = np.arange(0.0, 0.32, spacing)
ax.set_xlim([np.min(ranges)+0.5,len(ranges)-0.5])
ax.set_ylim([0,top_all+0.3])
plt.savefig('../Figures/boxplot_z_fmerg_'+str(run)+'_'+suffix+'.png', dpi=1000)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~mass~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# Same thing but for redshift:
# Okay but I'd actually like to bin up by B/T and then plot p_merg and merger fraction
# Also bin up the data





spacing = 0.5
ranges = np.arange(8.5, 12.5, spacing)# did start at 8.5

plt.clf()

if suffix == 'mendel':
	final_merged['category'] = pd.cut(final_merged['logMt'], ranges)
	counts = final_merged.groupby(['category'])['logMt'].count()
if suffix == 'color':
	final_merged['category'] = pd.cut(final_merged['log_stellar_mass_from_color'], ranges)
	counts = final_merged.groupby(['category'])['log_stellar_mass_from_color'].count()

ax = sns.violinplot(x='category', y='p_merg_x', data=final_merged, palette='Greens')
labels = [str(spacing)] + [f'{r + spacing:.2f}' for r in ranges[1:-1]]#-{r + 0.015:.2f}
ax.set_xticklabels([f'{label}\n{count / sum(counts) * 100:.1f}%' for label, count in zip(labels, counts)])
ax.set_ylim([0,1])
#plt.tight_layout()


plt.savefig('../Figures/violin_mass_'+str(run)+'_'+suffix+'.png', dpi=1000)


# Now what about pulling from the prior tables and getting the f_merg 
# distribution for each category

# So that involves pulling all of the IDs from each bin and then getting their f_merg from each of the tables
# Lets just do by category
f_merg = {}
for i in range(len(ranges)-1):
	bin_start = ranges[i]
	bin_end = ranges[i+1]
	print('start', bin_start, 'stop', bin_end)
	# build dataset
	if suffix == 'mendel':
		df = final_merged[(final_merged['logMt'] > bin_start) & (final_merged['logMt'] < bin_end)]
	if suffix == 'color':
		df = final_merged[(final_merged['log_stellar_mass_from_color'] > bin_start) & (final_merged['log_stellar_mass_from_color'] < bin_end)]
	

	# Now for each of these go through all of the LDA tables and match and pull a merger fraction from this :)

	
	# Okay go through all of these, load them up, and match to the object IDs:
	f_merg[round(bin_start+(bin_end-bin_start)/2,2)] = []
	for j in range(len(list_of_prior_files)):
		prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[j],header=[0],sep='\t')
		# cut it way down
		table = prior_file[['ID','p_merg']]
		merged = table.merge(df, left_on='ID', right_on='objID')
		
		# Count the fraction of things greater than 0.5:
		try:
			fmerg = len(np.where(merged['p_merg'] > 0.5)[0])/len(merged)
		except ZeroDivisionError:
			continue
		f_merg[round(bin_start+(bin_end-bin_start)/2,2)].append(fmerg)
		
	#df_LDA = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run+'_flags.txt',header=[0],sep='\t')

# Python 3.5+
labels, data = [*zip(*f_merg.items())]  # 'transpose' items to parallel key, value lists

# or backwards compatable    
labels, data = f_merg.keys(), f_merg.values()
plt.clf()
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)
ax.boxplot(data, showfliers=False)
ax.set_xticks(range(1, len(labels) + 1))


ax.set_xticklabels([f'{label}\n{format(count,"5.0e")}' for label, count in zip(labels, counts)])
#format(5482.52291, "5.2e")


'''
labels = [str(spacing)] + [f'{r + str(spacing):.2f}' for r in ranges[1:-1]]#-{r + 0.015:.2f}
ax.set_xticklabels([f'{label}\n{count / sum(counts) * 100:.1f}%' for label, count in zip(labels, counts)])

'''
ax.set_ylabel(r'$f_{\mathrm{merg}}$')
ax.set_xlabel(r'log total stellar mass')
x_bt = [0.5,7.5]
ax.fill_between(x_bt, y_whisker_top, y_top, facecolor='grey')
ax.fill_between(x_bt, y_whisker_bottom, y_bottom, facecolor='grey')
ax.fill_between(x_bt, y_Q3, y_whisker_top, facecolor='grey', alpha=0.5)
ax.fill_between(x_bt, y_Q1, y_whisker_bottom, facecolor='grey', alpha=0.5)

ax.axhline(y = median_all, color='orange', ls='-')
#ax.axhline(y = bottom_all, color='black', ls='--')
#ax.axhline(y = top_all, color='black', ls='--')

plt.tight_layout()
ax.set_ylim([0,top_all+0.5])
plt.savefig('../Figures/boxplot_mass_fmerg_'+str(run)+'_'+suffix+'.png', dpi=1000)

