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
from statsmodels.stats.outliers_influence import summary_table
from scipy.stats import iqr


# path
dir = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'

type_gal = 'predictors'
run = 'major_merger'
suffix = 'color_complete_z_bins_z_lt_0_pt_1_pozzetti'#_1'#_complete'
suffix = 'measurements'
type_marginalized = '_flags_cut_segmap'
num = None
savefigs = False
save_df = False

# The below are flags for 
s_n_cut = False
if s_n_cut:
	add_on_binned_table = 's_n_15_to_20'
else:
	add_on_binned_table = ''
low_s_n = 15
high_s_n = 20
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
masstable = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_'+str(suffix)+'.txt',header=[0],sep='\t')
print(masstable.columns)

masstable = masstable[masstable['logBD'] < 13]#log_stellar_mass_from_color


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

try:
	final_merged = final_merged[['objID','p_merg_x','LD1','z','log_stellar_mass_from_color']]
except KeyError:
	final_merged = final_merged[['objID','p_merg_x','LD1','z_x','log_stellar_mass_from_color']]
df = final_merged


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run OLS with predictor values and z and stellar mass and p_merg:
# Now merge these two and run an OLS:

df_pred_merged = df_predictors_clean.merge(df, left_on='ID', right_on='objID')# Now merging on dr8


print(df_pred_merged.columns)
try:
	X = df_pred_merged[[ 'z','Concentration (C)','Gini','Asymmetry (A)']]
except KeyError:
	X = df_pred_merged[[ 'z_x','Concentration (C)','Gini','Asymmetry (A)']]
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

# Can also run a similar analysis to see if z and S/N are related (spoiler: they are)

# Does z predict S/N?

# Do a 2D regression:
try:
	X = df_pred_merged[['z']] 
except KeyError:
	X = df_pred_merged[['z_x']] 
y = df_pred_merged['S/N']
## fit a OLS model with intercept on mass and z
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
print(est.summary())





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Do multi-linear regression without doing the following fancy marginalization as part of it
try:
	X = df[[ 'z']]
except KeyError:
	X = df[[ 'z_x']]
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

if suffix == 'color_complete_z_bins':
	centers_z = [0.032275000000000005, 0.073274, 0.091613, 0.1115435, 0.137801, 0.5676965]
	bins_z = [5.09000e-04, 6.40410e-02, 8.25070e-02, 1.00719e-01, 1.22368e-01, 1.53234e-01, 9.82159e-01]

if suffix == 'measurements' or suffix == 'color_complete_z_bins_z_lt_0_pt_1' or suffix == 'color_complete_z_bins_z_lt_0_pt_1_pozzetti' or suffix == 'color_complete_z_bins_z_lt_0_pt_1_pozzetti_50' or suffix == 'color_complete_z_bins_z_lt_0_pt_1_pozzetti_99' or suffix == 'color_complete_z_bins_z_lt_0_pt_1_pozzetti_1':
	centers_z = [0.024610499999999997, 0.0562255, 0.0687485, 0.0779165, 0.0863525, 0.0953145]
	bins_z = [0.000509, 0.048712, 0.063739, 0.073758, 0.082075, 0.09063,  0.099999]

if suffix == 'color_complete' or suffix == 'color':
	cats_z, bins_z = pd.qcut(df['z'], q=nbins, retbins=True, precision = 1)
	#df['z_bin'] = cats_z
	
	centers_z = [(bins_z[x+1] - bins_z[x])/2 + bins_z[x] for x in range(len(bins_z)-1)]
centers_mass = [(bins_mass[x+1] - bins_mass[x])/2 + bins_mass[x] for x in range(len(bins_mass)-1)]

print('~~~~~~~~~~~~~~')
print(bins_z)
print(centers_z)

print(bins_mass)
print(centers_mass)

# Now that you have the bins in both dimensions, make a figure of your different bins:
plt.clf()
'''
plt.scatter(df_pred_merged['z_x'].values, df_pred_merged['log_stellar_mass_from_color'].values, color='orange', s=0.2)
plt.annotate(str(len(df_pred_merged['z_x'].values)), 
	xy = (np.mean(df_pred_merged['z_x'].values), np.mean(df_pred_merged['log_stellar_mass_from_color'].values)), 
	xycoords='data', color='black')
'''
for i in range(len(bins_z)-1):
	bin_start_z = bins_z[i]
	bin_end_z = bins_z[i+1]
	for j in range(len(bins_mass)-1):
		bin_start_mass = bins_mass[j]
		bin_end_mass = bins_mass[j+1]
		df_select = df_pred_merged[(df_pred_merged['z_x'] > bin_start_z) 
			& (df_pred_merged['z_x'] < bin_end_z) 
			& (df_pred_merged['log_stellar_mass_from_color'] > bin_start_mass) 
			& (df_pred_merged['log_stellar_mass_from_color'] < bin_end_mass)]
		plt.scatter(df_select['z_x'].values, df_select['log_stellar_mass_from_color'].values, 
			s=0.2)
		plt.annotate(str(len(df_select['z_x'].values)), 
			xy = (np.mean(df_select['z_x'].values) - 0.005, np.mean(df_select['log_stellar_mass_from_color'].values - 0.05)), 
			xycoords='data', color='black')
plt.xlabel(r'$z$')
plt.ylabel('log stellar mass')
if savefigs:
	plt.savefig('../Figures/MCMC_scatter_'+str(run)+'_'+suffix+'_color.png', dpi=1000)
else:
	plt.show()






# try to load up the df of mass centers:
if save_df:

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
			if s_n_cut:#df_pred_merged
				try:
					df_select = df_pred_merged[(df_pred_merged['S/N'] > low_s_n) & (df_pred_merged['S/N'] < high_s_n)
						& (df_pred_merged['z'] > bin_start_z) 
						& (df_pred_merged['z'] < bin_end_z) 
						& (df_pred_merged['log_stellar_mass_from_color'] > bin_start_mass) 
						& (df_pred_merged['log_stellar_mass_from_color'] < bin_end_mass)]
				except KeyError:
					df_select = df_pred_merged[(df_pred_merged['S/N'] > low_s_n) & (df_pred_merged['S/N'] < high_s_n)
						& (df_pred_merged['z_x'] > bin_start_z) 
						& (df_pred_merged['z_x'] < bin_end_z) 
						& (df_pred_merged['log_stellar_mass_from_color'] > bin_start_mass) 
						& (df_pred_merged['log_stellar_mass_from_color'] < bin_end_mass)]
			else:
				try:
					df_select = df_pred_merged[(df_pred_merged['z'] > bin_start_z) 
						& (df_pred_merged['z'] < bin_end_z) 
						& (df_pred_merged['log_stellar_mass_from_color'] > bin_start_mass) 
						& (df_pred_merged['log_stellar_mass_from_color'] < bin_end_mass)]
				except KeyError:
					df_select = df_pred_merged[(df_pred_merged['z_x'] > bin_start_z) 
						& (df_pred_merged['z_x'] < bin_end_z) 
						& (df_pred_merged['log_stellar_mass_from_color'] > bin_start_mass) 
						& (df_pred_merged['log_stellar_mass_from_color'] < bin_end_mass)]
			count[centers_z[i],centers_mass[j]] = len(df_select)


			# Now for each of these go through all of the LDA tables and match and pull a merger fraction from this :)

			
			list_of_prior_files = glob.glob('../Tables/change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'*'+str(type_marginalized)+'.txt')
			
			# Okay go through all of these, load them up, and match to the object IDs:
			f_merg[centers_z[i],centers_mass[j]] = []
			for p in range(len(list_of_prior_files)):
				#if p > 10: 
				#	continue
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


	# find a way to put this into df format
	mass_val = []
	z_val = []
	f_merg_val = []
	f_merg_e_val = []
	count_val = []
	for i in range(len(bins_z)-1):
		for j in range(len(bins_mass)-1):
			f_merg_val.append(f_merg_avg[centers_z[i],centers_mass[j]])
			f_merg_e_val.append(f_merg_std[centers_z[i],centers_mass[j]])
			z_val.append(centers_z[i])
			mass_val.append(centers_mass[j])
			count_val.append(count[centers_z[i],centers_mass[j]])
	# Now make a df out of these lists
	try:
		df_fmerg = pd.DataFrame(list(zip(mass_val, z_val, f_merg_val, f_merg_e_val, count_val)),
	           columns =['mass', 'z', 'fmerg', 'fmerg_std', 'count'])
	except:
		df_fmerg = pd.DataFrame(list(zip(mass_val, z_val, f_merg_val, f_merg_e_val, count_val)),
	           columns =['mass', 'z_x', 'fmerg', 'fmerg_std', 'count'])
	print(df_fmerg)

	df_fmerg.to_csv('../Tables/df_mass_z_fmerg'+str(suffix)+str(add_on_binned_table)+'.csv', sep='\t')
else:
	df_fmerg = pd.read_csv('../Tables/df_mass_z_fmerg'+str(suffix)+str(add_on_binned_table)+'.csv', sep = '\t')

	
df_fmerg = df_fmerg.dropna()
print(df_fmerg)


# Do a 2D regression:
try:
	X = df_fmerg[['mass', 'z']] 
except KeyError:
	X = df_fmerg[['mass', 'z_x']] 
y = df_fmerg['fmerg']
## fit a OLS model with intercept on mass and z
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
print(est.summary())


count = df_fmerg['count'].values

max_count = max(count)
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
	print(zcen)
	print(str(round(bins_z[color_count],3))+'$ < z < $'+str(round(bins_z[color_count+1],3)))
	#color_count+=1
	#continue
	# Grab everything with a z value from the df
	df_select = df_fmerg[df_fmerg['z'] == zcen].dropna().reset_index()
	print(df_select)
	if len(df_select)==0:
		color_count+=1
		continue
	# should already be sorted by mass
	count = df_select['count'].values
	print('total count', np.sum(count))
	masscen = df_select['mass'].values


	
	# to do a linear regression, I'm going to need to define X and y
	x = masscen.reshape(-1,1)
	X = sm.add_constant(x)

	y = df_select['fmerg']
	error = df_select['fmerg_std']

	mu, sigma = 0, 1 # mean and standard deviation
	

	plt.clf()
	
	# iterate
	# save slope values
	slope_list = []
	for num in range(100):
		Y = [y[i] + error[i] * np.random.normal(mu, sigma, 1) for i in range(len(y))]

		#scaler = StandardScaler()
		#scaler.fit(X)
		#X_standardized = scaler.transform(X)


		res = sm.OLS(Y, X).fit()
		
		#plt.scatter(x, Y, s=0.1, color=colors[color_count])
		try:
			slope_list.append(res.params[1])
		except:
			slope_list.append(999)
		
		

		st, data, ss2 = summary_table(res, alpha=0.05)
		fittedvalues = data[:,2]
		predict_mean_se  = data[:,3]
		predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
		predict_ci_low, predict_ci_upp = data[:,6:8].T

		#print(summary_table(res, alpha=0.05))
		

		

		plt.plot(x, fittedvalues, 'black', alpha=0.5)#, label='OLS')
	#plt.plot(x, predict_ci_low, 'b--')
	#plt.plot(x, predict_ci_upp, 'b--')
	#plt.plot(x, predict_mean_ci_low, 'g--')
	#plt.plot(x, predict_mean_ci_upp, 'g--')

	plt.scatter(x, y, color=colors[color_count], label='data', zorder=100)
	plt.errorbar(x, y, yerr = error, linestyle='None', color=colors[color_count], zorder=100)
	for (count, x, y) in zip(count, x, y):
		plt.annotate(str(count), xy = (x, y+0.07), xycoords='data', color = colors[color_count])

	plt.xlabel('Mass bins')
	plt.ylabel('fmerg')
	plt.title('$z = $'+str(zcen))
	#plt.legend()
	plt.annotate(str(round(np.mean(slope_list),2))+'+/-'+str(round(np.std(slope_list),2)), 
		xy=(0.01,0.9), xycoords='axes fraction', color=colors[color_count])
	if savefigs:
		plt.savefig('../Figures/MCMC_mass_'+str(run)+'_'+suffix+'_z_'+str(zcen)+'_color.png', dpi=1000)
	else:
		plt.show()

	color_count+=1

	continue
	
	
	

	adjusted_location = 0.03*(color_count - nbins/2 +0.5) + masscen#1.35*(zcen - 0.14) + masscen
	
	plt.scatter(adjusted_location[0], df_select['fmerg'].values[0], 
			color=colors[color_count], label=str(round(bins_z[color_count],3))+'$ < z < $'+str(round(bins_z[color_count+1],3)), marker='s',
			alpha = df_select['count'].values/max_count)
	
	plt.scatter(adjusted_location, df_select['fmerg'].values, 
		color=colors[color_count],  marker='s',
		alpha = df_select['count'].values/max_count)

	for c in range(len(df_select)):
		if df_select['count'].values[c] < 0.5* max_count:
			plt.errorbar(adjusted_location[c], df_select['fmerg'].values[c], yerr = df_select['fmerg_std'].values[c], 
				color=colors[color_count], linestyle='None', marker='s',
				alpha = df_select['count'].values[c]/max_count)
		else:
			plt.errorbar(adjusted_location[c], df_select['fmerg'].values[c], yerr = df_select['fmerg_std'].values[c], 
				color=colors[color_count], linestyle='None', marker='s')

	

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
if savefigs:
	plt.savefig('../Figures/2D_hist_equal_bins_z_mass_'+str(run)+'_'+suffix+'.png', dpi=1000)
else:
	plt.show()

plt.clf()
# Make the same plot but for redshift on the x axis:
color_count = 0
for masscen in centers_mass:
	print(masscen)
	print(str(round(bins_mass[color_count],3))+'$ < M < $'+str(round(bins_mass[color_count+1],3)))
	#color_count+=1
	#continue
	# Grab everything with a z value from the df
	df_select = df_fmerg[df_fmerg['mass'] == masscen].dropna().reset_index()
	print(df_select)
	if len(df_select)==0:
		color_count+=1
		continue
	# should already be sorted by mass
	count = df_select['count'].values
	print('total count', np.sum(count))
	try:
		zcen = df_select['z'].values
	except KeyError:
		zcen = df_select['z_x'].values


	
	# to do a linear regression, I'm going to need to define X and y
	x = zcen.reshape(-1,1)
	X = sm.add_constant(x)

	y = df_select['fmerg']
	error = df_select['fmerg_std']

	mu, sigma = 0, 1 # mean and standard deviation

	
	plt.clf()
	
	# iterate
	# save slope values
	slope_list = []
	for num in range(100):
		Y = [y[i] + error[i] * np.random.normal(mu, sigma, 1) for i in range(len(y))]

		#scaler = StandardScaler()
		#scaler.fit(X)
		#X_standardized = scaler.transform(X)


		res = sm.OLS(Y, X).fit()
		
		#plt.scatter(x, Y, s=0.1, color=colors[color_count])
		try:
			slope_list.append(res.params[1])
		except:
			slope_list.append(999)
		
		

		st, data, ss2 = summary_table(res, alpha=0.05)
		fittedvalues = data[:,2]
		predict_mean_se  = data[:,3]
		predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
		predict_ci_low, predict_ci_upp = data[:,6:8].T

		#print(summary_table(res, alpha=0.05))
		

		

		plt.plot(x, fittedvalues, 'black', alpha=0.5)#, label='OLS')
	#plt.plot(x, predict_ci_low, 'b--')
	#plt.plot(x, predict_ci_upp, 'b--')
	#plt.plot(x, predict_mean_ci_low, 'g--')
	#plt.plot(x, predict_mean_ci_upp, 'g--')

	plt.scatter(x, y, color=colors[color_count], label='data', zorder=100)
	plt.errorbar(x, y, yerr = error, linestyle='None', color=colors[color_count], zorder=100)
	for (count, x, y) in zip(count, x, y):
		plt.annotate(str(count), xy = (x, y+0.05), xycoords='data', color = colors[color_count])

	plt.xlabel('Redshift bins')
	plt.ylabel('fmerg')
	plt.title('M = '+str(masscen))
	#plt.legend()
	plt.annotate(str(round(np.mean(slope_list),2))+'+/-'+str(round(np.std(slope_list),2)), 
		xy=(0.01,0.9), xycoords='axes fraction', color=colors[color_count])
	if savefigs:
		plt.savefig('../Figures/MCMC_redshift_'+str(run)+'_'+suffix+'_M_'+str(masscen)+'_color.png', dpi=1000)
	else:
		plt.show()

	color_count+=1

	continue
	
	
	

	adjusted_location = 0.005*(color_count - nbins/2 +0.5) + zcen#1.35*(zcen - 0.14) + masscen
	
	plt.scatter(adjusted_location[0], df_select['fmerg'].values[0], 
			color=colors[color_count], label=str(round(bins_mass[color_count],3))+'$ < M < $'+str(round(bins_mass[color_count+1],3)), marker='s',
			alpha = df_select['count'].values/max_count)
	
	plt.scatter(adjusted_location, df_select['fmerg'].values, 
		color=colors[color_count],  marker='s',
		alpha = df_select['count'].values/max_count)

	for c in range(len(df_select)):
		if df_select['count'].values[c] < 0.5* max_count:
			plt.errorbar(adjusted_location[c], df_select['fmerg'].values[c], yerr = df_select['fmerg_std'].values[c], 
				color=colors[color_count], linestyle='None', marker='s',
				alpha = df_select['count'].values[c]/max_count)
		else:
			plt.errorbar(adjusted_location[c], df_select['fmerg'].values[c], yerr = df_select['fmerg_std'].values[c], 
				color=colors[color_count], linestyle='None', marker='s')

	

	#adjusted_location = [1.3*(zcen - 0.14) + x for x in centers_mass]
	#plt.scatter(adjusted_location, avgs, color=colors[color_count], label='$z = $'+str(zcen), marker='s')
	#plt.errorbar(adjusted_location, avgs, yerr = stds, color=colors[color_count], linestyle='None', marker='s')

	color_count+=1
for boundary in bins_z:
	plt.axvline(x = boundary, ls=':', color='black')

leg = plt.legend()
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xlabel(r'redshift')
plt.ylabel(r'$f_{\mathrm{merg}}$')
plt.ylim([0,1])
plt.xlim([bins_z[0]-0.01,bins_z[-1]+0.01])
if savefigs:
	plt.savefig('../Figures/2D_hist_equal_bins_z_'+str(run)+'_'+suffix+'.png', dpi=1000)
else:
	plt.show()




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

