# This code is for measuring the masses of all galaxies using the empirical relation from
# Bell+2003
# Does not require that galaxies have masses from Mendel

from astropy.cosmology import FlatLambdaCDM 
import astropy.units as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import sem
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# What information do we need?
# 1. photometric redshift
# 2. g band mag
# 3. r band mag

# enter the path in where you save tables:
dir = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'

# This is how I merge a bunch of tables together to make the above table
# None of this is necessary if you have the above mass_comparisons.txt table saved

type_gal = 'predictors'
run = 'major_merger'
name = 'color_complete'
spacing_z = 0.02#'KB'#0.04
completeness = 95
red = 'z_spec'
mass = 'log_stellar_mass_from_color'
#red = 'z'
suffix = str(spacing_z)+'_'+str(red)+'_'+str(mass)+'_completeness_'+str(completeness)
plot_hist = False
plot = False

if os.path.exists(dir+'all_mass_color_complete_'+str(suffix)+'.txt'):
	STOP

if os.path.exists(dir+'all_mass_'+str(suffix)+'.txt'):
	# already have the mass table saved
	table = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_'+str(suffix)+'.txt',header=[0],sep='\t')
	print(table.columns)
	if plot:
		plt.clf()
		plt.scatter(table['z_x'].values, table['z_spec'].values, 
			c = table['g_minus_r'].values, s=0.1, vmax=2)
		plt.colorbar(label='g-r')
		plt.xlabel('photo z')
		plt.ylabel('spec z')
		plt.xlim([0,0.056])
		plt.ylim([0,0.056])
		plt.plot(np.linspace(0,0.4,10), np.linspace(0,0.4,10), color='black')
		plt.show()

		STOP

	

else:
	# Next, import the info you need to derive your own stellar masses:
	if red == 'z':
		zs = pd.io.parsers.read_csv(filepath_or_buffer=dir+'z_g_r_kcorrected_beckynevin_0.csv',sep=',', header=[0])
	# columns are: 'objID', 'dr7objid'
	if red == 'z_spec': # get these from Mendel
		zs = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_measurements.txt',header=[0],sep='\t')
		
		zs['z_spec'] = zs['z_y']


	# Cross-match with your sample:

	df_LDA = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run+'_flags.txt',header=[0],sep='\t')

	# Because the df_LDA doesn't have the final flag, use the predictor table to instead clean via merging

	# Run OLS with predictor values and z and stellar mass and p_merg:
	df_predictors = pd.io.parsers.read_csv(filepath_or_buffer=dir+'SDSS_predictors_all_flags_plus_segmap.txt',header=[0],sep='\t')

	if len(df_LDA) != len(df_predictors):
		print('these have different lengths cannot use one to flag')
		STOP

	# First clean this so that there's no segmap flags
	df_predictors_clean = df_predictors[(df_predictors['low S/N'] ==0) & (df_predictors['outlier predictor']==0) & (df_predictors['segmap']==0)]

	clean_LDA = df_LDA[df_LDA['ID'].isin(df_predictors_clean['ID'].values)]

	clean_LDA = clean_LDA[['ID','p_merg']]

	table = zs.merge(clean_LDA, left_on = 'objID', right_on = 'ID')

	# For this table, you have all the photometric info, info from mendel (to check), and LDA classifications
	print('length of final table', len(table))


	# Now, for every row in the table, go through and measure the stellar mass using the color-M/L ratio from Bell 2003:
	# The formula:
	# log10(M/L_lambda) = a_lambda + b_lambda * (color)
	# units are solar for M/L_lambda, which I think means you need the luminosity relative to the sun (so its a ratio)
	# color is in magnitudes?

	# for the g-r color row and the r-band filter from Table 7 of Bell+2003
	a_r = -0.306
	b_r = 1.097

	# the updated values from Du+2019:
	a_r = -0.61
	b_r =  1.19

	# from zibetti+2009
	a_r = -0.840
	b_r = 1.654

	# Luminosity = flux * d^2 because its measured at the outside of a sphere of light
	# so to get the luminosity in terms of solar, you will need to know the flux ratio relative to the sun
	# as well as the relative distance because:
	# L_gal / L_sun = (F_gal / F_sun) * (d_gal / d_sun)**2
	#               = 10**(-0.4(m_gal - m_sun)) * (d_gal / d_sun)**2

	sun_d = 149597870700 * u.m # distance in meters to the sun
	mag_sun_apparent = -27.05#-26.7 # is this the apparent mag in r_band?
	#mag_sun_apparent_r = -27.05
	mag_sun_absolute_AB = 4.67

	number = 0#100 # the number of galaxies you're running through the for loop
	if number != 0:
		table = table[0:number]

	log_stellar_mass_from_color = []
	log_stellar_mass_from_color_scott = []

	for i in range(len(table)):
		z = table[red].values[i] # this is the photometric redshift from the sdss photoZ catalog

		gal_d = cosmo.luminosity_distance(z).to(u.m) # distance to the galaxy in meters

		# I get an answer of 1.59e8 so within a factor of 2?
		#table['g_minus_r'].values[i]
		g_r = table['g'].values[i] - table['kcorrG'].values[i] - (table['r'].values[i] - table['kcorrR'].values[i])
		#print('compare g-r OG', table['g_minus_r'].values[i])
		#print('k corrected', g_r)
		mag_gal_r = table['r'].values[i] - table['kcorrR'].values[i]

		f_ratio = 10**(-0.4*(mag_gal_r - mag_sun_apparent))
		#print('flux ratio relative to sun', f_ratio)
		#print('distance ratio relative to sun', (gal_d/ sun_d)**2)

		# So if the galaxy distance is over-estimated, 
		# then L_ratio is over-estimated, which over-estimates mass
		L_ratio = f_ratio * (gal_d/ sun_d).value**2

		#print('L ratio',L_ratio)

		#print('log stellar mass',np.log10(10**(a_r + b_r*(g_r))*L_ratio))
		log_stellar_mass_from_color.append(np.log10(10**(a_r + b_r*(g_r))*L_ratio))

		#print('mendel mass', table['logMt'].values[i]) # this is the mass from mendel

		scott = a_r + b_r*(g_r) + 2. * np.log10(gal_d.to(u.pc).value) - 2.0 + 0.4*mag_sun_absolute_AB - 0.4*mag_gal_r
		#print('scott versus me')
		#print('log M/L', a_r + b_r*(g_r))
		#print(scott, np.log10(10**(a_r + b_r*(g_r))*L_ratio))
		log_stellar_mass_from_color_scott.append(scott)

		# method from scott:

		#log_M=logML+2.*log10(LD_pc)-2.0+0.4*M_Sun-0.4*m

		#where logML is logarithm of M/L, LD is the luminosity distance in pc, 
		#M_sun and m are the Sun's absolute AB magnitude and the apparent magnitude in whichever filter you are using, r i guess. Attached is a lookup table of Solar magnitudes that I find useful for this.

		# so it basically looks like
		#log M/L = color thing
		# M/L = 10**(color thing)
		# M = L*10**color
		# log M = log L + log 10**color
		# log M = log M/L + log L



		if i > number and number !=0:
			break


	table['log_stellar_mass_from_color'] = log_stellar_mass_from_color
	print('length of table bf dropping nans', len(table))
	table = table.dropna()
	print('length of table after dropping nans', len(table))



		


	# save this table
	#table['log_stellar_mass_from_color_scott'] = log_stellar_mass_from_color_scott
	print(table)

	table.to_csv(dir+'all_mass_'+str(suffix)+'.txt', sep='\t')


# Now do the completeness calculation from Darvish:
survey_limit = 17.77

table = table[(table[mass] < 13) & (table[mass] > 8)]

#table = table[(table[red] < 0.057) & (table[red] > 0.032)]
ranges_z = np.arange(0.0, 0.4, spacing_z)
centers_z= [round(x + spacing_z/2,2) for x in ranges_z[:-1]]


if spacing_z == 'KB':
	# This is the code to do KBdiscretizer bin spacing:

	from sklearn.preprocessing import KBinsDiscretizer
	strategy = 'kmeans'
	enc = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy=strategy)
	X = table[[red,mass]].values

	print('X', X)
	print('yys', X[:,1].min(), X[:,1].max())

	enc.fit(X)
	xx, yy = np.meshgrid(
	        np.linspace(X[:, 0].min(), X[:, 0].max(), 300),
	        np.linspace(X[:, 1].min(), X[:, 1].max(), 300),
	    )
	grid = np.c_[xx.ravel(), yy.ravel()]
	grid_encoded = enc.transform(grid)
	print(np.shape(grid_encoded), grid_encoded)

	'''
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# horizontal stripes
	horizontal = grid_encoded[:, 0].reshape(xx.shape)
	ax.contourf(xx, yy, horizontal, alpha=0.5)
	# vertical stripes
	vertical = grid_encoded[:, 1].reshape(xx.shape)

	# How do we get the edges?
	# So in the horizontal and the vertical directions do this:

	boundary_x = []
	boundary_x.append(0)
	for i in range(len(xx[0])-1):
		if horizontal[0][i+1] != horizontal[0][i]:
			# then this is a boundary:
			boundary_x.append((xx[0][i+1] - xx[0][i])/2 + xx[0][i])

	boundary_y = []
	for j in range(len(yy[0])-1):
		if vertical[0][j+1] != vertical[0][j]:
			# then this is a boundary:
			boundary_y.append((yy[0][j+1] - yy[0][j])/2 + yy[0][j])



	ax.contourf(xx, yy, vertical, alpha=0.5)

	#ax.scatter(X[:, 0], X[:, 1], edgecolors="k", s = 0.1)
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())

	ax.set_xlabel('z')
	ax.set_ylabel('log stellar mass')
	plt.show()




	ranges_z = boundary_x
	'''
if plot_hist:
	#ranges_z = np.arange(0.032, 0.057, 0.006)
	#centers_z= [round((ranges_z[i+1] - ranges_z[i])/2 + ranges_z[i],2) for i in range(len(ranges_z[:-1]))]
	print('ranges', ranges_z)



	_, bins = np.histogram(table[mass].values, bins=50)

	color_hist = ['#EEEBD0','#EBB3A9','#E87EA1','#E86252','#EE2677','#590004',
		'#EEEBD0','#EBB3A9','#E87EA1','#E86252','#EE2677','#590004',
		'#EEEBD0','#EBB3A9','#E87EA1','#E86252','#EE2677','#590004',
		'#EEEBD0','#EBB3A9','#E87EA1','#E86252','#EE2677','#590004',
		'#EEEBD0','#EBB3A9','#E87EA1','#E86252','#EE2677','#590004']
	color_hist = ['#3D348B','#7678ED','#F7B801','#F18701','#F35B04']
	# Make a histogram of the stellar masses at each redshift range:
	plt.clf()
M_lim_list = []

select_list = []

for i in range(len(ranges_z)-1):
	# First select only galaxies within a certain range of redshifts
	select = table[(table[red] > ranges_z[i]) & (table[red] < ranges_z[i+1])]

	if plot_hist:
		try:
			plt.hist(select[mass].values, bins=bins, color = color_hist[i], 
				label = f'{round(ranges_z[i],3)} < z < {round(ranges_z[i+1],3)}', histtype='step')#, density=True)
			#plt.hist(select['logBD'].values, bins=bins, color = color_hist[i], ls='--',
			#	label = f'Mendel masses, {round(ranges_z[i],3)} < z < {round(ranges_z[i+1],3)}', histtype='step')#, density=True)
		except IndexError:
			break

	log_mass_lim = select[mass] + 0.4 * (select['r'] - select['kcorrR'] - survey_limit)
		
	M_comp = np.percentile(log_mass_lim, completeness)#95)
	#log_mass_lim_BD = select['logBD'] + 0.4 * (select['r'] - select['kcorrR'] - survey_limit)
	#M_comp_BD = np.percentile(log_mass_lim_BD, completeness)#95)
	if plot_hist:
		plt.axvline(x = M_comp, color = color_hist[i])
		#plt.axvline(x = M_comp_BD, color = color_hist[i], ls='--')

		continue


	if plot:
		plt.clf()
		plt.scatter(select[mass].values, log_mass_lim, s=1, color='#3E8989')
		plt.scatter(log_mass_lim, log_mass_lim, s=1, color='#1A181B')
		
		plt.xlabel(mass)
		plt.ylabel('M_lim')
		plt.annotate(str(M_comp),xy=(0.02,0.95), xycoords='axes fraction', color='#564D65')
		plt.axhline(y = M_comp, color='#564D65')
		plt.axvline(x = M_comp, color='#564D65')
		plt.xlim([6,12])
		plt.ylim([6,12])
		plt.show()
	M_lim_list.append(M_comp)

	# Now, select everything with masses above the limit:
	select_list.append(select[select[mass] > M_comp])
if plot_hist:
	plt.legend()	
	plt.xlabel(mass)
	plt.title(f'Mass completeness limits for {completeness}% completeness')
	plt.show()
	STOP	

table_above_limit = pd.concat(select_list)

# Plot the entire thing yo

plt.clf()
#plt.scatter(table['z_x'].values, 
#	table['log_stellar_mass_from_color'].values, s=0.1, color='#D33E43')


nbins = 100
H, xedges, yedges = np.histogram2d(table[red].values, 
	table[mass].values,bins=nbins)
 
# H needs to be rotated and flipped
H = np.rot90(H)
H = np.flipud(H)
 
# Mask zeros
Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
 
# Plot 2D histogram using pcolor

plt.pcolormesh(xedges,yedges,Hmasked)

print('m_lim_list', M_lim_list)
plt.scatter(centers_z, M_lim_list, s=10.0, color='#F87060')
plt.ylim([7,13.5])
plt.xlim([0,0.4])
plt.plot(centers_z, M_lim_list, color='#F87060')
plt.show()

plt.clf()
#plt.scatter(table['z_x'].values, 
#	table['log_stellar_mass_from_color'].values, s=0.1, color='#D33E43')
nbins = 100
H, xedges, yedges = np.histogram2d(table_above_limit[red].values, 
	table_above_limit[mass].values,bins=nbins)
 
# H needs to be rotated and flipped
H = np.rot90(H)
H = np.flipud(H)
 
# Mask zeros
Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
 
# Plot 2D histogram using pcolor

plt.pcolormesh(xedges,yedges,Hmasked)

plt.scatter(centers_z, M_lim_list, s=10.0, color='#F87060')
plt.ylim([7,13.5])
plt.xlim([0,0.4])
plt.plot(centers_z, M_lim_list, color='#F87060')
plt.show()



table_above_limit.to_csv(dir+'all_mass_color_complete_'+str(suffix)+'.txt', sep='\t')

print('final length', len(table_above_limit))

# What about also saving the redshift bins?
np.save(dir+'all_mass_color_complete_'+str(suffix)+'_bins.npy', ranges_z)

