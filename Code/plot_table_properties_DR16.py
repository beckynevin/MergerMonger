#~~~~~~~
# Plots the distribution of SDSS galaxies you just saved
# in the table in ra and dec and makes a histogram of r-band
# magnitudes.
#~~~~~~~


import numpy as np
import astropy.io.fits as pyfits
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
import scipy
from astroML.stats import binned_statistic_2d

prefix = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger/Tables/'

# This is what was given in the beginning
file_path = prefix+'five_sigma_detection_saturated_mode1_beckynevin.csv'
f = open(file_path, 'r+')
data=f.readlines()[1:]

sdss = []
ra = []
dec = []
r_band = []
for m in range(len(data)):
    line = data[m].split(',')
    

    sdss.append(int(line[0]))
    ra.append(float(line[1]))
    dec.append(float(line[2]))
    r_band.append(float(line[3]))
    
print('overall length of sdss', len(sdss))




# Okay so plot sky coverage of full sdss photometric sample from DR16:


# First, get a polar plot
'''
plt.clf()
fig = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(1,1)

ax = plt.subplot(gs[0], polar=True)
# convert ra to radians:
ra_rad = [x / 180.0*np.pi for x in ra]

im = ax.scatter(ra_rad, dec, c=r_band, s=0.1, vmin=-20)
plt.xlabel('RA')
plt.ylabel('Dec')
plt.colorbar(im, label='$r-$band mag')
plt.savefig('../LDA_figures/sample_distribution_DR16_polar.png')

plt.clf()
fig = plt.figure(figsize=(10,5))
plt.scatter(ra, dec, c=r_band, s=0.1, vmin=-20)
plt.xlabel('RA')#, labelpad=10)
plt.ylabel('Dec', labelpad=10)
plt.colorbar(label='$r-$band mag')
plt.savefig('../Figures/sample_distribution_DR16.png')
'''



# Okay but how many of these have I actually run is the real question



type_gal = 'predictors'
#f = pd.read_csv(prefix+'SDSS_'+str(type_gal)+'_all.txt', sep='\t')

f = open(prefix+'SDSS_'+str(type_gal)+'_all.txt', 'r+')
data=f.readlines()[1:]

sdss_cannon = []

for m in range(len(data)):
    line = data[m].split('\t')
    

    sdss_cannon.append(int(line[1]))
    

    
#
print('IDs', sdss[0:10])
print('sdss preds', sdss_cannon[0:10])
N=np.where(np.isin(sdss,sdss_cannon))[0]

print('len run', len(N))
'''
print(N)

ys = [0 for x in N]

plt.clf()
plt.scatter(np.array(sdss)[N], ys)
plt.savefig('../LDA_figures/selected.png')
'''


ptsize = 0.5
ra_rad = [x / 180.0*np.pi for x in ra]
sns.set_context("talk")


plt.clf()


fig = plt.figure(figsize=(14,10))
gs = gridspec.GridSpec(1,1)

ax = plt.subplot(gs[0], polar=True)
# convert ra to radians:

#im0 = ax.scatter(np.array(ra_rad), np.array(dec), c='grey', s=ptsize, edgecolor='None')

im = ax.scatter(np.array(ra_rad)[N], np.array(dec)[N], c=np.array(r_band)[N], s=ptsize, vmin=14, cmap='viridis_r', edgecolor='None')#, norm=matplotlib.colors.LogNorm())
#plt.xlabel('RA')
#plt.ylabel('Dec', labelpad=40)
plt.colorbar(im, label='Extinction-corrected $r-$band mag', fraction=0.043)
plt.annotate('Number in SDSS = '+str(len(sdss))+', Number measured = '+str(len(N)), xy=(-0.1, 1.1), xycoords='axes fraction')
#plt.annotate(', xy=(0.01, 1.0), xycoords='axes fraction')
plt.savefig('../Figures/sample_distribution_DR16_polar_overlay.png', dpi=1000)


# Try to make a better plot of this
# Perhaps a 2D histogram type situation

plt.clf()


fig = plt.figure(figsize=(14,10))
gs = gridspec.GridSpec(1,1)

ax = plt.subplot(gs[0])#, polar=True)
# convert ra to radians:
stat, xedges, yedges = binned_statistic_2d(np.array(ra_rad)[N],np.array(dec)[N],
    np.array(r_band)[N], 'mean', bins=100)
#im0 = ax.scatter(np.array(ra_rad), np.array(dec), c='grey', s=ptsize, edgecolor='None')

#im = ax.scatter(np.array(ra_rad)[N], np.array(dec)[N], c=np.array(r_band)[N], s=ptsize, vmin=14, cmap='magma_r', edgecolor='None', norm=matplotlib.colors.LogNorm())
im = ax.imshow(stat, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],aspect='auto', cmap='viridis_r')
#plt.xlabel('RA')
#plt.ylabel('Dec', labelpad=40)
plt.colorbar(im, label='Extinction-corrected $r-$band mag', fraction=0.043)
plt.annotate('Number in SDSS = '+str(len(sdss))+', Number measured = '+str(len(N)), xy=(-0.1, 1.1), xycoords='axes fraction')
#plt.annotate(', xy=(0.01, 1.0), xycoords='axes fraction')
plt.savefig('../Figures/sample_distribution_DR16_polar_2dstat.png', dpi=1000)



# Maybe also make a histogram of this whole situation

counts, bin_edges = np.histogram(r_band,bins=30,range=[10,18])
print('overall sample', bin_edges, counts)
counts_subset, bins = np.histogram(np.array(r_band)[N], bins=bin_edges)
print(bins, counts_subset)

plt.clf()
fig = plt.figure(figsize=(10,4))



plt.bar(bin_edges[:-1], counts, color='grey', alpha=0.5, label='SDSS Photometric Sample',linewidth=0, width=bin_edges[1]-bin_edges[0])


#plt.hist(counts, bins=bins, color='purple', alpha=0.5)
plt.bar(bin_edges[:-1], counts_subset, color='purple', alpha=0.5, label='Subsample', linewidth=0, width=bin_edges[1]-bin_edges[0])

plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Extinction-corrected $r-$band magnitude')

#plt.yscale('log')

plt.tight_layout()
#plt.axvline(x=17.77, ls='--', color='black')
plt.savefig('../Figures/sample_distribution_histogram_r_band_DR16.png', dpi=1000)


