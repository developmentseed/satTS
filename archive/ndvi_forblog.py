
import gippy
import re
from os import listdir
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd

fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi'

# List of file paths to NDVI time-series
files = [fp + '/' + f for f in listdir(fp)]
del files[0]

# Grab dates from NDVI file names
dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in files]
dates = set(list(chain.from_iterable(dates)))
dates = list(dates)

# Feb. 2, 2017
t1 = gippy.GeoImage.open(filenames=[files[1]], nodata=0, gain=.0001)

ndvi_t1 = t1.read()
ndvi_t1[ndvi_t1 == -32768] = np.nan

t1 = None

# Jul. 4, 2017
t2 = gippy.GeoImage.open(filenames=[files[9]], nodata=0, gain=.0001)

ndvi_t2 = t2.read()
ndvi_t2[ndvi_t2 == -32768] = np.nan

t2 = None



# Plot 2 NDVI scense, side by side
min = -0.32
max = 0.9

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))

im1 = axes[0].imshow(ndvi_t1, vmin=min, vmax=max, aspect='equal')
axes[0].axis('off')
axes[0].set_title("NDVI: February 2, 2017")

im2 = axes[1].imshow(ndvi_t2, vmin=min, vmax=max, aspect='equal')
axes[1].axis('off')
axes[1].set_title("NDVI: July 4, 2017")

divider1 = make_axes_locatable(axes[0])
cax1 = divider1.append_axes("right", size="5%", pad=0.15)

divider2 = make_axes_locatable(axes[1])
cax2 = divider2.append_axes("right", size="5%", pad=0.15)

plt.colorbar(im1, cax=cax1)
plt.colorbar(im2, cax=cax2)

fig.savefig('/Users/jameysmith/Documents/sentinel2_tanz/blog_images/ndviscense.png', bbox_incehs='tight')




# Some non-smoothed examples
crop = ndvi[ndvi['lc'] == 'crop']
cropsub = crop[(crop['pixel'] > 99) & (crop['pixel'] < 105)]
cropsub.set_index('date', drop=True, inplace=True)
cropsub.index = pd.to_datetime(cropsub.index)
cropsub = cropsub.reindex(cropsub.index.rename('Date'))

# Same curves with smoothing applied
csmth = ndvi_smooth[ndvi_smooth['lc'] == 'crop']
smthsub = csmth[(csmth['pixel'] > 99) & (csmth['pixel'] < 105)]
smthsub.set_index('date', drop=True, inplace=True)
smthsub.index = pd.to_datetime(smthsub.index)
smthsub = smthsub.reindex(smthsub.index.rename('Date'))


fig, ax = plt.subplots(figsize=(16,8), nrows=1, ncols=2)
for key, grp in cropsub.groupby(['pixel']):
    grp.plot(ax=ax[0], kind='line', y='ndvi', label=key,
             legend=None)
ax[0].set_title('5-day NDVI time-series')

for key, grp in smthsub.groupby(['pixel']):
    grp.plot(ax=ax[1], kind='line', y='ndvi', label=key,
             legend=None)
ax[1].set_title('5-day NDVI time-series, with Savitzky-Golay filter')

plt.tight_layout(pad=4)
fig.savefig('/Users/jameysmith/Documents/sentinel2_tanz/blog_images/ndvi_ts.png')


### BLOG POST 2 ###
import gippy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

m = '/Users/jameysmith/Documents/sentinel2_tanz/lcrast/lcrast.tif'
n = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi/2017-02-14_ndvi.tif'

ndvi = gippy.GeoImage.open(filenames=[n], bandnames=['ndvi'], nodata=0)
ndvi_vals = ndvi.read()

img = gippy.GeoImage.open(filenames=[m], bandnames=['lc'])
mask = img.read()

vals = np.unique(mask)

labs = ['No Data', 'Water', 'Vegetation', 'Cropped', 'Urban']

fig = plt.figure(figsize=(10, 8))
im = plt.imshow(mask)
colors = [im.cmap(im.norm(val)) for val in vals]
patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=labs[i])) for i in range(len(vals))]
plt.legend(handles=patches, loc='upper right', borderaxespad=0.)
plt.title('Rasterized Land Cover Polygons')
fig.savefig('/Users/jameysmith/Documents/sentinel2_tanz/blog_images/rasterized_lc')

fig = plt.figure(figsize=(10,8))
plt.imshow(ndvivals)
plt.title('NDVI February 14, 2017')
fig.savefig('/Users/jameysmith/Documents/sentinel2_tanz/blog_images/ndvi_2-14-17')

import pandas as pd
res = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/gridsearch_results.csv')
res = res.drop(columns=['score'])