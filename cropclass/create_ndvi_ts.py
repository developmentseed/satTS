from cropclass import tsmask
from cropclass import tsclust
from archive import ndvicalc
from os import listdir
import re
import matplotlib.pyplot as plt
import numpy as np


### Step 1: rasterize land cover shapefile
fp = '/Users/jameysmith/Documents/sentinel2_tanz'

# landcover shapefiles to be rasterized
shp = fp + '/land_cover/lc_polygons.shp'

# output file (rasterized land cover)
outimg = fp + '/lcrast/lcrast_v2.tif'

# Reference image (Sentinel-2 tile)
refimg = fp + '/aoiTS/2016-11-16_B04.jp2'

# Rasterize the shapefile
tsmask.rasterize(shapefile=shp, outimg=outimg, refimg=refimg, attribute='id')

# Confirm land cover classes were burned into raster correctly
tsmask.check_rasterize(fp + '/lcrast/lcrast_v2.tif')



### Step 2: Calculate NDVI bands for image time-series ###
fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs'
ndvicalc.calulate_ndvi(fp)



### Step 3: Create a masked time-series of the NDVI bands for each land cover class ###
# land cover classes in 'lcrast.tif': 1 = water, 2 = veg, 3 = cropped, 4 = urban

fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi'

# List of file paths to NDVI time-series
files = [fp + '/' + f for f in listdir(fp)]
files.sort()
del files[0] #.DS_Store file

# Grab dates from NDVI file names
dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in files]
dates = [date for sublist in dates for date in sublist]

# Land cover mask
mask = '/Users/jameysmith/Documents/sentinel2_tanz/lcrast/lcrast.tif'

# Generate masked time-series
water_mask = tsmask.mask_to_array(files, dates, mask, 1)

veg_mask = tsmask.mask_to_array(files, dates, mask, 2)

crop_mask = tsmask.mask_to_array(files, dates, mask, 3)

urban_mask = tsmask.mask_to_array(files, dates, mask, 4)



### Step 4: Create a masked time-series of the NDVI bands for 'crop' land cover class ###
cropts = tsmask.BandTimeSeries(mask=crop_mask, lc_class='crop', ts_var='ndvi', dates=dates)
cropdf = cropts.time_series_dataframe(frequency='5d')



### Step 5: Perform time-series clustering with grid-search of hyperparameters
pg = {
    'time_seriesdf': [cropdf],
    'n_samples': [10],
    'cluster_alg': ['GAKM', 'TSKM'],
    'n_clusters': list(range(2, 4)),
    'smooth': [True],
    'ts_var': ['ndvi'],
    'window': [7],
    'poly': [3],
    'cluster_metric': ['dtw', 'softdtw'],
    'score': [True]
}

# Grid search on crop land cover class
pg_dict, pg_df = tsclust.cluster_grid_search(pg)

# Get cluster dataframe corresponding to parameter combination with largest silhouette score
x = pg_dict['clusters'][pg_df['sil_score'].idxmax()]

# Get cluster means and 10th, 90th quantiles from lowest silhouette score combination
m, q = tsclust.cluster_mean_quantiles(x)

# Plot cluster results
nclusts = len(x.cluster.unique())
color = iter(plt.cm.Set2(np.linspace(0, 1, nclusts)))

fig = plt.figure(figsize=(10, 8))
cnt = 0
for i in range(0, nclusts):
    c = next(color)
    plt.plot(m.index, m[i], 'k', color=c)
    plt.fill_between(m.index, q.iloc[:, [cnt]].values.flatten(), q.iloc[:, [cnt+1]].values.flatten(),
                     alpha=0.5, edgecolor=c, facecolor=c)
    cnt += 2
