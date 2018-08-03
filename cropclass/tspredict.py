import os
import gippy
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from itertools import chain


fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing'

# Filepath points to folder of geotiffs of Sentinel 2 time-series of bands 4 (red) and 8 (nir)
scenes = [fp + '/' + f for f in os.listdir(fp) if not f.startswith('.')]
scenes.sort()

allbands = []
for s in scenes:
    bands = [s + '/' + b for b in os.listdir(s) if not b.startswith('.')]
    allbands.append(bands)

all_vals = []
for date in allbands:

    img = gippy.GeoImage.open(filenames=date, bandnames=['ndvi', 'green', 'blue'], nodata=0, gain=0.0001)

    flat_list = []
    for b in img.bandnames():
        # Read a single band from a single date
        band_vals = img[b].read()

        # Flatten
        flat_vals = band_vals.flatten()
        flat_list.append(flat_vals)

    # A flattened array of all band values from a single scene
    band_flat = np.stack([arr for arr in flat_list]).T
    all_vals.append(band_flat)

# All data from all dates/bands
av = np.stack([arr for arr in all_vals])

# Reshape to (num_samples, num_timesteps, num_features)
avs = av.swapaxes(1, 0)


# With Chunking
scene_files = list(chain.from_iterable(allbands))
bandnames=['ndvi', 'green', 'blue']

geoimg = gippy.GeoImage.open(filenames=scene_files, nodata=0, gain=0.0001)

for ch in geoimg.chunks(numchunks=100):
    flat_list = []
    for band in geoimg.bandnames():
        arr = geoimg[band].read(chunk=ch)
        flat_band = arr.flatten()
        flat_list.append(flat_band)






# Make predictions
pred = model.predict(avs)
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/predictions.npy', pred)
pred = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/predictions.npy')
pred_bool = (pred > 0.5)

pred_class = pred_bool.argmax(axis=1)

# NEED TO STORE band_vals.shape OUTSIDE OF LOOP
pred_mat = pred_class.reshape(-1, band_vals.shape[1])

unique, counts = np.unique(pred_mat, return_counts=True)
print(np.asarray((unique, counts)).T)

classes = np.unique(pred_mat)
labs = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Urban', 'Vegetation', 'Water']

labels = {0: 'Cluster 1',
          1: 'Cluster 2',
          2: 'Cluster 3',
          3: 'Cluster 4',
          4: 'Cluster 5',
          5: 'Urban',
          6: 'Vegetation',
          7: 'Water'}

cols = {0: '#e41a1c',
        1: '#377eb8',
        2: '#4daf4a',
        3: '#984ea3',
        4: '#ff7f00',
        5: '#ffff33',
        6: '#a65628',
        7: '#f781bf'}

cmap = colors.ListedColormap(['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                              '#ff7f00', '#ffff33', '#a65628', '#f781bf'])

fig = plt.figure(figsize=(10, 8))
im = plt.imshow(pred_mat, cmap=cmap)
patches = [mpatches.Patch(color=cols[i], label=labels[i]) for i in cols]
plt.legend(handles=patches, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.)


#x_test[40] = water
#pred_class[11233123] = water

f = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing/2016-11-16/S2A_OPER_MSI_L1C_TL_SGS__20161116T132549_A007325_T36MUS_N02.04_B02.tif'
img = gippy.GeoImage.open(filenames=[f], bandnames=['blue'], nodata=0)
v = img.read()