from cropclass import tstrain
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# Clustered NDVI time-series data for cropped area; 5 clusters
clust5 = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/final_clusters/5_clusters.csv')

# Visualize cluster results (shows cluster mean and 10th, 90th percentile for each date in time-series)
#tsclust.plot_clusters(clust5, fill=True, title='5 Clusters', save=False)


# STEP 1: Combine samples from clustered, cropped area, and other land cover clusters into single dataset
#         for model fitting. `ndvi_lc` is file path to .csv files containing NDVI time-series' from vegetation,
#         urban, and water land cover classes.
ndvi_lc = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/lc_ndvi_interp'

# Take n_samples of each non-crop land cover class
noncrop_samples = tstrain.random_ts_samples(ndvi_lc, n_samples=5000, seed=0)

# Rename and drop columns to allow concatination of crop and non-crop samples
clust5 = clust5.rename(columns={'cluster': 'label'})
clust5 = clust5.drop(['lc'], axis=1)

# Combine datasets
dlist = [clust5, noncrop_samples]
allsamples = pd.concat(dlist, ignore_index=True)



# STEP 2: Using raster index locations from `allsamples` (Step 1), extract band reflectance values from a
#         time-series of scenes contained in a directory generated using the default sat-search directory structure

# Directory containing time-series of Sentinel-2 scenes corresponding to AOI
fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing'

# Band names are stripped from file-path names. `bd` provides a lookup table for Sentinel-2 band numbers
bd = {'B02': 'blue',
      'B03': 'green',
      'B04': 'red',
      'B08': 'nir'}

# Extract training data from Sentinel-2 time-series
#training_data_stand = tstrain.get_training_data(fp, bd, allsamples, standardize=True)
training_data = tstrain.get_training_data(fp, bd, allsamples, standardize=False)


# STEP 3: Fit a LSTM recurrent neural network. In this 'toy' example, a total of 25,000 samples are used to fit a model.
#         including 10,000 from the clustered "cropped" class, and 5,000 from each of the "water", "urban" and
#         "vegetation" classes. The bands (features) include red, blue, green, and nir. Y labels are numerically
#         encoded, and converted to "one-hot" vectors.

# Format training data into correct 3D array of shape (n_samples, n_timesetps, n_features) required to fit a
# Keras LSTM model. N_features corresponds to number of bands included in training data
class_codes, x, y = tstrain.format_training_data(training_data)

# Split training and test data
x_train, x_test, y_train, y_test = tstrain.split_train_test(x, y, seed=0)

# Standardize features
mu, sd, x_train_norm, x_test_norm = tstrain.standardize_features(x_train, x_test)


# Train LSTM model
n_timesteps = len(training_data['date'].unique())
n_features = len(training_data['feature'].unique())

model = Sequential()
model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(32))
model.add(Dense(activation='softmax', units=y.shape[1]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train_norm, y_train, epochs=50, batch_size=32, verbose=2)

# Model accuracy
_, accuracy = model.evaluate(x_test_norm, y_test, batch_size=32)
accuracy

# Model predictions on test set
predictions = model.predict(x_test_norm)
y_pred = (predictions > 0.5)

# Confusion matrix
tstrain.conf_mat(x_test_norm, y_test, model, class_codes)



# --------------------- #
# PREDICT ON ENTIRE IMAGE
# --------------------- #
import os
import gippy
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing'

# Filepath points to folder of geotiffs of Sentinel 2 time-series of bands 4 (red) and 8 (nir)
scenes = [fp + '/' + f for f in os.listdir(fp) if not f.startswith('.')]
scenes.sort()

allbands = []
for s in scenes:
    bands = [s + '/' + b for b in os.listdir(s) if not b.startswith('.')]
    allbands.append(bands)

# TODO: Confirm the problem is not the GAIN argument in gippy
all_vals = []
for date in allbands:

    img = gippy.GeoImage.open(filenames=date, bandnames=['ndvi', 'green', 'blue'], nodata=0, gain=0.0001)

    flat_list = []
    for b in img.bandnames():
        # Read a single band from a single date
        band_vals = img[b].read()

        # Flatten and standardize values
        flat_vals = band_vals.flatten()
        flat_vals = preprocessing.scale(flat_vals)

        flat_list.append(flat_vals)

    # A flattened array of all band values from a single scene
    band_flat = np.stack([arr for arr in flat_list]).T
    all_vals.append(band_flat)

av = np.stack([arr for arr in all_vals])

avs = av.swapaxes(1, 0)

pred = model.predict(avs)
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/predictions.npy', pred)
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