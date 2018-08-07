from cropclass import tstrain
from cropclass import tspredict
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches


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
#model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add(Dense(activation='softmax', units=y.shape[1]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train_norm, y_train, epochs=50, batch_size=32, verbose=2)

# Model accuracy
_, accuracy = model.evaluate(x_test_norm, y_test, batch_size=32)
accuracy

# # Model predictions on test set
# predictions = model.predict(x_test_norm)
# y_pred = (predictions > 0.5)

# Confusion matrix
tstrain.conf_mat(x_test_norm, y_test, model, class_codes)


# STEP 4: Predict on full scene using trained model
fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing'

# Predict to full extent of the scene
predicted_scene = tspredict.predict_scene(file_path=fp, model=model, mu=mu, sd=sd)

# Prediction layer is last layer in the geoimage object returned from predict_scene()
predict_rast = predicted_scene[predicted_scene.nbands() - 1].read()
plt.imshow(predict_rast)



import numpy as np
import os
import gippy


file_path = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing'
scenes = [file_path + '/' + f for f in os.listdir(file_path) if not f.startswith('.')]
scenes.sort()

all_dates = []
for s in scenes:
    bands = [s + '/' + b for b in os.listdir(s) if not b.startswith('.')]
    bands.sort()
    all_dates.append(bands)

# Get dimensions for the final 3D input array for Keras model
get_shape = gippy.GeoImage.open(filenames=[all_dates[0][0]])

n_samples = get_shape.xsize() * get_shape.ysize()

n_timesteps = len(scenes)

n_features = len(all_dates[0])

get_shape = None


full_scene = np.empty([n_samples, n_timesteps, n_features])
for date in range(0, len(all_dates)):
    geoimg = gippy.GeoImage.open(filenames=all_dates[date], nodata=0, gain=0.0001)

    scene_vals = np.empty([n_samples, n_features])
    for i in range(0, geoimg.nbands()):
        arr = geoimg[i].read()
        flat = arr.flatten()
        scene_vals[:, i] = flat

    geoimg = None

    full_scene[:, date, :] = scene_vals

full_norm = (full_scene - mu) / sd

preds = model.predict(full_scene)




geoimg = gippy.GeoImage.open(filenames=scene_files, nodata=0, gain=0.0001)
#geoimg.add_band(geoimg[0])

flat_list = []
for band in geoimg.bandnames()[0:len(geoimg.bandnames()) - 1]:
    arr = geoimg[band].read()
    flat_band = arr.flatten()
    flat_list.append(flat_band)

date_vals = []
for group in chunker(flat_list, n_features):
    x = np.stack([arr for arr in group]).T
    date_vals.append(x)

all_vals = np.stack([arr for arr in date_vals])
av_keras = all_vals.swapaxes(0, 1)

av_norm = (av_keras - mu) / sd

# Predict on chunk
preds = model.predict(full_chunk)
pred_bool = (preds > 0.5)
pred_class = pred_bool.argmax(axis=1)

# Convert predictions to matrix with shape shape as chunk
pred_mat = pred_class.reshape(arr.shape[0], arr.shape[1])



# TODO: store chunk matrix dimensions outside of loop
for ch in geoimg.chunks(numchunks=100):
    flat_list = []
    for band in geoimg.bandnames()[0:len(geoimg.bandnames()) - 1]:
        arr = geoimg[band].read(chunk=ch)
        flat_band = arr.flatten()
        flat_list.append(flat_band)

    chunk_vals = []
    for group in chunker(flat_list, n_timesteps):
        x = np.stack([arr for arr in group]).T
        chunk_vals.append(x)

    # All band values for each date in chunk; 3D array with shape (n_samples, n_timesteps, n_features)
    full_chunk = np.stack([arr for arr in chunk_vals])
    full_chunk = full_chunk.swapaxes(0, 1)

    # Standardize features using mu and sd from training data
    full_chunk = (full_chunk - mu) / sd

    # Predict on chunk
    preds = model.predict(full_chunk)
    pred_bool = (preds > 0.5)
    pred_class = pred_bool.argmax(axis=1)

    # Convert predictions to matrix with shape shape as chunk
    pred_mat = pred_class.reshape(arr.shape[0], arr.shape[1])

    geoimg[geoimg.nbands() - 1].write(pred_mat, chunk=ch)








import numpy as np
unique, counts = np.unique(test, return_counts=True)
print(np.asarray((unique, counts)).T)

classes = np.unique(pred_mat)

labels = {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4',
          4: 'Cluster 5', 5: 'Urban', 6: 'Vegetation', 7: 'Water'}

cols = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a', 3: '#984ea3',
        4: '#ff7f00', 5: '#ffff33', 6: '#a65628', 7: '#f781bf'}

cmap = colors.ListedColormap(['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                              '#ff7f00', '#ffff33', '#a65628', '#f781bf'])

fig = plt.figure(figsize=(10, 8))
im = plt.imshow(pred_mat, cmap=cmap)
patches = [mpatches.Patch(color=cols[i], label=labels[i]) for i in cols]
plt.legend(handles=patches, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.)
