from cropclass import tstrain
from cropclass import tsclust
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



# Clustered NDVI time-series data for cropped area; 5 clusters
clust4 = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/final_clusters/4_clusters.csv')
clust5 = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/final_clusters/5_clusters.csv')
clust6 = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/final_clusters/6_clusters.csv')

# Visualize cluster results (shows cluster mean and 10th, 90th percentile for each date in time-series)
tsclust.plot_clusters(clust4, fill=True, title='4 Clusters', save=True, filename='/Users/jameysmith/Documents/sentinel2_tanz/blog_images/4_clusters_fill.png')
tsclust.plot_clusters(clust4, fill=False, title='4 Clusters', save=True, filename='/Users/jameysmith/Documents/sentinel2_tanz/blog_images/4_clusters.png')

tsclust.plot_clusters(clust5, fill=True, title='5 Clusters', save=True, filename='/Users/jameysmith/Documents/sentinel2_tanz/blog_images/5_clusters_fill.png')
tsclust.plot_clusters(clust5, fill=False, title='5 Clusters', save=True, filename='/Users/jameysmith/Documents/sentinel2_tanz/blog_images/5_clusters.png')

tsclust.plot_clusters(clust6, fill=True, title='6 Clusters', save=True, filename='/Users/jameysmith/Documents/sentinel2_tanz/blog_images/6_clusters_fill.png')
tsclust.plot_clusters(clust6, fill=False, title='6 Clusters', save=True, filename='/Users/jameysmith/Documents/sentinel2_tanz/blog_images/6_clusters.png')

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
training_data_stand = tstrain.get_training_data(fp, bd, allsamples, standardize=True)
#training_data = tstrain.get_training_data(fp, bd, allsamples, standardize=False)


# STEP 3: Fit a LSTM recurrent neural network. In this 'toy' example, a total of 25,000 samples are used to fit a model.
#         including 10,000 from the clustered "cropped" class, and 5,000 from each of the "water", "urban" and
#         "vegetation" classes. The bands (features) include red, blue, green, and nir. Y labels are numerically
#         encoded, and converted to "one-hot" vectors.

# Format training data into correct 3D array of shape (n_samples, n_timesetps, n_features) required to fit a
# Keras LSTM model. N_features corresponds to number of bands included in training data
class_codes, x, y = tstrain.format_training_data(training_data_stand)

# Split training and test data
x_train, x_test, y_train, y_test = tstrain.split_train_test(x, y, seed=0)

# Train LSTM model
n_timesteps = len(training_data_stand['date'].unique())
n_features = len(training_data_stand['feature'].unique())

model = Sequential()
model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(32))
model.add(Dense(activation='softmax', units=y.shape[1]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=2)

# Model accuracy
_, accuracy = model.evaluate(x_test, y_test, batch_size=32)
accuracy

# Confusion matrix
tstrain.conf_mat(x_test, y_test, model, class_codes)
