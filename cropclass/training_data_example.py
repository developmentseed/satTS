from cropclass import tstrain
from cropclass import tsclust
from archive import ndvicalc
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



# Clustered NDVI time-series data for cropped area; 5 clusters
clust5 = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/final_clusters/5_clusters.csv')

# Visualize cluster results (shows cluster mean and 10th, 90th percentile for each date in time-series)
tsclust.plot_clusters(clust5, fill=True)



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


## ADD INDEX BANDS
filepath = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing'

asset_dict = {'B02': 'blue',
              'B03': 'green',
              'B04': 'red',
              'B08': 'nir'}

indices = ['ndvi', 'evi']

ndvicalc.calulate_indices(filepath, asset_dict, indices)


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
training_data = tstrain.get_training_data(fp, bd, allsamples, scale=False)
training_data_scaled = tstrain.get_training_data(fp, bd, allsamples, scale=True)

training_data = training_data[training_data['feature'] != 'evi']
# STEP 3: Fit a LSTM recurrent neural network. In this 'toy' example, a total of 25,000 samples are used to fit a model.
#         including 10,000 from the clustered "cropped" class, and 5,000 from each of the "water", "urban" and
#         "vegetation" classes. The bands (features) include red, blue, green, and nir. Y labels are numerically
#         encoded, and converted to "one-hot" vectors.

# Format training data into correct 3D array of shape (n_samples, n_timesetps, n_features) required to fit a
# Keras LSTM model. N_features corresponds to number of bands included in training data
class_codes, x, y = tstrain.format_training_data(training_data, shuffle=True, seed=0)

# Split training and test data
x_train, x_test = x[0:int(x.shape[0]*0.8)], x[int(x.shape[0]*0.8):len(x)]
y_train, y_test = y[0:int(y.shape[0]*0.8)], y[int(y.shape[0]*0.8):len(y)]

# Train LSTM model
n_timesteps = len(training_data['date'].unique())
n_features = len(training_data['feature'].unique())

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dense(activation='softmax', units=y.shape[1]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=2)
_, accuracy = model.evaluate(x_test, y_test, batch_size=32)

# Model accuracy
accuracy
