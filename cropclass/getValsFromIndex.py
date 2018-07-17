import pandas as pd
import ast
import gippy
from cropclass import tsclust
from os import listdir
import re
import numpy as np
import pickle
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
import matplotlib.pyplot as plt

# TODO: Generalize this to a function that can grap values within a 3d raster stack
# A cluster results dataframe with pixels assigned to a cluster
df = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/best_clusters.csv',
                 converters={"array_index": ast.literal_eval})

# Un-smoothed
pick = '/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/full_results.p'
all_results = pickle.load(open(pick, "rb"))
clustscores = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/gridsearch_results.csv')

# Smoothed
clustsmooth = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/best_clusters.csv')
pick_smooth = '/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/full_results.p'
smooth_results = pickle.load(open(pick_smooth, "rb"))
smooth_summary = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/gridsearch_results.csv')

tsclust.plot_clusters(smooth_results, 24, fill=True)



test = smooth_results['clusters'][24]
test.cluster.value_counts()

# A random ndvi Sentinel-2 scene
#fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/extract_test'
# NDVI folder
fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi'
files = [fp + '/' + f for f in listdir(fp)]
files.sort()
del files[0]

dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in files]
dates = [date for sublist in dates for date in sublist]

# For reshaping dataset to fit model
n_timesteps = len(set(dates))
n_features = 1 # Only Red and NIR bands in this example
n_samples = len(test.pixel.unique())

#names = ['red_01', 'nir_01', 'ndvi_01', 'red_02', 'nir_02', 'ndvi_02']

# Open images in file path
ts = gippy.GeoImage.open(filenames=files, bandnames=dates, nodata=0, gain=.0001)

# Read images to np.arrays
bandvals = ts.read()

# Grab array indices from cluster operation
x = list(test.array_index)
x = [elem.strip('()') for elem in x]
x = [elem.split(',') for elem in x]
x = [list(map(int, elem)) for elem in x]
indices = np.array([*x])

## KERAS WANTS INPUT SHAPE: [n_samples, n_timesteps, n_features]
## This example: 10 samples, 2 timesteps, 2 features at each timestep
x_vals = bandvals[:, indices[:, 0], indices[:, 1]].T

# Center and
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x_vals)

# Reshape to 3D array for LSTM input
X = x_scaled.reshape((n_samples, n_timesteps, n_features))

# Class labels for training data
y_labels = test.cluster

# Convert labels to one-hot encoding
y_onehot = to_categorical(y_labels, num_classes=len(y_labels.unique()))

# Training and Test sets
x_train, x_test = X[0:int(X.shape[0]*0.8)], X[int(X.shape[0]*0.8):len(X)]
y_train, y_test = y_onehot[0:int(y_onehot.shape[0]*0.8)], y_onehot[int(y_onehot.shape[0]*0.8):len(y_onehot)]

# Train an LSTM model
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dense(activation='softmax', units=y_onehot.shape[1]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=32)
score

