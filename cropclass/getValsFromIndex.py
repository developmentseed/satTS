import pandas as pd
import ast
import gippy
from cropclass import tsclust
import sys
import os
import re
import numpy as np
import pickle
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import random


# Final clusters (10k samples)
clust4 = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/final_clusters/4_clusters.csv')
clust4.cluster.value_counts()
tsclust.plot_clusters(clust4, fill=True)


def random_ts_samples(file_path, n_samples, seed=None):

    # CSV files containing time-series' to be samples
    ts_files = [file_path + '/' + file for file in os.listdir(file_path)]

    np.random.seed(seed)

    # Sample each dataframe corresponding to a land cover class, store in list
    dfs = []
    for file in ts_files:
        df = pd.read_csv(file)
        g = df.groupby('pixel')
        a = np.arange(g.ngroups)
        np.random.shuffle(a)
        s = df[g.ngroup().isin(a[:n_samples])]

        dfs.append(s)

    # Convert list to single sample dataframe, convert to same shape as cluster results
    lc_samples = pd.concat(dfs)
    lc_samples = lc_samples.rename(columns={'lc': 'label', 'array_ind': 'array_index'})
    lc_samples = lc_samples.pivot_table(index=['array_index', 'label', 'pixel'], columns='date', values='ndvi').reset_index()

    return lc_samples


ndvi_lc = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/lc_ndvi_interp'

test = random_ts_samples(ndvi_lc, 5, seed=0)

clust5 = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/clustering/cluster_results/smooth_results/final_clusters/5_clusters.csv')
clust5 = clust5.rename(columns={'cluster': 'label'})
clust5 = clust5.drop(['lc'], axis=1)

dlist = [clust5, test]
allsamples = pd.concat(dlist)


def get_training_data(asset_dir, asset_dict, samples_df):
    # Array indices corresponding to sample locations in
    ind = list(samples_df.array_index)
    ind = [elem.strip('()').split(',') for elem in ind]
    ind = [list(map(int, elem)) for elem in ind]
    sample_ind = np.array([*ind])

    # Class labels
    labels = samples_df.label

    # Full file-path for every asset in `fp` (directory structure = default output of sat-search)
    file_paths = []
    for path, subdirs, files in os.walk(asset_dir):
        for name in files:
            # Address .DS_Store file issue
            if not name.startswith('.'):
                file_paths.append(os.path.join(path, name))

    # Scene dates
    dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in file_paths]
    dates = [date for sublist in dates for date in sublist]

    # Asset (band) names
    pattern = '[^_.]+(?=\.[^_.]*$)'
    bands = [re.search(pattern, f).group(0) for f in file_paths]

    # Match band names
    bands = [asset_dict.get(band, band) for band in bands]

    samples_list = []
    for i in range(0, len(file_paths)):

        img = gippy.GeoImage.open(filenames=[file_paths[i]], bandnames=[bands[i]], nodata=0, gain=0.0001)
        bandvals = img.read()

        # Extract values at sample indices for band[i] in time-step[i]
        sample_values = bandvals[sample_ind[:, 0], sample_ind[:, 1]]

        # Store extracted band values as dataframe
        d = {'feature': bands[i],
             'value': sample_values,
             'date': dates[i],
             'label': labels,
             'ind': [*sample_ind]}

        # Necessary due to varying column lengths
        samp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()])).ffill()
        samples_list.append(samp)

    # Combine all samples into single, long-form dataframe
    training = pd.concat(samples_list)

    # Reshape for time-series generation
    training['ind'] = tuple(list(training['ind']))
    training = training.sort_values(by=['ind', 'date'])

    return training


# Directory containing time-series of Sentinel-2 scenes corresponding to AOI
#fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/Sentinel-2A'
fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing'

bd = {'B02': 'blue',
      'B03': 'green',
      'B04': 'red',
      'B08': 'nir'}

t = get_training_data(fp, bd, clust5)



def format_training_data(training_data, one_hot=True):
    # Create 3D numpy array from sample values
    i = training_data.set_index(['date', 'ind', 'feature'])
    shape = list(map(len, i.index.levels))
    arr = np.full(shape, np.nan)
    arr[i.index.labels] = i.values[:,0].flat

    # Kereas LSTM shape: [n_samples, n_timesteps, n_feaures]
    x = arr.swapaxes(0, 1)

    # Data labels
    group = training_data.groupby('ind')
    y = np.array([group.apply(lambda x: x['label'].unique())]).flatten()

    if one_hot:
        y = to_categorical(y, num_classes=len(training_data['label'].unique()))

    return x, y


x, y = format_training_data(t)

# Training and Test sets
x_train, x_test = x[0:int(x.shape[0]*0.8)], x[int(x.shape[0]*0.8):len(x)]
y_train, y_test = y[0:int(y.shape[0]*0.8)], y[int(y.shape[0]*0.8):len(y)]

# Train LSTM model
n_timesteps = len(t['date'].unique())
n_features = len(t['feature'].unique())

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dense(activation='softmax', units=y.shape[1]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=2)
_, accuracy = model.evaluate(x_test, y_test, batch_size=32)
accuracy

