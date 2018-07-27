import pandas as pd
import gippy
from gippy import GeoImage
import gippy.algorithms as alg
import os
import re
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


def random_ts_samples(file_path, n_samples, seed=None):
    ''' Sample of locations for each land cover class included in file_path

    :param file_path (str): Full path to directory containing .csv files with location data for land class samples
    :param n_samples (int): Number of samples to select from full dataset
    :param seed (int): Set a seed to generate same dataset repeatedly

    :return: pd.DataFrame with locations for n_samples of each land cover class
    '''

    # CSV files containing time-series' to be samples
    ts_files = [file_path + '/' + file for file in os.listdir(file_path) if not file.startswith('.')]

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


def calulate_indices(filepath, asset_dict, indices):
    ''' Create image files for indices

    :param filepath (str): Full path to directory containing satellite scenes in default structure created
                           by sat-search load --download
    :param asset_dict (dict): Keys = asset (band) names in scene files (e.g. 'B01', 'B02'); Values = value names
                            corresponding to keys (e.g. 'red', 'nir')
    :param indices (list): Which indices to generate? Options include any index included in gippy.alg.indices

    :return: None (writes files to disk)
    '''

    subdirs = [x[0] for x in os.walk(filepath)]
    subdirs = subdirs[1:len(subdirs)]

    for folder in subdirs:

        # Filepath points to folder of geotiffs of Sentinel 2 time-series of bands 4 (red) and 8 (nir)
        files = [folder + '/' + f for f in os.listdir(folder) if not f.startswith('.')]

        # Asset (band) names
        pattern = '[^_.]+(?=\.[^_.]*$)'
        bands = [re.search(pattern, f).group(0) for f in files]

        # Match band names
        bands = [asset_dict.get(band, band) for band in bands]

        img = GeoImage.open(filenames=files, bandnames=bands, nodata=0)

        for ind in indices:
            alg.indices(img, products=[ind], filename=folder + '/index_' + ind + '.tif')

        img = None


def get_training_data(asset_dir, asset_dict, samples_df, standardize=True):
    ''' Create a dataset of n_features (bands) at each samples location for n_timeseteps

    :param asset_dir (str): File path to directory containing satellite scenes downloaded using the default
                            output of sat-search load
    :param asset_dict (dict): Keys = asset (band) names in scene files (e.g. 'B01', 'B02'); Values = value names
                              corresponding to keys (e.g. 'red', 'nir')
    :param samples_df (pd.DataFrame): pd.DataFrame with samples locations for each land cover class
    :param scale (bool): Scale features using sklearn.preprecessing.MinMaxScaler()

    :return: pd.DataFrame with time-series of n_features (band refectance values) for each sample location
    '''

    # Array indices corresponding to sample locations
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

        # Zero mean and unit variance for feature
        if standardize:
            samp['value'] = preprocessing.scale(samp['value'])

        samples_list.append(samp)

    # Combine all samples into single, long-form dataframe
    training = pd.concat(samples_list)

    # Reshape for time-series generation
    training['ind'] = tuple(list(training['ind']))
    training = training.sort_values(by=['ind', 'date'])

    return training


def format_training_data(training_data, one_hot=True, seed=None):
    ''' Format time-series of reflectance data for fitting a Keras Sequential model

    :param training_data (pd.DataFrame): output of get_training_data
    :param one_hot (bool): Format response variable to one-hot encoded vectors?
    :param seed (bool): Set a seed to generate same train/test datasets repeatedly

    :return: X (feature) matrix , Y (response) matrix, codes (dict) for Y-labels
    '''

    np.random.seed(seed)

    # Create 3D numpy array from sample values
    i = training_data.set_index(['date', 'ind', 'feature'])
    shape = list(map(len, i.index.levels))
    arr = np.full(shape, np.nan)
    arr[i.index.labels] = i.values[:, 0].flat

    # Kereas LSTM shape: [n_samples, n_timesteps, n_feaures]
    x = arr.swapaxes(0, 1)

    # Data labels (Y values); first encode labels as int
    training_data['label'] = training_data['label'].astype('category')

    # Store categorical codes
    label_codes = dict(enumerate(training_data['label'].cat.categories))

    # Convert labels to int
    training_data['label'] = training_data['label'].cat.codes.astype('str').astype('int')

    # Get Y
    group = training_data.groupby('ind')

    y = group.apply(lambda x: x['label'].unique())
    y = y.apply(pd.Series)
    y = y[0].values

    if one_hot:
        y = to_categorical(y, num_classes=len(training_data['label'].unique()))

    return label_codes, x, y


def split_train_test(x, y, seed=0, prop_train=0.8):
    ''' Generate training and test datasets for keras LSTM

    :param x (np.array): dataset of shape (n_samples, n_features, n_timesteps)
    :param y (np.array): data labels of shape (n_classes, n_samples); likely one-hot encoded vectors
    :param seed (int): for shuffling data
    :param prop_train (float): proportion of dataset to use for training; default = 0.8 for 80/20 train/test split

    :return:  x and y matrices for train/test sets
    '''
    x, y = shuffle(x, y, random_state=seed)

    x_train, x_test = x[0:int(x.shape[0] * prop_train)], x[int(x.shape[0] * prop_train):len(x)]
    y_train, y_test = y[0:int(y.shape[0] * prop_train)], y[int(y.shape[0] * prop_train):len(y)]

    return x_train, x_test, y_train, y_test


def conf_mat(x_test, y_test, model, label_dict):

    # Model predictions on test set
    predictions = model.predict(x_test)
    y_pred = (predictions > 0.5)

    # Generate confusion matrix
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    # Add data labels and per-class recall
    tp = matrix.diagonal()
    cm = pd.DataFrame(matrix, index=list(label_dict.values()), columns=list(label_dict.values()))
    cm['recall'] = tp / cm.sum(axis=1)

    return cm