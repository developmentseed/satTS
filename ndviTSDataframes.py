import pandas as pd
from os import listdir
import re
from scipy import signal
import numpy as np


def apply_savgol(x, window, poly):
    ''' Perform Savgol signal smoothing on NDVI time-series in dataframe group object (x)
    :param x: Grouped dataframe object
    :param window: smoothing window - pass to signal.savgol_filter 'window_length' param
    :param poly: polynomial order used to fit samples - pass to signal.savgol_filter 'polyorder' param
    :return: "Smoothed" NDVI time-series
    '''

    x['ndvi'] = signal.savgol_filter(x['ndvi'], window_length=window, polyorder=poly)

    return x


def ndvi_dataframe(filepath, lc_list, lc_names, n_samples, long=True, interpolate=True):
    ''' Generates 1000 random samples of NDVI time-series from each land cover class stored in a list of numpy arrays
    and stores in a single dataframe object
    :param filepath: file patch to
    :param lc_list: list of ndvi time-series' numpy arrays
    :param lc_names: str of land cover types included in lc_list
    :param n_samples: int number of samples to select from each lc_list element
    :return: Single pandas dataframe of NDVI time-series'
    '''

    # List of file paths to NDVI time-series
    files = [filepath + '/' + f for f in listdir(filepath)]
    del files[0]  # .DS Store

    # Grab dates from NDVI file names
    dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in files]
    dates = [date for sublist in dates for date in sublist]

    # Select 1000 random samples from each land cover class
    class_samples = []
    for cls in lc_list:
        class_samples.append(cls[:, np.random.randint(cls.shape[1], size=n_samples)])

    # Convert numpy arrays to pandas dataframes
    dfs = []
    for c in range(0, len(classes)):
        df = pd.DataFrame(class_samples[c])
        df['date'] = pd.to_datetime(dates, format="%Y-%m-%d")
        df['lc'] = lc_names[c]

        if long:
            # Convert to long-format, append to df list
            df = pd.melt(df, id_vars=['date', 'lc'], var_name='pixel', value_name='ndvi')

        dfs.append(df)

    # Bind all data frames and return
    class_dfs = pd.concat(dfs)

    # Interpolate NDVI time-series to 5-day series using linear interpolation
    if interpolate:
        class_dfs = class_dfs.set_index('date').groupby(['lc', 'pixel'])
        class_dfs = class_dfs.resample('5d')['ndvi'].asfreq().interpolate(method='linear').reset_index()

        # Subset to dates after Jan 1, 2017
        class_dfs = class_dfs[class_dfs['date'] >= '2017-01-01']

    return class_dfs




# Numpy arrays of ndvi time-series'
water = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/water_tsMatrix.npy')
veg = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/veg_tsMatrix.npy')
crop = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/crop_tsMatrix.npy')
urban = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/urb_tsMatrix.npy')


fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi'

# List of NDVI time-series 3D numpy arrays for each generic land cover category
classes = [water, veg, crop, urban]
names = ['water', 'veg', 'crop', 'urban']

# Create land cover dataframes
np.random.seed(1)
ndvi_ts = ndvi_dataframe(filepath=fp, lc_list=classes, lc_names=names, n_samples=1000)

ndvi_ts.to_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/ndvi_ts.csv', index=False)


# Smoothed time-series'
ndvi_smth = ndvi_ts.groupby(['lc', 'pixel']).apply(apply_savgol, 7, 3)

ndvi_smth.to_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/ndvi_ts_smooth.csv', index=False)



