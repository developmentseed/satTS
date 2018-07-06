import pandas as pd
from os import listdir
import re
import numpy as np



# Indices of non-nan values in crop mask returns as np.array of dim: (#non-nan cells, (rowindex, colindex))
def get_mask_indices(mask):
    w = np.argwhere(np.logical_not(np.isnan(mask)))
    wdf = pd.DataFrame(w)
    wsub = wdf.loc[wdf[0] == 0, [1,2]]
    ind = list(zip(wsub[1], wsub[2]))
    return ind



def ndvi_arr_indx(filepath, ndvi_matrix, mask, name, interpolate=True):
    """Create dataframe with ndvi time-series for each pixel in land cover class

    :param filepath: file path to nvdi images (for grabbing dates from file names)
    :param ndvi_matrix: np.array of shape (#time-steps, #non-nan pixels)
    :param mask: np.array corresponding to land cover mask
    :param name: (str) land cover class name
    :param interpolate: if true, interpolate time-series to 5-day interval

    :return: Dataframe with ndvi time-series per-pixel/land cover class
    """
    # List of file paths to NDVI time-series
    files = [filepath + '/' + f for f in listdir(filepath)]
    files.sort()
    del files[0]  # .DS Store

    # Grab dates from NDVI file names
    dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in files]
    dates = [date for sublist in dates for date in sublist]

    # Array indices (from original image) of non-nan values
    lc_ind = get_mask_indices(mask)

    # Transpose ndvi matrix of dim (# time steps, # non-nan pixels)
    mat_transpose = ndvi_matrix.T

    # Convert to dataframe, change col names to dates
    ndvi_df = pd.DataFrame(mat_transpose)
    ndvi_df.columns = dates

    # append array indices as column
    ndvi_df['array_ind'] = lc_ind

    # Create land cover value and pixel value columns
    ndvi_df['lc'] = name
    ndvi_df['pixel'] = ndvi_df.index

    # Convert to long-format and sort
    ndvi_df = pd.melt(ndvi_df, id_vars=['lc', 'pixel', 'array_ind'], var_name='date', value_name='ndvi')
    ndvi_df = ndvi_df.sort_values(['lc', 'pixel', 'date'])

    # Convert date column to datetime object (can be used as datetime index for interpolation)
    ndvi_df['date'] = pd.to_datetime(ndvi_df['date'], format="%Y-%m-%d")

    if interpolate:
        ndvi_df = ndvi_df.set_index('date').groupby(['lc', 'pixel', 'array_ind'])
        ndvi_df = ndvi_df.resample('5d')['ndvi'].asfreq().interpolate(method='linear').reset_index()

    return ndvi_df




# 28 <15% cloudy scenes total between 2016-11-16 and 2017-12-31
# Land cover NDVI time-series': shape = (28, #non-nan pixels)
water = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/water_tsMatrix.npy')
veg = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/veg_tsMatrix.npy')
crop = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/crop_tsMatrix.npy')
urban = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/urb_tsMatrix.npy')


# File path to ndvi image files (to grab dates)
fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi'

# Land cover masks: shape = (28, 10980, 10980) - load one at a time (memory usage issues)
veg_mask = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_masks/veg_mask.npy')
veg_df = ndvi_arr_indx(fp, veg, veg_mask, 'veg')
veg_mask = None
#veg_df.to_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/veg_ndvi_interp.csv', index=False)

water_mask = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_masks/water_mask.npy')
water_df = ndvi_arr_indx(fp, water, water_mask, 'water')
water_mask = None
#water_df.to_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/water_ndvi_interp.csv', index=False)

crop_mask = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_masks/crop_mask.npy')
crop_df = ndvi_arr_indx(fp, crop, crop_mask, 'crop')
crop_mask = None
#crop_df.to_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/crop_ndvi_interp.csv', index=False)

urban_mask = np.load('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_masks/urban_mask.npy')
urban_df = ndvi_arr_indx(fp, urban, urban_mask, 'urban')
urban_mask = None
#urban_df.to_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/urban_ndvi_interp.csv', index=False)