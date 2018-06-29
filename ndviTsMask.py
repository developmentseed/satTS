import gippy
import re
from os import listdir
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd



def mask_to_array(files, dates, mask, class_num):
    # Grab dimensions to set empty array
    ts = gippy.GeoImage.open(filenames=files, bandnames=(dates), nodata=0, gain=.0001)

    nbands = ts.nbands()
    nrows = ts.ysize()
    ncols = ts.xsize()

    # Close connection
    ts = None

    arr = np.empty((nbands, nrows, ncols))

    for band in range(0, nbands):
        # Open NDVI time-series
        ndvi_ts = gippy.GeoImage.open(filenames=files, bandnames=(dates), nodata=0, gain=.0001)

        # Open rasterized landcover
        land_cover = gippy.GeoImage.open(filenames=[mask], bandnames=(['land_cover']), nodata=0)

        # Create land cover mask
        lc_mask = ndvi_ts.add_mask(land_cover['land_cover'] == class_num)

        # Read mask for time-step[band] into np.array
        lc_mask = lc_mask[band].read()

        # Deal with no-data values
        lc_mask[lc_mask == -32768] = np.nan

        # Append water mask np.array
        arr[band] = lc_mask

        # Close image connections
        ndvi_ts = None
        land_cover = None

    return arr



# Indices of non-nan values in crop mask returns as np.array of dim (#non-nan cells, (rowindex, colindex))
def get_mask_indices(mask):
    w = np.argwhere(np.logical_not(np.isnan(mask)))
    wdf = pd.DataFrame(w)
    wsub = wdf.loc[wdf[0] == 0, [1,2]]
    ind = np.array(list(zip(wsub[1], wsub[2])))
    return ind


fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi'

# List of file paths to NDVI time-series
files = [fp + '/' + f for f in listdir(fp)]
del files[0]

# Grab dates from NDVI file names
dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in files]
dates = set(list(chain.from_iterable(dates)))
dates = list(dates)

mask = '/Users/jameysmith/Documents/sentinel2_tanz/lcrast/lcrast.tif'


# Crop class: 1 = water, 2 = veg, 3 = cropped, 4 = urban
water_mask = mask_to_array(files, dates, mask, 1)

veg_mask = mask_to_array(files, dates, mask, 2)

crop_mask = mask_to_array(files, dates, mask, 3)

urban_mask = mask_to_array(files, dates, mask, 4)


# We want a matrix for each mask of shape (28, #non-nan pixels)
water_ts = water_mask[np.logical_not(np.isnan(water_mask))]
water_mat = water_ts.reshape((len(water_mask), int(water_ts.shape[0] / len(water_mask))))
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/water_tsMatrix.npy', water_mat)

veg_ts = veg_mask[np.logical_not(np.isnan(veg_mask))]
veg_mat = veg_ts.reshape((len(veg_mask), int(veg_ts.shape[0] / len(veg_mask))))
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/veg_tsMatrix.npy', veg_mat)

crop_ts = crop_mask[np.logical_not(np.isnan(crop_mask))]
crop_mat = crop_ts.reshape((len(crop_mask), int(crop_ts.shape[0] / len(crop_mask))))
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/crop_tsMatrix.npy', crop_mat)

urb_ts = urban_mask[np.logical_not(np.isnan(urban_mask))]
urb_mat = urb_ts.reshape((len(urban_mask), int(urb_ts.shape[0] / len(urban_mask))))
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/urb_tsMatrix.npy', urb_mat)


# TODO: incorporate array locations into time-series data frames.
arr = get_mask_indices(crop_mask)







