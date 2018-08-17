import gippy
import re
from os import listdir
import numpy as np
from itertools import chain



def mask_to_array(files, dates, mask, class_num):
    ''' Generate a 3d array of values corresponding to a time-series of image masks for a land cover class

    :param files (list): List of files containing the image time-series (e.g. a stack of NDVI images)
    :param dates (list): List of dates corresponding in the image time-series
    :param mask (str): File path to a land cover mask
    :param class_num (int): ID number of land cover class in `mask`

    :return: 3d array of time-series mask for a specified land cover class
    '''

    # Grab dimensions to set empty array
    ts = gippy.GeoImage.open(filenames=files, bandnames=(dates), nodata=0, gain=.0001)

    nbands = ts.nbands()
    nrows = ts.ysize()
    ncols = ts.xsize()

    # Close connection
    ts = None

    arr = np.empty((nbands, nrows, ncols))

    for band in range(0, nbands):
        # Open image time-series
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






fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi'

# List of file paths to NDVI time-series
files = [fp + '/' + f for f in listdir(fp)]
files.sort()
del files[0] #.DS_Store file

# Grab dates from NDVI file names
dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in files]
dates = set(list(chain.from_iterable(dates)))
dates = list(dates)
dates.sort()

# Land cover mask
mask = '/Users/jameysmith/Documents/sentinel2_tanz/lcrast/lcrast.tif'



# Crop class: 1 = water, 2 = veg, 3 = cropped, 4 = urban
water_mask = mask_to_array(files, dates, mask, 1)
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_masks/water_mask.npy', water_mask)

veg_mask = mask_to_array(files, dates, mask, 2)
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_masks/veg_mask.npy', veg_mask)

crop_mask = mask_to_array(files, dates, mask, 3)
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_masks/crop_mask.npy', crop_mask)

urban_mask = mask_to_array(files, dates, mask, 4)
#np.save('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_masks/urban_mask.npy', urban_mask)



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