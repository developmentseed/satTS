from gippy import GeoImage
import gippy.algorithms as alg
import re
from os import listdir
from itertools import chain



def ndviCalc(filepath):
    # Filepath points to folder of geotiffs of Sentinel 2 time-series of bands 4 (red) and 8 (nir)
    files = [filepath + '/' + f for f in listdir(filepath)]
    del files[0] #.DS_store file - remove from list

    # Get unique dates in image file names
    dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in files]
    dates = set(list(chain.from_iterable(dates)))
    dates = list(dates)

    for date in dates:
        # Full filepath to bands per-timestep
        bandmatch = re.compile(".*" + date)
        bands = list(filter(bandmatch.match, files))

        # Open red and nir bands
        geoimg = GeoImage.open(filenames=bands, bandnames=(['red', 'nir']), nodata=0)

        # Calculate NDVI
        alg.indices(geoimg, products=['ndvi'], filename=filepath + '/ndvi/' + date + '_ndvi.tif')

        geoimg = None


fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs'
ndviCalc(fp)





ds1 = gdal.Open(fp + '/ndvi/2017-06-14_ndvi.tif', gdal.GA_ReadOnly)

ds1vals = ds1.GetRasterBand(1).ReadAsArray()

ds1 = None

ds2 = gdal.Open(fp + '/ndvi/2017-08-03_ndvi.tif', gdal.GA_ReadOnly)

ds2vals = ds2.GetRasterBand(1).ReadAsArray()

ds1 = None

import matplotlib.pyplot as plt
plt.imshow(ds1vals)
plt.imshow(ds2vals)