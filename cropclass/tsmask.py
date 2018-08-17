from osgeo import ogr, gdal
import gippy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

GIT TEST

def rasterize(shapefile, outimg, refimg, attribute):
    ''' Rasterize a shapefile containing land cover polygons. Shapefile should have an attribute
    called 'id' corresponding to unique land cover class or other label

    :param shapefile (str): file path to shapefile to be rasterized
    :param outimg (str): file path to rasterized image
    :param refimg (str): file path to a reference image. Used to fetch dimensions and other metadata for rasterized img
    :param attribute (str): name of attribute in `shapefile` to burn into raster layer

    :return: None (saves image to file)
    '''

    # Open reference raster image to grab projection info and metadata
    img = gdal.Open(refimg, gdal.GA_ReadOnly)

    # Fetch dimensions of reference raster
    ncol = img.RasterXSize
    nrow = img.RasterYSize

    # Projection and extent of raster reference
    proj = img.GetProjectionRef()
    ext = img.GetGeoTransform()

    # Close reference image
    img = None

    # Create raster mask
    memdrive = gdal.GetDriverByName('GTiff')
    outrast = memdrive.Create(outimg, ncol, nrow, 1, gdal.GDT_Byte)

    # Set rasterized image's projection and extent to  input raster's projection and extent
    outrast.SetProjection(proj)
    outrast.SetGeoTransform(ext)

    # Fill output band with the 0 blank (no class) label
    b = outrast.GetRasterBand(1)
    b.Fill(0)

    # Open the shapefile
    polys = ogr.Open(shapefile)
    layer = polys.GetLayerByIndex(0)

    # Rasterize the shapefile layer to new dataset
    status = gdal.RasterizeLayer(outrast, [1], layer, None, None, [0], ['ALL_TOUCHED=TRUE', 'ATTRIBUTE=' + attribute])

    # Close rasterized dataset
    outrast = None


def check_rasterize(rasterized_file, plot=True):
    '''Checks how many pixels are in each class of a rasterized image

    :param rasterized_file (str): File path to a rasterized image
    :param plot (bool): Should the result of the rasterized layer be plotted?

    :return: None
    '''

    # Read rasterized image
    roi_ds = gdal.Open(rasterized_file, gdal.GA_ReadOnly)
    roi = roi_ds.GetRasterBand(1).ReadAsArray()

    # How many pixels are in each class?
    classes = np.unique(roi)

    # Iterate over all class labels in the ROI image, print num pixels/class
    for c in classes:
        print('Class {c} contains {n} pixels'.format(c=c, n=(roi == c).sum()))

    if plot:
        plt.imshow(roi)


def mask_to_array(files, dates, mask, class_num):
    ''' Generate a 3d array of values corresponding to a time-series of image masks for a land cover class

    :param files (list): List of files containing the image time-series (e.g. a stack of NDVI images)
    :param dates (list): List of dates corresponding in the image time-series
    :param mask (str): File path to a land cover mask
    :param class_num (int): ID number of land cover class in `mask`

    :return: 3d array of time-series mask for a specified land cover class
    '''

    # Grab dimensions to set empty array
    ts = gippy.GeoImage.open(filenames=files, bandnames=dates, nodata=0, gain=.0001)

    nbands = ts.nbands()
    nrows = ts.ysize()
    ncols = ts.xsize()

    # Close connection
    ts = None

    arr = np.empty((nbands, nrows, ncols))

    for band in range(0, nbands):
        # Open image time-series
        ndvi_ts = gippy.GeoImage.open(filenames=files, bandnames=dates, nodata=0, gain=.0001)

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


class BandTimeSeries:
    """Time-series of image band values for a (masked) land cover class"""

    def __init__(self, mask, lc_class, ts_var, dates):
        """
        :param mask (numpy array): 3D numpy array corresponding to masked time-series for an image band or index
        :param lc_class (str): name of land cover class
        :param ts_var (str): name of variable contained in masked time-series (e.g. 'red', 'ndvi')
        :param dates (list): list of dates corresponding to time-series
        """
        self.land_cover_class = lc_class
        self.mask = mask
        self.ts_var = ts_var
        if len(dates) == len(mask):
            self.ts_dates = dates
        else:
           raise ValueError('length of dates must match number of time-steps in mask')

        # 2D time-series array of shape (num_timesteps, num_non-nan-pixels)
        mask_vals = self.mask[np.logical_not(np.isnan(self.mask))]
        self.ts_matrix = mask_vals.reshape((len(self.mask), int(mask_vals.shape[0] / len(self.mask))))
        self.num_timesteps = self.ts_matrix.shape[0]
        self.num_timeseries = self.ts_matrix.shape[1]

    def mask_indices(self):
        """Get the indices of non-nan values in crop mask
        :return: list of length #non-nan cells with each element a tuple: (rowindex, colindex)
        """
        w = np.argwhere(np.logical_not(np.isnan(self.mask)))
        wdf = pd.DataFrame(w)
        wsub = wdf.loc[wdf[0] == 0, [1, 2]]
        ind = list(zip(wsub[1], wsub[2]))

        return ind

    def time_series_dataframe(self, frequency, interpolate=True):
        """Create dataframe with band-value time-series for each pixel in land cover class
        :param interpolate (bool): Should time-series be interpolated?
        :param frequency (str): interpolation frequency, e.g. '1d' for daily, '5d' for 5 days
        :return: Dataframe with band-value time-series per-pixel/land cover class
        """

        # Array indices (from original image) of non-nan values
        lc_ind = self.mask_indices()

        # Transpose time-series matrix of dim (# time steps, # non-nan pixels)
        mat_transpose = self.ts_matrix.T

        # Convert to dataframe, change col names to dates
        ts_df = pd.DataFrame(mat_transpose)
        ts_df.columns = self.ts_dates

        # append array indices as column
        ts_df['array_index'] = lc_ind

        # Create land cover value and pixel value columns
        ts_df['lc'] = self.land_cover_class
        ts_df['pixel'] = ts_df.index

        # Convert to long-format and sort
        ts_df = pd.melt(ts_df, id_vars=['lc', 'pixel', 'array_index'], var_name='date', value_name=self.ts_var)
        ts_df = ts_df.sort_values(['lc', 'pixel', 'date'])

        # Convert date column to datetime object (can be used as datetime index for interpolation)
        ts_df['date'] = pd.to_datetime(ts_df['date'], format="%Y-%m-%d")

        if interpolate:
            ts_df = ts_df.set_index('date').groupby(['lc', 'pixel', 'array_index'])
            ts_df = ts_df.resample(frequency)[self.ts_var].asfreq().interpolate(method='linear').reset_index()

        return ts_df