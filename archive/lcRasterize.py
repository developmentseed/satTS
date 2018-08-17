from osgeo import ogr, gdal
import numpy as np
import matplotlib.pyplot as plt


def rasterize(shapefile, outimg, refimg):
    ''' Rasterize a shapefile containing land cover polygons. Shapefile should have an attribute
    called 'id' corresponding to unique land cover class

    :param shapefile (str): file path to shapefile to be rasterized
    :param outimg (str): file path to rasterized image
    :param refimg (str): file path to a reference image. Used to fetch dimensions and other metadata for rasterized img

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
    status = gdal.RasterizeLayer(outrast, [1], layer, None, None, [0], ['ALL_TOUCHED=TRUE', 'ATTRIBUTE=id'])

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


fp = '/Users/jameysmith/Documents/sentinel2_tanz'

# landcover shapefiles to be rasterized
shp = fp + '/land_cover/lc_polygons.shp'

# output file (rasterized land cover)
outimg = fp + '/lcrast/lcrast_v2.tif'

# Reference image (Sentinel-2 tile)
refimg = fp + '/aoiTS/2016-11-16_B04.jp2'


# Rasterize the shapefile
rasterize(shapefile=shp, outimg=outimg, refimg=refimg)

# Confirm land cover classes were burned into raster correctly
check_rasterize(fp + '/lcrast/lcrast_v2.tif')


