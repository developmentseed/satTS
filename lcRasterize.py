# A script to rasterise a shapefile to the same projection & pixel resolution as a reference image.
from osgeo import ogr, gdal
import numpy as np

fp = '/Users/jameysmith/Documents/sentinel2_tanz'


# landcover shapefiles to be rasterized
shp = fp + '/land_cover/lc_polygons.shp'

# output file (rasterized land cover)
outimg = fp + '/lcrast/lcrast.tif'

# Reference image (sentinel 2 tile)
refimg = fp + '/aoiTS/2016-11-16_B04.jp2'



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
polys = ogr.Open(shp)
layer = polys.GetLayerByIndex(0)

# Rasterize the shapefile layer to new dataset
status = gdal.RasterizeLayer(outrast, [1], layer, None, None, [0], ['ALL_TOUCHED=TRUE', 'ATTRIBUTE=id'])

# Close rasterized dataset
outrast = None




# Confirm values were burned into raster correctly
roi_ds = gdal.Open(fp + '/lcrast/lcrast.tif', gdal.GA_ReadOnly)

roi = roi_ds.GetRasterBand(1).ReadAsArray()

# How many pixels are in each class?
classes = np.unique(roi)

# Iterate over all class labels in the ROI image, printing out some information
for c in classes:
    print('Class {c} contains {n} pixels'.format(c=c, n=(roi == c).sum()))

import matplotlib.pyplot as plt
plt.imshow(roi)