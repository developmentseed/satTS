from osgeo import ogr, gdal
from os import listdir

fp = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS'

files = [fp + '/' + f for f in listdir(fp)]
del files[0]

driv = "GTiff"
outpath = fp + "geotiffs"

for f in files:
    inimg = gdal.Open(f)

    driver = gdal.Open(driv)

    outimg = driver.CreateCopy()


in_image = gdal.Open(files[0])

driver = gdal.GetDriverByName("GTiff")

out_image = driver.CreateCopy(fp + "/geotiffs/test.tif", in_image, 0)

roi_ds = gdal.Open(fp + "/geotiffs/test.tif", gdal.GA_ReadOnly)

roi = roi_ds.GetRasterBand(1).ReadAsArray()
import matplotlib.pyplot as plt

plt.imshow(roi)

in_image = None
out_image = None