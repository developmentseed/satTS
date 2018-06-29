import numpy as np
import geopandas as gpd
import pandas as pd
import re
import os
import rasterio
import rasterio.plot as rsplot
from rasterio.mask import mask
import matplotlib.pyplot as plt


fp = '/Users/jameysmith/Documents/sentinel2_tanz'

# Want sintenel bands 04 (red) and 08 (NIR) for NDVI calculation
bandre = re.compile("(B04|B08).jp2$")

# Find file names of .jp2 files for "bands of interest" (boi)
boi = []
for file in os.listdir(fp + "/rukwa_2-4-17"):
    if bandre.match(file):
        boi.append(os.path.join(fp, "rukwa_2-4-17/" + file))

# We handle the connections with "with"
with rasterio.open(boi[0]) as src:
    red = src.read(1)

with rasterio.open(boi[1]) as src:
    nir = src.read(1)

# Allow division by zero
np.seterr(divide='ignore', invalid='ignore')

ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)

# Convert ndvi array to rasterio object
profile = src.meta
profile.update(driver='GTiff', dtype=rasterio.float32)

# Write ndvi layer to disk
outfile = fp + '/rukwa_2-4-17/ndvi.tif'
with rasterio.open(outfile, 'w', **profile) as dst:
    dst.write(ndvi.astype(rasterio.float32), 1)


rast = plt.imshow(ndvi)

# Shapefiles with land cover sample generated in QGIS and R
samp = gpd.read_file(fp + "/land_cover/landcover_samples.shp")

# Get shapely geometries
features = samp['geometry']

src = rasterio.open(boi[0])

