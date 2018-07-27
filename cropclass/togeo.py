import gippy
import rasterio
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd

# Rasterized shapefile:
lcrast = '/Users/jameysmith/Documents/sentinel2_tanz/lcrast/lcrast.tif'

# A single NDVI scene
scene = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi/2016-11-16_ndvi.tif'

# Open ndvi scene (1 date only)
img = rasterio.open(scene)

# Open rasterized land cover layer, add mask for land cover
m = gippy.GeoImage.open(filenames=[scene], bandnames=['ndvi'], nodata=0, gain=0.0001)
land_cover = gippy.GeoImage.open(filenames=[lcrast], bandnames=['land_cover'], nodata=0)

# Create land cover mask for land_cover.ID == 2
lc_mask = m.add_mask(land_cover['land_cover'] == 2)

# Read mask into np.array
vals = lc_mask.read()

# Deal with no-data values
vals[vals == -32768] = np.nan


######## DIFFERENT APPROACH #####
scene = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi/2016-11-16_ndvi.tif'

# Open ndvi scene (1 date only)
img = rasterio.open(scene)

# X,Y index for non-nan pixels
data_loc = np.argwhere(np.logical_not(np.isnan(vals)))

# Get x, y indices from non-nan values in array; 2D array with 503143, 2 columns. Column 0 is row index (lat);
# Column 1 is column index (long)
x = data_loc[:, 1]
y = data_loc[:, 0]

# Get lat long from x y
long_lat = img.xy(x, y)
long = long_lat[0]
lat = long_lat[1]

coords = list(zip(long, lat))

df = pd.DataFrame({"coordinates": coords})
df['coordinates'] = df['coordinates'].apply(Point)
gdf = gpd.GeoDataFrame(df, geometry='coordinates')
